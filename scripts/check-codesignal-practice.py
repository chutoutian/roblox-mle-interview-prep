#!/usr/bin/env python3
"""Validate both notebooks and execute the reference in an isolated real kernel.

Run: .venv-codesignal/bin/python scripts/check-codesignal-practice.py
Only the reference notebook's execution artifact is saved; practice stays intact.
"""

import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unicodedata

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager


ROOT = Path(__file__).resolve().parent.parent
PRACTICE = ROOT / "codesignal_roblox_retention_practice.ipynb"
SOLUTION = ROOT / "codesignal_roblox_retention_solution.ipynb"

# This audit runs only in an in-memory copy, after the supplied notebook cells.
# Its oracle uses csv/dict arithmetic, independently of the pandas aggregation.
AUDIT_SOURCE = r'''
import csv
import json
import statistics
from collections import Counter, defaultdict
from sklearn import metrics as audit_metrics

passed = []
with (DATA_DIR / "players.csv").open(newline="", encoding="utf-8") as handle:
    raw_players = list(csv.DictReader(handle))
with (DATA_DIR / "sessions.csv").open(newline="", encoding="utf-8") as handle:
    raw_sessions = list(csv.DictReader(handle))
pd.testing.assert_frame_equal(players, pd.read_csv(DATA_DIR / "players.csv"))
pd.testing.assert_frame_equal(sessions, pd.read_csv(DATA_DIR / "sessions.csv"))
passed.append("raw data unchanged in memory")

seen, expected_rows, histories = set(), [], defaultdict(list)
for row in raw_sessions:
    key = tuple(row.items())
    if key in seen:
        continue
    seen.add(key)
    if not -7 <= int(row["day_offset"]) <= -1:
        continue
    duration = float(row["minutes_played"]) if row["minutes_played"] else float("nan")
    if duration < 0:
        duration = float("nan")
    clean_row = dict(row, day_offset=int(row["day_offset"]),
                     minutes_played=duration, completed=int(row["completed"]))
    expected_rows.append(clean_row)
    histories[row["user_id"]].append(clean_row)
expected_clean = pd.DataFrame(expected_rows)
pd.testing.assert_frame_equal(
    clean_sessions.sort_values("session_id").reset_index(drop=True),
    expected_clean.sort_values("session_id").reset_index(drop=True),
    check_dtype=False,
)
passed.append("cleaned rows match independent CSV oracle")

activity_columns = ["session_count", "total_minutes", "avg_minutes",
                    "completion_rate", "known_duration_count"]
expected_activity = {}
for player in raw_players:
    rows = histories[player["user_id"]]
    known = [row["minutes_played"] for row in rows if np.isfinite(row["minutes_played"])]
    count = len(rows)
    total = sum(known) if known else (float("nan") if count else 0.0)
    expected_activity[player["user_id"]] = {
        "session_count": count,
        "total_minutes": total,
        "avg_minutes": total / len(known) if known else total,
        "completion_rate": sum(row["completed"] for row in rows) / count if count else 0.0,
        "known_duration_count": len(known),
    }
expected_frame = pd.DataFrame.from_dict(expected_activity, orient="index")
expected_frame.index.name = "user_id"
actual_frame = player_features.set_index("user_id")
pd.testing.assert_frame_equal(actual_frame[activity_columns].sort_index(),
                              expected_frame[activity_columns].sort_index(), check_dtype=False)
pd.testing.assert_frame_equal(player_features[players.columns].reset_index(drop=True), players)
passed.append("every player aggregation matches independent oracle")

assert actual_frame.loc["user_0001", activity_columns].eq(0).all()
assert actual_frame.loc["user_0002", "session_count"] == 3
assert actual_frame.loc["user_0002", "known_duration_count"] == 0
assert actual_frame.loc["user_0002", ["total_minutes", "avg_minutes"]].isna().all()
passed.append("no-history and all-unknown-duration fixtures")

device_groups = defaultdict(list)
for player in raw_players:
    device_groups[player["device"]].append(expected_activity[player["user_id"]]["total_minutes"])
assert len(device_summary) == len(device_groups)
for device, totals in device_groups.items():
    observed = [value for value in totals if np.isfinite(value)]
    summary = (device_summary.loc[device] if device else
               device_summary.loc[device_summary.index.isna()].iloc[0])
    assert summary["players"] == len(totals)
    assert summary["players_with_observed_total"] == len(observed)
    np.testing.assert_allclose(summary["mean_observed_minutes"], sum(observed) / len(observed))
passed.append("device averages and missing-duration coverage")

raw_by_user = {row["user_id"]: row for row in raw_players}
future_minutes = defaultdict(float)
for row in raw_sessions:
    if int(row["day_offset"]) >= 0:
        player = raw_by_user[row["user_id"]]
        assert int(player["returned_next_week"]) == 1
        assert 0 <= int(row["day_offset"]) <= 6
        assert float(row["minutes_played"]) >= 0
        future_minutes[row["user_id"]] += float(row["minutes_played"])
for user_id, minutes in future_minutes.items():
    assert minutes <= float(raw_by_user[user_id]["next_week_minutes"]) + 1e-9
assert future_minutes, "The fixture should exercise post-cutoff filtering."
passed.append("future-session data agree with labels and outcome totals")

expected_numeric = ["account_age_days", "session_count", "total_minutes", "avg_minutes", "completion_rate"]
assert numeric_features == expected_numeric
assert categorical_features == ["country", "device"]
assert feature_cols == expected_numeric + ["country", "device"]
assert not {"user_id", "returned_next_week", "next_week_minutes"}.intersection(X.columns)
pd.testing.assert_frame_equal(X, actual_frame[feature_cols])
pd.testing.assert_series_equal(y, actual_frame["returned_next_week"])
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_train.index) | set(X_test.index) == set(raw_by_user)
assert len(X_train) == int(0.8 * len(X)) and len(X_test) == len(X) - len(X_train)
pd.testing.assert_frame_equal(X_train, X.loc[X_train.index])
pd.testing.assert_frame_equal(X_test, X.loc[X_test.index])
pd.testing.assert_series_equal(y_train, y.loc[X_train.index])
pd.testing.assert_series_equal(y_test, y.loc[X_test.index])
assert set(y_train) == set(y_test) == {0, 1}
passed.append("explicit feature boundary and disjoint complete player split")

expected_medians = []
for column in numeric_features:
    values = []
    for user_id in X_train.index:
        value = (float(raw_by_user[user_id][column]) if raw_by_user[user_id][column] else float("nan")) if column == "account_age_days" else expected_activity[user_id][column]
        if np.isfinite(value):
            values.append(value)
    expected_medians.append(statistics.median(values))
fitted_numeric = model.named_steps["preprocess"].named_transformers_["num"]
np.testing.assert_allclose(fitted_numeric.named_steps["imputer"].statistics_, expected_medians)
train_imputed = X_train[numeric_features].fillna(dict(zip(numeric_features, expected_medians)))
np.testing.assert_allclose(fitted_numeric.named_steps["scaler"].mean_, train_imputed.mean().to_numpy())
np.testing.assert_allclose(fitted_numeric.named_steps["scaler"].var_, train_imputed.var(ddof=0).to_numpy())
assert fitted_numeric.named_steps["scaler"].n_samples_seen_ == len(X_train)
passed.append("imputer medians and scaling learned from training rows only")

fitted_categorical = model.named_steps["preprocess"].named_transformers_["cat"]
for index, column in enumerate(categorical_features):
    counts = Counter(raw_by_user[user_id][column] for user_id in X_train.index if raw_by_user[user_id][column])
    expected_mode = min(counts, key=lambda value: (-counts[value], value))
    assert fitted_categorical.named_steps["imputer"].statistics_[index] == expected_mode
    assert set(fitted_categorical.named_steps["encoder"].categories_[index]) == set(counts)
np.testing.assert_allclose(baseline.class_prior_, [1 - y_train.mean(), y_train.mean()])
passed.append("categorical vocabulary, modes and baseline prior use training rows")

np.testing.assert_array_equal(model_prob, model.predict_proba(X_test)[:, 1])
np.testing.assert_array_equal(model.predict_proba(X_test), model.predict_proba(X_test))
np.testing.assert_array_equal(new_prob, model.predict_proba(new_players[feature_cols])[:, 1])
np.testing.assert_array_equal(model.predict_proba(new_players[feature_cols]), model.predict_proba(new_players[feature_cols]))
assert "NEW_COUNTRY" not in fitted_categorical.named_steps["encoder"].categories_[0]
assert new_players.isna().any().any()
stress_input = new_players.astype({column: float for column in numeric_features})
stress_input.loc[:, numeric_features] = np.nan
stress_input["country"] = "AUDIT_UNSEEN_COUNTRY"
stress_input["device"] = np.nan
stress_prob = model.predict_proba(stress_input[feature_cols])[:, 1]
assert np.isfinite(stress_prob).all() and ((stress_prob >= 0) & (stress_prob <= 1)).all()
passed.append("repeatable predictions with missing fields and unseen categories")

for name, probabilities in [("prior_baseline", baseline_prob), ("logistic_regression", model_prob)]:
    labels = (probabilities >= 0.5).astype(int)
    expected_metrics = {
        "accuracy": audit_metrics.accuracy_score(y_test, labels),
        "roc_auc": audit_metrics.roc_auc_score(y_test, probabilities),
        "average_precision": audit_metrics.average_precision_score(y_test, probabilities),
        "precision@0.5": audit_metrics.precision_score(y_test, labels, zero_division=0),
        "recall@0.5": audit_metrics.recall_score(y_test, labels, zero_division=0),
        "f1@0.5": audit_metrics.f1_score(y_test, labels, zero_division=0),
    }
    for metric, expected_value in expected_metrics.items():
        np.testing.assert_allclose(metrics.loc[name, metric], expected_value)
np.testing.assert_allclose(baseline_prob, np.full(len(X_test), y_train.mean()))
np.testing.assert_allclose(metrics.loc["prior_baseline", "roc_auc"], 0.5)
np.testing.assert_allclose(metrics.loc["prior_baseline", "average_precision"], y_test.mean())
passed.append("six reported metrics match sklearn; dummy AUC/AP identities")
print(f"Independent runtime checks: {len(passed)} groups passed.")
print(json.dumps(metrics.round(6).to_dict(orient="index"), indent=2))
'''


class ExplicitPythonKernel(KernelSpecManager):
    """Avoid relying on a globally installed or registered Python kernelspec."""

    def get_kernel_spec(self, kernel_name):
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="CodeSignal validation", language="python",
        )


def tagged_sources(notebook, tag):
    return [cell.source for cell in notebook.cells if tag in cell.metadata.get("tags", [])]


def validate_structure(practice, solution):
    for name, notebook in [("practice", practice), ("solution", solution)]:
        nbformat.validate(notebook)
        ids = [cell.id for cell in notebook.cells]
        assert len(ids) == len(set(ids)), f"Duplicate cell IDs in {name}"
        for index, cell in enumerate(notebook.cells):
            # English prose may contain Unicode punctuation, but no other scripts.
            assert all(not char.isalpha() or "LATIN" in unicodedata.name(char, "")
                       for char in cell.source), f"Non-Latin text in {name}, cell {index}"
            if cell.cell_type == "code":
                compile(cell.source, f"{name}:cell-{index}", "exec")
        assert "interviewer" in notebook.cells[0].source.lower()
    exercises = tagged_sources(practice, "exercise")
    assert len(exercises) == len(tagged_sources(solution, "solution")) == 7
    for source in exercises:
        body = ast.parse(source).body
        assert len(body) == 1 and isinstance(body[0], ast.Raise)
        assert isinstance(body[0].exc, ast.Call)
        assert isinstance(body[0].exc.func, ast.Name) and body[0].exc.func.id == "NotImplementedError"
    for notebook, expected in [(practice, 7), (solution, 0)]:
        raises = sum(isinstance(node, ast.Name) and node.id == "NotImplementedError"
                     for cell in notebook.cells if cell.cell_type == "code"
                     for node in ast.walk(ast.parse(cell.source)))
        assert raises == expected
    for tag, count in [("provided", 2), ("check", 7)]:
        assert len(tagged_sources(practice, tag)) == count
        assert tagged_sources(practice, tag) == tagged_sources(solution, tag)
    supplied_practice = [cell.source for cell in practice.cells if cell.cell_type == "code"
                        and "exercise" not in cell.metadata.get("tags", [])]
    supplied_solution = [cell.source for cell in solution.cells if cell.cell_type == "code"
                        and "solution" not in cell.metadata.get("tags", [])]
    assert supplied_practice == supplied_solution
    assert all(cell.execution_count is None and not cell.outputs
               for cell in practice.cells if cell.cell_type == "code")
    print("Static checks: schemas, unique IDs, English-script text, compilation, 7 TODOs, supplied cells passed.")


def main():
    if Path(sys.prefix).resolve() != (ROOT / ".venv-codesignal").resolve():
        raise SystemExit("Run with .venv-codesignal/bin/python scripts/check-codesignal-practice.py")
    protected = [PRACTICE, ROOT / "data/codesignal_roblox/players.csv",
                 ROOT / "data/codesignal_roblox/sessions.csv"]
    original_hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in protected}
    practice = nbformat.read(PRACTICE, as_version=4)
    solution = nbformat.read(SOLUTION, as_version=4)
    validate_structure(practice, solution)
    executed = copy.deepcopy(solution)
    executed.cells.append(nbformat.v4.new_code_cell(AUDIT_SOURCE, id="temporary-independent-audit"))
    with tempfile.TemporaryDirectory(prefix="codesignal-validation-") as temp:
        env = dict(os.environ)
        for variable in ["IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR"]:
            directory = Path(temp) / variable.lower()
            directory.mkdir()
            env[variable] = str(directory)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        manager = AsyncKernelManager(kernel_name="codesignal-validation",
                                     kernel_spec_manager=ExplicitPythonKernel(),
                                     connection_file=str(Path(temp) / "kernel.json"))
        client = NotebookClient(executed, km=manager, timeout=180,
                                allow_errors=False, record_timing=False)
        try:
            client.execute(cwd=str(ROOT), env=env, cleanup_kc=True)
        finally:
            for path, digest in original_hashes.items():
                assert hashlib.sha256(path.read_bytes()).digest() == digest, f"Protected file changed: {path}"
    audit = executed.cells.pop()
    for output in audit.outputs:
        if output.output_type == "stream":
            print(output.text, end="")
    nbformat.validate(executed)
    nbformat.write(executed, SOLUTION)
    count = sum(cell.cell_type == "code" for cell in executed.cells)
    print(f"Executed {count} reference code cells in {sys.executable}.")
    print("Saved reference outputs; practice, source CSVs, and notebook source cells unchanged.")


if __name__ == "__main__":
    main()
