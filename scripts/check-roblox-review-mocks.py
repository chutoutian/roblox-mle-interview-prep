"""Run only the new reference notebooks, preserving all existing learner work/data."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager


ROOT = Path(__file__).resolve().parent.parent
SLUGS = ["review_queue", "playtime_tail", "request_ranking", "high_engagement"]
SELECTED = next((value.split("=", 1)[1] for value in sys.argv[1:] if value.startswith("--only=")), None)
if SELECTED and SELECTED not in SLUGS:
    raise SystemExit(f"Unknown mock {SELECTED!r}.")


class ExplicitPythonKernel(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="Roblox review mock validation", language="python",
        )


AUDITS = {
    "high_engagement": '''
# Independent per-player aggregation and decision-policy oracles.
history = sessions.drop_duplicates().copy()
history = history.loc[history["day_offset"].between(-7, -1)].copy()
history["minutes_played"] = history["minutes_played"].mask(history["minutes_played"].lt(0))
pd.testing.assert_frame_equal(clean_sessions, history)
cols = ["session_count", "known_duration_count", "total_minutes", "active_days", "days_since_last_session"]
oracle = {}
for user in players["user_id"]:
    records = history.loc[history["user_id"].eq(user)]
    observed = [float(value) for value in records["minutes_played"] if pd.notna(value)]
    days = set(records["day_offset"])
    oracle[user] = [len(records), len(observed),
                    sum(observed) if observed else (0 if not days else np.nan),
                    len(days), -max(days) if days else np.nan]
for row in features.itertuples():
    np.testing.assert_allclose([getattr(row, name) for name in cols], oracle[row.user_id], equal_nan=True)
np.testing.assert_array_equal(no_history, features["session_count"].eq(0))
for inputs, target in [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]:
    assert inputs.index.equals(target.index)
    np.testing.assert_array_equal(target, features.loc[inputs.index, "next_week_minutes"].ge(120).astype(int))
numeric_branch = model.named_steps["preprocess"].named_transformers_["num"]
train_filled = X_train[numeric_features].fillna(X_train[numeric_features].median())
np.testing.assert_allclose(numeric_branch.named_steps["scaler"].mean_, train_filled.mean())
np.testing.assert_allclose(numeric_branch.named_steps["scaler"].var_, train_filled.var(ddof=0))
truth = y_valid.to_numpy().astype(bool)
oracle_rows = []
for cutoff in np.unique(valid_prob):
    selected = valid_prob >= cutoff
    tp = int((selected & truth).sum())
    oracle_rows.append({"threshold": cutoff, "selected_count": int(selected.sum()),
                        "precision": tp / selected.sum(), "recall": tp / truth.sum()})
expected = pd.DataFrame(oracle_rows)
pd.testing.assert_frame_equal(threshold_table, expected, check_dtype=False)
eligible = expected.loc[expected["precision"].ge(precision_target) & expected["selected_count"].ge(min_selected)]
pd.testing.assert_frame_equal(feasible, eligible, check_dtype=False)
best = eligible.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
assert threshold == best["threshold"]
positive_col = list(model.classes_).index(1)
np.testing.assert_allclose(test_prob, model.predict_proba(X_test)[:, positive_col])
np.testing.assert_allclose(valid_prob, model.predict_proba(X_valid)[:, positive_col])
for label, decision in [("prior_default", baseline.predict(X_test)),
                        ("threshold_0.5", test_prob >= 0.5),
                        ("chosen_threshold", test_prob >= threshold)]:
    truth = y_test.to_numpy().astype(bool)
    decision = np.asarray(decision).astype(bool)
    tp, fp = int((truth & decision).sum()), int((~truth & decision).sum())
    fn, tn = int((truth & ~decision).sum()), int((~truth & ~decision).sum())
    wanted = {"accuracy": (tp + tn) / len(truth),
              "precision": tp / (tp + fp) if tp + fp else 0,
              "recall": tp / (tp + fn), "selected_count": int(decision.sum())}
    for metric, value in wanted.items():
        np.testing.assert_allclose(policy_scores.loc[label, metric], value)
np.testing.assert_allclose(scores.loc["logistic", "AP"], metrics.average_precision_score(y_test, test_prob))
np.testing.assert_allclose(scores.loc["logistic", "ROC_AUC"], metrics.roc_auc_score(y_test, test_prob))
''',
    "review_queue": '''
# Independently verify the queue policy over every available validation threshold.
truth = y_valid.to_numpy()
assert (truth == 1).sum() > 0
feasible_counts = []
for value in np.unique(valid_prob):
    chosen = valid_prob >= value
    actual_recall = ((truth == 1) & chosen).sum() / (truth == 1).sum()
    if actual_recall >= recall_target:
        feasible_counts.append(int(chosen.sum()))
assert validation_decision.sum() == min(feasible_counts)
tp = int(((y_test.to_numpy() == 1) & (test_decision == 1)).sum())
fp = int(((y_test.to_numpy() == 0) & (test_decision == 1)).sum())
fn = int(((y_test.to_numpy() == 1) & (test_decision == 0)).sum())
np.testing.assert_allclose(policy_scores.loc["selected_threshold", "precision"], tp / (tp + fp) if tp + fp else 0)
np.testing.assert_allclose(policy_scores.loc["selected_threshold", "recall"], tp / (tp + fn))
np.testing.assert_array_equal(frame["needs_review"], frame["label"].eq("abuse").astype(int))
assert not {"label", "needs_review", "review_minutes"} & set(X.columns)
assert set(frame.loc[X_train.index, "creator_id"]).isdisjoint(frame.loc[X_test.index, "creator_id"])
np.testing.assert_allclose(scores.loc["logistic", "log_loss"], -np.mean(
    y_test.to_numpy() * np.log(test_prob) + (1 - y_test.to_numpy()) * np.log1p(-test_prob)
))
''',
    "playtime_tail": '''
# Independent aggregation and target-transform checks, not fixed expected scores.
oracle = {}
for user, rows in clean_sessions.groupby("user_id"):
    observed = [float(value) for value in rows["minutes_played"] if pd.notna(value)]
    oracle[user] = (len(rows), len(observed), sum(observed) if observed else np.nan)
for row in features.itertuples():
    count, known, total = oracle.get(row.user_id, (0, 0, 0))
    assert row.session_count == count and row.known_duration_count == known
    assert (pd.isna(row.total_minutes) and pd.isna(total)) or np.isclose(row.total_minutes, total)
assert no_history.any() and unknown.any()
assert best_name == validation_scores.loc[list(candidates), "MAE_min"].idxmin()
assert all(isinstance(candidate, Pipeline) for candidate in candidates.values())
assert isinstance(final_model, Pipeline)
np.testing.assert_allclose(raw_val_pred, candidates["raw_ridge"].predict(X_val))
np.testing.assert_allclose(log_val_pred, np.expm1(candidates["log_ridge"].predict(X_val)))
if best_name == "log_ridge":
    np.testing.assert_allclose(test_pred, np.expm1(final_model.predict(X_test)))
else:
    np.testing.assert_allclose(test_pred, final_model.predict(X_test))
final_pipeline = final_model
final_imputer = final_pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
np.testing.assert_allclose(final_imputer.statistics_, X_dev[numeric_features].median().to_numpy())
np.testing.assert_allclose(test_scores.loc[best_name, "MAE_min"], np.abs(y_test.to_numpy() - test_pred).mean())
np.testing.assert_allclose(test_scores.loc[best_name, "RMSE_min"], np.sqrt(np.mean((y_test.to_numpy() - test_pred) ** 2)))
np.testing.assert_allclose(final_baseline.constant_.ravel(), [y_dev.median()])
expected_new = final_model.predict(new_player)
expected_validation = candidates[best_name].predict(X_val)
if best_name == "log_ridge":
    expected_new = np.expm1(expected_new)
    expected_validation = np.expm1(expected_validation)
np.testing.assert_allclose(new_prediction, expected_new)
np.testing.assert_allclose(validation_errors["predicted"], expected_validation)
''',
    "request_ranking": '''
# Independent conflict policy based on group sizes after exact deduplication.
deduplicated = impressions.drop_duplicates()
expected_ids = {
    event for event, records in deduplicated.groupby("impression_id") if len(records) == 1
}
assert set(clean_impressions["impression_id"]) == expected_ids
assert not set(clean_impressions["impression_id"]) & set(conflicts["impression_id"])
assert set(frame.iloc[train_idx]["user_id"]).isdisjoint(frame.iloc[test_idx]["user_id"])
precision_values, recall_values, hit_values, positive_hit_values = [], [], [], []
excluded = 0
for request, rows in heldout.groupby("request_id"):
    ordered = sorted(rows.itertuples(), key=lambda row: (-row.score, row.experience_id))
    selected = ordered[:2]
    hits = sum(row.joined for row in selected)
    positives = sum(row.joined for row in ordered)
    precision_values.append(hits / len(selected))
    hit_values.append(hits > 0)
    if positives:
        recall_values.append(hits / positives)
        positive_hit_values.append(hits > 0)
    else:
        excluded += 1
    assert top_two.loc[top_two["request_id"].eq(request), "experience_id"].tolist() == [row.experience_id for row in selected]
np.testing.assert_allclose(request_scores.loc["model_top_two", "precision_at_2_actual_count"], np.mean(precision_values))
np.testing.assert_allclose(request_scores.loc["model_top_two", "recall_at_2_positive_requests"], np.mean(recall_values))
np.testing.assert_allclose(request_scores.loc["model_top_two", "hit_rate_at_2_all_requests"], np.mean(hit_values))
np.testing.assert_allclose(request_scores.loc["model_top_two", "hit_rate_at_2_positive_requests"], np.mean(positive_hit_values))
assert request_scores.loc["model_top_two", "excluded_zero_positive_requests"] == excluded
# Small edge cases: one card, no positives, and deterministic score ties.
edge = pd.DataFrame({
    "request_id": [1, 2, 2, 2], "experience_id": [10, 30, 20, 40],
    "score": [0.4, 0.8, 0.8, 0.1], "joined": [1, 0, 0, 0],
})
edge_selected = edge.sort_values(
    ["request_id", "score", "experience_id"], ascending=[True, False, True]
).groupby("request_id", sort=False).head(2)
assert edge_selected.loc[edge_selected["request_id"].eq(2), "experience_id"].tolist() == [20, 30]
edge_result = evaluate_requests(edge_selected, edge)
np.testing.assert_allclose(edge_result["precision_at_2_actual_count"], 0.5)
np.testing.assert_allclose(edge_result["recall_at_2_positive_requests"], 1.0)
np.testing.assert_allclose(edge_result["hit_rate_at_2_all_requests"], 0.5)
assert edge_result["excluded_zero_positive_requests"] == 1
no_positive = evaluate_requests(edge_selected.loc[edge_selected["request_id"].eq(2)],
                                edge.loc[edge["request_id"].eq(2)])
assert np.isnan(no_positive["recall_at_2_positive_requests"])
assert no_positive["hit_rate_at_2_all_requests"] == 0
''',
}


def tagged(notebook, tag):
    return [cell.source for cell in notebook.cells if tag in cell.metadata.get("tags", [])]


def main():
    assert Path(sys.prefix).resolve() == ROOT / ".venv-codesignal", "Use .venv-codesignal/bin/python."
    slugs = [SELECTED] if SELECTED else SLUGS
    targets = {ROOT / f"codesignal_roblox_{slug}_solution.ipynb" for slug in slugs}
    protected = [file for file in ROOT.glob("*.ipynb") if file not in targets]
    protected += list((ROOT / "data").rglob("*.csv"))
    hashes = {file: hashlib.sha256(file.read_bytes()).digest() for file in protected}
    try:
        for slug in slugs:
            practice = nbformat.read(ROOT / f"codesignal_roblox_{slug}_practice.ipynb", 4)
            output = ROOT / f"codesignal_roblox_{slug}_solution.ipynb"
            solution = nbformat.read(output, 4)
            for notebook in [practice, solution]:
                nbformat.validate(notebook)
                assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
                for cell in notebook.cells:
                    # Learner exercise cells can be partially typed, including
                    # temporarily invalid syntax. Validate reference/check code,
                    # not the student's unfinished answers.
                    if cell.cell_type == "code" and (notebook is solution or "check" in cell.metadata.get("tags", [])):
                        compile(cell.source, cell.id, "exec")
                    elif cell.cell_type == "markdown":
                        assert not any("\u4e00" <= char <= "\u9fff" for char in cell.source), "Notebook prose must be English."
            assert len(tagged(practice, "exercise")) == len(tagged(solution, "solution")) == 7
            assert len(tagged(practice, "setup-exercise")) == 1
            assert tagged(practice, "check") == tagged(solution, "check")
            executed = copy.deepcopy(solution)
            executed.cells.append(nbformat.v4.new_code_cell(AUDITS[slug], id="temporary-independent-audit"))
            with tempfile.TemporaryDirectory(prefix="roblox-review-check-") as directory:
                env = dict(os.environ)
                for name in ["IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR"]:
                    location = Path(directory) / name.lower()
                    location.mkdir()
                    env[name] = str(location)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                manager = AsyncKernelManager(
                    kernel_name="review-mock-check", kernel_spec_manager=ExplicitPythonKernel(),
                    connection_file=str(Path(directory) / "kernel.json"),
                )
                client = NotebookClient(executed, km=manager, timeout=120,
                                        allow_errors=False, record_timing=False)
                client.execute(cwd=str(ROOT), env=env, cleanup_kc=True)
            executed.cells.pop()
            for cell in executed.cells:
                for item in cell.get("outputs", []):
                    assert item.output_type != "error"
                    if item.output_type == "stream" and item.name == "stderr":
                        raise AssertionError(f"Unexpected stderr in {slug}: {item.text}")
            nbformat.validate(executed)
            nbformat.write(executed, output)
            print(f"{slug}: all reference cells and independent audits passed; outputs saved.", flush=True)
    finally:
        for file, digest in hashes.items():
            assert hashlib.sha256(file.read_bytes()).digest() == digest, f"Protected file changed: {file}"
    print("All prior notebooks, practice answers and CSVs were unchanged.")


if __name__ == "__main__":
    main()
