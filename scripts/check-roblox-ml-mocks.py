"""Execute new reference notebooks; never modify the original pair or practice work."""
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
ALL_SLUGS = ["regression", "softmax", "recommendation"]
ONLY = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--only=")), None)
if ONLY is not None and ONLY not in ALL_SLUGS:
    raise SystemExit(f"Unknown mock {ONLY!r}; choose one of: {', '.join(ALL_SLUGS)}")
SLUGS = [ONLY] if ONLY else ALL_SLUGS


class ExplicitPythonKernel(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                          display_name="Mock validation", language="python")


AUDITS = {
    "regression": '''
# Independent per-player oracle, including unknown/no-history cases.
expected = {}
for user, group in clean_sessions.groupby("user_id"):
    values = [float(v) for v in group["minutes_played"] if pd.notna(v)]
    expected[user] = (len(group), len(values), sum(values) if values else np.nan)
for row in features.itertuples():
    count, known, total = expected.get(row.user_id, (0, 0, 0))
    assert row.session_count == count and row.known_duration_count == known
    assert (pd.isna(row.total_minutes) and pd.isna(total)) or np.isclose(row.total_minutes, total)
assert no_history.any() and unknown.any()
assert scores.loc["ridge", "MAE_minutes"] >= 0
''',
    "softmax": '''
assert len(frame) == 1200 and submissions.duplicated().sum() == 15
assert frame["creator_id"].is_unique
assert frame["text_length"].isna().sum() == 35
assert creators["account_age_days"].isna().sum() == 45
assert not {"review_minutes", "label"} & set(X.columns)
assert set(frame.loc[X_train.index, "creator_id"]).isdisjoint(frame.loc[X_test.index, "creator_id"])
# Independent definition: compute F1 per class from label comparisons.
f1_values = []
for label in model.classes_:
    true = y_test.to_numpy() == label
    guessed = pred == label
    tp = (true & guessed).sum()
    fp = (~true & guessed).sum()
    fn = (true & ~guessed).sum()
    f1_values.append(2 * tp / (2 * tp + fp + fn))
np.testing.assert_allclose(scores.loc["softmax", "macro_f1"], np.mean(f1_values))
np.testing.assert_allclose(manual_prob, prob)
''',
    "recommendation": '''
assert len(frame) == 2400 and impressions.duplicated().sum() == 20
assert impressions.shape == (2420, 10) and experiences.shape == (60, 3)
assert impressions["impression_id"].nunique() == 2400
assert experiences["experience_id"].is_unique
assert impressions["experience_id"].isin(experiences["experience_id"]).all()
assert clean["prior_genre_minutes"].isna().sum() == 60
assert clean["device"].isna().sum() == 50
assert frame["user_id"].nunique() == 600
assert set(frame.iloc[train_idx]["user_id"]).isdisjoint(frame.iloc[test_idx]["user_id"])
pd.testing.assert_frame_equal(impressions, pd.read_csv(DATA_DIR / "impressions.csv"))
pd.testing.assert_frame_equal(experiences, pd.read_csv(DATA_DIR / "experiences.csv"))
assert frame["prior_genre_minutes"].isna().sum() == 60
assert set(frame.iloc[train_idx]["request_id"]).isdisjoint(frame.iloc[test_idx]["request_id"])
assert set(train_idx) | set(test_idx) == set(range(len(frame)))
assert (frame.groupby("request_id").size() == 4).all()
# Pairwise interpretation of binary ROC-AUC, independent of sklearn metric.
positive_scores = prob[y_test.to_numpy() == 1]
negative_scores = prob[y_test.to_numpy() == 0]
pairs = positive_scores[:, None] - negative_scores[None, :]
pair_auc = ((pairs > 0).sum() + 0.5 * (pairs == 0).sum()) / pairs.size
np.testing.assert_allclose(scores.loc["logistic", "roc_auc"], pair_auc)
manual_recalls = []
for _, group in ranked.groupby("request_id"):
    total = int(group["joined"].sum())
    if total:
        manual_recalls.append(group.head(2)["joined"].sum() / total)
np.testing.assert_allclose(recall_at_2, np.mean(manual_recalls))
''',
}


def tagged(nb, tag):
    return [c.source for c in nb.cells if tag in c.metadata.get("tags", [])]


def main():
    assert Path(sys.prefix).resolve() == ROOT / ".venv-codesignal", "Use the project CodeSignal environment."
    protected = [ROOT / "codesignal_roblox_retention_practice.ipynb", ROOT / "codesignal_roblox_retention_solution.ipynb"]
    protected += list((ROOT / "data/codesignal_roblox").glob("*.csv"))
    protected += list((ROOT / "data/codesignal_roblox_softmax").glob("*.csv"))
    protected += list((ROOT / "data/codesignal_roblox_recommendation").glob("*.csv"))
    protected += [ROOT / f"codesignal_roblox_{slug}_practice.ipynb" for slug in SLUGS]
    hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in protected}
    try:
        for slug in SLUGS:
            practice = nbformat.read(ROOT / f"codesignal_roblox_{slug}_practice.ipynb", 4)
            target = ROOT / f"codesignal_roblox_{slug}_solution.ipynb"
            solution = nbformat.read(target, 4)
            for nb in [practice, solution]:
                nbformat.validate(nb)
                assert len({c.id for c in nb.cells}) == len(nb.cells)
                for c in nb.cells:
                    if c.cell_type == "code":
                        compile(c.source, "notebook cell", "exec")
            assert len(tagged(practice, "exercise")) == len(tagged(solution, "solution")) == 7
            if slug == "recommendation":
                assert len(tagged(practice, "setup-exercise")) == 1
                assert not tagged(practice, "provided")
                assert len(tagged(solution, "provided")) == 1
            else:
                assert tagged(practice, "provided") == tagged(solution, "provided")
            assert tagged(practice, "check") == tagged(solution, "check")
            executed = copy.deepcopy(solution)
            executed.cells.append(nbformat.v4.new_code_cell(AUDITS[slug], id="temporary-audit"))
            with tempfile.TemporaryDirectory(prefix="roblox-ml-check-") as temp:
                env = dict(os.environ)
                for name in ["IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR"]:
                    directory = Path(temp) / name.lower()
                    directory.mkdir()
                    env[name] = str(directory)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                manager = AsyncKernelManager(kernel_name="mock-check", kernel_spec_manager=ExplicitPythonKernel(),
                                             connection_file=str(Path(temp) / "kernel.json"))
                client = NotebookClient(executed, km=manager, timeout=120, allow_errors=False, record_timing=False)
                client.execute(cwd=str(ROOT), env=env, cleanup_kc=True)
            executed.cells.pop()
            for cell in executed.cells:
                for out in cell.get("outputs", []):
                    assert out.output_type != "error"
                    if out.output_type == "stream" and out.name == "stderr":
                        raise AssertionError(f"Unexpected stderr in {slug}: {out.text}")
            nbformat.validate(executed)
            nbformat.write(executed, target)
            count = sum(c.cell_type == "code" for c in executed.cells)
            print(f"{slug}: {count} executed code cells; notebook checks and independent audits passed.", flush=True)
    finally:
        for p, digest in hashes.items():
            assert hashlib.sha256(p.read_bytes()).digest() == digest, f"Protected file changed: {p}"
    print("Original notebooks, all practice notebooks, and existing CSVs were unchanged.")


if __name__ == "__main__":
    main()
