# CodeSignal-style ML practice

An original Roblox-themed mock based on the interview-format description in the [shared conversation](https://chatgpt.com/share/6ad92cab-a9dc-83e8-b72e-654063228d54), not a verified company question.

**Task:** use historical session logs to analyze engagement and predict whether a player returns the following week. Two problems, 35 coding minutes: data analysis, then model training and evaluation. All text and interviewer questions are in English.

- [Practice notebook](codesignal_roblox_retention_practice.ipynb): supplied inputs/imports, seven TODO cells, per-step checks and follow-up questions.
- [Reference notebook](codesignal_roblox_retention_solution.ipynb): complete code, executed outputs, and expandable answers.
- [Players](data/codesignal_roblox/players.csv) and [sessions](data/codesignal_roblox/sessions.csv): synthetic CSV inputs; no download needed.

Open the practice notebook in VS Code or Jupyter and select `.venv-codesignal/bin/python` as the kernel. Run from the workspace root so the relative CSV paths resolve. The practice notebook intentionally stops at each unimplemented TODO; the reference notebook can run top to bottom. Do not change the raw data to make tests pass.

## Additional mocks

Continue with [three additional Roblox-style exercises](ROBLOX_ML_PRACTICE.md): regression, softmax multiclass classification, and experience-join prediction. Each has a separate practice notebook and reference solution. Your original notebooks and data remain unchanged.

## Local setup

The project-local environment is separate from the existing Transformer environment. To recreate it when needed:

```bash
python3 -m venv .venv-codesignal
.venv-codesignal/bin/python -m pip install -r requirements-codesignal.txt
```

The pinned versions support the computer's Python 3.9. They are a reproducible local setup, not a claim about CodeSignal's installed versions. On CodeSignal, use the provided kernel and packages; only update `DATA_DIR` if the interviewer uses another input directory.

## Validation and regeneration

```bash
.venv-codesignal/bin/python scripts/check-codesignal-practice.py
```

The check validates both notebook schemas, executes the solution in a fresh Python kernel, checks data boundaries and model behavior, and saves only the solution's cell outputs. It does not fill in or overwrite practice answers.

This is a release check for the original templates: it expects the seven practice TODOs to remain untouched. After filling in your answers, use the check cells inside the practice notebook instead.

The source builders are `scripts/build-codesignal-notebooks.mjs` and `scripts/generate-codesignal-data.py`. Do not regenerate a notebook containing your work: the notebook builder refuses to overwrite existing notebooks unless you explicitly pass `--force`. Dataset regeneration restores the synthetic fixtures and overwrites those two CSVs.
