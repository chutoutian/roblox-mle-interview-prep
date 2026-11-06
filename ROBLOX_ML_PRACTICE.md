# Roblox-style Data Analysis & Modeling practice

Original mock interviews, aligned to your supplied **Data Analysis & Modeling Coding Interview** guide, not verified Roblox questions. The guide describes interview format, not a team-specific job description or an exact syllabus.

**Each mock:** 5 minutes introduction, 35 minutes coding across two problems (data analysis, then model training), 5 minutes Q&A. Seven short steps include expected variables, runnable checks, a sentence to say before coding, and possible interviewer questions. Reference notebooks add complete code and expandable answers.

Each notebook includes an interviewer prompt, timing table, data contract, provided setup, Problem 1 Steps 1–3, Problem 2 Steps 4–7, and optional follow-ups. Each step has three interviewer questions. The regression exercise is self-contained: its tasks, explanations and questions assume no prior notebook. Existing practice answers are preserved when updating explanatory text.

## Recommended order

| Mock | Practice | Reference | What changes from your first binary classifier? |
| --- | --- | --- | --- |
| 1. Next-week play time | [Regression](codesignal_roblox_regression_practice.ipynb) | [Solution](codesignal_roblox_regression_solution.ipynb) | Ridge regression, continuous labels, MAE / RMSE / R2, zero-heavy targets |
| 2. Content categories | [Softmax](codesignal_roblox_softmax_practice.ipynb) | [Solution](codesignal_roblox_softmax_solution.ipynb) | Three classes, N-by-3 probabilities, argmax, macro-F1, log loss, 3-by-3 confusion matrix |
| 3. Experience joins | [Recommendation](codesignal_roblox_recommendation_practice.ipynb) | [Solution](codesignal_roblox_recommendation_solution.ipynb) | Impression-level examples, many-to-one joins, user-disjoint split, class imbalance; optional Recall@2 |

Start with **regression**. Its player/session data contract explains all inputs, prediction timing and aggregation rules. Work through it as an independent problem: load the data, inspect it, build features, fit a model and evaluate predictions. For a timed mock, start after the supplied setup and narrate your choices.

The softmax exercise reads two supplied CSVs from `data/codesignal_roblox_softmax/`. The recommendation exercise reads **impressions.csv** and **experiences.csv** from `data/codesignal_roblox_recommendation/`. Both use pre-generated synthetic data. Neither needs an account, download, or extra package. Synthetic-data parameters do not describe actual Roblox users or policies.

**Recommendation practice starts without imports.** Write your own imports, define SEED and the data path, then load both CSVs in Step 1. Its reference solution supplies complete imports and working code. Restart the kernel before starting so you can check that your own setup is sufficient. Include this setup in the timed attempt.

## Additional review mocks

Four independent exercises mix familiar operations with new decisions. Every practice notebook starts without imports, reuses the supplied CSVs read-only, and includes seven steps and 21 interviewer questions. Reference notebooks contain complete code, checks, saved outputs and expandable answers.

| Order | Practice | Reference | Main new material |
| --- | --- | --- | --- |
| 1 | [Human-review queue](codesignal_roblox_review_queue_practice.ipynb) | [Solution](codesignal_roblox_review_queue_solution.ipynb) | Train/validation/test, precision_recall_curve, recall versus review workload |
| 2 | [Zeros and long-tailed play time](codesignal_roblox_playtime_tail_practice.ipynb) | [Solution](codesignal_roblox_playtime_tail_solution.ipynb) | log1p/expm1, transformed targets, honest validation selection, original-minute errors |
| 3 | [Two experiences per request](codesignal_roblox_request_ranking_practice.ipynb) | [Solution](codesignal_roblox_request_ranking_solution.ipynb) | Conflicting event IDs, sort_values, groupby.head, per-request Top-K metrics |
| 4 | [High-engagement player shortlist](codesignal_roblox_high_engagement_practice.ipynb) | [Solution](codesignal_roblox_high_engagement_solution.ipynb) | Distinct active days, recency, maximizing recall under a precision constraint, accuracy versus AP/ROC-AUC |

See [ROBLOX_ML_REVIEW.md](ROBLOX_ML_REVIEW.md) for the prompts, source boundaries and new-function map. Each is self-contained; no variables from a previous notebook are needed. They are original adaptations of collected report topics, not newly verified interview questions.

The recommendation CSV deliberately contains a conflicting impression ID for the earlier debugging exercise. The new request-ranking mock specifies an explicit quarantine policy. The original recommendation reference still stops at its uniqueness check; this is intentional, and the new validator does not alter or execute it.

Validate only these new references with:

```bash
.venv-codesignal/bin/python scripts/check-roblox-review-mocks.py
```

## Run

Open from the workspace root and select the existing **.venv-codesignal/bin/python** kernel. Replace each `NotImplementedError` in the practice version, then run the next check cell. The solution can run top to bottom. New notebooks do not modify the original exercise or its CSVs.

Use the existing pinned environment in [requirements-codesignal.txt](requirements-codesignal.txt). On a real CodeSignal environment, use its provided packages. `root_mean_squared_error` requires sklearn >= 1.4; on an older environment use `np.sqrt(mean_squared_error(y_true, y_pred))`.

The mathematical softmax reconstruction and per-request ranking are **optional extensions**, not required to finish the core 35 minutes. You do not need to implement an optimizer from scratch for these mocks.

## Differences to remember

| | Binary classification | Regression | Single-label multiclass |
| --- | --- | --- | --- |
| Target | 0 or 1 | Number, such as minutes | One of K categories |
| Basic model | LogisticRegression | Ridge / LinearRegression | LogisticRegression with multinomial-capable solver |
| Predict | Class label | Numeric value | One class label |
| predict_proba | N-by-2; choose the positive-class column | Not available | N-by-K; inspect classes_ |
| Decision | Select a threshold if needed | No probability threshold | Argmax for the default decision |
| Metrics | ROC-AUC, AP, precision, recall, F1 | MAE, RMSE, R2 | Macro-F1, per-class recall, log loss, accuracy |

## Source boundary

The user-supplied guide explicitly calls for pandas/NumPy data processing, feature engineering, baseline models, evaluation, live explanation, and end-to-end sklearn workflows. It does **not** say that these exact tasks, softmax, or ranking will be asked.

The [official PhD Early Career MLE posting](https://careers.roblox.com/jobs/8242623) mentions engagement, safety and recommendation systems; those motivate the business settings only. Sources checked 2026-10-22. The three-way moderation taxonomy is an exercise simplification, not a Roblox policy.

## Validation

The reference notebooks have saved executed outputs. Recheck them with:

```bash
.venv-codesignal/bin/python scripts/check-roblox-ml-mocks.py
```

To check only the recommendation reference after editing other practice notebooks:

```bash
.venv-codesignal/bin/python scripts/check-roblox-ml-mocks.py --only=recommendation
```

This runs each reference in a fresh kernel, verifies its assertions plus independent metric/data checks, and saves only reference outputs. It preserves practice answers, original notebooks and existing datasets. `scripts/build-roblox-ml-mocks.mjs` is the original source generator and refuses to overwrite existing notebooks.
