# Roblox-style ML review: four additional mocks

These are **original practice questions**, not reported exact Roblox coding questions. They follow your supplied interview guide: 5 minutes introduction, 35 minutes coding across data analysis and modeling, and 5 minutes Q&A. All notebook content is English.

Each exercise starts from loading its inputs, with no assumption that another notebook has run. Write your own imports in the practice version. Input CSVs already exist, so data generation is not part of the exercise. Use the workspace root and the `.venv-codesignal/bin/python` kernel.

## 1. Which submissions should enter human review?

[Practice](codesignal_roblox_review_queue_practice.ipynb) · [Reference](codesignal_roblox_review_queue_solution.ipynb)

At submission time, predict whether content belongs in an abuse-specific human-review queue. Clean and join the two source tables, inspect positive-class rates, train a classifier, and choose a threshold that catches at least 80% of validation positives while minimizing the queue among available score thresholds. Freeze the policy, then evaluate on test.

| Review | New material |
| --- | --- |
| drop_duplicates, mask, merge, groupby.agg | Separate training, validation and test roles |
| LogisticRegression, Pipeline, DummyClassifier | precision_recall_curve and its threshold alignment |
| AP, ROC-AUC, log loss, precision, recall | Choosing a decision threshold for a stated action and workload |

Example follow-ups:

1. Why is the default threshold of 0.5 not automatically appropriate?
2. Why is selecting the threshold on test a problem?
3. What changes if the action is automatic deletion rather than human review?
4. What if the review capacity and recall requirements cannot both be met?
5. Why might precision change when the positive fraction changes?

Inputs: `data/codesignal_roblox_softmax/submissions.csv` and `creators.csv`. The binary label definition and 80% requirement are exercise assumptions. Measure the actual class balance; the fixture is not described as an extremely rare-event dataset.

## 2. Predict weekly play time: zeros and a long tail

[Practice](codesignal_roblox_playtime_tail_practice.ipynb) · [Reference](codesignal_roblox_playtime_tail_solution.ipynb)

Build player-level features from historical sessions. Some players have no history, some durations are unknown, and the future target has many zeros and a high upper tail. Compare a constant baseline with Ridge on minutes and Ridge on a transformed target. Select using validation MAE and evaluate the frozen choice on test in original minutes.

| Review | New material |
| --- | --- |
| groupby.agg, size versus count, left merge | Inspecting valid high targets without deleting them automatically |
| Numeric and categorical preprocessing, Ridge | np.log1p and np.expm1 |
| MAE, RMSE and R2 | Explicit target transformation and validation-based model selection |

Example follow-ups:

1. Does a high 99th percentile imply most players play for a long time?
2. Why use log1p rather than log?
3. Does a lower log-space loss guarantee lower RMSE in minutes?
4. Why might MAE improve while RMSE worsens?
5. What should you conclude if the model does not beat a constant baseline?

Inputs: `data/codesignal_roblox/players.csv` and `sessions.csv`. The log transformation is a hypothesis to test, not a guaranteed improvement. Inverse-transforming a predicted mean log target does not generally recover the conditional arithmetic mean in minutes.

## 3. Choose two experiences for each request

[Practice](codesignal_roblox_request_ranking_practice.ipynb) · [Reference](codesignal_roblox_request_ranking_solution.ipynb)

An impression is one displayed card; a request contains several cards. Clean exact duplicates, quarantine conflicting event IDs under the supplied policy, join experience metadata, and hold out users. Train a simple classifier, then select up to two cards within each held-out request and compare with the historical first two positions.

| Review | New material |
| --- | --- |
| nunique, merge, GroupShuffleSplit, iloc | Distinguishing identical copies from conflicting event IDs |
| LogisticRegression, AP and ROC-AUC | sort_values, deterministic ties, groupby.head and reindex |
| Correct feature/label alignment | Per-request precision, recall and hit-rate denominators |

Example follow-ups:

1. Why can global ROC-AUC improve while request-level Top-K results do not?
2. How should precision handle a request with only one surviving card?
3. What is recall when a request has no observed positives?
4. Does setting every position feature to 1 eliminate position bias?
5. Can these logged outcomes prove that reordering increases joins?

Inputs: `data/codesignal_roblox_recommendation/impressions.csv` and `experiences.csv`. The deliberate conflicting-ID example remains in the source data. Quarantine happens only inside this exercise's working DataFrame; no source record is deleted. The ranking result is a diagnostic under logged outcomes, not an unbiased causal estimate of a new display policy.

## 4. Which players will play at least two hours next week?

[Practice](codesignal_roblox_high_engagement_practice.ipynb) · [Reference](codesignal_roblox_high_engagement_solution.ipynb)

At day 0, identify players whose future seven-day play time will reach 120 minutes. Build historical session counts, distinct active days and recency while retaining players with missing durations or no history. Train a simple classifier and prior baseline. On validation, maximize recall subject to precision of at least 30% and at least ten selected players, then evaluate the frozen policy on test.

| Review | New decisions |
| --- | --- |
| Cleaning, aggregation, left merge and preprocessing | Distinct active days versus session frequency, and recency from negative day offsets |
| Train/validation/test and LogisticRegression | Precision-constrained recall maximization with deterministic threshold ties |
| Accuracy, precision, recall, AP and ROC-AUC | Separating ranking quality, probability quality and the actual shortlist policy |

Inputs: the existing `data/codesignal_roblox/players.csv` and `sessions.csv`, read-only. The practice notebook supplies no imports or implementation answers. The reference runs independently, includes answers to 21 per-step questions, and does not require meeting the validation precision requirement on the test set. A missed test target is an evaluation result, not something to hide by retuning.

Optional follow-ups cover an infeasible stronger precision requirement, class-prevalence changes, no-history subgroup evaluation, an additional recent-activity feature, and future-time generalization. Predicting engagement does not establish the causal effect of a promotion.

This is an original mock aligned to the user-supplied interview format, not a newly discovered or verified company question.

## Collected-report connection

| Collected topic | Report | Used to motivate |
| --- | --- | --- |
| Harmful-content detection | [1125288](https://www.1point3acres.com/bbs/thread-1125288-1-1.html) | Human-review queue |
| Classification loss and categorical features | [1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html) | Explaining preprocessing and classification objectives |
| Features, training examples, loss and evaluation; PhD new grad | [1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html) | All three workflows |
| Classification/regression fundamentals; PhD new grad | [1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html) | Regression comparison |
| Recommendation play-time objective; more senior interview | [995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html) | Engagement regression setting |
| Processing interaction data; senior interview | [1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html) | Request-level data and evaluation |

This map refers to the previously collected **Question/Asked** topics in [08_company_interviews.md](08_company_interviews.md), not a new scrape or verification of paywalled reports. Reference explanations in that collection are not verbatim interviewer prompts. None of the reports establishes these exact CSVs, functions, timings, thresholds or complete notebook exercises as actual interview content.

## Reference validation

All four reference notebooks run top to bottom in fresh local Jupyter kernels. Their checks cover train-only preprocessing, split alignment, threshold selection, target inversion, duplicate conflicts and ranking denominators. Independent audits include one-card requests, zero-positive requests, score ties, per-player activity/recency, and precision-constrained recall selection.

```bash
.venv-codesignal/bin/python scripts/check-roblox-review-mocks.py
```

To validate one reference only:

```bash
.venv-codesignal/bin/python scripts/check-roblox-review-mocks.py --only=review_queue
```

The validator saves only the new reference outputs and verifies that all prior notebooks, practice answers and source CSVs are unchanged. Optional extensions are outside the 35-minute core. Reference results are reported honestly; neither the transformed regression nor the model's Top-K selection is required to beat its baseline.
