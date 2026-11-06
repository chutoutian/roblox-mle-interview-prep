// Build the original practice and reference notebooks. No network access.
// Existing notebooks are protected unless --force is explicitly supplied.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const filenames = ['codesignal_roblox_retention_practice.ipynb', 'codesignal_roblox_retention_solution.ipynb'];
if (!process.argv.includes('--force') && filenames.some(f => fs.existsSync(path.join(root, f)))) {
  throw new Error('Notebook already exists. Keep your practice edits; use --force only to intentionally regenerate both notebooks.');
}

const intro = `# Data Analysis & Modeling: Will a Player Return?

**Original interview mock, not a reported Roblox question or an official CodeSignal assessment.**
The format follows the interview description in your [shared conversation](https://chatgpt.com/share/6ad92cab-a9dc-83e8-b72e-654063228d54): **5 minutes introduction + 35 minutes coding + 5 minutes Q&A**. All data are synthetic; do not interpret these results as Roblox player behavior.

## The interviewer prompt

You work on a game platform. At the start of day 0, predict whether each player will return during days 0 through 6, using only information available before day 0.

**Problem 1 — Data analysis (14 minutes).** Load two CSVs, inspect their quality, clean the session log, and build one row per player. Report the percentage of players with no historical sessions and average historical play time per device, including inactive players.

**Problem 2 — Modeling (21 minutes).** Select valid features, create a holdout set, train a simple classification baseline, evaluate it against a no-skill baseline, and predict for two new players. Explain your choices as you code. No deep learning or from-scratch optimizer is required.

| Coding time | Task |
| --- | --- |
| 0–4 min | Load and inspect |
| 4–9 min | Clean the session log |
| 9–14 min | Aggregate and join |
| 14–18 min | Select features and split |
| 18–27 min | Preprocess and train |
| 27–32 min | Evaluate |
| 32–35 min | Predict and summarize |

These are suggested checkpoints, not official sub-round timings. Ask during the introduction: What is one example? When is a prediction made? Which mistake is more costly?
`;

const schema = `## Data contract — supplied by the interviewer

**players.csv:** one row per player; attributes are snapshots taken before day 0. All players' outcome windows have completed, so labels are known. Every player belongs in the analysis, even if they have no session history.

| Column | Meaning |
| --- | --- |
| user_id | Unique identifier; not a numeric predictor |
| account_age_days | Account age at the cutoff; may be missing |
| country, device | Categorical attributes at the cutoff; may be missing |
| returned_next_week | Label: 1 if the player returns during days 0–6, otherwise 0 |
| next_week_minutes | Total minutes during days 0–6; present in the offline export, **unavailable at prediction time** |

**sessions.csv:** one row per logged session before cleaning.

| Column | Meaning |
| --- | --- |
| session_id | Session identifier; repeated exports may contain exact duplicate rows |
| user_id | Player identifier, always referencing a supplied player |
| day_offset | Relative to the cutoff: only **-7 through -1** are valid historical features |
| minutes_played | Session duration; blank or a negative sentinel means unknown, not zero |
| completed | 0/1 session-completion indicator; valid even if duration is unknown |

All session rows are available in this small batch exercise; the export deliberately contains some post-cutoff rows. For this mock, distinct players are assumed to be independent samples from the same population. A future-deployment evaluation would instead need multiple cutoffs and a chronological holdout.
`;

const setup = `from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

SEED = 42
DATA_DIR = Path("data/codesignal_roblox")
assert (DATA_DIR / "players.csv").exists(), "Run from the workspace root, or change DATA_DIR to your CSV folder."
assert (DATA_DIR / "sessions.csv").exists()
`;

const steps = [
  {
    title: 'Problem 1 · Step 1 — Load and inspect (0–4 min)',
    task: `Load **players** and **sessions**. Inspect shapes, sample rows, dtypes, missingness and numeric summaries. Check player uniqueness, duplicate sessions, invalid durations, out-of-window records, and label balance. Do not start dropping every incomplete row.`,
    say: 'I will first check the unit of observation and data quality. The label is at player level, but the event log has many rows per player.',
    questions: [
      ['What does one training example represent?', 'One player at one prediction cutoff, not one session. Session data must be aggregated before modeling.'],
      ['Which columns must not become features?', 'The label, user_id, next_week_minutes, and any post-cutoff activity. The identifier is not meaningful ordered data, and the future fields leak the outcome.'],
      ['What do you check before filling missing values?', 'The meaning of missingness, its frequency, and whether it means unknown or no activity. A missing duration is not evidence of a zero-minute session.'],
    ],
    solution: `players = pd.read_csv(DATA_DIR / "players.csv")
sessions = pd.read_csv(DATA_DIR / "sessions.csv")

print("players:", players.shape, "sessions:", sessions.shape)
display(players.head(), sessions.head())
players.info()
sessions.info()
display(players.isna().sum(), sessions.isna().sum())
display(sessions.describe())
display(players["returned_next_week"].value_counts(normalize=True).rename("fraction"))
print("duplicate player IDs:", players["user_id"].duplicated().sum())
print("exact duplicate sessions:", sessions.duplicated().sum())
print("negative durations:", sessions["minutes_played"].lt(0).sum())
print("outside history window:", (~sessions["day_offset"].between(-7, -1)).sum())
`,
    checks: `assert players["user_id"].is_unique
assert players["returned_next_week"].notna().all()
assert set(players["returned_next_week"]) == {0, 1}
assert sessions["user_id"].isin(players["user_id"]).all()
print("Input contract checked.")
`,
  },
  {
    title: 'Problem 1 · Step 2 — Clean the session log (4–9 min)',
    task: `Create **clean_sessions** without modifying **sessions**. Remove exact duplicate rows, keep only day offsets -7 through -1, and convert negative durations to missing values. Keep a session with an unknown duration: it still counts as activity. Fail visibly if duplicate session IDs remain with conflicting rows.`,
    say: 'I will apply deterministic validity rules now. I will postpone learned imputation statistics until after the train/test split.',
    questions: [
      ['Why not use dropna() on the whole table?', 'It discards valid evidence of activity and may systematically remove particular users or devices. Keep valid fields and handle missing predictors later.'],
      ['Why drop exact duplicates rather than silently keep the first session ID?', 'The contract permits identical repeated exports. Conflicting rows require a documented resolution rule; arbitrary first-row selection could discard valid information.'],
      ['Is this cleaning before splitting a leak?', 'These fixed validity rules use neither labels nor statistics learned across players. Median imputation, scaling and category learning are different: fit them only on training data.'],
    ],
    solution: `clean_sessions = sessions.drop_duplicates().copy()
clean_sessions = clean_sessions.loc[
    clean_sessions["day_offset"].between(-7, -1)
].copy()
assert clean_sessions["session_id"].is_unique, "Conflicting records need a resolution rule."
clean_sessions["minutes_played"] = clean_sessions["minutes_played"].mask(
    clean_sessions["minutes_played"].lt(0)
)

print("raw rows:", len(sessions), "clean historical rows:", len(clean_sessions))
print("unknown durations retained:", clean_sessions["minutes_played"].isna().sum())
display(clean_sessions.head())
`,
    checks: `assert clean_sessions["session_id"].is_unique
assert clean_sessions["day_offset"].between(-7, -1).all()
assert clean_sessions["minutes_played"].dropna().ge(0).all()
assert clean_sessions["minutes_played"].isna().any()
print("Historical-session checks passed.")
`,
  },
  {
    title: 'Problem 1 · Step 3 — Aggregate and join (9–14 min)',
    task: `Build **player_features**, one row for every player. Add **session_count**, **total_minutes**, **avg_minutes**, and **completion_rate** from clean history. Keep a duration-observation count to distinguish unknown durations from inactivity. For a player with no sessions, use 0 for these activity features. For a player with sessions but no valid durations, keep total and average minutes missing.

Report (1) the fraction of players with no historical activity, and (2) average per-player observed total minutes by device. Include inactive players; keep an explicit missing-device group. Counts accompanying the averages should show missing-duration coverage.`,
    say: 'I will aggregate before joining, so players with many sessions do not accidentally receive more training weight. A left join preserves inactive players.',
    questions: [
      ['Why size rather than count for the number of sessions?', 'size counts rows. Counting minutes_played skips unknown durations and undercounts valid sessions.'],
      ['What happens if you join players directly to all sessions?', 'You get multiple rows per player. That changes the unit of analysis and can leak the same player across a random row split.'],
      ['Does higher play time on one device prove that device increases engagement?', 'No. This is a descriptive association with possible confounding, missing data, and selection effects; it is not a randomized causal comparison.'],
    ],
    solution: `activity = clean_sessions.groupby("user_id").agg(
    session_count=("session_id", "size"),
    total_minutes=("minutes_played", "sum"),
    avg_minutes=("minutes_played", "mean"),
    completion_rate=("completed", "mean"),
    known_duration_count=("minutes_played", "count"),
)
# A sum over entirely missing durations should be unknown, not zero.
activity.loc[activity["known_duration_count"].eq(0), "total_minutes"] = np.nan
player_features = players.merge(
    activity, on="user_id", how="left", validate="one_to_one"
)
no_sessions = player_features["session_count"].isna()
player_features.loc[no_sessions, activity.columns] = 0

print(f"No historical sessions: {no_sessions.mean():.1%}")
device_summary = player_features.groupby("device", dropna=False).agg(
    players=("user_id", "size"),
    players_with_observed_total=("total_minutes", "count"),
    mean_observed_minutes=("total_minutes", "mean"),
)
display(device_summary, player_features.head())
print("Totals sum observed durations; partially missing sessions can understate true play time.")
`,
    checks: `assert len(player_features) == len(players)
assert player_features["user_id"].is_unique
assert player_features["session_count"].notna().all()
assert player_features["session_count"].sum() == len(clean_sessions)
inactive = player_features["session_count"].eq(0)
assert inactive.any()
assert player_features.loc[inactive, ["total_minutes", "avg_minutes"]].eq(0).all().all()
unknown = player_features["session_count"].gt(0) & player_features["known_duration_count"].eq(0)
assert unknown.any()
assert player_features.loc[unknown, ["total_minutes", "avg_minutes"]].isna().all().all()
print("One-row-per-player and missing-versus-zero checks passed.")
`,
  },
  {
    title: 'Problem 2 · Step 4 — Select features and split (14–18 min)',
    task: `Define **numeric_features**, **categorical_features**, and **feature_cols**. Use the five numeric predictors account_age_days, session_count, total_minutes, avg_minutes, completion_rate, plus country and device. Create **X** and **y**, then an 80/20 stratified split: **X_train**, **X_test**, **y_train**, **y_test**. Use SEED. Keep user_id as the row index for auditing, not a predictor.`,
    say: 'My feature list is explicit. I will reserve the test set now and fit every learned preprocessing step only on training data.',
    questions: [
      ['Why use stratify=y?', 'It approximately preserves class proportions in both subsets. It does not remove temporal leakage or make repeated observations independent.'],
      ['Why is a random split acceptable here, and when is it wrong?', 'This mock assumes independent players at a shared cutoff. To test future deployment, use later cutoffs with mature labels. To test unseen users with repeated rows, use a grouped split; grouping and time answer different evaluation questions.'],
      ['Is total_minutes already leaking because we computed it before the split?', 'No: it uses only each player\'s own pre-cutoff records, not other players, future activity, or labels. Fitting a global scaler on all players would leak test information.'],
    ],
    solution: `numeric_features = [
    "account_age_days", "session_count", "total_minutes", "avg_minutes", "completion_rate"
]
categorical_features = ["country", "device"]
feature_cols = numeric_features + categorical_features

indexed = player_features.set_index("user_id")
X = indexed[feature_cols]
y = indexed["returned_next_week"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print("train:", X_train.shape, "test:", X_test.shape)
print("positive rate:", {"train": y_train.mean(), "test": y_test.mean()})
`,
    checks: `assert not {"user_id", "returned_next_week", "next_week_minutes"}.intersection(feature_cols)
assert X_train.index.intersection(X_test.index).empty
assert len(X_train) + len(X_test) == len(player_features)
assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)
assert set(y_train) == set(y_test) == {0, 1}
print("Split and feature-boundary checks passed.")
`,
  },
  {
    title: 'Problem 2 · Step 5 — Preprocess and train (18–27 min)',
    task: `Fit **baseline**, a DummyClassifier using the training-label prior. Then fit **model**, one Pipeline containing a ColumnTransformer and LogisticRegression. Numeric columns: median imputation, then standardization. Categorical columns: most-frequent imputation, then one-hot encoding that tolerates unseen categories. Fit only X_train and y_train. Keep default regularization; set max_iter=500.`,
    say: 'A dummy baseline tells me whether I learned more than the class frequency. Logistic regression is fast, interpretable, and a reasonable first model for mixed tabular features.',
    questions: [
      ['What does Pipeline prevent?', 'At fit, preprocessing learns from training data. At prediction, the same learned transforms are reused. In cross-validation, each fold fits its own preprocessing. It does not fix bad feature timing or an invalid split.'],
      ['Why scale numeric features for logistic regression?', 'It improves optimization and makes the regularization penalty less dependent on arbitrary feature units. Trees generally do not need this scaling.'],
      ['What is logistic regression optimizing?', 'A linear score z = w^T x + b becomes p = 1 / (1 + exp(-z)). Training minimizes binary log loss plus regularization. predict_proba produces probabilities; a threshold produces labels.'],
    ],
    solution: `numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=500)),
])
model.fit(X_train, y_train)
print("Both baselines fitted on training players only.")
`,
    checks: `assert list(model.classes_) == [0, 1]
fitted_imputer = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
np.testing.assert_allclose(fitted_imputer.statistics_, X_train[numeric_features].median().to_numpy())
print("The numeric imputer learned the training medians.")
`,
  },
  {
    title: 'Problem 2 · Step 6 — Evaluate (27–32 min)',
    task: `Create **baseline_prob** and **model_prob** for the positive class on X_test. Build a **metrics** table with accuracy, ROC-AUC, average precision (AP), precision, recall and F1. Use a fixed 0.5 threshold for label metrics; pass probabilities to ROC-AUC and AP. Show a confusion matrix with clearly labeled rows and columns. Briefly compare training and test ROC-AUC. Do not tune against the test results.`,
    say: 'I will evaluate ranking with probability scores and decision quality at a stated threshold. Accuracy alone may hide poor recall for returning players.',
    questions: [
      ['Why not compute ROC-AUC using model.predict()? ', 'Hard labels discard score ordering. Use predict_proba[:, 1] for this binary model; check classes_ to verify which column is the positive class.'],
      ['What do ROC-AUC and AP tell you?', 'ROC-AUC measures positive-versus-negative ranking across thresholds. AP summarizes precision-recall performance and its no-skill reference is the positive fraction. AP is not exactly trapezoidal PR-AUC; neither metric establishes calibration.'],
      ['What if the business needs 80% recall instead of a 0.5 threshold?', 'Choose the operating threshold using training cross-validation or a separate validation set, considering precision and cost. Then evaluate the frozen choice once on the untouched test set.'],
    ],
    solution: `baseline_prob = baseline.predict_proba(X_test)[:, 1]
model_prob = model.predict_proba(X_test)[:, 1]

def evaluate(y_true, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "precision@0.5": precision_score(y_true, predicted, zero_division=0),
        "recall@0.5": recall_score(y_true, predicted, zero_division=0),
        "f1@0.5": f1_score(y_true, predicted, zero_division=0),
    }

metrics = pd.DataFrame({
    "prior_baseline": evaluate(y_test, baseline_prob),
    "logistic_regression": evaluate(y_test, model_prob),
}).T
display(metrics.round(3))
display(pd.DataFrame(
    confusion_matrix(y_test, (model_prob >= 0.5).astype(int), labels=[0, 1]),
    index=["actual: no return", "actual: return"],
    columns=["predicted: no return", "predicted: return"],
))
print("train ROC-AUC:", round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 3))
print("test ROC-AUC:", round(roc_auc_score(y_test, model_prob), 3))
`,
    checks: `assert model_prob.shape == (len(X_test),)
assert np.isfinite(model_prob).all() and ((0 <= model_prob) & (model_prob <= 1)).all()
assert np.isclose(metrics.loc["prior_baseline", "roc_auc"], 0.5)
assert np.isclose(metrics.loc["prior_baseline", "average_precision"], y_test.mean())
print("Probability and baseline-metric checks passed. Model scores are observations, not a pass/fail target.")
`,
  },
  {
    title: 'Problem 2 · Step 7 — Predict and summarize (32–35 min)',
    task: `The next cell supplies **new_players** with already-aggregated pre-cutoff features. Compute **new_prob** using the fitted pipeline, without refitting. Include the unknown country and missing attributes. Return the positive-class probability, and give a 30-second summary of your approach, one limitation, and one next experiment.`,
    supplied: `new_players = pd.DataFrame([
    {"account_age_days": 15, "session_count": 3, "total_minutes": 45,
     "avg_minutes": 15, "completion_rate": 2/3, "country": "US", "device": "mobile"},
    {"account_age_days": np.nan, "session_count": 0, "total_minutes": 0,
     "avg_minutes": 0, "completion_rate": 0, "country": "NEW_COUNTRY", "device": np.nan},
], index=["new_player_A", "new_player_B"])
`,
    say: 'I will reuse the fitted preprocessing and classifier. The second input deliberately tests missing data and an unseen categorical value.',
    questions: [
      ['What happens to the unseen country?', 'handle_unknown="ignore" produces all zeros in that categorical feature\'s one-hot block. The model still uses other features. This avoids a crash, but does not guarantee good predictions for new populations.'],
      ['What would you try next?', 'Use a validation set or cross-validation to compare a tree-based baseline, missingness indicators, and modest feature changes. Evaluate future-cutoff data, calibration and relevant slices before claiming production readiness.'],
      ['What must training and serving share?', 'The prediction cutoff, historical window, deduplication and missing-value rules, feature schema, and fitted preprocessing/model. This prediction cell assumes upstream raw-event aggregation follows the same contract.'],
    ],
    solution: `new_prob = model.predict_proba(new_players[feature_cols])[:, 1]
display(pd.DataFrame({"probability_of_return": new_prob}, index=new_players.index))
`,
    checks: `assert new_prob.shape == (2,)
assert np.isfinite(new_prob).all() and ((0 <= new_prob) & (new_prob <= 1)).all()
print("Inference handles missing values and an unseen country.")
`,
    closing: `**Example closing statement:** "I built player-level historical features, excluded future information, and trained an imputation/encoding/scaling pipeline with logistic regression. I compared holdout ranking and threshold metrics to a prior baseline. This random split estimates performance on independent players at this cutoff, not robustness over time. My next step would be a chronological validation experiment and a simple tree-based comparison." Insert the scores from your own run; do not invent a business lift.`,
  },
];

const ending = `## If the interviewer extends the problem

1. **Predict next-week minutes instead.** This becomes regression: use next_week_minutes as the target, never as an input. Try a mean baseline and a regularized regressor; compare MAE and RMSE. Discuss the many zeros and long tail. Keep the same feature-time boundary.
2. **Many daily rows per player.** Clarify whether the goal is future predictions for existing players or generalization to unseen players. Use time, group, or a combined split accordingly; mature label windows must not reach into the evaluation period.
3. **A 100 GB session log.** Push filtering/deduplication/aggregation into a database or distributed job. Chunk sums and counts merge easily; an exact distinct-session count needs a deliberate cross-chunk deduplication strategy. Do not average chunk averages without weighting.
4. **A nearly perfect model.** First audit future columns, duplicated entities across subsets, target-derived transformations, and whether preprocessing was fitted on all data. High accuracy is not proof of a good model.

## References and provenance

The interview **format** comes from the description pasted in your [shared conversation](https://chatgpt.com/share/6ad92cab-a9dc-83e8-b72e-654063228d54). The problem, data, code and questions here are original practice material. They are not evidence that Roblox asks this exact task, and this local notebook is not connected to CodeSignal.

Implementation references: [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [pandas merge validation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html).
`;

function createNotebook(solution) {
  const cells = [];
  const add = (type, source, tags = []) => {
    const cell = {cell_type: type, id: `cell-${String(cells.length + 1).padStart(3, '0')}`,
      metadata: tags.length ? {tags} : {}, source: source.trim().split('\n').map(s => `${s}\n`)};
    if (type === 'code') Object.assign(cell, {execution_count: null, outputs: []});
    cells.push(cell);
  };
  add('markdown', intro);
  add('markdown', solution
    ? '**Reference version.** Run cells top to bottom. Code is complete; each step includes checks and expandable follow-up answers. Try codesignal_roblox_retention_practice.ipynb first if you want a timed attempt.'
    : '**Practice version.** Replace each NotImplementedError cell with your code, then run its supplied check cell. Imports, inputs and checks are provided. The intentional TODO stops mean Run All will not finish until you implement the steps. Use codesignal_roblox_retention_solution.ipynb only when reviewing.');
  add('markdown', schema);
  add('markdown', `## Provided setup\n\nOpen this notebook from the workspace root in VS Code or Jupyter. Select the Python environment at **.venv-codesignal/bin/python** (or another environment with pandas, NumPy and scikit-learn). The two CSVs live in **data/codesignal_roblox/**. No network or GPU is used during the exercise. For a fresh local setup, see **CODESIGNAL_PRACTICE.md**; the real CodeSignal environment has its own libraries.`);
  add('code', setup, ['provided']);
  steps.forEach((step, index) => {
    add('markdown', `## ${step.title}\n\n${step.task}\n\n**Say before coding:** "${step.say}"\n\n**Possible interviewer questions**\n\n${step.questions.map(([q], i) => `${i + 1}. ${q}`).join('\n')}`);
    if (step.supplied) add('code', step.supplied, ['provided']);
    add('code', solution ? step.solution : `# Step ${index + 1}: write your implementation here.\n# Create the variables named in the task above.\nraise NotImplementedError("Complete Step ${index + 1}, then run the checks below.")`, [solution ? 'solution' : 'exercise']);
    add('code', step.checks, ['check']);
    if (solution) {
      add('markdown', `<details>\n<summary>Follow-up answers</summary>\n\n${step.questions.map(([q, a], i) => `**${i + 1}. ${q}**\n\n${a}`).join('\n\n')}\n\n</details>${step.closing ? `\n\n${step.closing}` : ''}`);
    }
  });
  add('markdown', ending);
  return {
    cells,
    metadata: {
      kernelspec: {display_name: 'Python 3', language: 'python', name: 'python3'},
      language_info: {name: 'python', version: '3.9', file_extension: '.py', mimetype: 'text/x-python', pygments_lexer: 'ipython3'},
    },
    nbformat: 4, nbformat_minor: 5,
  };
}

for (const [index, file] of filenames.entries()) {
  fs.writeFileSync(path.join(root, file), `${JSON.stringify(createNotebook(index === 1), null, 1)}\n`);
  console.log(`Built ${file}`);
}
