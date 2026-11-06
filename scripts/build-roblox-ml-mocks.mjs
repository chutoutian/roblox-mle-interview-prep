// New notebooks by default. --refresh-layout updates prose only, preserving all code and outputs.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const common = `from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, f1_score, log_loss, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, precision_score, recall_score)
SEED = 42
`;
const preprocessing = `numeric_pipeline = Pipeline([
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
])`;
const step = (title, task, say, questions, solution, check) => ({title, task, say, questions, solution, check});
const regression = {
  slug: 'regression', title: 'Predict Next-Week Play Time',
  standalone: true,
  prompt: `You work on a game platform. At the start of day 0, the team wants to estimate how many minutes each player will spend playing during the next seven days. You receive a player table and a session log.

Use only information available before day 0. Predict **next_week_minutes**, the total minutes during days 0–6. For example, a prediction of 75 means an estimated 75 minutes over that week. A player who does not play has a valid target of zero.

**Problem 1 — Data analysis (14 minutes).** Load both CSVs, inspect data quality, clean the session log, and build one row per player. Report the fraction of players without historical sessions and average observed historical play time by device.

**Problem 2 — Modeling (21 minutes).** Select features, create a train/test split, fit a simple regression model and a constant baseline, evaluate their errors, and predict for a new player. Explain your choices as you code. No neural network or from-scratch optimizer is required.`,
  schema: `**players.csv:** one row per player. Player attributes are snapshots taken before day 0. The export was assembled after every player's outcome window finished, so the target is known for training. Keep every player, including those without historical sessions.

| Column | Meaning |
| --- | --- |
| user_id | Unique player identifier; used for joins, not as a numeric predictor |
| account_age_days | Account age at the prediction cutoff; may be missing |
| country | Country category; may be missing |
| device | Device category; may be missing |
| next_week_minutes | Target: total minutes played during days 0–6; zero is valid |
| returned_next_week | Whether the player returns during days 0–6; a future outcome, not an input feature |

**sessions.csv:** one row per logged session before cleaning. A player can have multiple sessions or none.

| Column | Meaning |
| --- | --- |
| session_id | Session identifier; the export may repeat identical rows |
| user_id | Player identifier; always refers to a player in players.csv |
| day_offset | Day relative to the cutoff: -1 means yesterday, 0 means the prediction day; only -7 through -1 are valid history |
| minutes_played | Session duration; missing or a negative sentinel means unknown, not zero |
| completed | 1 if the session was completed, otherwise 0; remains valid when duration is unknown |

**Aggregation rules for this exercise:**

- Keep sessions whose durations are unknown: they still count as sessions.
- With no historical sessions, set session_count, known_duration_count and total_minutes to 0; leave avg_minutes and completion_rate missing.
- With sessions but no known durations, leave total_minutes and avg_minutes missing.
- With some known durations, total_minutes sums the observed durations only; it can understate true play time.

Distinct players at this shared cutoff are assumed to be independent samples for the mock. The random holdout estimates performance on other players from this population; it does not measure robustness to changes over future weeks.`,
  setup: common + `\nDATA_DIR = Path("data/codesignal_roblox")
assert (DATA_DIR / "players.csv").exists(), "Run with the workspace root as your working directory."
`,
  steps: [
    step('1. Load and inspect (0–4 min)',
      `Load **players** and **sessions** with pd.read_csv. Inspect their shapes, sample rows, column types and missing values. Check unique player IDs, exact duplicate sessions, negative durations and records outside days -7 through -1.

Inspect **next_week_minutes** with describe, including the 50th, 90th and 99th percentiles. Report the fraction of labels equal to zero. A high upper percentile can reveal a long tail: a small number of players with much larger values.`,
      'I will first check what each row represents, which fields are available at prediction time, and whether the target contains missing or invalid values.',
      [['What does one training example represent, and what is its target?', 'One player at the start of day 0. The target is that player\'s total minutes played during days 0–6. Session rows must be aggregated to player level before training.'], ['Can returned_next_week be a feature?', 'No. It is an outcome from the future prediction window. The offline export contains it, but the serving system would not know it at day 0.']],
      `players = pd.read_csv(DATA_DIR / "players.csv")
sessions = pd.read_csv(DATA_DIR / "sessions.csv")
players.info()
display(players.head(), sessions.head(), players.isna().sum(), sessions.isna().sum())
display(players["next_week_minutes"].describe(percentiles=[0.5, 0.9, 0.99]))
print("zero target fraction:", players["next_week_minutes"].eq(0).mean())
print("exact duplicate sessions:", sessions.duplicated().sum())
print("outside history:", (~sessions["day_offset"].between(-7, -1)).sum())`,
      `assert players["user_id"].is_unique
assert players["next_week_minutes"].notna().all()
assert players["next_week_minutes"].ge(0).all()
assert sessions["user_id"].isin(players["user_id"]).all()`),
    step('2. Clean history (4–8 min)',
      `Create **clean_sessions** without modifying **sessions**. Remove exact duplicate rows, retain day_offset values from -7 through -1 inclusive, and replace negative minutes_played with NaN. Keep rows with unknown durations.

After removing exact duplicates, check that session_id is unique. Conflicting records with the same ID require investigation rather than silently keeping an arbitrary row. Do not fill missing values with learned statistics yet.`,
      'I am applying fixed validity rules, not learning statistics from the full dataset.',
      [['Why keep a session whose duration is missing?', 'The session still happened and its completion indicator can still be useful.'], ['Why not replace missing durations with zero?', 'Unknown duration is not evidence that a session lasted zero minutes.']],
      `clean_sessions = sessions.drop_duplicates().copy()
clean_sessions = clean_sessions.loc[clean_sessions["day_offset"].between(-7, -1)].copy()
assert clean_sessions["session_id"].is_unique
clean_sessions["minutes_played"] = clean_sessions["minutes_played"].mask(
    clean_sessions["minutes_played"].lt(0)
)`,
      `assert clean_sessions["day_offset"].between(-7, -1).all()
assert clean_sessions["minutes_played"].dropna().ge(0).all()
assert clean_sessions["session_id"].is_unique`),
    step('3. Aggregate and analyze (8–14 min)',
      `Group **clean_sessions** by user_id and create **activity** with:

- **session_count:** number of sessions, including unknown durations.
- **known_duration_count:** number of nonmissing durations.
- **total_minutes:** sum of observed durations; missing if every duration is unknown.
- **avg_minutes:** mean of observed durations.
- **completion_rate:** mean of the 0/1 completed indicator.

Left join activity into players to create **features**, one row per player. Create **no_history**, a Boolean mask for players without matching sessions, and apply the aggregation rules in the data contract.

Report the no-history fraction. Group features by device and report player count, count of observed totals, and mean observed total_minutes. Include players without history and retain a missing-device group.`,
      'I need one training row per player, including people without any history.',
      [['Why a left join?', 'Keep every labeled player, including those with no matching historical session.'], ['Can a partial total understate activity?', 'Yes. Keep known_duration_count so missingness is visible; do not claim the sum is the true total.']],
      `activity = clean_sessions.groupby("user_id").agg(
    session_count=("session_id", "size"),
    known_duration_count=("minutes_played", "count"),
    total_minutes=("minutes_played", "sum"),
    avg_minutes=("minutes_played", "mean"),
    completion_rate=("completed", "mean"),
)
activity.loc[activity["known_duration_count"].eq(0), "total_minutes"] = np.nan
features = players.merge(activity, on="user_id", how="left", validate="one_to_one")
no_history = features["session_count"].isna()
features.loc[no_history, ["session_count", "known_duration_count", "total_minutes"]] = 0
display(features.groupby("device", dropna=False).agg(
    players=("user_id", "size"),
    observed_totals=("total_minutes", "count"),
    mean_observed_minutes=("total_minutes", "mean"),
))
print("no-history fraction:", no_history.mean())`,
      `assert len(features) == len(players)
assert features["user_id"].is_unique
assert features.loc[no_history, "total_minutes"].eq(0).all()
unknown = features["session_count"].gt(0) & features["known_duration_count"].eq(0)
assert features.loc[unknown, "total_minutes"].isna().all()`),
    step('4. Select features and split (14–18 min)',
      `Define **numeric_features** as account_age_days, session_count, known_duration_count, total_minutes, avg_minutes and completion_rate. Define **categorical_features** as country and device.

Create **X** using only these columns and **y** from next_week_minutes. Exclude user_id and both future-outcome columns from X. Create **X_train**, **X_test**, **y_train**, **y_test** with an 80/20 train_test_split and random_state=SEED. Do not stratify directly on the continuous target.`,
      'I will keep the holdout untouched and learn preprocessing only on training players.',
      [['Should we stratify by this continuous target?', 'Not directly: many target values are unique. A simple random split is sufficient under this mock contract.'], ['What if rows contain repeated cutoffs for the same user?', 'Choose a split matching deployment, often chronological with non-overlapping outcome windows; consider user grouping for a new-user evaluation.']],
      `numeric_features = ["account_age_days", "session_count", "known_duration_count",
                    "total_minutes", "avg_minutes", "completion_rate"]
categorical_features = ["country", "device"]
X = features[numeric_features + categorical_features]
y = features["next_week_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)`,
      `assert set(X_train.index).isdisjoint(X_test.index)
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert not {"user_id", "returned_next_week", "next_week_minutes"} & set(X.columns)`),
    step('5. Fit the model (18–26 min)',
      `Fit **baseline**, a DummyRegressor(strategy="mean"), which predicts the training-label mean for every player.

Create **numeric_pipeline** with SimpleImputer(strategy="median") followed by StandardScaler. Create **categorical_pipeline** with SimpleImputer(strategy="most_frequent") followed by OneHotEncoder(handle_unknown="ignore"). Combine them in a ColumnTransformer named **preprocess**.

Create **model**, a Pipeline with preprocess and a regressor named **regressor**, using Ridge(alpha=1.0). Fit only X_train and y_train. Ridge predicts a weighted sum of the transformed features plus an intercept; its L2 penalty discourages large coefficients.`,
      'I will compare a constant training-mean baseline with a regularized linear model. The pipeline will learn imputation, scaling and encoding from training data only.',
      [['Why scale for Ridge?', 'The penalty acts on coefficients. Comparable feature scales make regularization more comparable across features.'], ['Is the mean the best constant for every metric?', 'The mean minimizes training squared error; the median minimizes training absolute error. A median baseline is an optional MAE comparison.']],
      `${preprocessing}
model = Pipeline([("preprocess", preprocess), ("regressor", Ridge(alpha=1.0))])
model.fit(X_train, y_train)
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)`,
      `assert model.named_steps["regressor"].coef_.size > 0
learned = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
np.testing.assert_allclose(learned, X_train[numeric_features].median().to_numpy())`),
    step('6. Evaluate (26–32 min)',
      `Create **pred** and **baseline_pred** by predicting on X_test. The application requires nonnegative minutes, so apply np.maximum(prediction, 0) to both outputs as a fixed serving rule, not a choice tuned on test results.

Create a **scores** table with rows **mean_baseline** and **ridge** and columns **MAE_minutes**, **RMSE_minutes** and **R2**:

- **MAE:** average absolute prediction error, in minutes; lower is better.
- **RMSE:** square root of average squared error, in minutes; gives more weight to large errors; lower is better.
- **R2:** 1 minus model squared error divided by squared error around the test-label mean; higher is better. It can be negative.

Create **errors** containing actual_minutes, predicted_minutes and absolute_error. Display the five largest errors and give a short interpretation. Do not change hyperparameters based on this test set.`,
      'I will report errors in minutes, compare against the constant baseline, and inspect where the model misses badly. I will keep model selection separate from final test evaluation.',
      [['Can R2 be negative?', 'Yes. It means squared error is worse than predicting the test-set mean. This definition is distinct from comparing against our training-mean deployable baseline.'], ['What if the target has many zeros and a long tail?', 'Discuss a log1p target or a two-stage return-probability times conditional-duration model. Select alternatives on validation data, not this test set.']],
      `pred = np.maximum(model.predict(X_test), 0)
baseline_pred = np.maximum(baseline.predict(X_test), 0)
def evaluate_regression(actual, prediction):
    return {"MAE_minutes": mean_absolute_error(actual, prediction),
            "RMSE_minutes": root_mean_squared_error(actual, prediction),
            "R2": r2_score(actual, prediction)}
scores = pd.DataFrame({"mean_baseline": evaluate_regression(y_test, baseline_pred),
                       "ridge": evaluate_regression(y_test, pred)}).T
display(scores)
errors = pd.DataFrame({"actual_minutes": y_test, "predicted_minutes": pred})
errors["absolute_error"] = (errors["actual_minutes"] - errors["predicted_minutes"]).abs()
display(errors.nlargest(5, "absolute_error"))`,
      `assert pred.shape == (len(y_test),)
assert np.isfinite(pred).all() and (pred >= 0).all()
np.testing.assert_allclose(scores.loc["ridge", "MAE_minutes"], np.abs(y_test.to_numpy() - pred).mean())
np.testing.assert_allclose(scores.loc["ridge", "RMSE_minutes"], np.sqrt(np.mean((y_test.to_numpy() - pred) ** 2)))`),
    step('7. Inference and handoff (32–35 min)',
      `Create **new_player**, a one-row DataFrame with these already-aggregated pre-cutoff features:

| Feature | Value |
| --- | --- |
| account_age_days | np.nan |
| session_count | 3 |
| known_duration_count | 2 |
| total_minutes | 55 |
| avg_minutes | 27.5 |
| completion_rate | 2/3 |
| country | "UNSEEN_COUNTRY" |
| device | "mobile" |

Use the fitted model to create **new_prediction**, clip it at zero using the serving rule, and report the result in minutes. Do not fit again. Give a 30-second summary of the input features, split, model, evaluation, one limitation and one next experiment.`,
      'Inference reuses the training transformations. An unseen category must not crash the pipeline.',
      [['Why not call fit_transform on the new player?', 'That would replace training statistics and category definitions. Use the fitted pipeline predict method.'], ['Does better predicted play time imply a better product?', 'No. Engagement is a proxy; quality, safety and user satisfaction need separate guardrails and online evaluation.']],
      `new_player = pd.DataFrame([{
    "account_age_days": np.nan, "session_count": 3, "known_duration_count": 2,
    "total_minutes": 55, "avg_minutes": 27.5, "completion_rate": 2/3,
    "country": "UNSEEN_COUNTRY", "device": "mobile",
}])
new_prediction = np.maximum(model.predict(new_player), 0)
print("Predicted next-week minutes:", new_prediction[0])`,
      `assert new_prediction.shape == (1,)
assert np.isfinite(new_prediction).all()`),
  ],
  extension: `These questions are optional, outside the 35-minute core.

1. **MAE is the business metric. Which constant baseline would you choose?** A training-label median minimizes training absolute error. Compare it with the training-mean baseline, which minimizes squared error.
2. **A few players have extremely large targets. What would you try?** Check that the values are valid, then consider a log1p target or a different model on validation data. Evaluate final predictions in minutes; exponentiating a predicted mean log value does not generally recover the mean in the original units.
3. **Does the model work equally well across devices?** Report per-device errors together with group counts. Small groups have uncertain estimates; avoid concluding that a device causes an error difference.
4. **How would you improve the model?** Try one tree-based model on a validation split, compare it with Ridge, and keep the final test set untouched until the choice is fixed.
5. **How would you evaluate future weeks?** Collect multiple prediction cutoffs, ensure every feature precedes its cutoff, and hold out a later period with fully observed outcome windows.`,
  sources: `## References and provenance

This is an original synthetic regression exercise, not a reported Roblox question or an official CodeSignal assessment. The interview format follows the user-supplied **Data Analysis & Modeling Coding Interview** guide: 5 minutes introduction, 35 minutes for data analysis and model training, and 5 minutes Q&A. Step timings are suggested practice checkpoints.

The task, synthetic data and reference implementation are practice material. They do not describe actual Roblox player behavior or guarantee that this topic will be asked.

Implementation references: [scikit-learn model evaluation](https://scikit-learn.org/1.6/modules/model_evaluation.html), [avoiding data leakage](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage).`,
};

const softmax = {
  slug: 'softmax', title: 'Classify a Content Submission into Three Categories',
  prompt: `Before a content submission is reviewed, predict one coarse category: **clean**, **spam**, or **abuse**. Treat this as a mutually exclusive three-class teaching task, not Roblox's actual moderation policy.

**Problem 1 (14 minutes):** inspect and clean two supplied tables, join metadata, and summarize class balance by content type.
**Problem 2 (21 minutes):** train multinomial logistic regression, inspect the three probabilities, evaluate per-class errors, and score a new submission.

This mock practices tabular ML. It does not ask you to train a text encoder. Provided numeric signals stand in for features computed at upload time.`,
  schema: `The interviewer supplies **submissions.csv** (one row per submission, plus repeated export rows) and **creators.csv** (unique creator metadata). Data are synthetic. One unique creator per submission is deliberate, making a simple stratified split valid for this mock; real repeated creators may require a group split.

| Column | Meaning / policy |
| --- | --- |
| submission_id, creator_id | Join/identity columns, not numeric features |
| text_length | Number of characters at upload; negative sentinel means unknown |
| link_count | Number of links at upload |
| uppercase_ratio | Fraction of letters in uppercase, in [0,1] |
| repeated_token_ratio | Fraction of repeated tokens, in [0,1] |
| content_type | chat, description, or comment; may be missing |
| creators.account_age_days | Account age at upload; may be missing |
| label | Reviewer-provided class, available only after review |
| review_minutes | Time spent on the subsequent review; **leakage**, never a feature |

Every submission has a known label and matching creator. Drop exact duplicates; conflicting submission IDs should fail a check. Keep rows with incomplete predictors.`,
  setup: common + `
DATA_DIR = Path("data/codesignal_roblox_softmax")
assert (DATA_DIR / "submissions.csv").exists(), "Run from the workspace root."
assert (DATA_DIR / "creators.csv").exists()
`,
  steps: [
    step('1. Inspect inputs (0–4 min)',
      'Inspect submissions and creators. Count duplicate exports, invalid text lengths and missing fields. Inspect the label counts, but do not treat arbitrary category codes as ordered targets.',
      'This is single-label multiclass classification: each example belongs to exactly one of three classes.',
      [['Why not regression with clean=0, spam=1, abuse=2?', 'Those codes would invent distances and an ordering. Use a categorical target instead.'], ['Could content actually be both spam and abuse?', 'Yes. That would require a multilabel contract, typically independent sigmoid outputs, not this mutually exclusive softmax exercise.']],
      `submissions = pd.read_csv(DATA_DIR / "submissions.csv")
creators = pd.read_csv(DATA_DIR / "creators.csv")

print("submissions:", submissions.shape, "creators:", creators.shape)
submissions.info()
display(submissions.head(), creators.head(), submissions.isna().sum())
display(submissions["label"].value_counts())
print("exact duplicates:", submissions.duplicated().sum())
print("invalid length:", submissions["text_length"].lt(0).sum())`,
      `assert creators["creator_id"].is_unique
assert submissions["creator_id"].isin(creators["creator_id"]).all()
assert set(submissions["label"]) == {"clean", "spam", "abuse"}`),
    step('2. Clean and join (4–9 min)',
      'Create clean by dropping exact duplicates and replacing negative text_length with NaN. Left merge creators into frame with validate="many_to_one". Do not use review_minutes as a feature.',
      'The creator table has one row per key. Merge validation prevents accidental row multiplication.',
      [['Why many_to_one?', 'The right-side creator key must be unique. The general submission schema allows many submissions per creator, although this fixture uses one.'], ['Why is review_minutes leakage?', 'The review occurs after the moment we must produce the prediction.']],
      `clean = submissions.drop_duplicates().copy()
assert clean["submission_id"].is_unique
clean["text_length"] = clean["text_length"].mask(clean["text_length"].lt(0))
frame = clean.merge(creators, on="creator_id", how="left", validate="many_to_one")`,
      `assert len(frame) == submissions["submission_id"].nunique()
assert frame["text_length"].dropna().ge(0).all()
assert frame["submission_id"].is_unique`),
    step('3. Analyze class balance (9–14 min)',
      'Create class_fraction from the cleaned frame. Display counts by content_type and label, including missing content types. Identify the largest class and say why accuracy alone may hide poor minority recall.',
      'I will check each class instead of relying only on the overall score.',
      [['What does normalize=True change?', 'Counts become fractions of the nonmissing labels.'], ['Should we immediately balance the dataset?', 'No. Start with a baseline and per-class metrics. Discuss business costs, class weights and probability calibration before changing the training distribution.']],
      `class_fraction = frame["label"].value_counts(normalize=True)
display(class_fraction)
display(frame.groupby(["content_type", "label"], dropna=False).size().unstack(fill_value=0))`,
      `np.testing.assert_allclose(class_fraction.sum(), 1)
assert len(class_fraction) == 3`),
    step('4. Select and split (14–18 min)',
      'Use text_length, link_count, uppercase_ratio, repeated_token_ratio, account_age_days as numeric_features and content_type as categorical_features. Create an 80/20 stratified split with SEED.',
      'Stratification approximately preserves the three-class proportions; preprocessing still fits only on train.',
      [['Does stratification make the classes equally common?', 'No. It preserves their original proportions approximately.'], ['What if creators appear in both splits in production data?', 'For evaluating unseen creators, split by creator and verify disjoint groups. For future submissions, use a time-aware split matched to the intended deployment.']],
      `numeric_features = ["text_length", "link_count", "uppercase_ratio",
                    "repeated_token_ratio", "account_age_days"]
categorical_features = ["content_type"]
X = frame[numeric_features + categorical_features]
y = frame["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)`,
      `assert set(X_train.index).isdisjoint(X_test.index)
assert X_train.index.equals(y_train.index)
assert not {"label", "review_minutes", "creator_id", "submission_id"} & set(X.columns)
assert y_train.nunique() == y_test.nunique() == 3`),
    step('5. Train softmax regression (18–26 min)',
      'Use the familiar numeric and categorical pipelines. Create model with LogisticRegression(solver="lbfgs", max_iter=500), and baseline with DummyClassifier(strategy="prior"). Fit both on train only.',
      'With three classes, this solver fits multinomial logistic regression: one logit per class, followed by softmax.',
      [['What is the model mathematically?', 'For transformed X of shape (N,D), logits Z = X @ W.T + b have shape (N,3). Softmax normalizes exp(Z) across each row. The objective is multiclass cross-entropy plus regularization.'], ['Does each class probability use an independent sigmoid?', 'No. Softmax probabilities compete and sum to one. Independent sigmoids fit multilabel outcomes instead.']],
      `${preprocessing}
model = Pipeline([("preprocess", preprocess),
                  ("classifier", LogisticRegression(solver="lbfgs", max_iter=500))])
model.fit(X_train, y_train)
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
print("class order:", model.classes_)`,
      `assert model.named_steps["classifier"].coef_.shape[0] == 3
learned = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
np.testing.assert_allclose(learned, X_train[numeric_features].median().to_numpy())`),
    step('6. Evaluate probabilities and labels (26–32 min)',
      'Create prob and pred. Display the first five probability rows with model.classes_ as column names. Evaluate model and baseline with accuracy, macro-F1 and log loss. Display classification_report and a labeled 3x3 confusion matrix.',
      'Argmax selects one class. Macro-F1 weights classes equally; log loss also evaluates confidence.',
      [['Why not threshold every probability at 0.5?', 'With three classes, the highest probability may be below 0.5. Argmax still picks exactly one class under this contract.'], ['Why does log loss need probabilities?', 'It penalizes assigning low probability to the true class, especially confident wrong predictions. Accuracy discards that confidence information.']],
      `prob = model.predict_proba(X_test)
pred = model.predict(X_test)
display(pd.DataFrame(prob[:5], columns=model.classes_))
def evaluate_multiclass(estimator):
    p = estimator.predict_proba(X_test)
    labels = estimator.predict(X_test)
    return {"accuracy": accuracy_score(y_test, labels),
            "macro_f1": f1_score(y_test, labels, average="macro", zero_division=0),
            "log_loss": log_loss(y_test, p, labels=estimator.classes_)}
scores = pd.DataFrame({"prior": evaluate_multiclass(baseline),
                       "softmax": evaluate_multiclass(model)}).T
display(scores)
print(classification_report(y_test, pred, zero_division=0))
display(pd.DataFrame(confusion_matrix(y_test, pred, labels=model.classes_),
                     index=["actual: " + c for c in model.classes_],
                     columns=["predicted: " + c for c in model.classes_]))`,
      `assert prob.shape == (len(y_test), 3)
np.testing.assert_allclose(prob.sum(axis=1), 1)
np.testing.assert_array_equal(pred, model.classes_[prob.argmax(axis=1)])
true_col = pd.Index(model.classes_).get_indexer(y_test)
np.testing.assert_allclose(scores.loc["softmax", "log_loss"],
    -np.log(prob[np.arange(len(y_test)), true_col]).mean())`),
    step('7. Predict and explain (32–35 min)',
      'Create new_submission with a missing account age and unseen content_type="profile_bio". Produce new_prob and new_label. Explain how you would handle uncertain predictions without treating this toy model as a production moderation system.',
      'The model returns a category and a confidence distribution; deployment needs a separate reviewed policy.',
      [['Does a probability of 0.8 guarantee 80% correctness?', 'No. Check calibration on held-out data. Synthetic performance is not evidence of real-world moderation reliability.'], ['How might we use a human review queue?', 'Define an abstention or risk-based review policy, tune it on validation data for coverage and error costs, and audit performance across groups.']],
      `new_submission = pd.DataFrame([{
    "text_length": 120, "link_count": 2, "uppercase_ratio": 0.1,
    "repeated_token_ratio": 0.2, "account_age_days": np.nan,
    "content_type": "profile_bio",
}])
new_prob = model.predict_proba(new_submission)
new_label = model.predict(new_submission)
display(pd.DataFrame(new_prob, columns=model.classes_))
print("Predicted category:", new_label[0])`,
      `assert new_prob.shape == (1, 3)
np.testing.assert_allclose(new_prob.sum(axis=1), 1)`),
  ],
  extension: `**Optional 5-minute mathematical follow-up:** reconstruct predict_proba from logits. In the reference, the next cell verifies it numerically. This is not required in the 35-minute exercise.

For one row: subtract its maximum logit, exponentiate, then divide by the row sum. Subtracting a constant leaves softmax unchanged and avoids overflow.`,
  extensionCode: `transformed = model.named_steps["preprocess"].transform(X_test)
classifier = model.named_steps["classifier"]
logits = transformed @ classifier.coef_.T + classifier.intercept_
shifted = logits - logits.max(axis=1, keepdims=True)
exp_logits = np.exp(shifted)
manual_prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)
np.testing.assert_allclose(manual_prob, prob)
print("X:", transformed.shape, "W:", classifier.coef_.shape,
      "logits:", logits.shape, "probabilities:", manual_prob.shape)`,
};

const recommendation = {
  slug: 'recommendation', title: 'Predict Whether a Player Joins an Experience',
  standalone: true,
  practiceSetup: `# Write your imports here.
# Define SEED as 42 and DATA_DIR for the supplied CSV directory.
raise NotImplementedError("Complete setup, then start Step 1.")`,
  prompt: `At the time an experience card is shown, predict whether the player joins that experience within the next ten minutes. Each row is one displayed card (an impression), not one player.

**Problem 1 (14 minutes):** inspect and clean impression logs, attach experience metadata, and report join rate by genre.
**Problem 2 (21 minutes):** build a probability model, use a user-disjoint holdout, evaluate imbalanced classification, and score new cards.

This is a small pointwise recommendation baseline, not a full recommendation system. Ranking cards is an optional follow-up after the timed exercise.`,
  schema: `The interviewer supplies **impressions.csv** and **experiences.csv** in **data/codesignal_roblox_recommendation/**. Load these into DataFrames named **impressions** and **experiences** in Step 1. Each user has one request with four shown candidates. All ten-minute outcomes are observed. Historical affinity was computed before the request; experience attributes are snapshots available at request time.

| Column | Meaning / policy |
| --- | --- |
| impression_id | Unique displayed-card ID, except exact repeated exports |
| user_id, request_id, experience_id | Group/join identifiers, not numeric features |
| device | Device at request time, may be missing |
| prior_genre_minutes | User's minutes in that genre during the previous seven days; negatives mean unknown |
| friends_playing | Number of friends playing at request time |
| position | Display position 1–4; affects observed behavior |
| joined | 0/1 label: join this experience within ten minutes |
| minutes_after_impression | Subsequent play time; unavailable for the current prediction |
| experiences.genre, experience_age_days | Static/as-of-request experience metadata |

The evaluation question is explicitly **generalization to unseen users**. Users must not cross the train/test boundary. Multiple joins per request are possible over ten minutes; this is not a mutually exclusive softmax task. Logs contain only displayed candidates, not all possible experiences.`,
  setup: common + `
DATA_DIR = Path("data/codesignal_roblox_recommendation")
assert (DATA_DIR / "impressions.csv").exists(), "Run from the workspace root."
assert (DATA_DIR / "experiences.csv").exists()
`,
  steps: [
    step('1. Inspect impressions (0–4 min)',
      'Load impressions.csv and experiences.csv into impressions and experiences with pd.read_csv. Inspect both tables. Count users, requests, duplicate impressions, missingness and joined class balance.',
      'One example is a displayed user-experience pair. Repeated users make a naive random row split risky for our unseen-user evaluation.',
      [['What does one training example represent?', 'One game card shown to one user in one request. The joined label records whether that user entered the shown experience within ten minutes.'], ['Are unshown items negative examples?', 'No. Their outcomes are unobserved. Exposure and logging policy determine which labeled rows we see.']],
      `impressions = pd.read_csv(DATA_DIR / "impressions.csv")
experiences = pd.read_csv(DATA_DIR / "experiences.csv")
print("impressions:", impressions.shape, "experiences:", experiences.shape)
impressions.info()
display(impressions.head(), experiences.head(), impressions.isna().sum())
print("users:", impressions["user_id"].nunique())
print("requests:", impressions["request_id"].nunique())
print("exact duplicate rows:", impressions.duplicated().sum())
display(impressions["joined"].value_counts(normalize=True))`,
      `assert experiences["experience_id"].is_unique
assert impressions["experience_id"].isin(experiences["experience_id"]).all()
assert set(impressions["joined"]) == {0, 1}`),
    step('2. Clean and merge (4–9 min)',
      'Create clean: remove exact duplicates and turn negative prior_genre_minutes into NaN. Join experiences into frame using a left many-to-one merge. Do not aggregate impressions away.',
      'A popular experience can appear many times, but its metadata must match only one row per impression.',
      [['Why not one_to_one?', 'Many impressions refer to the same experience; only the experience table key must be unique.'], ['Can we use minutes_after_impression?', 'No. It is recorded after the prediction and directly exposes the outcome.']],
      `clean = impressions.drop_duplicates().copy()
assert clean["impression_id"].is_unique
clean["prior_genre_minutes"] = clean["prior_genre_minutes"].mask(clean["prior_genre_minutes"].lt(0))
frame = clean.merge(experiences, on="experience_id", how="left", validate="many_to_one")`,
      `assert len(frame) == impressions["impression_id"].nunique()
assert frame["impression_id"].is_unique
assert frame["prior_genre_minutes"].dropna().ge(0).all()`),
    step('3. Analyze join rate (9–14 min)',
      'Create genre_summary with impression_count, joins and join_rate, grouped by genre. Also report join_rate by position. Explain why observed rates are not causal effects.',
      'The mean of a 0/1 outcome is its observed rate. Group counts tell us how much evidence each rate uses.',
      [['Why include counts next to rates?', 'A rate from ten impressions is much less stable than a rate from ten thousand.'], ['Does higher join rate prove a genre is better?', 'No. User mix, item quality, position and the existing recommendation policy can confound the comparison.']],
      `genre_summary = frame.groupby("genre", dropna=False).agg(
    impression_count=("impression_id", "size"),
    joins=("joined", "sum"),
    join_rate=("joined", "mean"),
)
display(genre_summary)
display(frame.groupby("position")["joined"].agg(["size", "mean"]))`,
      `assert genre_summary["impression_count"].sum() == len(frame)
np.testing.assert_allclose(genre_summary["join_rate"], genre_summary["joins"] / genre_summary["impression_count"])`),
    step('4. Split by user (14–19 min)',
      'Which columns would you use as features, which column is the target, and which fields would you exclude? We want to evaluate how well the model predicts joins for users it has never seen during training. What kind of train/test split would you choose, and why? Implement your choice and explain how you would verify that the split matches this evaluation goal.',
      'I will clarify the evaluation goal, choose a splitting strategy, and check that the resulting sets support that goal.',
      [['Would a random split of impression rows answer this evaluation question?', 'Not reliably: rows from one user can appear in both sets. For the stated unseen-user goal, keep all impressions from each user in only one set.'], ['What would you check after splitting?', 'Check that user IDs do not overlap, X and y remain aligned, and each subset has enough positive and negative examples. A user-group split does not guarantee equal label proportions.']],
      `numeric_features = ["prior_genre_minutes", "friends_playing", "position", "experience_age_days"]
categorical_features = ["device", "genre"]
X = frame[numeric_features + categorical_features]
y = frame["joined"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=frame["user_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print("train/test positive fractions:", y_train.mean(), y_test.mean())`,
      `assert set(frame.iloc[train_idx]["user_id"]).isdisjoint(frame.iloc[test_idx]["user_id"])
assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)
assert y_train.nunique() == y_test.nunique() == 2
assert not {"user_id", "request_id", "experience_id", "impression_id", "joined", "minutes_after_impression"} & set(X.columns)`),
    step('5. Fit probability baselines (19–26 min)',
      'Create numeric_pipeline with median imputation and StandardScaler, and categorical_pipeline with most-frequent imputation and OneHotEncoder(handle_unknown="ignore"). Combine them in preprocess using ColumnTransformer. Create model as a Pipeline with preprocess and LogisticRegression(max_iter=500), and baseline as DummyClassifier(strategy="prior"). Fit both using training data only.',
      'I will start with an interpretable pointwise probability model, not a complex recommender.',
      [['Should class_weight="balanced" always be on?', 'No. It changes the effective training objective and probability interpretation. Compare business-relevant performance and calibration on validation data.'], ['Why keep position for the classification exercise?', 'Position is known when evaluating response to an already displayed card. For ranking before placement, its dependence on the decision needs separate handling.']],
      `${preprocessing}
model = Pipeline([("preprocess", preprocess),
                  ("classifier", LogisticRegression(max_iter=500))])
model.fit(X_train, y_train)
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)`,
      `np.testing.assert_array_equal(model.classes_, [0, 1])
learned = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
np.testing.assert_allclose(learned, X_train[numeric_features].median().to_numpy())`),
    step('6. Evaluate imbalanced classification (26–32 min)',
      'Create prob=model.predict_proba(X_test)[:,1] and pred at threshold 0.5. Compare baseline and model on ROC-AUC, average precision, log loss, precision and recall at 0.5. Show the confusion matrix.',
      'A model can rank useful candidates well while producing few positive decisions at the default threshold.',
      [['Why might accuracy look good for a useless model?', 'If joins are rare, predicting no join almost everywhere can achieve high accuracy.'], ['Where should we choose a different threshold?', 'On a separate validation split, using a stated cost or precision/recall goal. Keep the final test set for one evaluation.']],
      `prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
def evaluate_binary(estimator):
    p = estimator.predict_proba(X_test)[:, 1]
    labels = (p >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y_test, p),
            "average_precision": average_precision_score(y_test, p),
            "log_loss": log_loss(y_test, p, labels=[0, 1]),
            "precision@0.5": precision_score(y_test, labels, zero_division=0),
            "recall@0.5": recall_score(y_test, labels, zero_division=0)}
scores = pd.DataFrame({"prior": evaluate_binary(baseline),
                       "logistic": evaluate_binary(model)}).T
display(scores)
display(pd.DataFrame(confusion_matrix(y_test, pred, labels=[0, 1]),
    index=["actual: no join", "actual: join"],
    columns=["predicted: no join", "predicted: join"]))`,
      `assert prob.shape == (len(y_test),)
assert ((prob >= 0) & (prob <= 1)).all()
np.testing.assert_allclose(scores.loc["prior", "average_precision"], y_test.mean())
np.testing.assert_allclose(scores.loc["prior", "roc_auc"], 0.5)`),
    step('7. Score new cards (32–35 min)',
      'Create new_cards with two rows containing all input features. Use the same assumed position=1 for both candidate cards and compute join_prob. Explain why score-based ranking still needs a separate evaluation.',
      'For comparing candidates I hold position constant, but these observational probabilities are not causal estimates.',
      [['Is global ROC-AUC the same as ranking quality per request?', 'No. It compares positive and negative rows globally. Ranking evaluation compares candidates within a request using metrics such as Recall@K or NDCG@K.'], ['Can this model score every unseen experience?', 'Only if the required content and historical features are available. A full system also needs candidate retrieval, safety filtering and online evaluation.']],
      `new_cards = pd.DataFrame([
    {"prior_genre_minutes": 80, "friends_playing": 2, "position": 1,
     "experience_age_days": 200, "device": "mobile", "genre": "adventure"},
    {"prior_genre_minutes": 10, "friends_playing": 0, "position": 1,
     "experience_age_days": 20, "device": "mobile", "genre": "puzzle"},
])
join_prob = model.predict_proba(new_cards)[:, 1]
display(new_cards.assign(join_probability=join_prob))`,
      `assert join_prob.shape == (2,)
assert np.isfinite(join_prob).all()`),
  ],
  extension: `**Optional ranking follow-up (not in the 35-minute core):** score held-out requests with position held at 1, sort each request by score, and compute Recall@2 over requests with at least one observed join. State that these are outcomes under the old display policy, not unbiased counterfactual labels for a new ordering. Report how many requests were excluded because their denominator is zero. This diagnostic is not proof that re-ranking will improve joins.`,
  extensionCode: `ranking_X = X_test.copy()
ranking_X["position"] = 1
ranked = frame.iloc[test_idx][["request_id", "experience_id", "joined"]].copy()
ranked["score"] = model.predict_proba(ranking_X)[:, 1]
ranked = ranked.sort_values(["request_id", "score", "experience_id"], ascending=[True, False, True])
top_two = ranked.groupby("request_id", sort=False).head(2)
positives = ranked.groupby("request_id")["joined"].sum()
hits = top_two.groupby("request_id")["joined"].sum().reindex(positives.index, fill_value=0)
eligible = positives.gt(0)
recall_at_2 = (hits.loc[eligible] / positives.loc[eligible]).mean()
print("Observed-label Recall@2:", recall_at_2)
print("Eligible requests:", eligible.sum(), "excluded no-positive requests:", (~eligible).sum())
assert 0 <= recall_at_2 <= 1`,
};

const sources = `## Scope and sources

These are **original synthetic exercises**, not reported Roblox questions and not official CodeSignal assessments. The user-supplied **Data Analysis & Modeling Coding Interview** guide defines the format: 5 minutes intro, 35 minutes for two problems (analysis and modeling), 5 minutes Q&A; pandas, NumPy, sklearn and SciPy are available. The per-step times here are suggested practice checkpoints, not official grading criteria.

The guide does not identify a team or prescribe regression/softmax topics. We chose those to practice task-appropriate modeling and metrics. Business themes are informed by the [official PhD Early Career MLE posting](https://careers.roblox.com/jobs/8242623), which mentions engagement, safety and recommendation systems. Source checked 2026-10-22; exact interview questions are unknown.

Implementation references: [sklearn 1.6 LogisticRegression](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html), [model evaluation](https://scikit-learn.org/1.6/modules/model_evaluation.html), [data leakage](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage).
`;

const checkpoints = ['0–4', '4–9', '9–14', '14–18', '18–27', '27–32', '32–35'];
const stepNames = ['Load and inspect', 'Clean the data', 'Analyze and prepare features',
  'Select features and split', 'Preprocess and train', 'Evaluate', 'Predict and summarize'];
const templateQuestions = [
  ['What should you inspect before dropping missing rows?',
   'Check which columns are missing and what missingness means. Missing predictors do not automatically invalidate the label or the other fields.'],
  ['Why postpone median imputation until after splitting?',
   'A median is learned from data. Fit it on training rows only; the held-out rows must not influence preprocessing.'],
  ['How do you check that a merge did not multiply rows?',
   'Declare the expected key relationship with validate, then check row counts and the training-example identifier.'],
  ['How do you check that X and y still correspond?',
   'Select both from the same ordered table and apply the same split indices. Verify their indices match within each subset.'],
  ['What happens when you call model.fit on the Pipeline?',
   'The preprocessing steps fit and transform the training features. The final estimator then fits those transformed features against the training labels.'],
  ['Can you keep changing the model after looking at test metrics?',
   'Use a validation split or cross-validation for those choices. Repeatedly selecting against the test set makes its reported performance optimistic.'],
  ['What must training and inference share?',
   'The feature definitions, column meanings and fitted preprocessing. Reuse the saved pipeline rather than relearning imputation, scaling or categories.'],
];
const differences = {
  softmax: 'Keep the familiar cleaning, merge, stratified split and sklearn Pipeline. The main changes are **three labels instead of two**, **N-by-3 probabilities**, **argmax instead of a 0.5 threshold**, and **multiclass metrics**.',
  recommendation: 'Keep the familiar binary LogisticRegression Pipeline and probability metrics. The main changes are **one row per impression**, **a many-to-one metadata join**, and **splitting by user because each user has multiple rows**. Ranking remains an optional follow-up.',
};

function notebook(mock, solved) {
  const cells = [];
  function add(type, source, tag) {
    const cell = {cell_type: type, id: `${mock.slug}-${cells.length + 1}`,
      metadata: tag ? {tags: [tag]} : {}, source: source.trim() + '\n'};
    if (type === 'code') Object.assign(cell, {execution_count: null, outputs: []});
    cells.push(cell);
  }
  const formatIntro = mock.standalone
    ? 'Interview format: **5 minutes introduction + 35 minutes coding + 5 minutes Q&A**. All data are synthetic.'
    : 'This follows the same format as [Will a Player Return?](codesignal_roblox_retention_practice.ipynb): **5 minutes introduction + 35 minutes coding + 5 minutes Q&A**. All data are synthetic.';
  const comparison = mock.standalone ? '' : `**Compared with Will a Player Return?** ${differences[mock.slug]}\n\n`;
  const practiceDescription = mock.practiceSetup
    ? '**Practice version.** Write your own imports and setup, then complete the seven exercise steps. Input CSVs and per-step checks are provided. Replace each NotImplementedError with your implementation. Use the matching solution notebook when reviewing.'
    : '**Practice version.** Replace each NotImplementedError cell with your code, then run its supplied check cell. Imports, inputs and checks are provided. Run All intentionally stops at unfinished exercises. Use the matching solution notebook only when reviewing.';
  add('markdown', `# Data Analysis & Modeling: ${mock.title}\n\n**Original interview mock, not a reported Roblox question or an official CodeSignal assessment.**\n${formatIntro}\n\n## The interviewer prompt\n\n${mock.prompt}\n\n| Coding time | Task |\n| --- | --- |\n${stepNames.map((name, i) => `| ${checkpoints[i]} min | ${name} |`).join('\n')}\n\nThese are suggested checkpoints, not official sub-round timings. Ask during the introduction: What is one example? When is a prediction made? What exactly is the target?\n\n${comparison}${solved ? '**Reference version.** Run cells top to bottom. Code is complete; each step includes checks and expandable follow-up answers. Try the matching practice notebook first.' : practiceDescription}\n\n## Data contract — supplied by the interviewer\n\n${mock.schema}`);
  const setupNote = mock.slug === 'regression'
    ? 'The two input files are **data/codesignal_roblox/players.csv** and **data/codesignal_roblox/sessions.csv**. The provided cell imports libraries and sets DATA_DIR, a pathlib.Path directory object. In Step 1, use pd.read_csv(DATA_DIR / "players.csv") and pd.read_csv(DATA_DIR / "sessions.csv") to load them. Do not modify the source CSVs.'
    : mock.slug === 'softmax'
      ? 'The two input files are **data/codesignal_roblox_softmax/submissions.csv** and **creators.csv**. The provided cell imports libraries and sets DATA_DIR. Load both files with pd.read_csv in Step 1; do not modify the source CSVs.'
      : 'The two input files are **data/codesignal_roblox_recommendation/impressions.csv** and **data/codesignal_roblox_recommendation/experiences.csv**. Load them into impressions and experiences in Step 1. The source files are already supplied; data generation is not part of the exercise.';
  const ownSetup = Boolean(mock.practiceSetup && !solved);
  const setupTask = ownSetup
    ? 'Write the imports you need in the next cell. Define **SEED = 42** and **DATA_DIR** for the CSV directory. You may use a pathlib directory or ordinary string paths. Restart the kernel before attempting the exercise so imports from other notebooks do not hide missing imports. If doing a timed mock, include writing imports and setup in your 35 minutes.'
    : 'The next cell supplies imports and DATA_DIR. Start the 35-minute coding timer after setup.';
  add('markdown', `## ${ownSetup ? 'Your setup — write the imports' : 'Provided setup'}\n\nOpen this notebook from the workspace root in VS Code or Jupyter and select **.venv-codesignal/bin/python**. No network, GPU or additional package installation is required.\n\n${setupNote}\n\n${setupTask}`);
  add('code', ownSetup ? mock.practiceSetup : mock.setup, ownSetup ? 'setup-exercise' : 'provided');
  for (const [i, s] of mock.steps.entries()) {
    const questions = [...s.questions, templateQuestions[i]];
    add('markdown', `## Problem ${i < 3 ? '1' : '2'} · Step ${i + 1} — ${stepNames[i]} (${checkpoints[i]} min)\n\n${s.task}\n\n**Say before coding:** "${s.say}"\n\n**Possible interviewer questions**\n\n${questions.map(([q], j) => `${j+1}. ${q}`).join('\n')}`);
    add('code', solved ? s.solution : `# Create the variables named in the task above.\nraise NotImplementedError("Complete step ${i+1}, then run the checks below.")`, solved ? 'solution' : 'exercise');
    add('code', s.check, 'check');
    if (solved) add('markdown', `<details>\n<summary>Follow-up answers</summary>\n\n${questions.map(([q,a]) => `**${q}**\n\n${a}`).join('\n\n')}\n\n</details>`);
  }
  add('markdown', `## If the interviewer extends the problem\n\n${mock.extension}`);
  if (solved && mock.extensionCode) add('code', mock.extensionCode, 'optional');
  add('markdown', mock.sources ?? sources);
  return {cells, metadata: {kernelspec: {display_name: 'Python 3', language: 'python', name: 'python3'}, language_info: {name: 'python', version: '3.9'}}, nbformat: 4, nbformat_minor: 5};
}
const selected = process.argv.find(a => a.startsWith('--only='))?.slice('--only='.length);
if (selected && !['regression', 'softmax', 'recommendation'].includes(selected)) throw new Error('Unknown mock selection.');
const outputs = [regression, softmax, recommendation].filter(m => !selected || m.slug === selected).flatMap(m => [false,true].map(s => ({
  file: path.join(root, `codesignal_roblox_${m.slug}_${s ? 'solution' : 'practice'}.ipynb`), mock:m, solved:s,
})));
const refreshLayout = process.argv.includes('--refresh-layout');
const migrateSoftmaxCsv = process.argv.includes('--migrate-softmax-csv');
const migrateRecommendationCsv = process.argv.includes('--migrate-recommendation-csv');
if (!refreshLayout && outputs.some(o => fs.existsSync(o.file))) throw new Error('New mock notebooks already exist. Refusing to overwrite learner work.');
// Validate all targets before writing any. A layout refresh never regenerates code.
const prepared = outputs.map(o => {
  const generated = notebook(o.mock, o.solved);
  if (!refreshLayout) return {...o, result: generated};
  const existing = JSON.parse(fs.readFileSync(o.file, 'utf8'));
  const byId = new Map(existing.cells.map(c => [c.id, c]));
  const migrationSlug = migrateSoftmaxCsv && o.mock.slug === 'softmax' ? 'softmax'
    : migrateRecommendationCsv && o.mock.slug === 'recommendation' ? 'recommendation' : null;
  const allowedCodeChanges = new Set(migrationSlug
    ? [`${migrationSlug}-3`, ...(o.solved ? [`${migrationSlug}-5`] : [])] : []);
  const unchangedCodeBefore = JSON.stringify(existing.cells.filter(c => c.cell_type === 'code' && !allowedCodeChanges.has(c.id)));
  for (const cell of generated.cells) {
    const target = byId.get(cell.id);
    if (!target || target.cell_type !== cell.cell_type) throw new Error(`Structure changed in ${o.file}; review manually.`);
    if (cell.cell_type === 'markdown') target.source = cell.source;
  }
  if (migrationSlug) {
    const setupCell = byId.get(`${migrationSlug}-3`);
    if (!setupCell || !setupCell.metadata?.tags?.some(t => ['provided', 'setup-exercise'].includes(t))) throw new Error('Setup cell missing.');
    const generatedSetup = generated.cells.find(c => c.id === `${migrationSlug}-3`);
    setupCell.source = generatedSetup.source;
    setupCell.metadata = generatedSetup.metadata;
    setupCell.execution_count = null;
    setupCell.outputs = [];
    if (o.solved) {
      const firstSolution = byId.get(`${migrationSlug}-5`);
      if (!firstSolution || !firstSolution.metadata?.tags?.includes('solution')) throw new Error('Step 1 solution missing.');
      firstSolution.source = generated.cells.find(c => c.id === `${migrationSlug}-5`).source;
    }
  }
  const unchangedCodeAfter = JSON.stringify(existing.cells.filter(c => c.cell_type === 'code' && !allowedCodeChanges.has(c.id)));
  if (unchangedCodeAfter !== unchangedCodeBefore) throw new Error('Code preservation check failed.');
  return {...o, result: existing};
});
for (const o of prepared) {
  fs.writeFileSync(o.file, JSON.stringify(o.result, null, 1) + '\n');
  console.log(path.basename(o.file));
}
