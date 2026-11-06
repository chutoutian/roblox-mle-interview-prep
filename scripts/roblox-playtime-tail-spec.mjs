// Independent mock: reuse the supplied CSVs, not another notebook's variables.
export default {
  slug: 'playtime_tail',
  title: 'Predict Weekly Play Time: Zeros and a Long Tail',
  dataDir: 'data/codesignal_roblox',
  prompt: `At the beginning of day 0, a game platform wants to predict each player's total play time during the next seven days. Some players will play zero minutes; a few may play much more than most others.

**Problem 1 — Data analysis (14 minutes).** Inspect two CSVs, clean the historical sessions, and create one training row per player. Explain missing-duration handling and summarize the target distribution without discarding legitimate high-play-time users.

**Problem 2 — Modeling (21 minutes).** Compare a simple regression model with a model that learns a transformed target. How would you decide which one to use without repeatedly checking the test set? Include a constant baseline, evaluate predictions in minutes, and demonstrate prediction for a new player.

Explain your assumptions and choices while coding. A scikit-learn solution is sufficient; no neural network or custom optimizer is required.`,
  contract: `**Prediction cutoff:** the beginning of day 0. Attributes are snapshots available at that time. Each player appears once. The export includes completed future outcomes for supervised training, but those outcomes cannot be input features.

**players.csv — one row per player**

| Column | Meaning |
| --- | --- |
| user_id | Unique player ID, for joining and checking identity, not a numerical feature |
| account_age_days | Account age at the cutoff; may be missing |
| country, device | Categorical attributes at the cutoff; may be missing |
| next_week_minutes | Target: total play time during days 0–6; nonnegative and observed for every player |
| returned_next_week | A future outcome from days 0–6; exclude it from features |

**sessions.csv — one historical session per row before cleaning**

| Column | Meaning |
| --- | --- |
| session_id | Session ID; exact duplicate export rows may occur |
| user_id | Player ID, always belonging to players.csv |
| day_offset | Day relative to the cutoff; only -7 through -1 inclusive is usable history |
| minutes_played | Duration; a negative sentinel or missing value means unknown, not zero |
| completed | Observed 0/1 completion indicator, valid even when duration is unknown |

**Fixed cleaning and aggregation rules:** Remove exact duplicate session rows, filter to the history window, and turn negative durations into missing values. Keep sessions whose durations are unknown. If an ID still has conflicting rows after exact deduplication, stop and investigate rather than choosing an arbitrary version.

Build session_count (all sessions), known_duration_count (nonmissing durations), total_minutes (sum of observed durations), avg_minutes (mean of observed durations), and completion_rate (mean completed). With sessions but no known durations, total_minutes and avg_minutes stay missing. With no historical sessions, set session_count, known_duration_count, and total_minutes to zero; leave avg_minutes and completion_rate missing. A partial observed total can understate actual history.

Preserve every labeled player. Zero targets are valid. An unusually large nonnegative target is not automatically an error; do not cap or remove authentic high targets merely because they are rare.

For this mock, distinct players at a shared cutoff are independent samples from the same population. Use approximately 60% training, 20% validation, and 20% final test, with SEED = 42. This split evaluates other players from this population, not robustness to a future calendar period. Select by validation MAE in original minutes. Inspect the final test only after the choice is fixed.`,
  highlights: `**Familiar:** missing-value checks, Boolean filtering, groupby/agg, left merge, Pipeline, Ridge, MAE and RMSE.

**New:** a three-way split, validation-based model selection, and explicit target transformations with np.log1p/np.expm1. A target transformation is a candidate, not a guaranteed improvement.`,
  sourceNotes: `This original practice task is inspired by collected report topics: 1161498 (feature construction and evaluation), 1124594 (classification/regression fundamentals), and 995525 (play-time objectives; a more senior interview). The exact CSV task, long-tail comparison, target transformation, and three-way split are teaching adaptations, not a claimed verbatim Roblox question.`,
  steps: [
    {
      title: '1. Understand the rows and inspect the target',
      time: '0–4 min',
      task: `What does one row represent in each file? Load **players** and **sessions**, inspect missingness and suspicious session records, and summarize **next_week_minutes**, including upper percentiles and the fraction equal to zero. Which available columns would be unavailable at prediction time?`,
      say: 'I will establish the prediction cutoff, inspect the data contract, and separate legitimate variation from invalid records.',
      questions: [
        ['Does a high 99th percentile mean most players play that much?', 'No. It describes an upper-tail boundary: roughly 99% are at or below it. A large gap from the median suggests a relatively small high-play-time tail.'],
        ['Should zero targets or unusually high targets be removed?', 'Zero is a valid outcome. Large values need domain and logging checks, not automatic removal. Removing authentic high users changes the prediction problem.'],
        ['Why exclude returned_next_week even if it correlates strongly with play time?', 'It is only known after the future outcome window. Using it would leak information unavailable at the cutoff.'],
      ],
      code: `players = pd.read_csv(DATA_DIR / "players.csv")
sessions = pd.read_csv(DATA_DIR / "sessions.csv")
display(players.head(), sessions.head())
players.info()
display(players.isna().sum(), sessions.isna().sum())
display(players["next_week_minutes"].describe(percentiles=[0.5, 0.9, 0.99]))
print("zero target fraction:", players["next_week_minutes"].eq(0).mean())
print("exact duplicate sessions:", sessions.duplicated().sum())
print("outside history:", (~sessions["day_offset"].between(-7, -1)).sum())
print("negative durations:", sessions["minutes_played"].lt(0).sum())`,
      check: `assert players["user_id"].is_unique
assert players["next_week_minutes"].notna().all()
assert players["next_week_minutes"].ge(0).all()
assert sessions["user_id"].isin(players["user_id"]).all()`,
    },
    {
      title: '2. Clean the usable history',
      time: '4–8 min',
      task: `Create **clean_sessions** using the fixed validity rules. How will you distinguish an unknown duration from a session that did not happen? Preserve the raw tables and verify that session IDs are unique after exact deduplication.`,
      say: 'I will apply deterministic validity rules now and leave learned imputation statistics until after the split.',
      questions: [
        ['Why not drop every row containing a missing value?', 'A session with unknown duration still contributes to session count and completion rate. Dropping it would discard valid information and potentially bias the sample.'],
        ['Why filter out sessions on day 0 or later?', 'They occur during or after the prediction window begins. Only history available before the cutoff can be used.'],
        ['Does drop_duplicates resolve two rows with the same session ID but different durations?', 'No. It removes identical rows by default. Conflicting versions require an authoritative resolution rule or investigation; a uniqueness check catches that separate problem.'],
      ],
      code: `clean_sessions = sessions.drop_duplicates().copy()
assert clean_sessions["session_id"].is_unique, "Investigate conflicting session IDs."
clean_sessions = clean_sessions.loc[
    clean_sessions["day_offset"].between(-7, -1)
].copy()
clean_sessions["minutes_played"] = clean_sessions["minutes_played"].mask(
    clean_sessions["minutes_played"].lt(0)
)`,
      check: `assert clean_sessions["day_offset"].between(-7, -1).all()
assert clean_sessions["minutes_played"].dropna().ge(0).all()
assert clean_sessions["session_id"].is_unique
assert len(clean_sessions) <= len(sessions)`,
    },
    {
      title: '3. Build one feature row per player',
      time: '8–14 min',
      task: `Create **activity** with the five historical aggregates in the contract, then merge it into players as **features**. Keep players without history. Explain which missing aggregates become zero and which remain unknown. Report the fraction with no history.`,
      say: 'I will aggregate before joining so that each label still corresponds to exactly one player row.',
      questions: [
        ['What is the difference between size and count for session duration?', 'size counts all rows, including sessions with missing duration; count counts nonmissing durations. We need both to distinguish no activity from incomplete measurement.'],
        ['Why correct the sum when every duration is missing?', 'The default sum can return zero for an all-missing group. That would incorrectly claim no play time was observed as a true zero total.'],
        ['Why retain a player without historical sessions?', 'They still have a valid future label and may appear at inference. Their absence of history is a useful case, not a reason to remove the player.'],
      ],
      code: `activity = clean_sessions.groupby("user_id").agg(
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
print("no-history fraction:", no_history.mean())
display(features.head())`,
      check: `assert len(features) == len(players)
assert features["user_id"].is_unique
assert features.loc[no_history, "total_minutes"].eq(0).all()
assert features.loc[no_history, ["avg_minutes", "completion_rate"]].isna().all().all()
unknown = features["session_count"].gt(0) & features["known_duration_count"].eq(0)
assert features.loc[unknown, "total_minutes"].isna().all()`,
    },
    {
      title: '4. Protect a final test set',
      time: '14–18 min',
      task: `Which features are available at the cutoff? Create **X** and **y**, then separate approximately 60% training, 20% validation, and 20% final test. Name the six subsets **X_train**, **X_val**, **X_test**, **y_train**, **y_val**, and **y_test**. Explain why two subsets are not enough when choosing between models.`,
      say: 'I will use validation to choose the model and reserve the test set for one final estimate after that decision.',
      questions: [
        ['Why is the second split 25% rather than 20%?', 'After holding out 20% for test, 80% remains. Taking 25% of that remainder gives 20% of the original data for validation, leaving 60% for training.'],
        ['Why not choose the transformation based on the best test score?', 'Then the test set participates in model selection and the reported score is optimistically selected. Validation should make the choice; test should only evaluate the frozen choice.'],
        ['Would this random split answer a future-week deployment question?', 'Not necessarily. A shared-cutoff random split tests other players in this population. Repeated user snapshots and future-time deployment require an appropriate chronological and possibly grouped split.'],
      ],
      code: `numeric_features = ["account_age_days", "session_count", "known_duration_count",
                    "total_minutes", "avg_minutes", "completion_rate"]
categorical_features = ["country", "device"]
X = features[numeric_features + categorical_features]
y = features["next_week_minutes"]
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=SEED
)
print("train / validation / test:", len(X_train), len(X_val), len(X_test))`,
      check: `assert set(X_train.index).isdisjoint(X_val.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_val.index).isdisjoint(X_test.index)
assert len(X_train) + len(X_val) + len(X_test) == len(X)
assert X_train.index.equals(y_train.index)
assert X_val.index.equals(y_val.index)
assert X_test.index.equals(y_test.index)
assert not {"user_id", "returned_next_week", "next_week_minutes"} & set(X.columns)`,
    },
    {
      title: '5. Compare raw and log-target regression on validation',
      time: '18–26 min',
      task: `Fit a constant baseline and two fresh Ridge models: one learns minutes directly, while the other learns log1p(minutes). Write the target transformation and inverse transformation explicitly: fit the log model on np.log1p(y_train), then apply np.expm1 to its validation predictions to recover minutes. Do not use a target-transform wrapper.

Create **make_ridge**, **raw_ridge**, **log_ridge**, **candidates**, **baseline**, **evaluate**, and a **validation_scores** table. Keep **raw_val_pred** and **log_val_pred** in original minutes. Use MAE in minutes as the primary selection metric; also report RMSE and R2. Fit all preprocessing on training players only. Do not assume the transformed target will win.`,
      say: 'I will compare both target choices in the same original units; a log-space loss by itself cannot decide which model best meets a minutes-based objective.',
      questions: [
        ['Why log1p rather than log?', 'log(0) is undefined, while log1p(0) is zero. log1p(y) compresses large positive values without removing valid zero labels. Its inverse is expm1.'],
        ['Does inverse-transforming a log prediction recover the conditional mean minutes?', 'Not generally: exp(E[log(1+Y) | X]) - 1 is not E[Y | X]. Nonlinearity and model error matter. Treat inverse-transformed predictions as candidates and validate them against the actual business metric.'],
        ['Why use a median constant baseline when choosing by MAE?', 'The training median minimizes training absolute error among constants; the training mean instead minimizes squared error. Both are legitimate references, but their optimization targets differ.'],
      ],
      code: `def make_ridge():
    numeric_pipeline = Pipeline([
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
    return Pipeline([("preprocess", preprocess), ("regressor", Ridge(alpha=1.0))])


def evaluate(gt, prediction):
    return {
        "MAE_min": metrics.mean_absolute_error(gt, prediction),
        "RMSE_min": metrics.root_mean_squared_error(gt, prediction),
        "R2": metrics.r2_score(gt, prediction),
    }


baseline = DummyRegressor(strategy="median").fit(X_train, y_train)

raw_ridge = make_ridge()
raw_ridge.fit(X_train, y_train)
raw_val_pred = raw_ridge.predict(X_val)

log_ridge = make_ridge()
y_train_log = np.log1p(y_train)
log_ridge.fit(X_train, y_train_log)
log_val_pred = np.expm1(log_ridge.predict(X_val))

candidates = {"raw_ridge": raw_ridge, "log_ridge": log_ridge}
validation_scores = pd.DataFrame({
    "median_baseline": evaluate(y_val, baseline.predict(X_val)),
    "raw_ridge": evaluate(y_val, raw_val_pred),
    "log_ridge": evaluate(y_val, log_val_pred),
}).T
display(validation_scores)`,
      check: `assert set(validation_scores.index) == {"median_baseline", "raw_ridge", "log_ridge"}
assert np.isfinite(validation_scores.to_numpy()).all()
raw_imputer = candidates["raw_ridge"].named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
np.testing.assert_allclose(raw_imputer.statistics_, X_train[numeric_features].median().to_numpy())
log_pipeline = candidates["log_ridge"]
log_imputer = log_pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
np.testing.assert_allclose(log_imputer.statistics_, X_train[numeric_features].median().to_numpy())
np.testing.assert_allclose(raw_val_pred, candidates["raw_ridge"].predict(X_val))
np.testing.assert_allclose(log_val_pred, np.expm1(candidates["log_ridge"].predict(X_val)))
np.testing.assert_allclose(np.expm1(np.log1p([0.0, 1.0, 100.0])), [0.0, 1.0, 100.0])`,
    },
    {
      title: '6. Freeze the choice and evaluate the held-out test',
      time: '26–32 min',
      task: `Choose **best_name** using validation MAE among the two learned models. Create **use_log_target** to record whether the chosen model learns a log target. Refit a fresh **final_model** on the combined training and validation players: pass np.log1p(y_dev) if it is the log model, otherwise pass y_dev. Convert log-model predictions back with np.expm1 before evaluation.

Evaluate once on the held-out test. Refit a **final_baseline** on the same development players and create **test_scores**. What would you conclude if the baseline is competitive?`,
      say: 'The choice is fixed before I inspect test labels. I can now refit on all development data, but I will not tune further based on this test result.',
      questions: [
        ['What if raw Ridge has better RMSE but log-target Ridge has better MAE?', 'That is a real trade-off: RMSE weights large errors more heavily. For this mock, the primary metric was fixed as MAE. If the business instead prioritizes heavy-user error, agree on that objective before selection.'],
        ['What if the chosen model does not beat the constant baseline?', 'Report it honestly. Check feature signal, target quality, split assumptions, and error segments. Do not claim a successful improvement or switch to another model because of the final test.'],
        ['Can we clamp negative predictions to zero?', 'That is a plausible serving constraint, but it is a modeling/postprocessing choice. Specify it before selection and apply it consistently in validation and test. This reference reports the original un-clipped predictions.'],
      ],
      code: `best_name = validation_scores.loc[list(candidates), "MAE_min"].idxmin()
print("selected using validation MAE:", best_name)
use_log_target = best_name == "log_ridge"
final_model = make_ridge()
if use_log_target:
    final_model.fit(X_dev, np.log1p(y_dev))
else:
    final_model.fit(X_dev, y_dev)
final_baseline = DummyRegressor(strategy="median").fit(X_dev, y_dev)
test_pred = final_model.predict(X_test)
if use_log_target:
    test_pred = np.expm1(test_pred)
test_scores = pd.DataFrame({
    "median_baseline": evaluate(y_test, final_baseline.predict(X_test)),
    best_name: evaluate(y_test, test_pred),
}).T
display(test_scores)`,
      check: `assert best_name in candidates
assert best_name == validation_scores.loc[list(candidates), "MAE_min"].idxmin()
assert use_log_target == (best_name == "log_ridge")
assert set(X_dev.index) == set(X_train.index) | set(X_val.index)
assert set(X_dev.index).isdisjoint(X_test.index)
assert test_pred.shape == y_test.shape
assert np.isfinite(test_pred).all()
assert np.isfinite(test_scores.to_numpy()).all()`,
    },
    {
      title: '7. Predict for a new player and explain the result',
      time: '32–35 min',
      task: `Create **new_player**, a single feature row with a missing account age and a previously unseen country. Obtain **new_prediction** using the fitted final model. If use_log_target is true, apply np.expm1 exactly once so the final prediction is in minutes; leave a raw-model prediction unchanged. Explain what preprocessing is reused.`,
      say: 'Serving must reuse exactly the fitted preprocessing and the chosen target transformation; prediction should not learn anything from the incoming row.',
      questions: [
        ['Should the new row be imputed with its own median or fitted again?', 'No. Reuse the training/development-fitted imputer, scaler and encoder through predict. Fitting on incoming rows changes the learned transformation and breaks consistency.'],
        ['When should we apply expm1 to final_model.predict?', 'Only when the fitted model learned log1p(minutes). This plain Ridge pipeline returns values in the units of its training target, so the log model needs one explicit inverse transformation. Raw Ridge already predicts minutes. Never apply expm1 twice.'],
        ['What new situation would make you reconsider this evaluation?', 'A future-week shift, changed logging, or repeated user snapshots can invalidate the shared-cutoff assumptions. Redesign the split and monitor relevant segments rather than assuming this random holdout guarantees future performance.'],
      ],
      code: `new_player = pd.DataFrame([{
    "account_age_days": np.nan,
    "session_count": 3,
    "known_duration_count": 2,
    "total_minutes": 55.0,
    "avg_minutes": 27.5,
    "completion_rate": 2 / 3,
    "country": "UNSEEN_COUNTRY",
    "device": "mobile",
}])
new_prediction = final_model.predict(new_player)
if use_log_target:
    new_prediction = np.expm1(new_prediction)
print("predicted next-week minutes:", new_prediction[0])`,
      check: `assert new_prediction.shape == (1,)
assert np.isfinite(new_prediction).all()
assert set(new_player.columns) == set(X.columns)`,
    },
  ],
  extension: {
    prompt: `Optional discussion after the timed mock: a single overall metric can hide errors for heavy users. Using validation predictions only, compare absolute errors for zero, ordinary-positive, and high-play-time targets. Define the high-play-time cutoff using training labels, not the test set. Why might a model with the lowest overall MAE still be unsuitable if heavy-user underprediction is expensive? Do not revise the already reported final test result after this exploration.`,
    code: `high_cutoff = y_train.quantile(0.9)
validation_pred = candidates[best_name].predict(X_val)
if best_name == "log_ridge":
    validation_pred = np.expm1(validation_pred)
validation_errors = pd.DataFrame({
    "actual": y_val,
    "predicted": validation_pred,
})
validation_errors["segment"] = np.select(
    [validation_errors["actual"].eq(0), validation_errors["actual"].gt(high_cutoff)],
    ["zero", "high"], default="positive_not_high",
)
validation_errors["absolute_error"] = (
    validation_errors["actual"] - validation_errors["predicted"]
).abs()
display(validation_errors.groupby("segment").agg(
    players=("actual", "size"),
    MAE_min=("absolute_error", "mean"),
))`,
  },
};
