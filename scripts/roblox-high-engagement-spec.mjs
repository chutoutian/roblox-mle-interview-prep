// Original interview practice; inputs are existing, read-only synthetic CSVs.
export default {
  slug: 'high_engagement',
  title: 'Which Players Will Play at Least Two Hours Next Week?',
  dataDir: 'data/codesignal_roblox',
  prompt: `At the beginning of day 0, a game platform wants to identify players who will play for at least **120 minutes during the next seven days**. A product team wants a useful shortlist, not simply a high overall accuracy.

**Problem 1 — Data analysis (14 minutes).** Inspect two supplied CSVs, clean historical sessions, and create one feature row per player. Include how many distinct days a player was active and how recently they had a session. Preserve players with no history and distinguish missing durations from zero activity.

**Problem 2 — Modeling (21 minutes).** Build a simple classifier and a constant baseline. Use validation data to choose a shortlist with precision of at least **30%**, containing at least **10 validation players**, while recovering as many true high-engagement players as possible. Freeze that choice and report its performance on a separate test set.

Explain your metric choices and decisions out loud. Scikit-learn is sufficient; you do not need to implement a neural network or optimizer. The 120-minute label, 30% requirement and minimum count are mock assumptions, not Roblox product rules.`,
  highlights: `**Review:** cleaning, player-level aggregation, joins, missing values, train/validation/test, preprocessing, LogisticRegression and a constant baseline.

**Practice decisions:** distinct active days versus session count, recency, precision-constrained threshold selection, AP versus ROC-AUC, and why a validation requirement might fail on test.`,
  contract: `**Prediction time:** the beginning of day 0. The seven future days are days 0–6. The historical files include extra invalid records that must not become features.

**players.csv — one row per player**

| Column | Meaning |
| --- | --- |
| user_id | Unique player key; use for joins, not as a numerical feature |
| account_age_days | Account age at prediction time; can be missing |
| country, device | Categorical attributes available at prediction time; can be missing |
| next_week_minutes | Fully observed future play time in days 0–6; used only to construct the label |
| returned_next_week | Another future outcome; never an input feature |

The target **high_engagement** is 1 when next_week_minutes is at least 120, otherwise 0. Zero future minutes is a valid negative label. All players have complete future outcomes; do not remove players just because they lack historical sessions.

**sessions.csv — one session per row, before cleaning**

| Column | Meaning |
| --- | --- |
| session_id | Session key; identical export rows can repeat |
| user_id | A player key that exists in players.csv |
| day_offset | Relative day; only -7 through -1 inclusive is usable history |
| minutes_played | Session duration; a missing or negative value means unknown, not zero |
| completed | Historical 0/1 completion indicator; not required in this exercise |

**Cleaning rules:** remove exact duplicate session rows; investigate any remaining conflicting session IDs. Keep only usable historical days. Replace negative durations with missing values, but keep those sessions.

**Historical features:** session_count counts all usable sessions; known_duration_count counts sessions with observed durations; total_minutes sums observed durations; active_days counts distinct usable day_offset values; days_since_last_session is the negated largest usable day_offset. A session on day -1 therefore has recency 1, and a session on day -7 has recency 7.

If sessions exist but every duration is unknown, total_minutes stays missing, while counts, active_days and recency are still meaningful. If no session exists, session_count, known_duration_count, total_minutes and active_days are 0; days_since_last_session stays missing because the last session time is unknown. Partially observed totals can understate true historical minutes.

Use account_age_days and these historical features as numeric inputs; country and device are categorical inputs. Exclude all IDs, high_engagement, next_week_minutes and returned_next_week from the model.

For this mock, each player appears once at a shared cutoff, and different players are independent samples from the same population. Estimate performance on held-out players from this population using approximately 60% training, 20% validation and 20% test, with similar class balance and SEED = 42. This does not establish performance in a future calendar period.

Choose a threshold on validation scores only: maximize recall subject to precision >= 0.30 and selected_count >= 10. Consider all distinct validation scores with the decision rule score >= threshold. Break equal-recall ties by choosing the larger threshold. If there is no feasible candidate, state that the requirement cannot be met; do not invent a successful policy.`,
  sourceNotes: `This is an original extension of the supplied Data Analysis & Modeling interview format. It uses the already supplied player/session fixture read-only. It is not a reported company question, an actual experiment, or evidence that sending a promotion changes play time. No new web verification or scrape is claimed.`,
  steps: [
    {
      title: 'Understand the rows and prediction time', time: '0–4',
      task: `Load **players** and **sessions**. What does a row represent in each table? Inspect types, missing values, duplicate records and suspicious durations or dates. What fraction of players meet the future 120-minute criterion? Which fields would be unavailable when the prediction is made?`,
      say: 'I will define the target and cutoff before selecting inputs or interpreting a suspicious value.',
      questions: [
        ['Is this a regression problem because the raw outcome is minutes?', 'Not for this task. The requested decision is whether future minutes reaches 120, so we construct a binary target. Predicting the number of minutes would be a different modeling objective.'],
        ['Why can we use next_week_minutes as a label but not a feature?', 'The historical export contains the completed outcome for supervision. At day 0 that outcome is not known, so using it as an input would leak the future.'],
        ['Would a high accuracy immediately show that the shortlist is useful?', 'No. With a minority positive class, predicting everybody negative can have high accuracy while identifying no high-engagement players.'],
      ],
      code: `players = pd.read_csv(DATA_DIR / "players.csv")
sessions = pd.read_csv(DATA_DIR / "sessions.csv")
display(players.head(), sessions.head())
players.info()
sessions.info()
display(players.isna().sum(), sessions.isna().sum())
print("players:", players["user_id"].nunique())
print("high-engagement fraction:", players["next_week_minutes"].ge(120).mean())
print("exact duplicate sessions:", sessions.duplicated().sum())
print("outside history:", (~sessions["day_offset"].between(-7, -1)).sum())
print("negative durations:", sessions["minutes_played"].lt(0).sum())`,
      check: `assert players["user_id"].is_unique
assert players["next_week_minutes"].notna().all()
assert players["next_week_minutes"].ge(0).all()
assert sessions["user_id"].isin(players["user_id"]).all()
assert set(players["next_week_minutes"].ge(120).astype(int)) == {0, 1}`,
    },
    {
      title: 'Clean history and construct the target', time: '4–8',
      task: `Create **clean_sessions** using the cleaning contract and **labeled_players** with the new high_engagement target. Preserve both raw tables. How will you avoid treating an unknown duration as evidence that a session never happened?`,
      say: 'I will apply fixed validity rules now, but learn imputation values only after splitting the data.',
      questions: [
        ['Why keep a session when its duration is unknown?', 'Its existence, day and player ID still provide information for session count, active days and recency.'],
        ['Why not keep day 0 through day 3 sessions if they are in the CSV?', 'They occur after the prediction cutoff and would reveal information from the outcome period.'],
        ['Does removing identical rows guarantee a unique session_id?', 'No. Two records may share an ID while disagreeing in another field. Check the key separately and investigate conflicts.'],
      ],
      code: `clean_sessions = sessions.drop_duplicates().copy()
assert clean_sessions["session_id"].is_unique, "Investigate conflicting session IDs."
clean_sessions = clean_sessions.loc[
    clean_sessions["day_offset"].between(-7, -1)
].copy()
clean_sessions["minutes_played"] = clean_sessions["minutes_played"].mask(
    clean_sessions["minutes_played"].lt(0)
)
labeled_players = players.copy()
labeled_players["high_engagement"] = labeled_players["next_week_minutes"].ge(120).astype(int)
display(labeled_players["high_engagement"].value_counts(normalize=True))`,
      check: `assert clean_sessions["session_id"].is_unique
assert clean_sessions["day_offset"].between(-7, -1).all()
assert clean_sessions["minutes_played"].dropna().ge(0).all()
np.testing.assert_array_equal(
    labeled_players["high_engagement"], players["next_week_minutes"].ge(120).astype(int)
)
assert "high_engagement" not in players.columns`,
    },
    {
      title: 'Build activity and recency features', time: '8–14',
      task: `Create **activity** with the five historical features in the contract, then attach them to labeled_players as **features**, preserving one row per player. Identify **no_history** players. Which missing values should become zero, and which should stay missing? Show a few rows and report the no-history fraction.`,
      say: 'I will preserve the player-level prediction unit and distinguish frequency, active days, recency and incomplete measurement.',
      questions: [
        ['If a player has five sessions on the same day, what are session_count and active_days?', 'The session count is five, but active_days is one. Counts measure frequency; distinct days measure spread across the week.'],
        ['Why use the largest day_offset to find the most recent session?', 'Among negative day offsets, -1 is more recent than -7. Negating the maximum gives a positive number of days since the last observed session.'],
        ['Why is no history different from all session durations being unknown?', 'No history means no session was observed in the window. Unknown durations still establish activity and recency, but do not establish a zero-minute total.'],
      ],
      code: `activity = clean_sessions.groupby("user_id").agg(
    session_count=("session_id", "size"),
    known_duration_count=("minutes_played", "count"),
    total_minutes=("minutes_played", "sum"),
    active_days=("day_offset", "nunique"),
    last_day=("day_offset", "max"),
)
activity["days_since_last_session"] = -activity["last_day"]
activity = activity.drop(columns="last_day")
activity.loc[activity["known_duration_count"].eq(0), "total_minutes"] = np.nan
features = labeled_players.merge(activity, on="user_id", how="left", validate="one_to_one")
no_history = features["session_count"].isna()
features.loc[no_history, ["session_count", "known_duration_count", "total_minutes", "active_days"]] = 0
print("no-history fraction:", no_history.mean())
display(features.head())`,
      check: `assert len(features) == len(players)
assert features["user_id"].is_unique
assert features["active_days"].between(0, 7).all()
assert features["active_days"].le(features["session_count"]).all()
assert features.loc[no_history, "total_minutes"].eq(0).all()
assert features.loc[no_history, "days_since_last_session"].isna().all()
assert features.loc[~no_history, "days_since_last_session"].between(1, 7).all()
unknown = features["session_count"].gt(0) & features["known_duration_count"].eq(0)
assert features.loc[unknown, "total_minutes"].isna().all()`,
    },
    {
      title: 'Separate fitting, decision-making and evaluation', time: '14–18',
      task: `Which columns should be inputs? Create **numeric_features**, **categorical_features**, **X** and **y**. What split would let you fit a model, choose a decision threshold, and then estimate performance on untouched players? Create **X_train / y_train**, **X_valid / y_valid** and **X_test / y_test**, approximately 60% / 20% / 20%, with similar class balance.`,
      say: 'I will reserve validation for the threshold and keep the test set out of both model fitting and policy selection.',
      questions: [
        ['Why not select the threshold using the test labels?', 'The test set would become part of policy selection, making its final result an optimistically selected estimate.'],
        ['Is a user-grouped split necessary in this fixture?', 'Each final row already represents a distinct user, so a disjoint row split is user-disjoint. Repeated user snapshots would require a split matched to the intended deployment setting.'],
        ['Does preserving class proportions make the train and test sets identical?', 'No. It approximately preserves label balance while keeping different rows in the subsets. It does not solve temporal shift or leakage.'],
      ],
      code: `numeric_features = ["account_age_days", "session_count", "known_duration_count",
                    "total_minutes", "active_days", "days_since_last_session"]
categorical_features = ["country", "device"]
X = features[numeric_features + categorical_features]
y = features["high_engagement"]
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED
)
print("train / validation / test:", len(X_train), len(X_valid), len(X_test))`,
      check: `for left, right in [(X_train, X_valid), (X_train, X_test), (X_valid, X_test)]:
    assert set(left.index).isdisjoint(right.index)
for inputs, target in [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]:
    assert inputs.index.equals(target.index)
    assert set(target) == {0, 1}
assert len(X_train) + len(X_valid) + len(X_test) == len(features)
assert not {"user_id", "high_engagement", "next_week_minutes", "returned_next_week"} & set(X.columns)`,
    },
    {
      title: 'Fit a classifier and a constant reference', time: '18–25',
      task: `Fit **model**, a LogisticRegression pipeline that handles missing numeric values and categorical inputs. Fit **baseline**, a constant-probability classifier based on training label frequencies. Which parts learn from data, and which split should they learn from? Inspect the positive-class mapping before extracting probabilities.`,
      say: 'I will start with a small model and make sure preprocessing sees training players only.',
      questions: [
        ['What does the prior baseline predict for every player?', 'The training fraction of high-engagement players. It has no ranking ability; its default class prediction is the majority training class.'],
        ['Why are the imputer and scaler fitted on training rows only?', 'Their statistics are learned information. Fitting them on validation or test would let held-out inputs influence training.'],
        ['Does setting class_weight="balanced" automatically meet the precision requirement?', 'No. It changes the objective and can change probability calibration. The operating threshold and resulting precision still require validation.'],
      ],
      code: `numeric_pipeline = Pipeline([
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
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=500, random_state=SEED)),
])
model.fit(X_train, y_train)
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
print("probability columns:", model.classes_)`,
      check: `np.testing.assert_array_equal(model.classes_, [0, 1])
np.testing.assert_allclose(baseline.class_prior_[1], y_train.mean())
learned = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
np.testing.assert_allclose(learned, X_train[numeric_features].median().to_numpy())`,
    },
    {
      title: 'Choose a precision-constrained shortlist', time: '25–31',
      task: `Create **valid_prob**. For each distinct validation score used as a threshold, calculate the selected count, precision and recall. Store the results in **threshold_table**.

Set **precision_target = 0.30** and **min_selected = 10**. Create **feasible**, the candidates meeting both requirements, then choose **threshold** with the greatest recall, breaking recall ties with the higher threshold. Use score >= threshold consistently. Explain what you would do if no candidate were feasible, and why this objective is not the same as maximizing accuracy.`,
      say: 'I want to find as many true positives as possible, but only among shortlists that meet the stated precision and size requirements.',
      questions: [
        ['Why does this task constrain precision and maximize recall?', 'The shortlist must have enough genuine high-engagement players to be useful, while covering as many eligible players as possible. These are different requirements.'],
        ['Does lowering the threshold always improve precision?', 'No. Recall and the number selected cannot decrease on fixed scores, but precision can go up or down depending on the labels of newly included examples.'],
        ['Why include at least 10 selected validation players?', 'A precision estimate from only one or two selected players is fragile. Ten is an exercise guardrail, not a statistical guarantee or a sufficient sample-size justification for production.'],
      ],
      code: `valid_prob = model.predict_proba(X_valid)[:, 1]
rows = []
for value in np.unique(valid_prob):
    decision = (valid_prob >= value).astype(int)
    rows.append({
        "threshold": value,
        "selected_count": int(decision.sum()),
        "precision": metrics.precision_score(y_valid, decision, zero_division=0),
        "recall": metrics.recall_score(y_valid, decision, zero_division=0),
    })
threshold_table = pd.DataFrame(rows)
precision_target = 0.30
min_selected = 10
feasible = threshold_table.loc[
    threshold_table["precision"].ge(precision_target)
    & threshold_table["selected_count"].ge(min_selected)
].copy()
if feasible.empty:
    raise ValueError("No feasible shortlist; discuss the requirement instead of claiming success.")
chosen = feasible.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
threshold = float(chosen["threshold"])
display(chosen.to_frame("selected_policy").T)`,
      check: `assert len(threshold_table) == len(np.unique(valid_prob))
assert chosen["precision"] >= precision_target
assert chosen["selected_count"] >= min_selected
assert chosen["recall"] == feasible["recall"].max()
best_recall = feasible.loc[feasible["recall"].eq(feasible["recall"].max())]
assert threshold == best_recall["threshold"].max()`,
    },
    {
      title: 'Evaluate ranking and the frozen decision', time: '31–35',
      task: `Create **test_prob** and **test_decision** using the frozen model and threshold. Create **scores**, comparing model and baseline on ROC-AUC, AP and log loss. Create **policy_scores**, comparing the baseline's default predictions, threshold 0.5, and the chosen threshold on accuracy, precision, recall and selected count.

Does the 30% precision target hold on the untouched test set? Report the result honestly without changing the threshold. Explain why the model with the highest accuracy need not produce the most useful shortlist.`,
      say: 'Ranking quality, probability quality and the final decision answer different questions; I will report them separately and will not retune on test.',
      questions: [
        ['Why do AP and ROC-AUC use raw probabilities rather than the final 0/1 decision?', 'They evaluate score ordering across thresholds. Binarizing the scores discards most of that ordering.'],
        ['What if validation precision is at least 30% but test precision is lower?', 'The validation requirement is not a guarantee. Report sampling variation or potential distribution shift, avoid retuning on test, and validate a revised policy on new data if needed.'],
        ['Would this show that sending a promotion to selected players increases play time?', 'No. The model predicts engagement under historical conditions, not the causal effect of a promotion. That requires an appropriately designed experiment or another defensible causal approach.'],
      ],
      code: `test_prob = model.predict_proba(X_test)[:, 1]
test_decision = (test_prob >= threshold).astype(int)
def evaluate_scores(gt, probability):
    return {
        "ROC_AUC": metrics.roc_auc_score(gt, probability),
        "AP": metrics.average_precision_score(gt, probability),
        "log_loss": metrics.log_loss(gt, probability, labels=[0, 1]),
    }
scores = pd.DataFrame({
    "prior": evaluate_scores(y_test, baseline.predict_proba(X_test)[:, 1]),
    "logistic": evaluate_scores(y_test, test_prob),
}).T
def evaluate_policy(gt, decision):
    return {
        "accuracy": metrics.accuracy_score(gt, decision),
        "precision": metrics.precision_score(gt, decision, zero_division=0),
        "recall": metrics.recall_score(gt, decision, zero_division=0),
        "selected_count": int(decision.sum()),
    }
policy_scores = pd.DataFrame({
    "prior_default": evaluate_policy(y_test, baseline.predict(X_test)),
    "threshold_0.5": evaluate_policy(y_test, (test_prob >= 0.5).astype(int)),
    "chosen_threshold": evaluate_policy(y_test, test_decision),
}).T
display(scores, policy_scores)
print("test precision target met:", policy_scores.loc["chosen_threshold", "precision"] >= precision_target)`,
      check: `assert test_prob.shape == test_decision.shape == (len(y_test),)
assert np.isfinite(test_prob).all()
assert ((test_prob >= 0) & (test_prob <= 1)).all()
np.testing.assert_array_equal(test_decision, (test_prob >= threshold).astype(int))
assert scores.loc["prior", "ROC_AUC"] == 0.5
np.testing.assert_allclose(scores.loc["prior", "AP"], y_test.mean())
assert policy_scores.loc["chosen_threshold", "selected_count"] == test_decision.sum()
# Do not assert that a validation precision requirement must also hold on test.`,
    },
  ],
  extension: {
    prompt: `1. Raise the validation precision requirement to 50%. Is there still a feasible nonempty shortlist? Describe the result without selecting a new policy on test.
2. What happens to AP and precision if high-engagement players become much rarer, even when the conditional score distributions stay the same?
3. Compare test precision and recall for players with and without history. Why would a small subgroup need uncertainty estimates?
4. Add a feature for the number of sessions in the last three historical days. How would you test its value without repeatedly checking the final test set?
5. Can a player-level random split estimate performance on next month's users and game styles? What additional data and evaluation would you need?`,
  },
};
