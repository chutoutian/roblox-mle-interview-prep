export default {
  slug: 'review_queue',
  title: 'Which Submissions Should Enter Human Review?',
  dataDir: 'data/codesignal_roblox_softmax',
  prompt: `You work on a platform's content-review team. At submission time, a model should decide which submissions to send to a human reviewer. This is a queueing decision, not an automatic deletion or account ban.

**Problem 1 — Data analysis (14 minutes).** Load the submission and creator tables, inspect data quality, clean fixed validity errors, and build one row per submission. How common is the positive class, and does its observed rate vary by content type?

**Problem 2 — Modeling (21 minutes).** Train a simple binary classifier and a constant baseline. The team's practice requirement is to catch **at least 80% of abusive submissions on validation**, while making the review queue as small as possible among the model's available score thresholds. How would you select a threshold and evaluate it honestly? Report final test ranking metrics, precision, recall and the fraction of submissions entering review.

Explain your assumptions and choices out loud. Use a train/validation/test split; do not use test results to change the threshold. The 80% requirement is an exercise assumption, not a Roblox policy.`,
  highlights: `**What this mock exercises:** familiar cleaning, many-to-one joins, binary LogisticRegression and probability metrics; a separate validation set; **precision_recall_curve**; turning probabilities into a review decision.

This dataset is not assumed to have an extremely rare positive class: measure its actual prevalence. A useful classifier and a useful operating threshold are different things.`,
  contract: `The files are already supplied. Do not generate data or modify the source CSVs.

**submissions.csv — one logged submission per row before cleaning.**

| Column | Meaning |
| --- | --- |
| submission_id | Event identifier; identical exported rows can repeat |
| creator_id | Key into creators.csv |
| text_length | Number of characters; negative sentinel means unknown |
| link_count | Number of links |
| uppercase_ratio | Uppercase share, between 0 and 1 |
| repeated_token_ratio | Repetition share, between 0 and 1 |
| content_type | chat, description or comment; can be missing |
| label | Historical completed review: clean, spam or abuse |
| review_minutes | Time spent reviewing, available only after submission; not an input |

**creators.csv — one row per creator:** creator_id and account_age_days, measured at submission time; account age may be missing.

**Target for this exercise:** needs_review = 1 exactly when label is **abuse**, otherwise 0. Treating spam as negative is only a simplified abuse-specific queue definition, not a real moderation policy. Assume every historical row received a completed review, so the labels are available for training without selecting only previously flagged submissions.

All listed predictor fields are available at submission time. Exclude IDs, label, needs_review and review_minutes from model inputs. Each creator has one distinct submission in this fixture, so disjoint submission splits are also creator-disjoint. A future dataset with repeated creators would need a split matched to the deployment goal.

Negative text lengths become missing; keep their rows. Unknown predictors are not zero and do not make a valid label unusable. Fit imputers, scalers and category vocabularies on training data only.`,
  steps: [
    {
      title: 'Load the inputs and define a positive example', time: '0–4',
      task: `Load **submissions** and **creators**. What does each row represent? Inspect shapes, types, missing values, label counts, exact duplicate rows and invalid text lengths. Confirm which fields are known when a new submission arrives.`,
      say: 'I will define the positive class and prediction time before choosing a model or dropping rows.',
      questions: [
        ['What is a positive example in this task?', 'A submission whose completed historical label is abuse. The model will estimate that probability at submission time.'],
        ['Does having a label in the training file make it a valid feature?', 'No. A historical outcome is known offline but not when the live decision is made.'],
        ['What would change if only previously flagged submissions were reviewed?', 'The labels would be selectively observed. Offline evaluation on that sample would not directly estimate performance on all submissions; collect a suitable random audit sample or address selection bias.'],
      ],
      code: `submissions = pd.read_csv(DATA_DIR / "submissions.csv")
creators = pd.read_csv(DATA_DIR / "creators.csv")
submissions.info()
display(submissions.head(), creators.head())
display(submissions.isna().sum(), creators.isna().sum())
display(submissions["label"].value_counts())
print("exact duplicate rows:", submissions.duplicated().sum())
print("negative text lengths:", submissions["text_length"].lt(0).sum())`,
      check: `assert creators["creator_id"].is_unique
assert submissions["label"].notna().all()
assert set(submissions["label"]) == {"clean", "spam", "abuse"}
assert submissions["creator_id"].isin(creators["creator_id"]).all()`,
    },
    {
      title: 'Clean without deleting useful examples', time: '4–8',
      task: `Create **clean** without changing submissions. Remove identical exported rows. Check that submission_id is now unique; stop and investigate if it is not. Replace invalid negative text_length values with missing values, keeping those submissions. Do not learn imputation values yet.`,
      say: 'I will separate fixed validity rules from preprocessing statistics that must be fitted on training data.',
      questions: [
        ['Why not drop every row with a missing field?', 'A row may still contain a reliable label and several useful features. Blanket deletion wastes data and can change the population.'],
        ['Why is missing text length different from a length of zero?', 'Missing means unknown. Zero would assert that the text was empty.'],
        ['Can removing exact duplicate rows guarantee unique event IDs?', 'No. Two rows with the same event ID can disagree in another column. Check the key separately.'],
      ],
      code: `clean = submissions.drop_duplicates().copy()
assert clean["submission_id"].is_unique, "Investigate conflicting event records."
clean["text_length"] = clean["text_length"].mask(clean["text_length"].lt(0))
display(clean.loc[clean["text_length"].isna()].head())`,
      check: `assert clean["submission_id"].is_unique
assert clean["text_length"].dropna().ge(0).all()
assert len(clean) == len(submissions.drop_duplicates())
assert clean["text_length"].isna().sum() >= submissions["text_length"].lt(0).sum() - submissions.duplicated().sum()`,
    },
    {
      title: 'Join metadata and analyze the queue population', time: '8–14',
      task: `Create **frame** with one row per submission and creator metadata attached. Define **needs_review** from the historical label. Report its prevalence overall and a **content_summary** with submission counts and abuse rates by content_type, retaining the missing-type group. Explain what these rates can and cannot establish.`,
      say: 'I will preserve the submission unit and check that the metadata join does not multiply examples.',
      questions: [
        ['Why validate this join as many-to-one?', 'Submission rows may share a creator, but the creator table must have one row per key. This prevents an accidental many-to-many expansion.'],
        ['Does a higher abuse rate for one content type prove that the type causes abuse?', 'No. These are descriptive observational rates. Other user and content differences may explain them.'],
        ['Would accuracy alone tell us whether this review queue is useful?', 'No. Compare against prevalence and a constant baseline, then inspect ranking quality, missed abuse and review workload.'],
      ],
      code: `frame = clean.merge(creators, on="creator_id", how="left", validate="many_to_one")
frame["needs_review"] = frame["label"].eq("abuse").astype(int)
content_summary = frame.groupby("content_type", dropna=False).agg(
    submissions=("submission_id", "size"),
    abuse_rate=("needs_review", "mean"),
)
print("positive fraction:", frame["needs_review"].mean())
display(content_summary)`,
      check: `assert len(frame) == len(clean)
assert frame["submission_id"].is_unique
np.testing.assert_array_equal(frame["needs_review"], frame["label"].eq("abuse").astype(int))
assert content_summary["submissions"].sum() == len(frame)`,
    },
    {
      title: 'Separate training, threshold selection and final evaluation', time: '14–18',
      task: `Which columns should be features, and which should be excluded? Create **numeric_features**, **categorical_features**, **X** and **y**. Create disjoint **X_train / y_train**, **X_valid / y_valid** and **X_test / y_test**, approximately 60% / 20% / 20%, preserving class balance. Explain the role of each set.`,
      say: 'Training fits the model, validation chooses the threshold, and the untouched test set estimates the final policy.',
      questions: [
        ['Why does this task need validation as well as test?', 'The threshold is a learned decision. Choosing it on test labels would make the final test report optimistic.'],
        ['Why stratify these splits?', 'Maintain approximately similar class proportions across subsets, especially when the positive class is smaller. It does not solve dependence or time leakage.'],
        ['What if the deployment goal were performance on next month\'s submissions?', 'Use a chronological evaluation with features and fully matured labels available at each cutoff. A random split answers a different question.'],
      ],
      code: `numeric_features = ["text_length", "link_count", "uppercase_ratio",
                    "repeated_token_ratio", "account_age_days"]
categorical_features = ["content_type"]
X = frame[numeric_features + categorical_features]
y = frame["needs_review"]
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
assert len(X_train) + len(X_valid) + len(X_test) == len(frame)
assert not {"label", "needs_review", "review_minutes", "creator_id", "submission_id"} & set(X.columns)`,
    },
    {
      title: 'Fit a classifier and a constant baseline', time: '18–25',
      task: `Fit **model**, a LogisticRegression pipeline with numeric imputation/scaling and categorical imputation/encoding. Fit **baseline**, a DummyClassifier using training class frequencies. Use training data only. What objective is the classifier minimizing, and why is the constant model a useful comparison?`,
      say: 'I will start with a small interpretable model and keep every learned preprocessing statistic inside its pipeline.',
      questions: [
        ['What does LogisticRegression learn?', 'A weighted sum plus an intercept, converted to a positive-class probability by sigmoid. With default L2 settings it minimizes binary cross-entropy plus a weight penalty.'],
        ['Does class_weight="balanced" remove the need to choose a threshold?', 'No. It changes the training objective and can change the probability scale. Threshold and probability quality still need validation. We do not turn it on automatically here.'],
        ['What does the prior baseline predict?', 'The same training positive-class fraction for every row; it has no ability to rank submissions.'],
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
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)`,
      check: `np.testing.assert_array_equal(model.classes_, [0, 1])
learned_medians = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
np.testing.assert_allclose(learned_medians, X_train[numeric_features].median().to_numpy())
np.testing.assert_allclose(baseline.class_prior_[1], y_train.mean())`,
    },
    {
      title: 'Choose the review threshold on validation', time: '25–31',
      task: `Create **valid_prob**. Build a **threshold_table** describing precision and recall at the available validation score thresholds. Choose **threshold** to meet validation recall of at least **recall_target = 0.80**, while sending as few validation submissions as possible for review.

What does lowering the threshold do to queue volume? Report **validation_decision**, the selected precision, recall and review fraction. Use validation only. If the requirement cannot be evaluated or no candidate is feasible, explain that rather than inventing a successful result.`,
      say: 'I am choosing an operating point for a stated review objective; this is not a promise that future recall will equal validation recall.',
      questions: [
        ['How does precision_recall_curve align its outputs?', 'Precision and recall have one extra terminal entry with no associated threshold. Pair thresholds with precision[:-1] and recall[:-1].'],
        ['Why choose the largest threshold that still meets the recall requirement?', 'For decisions score >= threshold on these fixed validation scores, a larger threshold selects a subset and cannot increase review volume. Ties are kept together. This minimizes volume among the available score thresholds, not necessarily maximizes precision.'],
        ['Would you use the same threshold for automatic deletion?', 'Not automatically. The action changes the cost of false positives. State a new precision/cost requirement and validate the corresponding policy.'],
      ],
      code: `valid_prob = model.predict_proba(X_valid)[:, 1]
precision, recall, thresholds = metrics.precision_recall_curve(y_valid, valid_prob)
threshold_table = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1],
})
recall_target = 0.80
feasible = threshold_table.loc[threshold_table["recall"].ge(recall_target)]
if feasible.empty:
    raise ValueError("No feasible validation threshold; discuss the requirement.")
threshold = float(feasible["threshold"].max())
validation_decision = (valid_prob >= threshold).astype(int)
display(threshold_table.loc[threshold_table["threshold"].eq(threshold)])
print("validation review fraction:", validation_decision.mean())
print("validation recall:", metrics.recall_score(y_valid, validation_decision))`,
      check: `assert len(threshold_table) == len(thresholds)
assert len(precision) == len(recall) == len(thresholds) + 1
assert metrics.recall_score(y_valid, validation_decision) >= recall_target
assert threshold == feasible["threshold"].max()
assert ((validation_decision == 0) | (validation_decision == 1)).all()`,
    },
    {
      title: 'Evaluate the frozen policy and hand it off', time: '31–35',
      task: `Freeze the fitted model and selected threshold. Create **test_prob**, **test_decision** and **scores** comparing the prior baseline and model on ROC-AUC, AP and log loss. Create **policy_scores** comparing threshold 0.5 and the selected threshold on precision, recall, F1 and review fraction.

Does the validation recall target also hold on test? Report the result honestly without adjusting the policy. Summarize the trade-off and what you would monitor after deployment. Confusion-matrix display and new-submission inference are optional handoff follow-ups below.`,
      say: 'The test set evaluates one frozen policy. I will report both abuse coverage and the human-review workload, including any missed target.',
      questions: [
        ['Why do ROC-AUC and AP receive probabilities rather than 0/1 decisions?', 'They evaluate ranking over possible thresholds. A single binary decision discards most score ordering information.'],
        ['Why might test recall fall below 80%?', 'Finite samples, a different population or drift can move the operating point. The validation requirement is not a guarantee; do not retune on the final test set.'],
        ['What if there are not enough reviewers for the resulting queue?', 'The recall and capacity requirements may conflict. Quantify the trade-off on validation, discuss prioritization or added capacity, and do not silently claim both are satisfied.'],
      ],
      code: `test_prob = model.predict_proba(X_test)[:, 1]
test_decision = (test_prob >= threshold).astype(int)
def ranking_scores(probability):
    return {
        "ROC_AUC": metrics.roc_auc_score(y_test, probability),
        "AP": metrics.average_precision_score(y_test, probability),
        "log_loss": metrics.log_loss(y_test, probability, labels=[0, 1]),
    }
scores = pd.DataFrame({
    "prior": ranking_scores(baseline.predict_proba(X_test)[:, 1]),
    "logistic": ranking_scores(test_prob),
}).T
def policy_metrics(decision):
    return {
        "precision": metrics.precision_score(y_test, decision, zero_division=0),
        "recall": metrics.recall_score(y_test, decision, zero_division=0),
        "F1": metrics.f1_score(y_test, decision, zero_division=0),
        "review_fraction": decision.mean(),
    }
policy_scores = pd.DataFrame({
    "threshold_0.5": policy_metrics((test_prob >= 0.5).astype(int)),
    "selected_threshold": policy_metrics(test_decision),
}).T
display(scores, policy_scores)`,
      check: `assert test_prob.shape == test_decision.shape == (len(y_test),)
assert np.isfinite(test_prob).all()
np.testing.assert_allclose(scores.loc["prior", "AP"], y_test.mean())
np.testing.assert_allclose(scores.loc["prior", "ROC_AUC"], 0.5)
np.testing.assert_allclose(policy_scores.loc["selected_threshold", "review_fraction"], test_decision.mean())`,
    },
  ],
  extension: {
    prompt: `**Handoff follow-up:** Show a confusion matrix. Score one new_submission with a missing numeric field and an unseen content_type, creating new_probability and send_to_review with the frozen model and threshold. Do not fit again.

**Capacity follow-up:** A reviewer can inspect only 10% of submissions. Is that compatible with your recall goal? Use validation to inspect both constraints before touching test. Discuss why a fixed score threshold does not guarantee a fixed review count after drift.

**New-data follow-up:** The abuse fraction changes next month. What would you monitor, and why might precision move even if the model's conditional detection rates did not change?

**API notes, if needed:** precision_recall_curve produces arrays across score thresholds; unlike ROC-AUC/AP, precision_score and recall_score expect the chosen class decisions.`,
    code: `display(pd.DataFrame(metrics.confusion_matrix(y_test, test_decision, labels=[0, 1]),
    index=["actual: not abuse", "actual: abuse"],
    columns=["not queued", "queued"]))
new_submission = pd.DataFrame([{
    "text_length": np.nan, "link_count": 1, "uppercase_ratio": 0.6,
    "repeated_token_ratio": 0.1, "account_age_days": 30,
    "content_type": "NEW_TYPE",
}])
new_probability = model.predict_proba(new_submission)[:, 1]
send_to_review = new_probability >= threshold
print("new probability / review decision:", new_probability[0], send_to_review[0])
assert new_probability.shape == send_to_review.shape == (1,)

capacity = 0.10
capacity_table = threshold_table.copy()
capacity_table["review_fraction"] = [
    (valid_prob >= value).mean() for value in capacity_table["threshold"]
]
feasible_both = capacity_table.loc[
    capacity_table["recall"].ge(recall_target)
    & capacity_table["review_fraction"].le(capacity)
]
print("thresholds satisfying both validation constraints:", len(feasible_both))
if feasible_both.empty:
    print("No available threshold satisfies both; discuss the trade-off.")
else:
    display(feasible_both.tail())`,
  },
  sourceNotes: `The collected report topics motivating this original mock are harmful-content detection ([1125288](https://www.1point3acres.com/bbs/thread-1125288-1-1.html)), classification loss ([1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html)), and evaluation ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html)). The source index is [08_company_interviews.md](08_company_interviews.md). Those reports do **not** establish this exact notebook task, the 80% requirement or these CSVs as actual interview material.

Implementation reference: [sklearn precision_recall_curve](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.precision_recall_curve.html).`,
};
