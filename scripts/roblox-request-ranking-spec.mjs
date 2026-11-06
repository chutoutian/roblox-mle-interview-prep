export default {
  slug: 'request_ranking',
  title: 'Choose Two Experiences for Each Request',
  dataDir: 'data/codesignal_roblox_recommendation',
  prompt: `A game platform has logged recommendation requests. Each request contains several displayed experience cards, and each card has a label indicating whether the user joined that experience within ten minutes.

**Problem 1 — Data analysis (13 minutes).** Inspect the two exports, resolve duplicate records using the policy below, join experience metadata, and summarize join rates with their sample sizes.

**Problem 2 — Modeling and ranking (22 minutes).** Train a simple join-probability model and evaluate it on unseen users. Then choose up to two cards per held-out request and compare this diagnostic selection with keeping the earliest historical positions. Explain what this offline comparison can and cannot establish.

This is an original practice exercise, not a claim that Roblox asks this exact coding question. Work in ordinary pandas and scikit-learn; a neural network and a complete recommendation service are not required.`,
  contract: `The supplied files are **impressions.csv** and **experiences.csv**. One impression is one displayed experience card. One request contains several cards for one user; an experience can appear in many requests. A user must not appear in both training and test sets.

| File / field | Meaning |
| --- | --- |
| impressions.impression_id | Event identifier; exact re-exports and conflicting records are deliberately included |
| impressions.user_id / request_id / experience_id | User, recommendation batch and experience identifiers; not numeric predictors |
| impressions.prior_genre_minutes | Historical play time in this genre, available before the impression; negative means unknown |
| impressions.friends_playing | Friends playing at impression time |
| impressions.position | Historical display position, starting at 1 |
| impressions.device | Device category; may be missing |
| impressions.joined | Target: 1 if the user joined within ten minutes, otherwise 0; all windows have finished |
| impressions.minutes_after_impression | Future behavior; unavailable when scoring and excluded from features |
| experiences.experience_id | Unique metadata key |
| experiences.genre / experience_age_days | Category and age available at impression time |

**Conflict policy for this exercise:** remove exact duplicate rows first. If an impression_id still occurs more than once, quarantine every row for that ID because there is no timestamp or authoritative version. Report the number of rows and event IDs affected. Do not arbitrarily keep the first or last conflicting version. Do not edit the source CSVs.

After cleaning, requests can contain different numbers of cards. Never assume every request has four cards or that a particular row count must survive. Check that each request belongs to one user and contains no repeated experience.

For the ranking diagnostic, score all candidate cards as if their **position feature were 1**. This removes differing historical position values from the scoring input, but does not identify how users would respond to a different display order. Logged labels were collected under the old policy.`,
  highlights: [
    'Review: missing values, many-to-one joins, train-only pipelines, user-grouped splits and binary metrics.',
    'New situation: conflicting event IDs require quarantine, not just drop_duplicates().',
    'New operations: sort_values, groupby(...).head(2), and reindex for request-level evaluation.',
    'New reasoning: global classification metrics and within-request ranking metrics answer different questions.',
  ],
  steps: [
    {
      title: '1. Understand the exports', time: '0–4 min',
      task: `What does one row represent, and what is the prediction target? Load **impressions** and **experiences**. Inspect their shapes, column types, missing values, unique users and requests, exact duplicates, and negative historical play time. Which columns would be unavailable when a card is shown?`,
      say: 'I will identify the event grain and prediction time before deciding what can be a feature.',
      questions: [
        ['Why is user_id.count() not the number of users?', 'It counts nonmissing rows, so the same user is counted once per impression. nunique() counts distinct user IDs.'],
        ['Does a repeated impression_id always mean two identical rows?', 'No. An export can contain conflicting versions of the same event. Exact duplicates and duplicate event keys need separate checks.'],
        ['Can minutes_after_impression help predict joined?', 'It is a future outcome. Using it would leak information unavailable at scoring time.'],
      ],
      code: `impressions = pd.read_csv(DATA_DIR / "impressions.csv")
experiences = pd.read_csv(DATA_DIR / "experiences.csv")
impressions.info()
display(impressions.head(), experiences.head(), impressions.isna().sum())
print("users:", impressions["user_id"].nunique())
print("requests:", impressions["request_id"].nunique())
print("exact duplicate rows:", impressions.duplicated().sum())
print("negative history:", impressions["prior_genre_minutes"].lt(0).sum())
display(impressions["joined"].value_counts(normalize=True))`,
      check: `assert experiences["experience_id"].is_unique
assert impressions["joined"].notna().all()
assert set(impressions["joined"]) == {0, 1}
assert impressions[["impression_id", "user_id", "request_id", "experience_id"]].notna().all().all()`,
    },
    {
      title: '2. Clean events without inventing a version', time: '4–9 min',
      task: `How would you distinguish repeated exports from conflicting events? Apply the contract's conflict policy to create **clean_impressions** and preserve **conflicts** for investigation. Report both kinds of removal, replace invalid historical durations with missing values, and explain why the surviving number of cards can differ between requests.`,
      say: 'I will remove identical copies, then quarantine all conflicting versions rather than guess which one is correct.',
      questions: [
        ['Why use duplicated(keep=False) after removing exact duplicates?', 'It marks every remaining row belonging to a repeated event ID, including the first occurrence. The default would flag only later occurrences.'],
        ['Why not keep the final CSV row for an event?', 'File order does not establish which record is authoritative or which values existed at impression time. We would need timestamps and a source contract.'],
        ['Does quarantine have a cost?', 'Yes. It reduces coverage and can introduce selection bias. Report the affected IDs and investigate the cause; this is an explicit temporary policy, not a universal fix.'],
      ],
      code: `clean_impressions = impressions.drop_duplicates().copy()
exact_copies_removed = len(impressions) - len(clean_impressions)
conflict_mask = clean_impressions["impression_id"].duplicated(keep=False)
conflicts = clean_impressions.loc[conflict_mask].copy()
clean_impressions = clean_impressions.loc[~conflict_mask].copy()
print("exact copies removed:", exact_copies_removed)
print("quarantined rows:", len(conflicts))
print("quarantined event IDs:", conflicts["impression_id"].nunique())
display(conflicts)
clean_impressions["prior_genre_minutes"] = clean_impressions["prior_genre_minutes"].mask(
    clean_impressions["prior_genre_minutes"].lt(0)
)
display(clean_impressions.groupby("request_id").size().value_counts().sort_index())`,
      check: `assert clean_impressions["impression_id"].is_unique
assert not clean_impressions["impression_id"].isin(conflicts["impression_id"]).any()
assert len(impressions) == exact_copies_removed + len(conflicts) + len(clean_impressions)
assert clean_impressions["prior_genre_minutes"].dropna().ge(0).all()`,
    },
    {
      title: '3. Join metadata and examine join rates', time: '9–13 min',
      task: `How would you attach one metadata record to each impression without multiplying events? Create **frame**, check the resulting grain, and report impression counts and join rates by genre and historical position. What can these summaries tell you, and what would be unsafe to conclude?`,
      say: 'I will keep the impression grain and place sample sizes next to rates so that small groups are not mistaken for strong evidence.',
      questions: [
        ['Why validate="many_to_one"?', 'Many impression rows may reference one experience, but the metadata table must have at most one row per experience_id. Duplicate metadata keys could multiply events.'],
        ['Does a high position-1 join rate prove the first card is intrinsically best?', 'No. Position affects exposure, and the previous policy chose which items to place there. User and item differences also confound the comparison.'],
        ['Why verify request_id maps to one user?', 'The request is the ranking group. If a request accidentally contains multiple users, its candidate set and the grouped evaluation would be ill-defined.'],
      ],
      code: `frame = clean_impressions.merge(
    experiences, on="experience_id", how="left", validate="many_to_one"
)
assert clean_impressions["experience_id"].isin(experiences["experience_id"]).all()
assert frame.groupby("request_id")["user_id"].nunique().eq(1).all()
assert not frame.duplicated(["request_id", "experience_id"]).any()
display(frame.groupby("genre", dropna=False).agg(
    impressions=("impression_id", "size"), join_rate=("joined", "mean")
))
display(frame.groupby("position").agg(
    impressions=("impression_id", "size"), join_rate=("joined", "mean")
))`,
      check: `assert len(frame) == len(clean_impressions)
assert frame["impression_id"].is_unique
assert frame["experience_age_days"].notna().all()
assert frame["genre"].notna().all()`,
    },
    {
      title: '4. Choose the holdout unit', time: '13–17 min',
      task: `We want to estimate performance on users the model has never seen. Which features would you select, what split would support that goal, and how would you verify it? Create **numeric_features**, **categorical_features**, **X**, **y**, **train_idx**, **test_idx**, and the four train/test objects. Reserve about 20% of users for testing with a reproducible seed.`,
      say: 'I will split users, not individual impressions, and use positional split indices consistently for features and labels.',
      questions: [
        ['Why not split random rows?', 'The same user could occur in both sets. A row-level split would estimate a different, usually easier, evaluation setting than unseen-user generalization.'],
        ['Why use iloc with the indices returned by the splitter?', 'The splitter returns integer row positions, not pandas index labels. Filtering or deduplication can leave index labels different from positions.'],
        ['Does test_size=0.2 guarantee exactly 20% of rows?', 'No. In GroupShuffleSplit it refers to groups; row counts depend on how many impressions those selected users have.'],
      ],
      code: `numeric_features = ["prior_genre_minutes", "friends_playing", "position", "experience_age_days"]
categorical_features = ["device", "genre"]
X = frame[numeric_features + categorical_features]
y = frame["joined"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=frame["user_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_users = set(frame.iloc[train_idx]["user_id"])
test_users = set(frame.iloc[test_idx]["user_id"])
print("train/test users:", len(train_users), len(test_users))`,
      check: `assert train_users.isdisjoint(test_users)
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert set(frame.iloc[train_idx]["request_id"]).isdisjoint(frame.iloc[test_idx]["request_id"])
assert not {"impression_id", "user_id", "request_id", "experience_id", "joined", "minutes_after_impression"} & set(X.columns)
assert y_train.nunique() == 2 and y_test.nunique() == 2`,
    },
    {
      title: '5. Fit a reproducible pointwise model', time: '17–24 min',
      task: `How would you handle missing numeric values, missing categories and categories not seen during training? Build **model**, using a train-only preprocessing pipeline and a regularized LogisticRegression classifier. Fit **baseline**, a constant class-prior predictor. Explain what each model learns and why no preprocessing statistic should come from the test users.`,
      say: 'I will start with a simple probability model and a constant reference, keeping imputation and encoding inside the fitted pipeline.',
      questions: [
        ['Why median imputation rather than dropping every incomplete impression?', 'A missing historical value does not make the labeled impression useless. Median is a simple robust design choice learned from the training set, not proof that the unknown value equals that median.'],
        ['What does handle_unknown="ignore" do?', 'An unseen category produces zeros for that feature\'s learned one-hot columns instead of raising an error. It does not learn a new category during prediction.'],
        ['What does the prior baseline predict?', 'Every row receives the training class proportions. It checks whether the model learns useful discrimination beyond the base join rate.'],
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
learned = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
np.testing.assert_allclose(learned, X_train[numeric_features].median().to_numpy())
assert model.named_steps["classifier"].coef_.shape[0] == 1`,
    },
    {
      title: '6. Separate prediction quality from ranking input', time: '24–28 min',
      task: `Does the model improve on a constant predictor on held-out impressions? Produce **test_prob** and **scores** using ROC-AUC, average precision and log loss. Then create **ranking_scores** for the same rows with every position feature set to 1. Why should the original test probabilities and these hypothetical ranking scores be kept separate?`,
      say: 'I will evaluate observed-impression predictions first, then make the position convention for a hypothetical ranking explicit.',
      questions: [
        ['Why pass probabilities rather than thresholded 0/1 predictions to AUC and AP?', 'These metrics evaluate score ordering across thresholds. Thresholding discards that ordering and gives only one binary decision.'],
        ['Why set position=1 for all ranking inputs?', 'Otherwise the historical positions directly influence the new ordering. A shared value removes this input difference, but it does not correct historical selection bias or estimate causal outcomes.'],
        ['Does better global AUC guarantee better top-two choices for each request?', 'No. Global AUC compares positives and negatives across the whole test set. Per-request ranking concerns only the candidate cards competing within each request.'],
      ],
      code: `test_prob = model.predict_proba(X_test)[:, 1]
baseline_prob = baseline.predict_proba(X_test)[:, 1]
def evaluate_probability(gt, probability):
    return {
        "roc_auc": metrics.roc_auc_score(gt, probability),
        "average_precision": metrics.average_precision_score(gt, probability),
        "log_loss": metrics.log_loss(gt, probability, labels=[0, 1]),
    }
scores = pd.DataFrame({
    "class_prior": evaluate_probability(y_test, baseline_prob),
    "logistic": evaluate_probability(y_test, test_prob),
}).T
display(scores)
ranking_input = X_test.copy()
ranking_input["position"] = 1
ranking_scores = model.predict_proba(ranking_input)[:, 1]`,
      check: `assert len(test_prob) == len(X_test) == len(ranking_scores)
assert np.isfinite(test_prob).all() and np.isfinite(ranking_scores).all()
assert ((ranking_scores >= 0) & (ranking_scores <= 1)).all()
assert ranking_input["position"].eq(1).all()
pd.testing.assert_frame_equal(X_test, X.iloc[test_idx])`,
    },
    {
      title: '7. Select two cards and explain the handoff', time: '28–35 min',
      task: `For each held-out request, which up to two experiences would you select? Create **heldout**, **ranked**, **top_two**, and **request_scores**. Sort model scores from high to low and break ties by experience_id. Compare with selecting the first two historical positions, also breaking ties deterministically.

Report mean precision over the actual selected count (one card means a denominator of one), mean recall among requests with at least one observed join, and hit rate both for all requests and for positive requests only. Report how many zero-positive requests are excluded from recall. Keep requests with fewer than two surviving cards.

Finally, show a small request with its original positions, observed labels, scores and selected cards. If asked to ship this ranking, what evidence would you need beyond this notebook?`,
      say: 'I will evaluate within each request, name each denominator, and treat the result as a logged-data diagnostic rather than evidence that reordering causes more joins.',
      questions: [
        ['Why divide precision by the actual number selected?', 'A request may have fewer than two surviving candidates. Dividing by two would penalize missing candidates as if they were selected negatives.'],
        ['How should recall handle a request with no observed positive?', 'Its recall denominator is zero, so exclude it from mean positive-request recall and report the excluded count. It remains in all-request hit rate, where it contributes zero.'],
        ['Can this comparison prove the proposed ranking increases joins?', 'No. Outcomes were observed at historical positions under the previous policy. Reordering changes exposure, and unseen outcomes are not available. An appropriately designed randomized online evaluation or valid counterfactual data would be needed.'],
      ],
      code: `heldout = frame.iloc[test_idx][[
    "request_id", "experience_id", "position", "joined"
]].copy()
heldout["score"] = ranking_scores
ranked = heldout.sort_values(
    ["request_id", "score", "experience_id"], ascending=[True, False, True]
)
top_two = ranked.groupby("request_id", sort=False).head(2)
historical_top_two = heldout.sort_values(
    ["request_id", "position", "experience_id"]
).groupby("request_id", sort=False).head(2)

def evaluate_requests(selected, all_cards):
    positives = all_cards.groupby("request_id")["joined"].sum()
    hits = selected.groupby("request_id")["joined"].sum().reindex(positives.index, fill_value=0)
    shown = selected.groupby("request_id").size().reindex(positives.index, fill_value=0)
    eligible = positives.gt(0)
    return {
        "precision_at_2_actual_count": (hits / shown).mean(),
        "recall_at_2_positive_requests": (hits[eligible] / positives[eligible]).mean(),
        "hit_rate_at_2_all_requests": hits.gt(0).mean(),
        "hit_rate_at_2_positive_requests": hits[eligible].gt(0).mean(),
        "request_count": len(positives),
        "excluded_zero_positive_requests": (~eligible).sum(),
    }

request_scores = pd.DataFrame({
    "historical_first_two": evaluate_requests(historical_top_two, heldout),
    "model_top_two": evaluate_requests(top_two, heldout),
}).T
display(request_scores)
example_request = heldout["request_id"].iloc[0]
display(ranked.loc[ranked["request_id"].eq(example_request)])
display(top_two.loc[top_two["request_id"].eq(example_request)])`,
      check: `expected_sizes = heldout.groupby("request_id").size().clip(upper=2)
actual_sizes = top_two.groupby("request_id").size().reindex(expected_sizes.index, fill_value=0)
pd.testing.assert_series_equal(actual_sizes, expected_sizes)
assert not top_two.duplicated(["request_id", "experience_id"]).any()
assert request_scores["request_count"].eq(heldout["request_id"].nunique()).all()
assert set(top_two.index).issubset(heldout.index)`,
    },
  ],
  extension: {
    prompt: `A new request arrives with three candidate cards. How would you reuse the fitted preprocessing and model without fitting again? Rank the supplied cards, including an unknown device category and one missing historical value. Do not claim that these scores are validated probabilities for an unseen category or a new display policy.`,
    code: `new_cards = pd.DataFrame({
    "experience_id": [9001, 9002, 9003],
    "prior_genre_minutes": [40.0, np.nan, 10.0],
    "friends_playing": [1, 0, 2],
    "position": [1, 1, 1],
    "experience_age_days": [30, 200, 1000],
    "device": ["NEW_DEVICE"] * 3,
    "genre": ["adventure", "puzzle", "simulation"],
})
new_cards["score"] = model.predict_proba(new_cards[X.columns])[:, 1]
display(new_cards.sort_values(["score", "experience_id"], ascending=[False, True]).head(2))`,
  },
  sourceNotes: `This original mock combines themes from the collected Roblox interview notes: [report 1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html) discusses training examples, features and evaluation in an ML-design context; [report 1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html) includes interaction-data processing. These reports do **not** establish that this exact notebook, conflict policy or request-ranking coding exercise was asked. The exercise follows the user's supplied Data Analysis & Modeling interview format.`,
};
