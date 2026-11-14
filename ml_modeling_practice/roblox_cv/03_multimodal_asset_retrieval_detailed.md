# Multimodal Asset Retrieval in Detail: Interview Answers, Equations, and Examples

This is the detailed companion to the [reference answer](03_multimodal_asset_retrieval_solution.md) and [question document](03_multimodal_asset_retrieval_questions.md). It covers the seven stages, Q1–Q21, and three optional changes, with spoken answers, concrete examples, tensor shapes, and worked mathematics.

**Exercise boundaries:** The two-million-asset catalog, six rendered views, top twenty results, and 150 ms server-side p95 target are invented interview assumptions. This asset-search task is an original adaptation of broader retrieval topics, not a reported Roblox prompt or a production architecture claim.

## 0. The complete retrieval pipeline

> “I would define relevance before choosing a model: exact identity, visual similarity, and a useful substitute are different targets. I would render assets into images, establish a compatible pretrained text–image embedding baseline, and retrieve candidates from precomputed catalog vectors. Reviewed pairs would guide adaptation, and a shortlist reranker would be added only if it improves important errors within the latency budget. Evaluation would distinguish representation quality, approximate-search errors, final ranking, and user success.”

```text
TRAINING
Query + asset version + relevance judgment
  → query encoder ───────────────────────────┐
Asset → rendered views → asset encoder ─────┴→ pair scores
  → contrastive / relevance objective → parameter updates
  → development retrieval metrics → locked test

CATALOG PREPARATION
Eligible asset version
  → validated renderings
  → compatible image encoder + view pooling + normalization
  → asset vector + metadata → search index

ONLINE QUERY
Text or reference image
  → preprocessing + query encoder → normalized query vector
  → approximate candidate search
  → eligibility / version / duplicate handling
  → optional query–candidate reranking
  → return twenty eligible results

FEEDBACK
Independent relevance reviews + exposure-aware behavior + failures
  → revised judgments and subsequent model versions
```

The expensive catalog encoding is reusable across queries. A dual encoder creates vectors that can be scored cheaply; a richer pairwise model can inspect a small candidate set afterward.

## 1. Problem formulation · 0–4 minutes

### Spoken interview answer

> “One prediction is the relevance of a query–asset pair, and the product output is an ordered list of eligible asset versions. I would clarify whether users want the identical asset or a substitute with specific appearance and functional attributes. I would define graded relevance examples, resolve multi-object image intent explicitly, and separate relevance from popularity and eligibility.”

### 1.1 Define the score and the output

Let $q$ be a text or reference-image query and $a$ an asset version. The system learns a scoring function:

$$
s(q,a)\in\mathbb R,
\qquad\operatorname{Results}(q)=\operatorname{TopK}_{a\in\mathcal E(q)}s(q,a).
$$

$\mathcal E(q)$ is the eligible catalog for that request. A cosine similarity is a ranking score, not automatically the probability of relevance. If the product needs abstention or a match-confidence display, validate that separately.

For “red wooden chair,” one possible reviewed relevance scale is:

| Grade | Example interpretation |
| --- | --- |
| 3 | A usable chair matching the requested red color and wood material |
| 2 | A useful substitute with a minor mismatch accepted under the guide |
| 1 | Related object with a substantial attribute mismatch |
| 0 | Unrelated or unsuitable asset |

These boundaries must be agreed with reviewers and product. If material is mandatory, a metal chair should not be called relevant merely because it looks red. For binary metrics, state which grades count as relevant; retain the original grades for ranking evaluation.

### Q1. Exact duplicate or semantic substitute?

An exact-match task asks for the same file, mesh, or identifiable asset. File hashes, geometry checks, or instance retrieval may help. A semantic query permits many different assets if they satisfy the user's intent.

For example, two chairs by different creators can both match “small red wooden chair.” A duplicate detector that returns only the original screenshot cannot solve that broader task. Conversely, semantic similarity alone may return a substitute when the user explicitly needs the original asset.

Clarify the target, label both cases appropriately, and report their quality separately. Also decide whether the top twenty should show twenty near-identical variants or diverse valid alternatives.

### Q2. Multiple attributes or several objects in a reference image?

“Red wooden chair” requires combining category, color, and material. Separate matches for each word are insufficient: a red metal chair and a blue wooden chair each satisfy only part of the request.

If an uploaded room screenshot contains a chair, table, and lamp, the desired object is ambiguous. A crop or object-selection interaction can make the query explicit. If the product instead supports scene-wide similarity, use labels and evaluation for that different target. Do not silently assume that the largest object is always the user's intent.

### Q3. What requirements change the representation?

Clarify supported languages, asset types, text and image query proportions, exact versus substitute matching, geometry versus appearance, size/material requirements, eligibility filters, freshness, and the latency boundary.

For instance, a standardized rendering can show a red surface but may not reveal whether a mesh is compatible with an attachment point. Normalizing every asset to the same apparent size can also remove evidence of physical dimensions. If those attributes matter, use available geometry or metadata, or explicitly state the limitation of a visual baseline.

## 2. Data and labels · 4–10 minutes

### Spoken interview answer

> “I would collect query–asset relevance judgments with modality, event time, and asset versions. I would combine independently reviewed pairs with reliable content pairs and cautiously interpreted behavior, while preserving multiple valid matches. Unclicked and unexposed assets would not automatically be negatives. Evaluation would include representative queries, new creators and styles, and later-time content, with copied assets and changing metadata controlled.”

### 2.1 What is one training example?

```text
query_id, query_text or reference_image
query_modality, language, event_timestamp
asset_id, asset_version, asset_family_id, creator_id
render_version, metadata_snapshot
relevance_grade, judgment_source, judgment_status
exposure_position and qualified_action, when available
```

Human judgments should evaluate the asset, not only a misleading thumbnail. Store the content actually reviewed. Useful supervision includes a manually written query paired with a relevant asset, a reliable description paired with its renderings, and verified behavioral evidence such as successful asset use.

Descriptions are weak supervision if vague or incorrect. The fact that a description belongs to an asset does not guarantee it correctly describes every visual or geometric attribute. Keep the source distinguishable and evaluate on independent reviewed queries.

Include category, color/material combinations, unfamiliar art styles, query clutter, occluded objects, and difficult alternatives. Collect both text and image-query examples; training text retrieval alone does not establish image-query performance.

### Q4. Does “not clicked” mean irrelevant?

No. An asset might never have been shown, appeared far down the page, or been relevant but ignored after the user found an earlier result.

```text
Asset A: shown at rank 1, clicked
Asset B: shown at rank 20, not clicked
Asset C: not shown
```

These observations do not prove that B and C are irrelevant. Logs reflect exposure, position, the previous ranking policy, preview attractiveness, and user intent.

Record actual exposure and qualified later outcomes. Independently judged examples remain important. Randomized exposure or propensity methods can help when their assumptions and support hold; simply adding position as a feature or setting it to a constant does not remove exposure bias. An inverse-propensity weight requires a meaningful, nonzero observation probability and does not turn every click into reliable relevance.

### Q5. What if several assets match one query?

Keep all known valid alternatives positive. A different asset ID does not imply a negative.

```text
Query: red wooden chair
Asset A: red wooden chair, creator 1 → positive
Asset B: red wooden chair, creator 2 → positive
Asset C: red metal chair             → reviewed negative if material is mandatory
```

If A and B are present in the same batch, a one-positive diagonal loss may incorrectly push B away when training the query paired with A. Deduplicate equivalent pairs, mask known alternative positives from the negative set, or use a multi-positive objective. Unknown relevance is still unknown; some false negatives remain possible, so audit the negative-selection process.

### Q6. How do you prevent leakage?

Define the claim first. Searching an existing catalog legitimately uses its indexed images at test time; this is not equivalent to training on test relevance judgments. An unseen-asset generalization test is a stricter, different evaluation.

For that stricter test, group copied and near-duplicate asset families and keep them out of training relevance pairs. Add creator-held-out and future-time slices where required. Group paraphrases or templates of held-out evaluation queries when repeated wording would otherwise inflate results.

Snapshot metadata, renderings, eligibility, and versions at the relevant decision time. A historical query must not receive a later improved description during training. Keep future asset versions out of earlier training snapshots. Do not move future images backward in time merely to keep duplicate groups together.

Report ordinary known-catalog search, unseen assets, and new creators separately. Randomly splitting pair rows alone can leave nearly identical queries, copied meshes, and later versions on both sides.

## 3. Baseline · 10–15 minutes

### Spoken interview answer

> “I would start with a compatible pretrained text–image model, render each mesh into several views, pool the projected view embeddings, and normalize the asset vector. Text and reference-image queries would use the matching pretrained encoders. On a manageable catalog I would first use exact vector search to measure representation quality without ANN approximation, and compare a lexical metadata baseline for text queries.”

### 3.1 Why a compatible pretrained pair matters

A CLIP-style model learns paired text and image representations; this is the relevant starting point from the [CLIP paper](https://arxiv.org/abs/2103.00020). Keep the checkpoint's trained projections and preprocessing when establishing the baseline. Two unrelated encoders with equal output dimensions do not automatically support meaningful cross-modal similarity.

The initial asset representation can be the normalized average of its already projected view embeddings. Text queries use the text branch; reference images use the image branch. This is an engineering baseline to evaluate, not a guarantee that visual similarity captures all asset relevance.

For a toy normalized query $q=[1,0]$ and normalized assets $a_1=[0.8,0.6]$, $a_2=[0,1]$:

$$
s(q,a_1)=q^\top a_1=0.8,\qquad s(q,a_2)=0.
$$

Exact search computes scores for every catalog vector and selects the highest. Approximate nearest-neighbor search, or ANN, trades some exact-neighbor recovery for speed and memory characteristics. Establishing an exact baseline helps separate model errors from search-index errors.

### Q7. Can the image encoder consume a raw mesh?

No. A normal image encoder accepts pixels, not a list of vertices, faces, and materials. Render the mesh into images using a defined camera and lighting setup, then encode the images.

```text
Mesh + materials
  → render front / side / back / other useful views
  → RGB images
  → image encoder
```

A geometry-specific encoder is another design, requiring suitable inputs and supervision. Rendering is a practical way to reuse visual pretraining, but it cannot guarantee physical dimensions, collision behavior, or invisible internal geometry.

### Q8. How do you detect rendering shortcuts?

Standardize useful scale, framing, backgrounds, and lighting, then deliberately perturb those choices in evaluation. Keep views that reveal important sides rather than repeatedly rendering the same front face.

For example, if all training chairs have a gray background and tables have a blue background, the model may exploit background color. Test the same object under changed backgrounds and test different objects under the same background. Inspect back-side details, transparency, thin structures, materials, and camera orientation.

A high-quality rendering is not necessarily an informative one. If the attribute requested by the user is hidden in every view, retraining the final projection cannot recover it.

### Q9. When is domain adaptation justified?

Audit failures before changing weights. Missing assets, invalid renderings, stale eligibility, or a broken image crop require pipeline fixes. Relevant assets ranked poorly even under exact search, with correct content and labels, suggest a representation or scoring problem.

Unfamiliar stylized assets, small details, and repeated color/material confusion may justify adaptation. Compare held-out relevance, important slices, and compute before and after adaptation. Falling training loss alone does not establish improved search.

## 4. Model design · 15–20 minutes

### Spoken interview answer

> “I would use dual encoders so catalog vectors can be computed ahead of requests. Each asset's rendered views become image embeddings, which are pooled and normalized. A text query or reference image becomes a compatible normalized query vector. Dot products produce candidate scores. If attribute composition remains weak, I would compare a cross-encoder on a small retrieved shortlist rather than run rich interactions against the entire catalog.”

### 4.1 Trace the tensors

Use illustrative batch size eight, six views, visual hidden width 768, and retrieval width 512:

```text
Asset rendering batch                     [8, 6, 3, 224, 224]
Visual encoder + per-image pooling        [8, 6, 768]
Trained compatible projection             [8, 6, 512]
View pooling + L2 normalization           [8, 512]

Text query IDs                            [8, L]
Text encoder                              [8, L, d_text]
Checkpoint's text pooling + projection    [8, 512]
L2 normalization                         [8, 512]

All query–asset scores in training        [8, 8]
```

Here `d_text` and the visual width depend on the checkpoint; 768 and 512 are examples. A reference-image query uses the compatible visual path and projection, perhaps on an explicitly selected crop.

For unnormalized projected vectors $u_i$ and $v_j$:

$$
q_i=\frac{u_i}{\|u_i\|_2},\qquad
a_j=\frac{v_j}{\|v_j\|_2},\qquad
s_{ij}=\frac{q_i^\top a_j}{\tau}.
$$

Handle zero or invalid vectors explicitly. With row-stacked matrices $Q:[B,512]$ and $A:[B,512]$:

$$
S=QA^\top/\tau:[B,B].
$$

At serving time the candidate similarity can simply be $q^\top a$. A single positive temperature shared across all assets for that query rescales scores without changing their ordering; it matters for training probabilities and score-dependent decisions.

### Q10. How should multiple views be combined?

For valid-view mask $r_{jv}$ and projected view vectors $e_{jv}$, a baseline is:

$$
\bar e_j=\frac{\sum_vr_{jv}e_{jv}}{\sum_vr_{jv}},
\qquad a_j=\frac{\bar e_j}{\|\bar e_j\|_2}.
$$

One asset produces one vector, regardless of how many valid views it has. Missing views do not enter the mean. An all-missing or invalid asset needs rendering repair or an explicit fallback, not a normal-looking vector from padding.

Mean pooling is cheap but may dilute a distinctive back-side detail. Compare learned view attention or storing per-view vectors. Per-view search can retrieve any matching side, but uses more memory, returns multiple hits for the same asset, and needs asset-level score aggregation and deduplication.

Max-over-views scoring favors the most similar view, but more views provide more opportunities for accidental high scores. Reevaluate ranking and score thresholds under the actual view-count distribution. Query-conditioned view attention can be richer, but cannot generally be condensed into one fixed asset vector for all queries; it fits naturally in a shortlist reranker.

### Q11. Why is a fixed object classifier insufficient?

A classifier that outputs “chair” loses distinctions such as red versus blue, wood versus metal, small versus large, and stylistic similarity. Arbitrary natural-language requests cannot be reduced to a fixed category label without losing information.

A classifier may help with category filtering. Retrieval still needs a score for the specific query and asset. Training one output class per asset would also tie the head to a changing catalog, while content embeddings let a new asset be inserted without creating a new classifier output.

This is also where retrieval differs from genre classification. For genres, concatenating unrelated image and text features followed by a trained head can work. For retrieval, directly taking a dot product between two vectors requires trained compatibility; equal dimensions or L2 normalization alone do not establish semantic alignment.

### Q12. Where does a cross-encoder belong?

```text
Query → cheap vector search → 200 candidate assets
                                  ↓
                  Query + candidate visual/text tokens
                                  ↓
                     cross-encoder relevance score
                                  ↓
                           reordered top 20
```

The candidate count is illustrative. Unlike the dual encoder, a cross-encoder jointly processes the query and each candidate's evidence. Attention can connect “wooden” to candidate material cues or a query object's shape to particular visual patches.

The asset representation is now query-dependent, so the full pair score cannot generally be precomputed once per asset. Running it for two million assets per request is incompatible with the intended cheap candidate stage. Profile shortlist size, tokens, views, and p95; keep it only if end-to-end gains justify cost.

A reranker cannot repair missing candidates. First ensure candidate retrieval has sufficient relevance recall, then improve ordering within that set.

## 5. Loss and training · 20–26 minutes

### Spoken interview answer

> “I would begin with the pretrained compatible embeddings, then compare lightweight adaptation using reviewed query–asset pairs. A contrastive batch scores each query against several assets, increasing relevant-pair scores relative to competitors. I would prevent known alternative matches from becoming false negatives, train both query modalities, and use a multi-positive objective when appropriate. Normalization controls vector magnitude, and temperature controls softmax sharpness.”

### 5.1 What is in a contrastive batch?

In a simple one-positive example, each row is a query and each column is an asset:

```text
                   Asset A          Asset B          Asset C
Query 1             match            nonmatch         nonmatch
Query 2             nonmatch         match            nonmatch
Query 3             nonmatch         nonmatch         match
```

Assume for this toy batch that off-diagonal pairs are genuinely nonmatching. This is an assumption to check, not something guaranteed by different IDs.

Compute logits $s_{ij}=q_i^\top a_j/\tau$ and normalize within each query row:

$$
p_{ij}=\frac{e^{s_{ij}}}{\sum_{k=1}^{B}e^{s_{ik}}},
\qquad L_{Q\to A}=-\frac1B\sum_i\log p_{ii}.
$$

This is cross-entropy over candidate assets. It rewards the correct candidate relative to the competitors in the batch; its softmax probabilities are conditional on that candidate set, not calibrated relevance probabilities over the whole catalog.

For a symmetric objective, reverse the direction by applying the same operation over columns:

$$
L_{A\to Q}=-\frac1B\sum_j\log
\frac{e^{s_{jj}}}{\sum_ie^{s_{ij}}},
\qquad L=\frac12(L_{Q\to A}+L_{A\to Q}).
$$

Symmetric image–text matching is the relevant CLIP-style training idea. Our multi-view assets, image queries, negative selection, and deployment choices extend that idea for this exercise. A reverse loss is a candidate to validate, not mandatory for every relevance dataset; several queries can also validly describe one asset and need appropriate positive handling.

### 5.2 Work through a numerical loss

Suppose the already temperature-scaled score matrix is:

$$
S=\begin{bmatrix}3&1&0\\0&3&1\\1&0&3\end{bmatrix}.
$$

For the first row, the positive probability is:

$$
p_{11}=\frac{e^3}{e^3+e^1+e^0}
\approx\frac{20.086}{20.086+2.718+1}
\approx0.844,
$$

$$
\ell_1=-\log0.844\approx0.170.
$$

All rows and columns have the same positive and negative score pattern, so their losses are approximately 0.170 and the symmetric average is also approximately 0.170.

For a one-positive row:

$$
\frac{\partial\ell_i}{\partial s_{ij}}=p_{ij}-\mathbf1[j=i].
$$

The positive gradient is about $0.844-1=-0.156$, pushing its logit upward during gradient descent. Competitors have positive gradients and are pushed downward. Shared encoder parameters change all these embeddings through backpropagation.

### Q13. How do you choose negatives without punishing valid alternatives?

Use a mixture of random catalog negatives, in-batch competitors, and reviewed hard negatives. A red metal chair is a useful hard negative only when the query and annotation guide require wood. A second red wooden chair is an alternative positive, not a hard negative merely because it has another ID.

Hard-negative mining finds high-scoring candidates under the current model, then checks whether they are truly unsuitable. Do not repeatedly train on unreviewed high-scoring alternatives as definite negatives. Deduplicate copied assets and track known relevance sets.

One multi-positive objective rewards probability assigned to the positive set $P_i$:

$$
\ell_i^{set}=-\log
\frac{\sum_{j\in P_i}e^{s_{ij}}}{\sum_{k\in C_i}e^{s_{ik}}}.
$$

$C_i$ is the chosen candidate set, including positives and selected competitors. If two known positives receive probabilities 0.40 and 0.35, the set loss is $-\log(0.75)\approx0.288$. A one-positive target using only the first would instead give $-\log(0.40)\approx0.916$ and wrongly oppose the other valid match.

The set objective can concentrate on one easy positive. If the goal is to support all known alternatives, compare averaging log-probability over positives:

$$
\ell_i^{all}=-\frac1{|P_i|}\sum_{j\in P_i}
\log\frac{e^{s_{ij}}}{\sum_{k\in C_i}e^{s_{ik}}}.
$$

These objectives encode different preferences; choose using reviewed retrieval quality. Neither reveals unknown positives automatically. Also examine popularity and exposure bias in logged competitors; a sampled-softmax correction depends on the chosen objective and sampling process, and is not a universal repair for click bias.

### Q14. Why normalization, and what does temperature do?

Without normalization, multiplying an embedding by ten can increase dot products without changing its direction. L2 normalization makes the dot product equal cosine similarity:

$$
\cos\theta=\frac{u^\top v}{\|u\|_2\|v\|_2}=q^\top a.
$$

It removes norm inflation as a way to improve the contrastive score. It does not align unrelated embedding spaces or guarantee semantic relevance.

Temperature $\tau>0$ sets the softmax scale. For cosine scores 0.8 and 0.6:

$$
P(\text{first}\mid\tau=1)
=\frac{e^{0.8}}{e^{0.8}+e^{0.6}}\approx0.550,
$$

$$
P(\text{first}\mid\tau=0.1)
=\frac{e^8}{e^8+e^6}\approx0.881.
$$

A lower temperature sharpens relative probabilities and introduces a $1/\tau$ factor in gradients with respect to unscaled similarities. It can amplify noisy or false-negative pairs; the probabilities themselves also change, so lower temperature does not uniformly improve learning. Tune or constrain it on development data.

### Q15. What should be trained with limited reviewed data?

Keep the unmodified pretrained baseline first. Compare learned view pooling or small projections, selective encoder unfreezing, and low-rank adaptation. If adding new projections, train them with the paired objective; inserting random layers and normalizing their outputs does not preserve the original aligned space automatically.

Use a lower learning rate for unfrozen pretrained layers than for a new head as a candidate schedule, and choose checkpoints by held-out retrieval. AdamW is one practical optimizer; optimizer choice does not fix wrong relevance labels. Keep text- and image-query quality, rare styles, and forgetting visible.

For a reranker trained on graded judgments, a separate relevance regression/classification or pairwise-ranking objective may be appropriate. For example, for a preferred asset $a^+$ over $a^-$:

$$
\ell_{pair}=\log\left(1+e^{-(r(q,a^+)-r(q,a^-))}\right).
$$

This encourages the preferred candidate's reranker score to be higher. It cannot repair an asset that the first-stage search never retrieves.

## 6. Evaluation · 26–31 minutes

### Spoken interview answer

> “I would evaluate judged queries against a versioned eligible catalog, reporting text and image queries separately. Candidate relevance Recall@K measures what the reranker can still recover, while MRR, NDCG, and AP describe the final ordering. I would measure ANN recovery against exact search separately from human relevance, and inspect new creators, attribute combinations, and styles. A randomized rollout would then measure successful asset selection and use, with latency and eligibility guardrails.”

### 6.1 Define the evaluation universe and metrics

Fix the query set, catalog snapshot, eligibility rules, and relevance judgments. The **evaluation universe** is the set of queries and eligible assets against which every model is compared. A catalog snapshot fixes which asset versions exist; eligibility rules determine which of them may be returned. Otherwise, a changed score could reflect a changed catalog rather than a better model.

**Model scores determine the order; human judgments evaluate that order.** The model might output similarities such as 0.82, 0.75, and 0.61. Sort by those scores, then look up the human relevance label of the asset at each position. These ranking metrics do not require mapping model similarity onto the human 0–3 scale. Any strictly increasing transformation of the scores preserves the ranking and therefore preserves these metrics, with the same tie handling.

**Shared notation and labels**

| Notation | Meaning in this section |
| --- | --- |
| $q$ | One evaluation query, either text or a reference image. Here it identifies the query, not its embedding vector. |
| $\mathcal Q$ | The set of evaluation queries included in an aggregate metric. |
| $Q$ | The number of queries in that set: $Q=\lvert\mathcal Q\rvert$. Lowercase $q$ is one query; uppercase $Q$ is a count. |
| $a$ | One candidate asset from the fixed eligible catalog. |
| $g(q,a)$ | The human relevance grade for query–asset pair $(q,a)$, for example 0: irrelevant, 1: weak partial match, 2: useful match, 3: excellent match. This is a label, not the model similarity. |
| $R_q$ | The set of assets judged relevant to query $q$ under a fixed binary relevance rule. The subscript $q$ means this set depends on the query. |
| $K$ | A positive integer cutoff, such as 20. The notation `@K` means only the first $K$ ranked results are considered. |
| $\operatorname{TopK}(q)$ | The set of the first $K$ assets in the model's ranking for query $q$. Recall and precision count membership in this set rather than the positions within it. |
| $r$ | A rank position, counted from 1: the first result has $r=1$. It is not a relevance score. |

For binary metrics, define which human grades count as relevant. For example, use grades 2 and 3 as positives when they mean a usable result; grade 1 remains a partial match that does not meet this threshold. This choice defines $R_q$ and must stay fixed across comparisons. NDCG instead retains the relevance grades. An unjudged asset is unknown, not automatically grade 0; specify the judgment and evaluation policy before scoring.

**Recall and precision: coverage versus result quality**

$$
\operatorname{Recall@K}(q)=
\frac{|\operatorname{TopK}(q)\cap R_q|}{|R_q|},
\qquad
\operatorname{Precision@K}(q)=
\frac{|\operatorname{TopK}(q)\cap R_q|}{K}.
$$

Read the set operations as follows:

- $\cap$ means **intersection**: the assets that belong to both sets.
- $\lvert S\rvert$ means the **number of distinct elements in set $S$**, not a vector norm or a numerical absolute value here.
- $\lvert\operatorname{TopK}(q)\cap R_q\rvert$ is the number of relevant assets among the first $K$ results. It is the numerator in both metrics.
- $\lvert R_q\rvert$ is the total number of known relevant assets for this query. Recall asks what fraction of them were retrieved.
- $K$ in the precision denominator is the number of result slots evaluated. Precision asks what fraction of those slots contain a relevant asset.
- The argument $(q)$ means the metric is for one query. To report mean recall or precision, calculate it per query and then average over the declared query set.

For example, if $R_q=\{A,C,D\}$ and the first two results are A and B, the intersection is $\{A\}$. Recall@2 is $1/3$ and Precision@2 is $1/2$.

The displayed precision formula uses $K$ result slots. Define how fewer than $K$ eligible results are handled, such as counting missing slots as nonrelevant, or explicitly reporting a different denominator. For queries with no known positives, recall is undefined; specify exclusion or a separate no-match evaluation rather than silently dividing by zero. Report the number of queries included in each aggregate.

**RR and MRR: how soon does the first useful result appear?**

RR means **Reciprocal Rank** for one query. MRR means **Mean Reciprocal Rank** across queries:

$$
\operatorname{RR}(q)=
\begin{cases}
1/r_q & \text{if a relevant result is found},\\
0 & \text{otherwise},
\end{cases}
\qquad
\operatorname{MRR}=\frac1Q\sum_{q\in\mathcal Q}\operatorname{RR}(q).
$$

- $r_q$ is the position of the **first** relevant result for query $q$. If the first hit is at rank 4, its RR is $1/4$; if there is no hit, use the zero case rather than an undefined rank.
- $\sum_{q\in\mathcal Q}$ means add one contribution for every query in the evaluation set. The symbol $\in$ means "belongs to."
- $1/Q$ converts the sum into an arithmetic mean: each included query has equal weight.
- MRR@K searches for that first hit only within the first $K$ positions; a query with no relevant result there contributes zero.

MRR does not reward retrieving many valid alternatives after the first hit. Rankings beginning with a relevant asset all have RR 1, regardless of how good their remaining results are.

**DCG, IDCG, and NDCG: put the most relevant assets earliest**

For one fixed query, use its human relevance grades and discount results at later positions:

$$
\operatorname{DCG@K}=\sum_{r=1}^{K}
\frac{2^{g_r}-1}{\log_2(r+1)},\qquad
\operatorname{NDCG@K}=\frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}.
$$

| Notation | Meaning |
| --- | --- |
| $g_r$ | The human grade of the asset that the **model placed at rank $r$**. It is shorthand for that asset's $g(q,a)$; it is not the model's score. |
| $2^{g_r}-1$ | The relevance **gain** assigned to that grade. Grades 0, 1, 2, 3 become gains 0, 1, 3, 7. The superscript is an exponent. |
| $\log_2(r+1)$ | The base-2 logarithm of the rank plus one. It is the discount denominator; $r+1$ makes the first-position denominator $\log_2 2=1$. |
| $1/\log_2(r+1)$ | The position weight, approximately 1, 0.631, and 0.5 at ranks 1, 2, and 3. Later results contribute less. |
| $\sum_{r=1}^{K}$ | Add the discounted gains at positions 1 through $K$, including both endpoints. |
| $\operatorname{DCG@K}$ | **Discounted Cumulative Gain**: the total relevance gain of the model's first $K$ results after applying the position discounts. |
| $\operatorname{IDCG@K}$ | **Ideal DCG**: the maximum DCG under the same catalog and human judgments, obtained by putting the highest-grade assets first. Use the best available judged assets, not only a reordered version of the model's retrieved candidates. |
| $\operatorname{NDCG@K}$ | **Normalized DCG**: the model's DCG divided by the ideal DCG for that query and cutoff. With nonnegative gains and a positive IDCG, it lies between 0 and 1; 1 achieves the ideal score. |

The query argument is omitted from these three metric names for readability; each is still computed for one query. A reported mean NDCG averages these per-query ratios, rather than dividing a sum of DCGs by a sum of IDCGs. Define the convention when IDCG is zero, for example excluding those queries and reporting their count separately. The exponential gain is a common choice, not a probability calibration: grade 3 does not mean 75% relevance. A linear gain is another possible policy; choose the gain rule to match how the product values relevance levels.

**AP and mAP: reward multiple relevant results appearing early**

The next example also uses **Average Precision (AP)**. For an evaluated ranking with $N_q$ positions and binary relevance labels:

$$
\operatorname{AP}(q)=\frac1{|R_q|}
\sum_{r=1}^{N_q}y_r\operatorname{Precision@r}(q),
\qquad
\operatorname{mAP}=\frac1Q\sum_{q\in\mathcal Q}\operatorname{AP}(q).
$$

- $N_q$ is the number of positions in the evaluated ranking for this query; $\sum_{r=1}^{N_q}$ adds over those positions.
- $y_r$ is 1 if the asset at rank $r$ is relevant under the binary rule, and 0 if it is nonrelevant. This formula assumes those binary labels are available under the declared judgment policy.
- $\operatorname{Precision@r}(q)$ means precision with the cutoff set to the current rank $r$. Multiplication by $y_r$ keeps this contribution only at relevant positions.
- $|R_q|$ is the number of known relevant assets, including those missed by the evaluated ranking. Missed positives contribute nothing to the numerator while remaining in the denominator.
- mAP means **mean AP**: average the per-query AP values over the $Q$ queries. It does not mean an average over asset classes in this retrieval setting.

This formula uses $|R_q|$ as the denominator. If reporting a truncated AP@K, explicitly state the cutoff and denominator convention, since implementations differ. As with recall, define how queries with no known positives and unjudged assets are handled. See 6.2 for a complete numerical example.

### 6.2 Work through one ranked list

Suppose the judged relevant set is A, C, and D. Grades are A:3, C:2, D:2, B:0. The model orders them:

```text
Rank 1: A, grade 3, relevant
Rank 2: B, grade 0, irrelevant
Rank 3: C, grade 2, relevant
Rank 4: D, grade 2, relevant
```

At cutoff three:

$$
\operatorname{Recall@3}=2/3,\quad
\operatorname{Precision@3}=2/3,\quad
\operatorname{RR}=1.
$$

MRR considers this query perfect because the first result is relevant, even though an irrelevant item occupies rank two and D is missing from the first three.

$$
\operatorname{DCG@3}=7+0+3/2=8.5,
$$

$$
\operatorname{IDCG@3}=7+\frac3{\log_2 3}+\frac32
\approx10.393,
\qquad \operatorname{NDCG@3}\approx0.818.
$$

For the complete binary ranking shown, average precision is:

$$
AP=\frac{1+2/3+3/4}{3}\approx0.806.
$$

It averages precision at the ranks containing relevant results. Mean AP in retrieval averages query AP values; genre-classification mAP instead averages AP over genres. State the averaging unit, cutoff, and treatment of incomplete judgments.

In practice, the full catalog is rarely exhaustively judged. Pool candidates from diverse systems, obtain independent reviews, and retain a representative query sample. Treat unjudged assets as unknown; do not claim that measured recall covers all truly relevant catalog assets when relevance judgments are incomplete. Targeted challenge sets diagnose failures but do not establish representative aggregate quality.

### Q16. How do you separate search errors from ordering errors?

Measure three layers:

1. **Representation quality:** Does exact vector search rank human-relevant assets highly?
2. **ANN approximation:** Does ANN recover the exact vector neighbors?
3. **Final ordering:** Does the reranker order the retrieved candidates well?

For an exact top-$M$ set $E_M$ and ANN top-$M$ set $A_M$:

$$
\operatorname{IndexRecall@M}=\frac{|A_M\cap E_M|}{M}.
$$

This measures agreement with exact vector search, not human relevance. A perfect index can faithfully retrieve irrelevant vectors if the embeddings are poor.

If a query has ten judged relevant assets, eight enter the candidate set, and five appear in the final top twenty, candidate relevance recall is 80% and final Recall@20 is 50%. The reranker can improve selection among the eight, but cannot recover the two that never reached it.

Also measure loss caused by filtering and duplicate handling. A candidate that disappears because of stale eligibility metadata is a different failure from a poor embedding.

### Q17. Can good aggregate results conceal important failures?

Yes. Report text and image queries separately, plus language, attribute composition, new assets, new creators, rendering style, clutter, missing descriptions, and rare categories. Include sample counts and uncertainty.

For example, if text queries are 95% of traffic, a traffic-weighted average can improve while image-query quality deteriorates substantially. Report both the traffic-weighted product metric and the separate modality results. A balanced challenge set answers a different question from the traffic-weighted average.

Track catalog coverage and duplicates too. A system that repeatedly returns popular near-copies may score well on common queries while failing the discovery of new creators. Decide whether and how diversity is part of the product objective rather than silently changing the ranking target.

### Q18. Why can better offline results fail online?

Offline queries and relevance pools may not reflect actual user intent. A model may improve attributes that users rarely request, introduce latency, or produce attractive previews that are difficult to use. Historical clicks reflect the previous system's exposure policy.

Run a suitably randomized experiment on the actual ranking change. Define successful asset selection or qualified use, not only thumbnail clicks, as candidate outcomes. Also track abandonment, no-result rate, eligibility failures, latency, and creator coverage. Choose the randomization unit and experiment duration according to the product and interference risks.

Inspect delayed qualified outcomes. More clicks without more useful selections can indicate worse search rather than success. Offline relevance, live product outcomes, and performance guardrails should be assessed together.

## 7. Serving and trade-offs · 31–35 minutes

### Spoken interview answer

> “For a text request such as ‘red wooden chair,’ I would encode the query once, search the compatible asset index, validate candidate eligibility, optionally rerank a bounded shortlist, and return twenty results. The image-query path adds decoding and crop handling before encoding. New assets enter through content rendering and encoding, while deletions are enforced immediately through eligibility checks. Model and index versions would move together.”

### 7.1 Walk through an actual request

```text
Query: red wooden chair
  → tokenize and encode → normalized 512-dimensional query
  → ANN → e.g. 200 candidate asset versions
  → filter eligibility and stale versions; resolve duplicates
  → optionally rerank using query and candidate evidence
  → select twenty eligible results
```

The candidate count is a parameter to measure. If filtering leaves too few results, compare filter-aware search, bounded over-retrieval, or bounded refill. Do not repeatedly increase search effort without a deadline. Filtering after retrieval can reduce recall; ANN algorithms differ in their ability to apply filters during search.

For a reference image, include server-side decoding, resizing, and any crop selection in the latency boundary. The mock excludes network upload transfer, not preprocessing after arrival. Automatic object selection can be evaluated, but a wrong crop can remove the requested object before retrieval starts.

### Q19. How much memory, and where does latency go?

For one 512-dimensional float16 vector per asset:

$$
2{,}000{,}000\times512\times2
=2{,}048{,}000{,}000\text{ bytes}.
$$

That is 2.048 decimal GB, approximately 1.91 GiB. It excludes index structures, IDs, metadata, extra replicas, raw renderings, and model memory. Six vectors per asset would require 12.288 GB of raw vectors before overhead.

Exact scoring performs approximately $N\times d$ multiply-accumulate operations per query, roughly 1.024 billion for this catalog, plus reading the vectors and selecting results. Real latency depends strongly on memory bandwidth, batching, hardware, and implementation. ANN reduces the search effort at a measured quality trade-off.

Define complete request latency:

$$
L=L_{queue}+L_{preprocess}+L_{query\ encoder}
+L_{search}+L_{filter}+L_{rerank}+L_{response}.
$$

An illustrative per-request budget might allocate 5, 10, 30, 15, 10, 50, and 5 ms to those components, totaling 125 ms. This is a planning example, not a measured p95. Stage-level p95 values cannot generally be summed to obtain end-to-end p95; trace complete requests under realistic load, including slow paths and timeouts.

Measure p95, p99, throughput, timeout rate, shortlist size, and route-specific latency. A cross-encoder can consume the budget even when ANN is fast. Define a validated fallback, such as returning the dual-encoder ranking when reranking times out, while preserving mandatory eligibility checks.

### Q20. Why must model and index versions be compatible?

Equal dimensions do not mean equal coordinates. A simple two-dimensional example makes this clear:

```text
Old model:
  chair query  → [1, 0]
  chair asset  → [1, 0]

New compatible model, with rotated coordinates:
  chair query  → [0, 1]
  chair asset  → [0, 1]
```

Each version matches internally, but a new query against the old asset has dot product 0 instead of 1. Real encoder changes can create more complex incompatibilities.

Version query and asset encoders, trained projections, preprocessing, rendering/view aggregation, and the index. Build the new compatible catalog vectors, validate retrieval and latency, switch the matching query encoder and index together, and keep rollback available. A new model must also remain compatible with newly arriving assets throughout index construction and cutover.

### Q21. What happens to a brand-new asset with no clicks?

Render its content, encode it with the active compatible asset pipeline, check eligibility, and insert its version into the index. Content supplies a cold-start representation without waiting for interactions.

Monitor time from upload to searchability, rendering failures, missing materials, and indexing lag. Content-based retrieval still needs quality audits for unseen styles and poor metadata; cold-start support does not guarantee perfect relevance.

For updates, replace the searchable active version and handle stale vectors explicitly. For deletion or loss of eligibility, prevent the asset from being returned immediately even if physical index cleanup happens later. A request-time eligibility check is still necessary when an index can be stale.

## 8. Three optional constraint changes

### 8.1 The catalog grows one hundred times

> “I would first measure index memory, search latency, and ANN recovery against exact search on suitable evaluation subsets. I would compare compression, sharding, filter-aware retrieval, and search-effort settings while keeping candidate and reranker budgets bounded. Any compression or approximation change must be evaluated for final relevance, not just speed.”

The catalog now has 200 million assets. Raw 512-dimensional float16 vectors alone require 204.8 GB, about 190.7 GiB, before overhead. Six views and replicas multiply that requirement.

Sharding distributes storage and search, but adds query fan-out, merge work, and tail-latency risks. Compression saves memory but may change neighbor ordering. Increasing ANN search effort can improve recovery while increasing latency. Measure the actual bottleneck before choosing a solution; there is no single threshold that makes a hundredfold growth free.

### 8.2 Attribute combinations fail

> “I would inspect whether the available views and metadata reveal each requested attribute, then check whether relevance judgments distinguish partial matches from full matches. I would add verified compositional positives and hard negatives, compare better views or resolution, and evaluate a shortlist cross-encoder on held-out attribute combinations.”

Use contrasts such as red wooden chair versus red metal chair, blue wooden chair, and red wooden table. Verify material rather than inferring it from a vague caption. Hold out some combinations to test whether the model composes attributes instead of memorizing frequent phrases.

If exact vector search already misses relevant combinations, improve representations or candidate scoring. If candidates contain the right items but the final ordering ignores attributes, improve reranking. If the attribute is invisible in every input, fix the evidence source rather than merely enlarge the model.

### 8.3 No click logs and only a small reviewed set

> “I would retain the pretrained compatible encoders, content rendering, and indexing. I would reserve independent reviewed queries for evaluation, cautiously train only lightweight components if they help, and spend additional annotation on representative queries and important failure slices.”

Behavioral logs are not required for content retrieval, but their absence limits preference learning and exposure analysis. Do not manufacture negatives from unobserved behavior. Prioritize a trustworthy evaluation set and a working baseline over training a large model on a handful of pairs.

Additional annotation can mix representative sampling with high-scoring uncertain or hard alternatives. Keep the sampling source explicit so active-learning candidates do not silently become the only estimate of population relevance.

## 9. A complete spoken answer

> “I would define graded query–asset relevance and clarify exact matching, substitutes, mandatory attributes, image-query intent, and eligibility. The training data would preserve multiple valid matches and asset versions, with unclicked content treated cautiously rather than as guaranteed negative.
>
> I would render each mesh into useful views and start with compatible pretrained text and image embeddings. View pooling and normalization produce reusable asset vectors. A query vector retrieves candidates cheaply; a bounded cross-encoder reranks only when it adds measurable value.
>
> Adaptation would use reviewed contrastive pairs, with known alternative matches handled as positives. I would compare lightweight updates before unfreezing encoders. Evaluation would separate exact representation quality, ANN recovery, candidate relevance recall, final ranking, and live qualified use.
>
> Serving would maintain a compatible encoder–index version, bounded latency, content-based indexing for new assets, and immediate eligibility enforcement for removed assets. The main trade-off is richer query–asset interaction versus reusable catalog computation and request cost.”

## 10. Self-check and source boundaries

Be able to explain why feature concatenation does not require prealigned spaces but cosine retrieval does; why a different asset ID is not automatically a negative; why ANN recall is not relevance recall; and why a reranker cannot recover missing candidates.

The [reference answer's source boundary](03_multimodal_asset_retrieval_solution.md#source-boundary) documents the provenance and seniority limits of the underlying reports. The specific task, numerical examples, multi-view design, metrics walkthroughs, and deployment decisions here are original educational material. Technical references support methods, not claims about Roblox production systems.
