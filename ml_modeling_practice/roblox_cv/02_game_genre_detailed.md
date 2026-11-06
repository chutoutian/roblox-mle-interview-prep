# Game Genre Classification in Detail: Interview Answers, Equations, and Examples

This is the detailed companion to the [reference answer](02_game_genre_solution.md) and [question document](02_game_genre_questions.md). It covers all seven design stages, Q1–Q21, and three optional changes. Each stage begins with a spoken interview answer, followed by the reasoning, examples, tensor shapes, and experiments behind it. Use the short answers to establish the complete design; expand individual sections when asked.

**Exercise boundaries:** The 100,000-game catalog, approximately 20 genres, 5,000-experience labeling budget, four possible images, and five-minute prediction deadline are hypothetical. This is an original teaching adaptation of a reported topic, not a transcript or a description of Roblox infrastructure. Genre examples below are illustrative.

## 0. The complete design before individual components

> “I would define genre labels for an experience version, clarify whether several genres can coexist, and build a reviewed dataset before choosing a complex model. I would compare simple text and visual baselines, then test a pretrained multimodal classifier that pools multiple images within each game. I would train only on known labels, choose genre-specific thresholds on development data, and store versioned predictions asynchronously after publishing. Offline classification and the effect of using those labels in discovery would be evaluated separately.”

```text
TRAINING
Experience-version snapshots
  → taxonomy and annotation guidelines
  → reviewed labels + known-label masks
  → related-game / duplicate grouping and time-aware splits
  → image features + text features
  → within-game image pooling → feature fusion → 20 logits
  → masked BCE → update trainable parameters
  → development: model selection, calibration, thresholds
  → locked final test

INFERENCE
Publish or update
  → immutable content snapshot
  → encode available images and text
  → fuse features → scores → selected labels / abstention
  → save content, taxonomy, model, and threshold versions

DISCOVERY
User request → read saved labels → filtering / recommendation

FEEDBACK
Random audits + disagreements + product outcomes
  → reviewed corrections and subsequent training versions
```

The classification model predicts what genres describe the game. A decision rule determines which labels are confident enough to display. The downstream discovery system decides how to use those labels. These are related but distinct responsibilities.

## 1. Problem formulation · 0–4 minutes

### Spoken interview answer

> “One example would be one experience version at publication or update time. I would agree on a taxonomy and explicit genre definitions with product, and allow multiple labels when gameplay genuinely spans genres. The model would output twenty sigmoid scores, selected labels, and an explicit uncertain state when evidence is insufficient. I would use only information available at that decision time and define both label-quality targets and downstream discovery outcomes.”

### 1.1 Define the prediction unit and output contract

Let $i$ identify a particular published version of an experience:

$$
x_i=\{\text{icon, screenshots, title, description, available metadata}\},
\qquad y_i\in\{0,1\}^{20}.
$$

The model aims to produce a score $p_{ic}$ for whether genre $c$ applies. Until validated and calibrated, treat sigmoid outputs as scores rather than guaranteed probabilities.

A response might contain:

```text
experience_id: game_17
content_version: v8
genre_scores: {obstacle_course: 0.87, roleplay: 0.74, ...}
selected_labels: [obstacle_course, roleplay]
status: predicted
taxonomy_version: genres_v2
model_version: classifier_v4
```

The status could instead be `insufficient_evidence`. Returning no confident genre is different from proving that none of the twenty genres applies. Clarify whether the taxonomy is intended to cover every game or permits out-of-taxonomy content.

### Q1. Is this multiclass or multi-label?

The answer follows the annotation contract. If a game can genuinely be both an obstacle course and roleplay, use multi-label classification:

$$
p_{ic}=\sigma(z_{ic}),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

Scores such as 0.87 and 0.74 can coexist. They need not sum to one. If product instead defines exactly one mutually exclusive primary genre, a softmax is appropriate:

$$
p_{ic}=\frac{e^{z_{ic}}}{\sum_{j=1}^{20}e^{z_{ij}}}.
$$

Do not use softmax merely because there are twenty names in the taxonomy. Also distinguish genuine multi-genre content from reviewer uncertainty: two valid genres can both be positive, whereas an unresolved category should remain unknown.

### Q2. Why use an experience version instead of a screenshot?

Genre describes a game, not an isolated image. An icon, a menu screenshot, and a gameplay screenshot are several observations of the same target.

If a game has four images, creating four independently labeled training examples gives it four times the influence of a one-image game unless carefully corrected. It can also put screenshots of the same game into both training and test.

Instead, keep images inside one example and aggregate their representations. Version the example because a game can change its mechanics or content. Labels and features for version v3 must not silently be paired with screenshots from v9.

### Q3. Is matching creator tags the same as improving discovery?

No. Creator tags are a possible source of labels, but may be incomplete, inconsistent, or written for promotion. Matching those tags measures agreement with creators, not necessarily correct genre or user benefit.

Distinguish three outcomes:

| Outcome | Example measurement |
| --- | --- |
| Annotation quality | Agreement with independently reviewed genre labels |
| Prediction quality | Per-genre precision, recall, and ranking metrics |
| Product usefulness | Users successfully finding suitable games through filters or recommendations |

A wrong inclusion can contaminate a genre filter; a wrong omission can hide a relevant game. Discuss which cost matters most for each product use. Recommendation engagement is useful evidence, but it does not redefine what the genre means.

## 2. Data and labels · 4–10 minutes

### Spoken interview answer

> “I would turn the taxonomy into annotation guidelines with positive, negative, and borderline examples. The five-thousand-experience budget would cover representative traffic plus targeted rare genres and difficult styles, with some independent reviews and adjudication. Creator tags would remain weak supervision. I would store unknown labels separately, snapshot all content at decision time, and use related-game groups, duplicate checks, later-time evaluation, and creator-held-out slices to avoid overstating generalization.”

### 2.1 What does a labeled example contain?

```text
experience_id, content_version, publication_timestamp
creator_id, related_game_group
icon_reference, screenshot_references
title_snapshot, description_snapshot, creator_tag_snapshot
genre_values:     [1, 0, 0, 1, ...]
known_label_mask: [1, 1, 0, 1, ...]
label_source: reviewed / creator_tag / rule / teacher
taxonomy_version, annotation_status
```

For a reviewed category, 1 means applicable and 0 means explicitly not applicable. A zero placeholder with a mask of 0 means unknown. Keep model-input missingness separate from label missingness: a missing screenshot and an unreviewed genre are different facts.

Create rules based on gameplay rather than only appearance. Two visually similar games may have different mechanics. If reviewers need gameplay footage or a play session to establish ground truth, record that evidence. A classifier given only screenshots and text may still have irreducible uncertainty; either provide additional deployable evidence or allow abstention.

### 2.2 How would you spend 5,000 reviewed experiences?

Reserve training, development, and final-test data before iterative model development. A rough planning allocation could be 3,500 / 750 / 750 experiences, but actual counts depend on grouping, time windows, annotation cost, and positive support. These are illustrative allocations, not a required ratio.

Keep a representative evaluation component and explicitly label any enriched rare-genre component. Do not fill the test set entirely with easy or handpicked examples. Within the training-label budget, sample across creators, languages, styles, and candidate rare genres; obtain explicit negatives as well as positives.

Double-review ambiguous examples and a random subset of ordinary examples. Adjudicate disagreements, track systematic confusion between neighboring genres, and revise guidelines before blaming the encoder. Keep annotation-training examples separate from the final model test.

### Q4. What is a weak label versus a gold label?

A creator tag, keyword rule, or automated teacher gives a weak label: it is cheap but may be wrong or incomplete. A gold label follows a carefully reviewed protocol and adjudication, but still needs auditing.

Example: the description says “RPG racing simulator” for promotion, while reviewed gameplay supports only racing. The text can be an input; its keyword matches should not automatically become trusted targets.

For a small initial system, train the baseline on reviewed data first. If adding weak supervision, track provenance, compare performance with and without it, and validate source-specific weighting on reviewed development data. Evaluate on independent reviews rather than the teacher's own predictions. Assigning a high weight does not make a weak label true.

### Q5. Why is a random screenshot split misleading?

Suppose screenshot A from game 17 is in training, screenshot B from game 17 is in test, and both show the same obstacle course. The model may recognize the game rather than infer genre for a new game.

Group related versions and copied content, then apply time-aware splitting. Keep future images out of training. For a future game that reuses an old asset, report a repeated-content slice separately from a new-content slice. Overall future-traffic quality can include real repetitions, but those repetitions should not be presented as evidence of unseen-game generalization.

To claim new-creator performance, ensure the corresponding creators contribute no training examples. Creator separation does not replace duplicate checks: different creators can copy the same content.

### Q6. Does an absent creator tag mean a negative label?

No. A creator may list only the primary genre or leave tags blank. Absence is not an explicit assessment.

```text
Reviewed labels: [1, 0, ?, 1]
Known mask:     [1, 1, 0, 1]
```

Only the three assessed positions contribute supervised loss. A fully unannotated example contributes no supervised genre loss, but can remain available for future annotation. The mask is an implementation of this selection; computing loss only at known indices is equivalent.

Masking does not fix selective annotation. If reviewers only assess genres already suspected by an old rule, important positives and negatives can remain absent. Include randomly sampled experiences reviewed across the taxonomy.

## 3. Baseline · 10–15 minutes

### Spoken interview answer

> “I would first establish a constant prevalence baseline and a TF-IDF text classifier using reviewed labels. Then I would test frozen pretrained image features and compare text-only, image-only, and fused models on the same splits. This tells me whether vision adds value and which errors justify more complexity. Building the service from scratch does not require pretraining an encoder from random weights.”

### 3.1 A complete low-cost text baseline

Combine the title and available description into a document. TF-IDF increases the weight of words frequent within that document but uncommon across the training corpus. One illustrative definition is:

$$
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)
\left[\log\frac{1+D}{1+\operatorname{df}(t)}+1\right].
$$

$D$ is the number of training documents and $\operatorname{df}(t)$ is the number containing term $t$. TF normalization and other implementation choices can vary. Fit the vocabulary and IDF on training data only; otherwise evaluation content influences the feature construction.

Train one regularized logistic classifier per genre, using reviewed targets. For a vocabulary of size $M$:

$$
X_{text}:[B,M],\qquad W:[20,M],\qquad
Z=X_{text}W^\top+b:[B,20].
$$

This baseline is useful for discovering whether terms such as “obstacle course” carry predictive information. It can fail on synonyms, keyword stuffing, missing descriptions, and gameplay that text does not describe.

### Q7. Why build a baseline if a powerful model is available?

It supplies a controlled reference for both quality and cost. If multimodal fusion improves macro-F1 only slightly while quadrupling encoding cost, investigate whether the gain occurs in important slices and survives uncertainty.

Compare models on the same label definitions and evaluation population. Inspect disagreements: a useful visual model may fix missing-description cases even if the aggregate improvement is small. A larger architecture without measurable benefit is not a stronger design argument.

### Q8. Does “from scratch” require random initialization?

Clarify the scope. Here, the dataset, label taxonomy, training objective, deployment path, and evaluation are new. Reusing pretrained visual and text representations is compatible with building that system.

With 5,000 reviewed experiences, learning a large visual representation from random weights would need a separate data and compute justification. Start by freezing a pretrained encoder and training a small head. A CLIP-style zero-shot genre comparison can be another diagnostic, but gameplay concepts may not be visible in images or captured by prompts; it does not replace reviewed evaluation.

### Q9. Can play time or next-week retention be inputs?

Not for a newly published experience. Those values do not exist at prediction time. Including them during training would create a system whose features cannot be reproduced at serving time.

An updated, established game may have historical behavior. If using it, define a different availability contract: only behavior observed before the current decision is allowed. Evaluate new-game and established-game cases separately rather than silently relying on history for a cold-start task.

## 4. Model design · 15–20 minutes

### Spoken interview answer

> “My candidate would encode each available image with the same pretrained visual encoder, pool valid image representations within each game, and encode the title and description with a text encoder. I would project the pooled image and text vectors, concatenate them with presence flags, and train a nonlinear genre head. I would compare this with score-level fusion and test realistic missing and misleading modalities.”

### 4.1 Trace one batch through the model

Assume eight experiences, at most four images each, and an illustrative visual width of 768:

```text
Image batch                             [8, 4, 3, 224, 224]
Valid images through shared encoder     [number_of_valid_images, 768]
Restore padded game/view structure       [8, 4, 768]
Pool valid views within each game        [8, 768]

Text token IDs                          [8, L]
Contextual text-token features           [8, L, 384]
CLS or valid-token pooling               [8, 384]

Image projection                        [8, 256]
Text projection                         [8, 256]
Concatenate plus two presence flags      [8, 514]
MLP → genre logits                      [8, 20]
Sigmoid → genre scores                   [8, 20]
```

The text width 384 and all projection widths are examples. Follow the chosen checkpoint's preprocessing and pooling convention. Text has one feature vector per token before pooling; images have patch or spatial features before producing one vector per image. These are separate from pooling several images into a game representation.

### Q10. How do you choose a CNN or ViT?

A CNN applies learned local filters and shares their weights across positions. A ViT represents an image as patch tokens and uses attention to exchange information between them. Compare pretrained transfer quality, image detail, runtime, memory, and the target data rather than assuming either family wins.

For a ViT with patch size 16 and a 224-pixel square input:

$$
N=(224/16)^2=196.
$$

A CLS-based variant produces 197 tokens including CLS, each perhaps 768-dimensional. Take the final CLS representation, or use the pooling convention of the checkpoint, to obtain the per-image vector. A plain global attention layer has a pairwise interaction term $O(N^2d)$; fixed local convolution scales linearly with the number of output spatial positions when channels and kernel size are fixed. Full model cost also includes projections, MLPs, and hardware effects.

If failures concern tiny gameplay details lost by resizing, compare resolution or local views. A larger head cannot recover information that preprocessing removed. If the input contains only promotional art, even a strong encoder may lack evidence of actual mechanics.

### Q11. How do you combine screenshots without creating independent games?

Let $h_{iv}\in\mathbb R^{768}$ represent view $v$ of game $i$, and let $r_{iv}=1$ mean that view exists. A masked mean is:

$$
h_i^I=\frac{\sum_{v=1}^{4}r_{iv}h_{iv}}
{\sum_{v=1}^{4}r_{iv}}.
$$

Apply this only when at least one image exists. With no images, supply an explicit missing representation and set the image-presence flag to 0. A padded view must not count as a real observation.

For a two-dimensional toy example, three valid view vectors might be $[2,0]$, $[0,2]$, and $[1,1]$. Their mean is $[1,1]$. Adding a padded zero view and dividing by four instead gives $[0.75,0.75]$, incorrectly changing the representation because a screenshot is missing.

Mean pooling treats views equally. To learn their importance, compare masked attention pooling:

$$
e_{iv}=w^\top\tanh(Uh_{iv}),\qquad
\alpha_{iv}=\frac{r_{iv}e^{e_{iv}}}{\sum_ur_{iu}e^{e_{iu}}},
\qquad h_i^I=\sum_v\alpha_{iv}h_{iv}.
$$

Again handle the all-missing case separately. Attention weights are learned through the game-level loss. They are not guaranteed explanations of causality. Evaluate whether an irrelevant icon or repeated screenshot dominates, and compare icon-only, one gameplay image, and all valid views.

### Q12. What are early, feature-level, and late fusion?

Terminology varies, so describe the actual information being combined:

| Approach | What is combined? | Main trade-off |
| --- | --- | --- |
| Token-level fusion | Text tokens and image patch features interact | Richer local relationships; higher cost |
| Feature-level fusion | Encoded game-image and text vectors | Small trainable head; compression can lose details |
| Score-level late fusion | Separate text and image genre predictions | Easy to diagnose; less cross-modal information |

This candidate uses feature-level fusion:

$$
u_i=[P_Ih_i^I;P_Th_i^T;m_i^I;m_i^T]\in\mathbb R^{514},
\quad z_i=W_2\operatorname{ReLU}(W_1u_i+b_1)+b_2.
$$

Here $m_i^I,m_i^T$ are input-presence flags, not the known-label mask. With a hidden width of 256, $W_1:[256,514]$ and $W_2:[20,256]$.

The image and text encoders do not have to use aligned embedding spaces for concatenation. The classifier learns separate weights for their coordinates. Projection to 256 dimensions controls size; it does not by itself create semantic alignment. By contrast, directly comparing image and text by cosine similarity would require compatible representations.

A missing description can use a learned default vector and an explicit text-presence flag. Train with naturally missing cases and, where appropriate, modality dropout so the head can use one available modality. A missing field differs from a misleading field; test both. With no meaningful content, abstain rather than interpreting an all-zero vector as confident evidence.

## 5. Loss and training · 20–26 minutes

### Spoken interview answer

> “I would start with masked binary cross-entropy for the twenty coexisting genres. I would train the projections and fusion head with frozen encoders, then compare selective fine-tuning at a lower encoder learning rate. Rare-genre improvements would begin with label quality and data coverage, followed by sampling or loss weighting if justified. Development metrics would select the checkpoint, and augmentations would need to preserve genre evidence.”

### Q13. How do sigmoid, BCE, and missing-label masks fit together?

For one known genre label:

$$
\ell_{ic}=-y_{ic}\log p_{ic}-(1-y_{ic})\log(1-p_{ic}).
$$

A positive target penalizes a low genre score; a negative target penalizes a high one. For a batch, with annotation mask $m_{ic}$:

$$
L=\frac{\sum_{i,c}m_{ic}\ell_{ic}}{\sum_{i,c}m_{ic}}.
$$

Skip an optimizer update if the denominator is zero. Unknown targets can be stored as finite placeholders because the mask determines which entries are supervised. A NaN placeholder must be replaced or excluded before evaluating BCE; multiplying NaN by zero does not make it finite.

For example:

```text
Known targets: [1, 0, ?]
Scores:        [0.8, 0.1, 0.9]
Mask:          [1, 1, 0]
```

$$
L=\frac{-\log0.8-\log0.9}{2}\approx0.164.
$$

The third score is neither rewarded nor penalized directly for this sample. With unweighted BCE, the derivative of one known loss term with respect to its logit is:

$$
\frac{\partial\ell}{\partial z}=p-y.
$$

Masking removes an unknown target's direct contribution. The shared encoder can still change because other known targets update it. Use a stable logits-based loss implementation. Sigmoid is appropriate because genres can coexist; softmax would force their scores to compete.

### 5.1 What does class weighting actually change?

One option weights the whole category loss:

$$
L=\frac{\sum_{i,c}m_{ic}w_c\ell_{ic}}{\sum_{i,c}m_{ic}w_c}.
$$

This changes how much a category contributes relative to others. It is different from weighting positives within a category:

$$
\ell_{ic}^{weighted}=-\beta_c y_{ic}\log p_{ic}
-(1-y_{ic})\log(1-p_{ic}).
$$

The latter directly increases the cost of missing positives in that category and can shift scores. Validate the choice and check calibration and thresholds afterward. Neither formulation automatically fixes limited positive diversity or wrong labels.

### Q14. Why freeze, and when should you unfreeze or use adapters?

Initially freeze both encoders and train the projection, missing-modality representations, and fusion head. This limits trainable parameters and establishes whether existing features are sufficient. A frozen encoder still runs at inference; freezing is primarily a training choice.

If reviewed errors show that game-specific art or mechanics are poorly represented, compare unfreezing upper layers with a lower learning rate. For example, head and encoder learning rates might start around $10^{-3}$ and $10^{-5}$ in an experiment; tune them rather than treating them as requirements.

Lightweight adaptation is another candidate. For example, the low-rank update idea in [LoRA](https://arxiv.org/abs/2106.09685) replaces a full trainable update with:

$$
W'=W+\Delta W,\qquad \Delta W=AB,
\quad A\in\mathbb R^{d_{out}\times r},\ B\in\mathbb R^{r\times d_{in}}.
$$

The base $W$ remains frozen. This controls the number of trainable parameters; whether it improves genre prediction must be tested. Fewer trainable parameters do not eliminate forward computation through the encoder.

AdamW is a reasonable starting optimizer for the trainable components. Tune learning rate, decay, and checkpoint using development data. If training improves while validation worsens, inspect label quality, sample coverage, duplicate leakage, and overfitting before simply extending training.

### Q15. How do you handle rare genres and valid augmentation?

Start by inspecting the label definition and actual positive count. Fifty near-identical games are less informative than fifty diverse examples. Obtain additional reviewed positives and confusing benign examples from adjacent genres, while tracking how targeted sampling differs from production prevalence.

Then compare balanced sampling, category weights, or positive weights. Sampling a multi-label game for one rare genre also changes the distribution of its other genres. Focal loss can reduce easy-example dominance, but may emphasize mislabeled hard cases; it should replace the relevant BCE terms rather than being automatically added to them.

An augmentation is valid only if the transformed input still supports its label. Cropping away the only racing scene while keeping a racing target creates misleading supervision. Use mild transformations verified on the domain, and consider whether image-text relationships remain consistent. Do not randomly exchange descriptions between games while retaining original labels.

## 6. Offline and online evaluation · 26–31 minutes

### Spoken interview answer

> “I would report per-genre precision and recall, macro- and micro-F1 at chosen thresholds, and mean average precision for ranking quality. Thresholds would be chosen on development data according to genre-filter precision and coverage. I would compare text-only, image-only, and fusion models on the same data, including missing modalities and unfamiliar creators. A limited randomized rollout would then measure whether using the labels improves discovery, with incorrect-genre and latency guardrails.”

### Q16. What do macro-F1, micro-F1, and mAP measure?

For a genre, count TP, FP, and FN after applying its threshold:

$$
P_c=\frac{TP_c}{TP_c+FP_c},\quad
R_c=\frac{TP_c}{TP_c+FN_c},\quad
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}.
$$

Macro-F1 gives each evaluated genre equal weight:

$$
\operatorname{MacroF1}=\frac1C\sum_cF1_c.
$$

Micro-F1 pools the counts first:

$$
\operatorname{MicroF1}=\frac{2\sum_cTP_c}
{2\sum_cTP_c+\sum_cFP_c+\sum_cFN_c}.
$$

A two-genre example shows why both matter:

| Genre | TP | FP | FN | F1 |
| --- | --- | --- | --- | --- |
| Common genre | 90 | 10 | 10 | 0.900 |
| Rare genre | 1 | 1 | 9 | 0.167 |

Macro-F1 is about $(0.900+0.167)/2=0.533$. Micro-F1 is $182/(182+11+19)\approx0.858$. The strong common genre makes the pooled score look much better than the equally weighted category average.

For mAP, rank experiences by each genre's score and compute binary average precision for that genre. For a fully judged ranking without ties, one form is:

$$
AP_c=\frac1{n_c^+}\sum_{r:y_r=1}\operatorname{Precision@}r,
\qquad mAP=\frac1C\sum_cAP_c.
$$

If the ordered labels for one genre are $[1,0,1]$, then $AP=(1+2/3)/2\approx0.833$. AP evaluates ranking without choosing one threshold; F1 evaluates thresholded decisions. This is genre-classification AP over experiences, not object-detection mAP. The official metric definitions distinguish [F1 averaging](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Exclude unknown labels from that genre's supervised metric, and state the evaluation coverage. A selectively reviewed subset is not necessarily representative. Report positive counts, undefined metric handling, and uncertainty; do not silently treat unknowns as negatives or hide genres with no evaluable support.

### Q17. Must every threshold be 0.5?

No. Prevalence, calibration, and the cost of an incorrect label differ across genres. A strict discovery filter may require high precision, whereas an internal candidate feature may tolerate more uncertainty.

On development data, sweep each genre's threshold and examine the precision–recall trade-off. An example objective is maximizing recall subject to a minimum precision requirement and sufficient label support. The precision requirement is a product decision, not a universal constant.

If probabilities are used directly downstream, check calibration on representative data. An assigned score around 0.8 should correspond to an approximately 80% empirical positive rate among comparable cases. Reweighting or oversampling can disturb this interpretation. Use separate development data or appropriate cross-fitting for selection and calibration; leave final test untouched.

An all-low-score result should remain uncertain if the input lacks evidence. Forcing the highest score to become a label can introduce confident-looking mistakes. Conversely, several categories above their thresholds can all be selected when the taxonomy permits them.

### 6.1 Which ablations justify complexity?

Compare constant prevalence, TF-IDF, frozen image-only, pretrained text-only, and fused models on the same splits. Add experiments for one image versus multiple images, mean versus attention pooling, frozen versus adapted encoders, and realistic missing or keyword-stuffed descriptions.

Record quality gains and encoding cost together. Inspect creator-held-out, new-style, language, and rare-genre slices. A small aggregate gain can be valuable if it substantially helps missing-text games; a large average gain can conceal a failing rare genre.

### Q18. Why can better offline classification fail to improve engagement?

The classifier may improve labels that the recommender rarely uses. Existing discovery rules may already cover the same information, or a new label may send users to a technically correct but unsuitable game. Engagement also depends on rank, presentation, and user intent.

Test the actual downstream use through a randomized rollout, keeping unrelated ranking changes controlled where possible. Choose the randomization unit according to the product and interference risks. For example, measure successful genre-filter discovery, qualified play after selection, wrong-genre complaints, and latency. Define qualified outcomes before the experiment rather than selecting a favorable metric afterward.

Offline labels establish genre correctness; randomized deployment provides evidence about the effect of using those labels. Neither metric alone establishes the other.

## 7. Serving and trade-offs · 31–35 minutes

### Spoken interview answer

> “I would compute predictions asynchronously when an experience is published or updated, and save them against the content version. Discovery requests would read those saved results rather than run image and text encoders. I would monitor the five-minute freshness bound, queue age, prediction failures, and audited genre quality. Caches, quantization, and reduced views would be evaluated for actual savings and quality loss, with compatible versions and rollback.”

### Q19. What runs at publication versus discovery request time?

```text
Publish game version v8
  → fetch that version's images and text
  → encode, pool, fuse, predict
  → select labels with versioned thresholds
  → save prediction for v8

Discovery request
  → read saved labels
  → apply filter or recommendation logic
```

The five-minute requirement concerns how quickly the new version obtains a usable result. It includes waiting and preprocessing, not only model execution:

$$
L_{freshness}=L_{queue}+L_{fetch}+L_{preprocess}+L_{encode}
+L_{head}+L_{write}.
$$

Batching can increase throughput, but excessive batch waiting can violate freshness. Measure the percentage of versions predicted within the deadline, queue age, and sustained and peak throughput.

If v9 arrives before v8's job finishes, the late v8 result must not overwrite the active v9 prediction. Save and activate outputs with content-version checks. On failure, return unknown or an explicitly stale prior result if the product allows it; do not pretend that old content was freshly evaluated.

### Q20. How do caching and quantization help, and what can go wrong?

Cache per-image features by immutable image content and encoder/preprocessing version. A text-only edit can then reuse unchanged visual features. If the encoder is fine-tuned, its old cached features may no longer match the head and must be regenerated or kept within the old compatible pipeline.

Quantization uses lower-precision representations or operations to reduce storage or computation. Savings depend on runtime and hardware support. Measure complete encoding cost and throughput, and recheck rare genres, calibration, and threshold decisions after numerical changes.

For example, moving a score from 0.701 to 0.698 can flip a label at threshold 0.7 even though the numerical change is small. Aggregate accuracy alone can hide changes concentrated near operating thresholds.

### Q21. What changes for a new genre or rendering style?

A new genre needs a definition, annotation guidelines, reviewed positive and negative examples, new outputs, thresholds, and a taxonomy version. Expanding the head from twenty to twenty-one outputs is not sufficient. Old examples unreviewed for the new genre need an unknown mask entry, not an automatic negative.

A new style instead raises a representation and distribution question. Audit that slice, verify preprocessing and labels, collect targeted examples, and compare adaptation. Threshold changes alone cannot recover evidence absent from the representations.

Version the taxonomy, preprocessing, encoders, fusion head, calibration, and thresholds together. Monitor input missingness, label distributions, freshness, audited errors, and style-specific performance. Distribution changes prompt investigation; labeled audits establish whether quality changed.

## 8. Three optional interviewer changes

### E1. Descriptions are untrustworthy or keyword-stuffed

> “I would measure text-source reliability and compare image-only, text-only, and fused errors. I would train on realistic missing and misleading descriptions, consider source-quality features or learned modality gating, and obtain labels grounded in actual gameplay evidence. If the available content does not reveal the mechanics, I would abstain or request additional evidence.”

A gate can produce a text weight from the available evidence, but must be trained and tested; a confident text encoder is not proof of truth. Include counterexamples with the same misleading keywords on different games. Avoid defining gold labels from those same keywords, which would reward the shortcut you want to prevent.

### E2. A rare genre has only 50 reviewed examples

> “I would first check whether fifty means fifty positives or fifty total reviews, inspect annotation consistency, and count positives in each evaluation split. I would report uncertainty, inspect confusion with neighboring genres, and prioritize additional diverse reviews before increasing model capacity.”

If the test has only ten positives, one missed game changes recall by 10 percentage points. Enriched data can help training and diagnosis, but representative evaluation is still needed for production precision. Check whether the rare genre's defining mechanics are visible in the inputs; loss weights cannot invent missing evidence.

### E3. Explanations plus a fivefold reduction in GPU cost

> “I would define an auditable explanation as supporting images or text spans and the model's scores, without claiming that attention weights or generated rationales prove causality. I would profile the dominant compute, compare caching, fewer views, smaller encoders, distillation, or supported quantization, and measure quality after each change.”

Profile before promising a factor of five. If image encoding accounts for 80% of cost and becomes five times cheaper while everything else is unchanged:

$$
\text{new cost}=0.8/5+0.2=0.36,
\qquad\text{overall speedup}=1/0.36\approx2.78.
$$

Optimizing one component fivefold does not produce a fivefold end-to-end improvement. Cost and latency are also different measurements; define the GPU-cost denominator, such as cost per newly processed version, and quantify any remaining conflict with the quality target.

## 9. A complete spoken answer

> “I would define genre at the experience-version level and agree on a multi-label taxonomy. I would create reviewed labels with explicit unknowns, use creator tags only as tracked weak supervision, and prevent related games or future versions from leaking across evaluation boundaries.
>
> I would start with TF-IDF text and frozen visual baselines. My multimodal candidate pools valid image representations within each game, combines them with a pooled text representation and missingness flags, and produces twenty sigmoid scores. I would train the small head with masked BCE before comparing selective encoder adaptation.
>
> I would use per-genre precision and recall, macro- and micro-F1, and mAP, selecting thresholds on development data. Missing descriptions, rare genres, new creators, and unfamiliar styles need separate checks. A controlled rollout would establish whether using the labels helps discovery.
>
> Predictions would be computed at publish or update time and saved with compatible content and model versions. Discovery reads stored labels. The main trade-off is richer evidence versus annotation and encoding cost; the next experiment is a controlled text-only, image-only, and fusion comparison.”

## 10. Self-check and source boundaries

Before finishing, trace one experience through input snapshots, view pooling, text pooling, fusion, logits, loss, thresholds, saved labels, and product use. Be able to explain why a missing label is not negative and why a screenshot is not an independent game.

The provenance and level limitations of the collected topic are documented in the [reference answer](02_game_genre_solution.md#source-boundary-and-related-notes). This detailed design and its numerical examples are original practice material. Method links support the relevant modeling ideas, not claims about actual Roblox systems or interview frequency.
