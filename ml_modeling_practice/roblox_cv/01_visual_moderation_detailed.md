# Visual Moderation in Detail: Interview Answers, Equations, and Examples

This is a detailed English expansion of the [original solution](01_visual_moderation_solution.md), aligned with the [question set](01_visual_moderation_questions.md): seven stages, Q1–Q21, and three changes in requirements. Each section starts with a spoken interview answer, followed by concrete methods, mathematics, and follow-up explanations. Use the spoken answers to practice the main narrative, and the expanded material to build understanding and handle follow-up questions. You do not need to present all of it within 35 minutes.

**Scope of the exercise:** This is an original mock interview. The five policies, 250,000 training candidates, two million uploads per day, 300 ms p95 latency, and 1% human-review capacity are assumptions for practice. The original question does not name the five policies. Examples involving gore, hate symbols, and similar content are illustrative; they do not describe Roblox's official categories or actual infrastructure.

## 0. Start with the Complete Pipeline

> “I would treat this as multi-label image classification followed by a separate routing decision. I would first define the policies and error costs, build reliable labels and leakage-safe splits, then train a small head on a pretrained image encoder. I would evaluate per-policy detection and the complete routing policy under the review budget. Model improvements would follow error analysis, and independent audits would measure missed violations after deployment.”

```text
Offline training
Images + adjudicated labels + which policies were reviewed
  → Group duplicate assets; split into train / development / test
  → Apply preprocessing matched to the pretrained model
  → image encoder → 5 logits
  → Compute loss only for known labels → update trainable parameters
  → Development: select model, calibrate, select routing thresholds
  → Lock the complete configuration and evaluate on test

Online inference
New image → decode / resize / pad / normalize
  → encoder → policy scores → calibration
  → Apply policy-specific thresholds: allow / restrict / stronger model / human review

Feedback
Independent random audits + human-review outcomes + verified appeals
  → New adjudicated labels → subsequent training and evaluation versions
```

The encoder turns pixels into a feature vector; the head turns features into five scores; routing turns scores into actions. **Explaining these three jobs separately makes the overall design easier to communicate.**

## 1. Problem formulation · 0–4 minutes

### Spoken interview answer

> “One prediction example is one uploaded image at its initial moderation decision. I would clarify the five policy definitions, whether visual evidence alone is sufficient, and whether we only need image-level flags or also need locations. Since policies can coexist, the model outputs five sigmoid scores. These scores are separate from enforcement: a flagged image may be reviewed or restricted depending on policy severity and the confidence required for that action. I would prioritize severe-category recall while constraining human-review volume and false restrictions.”

### 1.1 First define x, y, and the action

Let the $i$th example be an uploaded image $x_i$, with the target:

$$
y_i=(y_{i1},y_{i2},y_{i3},y_{i4},y_{i5}),\qquad y_{ik}\in\{0,1\}.
$$

$y_{ik}=1$ means the image violates policy $k$. The stored data also needs a known mask, explained in the next section.

For example, an image that violates two policies could have the label:

```text
y = [0, 1, 1, 0, 0]
```

The target is “whether the image violates the current policy,” rather than “whether it was later reported” or “whether the historical system removed it.” The latter two outcomes are affected by user behavior and the previous system. They can generate annotation candidates, but cannot directly replace ground-truth labels.

The input must consist of information available at the initial moderation decision. Later report counts and final appeal outcomes cannot be inputs to that prediction; using them would leak future information.

The model produces $p_{ik}$, attempting to estimate $P(y_{ik}=1\mid x_i)$. Treat unvalidated sigmoid outputs as scores first; whether they can be interpreted as probabilities requires checking calibration. The final action comes from a separate rule:

$$
a_i=g(p_{i1},\ldots,p_{i5};\text{severity, thresholds, capacity}).
$$

### Q1. Why five sigmoids instead of softmax?

Softmax is suitable for mutually exclusive classes:

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^{5}e^{z_j}},\qquad \sum_kp_k=1.
$$

If an image violates two policies, we want both scores to be high, such as 0.95 and 0.90. The sum constraint in softmax prevents that representation. Instead, apply a sigmoid to each logit separately:

$$
p_k=\sigma(z_k)=\frac{1}{1+e^{-z_k}}.
$$

The five scores do not have to sum to 1. **Independent sigmoids describe the output and binary-loss formulation; they do not imply that real policy violations are statistically independent.** The shared encoder can still learn signals of co-occurrence, but the five marginal scores do not fully describe the joint distribution.

If all five policies have been reviewed and every label is 0, the image is benign under these five policies. If some policies are unreviewed, zeroes for all known labels are insufficient to conclude that the image is entirely safe. There is also no need to add a benign softmax class that competes with the five policies.

### Q2. How do you choose classification, detection, or segmentation?

| Output type | Question answered | Required supervision | When to use it |
| --- | --- | --- | --- |
| Image classification | Does the image contain a violation of a given policy? | Policy labels for each image | The only requirement is deciding whether the image needs further handling |
| Object detection | Where are the relevant objects? | Bounding boxes and object classes | Locating small objects, showing evidence, or generating crops |
| Segmentation | Which pixels belong to the relevant region? | Pixel masks | Precise obscuring or local editing |

Start with image-level classification because the task only requires identifying unsafe images. Consider detection or segmentation if later requirements include obscuring violating regions, or if missed violations are concentrated in very small symbols.

Detecting an object does not necessarily establish a policy violation. For example, whether a game weapon violates a policy may depend on the scene, how it is depicted, and the policy definition. A localization module supplies evidence; the policy decision may still require the full context.

### Q3. Are false negatives more serious than false positives? Specify the action

First fix one policy and one flagging threshold:

| Actual status | Model flags | Model does not flag |
| --- | --- | --- |
| Violation | TP: detected violation | FN: missed violation |
| Benign | FP: false alarm | TN: correctly allowed |

If a flag only triggers human review, the main costs of a false positive are review resources and publication delay; a false negative may allow harmful content to go live. If a flag triggers an automatic restriction, a false positive harms a legitimate creator and has a substantially higher cost. Therefore, it is too broad to say that false positives do not matter.

A simple binary decision model helps explain the threshold. Assume the probability $p$ is calibrated, the only actions are “allow” and “restrict,” and correct decisions have zero cost:

$$
\operatorname{Risk}(\text{allow})=C_{FN}p,\qquad
\operatorname{Risk}(\text{restrict})=C_{FP}(1-p).
$$

Choose restriction when:

$$
p>\frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

For example, if missing a severe violation costs 100 times as much as incorrectly restricting a benign image, this idealized model gives a threshold of $1/101$. **This number only explains how costs affect thresholds; it cannot be used directly as a production threshold.** Real systems also involve human review, multiple policies, probability errors, and review-capacity constraints.

The main interview objective can be: “Improve recall for severe categories while satisfying the review budget and limits on false restrictions.”

## 2. Data and labels · 4–10 minutes

### Spoken interview answer

> “I would store an image, a five-dimensional label vector, and a mask indicating which policies were actually reviewed. Reports and previous model flags would generate annotation candidates, but they would not automatically become ground truth. I would combine those candidates with randomly sampled uploads, using clear annotation guidelines and adjudication for ambiguous cases. Before splitting, I would group duplicate assets, use a later-time final test, and separately evaluate unseen creators and styles. Fully reviewed representative data would support calibration, threshold selection, and traffic-level evaluation.”

### 2.1 What exactly does a training example store?

```text
image_id:          img_123
image_reference:   reference to the original image
upload_timestamp:  upload time
creator_id:        used for creator slices / grouping
asset_family_id:   group containing the original image and near-duplicate variants
label_values:      [1, 0, 0, 0, 0]
known_mask:        [1, 1, 0, 0, 1]
label_source:      random_audit / report_review / appeal_review
policy_version:    v3
annotation_status: adjudicated
```

Here, `label_values=0` for policies 3 and 4 is merely a finite numerical placeholder. Its meaning is determined by `known_mask=0`; it does not mean the image was confirmed benign for those policies. Keeping the source and version makes label differences traceable. Keeping the creator ID supports analysis and splitting; it does not mean the ID must be a model input.

### Q4. How do you handle missing labels?

Define:

$$
m_{ik}=\begin{cases}
1,&\text{policy }k\text{ was assessed for image }i,\\
0,&\text{otherwise}.
\end{cases}
$$

If this image was reviewed only for policies 1, 2, and 5, it supplies supervision only for those three positions:

$$
\mathcal L_i\propto \ell_{i1}+\ell_{i2}+\ell_{i5}.
$$

Policies 3 and 4 produce no direct supervised gradient for this example. You cannot turn “not reviewed” into “reviewed and confirmed benign”; that would incorrectly teach the model not to predict those violations.

However, masking only addresses incorrect negative labels; **it does not automatically fix selection bias**. For example, if the previous system reviewed only images it considered suspicious, the training set may still lack violations that the old system could not detect, even with a correct mask. Supplement it with randomly sampled uploads that are fully reviewed for all five policies.

### 2.2 What does each data source contribute?

| Source | Purpose | Limitations and handling |
| --- | --- | --- |
| Historical human reviews | Initial supervised data | Check previous policy versions, selection bias, and review quality |
| User reports | Find violations, difficult examples, and new styles | A report is not a confirmed violation; review it before assigning a label |
| Randomly sampled uploads | Cover ordinary traffic and missed violations | Rare policies yield few positive examples, making annotation relatively expensive |
| High-scoring or uncertain model examples | Improve annotation efficiency and reveal decision-boundary problems | Model-selected examples are not representative of overall traffic |
| Appeals and their subsequent reviews | Find false restrictions and label errors | An appeal alone does not establish that the model was wrong |

The training set can deliberately enrich positives and hard negatives. Sets used to estimate production precision, review volume, and calibration must reflect real traffic or have reliable sampling weights.

For example, suppose 1% of actual traffic violates policy, but the training set is sampled to contain 50% positives. This training set helps the model see violating patterns, but its measured precision cannot be directly interpreted as production precision.

### Q5. Why is using only reported content biased?

Represent “was reported” with a selection variable $S$. Reported data approximately comes from:

$$
P(x,y\mid S=1),
$$

while the system needs to handle the overall upload distribution $P(x,y)$. These distributions usually differ.

For example, assets with more exposure are more likely to be reported; some niche styles may receive fewer reports; and violations that nobody notices never enter the dataset. The model may end up learning which styles are more likely to be reported instead of which content truly violates policy.

Use reports to add valuable examples while preserving independent random audits. If using stratified sampling, record the sampling probability for each stratum. When coverage is sufficient and probabilities are known, evaluation can use inverse-probability weights $w_i=1/q_i$. If some traffic always has a sampling probability of 0, simple weighting cannot recover performance on it.

### 2.3 How do you improve annotation quality?

First translate the policies into annotation guidelines: positive examples, benign counterexamples, borderline cases, and the context reviewers may use for each policy.

For example, blood-red game effects may resemble depictions of actual injury, and a symbol may have different meanings in different scenes. Reviewers should apply the same version of the rules. If the task requires context beyond the image, explicitly include that context in the input or mark the example as needing additional information. A model with only image inputs cannot be expected to infer invisible facts.

Arrange independent reviews for ambiguous examples, with disagreements adjudicated by a senior reviewer. Examine disagreement rates by policy and visual style. If disagreements cluster around a particular boundary, improve the rules and labels before considering a different model.

Keep a carefully adjudicated gold set to measure annotation quality and model performance. “Gold” is a quality objective; “representative” is a sampling objective. A handpicked difficult set can be highly reliable without representing production traffic, so retain both representative examples and a targeted challenge set.

### Q6. How do you split train / validation / test to prevent leakage?

One concrete approach is:

1. Find identical files using exact hashes, then use perceptual hashes or image-feature retrieval to identify near-duplicate candidates. Verify them and form asset groups.
2. Define training, development, and final-test windows by time. Within development, reserve data for model selection, calibration, and routing tuning. With limited data, cross-fitting is an option, but the final test must remain isolated.
3. Set an isolation rule for duplicate groups spanning time windows, such as excluding them from the main “new asset” test and reporting a separate repeated-upload slice. Do not move future images into training to keep groups together.
4. Prepare separate unseen-creator / unseen-style slices. To evaluate new creators, ensure those creators did not contribute to training.
5. Freeze test images, labels, and the policy version. Do not use test results to keep tuning thresholds.

Putting an original image in training and a recolored or bordered version in test lets the model earn an inflated score by recognizing similar assets. Grouping only by creator is insufficient because different creators may reupload the same image. Deduplication alone also does not simulate new styles that emerge in the future.

These splits answer different questions: a temporal split tests future traffic; isolating duplicate groups tests generalization to new assets; holding out creators tests new creators. Report them separately instead of claiming that a single ordinary random split validates every capability.

## 3. Baseline · 10–15 minutes

### Spoken interview answer

> “My baseline would be a pretrained image encoder with a linear five-output head. I would initially freeze the encoder, preserve aspect ratio during resizing, and use the preprocessing expected by the checkpoint. For a batch of eight images, the output is an eight-by-five logit matrix. I would train with masked binary cross-entropy, compare with a prevalence baseline, and inspect errors by policy, object size, text presence, and visual style before increasing complexity.”

### 3.1 Why start with a pretrained encoder and a linear head?

A pretrained model has already learned image features from other data or tasks. Reuse those features and first learn how they relate to the five policies. This quickly establishes a complete reference system and helps diagnose whether the main problem lies in labels, resolution, or representation quality.

Start with an existing CNN or ViT checkpoint. Choose based on performance on the current data, latency, and GPU memory; there is no need to claim that ViTs are always better than CNNs.

Example preprocessing: resize while preserving aspect ratio, pad to $224\times224$, convert to RGB, and apply the normalization required by the checkpoint. Padding reduces the risk of a direct center crop removing violating details near an edge. The padding color and rules must also be consistent.

### Q7. What changes with freezing, partial fine-tuning, and full fine-tuning?

Write the model as:

$$
h=f_\theta(x),\qquad z=Wh+b.
$$

| Strategy | Parameters updated | Purpose and cost |
| --- | --- | --- |
| Frozen encoder | Only $W,b$ | Quickly test whether pretrained representations are useful; limited ability to adapt to a new visual domain |
| Partial fine-tuning | Head + last few encoder layers | Adapt higher-level features to game images while controlling training cost |
| Full fine-tuning | All parameters | Greater adaptability, with greater needs for data, computation, and protection against overfitting |

A frozen encoder still performs a forward pass; freezing does not automatically make production inference faster. If offline features remain fixed, cache embeddings to speed up head experiments. Once the encoder or relevant preprocessing changes, regenerate the cache.

### Q8. How do tensor shapes change at each step?

The following is an educational example using ViT/16 with a hidden dimension of 768. The patch representation comes from the [original ViT paper](https://arxiv.org/abs/2010.11929).

Input batch:

$$
X\in\mathbb R^{8\times3\times224\times224}.
$$

Split each image into $16\times16$ patches:

$$
N=(224/16)^2=196,\qquad D_{patch}=3\times16\times16=768.
$$

Flatten each patch and project it into a 768-dimensional token. The raw patch length happens to be 768 here; these dimensions are not equal in every ViT configuration. After adding the summary token and position embeddings:

```text
RGB images                      [8, 3, 224, 224]
flattened patches               [8, 196, 768]
projected patch tokens          [8, 196, 768]
append CLS + add positions      [8, 197, 768]
Transformer blocks              [8, 197, 768]
take final CLS representation   [8, 768]
linear policy head              [8, 5]
sigmoid                         [8, 5]
```

In batch form:

$$
H\in\mathbb R^{8\times768},\quad W\in\mathbb R^{5\times768},\quad
Z=HW^\top+b\in\mathbb R^{8\times5}.
$$

The number of head parameters is:

$$
5\times768+5=3845.
$$

Each row represents one image, and the five columns represent the five policies; a row does not represent a class. Both labels $Y$ and masks $M$ have shape $[8,5]$.

If one image has logits:

$$
z=[-3,2,1,-2,-4],
$$

then:

$$
p\approx[0.047,0.881,0.731,0.119,0.018].
$$

These are only five policy scores. Policy-specific routing thresholds are still needed to determine an action.

### Q9. Why can predicting every image as benign still yield high accuracy?

Suppose only 100 of 100,000 images violate a particular policy. Predicting all of them as benign gives:

$$
\text{Accuracy}=99{,}900/100{,}000=99.9\%,\qquad
\text{Recall}=0/100=0.
$$

Therefore, do not report accuracy alone for the baseline. At minimum, examine AP, recall, precision, and review volume for each policy. A constant prevalence predictor provides a reference: if the model does not clearly outperform scores with no ability to distinguish examples, first inspect the data and training pipeline.

The baseline also establishes an error breakdown: are missed violations concentrated in text, small objects, or particular rendering styles? Those errors should drive component choices in the next section.

## 4. Model design · 15–20 minutes

### Spoken interview answer

> “I would add model complexity only for measured failure modes. If embedded text changes the meaning, I would extract it with OCR, encode it, and train a classifier on fused image and text features. If violations disappear after resizing, I would compare larger inputs and global-plus-local views. I would evaluate both the relevant error slices and the full pipeline, because an OCR stage or crop selector can miss evidence before the final classifier sees it.”

### Q10. How do you explain CNNs and self-attention at a technical level?

A CNN scans an image with shared local filters. A simplified convolution formula is:

$$
Y_{o,u,v}=b_o+\sum_{c,a,b}K_{o,c,a,b}\,X_{c,u+a,v+b}.
$$

The same kernel parameters are reused at different positions, making them useful for extracting local textures, edges, and shapes. Stacking layers expands the receptive field. Do not say that CNNs can only capture local information: deeper architectures can also incorporate information from a wider area.

Self-attention computes the following from the tokens:

$$
Q=TW_Q,\quad K=TW_K,\quad V=TW_V,
$$

$$
\operatorname{Attention}(T)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Each token can aggregate information from other tokens based on their content. For example, a symbol in one corner and a scene elsewhere in the image may jointly affect the decision. Positional encodings help the model distinguish token locations.

For global attention, the computation for pairwise token interactions is approximately $O(N^2d)$. This is not the total computational cost of a Transformer: linear projections and MLPs also contribute.

An interview explanation for your choice could be: “I would compare available pretrained CNN and ViT checkpoints at the required quality and latency, rather than choose only by architecture name.”

### Q11. What exactly is the difference between separate models and a joint model?

**Option A: Classify separately, then combine the scores.**

```text
image → image classifier → 5 image scores
image → OCR → text classifier → 5 text scores
                         ↓
               threshold / score fusion
```

This structure makes it easier to diagnose errors, upgrade components, and measure latency independently. It works well when either the image itself or the text itself violates a policy. However, if the two models each output only their own five violation scores, they may discard information needed to judge the relationship between the image and the text.

**Option B: Combine features, then make a joint decision.**

```text
image → image encoder → h_img [B, 768]
image → OCR → text encoder → h_txt [B, 384]
                          ↓
             concatenate → [B, 1152]
                          ↓
                   MLP → [B, 5]
```

For example:

$$
u=[h_{img};h_{txt}],\quad
r=\operatorname{ReLU}(W_1u+b_1),\quad
z=W_2r+b_2.
$$

With a hidden width of 256, the shapes are $W_1:[256,1152]$, $r:[B,256]$, and $W_2:[5,256]$. A nonlinear head can learn relationships between combinations of image and text features. If the concatenated vector feeds directly into a single linear layer, its score is still an additive combination of image and text terms, with more limited ability to model interactions.

For example, an ordinary character image with a neutral caption may be acceptable. The same image paired with text attacking that character's identity may change the overall meaning. The training data must include such pairs and similar nonviolating counterexamples. Simply connecting two encoders does not automatically teach the model the policy.

You could go further with cross-attention, allowing text tokens to interact with specific image patches. This is useful when a decision requires identifying exactly what the text refers to, but it makes training and inference more complex. First test whether simple fusion is sufficient.

For missing OCR output, distinguish “no text present” from “recognition failed.” You can provide a learned missing representation, a text-presence signal, and OCR quality signals, and cover these cases during training. When OCR misreads the critical text, the downstream text encoder cannot recover that information from nothing, so evaluation must cover the entire OCR pipeline.

A generative VLM is another, more expensive candidate, but it needs evaluation for false negatives, false positives, structured-output reliability, and latency. The ability to generate an explanation does not mean that the explanation is reliable. Whether it is worth adding still depends on its gains on examples requiring paired visual and textual context.

Useful ablations include image-only, text-only, score fusion, and feature fusion. Report results separately for purely visual cases, purely textual cases, and slices that require combining image and text.

### Q12. How do you decide whether to increase resolution or add localization?

Suppose the original image is 1000 pixels wide, and a policy-violating symbol is only 32 pixels wide. After resizing the image to 224 pixels, the symbol is approximately:

$$
32\times224/1000\approx7.2\text{ pixels}.
$$

If critical details have already disappeared, simply using a larger classification head cannot solve the problem. You can compare:

| Method | Benefit | Cost or risk |
| --- | --- | --- |
| Increase input size from 224 to 448 | Preserves more original detail | Higher computation, memory use, and latency |
| Full image + overlapping tiles | Preserves both global context and local detail | Multiple encoding passes; aggregation rules also need evaluation |
| Detector / region proposal + classifier | Focuses processing on relevant regions | Proposals may miss regions; suitable supervision is needed |

With a patch size of 16, increasing input size from 224 to 448 increases the patch count from 196 to 784, approximately 4 times as many. The pairwise interaction term in global attention grows by roughly 16 times. The latency of the whole model does not necessarily increase by exactly 16 times; measure it in practice. Also check how the checkpoint adapts its positional embeddings to the new input size.

If you use tiles, do not automatically copy the positive image-level label to every crop: an individual crop may contain no violating object at all. You can use region supervision with local annotations, or multiple-instance learning to train on a group of crops using image-level labels. Simple max pooling can express that “at least one crop is suspicious,” but it may also introduce more false positives as the number of crops increases. The final aggregated scores must be validated again.

The experiment should answer: “Does recall improve on the small-object slice? Is the overall system better under the same human-review budget? Does p95 latency still meet the requirement?” Do not look only at the detector's own accuracy.

## 5. Loss and training · 20–26 minutes

### Spoken interview answer

> “I would use masked binary cross-entropy from logits, because each policy is a separate binary target and some targets are unknown. I would first train the head, then compare partial fine-tuning with a smaller encoder learning rate. For rare violations, I would collect verified positives and hard negatives before trying positive weighting, resampling, or focal loss. I would choose these settings on development data, check calibration afterward, and only use augmentations that preserve the policy label.”

### 5.1 Why is BCE suitable for multi-label classification?

For a known binary label, with predicted score $p=\sigma(z)$, binary cross-entropy is:

$$
\ell(y,p)=-y\log p-(1-y)\log(1-p).
$$

When $y=1$, only $-\log p$ remains, encouraging the model to increase the violation score. When $y=0$, only $-\log(1-p)$ remains, encouraging it to lower the score. Compute this separately for all five policies, then aggregate.

Add a mask for known labels and optional positive-example weights:

$$
\mathcal L=-\frac{1}{\max(1,\sum_{i,k}m_{ik})}
\sum_{i,k}m_{ik}\left[
w_k^+y_{ik}\log p_{ik}+(1-y_{ik})\log(1-p_{ik})
\right].
$$

Start with $w_k^+=1$. This version averages over all known image-policy pairs. If review volume varies substantially across policies, commonly reviewed policies may receive more weight in training. You can compare averaging within each policy before applying policy weights, but explain that this changes the optimization objective; there is no single universally correct normalization.

**Numerical example:** For simplicity, consider just three policies:

```text
label:  [1, 0, ?]
mask:   [1, 1, 0]
score:  [0.8, 0.1, 0.9]
```

The unweighted loss is:

$$
\mathcal L=\frac{-\log0.8-\log0.9}{2}
\approx\frac{0.223+0.105}{2}=0.164.
$$

The third policy's score of 0.9 does not enter the loss because we do not know whether it is correct. If we incorrectly treated the unknown label as 0, we would add $-\log(1-0.9)=2.303$, heavily penalizing a high score that might actually be correct.

### Q13. How are logits, probabilities, and gradients connected?

A logit is an unrestricted real-valued score; sigmoid maps it to $(0,1)$:

$$
z=0\Rightarrow p=0.5,\qquad
z\approx2.197\Rightarrow p\approx0.9.
$$

For a single unweighted BCE term:

$$
\frac{\partial\ell}{\partial z}=p-y.
$$

For example, if the true label is a violation, $y=1$, but the model predicts only $p=0.2$, the gradient is $-0.8$. Gradient descent pushes the corresponding logit upward. If the true label is nonviolating, $y=0$, but the model predicts 0.9, the gradient is $+0.9$, pushing the logit downward. With masking, the direct loss gradient at an unknown position is 0. The shared encoder can still be updated through other known labels.

In practice, compute a numerically stable BCE directly from logits, instead of applying sigmoid and then manually taking logarithms. The unweighted form can be written as:

$$
\ell(z,y)=\max(z,0)-zy+\log(1+e^{-|z|}).
$$

This avoids directly computing $\log0$ at extreme scores. The [PyTorch BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) explains stable computation by combining sigmoid with BCE, as well as positive-example weighting for multi-label tasks.

Simplified pseudocode:

```python
# Use finite 0 placeholders for unknown positions in labels; known_mask determines their meaning.
loss_per_label = BCEWithLogits(logits, labels, reduction="none")
loss = (loss_per_label * known_mask).sum() / known_mask.sum().clamp_min(1)
```

If you represent unknowns with NaN, multiplying by 0 will not remove them, because NaN times 0 is still NaN. When an entire batch has no known labels, skip that optimizer update entirely, preventing weight decay from applying when there is no supervision signal.

### 5.2 What does the training process look like in practice?

First train the head and monitor per-policy loss and detection metrics on the training and development sets. Then compare unfreezing the final encoder layers. For example, you might initially try a learning rate of $10^{-3}$ for the head and $10^{-5}$ for the encoder. These are starting points for a search, not fixed answers.

A smaller encoder learning rate reduces the risk of damaging pretrained representations while adapting to the new task. If training performance is strong but development performance deteriorates, investigate overfitting, label noise, duplicate leakage, and differences between the splits, then adjust regularization, the number of trainable layers, or the data.

AdamW is one possible optimizer. Its update can be summarized as:

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

$\hat m_t$ and $\hat v_t$ are bias-corrected estimates of the first and second moments of the gradients. $\eta$ is the learning rate, and $\lambda$ is weight decay. The first term shrinks the parameters; the second adjusts the update scale using gradient history. Understanding it at this level is sufficient; you do not need to derive every state update in the main answer.

Checkpoint selection should also reflect the task objective, such as recall for severe policies at a fixed review rate, rather than only the lowest training loss. Keep the final test set isolated: do not use it to choose learning rates, models, or thresholds.

### Q14. How do you handle rare violations?

First determine whether the problem is “not enough positive examples” or “positive examples exist, but their training signal is overwhelmed by many easy negatives.”

1. Collect verified positives covering different styles, sizes, and borderline cases. Repeatedly copying a handful of positive images cannot replace real diversity.
2. Add reviewed hard negatives, such as normal game textures that resemble prohibited symbols, to help reduce false positives.
3. Compare moderate positive-example weighting or balanced sampling. In a multi-label setting, sampling positives for one rare policy also changes the distribution of other policies, so monitor all of them.
4. If easy examples do dominate training, then compare focal loss.

**Positive-example weighting changes the probabilistic meaning of the scores.** For a fixed $x$, let the true violation probability be $\eta=P(y=1\mid x)$, and let the positive weight be $w$. The ideal optimal score under weighted BCE is:

$$
q^*=\frac{w\eta}{w\eta+1-\eta}.
$$

For example, with $\eta=0.1$ and $w=9$, this gives $q^*=0.5$. It does not mean that the true violation probability has become 50%. After class weighting, validating or recalibrating scores on representative data becomes especially important. Do not assume a default threshold of 0.5 for the weighted outputs.

Focal loss is defined as:

$$
p_t=\begin{cases}p,&y=1,\\1-p,&y=0,\end{cases}
\qquad
\ell_{focal}=-\alpha_t(1-p_t)^\gamma\log p_t.
$$

When $\gamma=2$, a very easy example with $p_t=0.99$ has a modulation factor of $0.0001$, while a harder example with $p_t=0.6$ has a factor of $0.16$. The easy example's loss is therefore reduced much more strongly. This is the central motivation of the [original Focal Loss paper](https://arxiv.org/abs/1708.02002).

Focal loss does not create missing positive examples, and it is not equivalent to weighting only the minority class. It may place more emphasis on difficult or mislabeled examples, so first check label quality and then compare whether recall, false positives, and calibration actually improve.

### Q15. Why can augmentation make the labels incorrect?

Augmentation implicitly assumes that the label remains valid after the transformation:

$$
y(T(x))=y(x).
$$

However, the evidence for a moderation decision may be tiny. If a random crop removes the only violating region, the resulting image may be acceptable, yet retaining the positive label creates incorrect supervision.

Similarly, horizontal flipping may change the meaning of text or symbols, strong color perturbations may destroy critical visual cues, and blur may make text unreadable. First validate on actual examples that the transformations preserve the policy meaning, instead of directly adopting a standard augmentation recipe for general image classification.

If you have boxes or masks, update their annotations along with geometric transformations. If a crop removes all violating evidence, recompute the targets according to explicit local-label rules, or avoid that crop. Without local supervision, do not assume that you can accurately determine the label of every crop.

## 6. Evaluation · 26–31 minutes

### Spoken interview answer

> “I would separate ranking quality from operating-point quality. Per-policy average precision tells me how well the model ranks violations, while recall, precision, and review volume describe a chosen routing policy. I would calibrate on representative development data, select policy-specific thresholds under the shared review budget, and evaluate the locked pipeline on a later-time test. I would report severe categories and difficult slices separately, with uncertainty when positive counts are small.”

### 6.1 Four counts and the main metrics

First, count TP, FP, FN, and TN separately for each category:

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},\qquad
\operatorname{FPR}=\frac{FP}{FP+TN}.
$$

Precision answers, “Of the images we flagged, how many are actual violations?” Recall answers, “Of all actual violations, how many did we flag?” FPR answers, “Of all benign images, how many did we incorrectly flag?” Precision and FPR use different denominators and must not be confused.

### Q16. What is the difference between ranking quality and performance at a chosen threshold?

The model assigns a score to each image. Sorting the images from highest to lowest score and moving the threshold produces a precision–recall curve. A common definition of average precision is:

$$
AP=\sum_j(R_j-R_{j-1})P_j.
$$

It summarizes precision at different recall levels. AP is not necessarily identical to PR-AUC computed using trapezoidal integration, so state the implementation's definition when reporting it.

For example, image A is a violation with a score of 0.4, while image B is benign with a score of 0.3. The ranking is correct, but a fixed threshold of 0.5 allows both images, so A is still missed. Good ranking metrics therefore do not imply that the current threshold meets business requirements.

Report results for each of the five categories. Macro AP averages the categories' AP values, while micro metrics pool all known image-policy pairs and may be dominated by common categories. Severe, rare categories must also be reported explicitly rather than being hidden within an overall average.

**Why can a low FPR still produce many false positives?**

Let the true prevalence of a category be $\pi$:

$$
\operatorname{Precision}=
\frac{\pi\cdot\operatorname{Recall}}
{\pi\cdot\operatorname{Recall}+(1-\pi)\cdot\operatorname{FPR}}.
$$

The problem specifies 2,000,000 images per day. Suppose this category has a true prevalence of 0.1%, and a particular threshold gives 90% recall and a 1% FPR:

| Quantity | Calculation | Result |
| --- | --- | --- |
| Actual violations | $2{,}000{,}000\times0.001$ | 2,000 |
| TP | $2{,}000\times0.9$ | 1,800 |
| FN | $2{,}000-1{,}800$ | 200 |
| Benign images | $2{,}000{,}000-2{,}000$ | 1,998,000 |
| FP | $1{,}998{,}000\times0.01$ | 19,980 |
| All flagged images | $1{,}800+19{,}980$ | 21,780 |

$$
\operatorname{Precision}=1800/21780\approx8.26\%.
$$

Even if this were the only category being handled, sending every flagged image to human review would exceed the daily budget of 20,000 cases. A high ROC-AUC or an apparently low FPR is therefore insufficient to establish that the system is usable.

### Q17. What exactly is calibration, and how do you perform it?

If the model assigns a probability close to 0.8 for a category to 1000 comparable images, good calibration means roughly 800 of those images actually violate that policy. It does not guarantee that any particular image scored at 0.8 is correctly classified, nor does it mean that scores above 0.8 are automatically suitable for restriction.

Divide scores into bins and plot predicted probability against observed violation rate in a reliability plot. Also examine log loss or the Brier score:

$$
\operatorname{Brier}=\frac{1}{n}\sum_i(p_i-y_i)^2.
$$

Temperature scaling is a starting point to compare against alternatives. Fit $T>0$ on a separate calibration set:

$$
p^{cal}=\sigma(z/T).
$$

For example, $z=2$ originally produces 0.881. If the fitted temperature is $T=2$, the score becomes $\sigma(1)\approx0.731$. Within a category, a fixed positive temperature preserves the ranking and changes only the score scale. The [Calibration paper](https://proceedings.mlr.press/v70/guo17a.html) describes how modern neural networks can be miscalibrated and presents temperature scaling as a post-processing method.

Temperature scaling alone may not adequately correct shifts caused by sampling or positive-class weighting. You can compare it with sigmoid calibration that includes an intercept: $\sigma(a_kz_k+b_k)$. These are candidates to validate; no method is guaranteed to solve every calibration problem. For rare categories with little data, complex calibration methods can also overfit.

Calibration data must reflect the target traffic and must not be the final test set. Good overall calibration also does not guarantee good calibration for every new visual style or future time period, so slice checks and online audits remain necessary.

### Q18. With only 1% available for human review, how exactly do you choose thresholds?

**Step 1: Define the routing rules explicitly.** For each category $k$, set $t_k^{review}<t_k^{auto}$. For example:

```text
Any category meets the automatic restriction criteria → restrict
Otherwise, any category reaches its review threshold → human review
Otherwise → allow
```

This is one possible priority order. Automatic restriction requires rigorous precision validation and enough supporting samples; if the evidence is insufficient, you can initially leave it disabled. A low score for an ordinary category should not cancel out a severe category that already meets its restriction criteria. When policies conflict, encode the priority rules explicitly in routing.

For example, suppose one category has $t^{review}=0.2$ and $t^{auto}=0.98$. These numbers only illustrate routing: 0.10 is allowed, 0.60 goes to human review, and 0.995 meets that category's automatic restriction criteria. Actual thresholds come from a constrained search on development data, not from a probability simply appearing high.

**Step 2: Calculate the overall human review rate after deduplicating by image.**

Define $R_i=1$ to mean that image $i$ is ultimately routed to human review:

$$
R_i=\mathbf1\left[
\text{not auto-restricted and }\exists k:p_{ik}\ge t_k^{review}
\right],
$$

$$
\widehat{\operatorname{ReviewRate}}=\frac1n\sum_iR_i.
$$

An image that triggers three categories still counts as one review case. If each category has a 1% review rate, their combined review rate is not necessarily 1%; you cannot tune the five categories independently and immediately deploy them together.

The daily budget is:

$$
2{,}000{,}000\times1\%=20{,}000\text{ cases/day}.
$$

If those 20,000 cases include independent random audits, you could, for example, reserve 1,000 for auditing. That leaves a model-routing limit of 19,000 cases, rather than using all 20,000 and adding audits afterward. Count overlapping audit and existing review selections as one unique case. Clarify whether the stated budget also includes other manual work.

**Step 3: Search jointly over thresholds on the development set.**

Run the full routing logic for each candidate threshold combination, measuring recall for severe categories, the human review rate, automatic restriction precision, and the false restriction rate for benign images. First retain configurations that satisfy the constraints, then choose an operating point with fewer severe false negatives.

This can be summarized as:

$$
\min_{\mathbf t}\sum_k s_k\widehat{FN}_k(\mathbf t)
$$

$$
\text{subject to}\quad
\widehat{\operatorname{ReviewRate}}(\mathbf t)\le b,
\qquad
\widehat{\operatorname{FalseRestrictionRate}}(\mathbf t)\le\epsilon.
$$

$s_k$ represents a severity weight; $b$ is the budget remaining after other review work. If necessary, also impose a minimum recall for specific severe categories so the aggregate objective does not sacrifice one category. Define false restriction explicitly here: does it mean restricting an image that is benign under all five policies, or something else?

In offline evaluation, a violation sent to human review can initially count as “captured by routing,” but this does not guarantee that it will ultimately be blocked. Final online outcomes also depend on reviewer errors, review delays, and whether content is visible while waiting. Report flagging or routing recall separately from final enforcement outcomes.

If no threshold combination meets all targets, state that the targets are infeasible with the current model and resources. Improving ranking, collecting more labels, increasing review efficiency, or renegotiating constraints may change feasibility. Simply “tuning the thresholds well” cannot remove this conflict.

### 6.2 How do you report trustworthy results?

Report results for the main traffic population, and separately report slices such as rare categories, new creators, new visual styles, small objects, low resolution, and OCR failures. Include sample counts.

If a category has only 20 actual positives, missing one changes recall by 5 percentage points. A simple approximation is:

$$
\operatorname{SE}(\hat r)\approx\sqrt{\frac{\hat r(1-\hat r)}{n_+}}.
$$

With small samples or estimates near 0 or 1, use a more appropriate binomial confidence interval instead of treating this normal approximation as exact. Bootstrapping should also account for duplicate groups, creators, or temporal correlation; highly correlated image copies must not be treated as independent evidence.

Finally, lock the model, preprocessing, calibration, taxonomy, and thresholds, and evaluate them together on the final test set. Targeted challenge sets help identify capability limits, but cannot replace representative tests for estimating online precision and workload.

## 7. Serving and trade-offs · 31–35 minutes

### Spoken interview answer

> “For a new upload, I would decode and preprocess the image, run the encoder and policy head, apply calibration, and execute the versioned routing policy. I would measure the entire automated first-decision path against the 300-millisecond p95 budget, including queueing and optional OCR. I would monitor quality through independent random audits, because selected reviews alone cannot reveal all missed violations. Deployment would be gradual, with versioned artifacts and rollback.”

### 7.1 How does an image move through the production system?

```text
upload
  → validate format, decode, handle orientation and color
  → resize / pad / normalize
  → encoder + head
  → if required for this path: OCR / text fusion / stronger model
  → calibration
  → routing
  → return allow / restrict / pending review
  → log scores, actions, versions, and latency
```

A decoding failure must not default to benign. Depending on the product contract, explicitly return a processing failure, route to review, or allow the user to upload again. A timeout in the stronger model also needs a defined fallback action; failure must not silently be interpreted as low risk.

Here, 300 ms refers to the initial automated decision; final human review cannot fit within the same latency budget. If the slow path cannot finish within that budget, you can first return pending and evaluate asynchronously. However, confirm with the product team whether the content is visible in that state and when it can be published. Being placed in a background queue must not be described as having received a final moderation result.

### Q19. Why p95, and how do you break down latency?

Define latency for the complete automated path:

$$
L=L_{queue}+L_{decode}+L_{preprocess}+L_{model}
+L_{optional}+L_{routing}.
$$

An initial target breakdown for a single request could be: 20 ms for queueing, 30 ms for decoding and preprocessing, 80 ms for the model, 100 ms for optional OCR/fusion, and 10 ms for routing and returning the result, totaling 240 ms. This is only an illustrative budget and must be validated with actual load tests.

**You cannot add independently measured stage-level p95 latencies and claim that the sum is the overall p95**, because quantiles are generally not additive. Record the total latency for each real request, then compute the quantiles.

A p95 of 300 ms means roughly 95% of requests finish within 300 ms; it does not guarantee that the remaining 5% do. Also examine p99, timeout rates, and latency by route so the overall percentile does not obscure slow-path behavior.

The average throughput specified in the problem is:

$$
2{,}000{,}000/86{,}400\approx23.1\text{ uploads/s}.
$$

This number does not represent peak traffic and cannot directly determine how many GPUs are needed. Measure peak traffic, model throughput, batching effects, and hardware utilization. Larger batches may improve throughput, but waiting for a batch increases per-request latency, so the waiting time must be bounded.

### Q20. Why can high-confidence benign predictions still miss violations?

New visual styles, occlusion, compression, distorted text, or changes in the input distribution can all make the model highly confident in an incorrect prediction. Confidence alone cannot replace checking against true labels; uncertainty sampling also misses examples on which the model is “very confident but wrong.”

Therefore, do all of the following:

- Review flagged images to measure precision and review queue quality.
- Independently audit a random sample of allowed images to estimate violations that never entered the queue.
- Review appeals to discover false restrictions, without treating appeal samples as a direct estimate of the overall false positive rate.
- Track metrics and their uncertainty by policy, time, and visual style.

Suppose $A$ images are allowed in one day. A uniform random sample of $n_a$ allowed images contains $v$ violations of a particular category. A rough estimate of the number of missed violations in the allowed population is:

$$
\widehat{\operatorname{MissedCount}}=A\frac{v}{n_a}.
$$

However, auditing only allowed content does not directly yield overall recall. You also need to estimate all actual violations, including those in restricted and review-routed content. For stratified audits, weight observations according to the sampling design; do not simply pool samples with different selection probabilities.

For rare violations, a random sample may contain zero positives. This does not prove that the miss rate is zero. Report the sample size and a statistical upper bound, and complement the audit with labeled targeted tests.

**How do you calculate recall for a cascade?**

If all positives must first pass stage 1 and then be classified as positive by stage 2:

$$
\operatorname{Recall}_{cascade}=
P(\text{pass stage 1}\mid y=1)
\cdot P(\text{caught by stage 2}\mid\text{pass stage 1},y=1).
$$

For example, if stage 1 forwards 90% of actual violations and stage 2 catches 95% of those forwarded samples, the full chain has only $0.9\times0.95=85.5\%$ recall. Regardless of its quality, stage 2 cannot see violations that stage 1 allowed. This multiplication follows from conditional probability and does not require the stages to be independent. If other branches automatically block content, recalculate recall using all actual branches.

### Q21. How do you prevent training-serving skew?

Treat the following as one deployment version:

```text
decoder / preprocessing configuration
encoder + head weights
policy taxonomy + policy version
calibration parameters
routing thresholds + priority rules
optional OCR / fusion model versions
```

For example, training with RGB but mistakenly serving with BGR can change predictions. So can preserving the aspect ratio with padding during training while directly stretching images in serving. Use fixed images to check that preprocessing tensors and model outputs match across both environments within an allowed tolerance.

Log the versions used for each decision, deploy gradually, and retain rollback capability. A new model can first score traffic in shadow mode while you examine labeled evaluation results and capacity implications. Shadow scoring alone reveals score changes; assessing actual quality still requires labels. Then roll out to a small share of traffic and check false restrictions, audit results for missed violations, queue behavior, and latency.

Monitoring can be divided into three categories: system metrics such as p95, timeouts, and decoding failures; distribution metrics such as changes in scores and OCR failure rates; and labeled quality metrics such as missed violations and false restrictions for each category. A change in the score distribution is a signal to investigate, not automatic proof that model quality has deteriorated.

If you add embedding-based retrieval of similar assets, the encoder, preprocessing, and index must use compatible versions. Do not combine query embeddings from a new encoder with an incompatible old index.

Repeatedly retraining only on review data selected by the current model reinforces that model's selection bias. Independent random audits must therefore continue.

## 8. Three optional constraint changes

### 8.1 A new severe policy with only 0.1% prevalence

> “I would add the new target with its own annotation contract and known-label mask. I would collect verified positives for learning and a representative sample for estimating production precision and workload. I would not treat historical unreviewed examples as negatives. I would report positive counts and uncertainty, choose a severity-aware operating point, and check whether recall and review constraints are jointly feasible.”

The model output can expand from $[B,5]$ to $[B,6]$, but extending the head is the easiest part. Historical images were not reviewed against this new rule, so their new mask entry should be 0, rather than assigning them negative labels. Review appropriate historical images and new traffic, and clarify how the new policy overlaps with existing policies.

For learning, collect diverse verified positives and benign counterexamples that resemble violations of the new rule. If joint training affects existing categories, evaluate every category and retain suitable historical data. Also check whether the new rule requires additional context.

For evaluation, 0.1% prevalence means a random sample of 10,000 images contains only 10 positives on average. Assuming independent samples:

$$
P(\text{zero positives})=(1-0.001)^n.
$$

It takes $n\approx3000$ to have roughly a 95% chance of seeing at least one positive, but one positive is far too little to evaluate recall. A sample of 100,000 images contains about 100 positives on average. Even if estimated recall is 90%, its approximate standard error is still:

$$
\sqrt{0.9\times0.1/100}=0.03.
$$

Avoid reporting precise recall estimates without enough positive examples. A positive-enriched set can measure performance on its particular positive distribution, while representative audits estimate production prevalence, precision, and review volume. If the enriched set does not represent all types of violations, its recall cannot be directly extrapolated either.

This severe policy may require higher recall, but its review cases must still be counted jointly with the existing five categories. Start with review-only routing if needed, before enabling automatic restrictions. If the requirements are infeasible with current resources, explain the conflict explicitly.

### 8.2 Violations exist only in the image–text combination

> “I would train on paired image–text examples whose labels reflect the combined meaning. I would compare feature fusion with separate classifiers, include counterexamples where only the image or only the text changes, and evaluate the OCR-plus-fusion pipeline. If pooled features cannot represent the relevant interaction, I would compare token-level cross-attention or a suitable multimodal model.”

Supervision matters as much as architecture. Include paired examples such as:

```text
Image A + Text 1 → benign
Image A + Text 2 → violation
Image B + Text 2 → benign or a different meaning
```

Review each pair according to its overall meaning, rather than mechanically assigning the outcome shown here. Such counterexamples help the model learn image–text interactions instead of shortcuts such as flagging every occurrence of a particular word.

Add OCR, a text encoder, and a nonlinear fusion head to the image-only baseline. Measure the slice that requires combined context, then compare whether cross-attention or a larger model is worth the additional cost. Also inspect missing OCR text, incorrect image–text associations, and inference cost.

### 8.3 Sixty-second videos with a violation lasting only 0.5 seconds

> “The first issue is temporal coverage. A classifier cannot detect an event that no sampled frame contains. I would choose a sampling or windowing strategy based on the shortest event we need to catch, then add temporal modeling where motion or ordering matters. I would evaluate event-level recall and the entire sampling-plus-classification pipeline, not only accuracy on selected frames.”

Suppose we sample one frame per second, the event lasts 0.5 seconds, and its start has a uniformly random phase relative to the sampling grid. Under these idealized assumptions, about half of the events may fall entirely between sampled frames. This number depends on the sampling assumptions; it is not a fixed miss rate for all real videos.

Discuss coverage first. A regular sampling interval shorter than the shortest event can cover at least one instant under ideal visibility conditions, but one sampled frame may not be sufficient for classification. For example, 4 fps is a candidate to validate: it converts a 60-second video into $T=240$ frames. It does not automatically satisfy every short-event requirement.

The tensor shapes can be explained as:

$$
X:[B,T,3,224,224]
\longrightarrow H:[B,T,768]
\longrightarrow z:[B,5].
$$

The frame encoder can process the $B\times T$ images in batches, then restore the time dimension. If a violation only requires a particular pattern to appear in any frame, a frame classifier with aggregation is a useful baseline. If meaning depends on motion or ordering, add a temporal convolution, temporal attention, or a clip encoder.

Average pooling may dilute a short event: 0.5 seconds represents only $0.5/60\approx0.83\%$ of a 60-second video. Max pooling is more suited to presence detection, but more frames also create more opportunities for a high-scoring false alarm. Reevaluate thresholds at the video level.

If the product needs the event's location in time, collect event-interval labels and report event recall, short-event recall, and localization error. With only video-level labels, weakly supervised aggregation is possible, but it does not establish accurate temporal localization.

Do not distribute adjacent frames of the same video across training and test sets. Group videos and their near-duplicates, while also considering creator and time splits. Do not select frames solely by motion intensity: harmful text or imagery can remain static.

The original 300 ms budget for a single image does not transfer automatically. Redefine the time allowed for an initial video decision, available compute, whether to return a pending status first, and the required video-processing throughput.

## 9. A complete spoken answer connecting the seven stages

Use this passage to practice the full argument. Each sentence can be expanded using the relevant section above when the interviewer asks a follow-up.

> “I would define one example as one image at upload time, and clarify the five policy definitions, required context, and enforcement actions. Because policies can coexist, I would predict five sigmoid scores and separate prediction from routing. Severe false negatives are especially costly, but false positives also matter because they consume review capacity or incorrectly restrict creators.
>
> For data, I would combine reviewed candidates with randomly sampled uploads. Each image would have labels and a known-label mask, so unreviewed policies do not become fabricated negative labels. I would establish annotation guidelines, adjudicate disagreements, group duplicate assets before splitting, and evaluate later-time traffic and unseen creators separately.
>
> My baseline would freeze a pretrained image encoder and train a five-output linear head with masked binary cross-entropy. For a batch of eight images, the input is eight by three by 224 by 224, and the output is eight by five logits. I would then compare partial fine-tuning and collect verified rare positives and hard negatives.
>
> Model changes would follow error analysis. Text-dependent errors would motivate OCR and image–text fusion; small-object errors would motivate larger inputs or global-plus-local views. I would measure the full pipeline, including evidence lost by OCR or cropping.
>
> For evaluation, I would report per-policy average precision and operating-point recall, precision, and review volume. I would calibrate and jointly select policy-specific routing thresholds on development data. The one-percent budget means at most twenty thousand review cases per day, including any audits counted in that budget. If no operating point meets the constraints, I would identify that gap explicitly.
>
> In production, I would measure end-to-end automated p95 latency, version preprocessing, models, calibration and thresholds together, and deploy gradually. Independent random audits of approved content would estimate missed violations, while reviewed cases and verified appeals would support later improvements.”

## 10. Check your answers against the prompt

| Stage | What you should be able to explain | Follow-ups |
| --- | --- | --- |
| Problem formulation | Prediction unit; coexisting policies; scores versus actions; FN/FP costs | Q1–Q3 |
| Data and labels | Labels and masks; source bias; adjudication; duplicate, time, and creator splits | Q4–Q6 |
| Baseline | Pretrained encoder and head; freezing; input-to-output shapes; the accuracy trap | Q7–Q9 |
| Model design | CNNs and attention; image–text fusion; evidence for resolution and localization changes | Q10–Q12 |
| Loss and training | Masked BCE; logit gradients; rare categories; label-preserving augmentation | Q13–Q15 |
| Evaluation | AP versus thresholds; calibration; low-prevalence false alarms; joint routing and capacity | Q16–Q18 |
| Serving | Full-path p95; audits of missed violations; cascade recall; training–serving consistency and versions | Q19–Q21 |

For each important choice, practice explaining **what I would do → why → an example or formula → the experiment that would justify changing it**. Establish the complete argument first, then expand the equations and component details when asked.

## 11. Source boundaries

This guide is a teaching expansion of the local mock. All spoken answers, numerical examples, and design choices are practice material. For the provenance and level boundaries of the original interview topic, see the [original solution's Source boundaries](01_visual_moderation_solution.md#source-boundaries). Technical references appear near the relevant methods; they do not establish Roblox production configurations or the frequency of particular interview questions.
