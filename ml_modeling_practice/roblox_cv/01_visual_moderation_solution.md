# Mock 1 — Visual Safety Moderation: Reference Answer

For detailed explanations, spoken interview answers, worked numerical examples, and equations for every stage and Q1–Q21, read the [detailed walkthrough](01_visual_moderation_detailed.md).

## Opening spoken answer

“I would first clarify the policies, whether categories can coexist, and whether the product needs image-level decisions or locations. I would start with a pretrained image encoder and a small classification head, then add text understanding only where measured errors justify it. The difficult parts are trustworthy labels, rare violations, and choosing actions under a finite review budget. I would use separate development and final-test data, evaluate the complete routing policy, and audit approved content as well as reviewed content. High confidence alone is not a safety guarantee.”

All workload numbers below belong to this mock, not to Roblox's actual infrastructure.

## High-level picture before components

An **encoder** converts pixels into a learned feature vector. A **head** maps that vector to policy scores; a separate decision rule maps scores to actions.

```text
TRAINING
Reviewed images + known/unknown policy labels
  → deduplicate and split → image preprocessing
  → pretrained image encoder → policy head
  → loss on known labels → update selected parameters
  → validation: choose model, calibration, thresholds
  → locked test: evaluate the complete policy

INFERENCE
New image → identical preprocessing → encoder → policy scores
  → approved / restricted / stronger model / human review

FEEDBACK
Random audits + reviewed cases + appeals
  → adjudicated labels → subsequent training/evaluation versions
```

The initial model handles still images. OCR, meaning optical character recognition, extracts embedded text for an optional later model; video requires an additional temporal design.

## 1. Problem formulation · 0–4 minutes

One example is an uploaded image at its first decision time. Its target is a five-element policy vector, not the later fact that someone reported or removed it. Multiple labels may be positive. Define “unknown” separately from “negative,” and distinguish model predictions from enforcement decisions.

Clarify policy definitions, required context, which actions are reversible, and whether severe categories need stricter recall. **Recall** is the fraction of actual violations caught; **precision** is the fraction of flagged examples that really violate policy. Image-level classification is the simplest scope when presence is sufficient.

**Q1. Two simultaneous violations.** Use five independent output probabilities, one per policy. A softmax makes mutually exclusive probabilities sum to one and would incorrectly force categories to compete. A shared encoder can still learn relationships between labels.

**Q2. Labels versus locations.** Image labels answer whether a violation is present. Bounding boxes identify objects, while segmentation masks identify their pixels. Add localization when needed for redaction, reviewer evidence, or a demonstrated small-object failure—not merely because a detector exists.

**Q3. Error costs.** A false negative misses a genuine violation; a false positive flags benign content. Costs depend on severity and action: missing a dangerous upload differs from requesting an unnecessary review, and automatic restriction has a higher false-positive cost than a review suggestion.

## 2. Data and labels · 4–10 minutes

Create clear annotation guidelines, examples of difficult boundaries, and adjudication for disagreements. Include low-resolution images, small relevant details, benign lookalikes, embedded text, and varied rendering styles. Treat appeals as potentially corrective evidence, not automatically correct labels.

Keep a **gold set**, meaning carefully reviewed reference examples, including a random sample of actual traffic. Reports are useful for discovering difficult positives but cannot estimate prevalence alone. Track sampling probabilities or evaluate on a representative sample rather than interpreting an enriched training set as production traffic.

Prevent near-identical assets from crossing splits using duplicate groups. Use a later-time final test, plus held-out creator/style slices; if assessing unseen creators, explicitly keep those creators out of training. Do not claim an ordinary random image split tests all these generalization goals. Freeze evaluation labels and policy versions.

**Q4. Missing labels.** Store a mask $m_{ik}$ indicating whether category $k$ was reviewed for image $i$. An unassessed category contributes no supervised loss, rather than becoming a fabricated negative. Selective missingness can still bias training, so obtain fully reviewed random samples.

**Q5. Report bias.** Popular assets and certain communities may receive more reports; invisible violations and unreported benign content are underrepresented. The model can learn reporting behavior instead of policy. Random audits provide coverage of content that neither users nor existing models selected.

**Q6. Duplicate leakage.** A model may recognize an asset, creator-specific style, or a slightly modified copy instead of generalizing. Group related assets before splitting and inspect performance on new creators separately. Creator grouping does not replace time and near-duplicate checks.

## 3. Baseline · 10–15 minutes

Start with a pretrained image encoder, freeze it, and train a linear five-policy head. For example, resize and pad to $224\times224$, preserving aspect ratio and recording how padding is handled. This is an economical reference experiment, not proof that this resolution preserves every harmful detail.

For a ViT with $16\times16$ patches and hidden width 768:

$$
X:[8,3,224,224]\to[8,196,768]
\to[8,197,768]\to h:[8,768]
\to z:[8,5]\to p:[8,5].
$$

The extra token is a learned summary token; another encoder might use pooling instead. With $W:[5,768]$, compute $z=hW^\top+b$ and $p_k=\sigma(z_k)=1/(1+e^{-z_k})$. Probabilities need not sum to one. Compare against a trivial prevalence predictor, and inspect representative failures before adding complexity.

**Q7. Pretraining and freezing.** Pretraining means parameters were learned from an earlier dataset or objective. Freezing prevents their updates; only the new head learns. Partial fine-tuning updates selected encoder layers, while full fine-tuning updates the whole encoder.

**Q8. Shapes.** Eight RGB images become $[8,3,H,W]$ inputs. The model produces $[8,5]$ logits and probabilities, with $[8,5]$ labels and known-label masks. Each row represents one image, not one category.

**Q9. Misleading benign predictions.** With rare violations, predicting every label negative can yield high accuracy while detecting nothing. Compare per-category positive detection and false positives, not just the fraction of all labels predicted correctly.

## 4. Model design · 15–20 minutes

Choose additions from failure analysis. If text determines meaning, extract it using OCR, encode it, and combine text and image features in a trained classifier. For example, concatenate image features $[B,768]$ and pooled text features $[B,384]$ into $[B,1152]$, then produce $[B,5]$ logits. Missing or failed OCR must have explicit handling. A generative VLM is a more expensive alternative for genuinely contextual cases, not a mandatory upgrade.

**Q10. Convolution versus attention.** Convolution shares local filters across positions, giving useful spatial assumptions and efficient feature maps. Self-attention learns content-dependent interactions between patches and can connect distant evidence. Compare pretrained options on domain quality, resolution, memory, and latency rather than assuming one architecture always wins.

**Q11. Separate versus joint models.** Separate image/text models are easier to profile, retrain, and diagnose, but combining only their independent violation scores may miss harmful interactions. A joint fusion model learns how text changes the image's meaning using paired, context-aware labels. Evaluate both unimodal and cross-modal cases to justify the extra compute.

**Q12. Resolution and localization.** Measure errors by relevant-object size and compare larger inputs, overlapping tiles, or a localized model on that slice. Test the whole pipeline because a crop selector can miss the critical region. Localization requires suitable labels and computation; higher resolution is not free improvement.

## 5. Loss and training · 20–26 minutes

Use masked binary cross-entropy for the five coexisting labels:

$$
\mathcal L=-\frac{1}{\max(1,\sum_{i,k}m_{ik})}
\sum_{i,k}m_{ik}\left[w_k^+y_{ik}\log p_{ik}
+(1-y_{ik})\log(1-p_{ik})\right].
$$

Start with $w_k^+=1$; validate positive weighting or balanced sampling if rare categories are poorly learned. Implement the loss from logits using a stable library operation, not an explicit logarithm of probabilities near zero.

Use AdamW as a practical starting optimizer: it adapts update scales and applies weight decay separately. Train the head first, then compare unfreezing later encoder layers with a smaller learning rate. Validation selects learning rate, regularization, and checkpoint; the test is not a tuning loop. Do not claim AdamW universally beats SGD.

**Q13. Logits and probabilities.** A logit is an unrestricted real score, while sigmoid maps it into $[0,1]$. The logits-based loss incorporates that mapping numerically stably; backpropagation computes how to adjust trainable weights to reduce the loss. Sigmoid outputs are not automatically calibrated probabilities.

**Q14. Rare violations.** Collect verified positives and difficult verified benign examples, called hard negatives. A randomly chosen unreviewed upload is not a confirmed negative. Weighting, resampling, or focal loss $-\alpha_t(1-p_t)^\gamma\log p_t$ can help when easy examples dominate; $p_t$ is the probability assigned to the true binary label, and $\gamma$ controls easy-example downweighting. Focal loss is not automatic: noisy labels and calibration can worsen.

**Q15. Invalid augmentation.** Removing the only violating object while keeping a positive target teaches a contradiction. Prefer transformations verified to preserve meaning; maintain text readability and relevant details. For crops, use localization-aware label updates or avoid destructive crops. Apply geometry consistently to any boxes or masks.

## 6. Evaluation · 26–31 minutes

Evaluate fully reviewed representative data and separately report rare policies, new styles, small details, and OCR failures. Report uncertainty when a slice contains few positives. Maintain a targeted challenge set, but do not substitute its artificial prevalence for traffic estimates.

**Q16. Ranking versus decisions.** Average precision, $AP=\sum_j(R_j-R_{j-1})P_j$, summarizes a precision–recall ranking curve for each policy. ROC-AUC measures positive-versus-negative ordering; it is useful, but a small false-positive rate can still generate many false alarms at low prevalence. Neither fixes a decision threshold. At selected thresholds report recall, precision, false-positive rate, and review volume; do not call AP and every numerical PR-curve integral identical.

**Q17. Calibration.** Among comparable examples assigned about 0.8 probability, about 80% should truly violate that policy. Ranking can be strong even when probabilities are systematically too high. Check reliability plots and log loss on representative labels; any calibration transformation must be learned without using the final test. Training reweighting makes this check especially important.

**Q18. Thresholds and capacity.** Choose class-specific thresholds using validation labels and an explicit routing policy. An optional low threshold sends cases to review and a higher, precision-validated threshold permits automatic restriction; unresolved conflicts need a defined action. Count each reviewed image once even if it triggers three policies. For this mock, 1% of 2 million uploads means at most 20,000 review cases daily.

Optimize the routing policy for severe-category recall subject to review capacity and false-restriction limits. Lower thresholds generally catch more positives but increase workload; if no operating point satisfies the constraints, the honest result is infeasibility, not a magically correct threshold. Improve the model, collect labels, adjust scope, or negotiate capacity. Lock model, calibration, and thresholds before final testing.

## 7. Serving and trade-offs · 31–35 minutes

Follow the inference flow above, including decoding, resizing, padding, normalization, accelerator execution, optional OCR, and routing. Measure the complete automated first-decision path against the mock 300 ms budget, excluding human review. Throughput and batching matter, but a batch wait can harm per-upload latency.

**Q19. Why p95?** p95 is the latency within which 95% of measured requests finish. A good average can conceal a slow tail; include preprocessing, waiting, and model execution. Test realistic image sizes and concurrent load, not just a warm isolated model call.

**Q20. Confidence and audits.** Distribution shift can produce confident errors. Audit randomly sampled approved images as well as selected reviews, track per-policy missed violations with labels, and inspect appeals and new styles. A stronger model never sees false negatives filtered out earlier, so evaluate recall across the complete cascade rather than only its escalated subset.

**Q21. Consistency and versions.** Package preprocessing, model weights, label taxonomy, calibration, and routing thresholds together; test training–serving agreement on fixed images. Record versions for decisions, deploy gradually, and keep rollback available. If a known-asset embedding lookup is added, version its index with its encoder rather than silently mixing incompatible vectors.

Monitor quality, review selection, score shifts, and tail latency. Reviewed cases alone produce a biased feedback loop: the current model influences which labels arrive. Preserve independent random audits in later evaluation and retraining.

## Optional constraint changes — answers

**1. New policy at 0.1% prevalence.** Obtain enough verified positive examples and a representative labeled sample to measure precision and workload with uncertainty. Report policy-specific precision–recall behavior and counts rather than celebrate accuracy; enrichment helps training but cannot directly estimate production precision. Check whether review capacity and required recall remain feasible.

**2. Meaning exists only in image–text interaction.** Train a joint image/OCR fusion model or a suitable VLM using labels that include paired context. Add counterexamples where the image or text alone appears similar but the joint meaning differs. Compare against separate-model fusion, and audit OCR errors, cost, and grounded decisions.

**3. A 0.5-second event in 60 seconds.** Sparse image sampling can miss the event before classification begins. Use appropriately dense coverage or overlapping windows, then a temporal model when ordering matters; motion-only sampling is insufficient for static harmful content. A tensor $[B,T,3,H,W]$ becomes $[B,T,d]$ frame features and clip/event outputs. Obtain temporal labels where localization is needed and evaluate short-event recall, not just sampled-frame accuracy. Any cheap first stage must be audited for missed events.

## Source boundaries

The local [harmful chat and video report](../../08_company_interviews.html#harmful-chat-and-video-content-detection) supports the broad topic and the separate-versus-multimodal discussion; it is labeled target L6, not a verified new-graduate question. Its reference answer is authored study material, not a reporter's transcript. Upload-only scope, mock numbers, timing, follow-ups, and this solution are adaptations informed by the [CV moderation case](../../02_ml_domain.html#cv-case01) and [video case](../../02_ml_domain.html#cv-case03).

Method references: [ViT](https://arxiv.org/abs/2010.11929) for patch-based visual representations and [Focal Loss](https://arxiv.org/abs/1708.02002) for downweighting easy examples. These support modeling concepts, not interview provenance.
