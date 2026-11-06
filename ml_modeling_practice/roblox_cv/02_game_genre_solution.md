# Mock 2 — Game Genre Classification: Reference Answer

For detailed explanations, spoken interview answers, worked numerical examples, and equations for every stage and Q1–Q21, read the [detailed walkthrough](02_game_genre_detailed.md).

**One defensible worked design, not the only correct answer.** This is a modeling discussion, not a coding assessment. The assumptions and numerical examples below are hypothetical. Match the seven stages to the separate question document; use the core design as a 35-minute path and treat the follow-up answers as a reference bank.

## Opening: what I would say

> “I would clarify the genre taxonomy, whether multiple genres can coexist, and whether we need labels before a new game has interaction history. I’ll assume one prediction per published experience version, with multiple genres allowed. I’d establish a reviewed dataset and a simple baseline, then test a pretrained visual-and-text model. I would use validation data to select thresholds, and publish versioned labels asynchronously so discovery requests do not have to run the encoders.”

## The complete picture first

Genre describes gameplay, not merely visual style. A screenshot may suggest an obstacle course but cannot prove every gameplay mechanic. Text adds complementary evidence but may be incomplete or misleading.

```text
TRAINING
Experience snapshots + labeling guide → reviewed / weak labels
       → group-aware split → text and image representations
       → train genre predictor → validation thresholds → versioned model

INFERENCE
Publish or update → snapshot images + text → same preprocessing
       → encoders → fusion → genre probabilities → thresholds / abstention
       → save labels with content, taxonomy and model versions
       → discovery and recommendation read the saved labels
```

“Fusion” means combining information from modalities, such as images and text. “Abstention” means returning an explicit uncertain result rather than forcing an unsupported genre. Model probabilities and the rules that turn them into product labels are separate components.

## 1. Problem formulation (0–4 min)

I assume a reviewed taxonomy of $C=20$ genres and allow combinations such as obstacle-course and roleplay. The target is therefore $y_i\in\{0,1\}^{20}$ for experience version $i$, not a single mutually exclusive class. Store unknown annotations separately from negative labels.

The product contract returns genre scores, selected labels and model/taxonomy versions. Games may receive no confident label. Product should define useful distinctions and the costs of incorrect inclusion versus omission; adding a misleading genre can harm a filtered search, while omitting a genre hides relevant content. Start with content available at publication, not future engagement.

**Q1. Is this multiclass or multi-label?** Multi-label if multiple genres can simultaneously be correct. Softmax describes one mutually exclusive outcome; independent sigmoid outputs describe several binary decisions. Do not choose based on whether labels are stored as strings or integers.

**Q2. Why one example per experience version?** The target describes the game at a specific content snapshot. Treating each screenshot as independent inflates the influence of image-rich games and allows related content to leak across splits. Images are multiple observations inside one example.

**Q3. Are creator tags the objective?** No. They are a potentially noisy source of supervision. Annotation agreement is an offline proxy; discovery success requires measuring whether the resulting labels help users find suitable games.

## 2. Data and labels (4–10 min)

Write a labeling guide with positive examples, boundary cases and explicit yes/no/unknown decisions per genre. Use the 5,000-review budget to cover common and rare genres, languages and rendering styles; independently label a subset to diagnose disagreement. Agree how to adjudicate ambiguity rather than labeling it arbitrarily.

Creator tags and text-based rules can provide **weak labels**—cheap labels that may be wrong or incomplete. Keep them distinguishable from reviewed labels. Split by related game versions and near-duplicate content before using weak examples, then hold out creator groups and a later publication period for the generalization questions that matter. A random traffic audit estimates population quality; a deliberately rare-genre-enriched set diagnoses minority performance but has different prevalence.

At inference time some modalities may be missing. Preserve missingness indicators. For newly published games, no future play counts or retention labels exist. Existing-game behavior can be a later, separately evaluated model extension, not a hidden dependency of the cold-start model.

**Q4. Weak versus gold labels?** Weak labels come from unreliable or indirect rules, tags or automated teachers. Gold labels follow a reviewed annotation protocol; they can still disagree or contain errors. Evaluate on independently reviewed data rather than validating the model against its own pseudo-labels.

**Q5. Why not split screenshots randomly?** Different screenshots or versions of one game share visual assets, text and genre. The model may recognize the same game instead of generalizing. Group related content, and explicitly separate creator-held-out and future-time evaluation if both claims are needed.

**Q6. Is an absent creator tag a negative?** Not automatically. It may simply be unannotated. Track a known-label mask, obtain explicit negatives in reviewed examples, and avoid treating every unmentioned genre as false.

## 3. Baseline (10–15 min)

First compare a prevalence-based constant predictor with TF-IDF text features and one regularized binary classifier per genre, trained on reviewed labels. TF-IDF weights words according to their frequency in a document and rarity across the corpus. This is inexpensive and reveals how informative titles and descriptions are.

Then use frozen pretrained image features with a small genre head. Compare text-only, image-only and combined models on the same held-out data. CLIP-style zero-shot prompts can be another diagnostic baseline, but a domain genre may depend on mechanics that its prompts or visual representation do not capture. “From scratch” means building the data and service in this mock, not assuming we can afford representation pretraining.

**Q7. Why a baseline?** It establishes whether a complex system adds value at a meaningful cost. If a simple text model is strong, quantify which errors require vision instead of assuming a larger model is necessary.

**Q8. Must weights start randomly?** No. Clarify the phrase with the interviewer. Reusing pretrained representations is compatible with building a new dataset, objective, inference path and evaluation from scratch. Training an encoder from random initialization needs a separate data/compute justification.

**Q9. Can we use future play time?** Not at publication. A useful predictor must exist at the decision time. For an updated game with historical behavior, timestamp the feature snapshot and evaluate that different availability contract separately.

## 4. Model design (15–20 min)

For a concrete candidate, resize images to $224\times224$, preserve appropriate aspect-ratio handling, and use a pretrained ViT with $16\times16$ patches. Each image produces 196 patch tokens plus a CLS token; the CLS representation summarizes that image. With at most $V=4$ valid images:

$$
I\in\mathbb R^{B\times V\times3\times224\times224}
\rightarrow H_I\in\mathbb R^{B\times V\times768}
\rightarrow h_I\in\mathbb R^{B\times768}.
$$

Use a masked mean across valid images initially, so padding does not count as an image. Compare learned attention pooling if different views carry different evidence. Text goes through a pretrained multilingual encoder and pooling:

$$
T\in\mathbb N^{B\times L}
\rightarrow H_T\in\mathbb R^{B\times L\times384}
\rightarrow h_T\in\mathbb R^{B\times384}.
$$

Project each pooled vector to 256 dimensions. Concatenate them with two modality-presence flags and apply a small MLP:

$$
u=[P_Ih_I;P_Th_T;m_I;m_T]\in\mathbb R^{B\times514},
\qquad u\rightarrow\mathrm{MLP}\rightarrow z\in\mathbb R^{B\times20},
\qquad p=\sigma(z).
$$

Missing modality vectors use an explicit default and presence flag. A projection matches dimensions, while genre supervision teaches useful fusion; dimensional compatibility alone is not semantic alignment. These dimensions are an illustrative architecture, not a claim that this exact model is optimal.

**Q10. CNN or ViT?** Compare transferred task quality, available pretraining, input detail, memory and measured latency. CNN locality can help with limited data; pretrained ViTs offer flexible global patch interactions. A bigger architecture is not automatically better on 5,000 reviewed examples.

**Q11. How combine screenshots?** Pool their representations within the experience example, masking missing images. Mean pooling is a baseline; attention pooling can learn view importance. Evaluate whether additional views help and whether irrelevant screenshots dominate.

**Q12. Early or late fusion?** Early fusion allows cross-modal interactions before final prediction; late fusion combines separate modality predictions. This design combines encoded features before the final head, an intermediate feature-fusion approach. Compare it with score-level late fusion, and train with realistic missing modalities so a missing description is not a surprise.

## 5. Loss and training (20–26 min)

Use independent binary cross-entropy for each known genre label. Let $m_{ic}=1$ when genre $c$ is annotated for sample $i$; otherwise it is zero:

$$
\mathcal L=-\frac{\sum_{i,c}m_{ic}w_c\left[y_{ic}\log p_{ic}+(1-y_{ic})\log(1-p_{ic})\right]}
{\sum_{i,c}m_{ic}w_c}.
$$

Skip a batch with no annotated labels rather than dividing by zero. Implement the loss from logits for numerical stability. Class weights are a candidate to validate, not a guarantee of calibrated probabilities. Initially train the projections and fusion head while freezing encoders. Then compare unfreezing upper layers or lightweight adapters using lower learning rates and the same validation protocol.

AdamW is a reasonable optimizer for this pretrained-transformer candidate because it uses adaptive gradient updates with decoupled weight decay. Tune learning rate, decay and training duration on validation; keep the gold test set untouched. A train loss that falls while validation worsens suggests overfitting or distribution mismatch. Investigate sample coverage before merely adding epochs.

Use gentle color or geometric changes only when genre semantics remain intact. Aggressive crops may remove the gameplay evidence. Compare balanced sampling, class weights and more targeted annotation for rare genres. Avoid heavy users or prolific creators dominating batches; ordinary genre prediction does not require contrastive negative sampling.

**Q13. Why sigmoid, and what does the mask do?** Genres can coexist, so their probabilities do not need to sum to one. For BCE, the logit gradient is proportional to $p-y$; multiplying by $m=0$ removes the unknown label's contribution rather than pushing it toward negative. A softmax would incorrectly force genres to compete.

**Q14. Why freeze, then adapt?** Freezing reduces trainable parameters and compute when labels are scarce. Broader adaptation is justified if error analysis shows domain features are inadequate. Compare it with more data or a better head, monitoring overfitting and forgetting rather than assuming full fine-tuning always helps.

**Q15. What about rare genres and augmentation?** Check label quality and coverage first, then test sampling or weighting with per-class metrics. Focal loss can reduce easy-example dominance but is not automatic. An augmentation is valid only if the genre label still describes the transformed input; removing all relevant evidence creates misleading supervision.

## 6. Offline and online evaluation (26–31 min)

On reviewed held-out examples, report per-genre precision and recall, macro/micro-F1, and mean AP across genres with enough known positives and negatives. State the evaluation mask and class coverage; an absent class is not evidence of perfect performance. Use a deployment-representative audit set for population estimates and targeted slices for failure discovery.

Choose class-specific thresholds on validation according to filter precision and genre coverage. Evaluate text-only, image-only and fused ablations, missing-modality cases, unfamiliar creators/styles, and head versus tail genres. Inspect misleading title/image disagreements, not only aggregate scores.

After human review and shadow deployment, test whether using the new labels improves genre-filter success or recommendation quality. Keep other ranking changes fixed where possible, define a primary metric, and monitor wrong-genre complaints, safety and latency guardrails. Better classification is not automatically more engagement, and engagement is not the sole definition of correct genre.

**Q16. Macro-F1, micro-F1 and mAP?** Macro-F1 averages per-genre F1 equally; micro-F1 pools the label-level true/false positive/negative counts and emphasizes common decisions. Here mAP means the mean of each genre's binary average precision over experiences, not object-detection mAP. AP uses scores; F1 uses thresholded labels.

**Q17. Why not threshold 0.5 everywhere?** Genres differ in prevalence, score calibration and error costs. Select thresholds on validation, with adequate per-class support. Thresholds chosen on test would make the final report optimistic; a validation target also need not hold exactly on future data.

**Q18. Why can offline improvement fail online?** Genre accuracy is a proxy. Ranking may ignore the feature, existing labels may already be sufficient, or added labels may change exposure in an unhelpful way. Measure the effect of actually using the new labels through a controlled rollout rather than inferring causal lift from offline metrics.

## 7. Serving and trade-offs (31–35 min)

On publish or content update, snapshot the experience version, encode its available content, predict and save labels within the five-minute mock freshness budget. Discovery requests read stored labels instead of running a visual encoder. Cache representations only by immutable content and encoder version; changing the model can invalidate cached features.

Bound batches and measure queue age, throughput and freshness as well as model p95 latency. If prediction fails, keep an explicitly stale prior result where policy permits, or an unknown state; never label a new version confidently using unrelated content. Version the taxonomy, encoders, head and thresholds together. Monitor missing modalities, score/label distributions and audited errors, with rollback and targeted relabeling when styles change.

**Q19. Publishing versus request-time work?** Encoding is asynchronous at publication/update, while a discovery request reads already computed labels. This trades instantaneous updates for low request latency. State the freshness bound and invalidation behavior rather than assuming saved features stay correct forever.

**Q20. Caching and quantization risks?** Caching saves repeated encoder work but stale or mismatched features can silently alter predictions. Quantization reduces numerical precision and may improve memory/latency only with runtime support. Benchmark actual hardware and recheck rare-genre quality, calibration and thresholds after changing precision.

**Q21. New genre or new style?** A taxonomy change requires new label definitions, reviewed examples and versioned outputs; old unannotated labels are not negatives. A new visual style calls for audited slice evaluation, targeted data collection and adaptation. Changing thresholds alone cannot create a missing representation or label definition.

## Optional interviewer changes: reference responses

**E1. Untrustworthy descriptions.** Audit source quality, compare image-only and text-only errors, train on realistic missing or corrupted text, and use modality gating or lower-trust text features. Ground labels in actual game evidence, not copied keywords. No fusion model can reliably recover mechanics that are absent from every available input; request gameplay evidence or abstain.

**E2. Only 50 rare-genre examples.** First check definition consistency, inter-annotator disagreement and the number of positives in each split. Report uncertainty, inspect confusion with adjacent genres, and gather targeted reviewed examples. Compare a frozen encoder, weights/sampling and a simple head before increasing model capacity; do not promise stable recall from a tiny test slice.

**E3. Explanations and fivefold lower cost.** Return per-modality evidence and supporting screenshots or text spans for human audit, but do not describe attention maps or generated rationales as causal proof. Profile encoder, text and fusion costs; cache by content version, reduce views, distill or use supported quantization, and validate quality after each change. If the required quality and cost cannot both be achieved, quantify the conflict rather than promising an unsupported fivefold gain.

## Closing answer

> “I would build reviewed, versioned multi-label genre data and start with simple text and visual baselines. My candidate uses pretrained encoders, masked view pooling and a small fusion head, trained only on known labels. I would choose thresholds on validation and audit unseen styles and rare genres before using the labels downstream. The key trade-off is richer content understanding versus annotation and encoder cost; the next experiment is a controlled text-only, image-only and fusion comparison.”

## Source boundary and related notes

- The local collection reports [game genre classification from scratch](../../08_company_interviews.html#game-genre-classification-from-scratch), level unstated, linked to [1173576](https://www.1point3acres.com/bbs/thread-1173576-1-1.html). It does not report the dataset, label budget, timing breakdown, architecture or answers above.
- Connect to [transfer learning](../../02_ml_domain.html#cv-q09), [classification tasks](../../02_ml_domain.html#cv-q10), [class imbalance](../../02_ml_domain.html#cv-q20), and [multimodal fusion](../../02_ml_domain.html#cv-q39).
- Method references: [ViT](https://arxiv.org/abs/2010.11929) for patch-based visual encoding and [CLIP](https://arxiv.org/abs/2103.00020) for transferable image-text representations. These describe model families, not interview evidence or a universal architecture winner.
