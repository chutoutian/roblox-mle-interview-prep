# Mock 3 — Text and Image Search for 3D Assets: Reference Answer

For detailed explanations, spoken interview answers, worked numerical examples, and equations for every stage and Q1–Q21, read the [detailed walkthrough](03_multimodal_asset_retrieval_detailed.md).

This is an original adaptation, not a reported Roblox asset-retrieval interview question. All numerical constraints are illustrative. The proposed design is a starting hypothesis to validate, not a claim about Roblox's production architecture.

## Opening: a 30-second spoken answer

“I would first define relevance: do we want the same asset, a visually similar object, or a substitute satisfying the creator's description? I would represent assets using rendered views, establish a pretrained text/image retrieval baseline, and adapt it using reviewed relevance pairs. I would retrieve candidates cheaply, optionally rerank a shortlist, and evaluate both relevance and latency. New assets should be searchable from their content without needing clicks.”

## Whole-system picture before details

Training learns compatible representations; serving uses them to search the catalog.

```text
TRAINING
Assets → rendered views → asset encoder ──┐
Queries ────────────────→ query encoder ──┴→ batch matching scores ──┐
Reviewed query–asset relevance labels ─────────────────────────────┤
                                                    training loss → updates

CATALOG PREPARATION
Eligible asset version → rendered views → asset vector → search index

SERVING
Text or image query → query vector → candidate search → optional reranker
                                   → eligibility / duplicate checks → 20 results
```

**Exact nearest-neighbor search** scores every stored vector. **Approximate nearest-neighbor search (ANN)** searches a structured subset to find nearly the same high-scoring vectors more cheaply; it can miss candidates. That approximation is separate from whether the model's similarity reflects human relevance.

## 1. Problem formulation — minutes 0–4

One prediction is a relevance score for a query–asset pair, used to order eligible assets. Define graded relevance with examples: an exact match, a useful substitute, a partial match, and an irrelevant result. A query is not necessarily asking for the most popular asset or longest engagement; optimizing clicks alone can favor attractive thumbnails rather than usable assets.

**Q1. Exact duplicate versus semantic substitute.** Exact matching can use hashes or geometry-specific checks. Semantic search must allow multiple shapes and creators when assets satisfy the requested function, appearance, or attributes. Keep these tasks distinguishable in evaluation.

**Q2. Multiple attributes or objects.** “Red wooden chair” requires composition, not three independent keyword matches. For an image with several objects, clarify whether the user selects a crop or expects scene-wide similarity; crop-based query understanding is a useful explicit interaction rather than guessing their intent.

**Q3. Requirements before representation.** Ask about supported languages, asset types, appearance versus geometry, eligibility filters, update delay, and the latency budget's boundary. Also establish whether 20 diverse substitutes are preferable to 20 near-duplicates. These choices determine labels and output quality.

## 2. Data and labels — minutes 4–10

Build examples containing a query, candidate asset/version, query modality, event time, and relevance judgment. Combine manually reviewed pairs, reliable descriptions paired with rendered assets, and carefully interpreted behavioral feedback. Use graded judgments for final ranking and positive relevance pairs for initial representation learning. Include styles, rare categories, attribute combinations, difficult backgrounds, and rejected alternatives.

**Q4. Unclicked does not mean irrelevant.** Users only react to what the old system exposes, and low-ranked results may never be noticed. Even an examined, unclicked asset is an imperfect negative. Log exposure position and later qualified actions; use randomized exposure or appropriate propensity methods when feasible, and retain independently judged evaluation queries. Setting position to a constant is not a complete causal correction.

**Q5. Several valid assets.** Mark all known matches as positives, including equivalent variants where appropriate. A different catalog ID is not proof of irrelevance. Avoid treating another positive as a negative merely because it appears in a different training pair.

**Q6. Leakage and changing content.** Keep near-duplicate and copied asset families together when testing generalization, and add creator-held-out and future-time slices. Store rendering and metadata versions; a historical example must use information available then, not its later improved description. Do not let a future version of an evaluation asset leak into training. Separate these harder holdouts from ordinary known-catalog search evaluation, where the indexed assets are legitimately available.

## 3. Baseline — minutes 10–15

Start without custom training: a pretrained CLIP-style model embeds text and images into a common space. Text queries use its text encoder; reference images and asset renderings use its image encoder. Mean-pool several view features per asset, normalize, and initially search a small catalog exactly to establish model quality without index approximation. A lexical metadata baseline is also useful for text queries but cannot solve image queries alone.

**Q7. Raw meshes.** CLIP's image encoder accepts pixels, not vertices and faces. Render the mesh into images; a direct geometry encoder would require another model and suitable supervision. This baseline retrieves visual appearance, not guaranteed geometric compatibility.

**Q8. Rendering shortcuts.** Standardize useful camera coverage, scale, lighting, and backgrounds, then perturb them in evaluation. A model may match a shared background or miss the back of an object; high visual rendering quality does not establish coverage. Inspect occluded parts, thin structures, transparency, and orientation-sensitive queries.

**Q9. Evidence for adaptation.** Audit reviewed failures: unfamiliar art styles, tiny details, and color/material confusion suggest representation mismatch. Missing assets, stale eligibility, or invalid renderings require pipeline fixes first. Compare adaptation on held-out task relevance, not just falling training loss.

## 4. Model design — minutes 15–20

Use dual encoders so expensive asset encoding is reusable across queries. A CNN is a viable efficient visual backbone with local spatial priors; a ViT provides flexible patch interactions but depends on resolution and training. Choose measured retrieval quality and hardware cost, not the architecture name. Reuse a compatible pretrained text/image pair rather than start from random weights.

For $B$ assets and $V$ views:

$$
I:[B,V,3,H,W]\xrightarrow{f_I}F:[B,V,d_v]
\xrightarrow{\text{view pooling}}h:[B,d_v]
\xrightarrow{W_a}u:[B,d]
\xrightarrow{\text{L2 normalization}}a:[B,d].
$$

Text tokens $T:[B,L]$ pass through the text encoder, sequence pooling, projection, and normalization to give $q:[B,d]$. A reference image follows the visual path with one or several query crops. For matched text–asset batches, $S=qa^\top/\tau:[B,B]$ contains all query–asset scores; serving uses $q^\top a$ against catalog vectors.

**Q10. Combining views.** Mean pooling is simple but can dilute a distinctive side. Compare attention pooling or per-view retrieval with asset-level aggregation, accounting for extra storage and duplicate results. Test unseen views, not only the same render used for training.

**Q11. Why not a classifier?** A fixed class such as “chair” discards within-class attributes and cannot directly handle arbitrary queries. CLIP-style representations support cross-modal comparison and open-ended descriptions. A classifier may still help with category filters, but its class probability is not a complete relevance score.

**Q12. Rich interaction.** After fast vector retrieval, a cross-encoder can jointly attend to query text and candidate visual tokens, resolving attributes the dual encoder missed. It must process each pair, so apply it to a measured shortlist rather than two million assets. Keep it only if ranking gains justify latency and compute.

## 5. Loss and training — minutes 20–26

Initially freeze pretrained encoders and learn or adjust the pooling/projection components only if needed. Train on paired query–asset examples with symmetric contrastive cross-entropy, also called a CLIP-style InfoNCE objective. In the simple one-positive-per-row case:

$$
s_{ij}=q_i^\top a_j/\tau,\qquad
\mathcal L_{Q\to A}=-\frac1B\sum_i\log\frac{e^{s_{ii}}}{\sum_je^{s_{ij}}},\qquad
\mathcal L=\tfrac12(\mathcal L_{Q\to A}+\mathcal L_{A\to Q}).
$$

The reverse term swaps rows and columns. Train image-query pairs too; good text retrieval alone does not establish image-query quality. A reranker with graded judgments may use a separate relevance or pairwise-ranking loss.

**Q13. Negative selection.** Mix random catalog negatives for coverage, other batch assets for efficiency, and verified hard negatives for fine distinctions. A red metal chair can be useful against “red wooden chair,” provided the annotation genuinely excludes it. Deduplicate batches and mask known alternative matches; a multi-positive objective can reward total probability assigned to all relevant assets. Logged negatives reflect exposure and popularity, so examine sampling bias; a sampling-probability correction is conditional on the chosen sampled objective, not a universal repair for click bias.

**Q14. Magnitudes, gradients, and temperature.** L2 normalization makes dot product cosine similarity, preventing increased vector norms from improving scores alone. For a one-positive row, the logit gradient is $\partial\ell_i/\partial s_{ij}=p_{ij}-\mathbf1[j=i]$: it raises the positive and lowers competitors. Temperature scales embedding-score gradients by $1/\tau$ and changes softmax sharpness; too small a value amplifies difficult or noisy pairs. Tune or constrain it rather than assume smaller is better.

**Q15. Limited data and optimization.** Start with frozen features; compare projection-only, selective unfreezing, or LoRA on validation retrieval. Use AdamW for trainable dense parameters, a lower learning rate for unfrozen pretrained layers, and early stopping/checkpoint selection on held-out quality. Monitor forgetting and rare-style performance. Do not train a large encoder from scratch merely because the task is new.

## 6. Evaluation — minutes 26–31

Use judged queries and the eligible catalog snapshot as the evaluation universe. Report text and image queries separately, then aggregate with an explicit weighting rule. Retrieval Recall@K asks how many known relevant assets survive candidate selection:

$$
\operatorname{Recall@K}(q)=\frac{|\operatorname{TopK}(q)\cap R_q|}{|R_q|}.
$$

For final ordering, MRR emphasizes the first relevant result, $\operatorname{MRR}=\frac1Q\sum_q1/r_q$, with zero for no hit. NDCG handles graded relevance and discounts lower positions. AP rewards multiple binary-relevant results throughout a ranked list; specify truncation, averaging, and incomplete-judgment treatment. ROC-AUC can look strong against many easy negatives and does not directly measure the displayed top 20. Unlike the earlier notebook, the focus is query relevance over a catalog, not a fixed historical request's join labels.

**Q16. Search versus ordering errors.** Compare ANN candidates with exact vector-search results to measure index recall; compare both with human relevance to measure representation quality. Then evaluate the final ordering conditional on the candidate set. A reranker cannot rescue an asset absent from its input.

**Q17. Hidden slice failures.** Report modality, query language, asset style, new creators, rare categories, missing descriptions, and attribute composition. Group near-duplicates appropriately and track catalog coverage. A popularity-heavy aggregate may conceal failure on precisely the new content search should expose.

**Q18. Offline versus online.** Historical judgments and clicks do not fully predict satisfaction under a changed ranking. Run an appropriately randomized experiment with successful asset selection/use as a primary product outcome, plus latency, abandonment, and eligibility guardrails. Click lift alone can reward misleading previews; inspect delayed qualified use and creator coverage.

## 7. Serving and trade-offs — minutes 31–35

For “red wooden chair,” encode the text once; ANN might retrieve 200 candidates, eligibility checks remove unavailable assets, and an optional reranker promotes the best attribute matches before returning 20. Those counts are illustrative. For an image query, encode the selected image or crop instead; the catalog representations remain reusable.

**Q19. Memory and latency.** Two million 512-dimensional float16 vectors require $2{,}000{,}000\times512\times2=2{,}048{,}000{,}000$ bytes: 2.048 GB, about 1.91 GiB, before index overhead and metadata. Profile preprocessing, query encoding, search, filtering, reranking, and queues at realistic load; track p95/p99 as well as throughput. A generative LLM is unnecessary here, so there is no token-generation or time-to-first-token stage in this design.

**Q20. Version compatibility.** Equal vector length does not imply equal coordinates or semantics. Version both encoders, preprocessing, render configuration, and index; build a compatible replacement index, validate it, switch traffic atomically, and retain rollback. Do not silently mix old asset vectors with a newly trained query encoder.

**Q21. Cold start and freshness.** Render and encode a new asset from content, validate eligibility, and insert its version into the active compatible index. Updates replace old vectors; deletions must be filtered immediately even if physical index removal lags. Content enables cold-start retrieval, but poor descriptions or unseen styles still need quality checks. Monitor time-to-searchability and audit stale results.

## Optional constraint changes: reference responses

**1. Catalog grows 100 times.** Measure search latency, index memory, and ANN recall first. Compare sharding, compression, filter-aware search, and search effort; remeasure end-to-end relevance because compression and approximate search can lose matches. Keep candidate and reranker budgets bounded rather than expanding every stage proportionally.

**2. Attribute combinations fail.** Inspect whether views reveal the requested material/color and whether judgments distinguish partial matches. Add verified compositional pairs and hard negatives, compare higher-resolution views and a shortlist cross-encoder, and evaluate held-out combinations. A larger model cannot recover an attribute hidden from every rendering.

**3. No click logs, few reviewed pairs.** Retain the pretrained baseline and content-based indexing. Curate a small representative test set, train lightweight components cautiously, and choose additional labels from frequent or uncertain failure slices. Missing behavioral logs do not prevent retrieval, but they limit online preference learning; do not replace absent judgments with invented negatives.

## Source boundary

The collected [two-tower section](../../08_company_interviews.html#two-tower-models) attributes related questions to senior report 1128678 and principal report 995525. PhD new-grad report 1161498 supports recommender features, training examples, loss, optimization, and evaluation—not this exact asset-search prompt. The [CV retrieval case](../../02_ml_domain.html#cv-case02) is related instructional material; the answer above is our model-centered practice adaptation.

Method reference: [CLIP](https://arxiv.org/abs/2103.00020) for paired image/text representation learning. Multi-view mesh rendering, catalog search, reranking, numerical budgets, and product decisions here are our design choices, not claims about the paper's full system or Roblox's implementation.
