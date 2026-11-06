# Mock 3 — Text and Image Search for 3D Assets

This is an original ML Modeling Knowledge practice question, adapted from collected recommender/two-tower topics and the CV retrieval notes. It is **not a reported Roblox interview prompt**. Use a 45-minute session: five minutes of introduction, 35 minutes of technical discussion, and five minutes for your questions. This is a model-centered design discussion, not a pandas coding exercise.

## Open prompt

Design a system that helps a creator find reusable 3D assets using either a natural-language description or a reference image. A creator might type “a small red wooden chair” or upload a screenshot containing a chair and expect relevant assets, not necessarily identical images.

Explain the whole ML approach: what the system should predict, how you obtain training examples, a first working model, how you improve and train it, how you evaluate it, and how predictions reach the user. State assumptions and justify choices rather than naming an architecture immediately.

## Illustrative constraints

The following are **invented interview assumptions, not Roblox statistics or requirements**:

- The catalog contains two million assets, including new uploads and edited versions of existing assets.
- Each asset has a mesh, optional descriptive metadata, and up to six rendered views. Descriptions can be vague, incorrect, or missing.
- You have a modest set of reviewed query–asset relevance judgments and historical search/exposure logs. Different assets may all satisfy one query.
- Both text queries and reference-image queries matter. Uploaded images may have clutter, text overlays, or multiple objects.
- The interface displays 20 results. The initial target is 150 ms server-side p95 latency, excluding an upload's network transfer.
- New assets should become searchable without waiting for interaction history. Catalog eligibility restrictions still apply.

Clarify any assumption that changes your design. You are not expected to design every storage service or write training code during this interview.

## 1. Problem formulation — technical minutes 0–4

**Main question:** What exactly does “a relevant asset” mean for this product, and what should one prediction contain?

Follow-up questions:

- **Q1.** How would you distinguish finding an exact duplicate from finding a semantically suitable substitute?
- **Q2.** What should happen when a query contains several requested attributes or an image contains several objects?
- **Q3.** Which product requirements would you clarify before deciding how to represent the inputs?

## 2. Data and labels — technical minutes 4–10

**Main question:** What examples and labels would you collect, and how would you construct evaluation sets that reflect unfamiliar future content?

Follow-up questions:

- **Q4.** Does a search result that was not clicked necessarily mean the asset is irrelevant?
- **Q5.** If two distinct assets both match the query, how should that affect training examples?
- **Q6.** How would you prevent leakage from creators, copied assets, changing metadata, and later asset versions?

## 3. Baseline — technical minutes 10–15

**Main question:** What is the simplest complete system you could evaluate for both query types before committing to expensive training?

Follow-up questions:

- **Q7.** What can an off-the-shelf image model do with a raw 3D mesh?
- **Q8.** How would you test whether rendering choices, rather than asset semantics, drive matching?
- **Q9.** What would count as evidence that your first system needs domain adaptation rather than simply more engineering?

## 4. Model design — technical minutes 15–20

**Main question:** Describe the inputs, intermediate representations, and scores. Show how one query can be compared with a very large asset catalog.

Follow-up questions:

- **Q10.** How would you combine different views of one asset while preserving distinctive details?
- **Q11.** Why might a model that predicts a fixed set of object classes be insufficient for this search task?
- **Q12.** Where would you allow richer interaction between a query and an asset, and what limits the number of candidates you can process that way?

## 5. Loss and training — technical minutes 20–26

**Main question:** Specify a training objective and explain what a batch contains, what makes an example difficult, and which parameters you would update.

Follow-up questions:

- **Q13.** How do you choose nonmatching examples without repeatedly training against valid alternatives?
- **Q14.** What prevents a model from improving similarity scores merely by increasing representation magnitudes?
- **Q15.** If reviewed training data is limited, what would you train first and how would you decide whether to update more parameters?

## 6. Evaluation — technical minutes 26–31

**Main question:** How would you decide whether an improvement helps users find appropriate assets, including when several results are valid?

Follow-up questions:

- **Q16.** How would you separate errors in the initial candidate search from errors in the final ordering?
- **Q17.** Could a good aggregate score conceal poor image-query or new-creator performance?
- **Q18.** Why might stronger offline results fail to improve the live product?

## 7. Serving and trade-offs — technical minutes 31–35

**Main question:** Walk through one actual query, then explain how new uploads, deletions, and model updates change what is searchable.

Follow-up questions:

- **Q19.** How would you estimate representation memory and identify the dominant latency components?
- **Q20.** Why might vectors produced by two different model versions be incompatible even when their lengths match?
- **Q21.** How does the system handle a brand-new asset with no clicks or usage history?

## Optional constraint changes

1. The catalog grows 100 times larger, but the latency target stays unchanged. What would you measure and change first?
2. Users care strongly about attribute combinations such as “red wooden chair,” but the results often satisfy only “chair.” How would you diagnose and improve this?
3. You lose access to click logs and can afford only a small set of reviewed examples. What would you retain, simplify, and evaluate next?

## Source boundary

The collected [two-tower reports](../../08_company_interviews.html#two-tower-models) concern senior/principal candidates; PhD new-grad reports include recommender features, samples, loss, optimization, and evaluation. This specific asset-search task is our adaptation. The [CV retrieval case](../../02_ml_domain.html#cv-case02) provides related learning notes, not independent interview-report evidence.
