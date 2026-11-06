# Mock 2 — Game Genre Classification from Scratch

**Question document · 45-minute ML Modeling Knowledge interview · Computer Vision / Multimodal**

This is an original practice adaptation of a collected game-genre-classification topic, not a verbatim company question. The reported level was not stated. The data, budgets, taxonomy and follow-ups below are illustrative interview assumptions.

## Interviewer prompt

Design an ML system that assigns useful genre labels to newly published user-created games. Product wants these labels for discovery filters and recommendation features. There is no trusted genre-classification service today.

Explain the complete path from raw game content and annotations to training, evaluation and inference. Make and justify your choices rather than listing model names. No coding is required; draw the architecture and write the important tensor shapes and loss equations.

## Information available if you ask

- The mock catalog contains 100,000 experiences. Each may have an icon, up to three screenshots, a title, a description and creator-supplied tags. Some fields are missing or misleading.
- Assume product can agree on an initial taxonomy of approximately 20 genres. Decide whether one or several labels may apply, and how ambiguous cases should be represented.
- An initial human-labeling budget covers 5,000 experiences. Existing creator tags are not guaranteed to follow the taxonomy.
- A result is needed within five minutes of publishing or a content update; it is not a per-keystroke request. Compute is limited, and new games have no interaction history.
- The service must support unfamiliar creators and visual styles. These numbers and requirements are mock constraints, not Roblox measurements.

Use about five minutes for introduction, the 35 technical minutes below for the design, and five minutes for your questions. The interviewer chooses follow-ups; you are not expected to answer every question in the bank during one mock.

## 1. Problem formulation (0–4 min)

What exactly is the system predicting, what does one training example represent, and when is the prediction made? Translate discovery usefulness into an output contract and measurable success criteria. State what remains outside your initial scope.

**ML basics and follow-ups**

- **Q1.** Is genre prediction multiclass or multi-label? What observation determines the choice?
- **Q2.** Why should the prediction unit be an experience version rather than an individual screenshot?
- **Q3.** Is agreement with creator tags the same thing as improving discovery?

## 2. Data and labels (4–10 min)

How would you create the first reliable dataset? Describe annotation, ambiguous labels, weak supervision, missing content and evaluation splits. Explain which fields exist at publishing time.

**ML basics and follow-ups**

- **Q4.** What is the difference between a weak label and a human-reviewed gold label?
- **Q5.** How could a random screenshot split make the model appear better than it is?
- **Q6.** If a creator did not provide a genre tag, can you treat that genre as a negative label?

## 3. Baseline (10–15 min)

Propose a small first system and explain what it establishes. How would you decide whether visual information adds value? Explain what “from scratch” means in your assumed scope.

**ML basics and follow-ups**

- **Q7.** Why build a baseline when you already know a more powerful architecture?
- **Q8.** Does “build the system from scratch” require randomly initialized model weights?
- **Q9.** Could total play time or next-week retention be useful inputs at publishing time?

## 4. Model design (15–20 min)

Draw a candidate model from raw inputs to genre predictions. Explain multiple images, text, missing modalities and any trainable fusion layers. Show the shape of the representation at each boundary and name one alternative you would compare.

**ML basics and follow-ups**

- **Q10.** How would you choose between a CNN and a Vision Transformer for the visual encoder?
- **Q11.** How would you combine several screenshots without treating them as independent labeled games?
- **Q12.** What is early versus late fusion, and what happens when the description is missing?

## 5. Loss and training (20–26 min)

Write the objective for your output contract. Describe optimizer choice, which components you train first, and what you would change when training and validation diverge. Consider rare genres and partially annotated examples.

**ML basics and follow-ups**

- **Q13.** Why would you choose independent sigmoid outputs or a softmax output? How does unknown-label masking affect gradients?
- **Q14.** Why freeze a pretrained encoder initially, and when would you unfreeze it or use adapters?
- **Q15.** How would you address rare genres without destroying the meaning of the labels through augmentation?

## 6. Offline and online evaluation (26–31 min)

Select offline metrics and explain the intended operating point. Describe ablations, unfamiliar-style evaluation and a limited online rollout. How would you decide whether model complexity is justified?

**ML basics and follow-ups**

- **Q16.** What would macro-F1, micro-F1 and mean average precision tell you here?
- **Q17.** Must every genre use threshold 0.5? Where should thresholds be chosen?
- **Q18.** Why might better offline classification fail to improve recommendation engagement?

## 7. Serving and trade-offs (31–35 min)

Walk one new experience version through inference and into the product. Discuss latency, cost, content updates, failure behavior and monitoring. Close with the main trade-off and the next experiment.

**ML basics and follow-ups**

- **Q19.** Which work belongs at publishing time, and which work belongs on a discovery request?
- **Q20.** How could caching or quantization reduce cost, and what could go wrong?
- **Q21.** What should change when product adds a new genre or creators adopt a new rendering style?

## Optional interviewer changes

- **E1.** Creator descriptions are untrustworthy and often keyword-stuffed. How would the data and architecture change?
- **E2.** The model has strong average results but poor recall on a rare genre with only 50 reviewed examples. What do you investigate first?
- **E3.** Product asks for an explanation of each genre decision and requires a fivefold reduction in GPU cost. What would you promise, measure and change?

## Source boundary and related notes

- Collected topic: [Game genre classification from scratch](../../08_company_interviews.html#game-genre-classification-from-scratch), linked locally to report [1173576](https://www.1point3acres.com/bbs/thread-1173576-1-1.html). Its short reported prompt does not establish the mock's data availability or taxonomy assumptions.
- Foundations: [classification, detection and segmentation](../../02_ml_domain.html#cv-q10), [transfer learning](../../02_ml_domain.html#cv-q09), [multi-modal fusion](../../02_ml_domain.html#cv-q39), and [class imbalance](../../02_ml_domain.html#cv-q20).
- The complete questions and answers in this pack are original teaching material. The user-supplied interview guide establishes the format, not these exact questions or a guarantee about the assigned specialization.
