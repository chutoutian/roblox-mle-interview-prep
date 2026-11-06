# Mock 1 — Visual Safety Moderation

## Interview prompt

Design a computer vision system that identifies unsafe images uploaded to a user-generated gaming platform. Explain the workflow from prediction targets and training data to model evaluation and actions on new uploads. Justify your choices and revise them when requirements change.

This **45-minute original mock** has 5 minutes for introductions, 35 for technical discussion, and 5 for candidate questions. It is model-centered, not pandas coding or generic distributed-systems design. Use equations, tensor shapes, and diagrams where useful; implementation is unnecessary.

## Illustrative constraints — MOCK, not Roblox requirements

- The initial scope is uploaded still images, not live chat or video.
- The platform has five policy categories. An image may violate several categories, and some labels have not been reviewed.
- There are 250,000 reviewed training candidates, including partially labeled examples. Reports and appeals provide additional, imperfect signals.
- The mock workload is 2 million image uploads per day. The automated initial decision has a 300 ms p95 latency budget; human-review time is outside that budget.
- Reviewers can inspect at most 1% of uploads. Missed severe violations are costly, but incorrectly restricting benign creators is also harmful.
- Uploads include screenshots, rendered assets, small details, embedded text, and unfamiliar visual styles.

Ask for missing information and identify conflicting or infeasible requirements.

## Seven discussion steps

### 1. Problem formulation · 0–4 minutes

Define one prediction example, its target, and the decision consuming the output. Clarify missing requirements and initial exclusions before choosing an architecture.

- **Q1.** How should the output represent an image that violates two policies at once?
- **Q2.** When are image-level decisions sufficient, and when do you need object locations or pixel boundaries?
- **Q3.** What do false positives and false negatives mean here, and why might their costs differ by policy?

### 2. Data and labels · 4–10 minutes

Describe trustworthy labels and training, validation, and test data. Address partial annotations, repeated assets, reporting behavior, and new creators/styles.

- **Q4.** If reviewers did not assess a particular policy, can that missing label be treated as a negative?
- **Q5.** How could a dataset made only from user reports misrepresent production uploads?
- **Q6.** Why could placing the same creator's near-duplicate assets in both training and test overstate quality?

### 3. Baseline · 10–15 minutes

Propose a simple, defensible starting model and an experiment that determines whether it is useful. Show the information flowing through the model before adding more complicated components.

- **Q7.** What does using a pretrained model mean, and what changes when its parameters are frozen?
- **Q8.** For a batch of eight RGB images and five policy categories, what would your input and output tensor shapes be?
- **Q9.** Why could a model that labels every image benign appear strong under an unsuitable evaluation measure?

### 4. Model design · 15–20 minutes

Identify failures the baseline cannot resolve and explain which changes would address them. Discuss embedded text, fine visual details, and the choice between shared and separate components without assuming that a larger model is automatically better.

- **Q10.** What architectural differences between convolution and self-attention matter for this task?
- **Q11.** If image text changes the meaning of a scene, how would you incorporate it, and what are the trade-offs of separate versus joint models?
- **Q12.** How would you decide whether higher resolution or localized evidence is worth the additional cost?

### 5. Loss and training · 20–26 minutes

Choose a training objective and optimization strategy consistent with your outputs and annotation contract. Explain your approach to rare policies, model adaptation, and transformations of training images.

- **Q13.** What is the difference between a logit and a probability, and where does the training loss act?
- **Q14.** How would you improve learning for rare violations without assuming unreviewed images are confirmed negatives?
- **Q15.** What could go wrong if an augmentation removes the only harmful region but retains the original label?

### 6. Evaluation · 26–31 minutes

Choose evidence that would justify deploying the model and explain how you would turn scores into actions. Keep model selection separate from the final estimate of performance, and account for both review capacity and severe misses.

- **Q16.** How would you distinguish a model's ranking quality from the quality of a particular decision threshold?
- **Q17.** What would it mean for a predicted violation probability of 0.8 to be calibrated?
- **Q18.** How would you choose decision thresholds when improving detection also sends too many images to review?

### 7. Serving and trade-offs · 31–35 minutes

Follow one new image through your deployed model and describe failure handling, monitoring, and feedback collection. Concentrate on the modeling consequences of serving decisions rather than designing generic storage and messaging infrastructure.

- **Q19.** Why measure p95 latency instead of only average inference time?
- **Q20.** Is a high-confidence benign prediction sufficient to guarantee safety, and how would you detect missed violations after release?
- **Q21.** How would you prevent preprocessing or model updates from silently changing production behavior?

## Optional constraint changes

1. A new severe policy has only 0.1% prevalence. How would that change your data, evaluation, and decision-making plan?
2. Some violations become apparent only when otherwise benign image content and embedded text are interpreted together. How would you adapt the architecture and labels?
3. The product expands to 60-second videos, and a harmful event may last only 0.5 seconds. What assumptions in your image solution stop being adequate?

## Source boundaries

The collected [harmful chat and video report](../../08_company_interviews.html#harmful-chat-and-video-content-detection) supports the broad harmful-content modeling topic and the discussion of one multimodal model versus separate models. It is labeled target L6, not a confirmed new-graduate question. This upload-only scope, every numerical constraint, the seven-step structure, and the follow-ups are original adaptations. The [CV moderation case](../../02_ml_domain.html#cv-case01) is supporting study material, not additional evidence of an actual interview.
