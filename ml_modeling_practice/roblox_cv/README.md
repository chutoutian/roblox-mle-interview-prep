# ML Modeling Knowledge — CV Design Mocks

Three English, model-centered mock interviews for the user-supplied **ML Modeling Knowledge Interview** format: **5 minutes introduction + 35 minutes technical discussion + 5 minutes Q&A**. They are separate from the CodeSignal data-analysis/model-training notebooks.

The task is to turn a product goal into training examples, a model, a loss, an evaluation plan and a workable inference path. Be ready to explain equations and tensor shapes. These mocks do not require implementation; the actual interviewer can choose different questions or ask for pseudocode. The CV/multimodal focus is our preparation choice, not confirmation of the assigned vertical.

## The three cases

| Mock | Questions | Reference answer | Main modeling decision |
| --- | --- | --- | --- |
| 1. Visual safety moderation | [Question document](01_visual_moderation_questions.md) | [Worked answer](01_visual_moderation_solution.md) | Multi-label visual prediction, reliable labels, review thresholds and safety/cost trade-offs |
| 2. Game genre classification | [Question document](02_game_genre_questions.md) | [Worked answer](02_game_genre_solution.md) | Creating labels from scratch, pretrained encoders, multiple images and text fusion |
| 3. Text/image search for 3D assets | [Question document](03_multimodal_asset_retrieval_questions.md) | [Worked answer](03_multimodal_asset_retrieval_solution.md) | Rendered-view representations, contrastive learning, negative sampling and retrieval quality |

Each question document contains an open prompt, illustrative constraints, seven design stages, **21 numbered ML-basic follow-ups**, and three optional changes. Its paired answer starts with the complete system picture, then works through the same stages and answers every follow-up. The reference is one defensible design, not an architecture to recite regardless of the question.

Each case also has an English detailed walkthrough with explanations, spoken interview answers, numerical examples, tensor shapes, and equations for all seven stages, Q1–Q21, and the three optional changes: [visual moderation](01_visual_moderation_detailed.md), [game genre classification](02_game_genre_detailed.md), and [multimodal asset retrieval](03_multimodal_asset_retrieval_detailed.md).

## The shared 35-minute answer structure

| Technical minutes | Stage | What to establish |
| --- | --- | --- |
| 0–4 | Problem formulation | Prediction unit, output, decision, error costs and constraints |
| 4–10 | Data and labels | Input availability, trustworthy supervision, missing labels and leakage-safe splits |
| 10–15 | Baseline | A complete small system and what it tells you |
| 15–20 | Model design | Input-to-output architecture, tensor dimensions and alternatives |
| 20–26 | Loss and training | Objective, optimizer, trainable parameters, imbalance and overfitting |
| 26–31 | Evaluation | Offline metrics, validation decisions, error slices and online product evidence |
| 31–35 | Serving and trade-offs | Inference path, latency, freshness, memory, failures and monitoring |

This is a flexible allocation, not an official rubric. A real interviewer selects follow-ups; the 21-question bank is not an instruction to answer everything in 35 minutes. After showing the complete path, ask which trade-off they would like to explore.

## What to say before choosing components

> “Let me clarify the prediction unit, what information is available at that point, and which errors are most costly. I’ll establish a baseline, explain the model and training objective, then connect evaluation to the final product decision and serving constraints.”

For each important choice, explain **why it fits this task**, **what alternative you considered**, and **what experiment would justify changing it**. A component name alone is not a design argument.

## Terms used in the answers

- **Training example:** one labeled unit such as an image, experience version or query–asset pair.
- **Backbone / encoder:** a model that converts raw images or text into learned representations.
- **Embedding:** the resulting feature vector; similarity is useful only if training makes its geometry meaningful.
- **Head:** a small prediction module on top of a representation.
- **Logit:** an unrestricted score before sigmoid or softmax.
- **Fine-tuning:** updating pretrained parameters for the target task; freezing leaves selected parameters unchanged.
- **Ablation:** removing or replacing a component to test whether it actually contributes.
- **Inference / serving:** applying the trained model to new inputs inside the product.

## Source boundaries

The user-supplied interview guide establishes the six-part modeling flow and 45-minute format. The [company interview collection](../../08_company_interviews.html#13-system-design) contributes reported topics; the [CV question bank](../../02_ml_domain.html#26-interview-questions--computer-vision) contributes foundations. These are not new scrapes or newly verified interview reports.

- **Moderation:** the collection attributes harmful chat/video modeling and separate-versus-joint models to a target-L6 report. This mock narrows the initial scope to image uploads and adds its own requirements.
- **Genre:** a level-unspecified report asks for game genre classification from scratch. The taxonomy, label budget, data and model here are our assumptions.
- **Asset retrieval:** an original adaptation of reported two-tower/recommendation topics and the CV asset-retrieval case; not an exact reported Roblox prompt.

Every scale, latency target, label budget and numerical example is hypothetical. Model-method references support technical ideas, not claims about interview frequency or production systems. Existing study notes and notebooks remain intact.

## Self-check after one mock

Can you identify the training unit and label, trace every tensor to the output, explain the loss, prevent leakage, justify the metric and action threshold, and walk through one real inference? If not, revisit that stage before adding a more complicated model. Save the last interview minutes for your own questions about the team's modeling work.
