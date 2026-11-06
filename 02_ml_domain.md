# Part II — Machine Learning Domain Knowledge

> General ML/DL, modern transformer LMs, inference systems and VLMs. The learning order follows a
> dependency chain: **neural-net mechanics → mathematical tools → transformer LM → vision-language model → LoRA post-training → inference and
> scaling → hardware/parallelism → post-training → multimodality**. Generative models and learned compression are Chapter 6; codec fundamentals live in
> [01_video_domain.md](01_video_domain.md), and networking/streaming fundamentals live in
> [05_networking_streaming.md](05_networking_streaming.md).

**How to use these notes**

- Start with [Chapter 3: one sentence through the Transformer](#3-transformer-from-input-to-output). Follow the equations step by step, then run the PyTorch implementation. [Chapter 4](#4-vlm-from-image-and-question-to-answer) does the same for a vision-language model: one image and one question through ViT, projector, and LLM.
- Chapter 1 collects the basics to refer back to: neural-net mechanics, math and losses, training mechanics, and CNN / RNN / state-space architectures. Chapter 2 answers the common Transformer and LLM interview questions in short spoken form, with the derivations in Chapter 3. Chapter 5 explains post-training with LoRA, Chapter 6 generative models and learned compression, and chapters 7–9 inference and compute costs.
- Chapters 10–11 cover deployment and parallelism.
- Every Coding card retains a status: `READY` can serve as an interview answer; `PARTIAL` contains reusable pieces;
  `TOY` requires a clear explanation of its simplifications; `TODO` is only an implementation contract.

**Contents**

| § | Topic | What you should be able to explain |
|---|-------|----------------|
| 1 | Basics | 1.1 neural-net mechanics; 1.2 entropy/NLL, numerical stability, and losses; 1.3 regularization, optimizers, schedules, and normalization; 1.4 where CNNs, RNN/LSTMs, and state-space models fit |
| 2 | Interview questions: Transformers & LLMs | Short spoken answers on positions, normalization and training stability, attention and efficient attention, architecture and modern LLMs, and training and inference at scale |
| 3 | Transformer: from input to output | One sentence through every equation and the input and output shape of each step, the code read in VS Code; rotary positions; GPT-2 step by step as the decoder-only LLM; the vision transformers from ViT to DINOv3, with what each one changed |
| 4 | VLM: from image and question to answer | One image and one question through ViT, projector, and LLM with the shape of each step; the model line from CLIP to today; a PyTorch implementation |
| 5 | Post-training: LoRA | One frozen weight matrix through LoRA: the low-rank correction, which gradients exist, memory, merge, then a PyTorch implementation on the toy Transformer |
| 6 | Generative models & learned compression | Autoencoders and VAEs, VQ-VAE, VQGAN, FSQ, LFQ, RVQ and TiTok, GANs and diffusion, learned compression, and the Cosmos tokenizer, each with its notation and loss |
| 7 | Inference | KV cache, prefill/decode, batching, and sampling |
| 8 | Scaling laws & accounting | Order-of-magnitude estimates for parameters, FLOPs, memory, and training budgets |
| 9 | GPUs & roofline | Compute-bound vs bandwidth-bound execution, fusion, and utilization |
| 10 | Precision, efficiency & deployment | INT8/INT4, pruning, distillation, and on-device deployment |
| 11 | Parallelism | Collectives, DP/FSDP, TP, PP, and their communication costs |
| 12 | Metrics, evaluation & debugging | Metric selection, leakage, calibration, and generative evaluation |
| 13 | Rapid-fire drill | Oral self-test |
| 14 | Coding reverse index | prompt → knowledge point → implementation status |

---

# 1. Basics

> **Chapter guide:** The foundations to refer back to while reading the rest of the book: how a network computes and propagates gradients (1.1), the mathematics and losses behind training (1.2), the training loop itself (1.3), and the architectures that predate or complement the Transformer (1.4).

<a id="1-neural-network-fundamentals"></a>
## 1.1 Neural Net Basics

> **Section guide:** First build a mental model of computation graphs, gradients, and parameter initialization. The optimizer, Transformer, and post-training chapters all manipulate this same gradient chain.


### 1.1.1 Backpropagation

Computes gradients of the loss w.r.t. every parameter using the **chain rule**, layer by layer from output to input. An optimizer then uses those gradients to update weights.

```
Forward:   x → [layer 1] → [layer 2] → ... → loss L      (cache activations)
Backward:  ∂L/∂W_n ← ∂L/∂W_{n+1} · ∂W_{n+1}/∂W_n
```

**The memory point interviewers probe:** the forward pass must *cache activations* for the backward pass, which is why training memory ≫ inference memory. **Gradient checkpointing** trades compute for memory: store only a subset of activations, recompute the rest during backward (√N memory for ~33% extra compute). Essential for video models and long-context LLMs.

<a id="coding-linear-backward"></a>

##### Coding: Linear forward / backward `[READY · P0]`

- **Prompt:** implement a dense layer without autograd and derive `dX`, `dW`, and `db`.
- **Shape / cost:** `x:[B,Din]`, `W:[Din,Dout]`, `grad_out:[B,Dout]`; forward and backward are `O(B·Din·Dout)`.
- **Pitfalls:** cache the *input* from forward; sum `db` over the batch axis; this implementation is a single 2-D layer, not a complete MLP backprop engine.

```python
def backward(self, grad_out):
    x = self._cache
    grad_W = x.T @ grad_out
    grad_b = grad_out.sum(axis=0) if self.b is not None else None
    grad_x = grad_out @ self.W.T
    return grad_x, grad_W, grad_b
```

[Full implementation (NumPy)](./ml_coding.py#L242) · The PyTorch file instead relies on autograd for `Linear`; manual activation backward appears in [the custom autograd functions](./ml_coding_torch.py#L72). A finite-difference gradient checker and multi-layer chain are `[TODO]`.

### 1.1.2 Vanishing and exploding gradients

Gradients are products of many Jacobians:
```
gradient at layer k ≈ Π (∂layer_{i+1}/∂layer_i)

|Jacobian| < 1 repeatedly → vanishes exponentially
|Jacobian| > 1 repeatedly → explodes
```

Sigmoid's derivative maxes at 0.25, so a 10-layer sigmoid network attenuates gradient by up to 4^-10.

| Problem | Solution | Mechanism |
|---------|----------|-----------|
| Vanishing | **ReLU** | no saturation for positive inputs |
| Vanishing | **Residual connections** | `y = F(x)+x` → gradient highway, `∂y/∂x = 1 + ∂F/∂x` |
| Vanishing | **LSTM/GRU gates** | additive cell state, explicit memory path |
| Vanishing | **BatchNorm / LayerNorm** | keeps activations in a healthy range |
| Exploding | **Gradient clipping** | cap the global gradient norm |
| Both | **He / Xavier init** | keeps activation variance stable across depth |

### 1.1.3 Activation functions

| Activation | Formula | Issue | Use |
|-----------|---------|-------|-----|
| Sigmoid | 1/(1+e^−x) | saturates, not zero-centered | binary output only |
| Tanh | (e^x−e^−x)/(e^x+e^−x) | saturates | RNN hidden state |
| ReLU | max(0,x) | dying ReLU | CNN default |
| Leaky ReLU / PReLU | max(αx, x) | — | fixes dying ReLU; PReLU learns α |
| GELU | x·Φ(x) | slower | Transformers (BERT, GPT) |
| SiLU/Swish | x·σ(x) | slower | EfficientNet; MobileNetV3 uses the cheap hard-swish variant |
| **SwiGLU** | `(SiLU(xW₁) ⊙ xW₃)W₂`, `SiLU(z)=zσ(z)` | 3 matrices not 2 | **modern LLM FFN** (LLaMA, PaLM, Qwen) |

**Why ReLU over sigmoid in hidden layers**: sigmoid's gradient ≤ 0.25 → exponential attenuation. ReLU's gradient is exactly 1 for positive inputs, and it's a single comparison to compute.

**Dying ReLU**: if a neuron's pre-activation is always negative, its gradient is always 0 and it never recovers.

**Why SwiGLU in LLMs**: the gating gives the FFN a multiplicative interaction, empirically worth about 4% lower perplexity (log-perplexity 1.636 vs 1.677 for ReLU) at equal parameters and FLOPs in Shazeer (2020), while plain GELU or Swish FFNs were no better than ReLU. Note the FFN hidden dim is usually shrunk (to ~8/3·d instead of 4·d) to keep parameter count constant despite the third matrix — a nice detail to know.

<a id="coding-activations"></a>

##### Coding: Activation forward / backward `[READY · P0]`

- **Prompt:** write sigmoid/ReLU/GELU and the derivatives of sigmoid and ReLU; explain saturation and the derivative chosen at zero.
- **Shape / cost:** elementwise, shape-preserving, `O(number_of_elements)`.
- **Pitfalls:** the local sigmoid uses the compact `exp(-x)` formula and may emit an overflow warning for very negative inputs; ReLU convention here uses derivative `0` at exactly zero. SwiGLU is not implemented locally.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)
```

[Full implementation (NumPy)](./ml_coding.py#L84) · [Torch custom backward](./ml_coding_torch.py#L72). For an interview follow-up, make sigmoid branch-stable rather than claiming the compact version is overflow-free.

### 1.1.4 Weight initialization

```
All zeros:  every neuron computes the same gradient → symmetry never breaks
Too large:  activations saturate or explode
Too small:  activations → 0, gradients → 0
```

- **Xavier/Glorot** (for tanh/sigmoid): `σ² = 2/(fan_in + fan_out)` — keeps forward and backward variance balanced.
- **He** (for ReLU): `σ² = 2/fan_in`. ReLU zeroes ~half the units, halving output variance, so the factor of 2 compensates.
- **Transformer specifics**: scale residual-branch output projections by `1/√(2N)` (N = number of layers) so the residual stream's variance doesn't grow with depth. Related: **zero-init** the last layer of each residual branch (Fixup, ReZero) so the network starts as an identity map.

---

## 1.2 Mathy Things & Losses

> **Section guide:** This section keeps only the mathematics that directly affects an implementation or interview derivation: how probability distributions become bits or losses, why floating-point calculations fail, and how gradients pass through stochastic nodes.

### 1.2.1 Information theory: probability → bits → loss

The more surprising an event is under the model, the longer its description must be. If its probability is `p(x)`:

```
surprisal:       I(x) = -log₂ p(x)                  # information in one event
entropy:         H(p) = E_p[-log₂ p(x)]             # lower bound on optimal average code length
cross-entropy:   H(p,q) = E_p[-log₂ q(x)]
KL divergence:   D_KL(p||q) = H(p,q) - H(p) >= 0
```

When training a language model, the true data distribution `p` is fixed, so minimizing negative
log-likelihood / cross-entropy is equivalent to reducing `D_KL(p||q_θ)`. In compression,
`-log₂ q(x)` also approximates the number of bits an arithmetic coder spends on symbol `x`.
Accurate prediction and short compression are therefore two views of the same objective.

- **Conditional entropy** `H(X|C)`: how much uncertainty remains after observing context. LMs,
  hyperpriors, and inter prediction all use context to reduce it.
- **Mutual information** `I(X;C)=H(X)-H(X|C)`: how much uncertainty the context removes.
- **Perplexity**: if the loss uses natural logarithms, `PPL=exp(NLL)`; if it uses bits,
  `PPL=2^{bits/token}`. Compare it only under the same tokenizer and evaluation protocol.

Here `-log₂q(x)` is an ideal code length. A real file also pays for discretized probability tables,
coder termination, padding, headers, and side information; encoder and decoder must also use the
same quantized probability model bit-for-bit.

### 1.2.2 Numerical stability: algebraically equal, numerically different

A floating-point implementation cannot rely on algebraic equivalence alone. The most useful stability patterns are:

- **log-sum-exp**: `log Σ exp(xᵢ) = m + log Σ exp(xᵢ-m)`, where `m=max(x)`, prevents exponential overflow.
- **log-softmax + NLL**: compute directly from logits instead of first materializing probabilities that may underflow to zero.
- **Lower bounds on probability, variance, and scale**: `clamp_min(eps)` prevents `log(0)`, division by zero, and invalid negative variance.
- **Accumulation precision**: low-precision matmul can be fast, while reductions and optimizer states often accumulate in FP32.
- **Monitor before clipping**: gradient clipping can stop a single-step explosion, but may also conceal a bad loss scale or corrupt data.

In an interview, whenever you see `exp`, `log`, division, or a large reduction, first ask about input range, dtype, and axis. That checklist is more reliable than memorizing formulas.

### 1.2.3 Basic statistics for experiments

- `E[X]` is the mean over repeated samples; `Var(X)=E[(X-E[X])²]` measures variation.
  Covariance and correlation describe co-variation, not causation.
- **MLE** maximizes only the data likelihood; **MAP** also multiplies by a prior, which becomes an added regularizer in log-space.
- A point estimate is not enough: report a bootstrap/confidence interval and resample at the independent-sample level. For video tasks, that unit is usually the source video or subject, not highly correlated frames.
- **Aleatoric uncertainty** comes from data noise and may not disappear with more similar data;
  **epistemic uncertainty** comes from model ignorance and can often be reduced with broader data or model ensembles.

### 1.2.4 Gradient flow through sampling

The objective is often written as `∇θ E_{z~pθ}[f(z)]`, but a random sample is not an ordinary deterministic node. Distinguish these three approaches:

| Method | Core rewrite | Properties / use cases |
|------|----------|-----------------|
| **Pathwise / reparameterization** | `z=μθ+σθ⊙ε, ε~N(0,I)` | Low variance for continuous, differentiable distributions; the standard VAE answer |
| **Score-function / REINFORCE** | `E[f(z)∇θ log pθ(z)]` | Works with discrete variables and is unbiased, but has high variance; baselines/control variates reduce variance |
| **Relaxation / STE** | Gumbel-softmax or hard forward, identity backward | Practical but introduces bias or a train/test gap; common in VQ and quantization |

The key distinction is that reparameterization moves randomness into parameter-independent `ε` and
backpropagates through the sample's computation path. REINFORCE does not require `f` to be
differentiable with respect to the sample; it changes that sample's log-probability. An STE explicitly
uses a surrogate derivative during backward and must not be described as the true derivative.

**Primary sources:** [Shannon 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) ·
[VAE reparameterization](https://arxiv.org/abs/1312.6114) ·
[REINFORCE](https://doi.org/10.1007/BF00992696) ·
[stochastic-neuron estimators / STE](https://arxiv.org/abs/1308.3432) ·
[Gumbel-softmax](https://arxiv.org/abs/1611.01144)

<a id="2-loss-functions"></a>
### 1.2.5 Loss Functions

---

<a id="21-l1-vs-l2"></a>
#### 1.2.5.1 L1 vs L2

```
L1: |y − ŷ|          L2: (y − ŷ)²
```

| | L1 | L2 |
|--|----|----|
| Outliers | robust (linear) | sensitive (quadratic) |
| Gradient | constant ±1 | proportional to error |
| Optimum predicts | **median** | **mean** |
| Smoothness | non-differentiable at 0 | smooth |

**Why L2 blurs in image/video restoration** — the single most-asked loss question in this domain: given an ambiguous input, many outputs are plausible. L2 is minimized by the *mean* of that distribution, and the mean of many differently-positioned sharp textures is a blur. L1 predicts the median, which is slightly sharper but has the same fundamental problem. The real fix is a loss that measures *distribution* similarity rather than pointwise error: perceptual, adversarial, or diffusion.

**Charbonnier / smooth-L1**: `√(x²+ε²)` — L1's robustness with a differentiable origin. The standard choice in super-resolution and flow networks.

<a id="22-cross-entropy"></a>
#### 1.2.5.2 Cross-entropy

```
CE = − Σᵢ yᵢ log(pᵢ)      →  for one-hot labels:  −log p_correct
```

**Numerical stability** — worth being able to write:
```
log_softmax(z)ᵢ = zᵢ − max(z) − log Σⱼ exp(zⱼ − max(z))
```
Subtracting the max prevents `exp` overflow; computing log-softmax directly (rather than `log(softmax(z))`) prevents underflow of small probabilities to 0. This is why frameworks give you `cross_entropy(logits, target)` and warn you not to apply softmax first.

**Label smoothing**: replace the one-hot target with `(1−ε)` on the true class and `ε/(K−1)` elsewhere. Prevents overconfidence, improves calibration and generalization. ε = 0.1 typical.

<a id="coding-stable-ce"></a>

##### Coding: Stable softmax / cross-entropy `[READY + TOY · P0]`

- **Prompt:** compute CE directly from logits and derive `∂L/∂logits = (softmax − one_hot)/N`.
- **Shape / cost:** `logits:[N,C]`, `labels:[N]`; `O(NC)` time and `O(NC)` teaching-space.
- **Pitfalls:** subtract the maximum on the class axis and never call `log(softmax(...))` as two naive operations. The correct NumPy definition below exists at line 42, but a second same-name definition at line 59 shadows it and omits max subtraction; therefore the current module callable is **not** a reference answer.

```python
def stable_softmax(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def stable_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp
    return -np.mean(log_probs[np.arange(len(labels)), labels])
```

[Existing NumPy file (including the documented shadowing bug)](./ml_coding.py#L42) · [Torch implementation](./ml_coding_torch.py#L45) is numerically stable on CPU, but its `torch.arange` must follow `logits.device` before treating it as device-generic. MSE/MAE/BCE live beside both implementations.

<a id="23-perceptual-loss"></a>
#### 1.2.5.3 Perceptual loss

Compare *feature maps* from a pretrained network instead of pixels:
```
L_perc = Σ_l w_l ||φ_l(ŷ) − φ_l(y)||²
```
Lower layers capture texture/edges; higher layers capture semantics. Penalizes structurally different outputs even when pixel error is low, so it avoids the blur that L2 induces. **LPIPS** is the calibrated version: features are channel-wise reweighted by weights fit to human 2AFC judgments.

**Style/Gram loss**: match the Gram matrix (channel correlations) of features — matches *texture statistics* without matching position. Used in style transfer and texture synthesis.

<a id="24-focal-loss"></a>
#### 1.2.5.4 Focal loss

For heavy class imbalance (99% background in detection):
```
FL(p_t) = −α_t (1 − p_t)^γ · log(p_t)         γ ≈ 2
```
`(1−p_t)^γ` shrinks the loss of well-classified examples toward zero, so the gradient signal is dominated by hard examples instead of the sea of easy negatives.

<a id="25-contrastive-losses-needed-for-10-and-for-embedding-fine-tuning"></a>
#### 1.2.5.5 Contrastive losses (needed for §4 and for embedding fine-tuning)

**InfoNCE / NT-Xent** — the workhorse of CLIP, SimCLR, and every embedding model:
```
L = −log [ exp(sim(q, k⁺)/τ) / Σⱼ exp(sim(q, kⱼ)/τ) ]
```
- `sim` = cosine similarity, `τ` = temperature.
- Effectively a cross-entropy over "which of the N candidates is the true match."
- **Temperature τ** controls how much the loss concentrates on hard negatives. Small τ → sharp, focuses on the hardest negatives, can be unstable. CLIP *learns* τ (as a log-parameterized scalar, clamped).
- **Batch size matters enormously** because in-batch negatives are the negatives. CLIP used 32k. Tricks when you can't: memory banks (MoCo), gradient caching, cross-device negative gathering.
- **Hard negative mining** is the single biggest lever for a *fine-tuned* embedding model — random negatives become trivially easy after a few epochs. Mine negatives with the current model, filter out false negatives above a similarity threshold.

**Triplet loss**: `max(0, d(a,p) − d(a,n) + margin)`. Older; InfoNCE with many negatives generally beats it.

> **Directly relevant to you**: the Qwen3-VL embedding fine-tune for brand-conflict and context-similarity checking is exactly this. Expect "how did you pick negatives?" and "how did you evaluate the embedding — retrieval recall@k, or a threshold-based decision metric?"

<a id="coding-loss-suite"></a>

##### Coding: Focal loss / label smoothing / CLIP InfoNCE `[PARTIAL · P1]`

- **Prompt:** implement each loss from logits, then explain its reduction, temperature, and behavior on easy examples or hard negatives.
- **Shape / cost:** focal/label-smoothed CE use `[N,C]` logits and `O(NC)`; symmetric CLIP builds an `[N,N]` similarity matrix and costs `O(N²D)`.
- **Pitfalls:** normalize embeddings before cosine similarity; use both image→text and text→image CE; keep temperature positive (usually log-parameterized); do not confuse a single cosine primitive with the full InfoNCE loss.

```python
def clip_info_nce(image_embeddings, text_embeddings, logit_scale):
    """TODO: normalize, form N x N logits, and average row/column CE."""
    raise NotImplementedError

def label_smoothed_cross_entropy(logits, labels, epsilon=0.1):
    """TODO: mix one-hot targets with the documented smoothing convention."""
    raise NotImplementedError

def focal_loss(logits, labels, gamma=2.0, alpha=None):
    """TODO: gather p_t from stable log-softmax and define reduction."""
    raise NotImplementedError
```

The vector primitive is [available in NumPy](./ml_coding.py#L341) and [Torch](./ml_coding_torch.py#L407); the batched losses themselves are `[TODO]`.

---

## 1.3 Training Mechanics

> **Section guide:** Treat training as a closed loop: initialization sets signal scale, normalization and residual paths keep gradients healthy, the optimizer and schedule determine each step size, and regularization constrains what the model learns.

---

<a id="3-regularization"></a>
### 1.3.1 Regularization

**Overfitting**: good on training data, poor on unseen data — the model memorized rather than generalized.

- **Weight decay (L2)**: adds `λ||W||²`. Spreads evidence over many features. **AdamW** decouples it from the adaptive update, which matters because in vanilla Adam the L2 gradient gets divided by `√v̂`, making effective decay inversely proportional to gradient magnitude — i.e. inconsistent.
- **L1**: promotes exact sparsity. Rare in deep learning outside pruning.
- **Dropout**: randomly zero units during training; prevents co-adaptation. **Not recommended for low-level image/video tasks** — the stochastic feature noise propagates to pixel outputs as spatially inconsistent artifacts. Prefer weight decay + augmentation for restoration. Also largely absent from modern LLMs (huge datasets, single epoch → little overfitting to regularize).
- **Stochastic depth / DropPath**: randomly drop whole residual branches. Standard in deep ViTs and ConvNeXt.
- **Data augmentation**: crop, flip, color jitter; for video, temporal jitter, speed perturbation, and *consistent* spatial augmentation across a clip (augmenting frames independently destroys the temporal signal you're trying to learn).
- **Early stopping** on a validation metric.
- **Smaller capacity** — try this before piling on regularizers.
- **Split by scene/clip, not by frame, for video.** Adjacent frames are nearly identical; a random frame split leaks the test set into training and inflates validation metrics by several dB. This is the #1 data bug in video ML and a favorite interview trap.

---

<a id="4-optimization"></a>
### 1.3.2 Optimization

<a id="41-sgd-vs-adam"></a>
#### 1.3.2.1 SGD vs Adam

| | SGD + momentum | Adam / AdamW |
|--|-----------------|------|
| LR | single global | adaptive per-parameter |
| Convergence | slower, needs tuning | fast, forgiving |
| Generalization | often better final result (CNNs) | sometimes worse |
| Memory | 1× params (momentum) | **2× params** (m and v) |
| Default for | ResNet-era vision | transformers, everything modern |

**Momentum**:
```
v_t = β v_{t−1} + g_t
w_t = w_{t−1} − lr · v_t
```
Accumulates velocity along persistent gradient directions, damping oscillation across a ravine.

**Adam**:
```
m_t = β₁ m_{t−1} + (1−β₁) g_t          ← 1st moment (mean)
v_t = β₂ v_{t−1} + (1−β₂) g_t²         ← 2nd moment (uncentered variance)
m̂ = m_t/(1−β₁ᵗ),  v̂ = v_t/(1−β₂ᵗ)     ← bias correction (m,v start at 0)
w ← w − lr · m̂ / (√v̂ + ε)
```
Each parameter's effective step is normalized by its recent gradient magnitude → scale-invariant, which is why it works out of the box on architectures with wildly different layer scales.

**Bias correction matters**: `m_t` and `v_t` both start biased toward zero, but their ratio is not
biased by the same factor. Without correction, the update is scaled by
`(1−β₁ᵗ)/√(1−β₂ᵗ)` (about `3.16×` at step 1 for the default betas, ignoring ε), so it does not have
the intended Adam step size. Warmup addresses early optimization instability; it is not a substitute
for the correction.

**Memory note for large models**: AdamW states are 2× the parameters in fp32 — for a 7B model that's 56 GB of optimizer state alone. Hence 8-bit optimizers (bitsandbytes), ZeRO sharding, and why LoRA is so attractive (optimizer state only for the adapters).

<a id="coding-optimizers"></a>

##### Coding: SGD / Adam / AdamW boundary `[READY + TOY · P0]`

- **Prompt:** implement SGD with momentum and Adam with first/second moments and bias correction; then turn Adam into AdamW.
- **Shape / cost:** one `m` and one `v` tensor per Adam parameter; `O(P)` time, `2P` optimizer-state elements.
- **Pitfalls:** increment the step once per optimizer step, not once per tensor. AdamW decays the parameter outside the `m/v` update; adding `λp` to `g` is coupled L2.

```python
self.t += 1
for name in params:
    if name not in self.m:
        self.m[name] = np.zeros_like(params[name])
        self.v[name] = np.zeros_like(params[name])
    g = grads[name]
    self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g ** 2
    m_hat = self.m[name] / (1 - self.beta1 ** self.t)
    v_hat = self.v[name] / (1 - self.beta2 ** self.t)
    params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

[Full implementation (NumPy SGD/Adam)](./ml_coding.py#L123) · [Torch SGD/Adam](./ml_coding_torch.py#L121). Core Adam with `weight_decay=0` is `[READY]`; the Torch `weight_decay>0` branch currently inserts decay into `g` and is therefore `[TOY]` until fixed, despite its `AdamW-style` comment. A correct full AdamW implementation is `[TODO]`; its defining ordering is `p ← (1−lr·wd)p` followed by Adam moments computed from the raw loss gradient.

<a id="42-learning-rate-schedules"></a>
#### 1.3.2.2 Learning rate schedules

- **Warmup**: linear ramp from ~0 over the first N steps. Critical for transformers: Adam's second-moment estimate is noisy and near-zero early, producing enormous effective steps that damage the model before it can recover. Also stabilizes LayerNorm statistics.
  ```
  lr(t) = lr_max · t / warmup_steps       for t < warmup_steps
  ```
- **Cosine annealing**: smooth decay to lr_min. The default for basically everything.
  ```
  lr(t) = lr_min + ½(lr_max − lr_min)(1 + cos(π t/T))
  ```
- **Step decay**: ×γ every N epochs. Simple, abrupt, mostly legacy.
- **Inverse-sqrt**: original Transformer paper, still used in some NMT setups.

**Practical LLM/VLM recipe**: warmup 1–3% of steps → cosine to ~10% of peak. LR ~1e-5 to 2e-5 for full-parameter SFT of a 7B model; 1e-4 to 3e-4 for LoRA (adapters tolerate and need larger LRs since they start at zero).

<a id="43-gradient-clipping"></a>
#### 1.3.2.3 Gradient clipping

```
if ||g||₂ > threshold:  g ← g · threshold / ||g||₂
```
Applied to the **global** norm across all parameters (preserves gradient direction; per-parameter clipping distorts it). Standard value 1.0. Essential for RNNs, transformers, and RL.

<a id="coding-training-utils"></a>

##### Coding: Warmup/cosine schedule + global gradient clipping `[TODO · P1]`

- **Prompt:** implement the scheduler at all boundary steps and clip a *collection* of gradients by one global L2 scale.
- **Shape / cost:** scheduler is `O(1)`; norm and scaling are `O(P)` over all gradient elements.
- **Pitfalls:** define whether `step=warmup_steps` belongs to warmup or decay; protect zero norm; preserve one global direction; unscale mixed-precision gradients before clipping.

```python
def warmup_cosine_lr(step, warmup_steps, total_steps, peak_lr, min_lr=0.0):
    """TODO: linear warmup, then cosine decay with explicit endpoints."""
    raise NotImplementedError

def clip_grad_global_norm(grads, max_norm, eps=1e-12):
    """TODO: compute one norm over every tensor and apply one shared scale."""
    raise NotImplementedError
```

No local scheduler or global-clipping helper exists; both are `[TODO]`. Do not present per-tensor clipping as an equivalent shortcut.

<a id="44-mixed-precision"></a>
#### 1.3.2.4 Mixed precision

- **FP16**: 1 sign / 5 exp / 10 mantissa. 2× memory, tensor-core speed. Narrow exponent range → gradients underflow to zero. Needs **loss scaling**: multiply the loss by a large constant before backward, unscale before the optimizer step. `GradScaler` does this adaptively (raise the scale until an inf/nan appears, then back off).
- **BF16**: 1 / 8 / 7. Same exponent range as FP32 → no loss scaling needed, but less mantissa precision. **Preferred for training** on A100/H100/TPU.
- **FP8** (H100+): E4M3 for forward, E5M2 for gradients. Needs per-tensor scaling factors. Now standard for large-scale training and inference on Hopper/Blackwell.
- Master weights stay FP32; only forward/backward run in low precision.

<a id="45-distributed-training-name-drop-level"></a>
#### 1.3.2.5 Distributed training (name-drop level)

- **DDP / data parallel**: replicate the model, shard the batch, all-reduce gradients. Simple, memory-bound by model size.
- **ZeRO / FSDP**: shard optimizer states (stage 1), gradients (2), and parameters (3) across ranks. Stage 3 lets you train models that don't fit on one GPU with near-DDP throughput.
- **Tensor / pipeline / sequence parallel**: split within a layer / across layers / across the sequence dimension. Needed above ~30B or for very long contexts.
- **Gradient accumulation**: simulate a large batch on limited memory. Remember to scale the loss by 1/accum_steps.

---

<a id="5-normalization"></a>
### 1.3.3 Normalization

All normalize activations to ~zero mean / unit variance then apply learned scale γ and shift β. They differ in **which axes** they normalize over.

**BatchNorm** — per channel, across the batch:
```
x: [N, C, H, W]  → compute μ_c, σ_c over (N, H, W)
```
- Train: batch statistics. Inference: running averages accumulated during training.
- **Breaks at small batch size** — a real problem in video, where batch is memory-limited.
- Train/inference discrepancy is a classic source of "works in training, bad at eval" bugs.

**LayerNorm** — per sample, across features:
```
x: [..., C] → compute μ, σ over C
```
- No batch dependency; identical at train and inference; fine at batch size 1. Standard in transformers.

**GroupNorm** — split C into G groups, normalize within each group per sample. Batch-independent like LN, retains some channel structure like BN. Preferred for detection/segmentation/video with small batches.

```
InstanceNorm: G = C   (each channel alone)      → style transfer
GroupNorm:    G groups, C/G channels per group   → video, detection
GroupNorm with G = 1 normalizes all C×spatial values per sample; this matches
LayerNorm only when LayerNorm uses the same axes (affine parameterization can still differ).
```

**RMSNorm** — the modern LLM choice (LLaMA, Qwen, Gemma):
```
x̂ = x / √(mean(x²) + ε) · γ           ← no mean subtraction, no bias
```
~10–20% faster than LayerNorm with no quality loss. If you say "LayerNorm" about a modern LLM, expect to be corrected — say RMSNorm.

**Pre-norm vs post-norm**: the original Transformer put LN *after* the residual add (post-norm), which needs careful warmup to train deeply. Modern models use **pre-norm** (`x + F(LN(x))`), which keeps a clean identity path through the whole network and trains stably at depth. Some recent models use both (sandwich norm) or QK-norm to control attention logit growth.

**For video training with small batches**: prefer LayerNorm/GroupNorm, or freeze BatchNorm (use the pretrained running stats and don't update them) when fine-tuning an ImageNet backbone.

<a id="coding-normalization"></a>

##### Coding: BatchNorm / LayerNorm / RMSNorm `[READY + TODO · P0]`

- **Prompt:** implement BN train/eval and LN over the last feature axis, then remove LN's recentering step to obtain RMSNorm.
- **Shape / cost:** local implementations use BN `x:[N,C]` and LN `x:[...,C]`; both are linear in the number of elements. RMSNorm preserves `[...,C]`.
- **Pitfalls:** state the reduction axes; BN owns running state while LN/RMSNorm do not. Update Torch BN buffers under `no_grad`/in-place in production; do not reuse batch statistics at evaluation.

```python
class LayerNorm:
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class RMSNorm:
    def forward(self, x):
        """TODO: divide by RMS on the last axis, then apply learned scale."""
        raise NotImplementedError
```

[Full implementation (NumPy BN/LN)](./ml_coding.py#L176) · [Torch BN/LN](./ml_coding_torch.py#L191). BN/LN are `[READY]` for the stated interview shapes; RMSNorm and GroupNorm are `[TODO]`. The current Torch BN running-stat assignment is teaching code, not a production reference.

---

<a id="6-cnn-architecture"></a>
## 1.4 CNNs & Other Architectures

> **Section guide:** Transformers are the main line, but visual locality, streaming state, or very long sequences can make a CNN, RNN, or state-space model more appropriate. Compare inductive bias, parallelism, and state cost together.


<a id="61-convolution-basics"></a>
### 1.4.1 Convolution basics

```
output_size = floor((W + 2P − K)/S) + 1
W=32, K=3, P=1, S=1  →  32   (same-size)
FLOPs per layer ≈ H_out · W_out · K² · C_in · C_out
Params           ≈ K² · C_in · C_out  (+ bias)
```

**Receptive field**: the input region influencing one output unit. L layers of K×K stride-1: `RF = 1 + L(K−1)`. With stride/dilation it grows much faster. For video you also care about the **temporal receptive field** — too small and the model can't see the motion it's supposed to model.

**Why two 3×3 beats one 5×5**: same receptive field, `2·9=18` vs `25` parameters, and an extra nonlinearity. This is the VGG insight and a standard interview question.

<a id="coding-convolution"></a>

##### Coding: 2-D convolution / cross-correlation `[TOY · P1]`

- **Prompt:** hand-write a single-channel 2-D cross-correlation with stride and `valid`/`same` padding; derive the output dimensions before coding.
- **Shape / cost:** input `[H,W]`, kernel `[Kh,Kw]`, output `[Hout,Wout]`; `O(Hout·Wout·Kh·Kw)`.
- **Pitfalls:** DL libraries normally perform cross-correlation (no kernel flip); even-sized kernels make "same" padding asymmetric; validate stride and empty-output cases. The existing scratch file is not a clean canonical import because `List` is missing and unrelated demo code later in the module has bugs.

```python
for i in range(out_H):
    for j in range(out_W):
        total = 0.0
        for ki in range(kH):
            for kj in range(kW):
                total += padded[i * stride + ki][j * stride + kj] * kernel[ki][kj]
        result[i][j] = total
```

[Existing teaching implementation (module cleanup required)](./interview_practice.py#L82). A standalone tested Conv2D answer, multi-channel extension, and max-pool backward are `[TODO]`.

<a id="62-efficient-convolutions"></a>
### 1.4.2 Efficient convolutions

**Depthwise separable**:
```
Standard K×K, C_in→C_out:  K² · C_in · C_out
Depthwise (per channel):   K² · C_in
Pointwise (1×1 mixing):    C_in · C_out
Total:                     K² · C_in + C_in · C_out
Reduction ≈ 1/C_out + 1/K²  ≈ 8–9× for K=3, C_out=256
```
MobileNet, EfficientNet. **Caveat worth stating**: the FLOP reduction is far larger than the latency reduction, because depthwise convolutions have low arithmetic intensity and are memory-bandwidth-bound. This is *the* canonical example of FLOPs ≠ latency.

**Dilated (atrous)**:
```
Effective span = K + (K−1)(r−1)
K=3, r=1: ● ● ●            RF = 3
K=3, r=2: ●   ●   ●        RF = 5
K=3, r=4: ●       ●     ●  RF = 9
```
Bigger receptive field at constant parameter count. DeepLab, WaveNet. **Gridding artifacts** if you stack the same rate — vary them (1,2,4,2,1).

**Grouped convolution**: split channels into G groups, convolve independently. G=C is depthwise, G=1 is standard. ResNeXt's "cardinality."

**1×1 convolution**: channel mixing / dimensionality reduction. The bottleneck in ResNet blocks, and the "pointwise" half of depthwise-separable.

<a id="63-upsampling"></a>
### 1.4.3 Upsampling

- **Transposed conv**: learnable upsampling. **Checkerboard artifacts** when kernel size isn't divisible by stride, because output positions receive unequal numbers of kernel contributions.
- **Upsample + conv** (preferred): fixed bilinear/nearest upsample, then a regular conv. No checkerboarding. Standard in restoration networks.
- **PixelShuffle / sub-pixel conv**: produce `r²·C` channels at low resolution, then rearrange to `C` channels at `r×` resolution. It avoids the uneven-overlap mechanism of transposed-convolution checkerboards, but learned periodic artifacts are still possible; it is cheap because convolution happens at low resolution and is common in super-resolution (ESPCN onward).

<a id="64-pooling-and-skip-connections"></a>
### 1.4.4 Pooling and skip connections

- **Max pooling**: keep the strongest activation; gradient flows only to the argmax. Local translation invariance.
- **Average pooling**: mean; gradient spread evenly.
- **Global average pooling**: `[N,C,H,W] → [N,C]`. Replaces flatten+FC; far fewer parameters.
- **Strided conv**: learned downsampling, generally preferred over fixed pooling in modern nets.
- **For low-level tasks, minimize pooling** — every downsample permanently discards spatial detail. Use skip connections to route it around the bottleneck.

**ResNet residual**: `y = F(x) + x`. Two benefits: (1) `∂y/∂x = I + ∂F/∂x` gives a direct gradient path; (2) identity is easy to represent, so optimization usually degrades much less as depth grows. Extra depth can still hurt through optimization, overfitting, or compute constraints. **Pre-activation ResNet** (BN-ReLU-Conv order) gives an even cleaner identity path.

**U-Net skip**: concatenate encoder features into the decoder at each scale. Essential for pixel-level tasks — the decoder needs the high-frequency detail the bottleneck threw away.

<a id="65-frequency-intuition"></a>
### 1.4.5 Frequency intuition

- **Low frequency**: slow variation, smooth regions. Global structure.
- **High frequency**: edges, texture, noise. Fine detail.
- **Blur kernel** (low-pass): all positive, sums to 1.
- **Edge kernel** (high-pass): has negatives, sums to 0 → zero response on flat regions.

**Laplacian**:
```
∇²f = f(x+1,y) + f(x−1,y) + f(x,y+1) + f(x,y−1) − 4f(x,y)

 0 −1  0
−1  4 −1        ← 4-neighbour
 0 −1  0
```
Flat region → neighbors equal center → 0. Edge → large response. Isotropic (uses all directions) so it's orientation-independent, unlike Sobel.

**Sharpening = identity + Laplacian**: 4-neighbour → center 5; 8-neighbour → center 9.

**Why this matters for you**: "CNNs are biased toward low frequencies" (spectral bias) explains why L2-trained restoration nets look blurry, and why GAN/diffusion losses are needed to put the high frequencies back.

---

### 1.4.6 RNN, LSTM, and GRU

A vanilla RNN uses `h_t=f(W_x x_t + W_h h_{t-1})` to compress history into a fixed-size state. It
naturally supports streaming with O(hidden) state per step, but temporal recurrence prevents training
and token generation from being fully parallelized, while long Jacobian chains cause vanishing or exploding gradients.

An LSTM uses the additive path through its cell state and input/forget/output gates to control writes,
retention, and reads. A GRU merges some gates and is lighter. They mitigate rather than eliminate
long-range dependency problems. Recurrent video propagation has constant-size state, but a scene cut
or error can contaminate later state, so reset behavior must be designed explicitly.

### 1.4.7 State-space models

A continuous linear state-space model starts from `h'(t)=Ah(t)+Bx(t), y(t)=Ch(t)+Dx(t)`. After
discretization it can be viewed as a recurrence or, under some parameterizations, a parallel scan or
convolution. S4 models long sequences efficiently through a structured `A`; Mamba makes selected
parameters input-dependent for selective retention and forgetting, together with a hardware-aware scan.

Their attraction is near-linear state updates with sequence length and streaming inference without a
Transformer-style growing KV cache. The tradeoff is that arbitrary pairwise content interaction is no
longer explicit as in attention; ecosystem maturity, kernel support, and task quality must also be measured.
Do not declare a winner based only on asymptotic complexity.

**Primary sources:** [S4](https://arxiv.org/abs/2111.00396) · [Mamba](https://arxiv.org/abs/2312.00752)

---

# 2. Interview Questions — Transformers & LLMs

Short answers to Transformer and LLM questions, followed by [computer vision interview questions](#26-interview-questions--computer-vision). The derivations are in [Chapter 3](#3-transformer-from-input-to-output): the step-by-step walkthrough, rotary positions, the decoder-only GPT-2 walkthrough and the vision transformers.

## 2.1 Positions

**1. Why is an added positional encoding absolute while RoPE is relative?**

An added encoding mixes absolute position into each token once, at the input; RoPE rotates queries and keys inside every attention layer, so the score sees only the offset.
- Added: the score of $(e+P_p)W^Q$ with $(e'+P_{p'})W^K$ expands into four terms, and those with $P$ depend on $p$ and $p'$ separately, so shifting both tokens changes it.
- RoPE: pair $j$ of $q_p$ turns by $p\theta_j$ and of $k_{p'}$ by $p'\theta_j$, so the score is $q_p^{\top}R_{p'-p}k_{p'}$ and $(3,1)$ scores like $(103,101)$.
- RoPE adds no parameters and extrapolates better, so most current LLMs rotate queries and keys instead of adding $P$; see [Rotary positions](#rotary-positions-rope).

**2. Why does a Transformer need positional information, and what makes self-attention permutation equivariant?**

Without positions, unmasked self-attention is permutation-equivariant: permute the input tokens and the outputs are permuted the same way, so word order is invisible. Each score uses only two token vectors and each output is a weighted sum over a set, while the FFN, norms and residuals act on each token separately. So "we watch" and "watch we" give each word the same vector, just reordered. Position must be injected to break the symmetry: added absolute encodings at the input, or relative biases or RoPE inside attention. A causal mask breaks it partly, since each position sees a different prefix.

**3. What is RoPE, where is it applied inside a layer, and why did it become the default in LLMs?**

RoPE rotates each query and key by angles proportional to its token's position, so their dot product depends only on the offset.
- Where: every self-attention layer and head, after the projections and head split, before the scores; cross-attention usually has none.
- How: pair $j$ of the $d_k$ features turns by $p\theta_j$, $\theta_j = 10000^{-2j/d_k}$; no parameters, nothing added to the embeddings.
- Values stay unrotated: position should decide who attends, not what is carried.
- Why the default: relative position with no extra score term, so fused attention kernels work unchanged and the KV cache simply stores keys already rotated; rescaling $\theta_j$ extends context cheaply.

**4. What are the trade-offs between sinusoidal and learned absolute position embeddings?**

Within the trained length they perform about the same; they differ in parameters and beyond that length.
- Sinusoidal: no parameters, defined for any position, and a shift by $\Delta$ is a fixed rotation of each sin-cos pair, so offsets are easy to express.
- Learned (BERT, GPT-2, ViT): a table with $L_{\max}$ rows, flexible, but no row exists past $L_{\max}$, so longer inputs need interpolation or retraining.
- Both extrapolate poorly in practice, since the model never trained on those positions, which motivated relative schemes and RoPE.

**5. Why does sinusoidal encoding use many frequencies and sin-cos pairs?**

Many frequencies give a multi-scale code, like the hands of a clock: fast pairs separate neighbours, slow pairs separate distant positions, and together they stay unique over long ranges. Wavelengths grow geometrically from $2\pi$ toward $10000\cdot 2\pi$. The cos partner makes each pair a point on a circle, so a shift by $\Delta$ is a rotation by a fixed angle, a linear map independent of $p$, and $P_p\cdot P_{p'}$ depends only on $p-p'$. Sine alone could not express a shift linearly. RoPE reuses this frequency schedule.

**6. How would you extend a RoPE model's context beyond its trained length?**

Rescale the rotary angles so long positions land on angles seen in training, then fine-tune briefly on long sequences. Only slow pairs, which never completed a turn in training, meet new angles.
- Position interpolation: divide positions by the extension factor; simple, but crowds neighbours.
- NTK-aware scaling: raise the base, stretching slow pairs while fast pairs, which carry local order, barely change.
- YaRN: leave fast pairs alone, interpolate slow ones, add an attention temperature; little fine-tuning needed.
- In practice: a much larger base plus a final long-sequence stage.

**7. How do T5 relative position bias and ALiBi work?**

Both leave the embeddings alone and add a bias that depends on $p-p'$ to each attention score before the softmax.
- T5: one learned scalar per head for each bucket of relative distance, exact for small offsets and log-spaced for large ones, shared across layers.
- ALiBi: no parameters, a linear penalty $-m_h\lvert p-p'\rvert$ with a fixed slope per head, so distant tokens are down-weighted.
- ALiBi extrapolates well in perplexity, largely because it discounts distant tokens, which can hurt long-range retrieval; RoPE won in modern LLMs.

**8. Does RoPE make attention scores decay with distance?**

Only loosely. With fixed content, the pairs rotate at different speeds and fall out of phase as $\lvert p-p'\rvert$ grows, so an upper bound on the score decays with distance, with oscillations (the RoFormer argument). It is not a hard penalty like ALiBi: the model can still attend far by putting energy in slow pairs, whose angles barely move over long ranges and act almost as position-free channels for semantic matching.

**9. Can a decoder-only model work without any positional encoding (NoPE)?**

Yes, reasonably well, because the causal mask itself leaks position. Position $p$ attends over exactly $p+1$ tokens, so uniform attention gives the first token weight $1/(p+1)$, a signal that encodes $p$, and later layers can build relative position from it. An encoder without a mask cannot do this, since it stays permutation-equivariant. NoPE alone is not standard at scale, but some recent models (Llama 4) interleave NoPE layers with RoPE layers to help long context.

**10. How is RoPE extended to images and video?**

Split the rotary pairs among the coordinates. For an image, half the pairs turn by the row $h$ and half by the column $w$, so the score depends on $(h-h', w-w')$; video adds the frame $t$ (M-RoPE). Any resolution or aspect ratio only changes the coordinate range, so no position-table interpolation is needed. Text tokens get $t=h=w=p$, which reduces to 1D RoPE. Give each axis both fast and slow pairs (interleaved, as in Qwen3-VL), so far-apart frames stay distinguishable in long videos.

## 2.2 Normalization, residuals and training stability

**11. How do BatchNorm, LayerNorm and RMSNorm differ, and where does GroupNorm fit?**

They differ in the axes of the statistics and in whether the mean is subtracted.
- BatchNorm: per feature, over all $B \times n$ tokens in the batch; running averages at inference, where it folds into the previous linear layer.
- LayerNorm: per token, over its own $d$ features; batch-independent, identical at train and inference.
- RMSNorm: LayerNorm's axis, divided by the root mean square, with no mean subtraction and no $\beta$; slightly cheaper at equal quality, the LLaMA-style default.
- GroupNorm: per sample, over groups of channels; batch-independent, the small-batch CNN choice for detection and video.

**12. Why do Transformers use LayerNorm instead of BatchNorm?**

Because BatchNorm makes each token's output depend on the rest of the batch, which breaks on sequences, while LayerNorm uses only the token itself.
- PAD tokens pollute the statistics, and a sentence's output depends on its batch-mates.
- Decoding runs one token at a time at small batch, so batch statistics are unusable and the running averages used instead do not match training.
- Pooling over positions lets later tokens affect earlier ones, leaking the future past the causal mask.
- Batch statistics of text activations fluctuate strongly, which makes training noisy.

<a id="interview-pre-norm"></a>

**13. Why did the original Transformer use post-norm, and why do modern LLMs use pre-norm?**

Post-norm, $x \leftarrow \operatorname{LN}(x + \text{Sublayer}(x))$, the order of the Chapter 3 walkthrough, keeps each layer's output normalized and trained fine at the paper's 6 layers with warmup; most current models use pre-norm, $x \leftarrow x + \text{Sublayer}(\operatorname{LN}(x))$, which trains more stably at depth.
- Post-norm: often slightly better quality, but gradients to early layers cross every LayerNorm above and, at initialization, the gradients near the output are large, so deep stacks need long warmup and can diverge even with it.
- Pre-norm: an identity path from the top layer to the embeddings, so little warmup; the cost is a residual stream that grows with depth, later layers that count less, and a required final norm (DeepNorm and sandwich norm are compromises).
- Pre-norm usually comes with RMSNorm, $x / \sqrt{\operatorname{mean}(x^2)+\epsilon}\cdot\gamma$ with no mean subtraction, and a gated FFN such as SwiGLU, $(\operatorname{SiLU}(xW_1)\odot xW_3)W_2$.
- Do not mix the two orders and call it the same block; see [normalization](#133-normalization).

**14. What do residual connections buy you in a Transformer?**

They give gradients a direct path and let each layer learn only a correction. Since $y = x + F(x)$, the Jacobian is $I + \partial F / \partial x$, so gradients reach early layers even when the sublayers' Jacobians are small; in pre-norm the path is an exact identity, while post-norm puts a LayerNorm on it. A deep stack therefore starts near the identity and stays trainable. In the residual-stream view, every attention and FFN block reads from a shared $d$-dimensional stream and adds its output back.

**15. Why do Transformers need learning-rate warmup?**

Early in training, full-size steps are unreliable and can push the model somewhere it never recovers from.
- Adam's second-moment estimate rests on a few noisy gradients, so some per-parameter steps are far too large.
- Post-norm models have large gradients near the output at initialization, which the full learning rate turns into divergence.
- Warmup ramps the learning rate linearly from near zero over a small fraction of training, often 1 to 3 percent of steps, before the decay.
- Pre-norm needs less warmup but still uses some.

**16. What causes loss spikes in large Transformer training, and how do you fix them?**

Usually an optimization instability, sometimes a bad batch or a numerical overflow.
- Causes: too high a learning rate, attention logits growing until the softmax saturates, drifting output logits, fp16 overflow, corrupt data.
- Lower the peak learning rate or lengthen warmup, clip gradients, use bf16, and set Adam's $\beta_2$ to 0.95 instead of 0.999.
- QK-norm or logit soft-capping bounds attention scores; z-loss, a small penalty on the squared log of the softmax normalizer, keeps output logits in check.
- Recover by restarting from a checkpoint before the spike and skipping the batches around it.

**17. How should a deep Transformer be initialized, and why scale the output projections by depth?**

Use a small Gaussian init, typically standard deviation 0.02, and shrink the last projection of every residual branch. In pre-norm, each of the $2L$ sublayers adds its output to the residual stream, so the stream's variance at initialization grows linearly with depth. Scaling $W^O$ and the FFN's $W_2$ by $1/\sqrt{2L}$, as GPT-2 does, keeps the total at constant scale. The extreme version zero-initializes the last layer or a gate on each branch (Fixup, ReZero, adaLN-Zero), so every block starts as the identity.

**18. Why and how is gradient clipping used when training Transformers?**

It caps rare huge gradients so one bad batch cannot throw the weights far off or corrupt Adam's moment estimates. Compute one global $\ell_2$ norm over all gradients; if it exceeds a threshold, usually 1.0, scale every gradient by the same factor, which keeps the direction. Per-tensor or per-value clipping changes the direction and is avoided. With fp16 loss scaling, unscale before clipping. Frequent clipping points to a learning rate that is too high or to bad data, which clipping only hides.

**19. Why are Transformers trained with Adam rather than SGD?**

Adam normalizes each parameter's step, and a Transformer's parameter blocks have very different gradient scales and curvature. Embeddings, attention projections, norm gains and FFN weights would each need their own SGD learning rate, so one global rate is too large for some blocks and too small for others. Gradient noise in attention models is also heavy-tailed, which per-parameter scaling plus clipping tolerates. In practice SGD trains Transformers much more slowly and to a worse loss. The cost is two extra fp32 states per parameter.

**20. What is the difference between Adam with L2 regularization and AdamW?**

AdamW applies weight decay to the weights instead of adding it to the gradient. With L2 in Adam, $\lambda w$ joins the gradient and is divided by $\sqrt{\hat v}$, so parameters with large gradients are barely decayed. AdamW shrinks $w \leftarrow (1 - \eta \lambda) w$, then takes the usual Adam step on the raw loss gradient, so every weight decays at the same rate, independent of the adaptive scaling. For plain SGD the two coincide up to rescaling $\lambda$. Biases and norm gains are usually excluded from decay.

**21. Where is dropout applied in a Transformer, and why do large LLMs often train without it?**

The original base model applies dropout 0.1 to each sublayer's output before the residual add and to the sum of embeddings and positional encodings; implementations also drop attention weights. Large LLMs pretrain for about one epoch on huge data, so they barely overfit and dropout only slows learning; it is usually set to zero. Dropout still helps when fine-tuning on small datasets. ViT recipes such as DeiT use stochastic depth instead, which drops whole residual branches.

**22. What is label smoothing, and why did the original Transformer use it?**

Label smoothing puts $1 - \varepsilon$ on the correct token and spreads $\varepsilon$ over the rest of the vocabulary. The original Transformer used $\varepsilon = 0.1$: perplexity got worse, because the model learns to be less sure, but accuracy and BLEU improved. It stops logits from growing without bound toward a one-hot target and improves calibration. LLM pretraining skips it: the model would learn to put about $\varepsilon$ of its mass on random tokens, which sampling then emits.

## 2.3 Attention and efficient attention

**23. Why are attention scores divided by the square root of the key dimension?**

To keep the softmax out of saturation. At initialization, if the entries of $q$ and $k$ are independent with zero mean and unit variance, $q^\top k$ is a sum of $d_k$ such products, so its variance is $d_k$. Large scores push the softmax toward one-hot, where its gradients nearly vanish. Dividing by $\sqrt{d_k}$ brings the variance back to 1 at any head width. The divisor uses the per-head width $d_k$, not $d$.

**24. Why use multiple attention heads instead of a single head of full width?**

So that one layer can follow several relations at once. Each head has its own projections and its own softmax, so different heads can attend to different positions for different relations, such as syntax, coreference or the previous token. A single softmax gives one weighted average per query and blurs those relations together. The cost is about the same: $h$ heads of width $d_k = d/h$ use the same $4d^2$ weights and matmul FLOPs as one full-width head; only the number of $n \times n$ score matrices grows to $h$.

**25. What is the time and memory complexity of self-attention, and when does it dominate the FFN?**

The scores cost $O(n^2 d)$ compute per layer and, if materialized, $O(n^2)$ memory per head; everything else is linear in $n$. Multiply-adds per layer:
- Q, K, V and output projections: $4nd^2$.
- FFN: $2nd\,d_{ff}$, which is $8nd^2$ with $d_{ff} = 4d$.
- Scores plus the weighted sum of values: $2n^2 d$.
- So attention passes the FFN at $n > 4d$ and all linear work at $n > 6d$: it matters for long contexts and high-resolution images, not short sentences.

**26. What does the KV cache store, why is it valid, and how large is it?**

It stores every layer's keys and values for all past tokens, so a decode step computes $q$, $k$, $v$ only for the new token. It is valid because, under a causal mask, a token's keys and values depend only on tokens up to it, so they never change; queries are used once, so they are not cached. Per token it holds $2 \cdot L \cdot h_{kv} \cdot d_k$ values (K and V, $h_{kv}$ key/value heads), times bytes per value, tokens and batch. LLaMA-3-8B in BF16 ($L = 32$, $h_{kv} = 8$, $d_k = 128$): 128 KiB per token, 1.07 GB at 8,192 tokens.

**27. What are MQA, GQA and MLA, and what do they save?**

All three shrink the KV cache, cutting memory and the bandwidth of each decode step; query heads stay at $h$ and the per-head attention equation is unchanged.
- MHA, the baseline: $h$ key/value heads, one per query head.
- MQA: one $K, V$ head shared by all query heads; cache $h$ times smaller, some quality loss.
- GQA: one key/value head per group of query heads, $h_{kv}$ in all; the cache shrinks by $h/h_{kv}$ at near-MHA quality, the LLaMA 3 default ([GQA](https://arxiv.org/abs/2305.13245)).
- MLA (DeepSeek-V2/V3): caches one small latent per token and folds the K and V up-projections into the query and output weights; RoPE uses a small separate key, since rotation would block the folding.

<a id="75-efficient-attention"></a>

**28. Is FlashAttention an approximation, and why is it faster than standard attention?**

No, it is an exact algorithm: it computes exactly the softmax attention of [step 2 of Chapter 3](#transformer-math) and cuts memory traffic, not FLOPs. Standard attention writes the $n \times n$ scores and probabilities to HBM and reads them back. FlashAttention tiles $Q$, $K$, $V$ through on-chip SRAM with an online softmax (running max and sum, rescaling the partial output), so the $n \times n$ matrix never reaches HBM. Extra memory drops from $O(n^2)$ to $O(n)$, and because attention is bandwidth-bound, fewer HBM accesses make it much faster, even though the backward pass recomputes the scores.

**29. How do padding and causal masks work, and why use negative infinity instead of zero?**

The mask is added to the scores before the softmax: $0$ where a query may read a key, $-\infty$ where it may not.
- $e^{-\infty} = 0$ gives blocked keys exactly zero weight, and the allowed keys renormalize.
- A blocked score of $0$ would still get weight $e^0 = 1$.
- Padding masks block PAD keys (columns), never queries; the causal mask blocks later keys.
- A fully masked row (a PAD query under left padding with a causal mask) gives NaN, which reaches the gradients even if the row is dropped; use a large finite negative.

**30. What is the difference between self-attention and cross-attention, and where does each appear?**

Self-attention takes queries, keys and values from one sequence; cross-attention takes queries from one sequence and keys and values from another, so the output length follows the queries. In the encoder-decoder Transformer, the encoder uses self-attention with a padding mask, the decoder uses causal self-attention, and each decoder layer's cross-attention lets target queries read the encoder output $H$, an $m \times n$ score matrix. Decoder-only LLMs drop cross-attention; it remains in encoder-decoder models (T5, Whisper), some VLMs (Flamingo) and text-to-image U-Nets (Stable Diffusion).

**31. What are the trade-offs of sliding-window, sparse and linear attention compared with full attention?**

Each cuts the $O(n^2)$ cost by giving up exact attention over all token pairs.
- Sliding window: each query reads only $w$ nearby keys, $O(nw)$ compute and a KV cache capped at $w$ tokens; distant information hops across layers (Mistral, Longformer). Swin does the same for vision with shifted windows.
- Sparse patterns (BigBird, Longformer): a window plus a few global tokens, and random ones in BigBird; global tokens link any pair in two hops, but speed needs block-sparse kernels.
- Linear attention and state-space models (Mamba, RWKV): $O(n)$ with a fixed-size recurrent state, but still weaker on recall-heavy tasks such as exact copying.
- Hybrids that keep a few full-attention layers are the 2025–26 pattern.

**32. Why use separate query, key and value projections instead of the token vectors directly?**

Because the three roles differ: the query says what a token is looking for, the key what it offers for matching, and the value what it passes on once selected. Scores from $XX^\top$ would be symmetric, and with similar norms each token would score itself highest. Separate $W^Q$ and $W^K$ make the score $x_i W^Q (W^K)^\top x_j^\top$, an asymmetric learned bilinear form of rank at most $d_k$ per head. $W^V$ lets the content that is passed on differ from the content used for matching.

**33. Why does the Transformer use dot-product attention instead of additive attention?**

Dot-product attention is much faster and more memory-efficient, and the quality is about the same. All query-key scores come from one matrix multiplication, $QK^\top$, which GPUs run at full throughput, while additive attention passes every query-key pair through a small tanh layer. Additive attention did beat unscaled dot products at large $d_k$, because the dot products grow with $d_k$ and saturate the softmax. Dividing by $\sqrt{d_k}$ is the fix.

**34. Why is the softmax taken over the keys for each query rather than over the queries?**

Because each query's output is a weighted average of the values, and averaging weights must be non-negative and sum to 1 over what is averaged, the keys. That keeps every output on the scale of the values at any length. Normalizing each column over queries would instead give each key a fixed budget shared among queries, so a query's output would depend on how many other queries compete. Under a causal mask it would also leak the future, since each column's sum includes later queries.

**35. What are attention sinks, and why do they matter for streaming or long-context inference?**

They are the first few tokens, which trained LLMs give a large share of attention regardless of content. Softmax weights must sum to 1, so a head with nothing useful to read parks its weight on tokens every query can see, and under a causal mask that is the start. A plain sliding-window cache evicts them and perplexity blows up. StreamingLLM keeps them plus a recent window and stays stable on very long streams, though without long-range memory; some models learn a sink token or per-head sink logit instead.

**36. What are induction heads, and how do they relate to in-context learning?**

An induction head continues repeated patterns: having seen A B earlier, on the next A it predicts B. It takes two heads in different layers. A previous-token head writes each token's predecessor into its position. The induction head then matches the current token against those, attends to the token after the earlier A, and copies it. Induction heads form abruptly during training, at the same moment as a jump in in-context learning. They are the leading mechanistic account of simple in-context learning, best established in small models.

## 2.4 Architecture and modern LLMs

**37. What are encoder-only, decoder-only and encoder-decoder Transformers, and when would you use each?**

They differ in which tokens each position may attend to and in the training objective.
- Encoder-only (BERT, RoBERTa): bidirectional self-attention, masked-token objective; best for understanding tasks such as classification, retrieval and embeddings.
- Decoder-only (GPT, LLaMA, Qwen): causal mask, next-token prediction; the default for generation and general-purpose LLMs.
- Encoder-decoder (original Transformer, T5, BART, Whisper): bidirectional encoder and causal decoder joined by cross-attention; best when input and output are distinct sequences, as in translation, summarization or speech recognition.

**38. What does a modern decoder-only LLM look like, and how does it differ from the 2017 encoder-decoder Transformer?**

It keeps the 2017 skeleton and changes details, not structure: the prompt and the generated text form one causal token stream, so there is no encoder and no cross-attention ([steps 1–4 and 7 of Chapter 3](#transformer-math) disappear) and every other step keeps its role. Tokens pass through the embedding, then $L$ layers that each apply RMSNorm, attention with GQA and RoPE, and a residual add, then RMSNorm, a SwiGLU FFN and a residual add; a final RMSNorm and an LM head give the logits; the head is tied to the embedding in GPT-2 and many small models but untied in most large ones. The details that changed inside those steps: pre-norm RMSNorm instead of post-norm LayerNorm, RoPE instead of the added $P$, GQA to shrink the KV cache, SwiGLU instead of the ReLU FFN, and often MoE in place of the dense FFN. The math, step by step, is in [Decoder-only LLMs: GPT-2](#decoder-only-llms).

**39. Why do almost all current LLMs use a decoder-only architecture?**

Next-token prediction is one simple objective that scales on unlimited raw text and yields a generator.
- Every token is a training signal, whereas masked LM predicts only about 15%.
- Any task can be phrased as text continuation, which enables in-context learning and zero-shot use.
- Under a causal mask past keys and values never change, so the KV cache is exact and a new token computes only its own position.
- One stack without cross-attention is simpler to scale and serve, and at scale it matches encoder-decoder quality.

**40. How many parameters does a Transformer layer have, and how much compute does training cost?**

A standard dense layer has about $12d^2$ parameters, a number worth being able to estimate on a whiteboard: $4d^2$ for $W^Q, W^K, W^V, W^O$ plus $8d^2$ for an FFN with $d_{ff} = 4d$. A model therefore has about $12Ld^2$ plus the $Vd$ embedding table; a decoder layer with cross-attention adds another $4d^2$. For LLaMA 2 7B, $L = 32$ and $d = 4096$ give $12Ld^2 \approx 6.4\text{B}$, plus $0.26\text{B}$ for its two untied embedding tables, 6.7B in all. Training costs about 6 FLOPs per parameter per token, 2 forward and 4 backward, so about $6PT$ for $P$ parameters and $T$ tokens; inference costs about $2P$ per token. Both ignore the attention-score term, which matters mainly at long context.

**41. What does the FFN do in a Transformer layer, and why is its hidden width four times the model width?**

Attention moves information between tokens; the FFN transforms each token on its own, with the same weights at every position. It widens to $d_{ff}$, applies a nonlinearity and projects back to $d$ so the residual addition works, and interpretability work describes it as a key-value memory that holds much of the model's factual knowledge. The ratio $d_{ff} = 4d$ is an empirical convention from the original paper (512 to 2048), not a derived optimum; with it the FFN holds two thirds of a layer's parameters, $8d^2$ of $12d^2$.

**42. How do ReLU, GELU and SiLU differ, and why did Transformers move from ReLU to GELU?**

All three pass large positive inputs almost unchanged and differ near zero and below it.
- ReLU, $\max(0, x)$: a hard cutoff with gradient 0 or 1, the cheapest; units stuck negative get no gradient and die.
- GELU, $x\,\Phi(x)$ with $\Phi$ the standard normal CDF, is the expected output of keeping $x$ with probability $\Phi(x)$: smooth, with a minimum of $-0.17$ at $x \approx -0.75$ and a gradient for small negative inputs.
- SiLU (Swish), $x\,\sigma(x)$, is the same shape rescaled, as $\operatorname{GELU}(x) \approx x\,\sigma(1.702x)$; it needs one sigmoid and dips lower, to $-0.28$.

The switch was empirical: the GELU paper showed small gains over ReLU, and GPT, BERT, GPT-2, GPT-3 and ViT adopted it.

**43. What does the gate in a gated FFN do, and why do modern LLMs use SwiGLU or GEGLU?**

A gated FFN projects each token twice and multiplies the results elementwise: $\operatorname{FFN}(x) = (\operatorname{act}(xW_1) \odot xW_3)\,W_2$.
- Each hidden unit multiplies two different projections of $x$, an input-dependent scaling that a plain $\operatorname{act}(xW_1)\,W_2$ lacks.
- The original GLU (Dauphin et al., 2017) used a sigmoid gate in $(0, 1)$; SiLU and GELU gates are signed and unbounded.
- The name gives the gate: SwiGLU uses SiLU (LLaMA, PaLM, Qwen, Mistral) and GEGLU uses GELU (Gemma).

In Shazeer's 2020 T5 tests at equal parameters and FLOPs, GEGLU and SwiGLU did best, at log-perplexity 1.633 and 1.636 against 1.677 for ReLU, while plain GELU and Swish were no better than ReLU: the gain comes from the gating, which the paper leaves unexplained.

**44. Why is a SwiGLU FFN's hidden width often about 8/3 of the model width rather than four times?**

To keep the FFN's parameters and FLOPs equal to the standard FFN despite its third matrix. With $W_1$, $W_3$ and $W_2$, $d_{ff} = 4d$ would give the FFN alone $12d^2$ parameters instead of $8d^2$; $d_{ff} = \tfrac{8}{3}d$ gives $3 \cdot \tfrac{8}{3}d \cdot d = 8d^2$. That is the cut by $\tfrac{2}{3}$ Shazeer made for a fair comparison, 3072 to 2048 at $d = 768$.
- LLaMA-7B: $\tfrac{8}{3} \cdot 4096 \approx 10923$, rounded up to a multiple of 256, gives $d_{ff} = 11008$.
- The ratio is a choice, not a rule: PaLM kept $4d$ and paid $12d^2$, while LLaMA 3 8B and Mistral 7B use $3.5d$, 14336 at $d = 4096$.

**45. What is a Mixture of Experts layer, and what does it trade off?**

MoE replaces the FFN with $E$ expert FFNs and a router that sends each token to its top-$k$ experts, mixing their outputs by gate score.
- Parameters grow with $E$ while FLOPs per token do not, since only the active experts run, so quality per FLOP improves.
- $k$ is usually 1 or 2 (Switch, Mixtral); fine-grained designs such as DeepSeek-V3 and Qwen3-MoE route each token to more, smaller experts.
- Without a load-balancing auxiliary loss (or bias), the router collapses onto a few experts.
- Costs: memory still holds every expert, expert parallelism needs all-to-all communication, and training is less stable.

<a id="92-tokenization"></a>

**46. How do LLM tokenizers work, what does the vocabulary size trade off, and what problems do they cause?**

Most LLMs use BPE, which starts from characters and repeatedly merges the most frequent adjacent pair into a new token; byte-level BPE (GPT-2 onward) starts from the 256 byte values, so nothing is ever out of vocabulary.
- SentencePiece's unigram model is the probabilistic, language-agnostic alternative.
- Vocabulary size: a larger one gives fewer tokens per text, especially for non-English text and code, hence less compute and more content per context window, at the cost of a larger $Vd$ embedding and output layer and rare tokens with few updates; LLaMA grew from 32k (LLaMA 2) to 128k (LLaMA 3).
- Consequences: LLMs are weak at character-level tasks (counting letters, reversing strings) and at arithmetic, because numbers split inconsistently.
- Non-English text costs 2 to 4 times more tokens, and prompt length in tokens is not length in words.

**47. Why did Transformers replace RNNs, and what did they give up?**

Transformers train in parallel over the sequence, and any two tokens interact within one layer.
- An RNN's state at step $t$ needs step $t-1$, so even with teacher forcing it runs $n$ sequential steps; a Transformer computes all positions in one pass.
- The path between any two tokens has constant length, so long-range gradients do not vanish through time.
- The price: $O(n^2 d)$ attention per layer, a KV cache that grows with context, and explicitly injected positions, while an RNN keeps a fixed-size state and constant cost per token.

<a id="76-cnn-vs-transformer"></a>

**48. How do CNNs and Transformers compare, and when would you pick each?**

Pick a CNN when data, compute or edge hardware is limited and a Transformer when data is plentiful and long-range context matters; the root difference is inductive bias. Trained on ImageNet-1k alone, ViT-B/16 trails ResNets of its size, yet pretrained on JFT-300M it matches or beats the best CNNs for less pretraining compute.

| | CNN | Transformer |
|--|-----|-------------|
| Inductive bias | strong: local, translation-equivariant | weak, must be learned |
| Long-range interaction | limited, grows with depth | direct, in one layer |
| Data need | lower | higher, or augment + distill (DeiT) |
| Cost | efficient, hardware-friendly | $O(n^2)$ attention, memory-hungry |
| Edge deployment | better | heavier, though improving |
| Low-level vision | strong on local texture | window attention works well |

Hybrids win in practice: ConvNeXt showed a CNN with transformer-style design choices matches ViTs, and Swin showed local attention matches global at a fraction of the cost. For dense prediction and video at deployment scale, a convolutional stem plus windowed attention is the pragmatic default.

**49. How does a Vision Transformer turn an image into tokens?**

It cuts the image into non-overlapping $P \times P$ patches, flattens each one and maps it with one linear layer to width $d$, which equals a convolution with kernel and stride $P$. A learned class token is prepended and a learned position table is added: ViT-B/16 turns a 224 by 224 image into 196 patch tokens plus the class token. From there it is the standard Transformer encoder with no mask; at a new resolution the position table must be interpolated.

**50. How does BERT's masked language modeling work, and why the 80/10/10 masking rule?**

BERT selects 15% of the tokens and predicts each one from context on both sides with a bidirectional encoder. Of the selected tokens, 80% become a mask token, 10% a random token and 10% stay unchanged. The reason: the mask token never appears at fine-tuning, and because the model cannot tell which inputs were altered, it must keep a good representation of every token. The cost is that only the selected 15% give a loss, so MLM learns less per token than next-token prediction.

**51. Why tie the input embedding and the output projection weights, and when do models not tie them?**

Both map between token ids and the $d$-dimensional space, so one $V \times d$ matrix can serve as the input embedding and, transposed, as the output projection. Tying saves $Vd$ parameters, a large share of a small model, and regularizes: every softmax gradient also updates the input embeddings, including those of rare tokens. The original Transformer and GPT-2 tie them. Many large models, such as LLaMA 2 and LLaMA 3 8B and 70B, do not: $Vd$ is then a small fraction, and separate matrices let the input and output roles specialize.

**52. Why does the original Transformer multiply token embeddings by the square root of d?**

So that token and position signals have comparable scale when they are added. The embedding matrix is shared with the output projection, so it is initialized at scale $1/\sqrt{d}$. Multiplying by $\sqrt{d}$ raises its rows to norm about $\sqrt{d}$, comparable to the sinusoidal rows of $P$, whose norm is $\sqrt{d/2}$. It is a convention, not a necessity: with a unit-variance initialization it is not needed, and the first LayerNorm rescales anyway. GPT-2 and LLaMA omit it; Gemma keeps it.

## 2.5 Training and inference at scale

**53. Compare greedy decoding, beam search, temperature, top-k and top-p sampling; when does beam search hurt?**

Greedy takes the arg max at each step; beam search keeps the best few partial sequences by total log-probability. Temperature divides the logits by $T$, sharpening below one and flattening above; top-$k$ keeps the $k$ most likely tokens and top-$p$ the smallest set whose mass reaches $p$, then renormalizes. Beam search suits tasks with one right answer, such as translation or speech recognition. It hurts open-ended generation, where the most probable text is bland and repetitive, and without length normalization it favors short outputs.

**54. What is teacher forcing, and what is exposure bias?**

Teacher forcing feeds the decoder the true prefix: the target shifted right is the input, a causal mask hides the future, and all positions are trained in one parallel forward pass. Exposure bias is the resulting train-test mismatch: at inference the model conditions on its own outputs, which it never saw in training, so an early error can compound. Mitigations train on the model's own samples, such as scheduled sampling, sequence-level losses and RL fine-tuning.

**55. What do scaling laws say, and why are modern LLMs trained far past the Chinchilla-optimal point?**

Loss falls as a power law in parameters, data and compute, so small runs predict how to spend a large budget. Chinchilla found that at fixed training compute, parameters and tokens should grow together, about 20 tokens per parameter, an empirical point rather than a constant. That optimum ignores inference: a smaller model trained on far more tokens costs extra training compute but is cheaper to serve, and its loss keeps improving. So models are over-trained by design, such as LLaMA 3 8B on about 15T tokens.

**56. How much GPU memory does training an LLM need, and what takes it up?**

About 16 bytes per parameter for model state with Adam in mixed precision, before activations.
- bf16 weights and gradients: 2 + 2 bytes.
- fp32 master weights and the two Adam moments: 4 + 4 + 4 bytes.
- So a 7B model needs about 112 GB, more than one 80 GB GPU.
- Activations grow with batch, sequence length, $d$ and $L$ and often dominate at long context; gradient checkpointing keeps only layer inputs and recomputes the rest, for about one extra forward pass.
- Inference needs about 2 bytes per parameter in bf16, plus the KV cache.

**57. Why is LLM decoding memory-bandwidth-bound while prefill is compute-bound?**

Arithmetic intensity, the FLOPs done per byte moved. Prefill pushes all $n$ prompt tokens through each weight matrix at once, reusing each weight read from HBM $n$ times: about $n$ FLOPs per byte in BF16, far above the GPU's ridge point (about 300 on an H100). A decode step makes one token yet streams all weights and the KV cache, about 1 FLOP per byte, so its time is bytes over bandwidth. Hence batching raises decode throughput cheaply, and quantization, GQA and speculative decoding help by cutting or amortizing bytes.

**58. How do data, tensor and pipeline parallelism differ, and what do ZeRO and FSDP add?**

Each one splits something different across GPUs, and large runs combine all of them.
- Data parallel: every GPU holds a full replica and a slice of the batch; gradients are all-reduced every step.
- ZeRO / FSDP: data parallel that also shards optimizer states, gradients and weights, and gathers each layer's weights just before use.
- Tensor parallel: splits each matrix multiply (heads, FFN columns) across GPUs. It communicates inside every layer, so it stays within one node on fast links.
- Pipeline parallel: puts consecutive layers on different GPUs and streams micro-batches to shrink the idle bubble.

**59. What is the difference between fp16 and bf16, and when do you need loss scaling?**

Both use 16 bits: bf16 has 8 exponent and 7 mantissa bits, fp16 has 5 and 10. fp16 is more precise but has a narrow range, so small gradients underflow to zero and large activations can overflow. Loss scaling fixes the underflow: multiply the loss by a large factor before backward, unscale before the optimizer step, and on an inf or NaN skip the step and lower the factor. bf16 has fp32's range and needs no scaling, hence the default on A100, H100 and TPU. Master weights and optimizer states stay fp32.

**60. How does speculative decoding work, and why does it preserve the target model's distribution?**

A small draft model proposes $k$ tokens and the target model scores them all in one parallel forward pass. Each draft token $x$ is accepted with probability $\min(1, p(x)/q(x))$, with $p$ the target and $q$ the draft distribution; at the first rejection, a replacement is sampled from the normalized $\max(0, p - q)$ and the rest are discarded. This rejection-sampling rule makes every output token exactly distributed as $p$. It is faster because decoding is memory-bound, so verifying $k$ tokens costs about as much as generating one.

**61. What is perplexity, and how does it relate to the cross-entropy loss?**

Perplexity is the exponential of the average per-token cross-entropy in nats, so a loss of 2.0 means a perplexity of about 7.4. Read it as the effective number of equally likely next tokens the model is choosing among; a perfect model scores 1. Lower loss always means lower perplexity. But perplexity is comparable only between models with the same tokenizer and test set. Across tokenizers, compare bits per byte instead.

## 2.6 Interview Questions — Computer Vision

English Q&A adapted from the [shared CV conversation](https://chatgpt.com/share/6ae66ad9-9850-83e8-a381-1a1210d25df7): 50 core questions, five modeling design cases, and their follow-ups. The latest CNN-versus-fully-connected explanation is included under question 62. These are practice prompts, not verified Roblox interview reports.

For full 45-minute modeling interviews, open the [CV design practice pack](ml_modeling_practice/roblox_cv/index.html): visual safety moderation, game genre classification, and multimodal asset retrieval. Each has separate question and reference-answer documents, an end-to-end model flow, equations, tensor shapes, and 21 basic follow-ups.

**Computer vision and deep learning fundamentals**

<a id="cv-q01"></a>

**62. Why do CNNs use convolution instead of fully connected layers?**

Convolution builds in local connectivity, spatial weight sharing, and translation equivariance. These image-friendly inductive biases reduce parameters and often improve data efficiency compared with a dense pixel-level MLP. Stacking layers expands the receptive field from edges and textures to larger structures.

<details>
<summary>Detailed CNN versus fully connected example</summary>

**Dense layer versus convolution.** Flattening an RGB image gives

$$
X\in\mathbb R^{224\times224\times3}
\longrightarrow x\in\mathbb R^{150528}.
$$

A dense layer with 1,024 outputs computes $y=Wx+b$, where $W\in\mathbb R^{1024\times150528}$, requiring

$$
150528\times1024+1024=154141696\text{ parameters}.
$$

A $3\times3$ convolution with 3 input channels and 64 output channels requires

$$
3\times3\times3\times64+64=1792\text{ parameters}.
$$

With stride 1 and padding 1, its output is $224\times224\times64$, not a 1,024-dimensional vector: these are **different output shapes, not an apples-to-apples comparison**. The example illustrates that convolution parameters do not grow with image width or height. **Parameter count is not FLOPs**: the kernel is stored once but evaluated at many locations.

**Local connectivity and shared kernels.** For a grayscale image, a simple edge filter is

$$
K=\begin{bmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{bmatrix},\qquad
Y_{i,j}=\sum_{u=-1}^{1}\sum_{v=-1}^{1}K_{u,v}X_{i+u,j+v}.
$$

This measures left-to-right intensity changes, highlighting vertical boundaries. Deep-learning libraries generally implement this cross-correlation operation, conventionally called convolution. Each output uses only a local neighborhood, and the **same $K$** is applied everywhere. A learned detector therefore benefits from examples at different image locations rather than needing separate parameters for every location.

Two consecutive $3\times3$, stride-1, dilation-1 layers have receptive fields $3\times3\rightarrow5\times5$. Deeper layers combine local features into larger structures; this does not mean every learned channel follows a fixed edge-to-object progression.

Flattening does **not destroy pixel information**: fixed vector indices still identify the original positions. A dense layer simply does not enforce locality or spatial weight sharing; it must learn useful spatial relationships from data.

**Equivariance versus invariance.** For translation operator $T_\Delta$:

$$
\underbrace{f(T_\Delta X)=T_\Delta f(X)}_{\text{equivariance: features move}},
\qquad
\underbrace{g(T_\Delta X)=g(X)}_{\text{invariance: prediction stays unchanged}}.
$$

Stride-1 convolution is translation equivariant under appropriate boundary assumptions. Padding/cropping at boundaries and strided downsampling can break exact equivariance. Pooling or global aggregation can encourage invariance in classification, but arbitrary finite-image translations are not automatically handled perfectly.

**Short spoken answer.** “Convolution exploits local pixel relationships, shares one detector across image locations, and makes feature maps translation equivariant. These structural assumptions reduce parameters and often help with limited data. A dense MLP can learn image functions too, but it does not build in those spatial priors.”

**Follow-up: If CNNs are efficient, why use Vision Transformers?** Self-attention directly connects distant patches through content-dependent weights. ViTs benefit from large-scale pretraining and flexible global interactions, but standard global attention is costly at high resolution.

**Follow-up: Does a CNN provide translation invariance or equivariance?** Convolution principally provides equivariance; a classification head may encourage invariance through global pooling. They are not interchangeable.

**Follow-up: Why not use a huge MLP with unlimited compute and data?** An expressive MLP can approximate complex visual functions. CNNs still offer an efficient parameterization and useful spatial priors; unlimited resources do not establish that either architecture universally wins.

</details>

<a id="cv-q02"></a>

**63. What is the difference between convolution and self-attention?**

Convolution applies shared, learned local kernels; their weights do not depend on the current input content. Self-attention computes content-dependent interactions:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V.
$$

Global attention connects every token pair in one layer, whereas local convolutions expand their receptive field across layers. Its core attention cost is $O(N^2d)$; convolution cost is approximately $O(HWk^2C_{in}C_{out})$. Projection/MLP costs and hardware also matter. CNNs impose stronger local spatial priors; attention offers more flexible interactions.

<a id="cv-q03"></a>

**64. Explain ResNet. Why do residual connections help?**

A residual block learns a correction to an identity path:

$$
y=x+F(x),\qquad
\frac{\partial L}{\partial x}
=\left(I+J_F(x)\right)^\top\frac{\partial L}{\partial y}.
$$

The identity term supplies a direct gradient path, making deep networks easier to optimize and allowing a block to approximate identity by learning $F(x)\approx0$. It does not guarantee gradients never vanish or explode. If dimensions change, use a projection shortcut $y=P(x)+F(x)$.

<a id="cv-q04"></a>

**65. BatchNorm vs LayerNorm — what's the difference?**

Both use $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ followed by learned scale and shift. For CNN activations $[B,C,H,W]$, BatchNorm usually computes each channel's training statistics over $(B,H,W)$ and uses running statistics at evaluation. LayerNorm normalizes each example over specified feature dimensions; for Transformer tensors $[B,N,d]$, typically over $d$ independently at each token. BN can be sensitive to small or nonrepresentative batches; LN does not depend on other examples, which suits variable-length token processing.

<a id="cv-q05"></a>

**66. Why does Vision Transformer divide images into patches?**

Patches convert an image into a manageable token sequence:

$$
[B,C,H,W]\rightarrow[B,N,P^2C]\rightarrow[B,N,d],
\qquad N=(H/P)(W/P).
$$

For $224\times224$ inputs, $P=16$ gives $N=196$; $P=8$ gives $N=784$. Ignoring a possible CLS token, halving patch size yields four times as many tokens and **16 times the quadratic attention term**, not necessarily 16 times total runtime. Smaller patches preserve finer detail. High-resolution options include window attention, hierarchical features, token pruning/merging, and efficient attention variants.

<a id="cv-q06"></a>

**67. How does positional encoding work in ViT?**

Self-attention alone is permutation equivariant and does not identify spatial order. Add positional information to patch embeddings:

$$
Z_0=X_{patch}W_E+E_{pos},\qquad E_{pos}\in\mathbb R^{N\times d}.
$$

Learned or sinusoidal absolute embeddings can encode grid locations; relative position biases encode relationships between locations. A flattened patch grid can still represent two-dimensional position. Absolute embeddings often need grid interpolation when resolution changes.

<a id="cv-q07"></a>

**68. What are the advantages and limitations of ViT compared with CNN?**

ViTs flexibly model global dependencies and often benefit greatly from large-scale pretraining. Standard global attention becomes expensive as resolution increases, and weaker local priors can increase data/training requirements. CNNs efficiently exploit local structure and naturally produce multiscale features. Performance depends on dataset, training recipe, architecture, and deployment constraints; neither universally dominates.

<a id="cv-q08"></a>

**69. How would you handle variable image resolutions?**

Choose resize/crop for fixed-size inputs, or aspect-ratio-preserving resize plus padding when distortion or cropping is harmful. Use masks where padding would otherwise affect computation. Multiscale training/features and high-resolution crops preserve small-object detail. For ViTs, adapt absolute position embeddings by interpolating the patch grid. Explain the accuracy, memory, aspect-ratio, and batching trade-offs.

<a id="cv-q09"></a>

**70. What is transfer learning? When should the backbone be frozen?**

Reuse pretrained representations for a new task. A frozen backbone plus a trained head is a linear probe when the head is linear; partial or full fine-tuning adapts more features. Start frozen when labels/compute are limited or representations already transfer well. Consider unfreezing with sufficient data or substantial domain mismatch, often using a smaller backbone learning rate. Select the strategy on validation performance and monitor overfitting or forgetting.

<a id="cv-q10"></a>

**71. What is the difference between classification, detection, and segmentation?**

Classification predicts image-level labels, e.g. $[B,K]$ logits. Detection predicts a variable-size set of class labels and bounding boxes. Semantic segmentation predicts a class for each pixel, often $[B,K,H,W]$ logits; instance segmentation also separates individual objects with instance masks. Choose image classification if presence/absence is sufficient, detection for object locations, and segmentation when precise boundaries are required.

**Training, Optimization, and Loss Functions**

<a id="cv-q11"></a>

**72. Why use cross-entropy instead of MSE for image classification?**

Cross-entropy is the negative log-likelihood of a Bernoulli or categorical model, so it directly rewards probability assigned to the correct class. For binary labels, use a sigmoid probability $p$; for mutually exclusive classes, use softmax probabilities $p_k$:

$$
\mathcal L_{\mathrm{BCE}}=-[y\log p+(1-y)\log(1-p)],
\qquad
\mathcal L_{\mathrm{CE}}=-\sum_{k=1}^{K}y_k\log p_k.
$$

MSE on probabilities is possible, but cross-entropy with sigmoid/softmax has useful likelihood and gradient properties; for a softmax logit, its gradient is $p_k-y_k$. Independent multi-label classes normally use one sigmoid and a BCE term per class, rather than one softmax.

<a id="cv-q12"></a>

**73. When would you use focal loss?**

Focal loss is useful when numerous easy examples dominate training, such as background candidates in dense detection. It reduces their contribution while preserving more weight for difficult examples:

$$
\mathcal L_{\mathrm{focal}}=-\alpha_t(1-p_t)^\gamma\log p_t,
\quad p_t=\begin{cases}p,&y=1\\1-p,&y=0.\end{cases}
$$

Here $\gamma>0$ controls the focusing effect, and $\alpha_t$ can balance classes; it does not remove the need to inspect noisy labels and evaluate minority-class performance.

<a id="cv-q13"></a>

**74. What is label smoothing, and why does it work?**

Replace a one-hot target with a slightly softened distribution:

$$
q_k=(1-\varepsilon)\mathbf 1[k=y]+\frac{\varepsilon}{K},
\qquad \mathcal L=-\sum_k q_k\log p_k.
$$

This discourages driving the true-class logit arbitrarily far above all others and can regularize training. Its effects on accuracy and calibration depend on the setting; smoothing is not a guarantee of well-calibrated probabilities.

<a id="cv-q14"></a>

**75. Adam vs SGD with momentum — which would you choose?**

Adam uses running gradient moments to adapt updates per parameter; AdamW additionally decouples weight decay. SGD with momentum uses a shared learning rate and an accumulated update direction, and can work very well for CNN training with an appropriate schedule. I would compare validation quality, tuning cost, convergence, and optimizer-state memory rather than claim that one always generalizes better.

<a id="cv-q15"></a>

**76. Why is weight decay useful? Is it the same as L2 regularization?**

Weight decay shrinks parameters and can help control overfitting. For ordinary SGD, an L2 penalty $\frac{\lambda}{2}\|w\|^2$ gives

$$
w\leftarrow (1-\eta\lambda)w-\eta\nabla_w\mathcal L_{\mathrm{data}},
$$

which is the corresponding multiplicative decay update. With adaptive optimization, adding $\lambda w$ to the gradient changes the adaptive statistics too; AdamW instead applies decay separately, so the two procedures are not generally equivalent.

<a id="cv-q16"></a>

**77. How would you address overfitting in a vision model?**

First check a genuine training–validation gap, label quality, duplicate leakage, and whether the split reflects deployment. Then consider task-preserving augmentation, weight decay, early stopping, pretrained features, a smaller model, or additional representative data. Change a controlled set of factors and evaluate both aggregate and important slice metrics.

<a id="cv-q17"></a>

**78. How does contrastive learning work, and what is InfoNCE?**

Contrastive learning brings compatible representations closer and separates alternatives. For paired image/text embeddings $(v_i,t_i)$, image-to-text InfoNCE treats the paired text as the positive and other batch texts as negatives:

$$
\mathcal L_i^{I\to T}
=-\log\frac{\exp(\operatorname{sim}(v_i,t_i)/\tau)}
{\sum_{j=1}^{B}\exp(\operatorname{sim}(v_i,t_j)/\tau)}.
$$

Here $B$ is batch size and $\tau$ is temperature; CLIP-style training also includes the reverse text-to-image loss. Larger candidate sets provide more negatives, but their semantic quality matters as much as their count.

- **What happens when temperature becomes very small?** The softmax concentrates on the most similar candidates, emphasizing hard mistakes; overly small values can amplify noisy negatives and make optimization unstable.
- **Why is L2 normalization applied to embeddings?** With unit vectors, dot product equals cosine similarity, so increasing vector norms cannot improve scores; temperature separately controls score sharpness.
- **What happens if two different samples have the same semantic meaning?** They may be false negatives, and forcing them apart damages the representation; deduplication, multi-positive objectives, or masking known matches can help.
- **How would you build hard negatives for visual search?** Retrieve visually similar but genuinely nonmatching items, then verify relevance using labels or review; exclude duplicates and alternative valid matches before training.

<a id="cv-q18"></a>

**79. How does negative sampling affect training?**

Random negatives provide broad coverage but may become too easy; hard negatives teach finer distinctions but can include false negatives. In-batch negatives are efficient because embeddings are reused, while mined negatives can target current retrieval errors. Mix negative types and check whether improvements persist on a representative evaluation set rather than only a curated hard set.

<a id="cv-q19"></a>

**80. What data augmentation strategies are useful for CV?**

Common choices include random crops, flips, color jitter, MixUp, and CutMix. Transformations must preserve the task: a crop can remove a safety-relevant object, and a flip can alter text or direction labels. Detection boxes and segmentation masks must undergo the same geometry, while MixUp/CutMix require appropriate target mixing.

<a id="cv-q20"></a>

**81. How do you handle severe class imbalance?**

Use class weighting, resampling, or focal loss when justified, and inspect precision–recall behavior and per-class recall rather than relying on accuracy. Select a threshold using validation data to reflect false-positive/false-negative costs and review capacity. Reweighting or resampling changes the effective training distribution, so separately check calibration and the class prevalence expected in deployment.

**Classical CV Tasks and Model Design**

<a id="cv-q21"></a>

**82. How would you design an image classification system from scratch?**

Define whether labels are mutually exclusive or multi-label, annotation rules, evaluation slices, and deployment constraints. Split by the relevant user, asset, or time boundary, then start with a pretrained backbone and a small head: $I\to f_\theta(I)\in\mathbb R^d\to Wf_\theta(I)+b\in\mathbb R^K$. Train with CE or per-class BCE as appropriate, evaluate per-class metrics and calibration, and measure serving latency.

**Detection design follow-up: How would you detect inappropriate objects in Roblox screenshots?** First ask whether an image-level decision is sufficient or whether boxes/masks are required; do not choose YOLO before defining the output. For localization, compare detectors or segmenters while explicitly handling small objects, occlusion, low resolution, and different rendering styles. Set thresholds against missed violations, false accusations, and human-review cost, then evaluate those failure slices.

<a id="cv-q22"></a>

**83. What is the difference between one-stage and two-stage object detectors?**

One-stage detectors such as YOLO predict classes and locations directly from image features. Two-stage detectors such as Faster R-CNN first propose candidate regions, then classify and refine them. Compare measured accuracy, latency, object sizes, and deployment hardware; two-stage is not universally more accurate, nor is one-stage universally the right choice.

<a id="cv-q23"></a>

**84. What are IoU and Non-Maximum Suppression (NMS)?**

Intersection over Union measures overlap between predicted and reference regions:

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

NMS sorts candidate boxes by confidence, keeps a high-scoring box, and suppresses sufficiently overlapping alternatives, usually within a class. Its threshold trades duplicate removal against suppressing distinct nearby objects, especially in crowded scenes.

<a id="cv-q24"></a>

**85. How do you evaluate object detection?**

Evaluate confidence-ranked predictions using class labels and one-to-one matching to ground-truth boxes at a specified IoU threshold. A duplicate prediction or a poorly localized box can count as a false positive, so detection AP is not merely classification AP on image labels. Report mAP across classes and, where specified, IoU thresholds, together with size-specific recall and latency; explicitly distinguish AP50 from COCO-style AP averaged over IoU 0.50–0.95.

<a id="cv-q25"></a>

**86. What is the difference between semantic, instance, and panoptic segmentation?**

Semantic segmentation assigns each pixel a class but does not distinguish two objects of the same class. Instance segmentation produces a separate mask for each countable object. Panoptic segmentation combines both: every pixel receives a semantic label, and countable “thing” objects also receive instance identities, while “stuff” such as sky is not split into object instances.

<a id="cv-q26"></a>

**87. How would you handle small objects in high-resolution images?**

Avoid shrinking the entire image until relevant objects disappear: consider higher resolution, overlapping tiles/crops, and multi-scale features. Match the detection head's feature stride and training data to expected object sizes. Measure the added compute and merge detections across overlapping tiles without double-counting objects.

<a id="cv-q27"></a>

**88. What is Feature Pyramid Network (FPN)?**

FPN combines semantically rich, low-resolution backbone features with finer-resolution features through top-down upsampling and lateral connections. A typical pyramid has $P_3:[B,d,H/8,W/8]$, $P_4:[B,d,H/16,W/16]$, and $P_5:[B,d,H/32,W/32]$. Heads use these scales to detect differently sized objects, giving small objects finer spatial support without discarding high-level semantics.

<a id="cv-q28"></a>

**89. How would you build an image similarity search engine?**

Encode images into normalized embeddings, store them with asset metadata in an approximate-nearest-neighbor index, and retrieve candidates using cosine similarity or equivalent unit-vector distance. Optionally apply filters and a more expensive reranker to the shortlist. Evaluate Recall@K against relevance judgments, serving latency, index freshness, and representation drift; separate visual similarity from the product's actual relevance definition.

<a id="cv-q29"></a>

**90. How would you detect out-of-distribution images?**

Consider embedding distance from training examples, calibrated uncertainty, or energy-based scores, and choose thresholds using representative in-domain and OOD validation data. Include unfamiliar rendering styles, corrupted inputs, and new content categories, then evaluate OOD recall versus false alarms on normal traffic. High maximum softmax probability does not guarantee an input is familiar; uncertain cases may require abstention or review.

<a id="cv-q30"></a>

**91. What is the difference between zero-shot and fine-tuned classification?**

In CLIP-style zero-shot classification, encode class descriptions and compare an image embedding with those text embeddings without training on labeled examples from the target task. Fine-tuning updates some or all model parameters using target-domain data; a frozen encoder with a trained linear head is another useful baseline. Compare domain accuracy, annotation cost, compute, and robustness, and keep evaluation data separate from prompt and hyperparameter selection.

**VLM and multimodal learning**

<a id="cv-q31"></a>

**92. How is CLIP trained from scratch?**

Train an image encoder and a text encoder on paired images and captions, projecting both outputs into a shared embedding space. Normalize embeddings and use symmetric image-to-text and text-to-image contrastive losses so matching pairs receive higher similarity than mismatches; semantically related in-batch examples can be false negatives.

$$
v_i=\frac{f_I(I_i)}{\|f_I(I_i)\|_2},\qquad
t_j=\frac{f_T(T_j)}{\|f_T(T_j)\|_2},\qquad
s_{ij}=\frac{v_i^\top t_j}{\tau},\qquad
\mathcal L_{\mathrm{CLIP}}=\tfrac12(\mathcal L_{I\to T}+\mathcal L_{T\to I}).
$$

<a id="cv-q32"></a>

**93. What is the difference between CLIP and a generative VLM?**

CLIP is a dual-encoder representation model: similarity supports retrieval and text-prompt-based classification, but CLIP itself does not generate answers. A generative VLM conditions a language model on visual information and produces text autoregressively; generation adds flexibility but also decoding cost and hallucination risk.

<a id="cv-q33"></a>

**94. How does a vision-language model connect the visual encoder to an LLM?**

A common architecture projects visual patch features into the LLM hidden dimension, then combines visual and text embeddings for conditional generation. A resampler or Q-Former can instead extract a smaller set of visual tokens, while other architectures introduce cross-attention.

$$
I\xrightarrow{\text{vision encoder}}Z_v\in\mathbb R^{B\times N_v\times d_v},\qquad
H_v=Z_vW_p\in\mathbb R^{B\times N_v\times d_{\mathrm{LLM}}},\quad
W_p\in\mathbb R^{d_v\times d_{\mathrm{LLM}}}.
$$

$$
(H_v,\text{text embeddings})\xrightarrow{\text{LLM}}p(\text{answer}\mid I,\text{prompt}).
$$

<a id="cv-q34"></a>

**95. Why is a visual projector needed?**

It adapts visual feature dimensions and distributions to the language model's input representation. A linear layer or MLP supplies a trainable interface, but compatible dimensions alone do not guarantee semantic alignment: the interface must learn from suitable image-text supervision.

<a id="cv-q35"></a>

**96. When would you freeze the vision encoder versus fine-tune it?**

Freeze it initially when target data is limited, existing features transfer well, or compute is constrained. If evaluation reveals domain-specific failures, compare tuning later layers, adapters or LoRA, and full fine-tuning; control overfitting and forgetting with held-out domains and a suitable learning rate.

<a id="cv-q36"></a>

**97. What is the purpose of instruction tuning in a VLM?**

Instruction tuning teaches the model to answer a user's request using the image, rather than merely align image and text embeddings. In answer-only supervised fine-tuning, teacher forcing supplies the preceding correct tokens and a mask excludes prompt and visual positions from the loss.

$$
\mathcal L_{\mathrm{SFT}}=
-\frac{\sum_{i,t}m_{it}\log p_\theta(a_{it}\mid I_i,q_i,a_{i,<t})}
{\sum_{i,t}m_{it}},\qquad
m_{it}=1\text{ only for supervised answer tokens}.
$$

<a id="cv-q37"></a>

**98. How do you reduce visual token count?**

Use larger patches, spatial pooling, learned resampling, token merging, or task-aware pruning. These reduce LLM prefill and attention costs, but may remove small objects or text, so evaluate detailed visual tasks rather than only average accuracy.

<a id="cv-q38"></a>

**99. How would you evaluate hallucination in VLMs?**

Construct grounded tests for object presence, attributes, spatial relationships, and questions that the image cannot answer. Measure unsupported-claim rates and grounded answer quality using annotated evidence and human review, including negation and out-of-domain cases. Evaluate abstention too, so a model cannot appear reliable merely by refusing every question.

<a id="cv-q39"></a>

**100. Early fusion vs late fusion for multimodal learning?**

Early fusion permits interactions between modalities before final prediction, for example through joint token attention; it can capture detailed relationships but increases coupled computation. Late fusion combines independently produced embeddings, predictions, or scores and is easier to cache and modularize, but may miss fine-grained cross-modal interactions. Intermediate fusion is also possible, so define exactly where information is exchanged.

<a id="cv-q40"></a>

**101. How would you adapt a pretrained VLM to a specialized visual domain?**

Define the target tasks and curate representative, grounded examples with creator-, source-, or time-separated evaluation sets. Choose contrastive adaptation for retrieval or supervised instruction tuning for answer generation, beginning with a frozen backbone and lightweight trainable components. Compare broader fine-tuning only when justified, and evaluate domain transfer, hallucination, forgetting, and deployment cost.

**VLM architecture follow-ups**

**Why not train the vision encoder and LLM jointly from scratch?** Reusing pretrained components saves substantial data and compute and preserves useful representations. Joint training from scratch may be justified with sufficient resources and a strong reason existing representations are unsuitable, but it is not the default for a modest domain dataset.

**How do you select image-text pretraining data?** Check image-text alignment, diversity, domain coverage, provenance, permissions, and harmful or sensitive content. Deduplicate across training and evaluation sets, remove misleading captions, and inspect minority domains rather than maximizing raw pair count.

**Why might a linear projector be sufficient?** Strong pretrained visual features may already encode the information the LLM needs, leaving a relatively simple mapping to learn. Test that hypothesis against an MLP or resampler using grounded accuracy and cost; linear capacity is not sufficient for every task.

**What if the domain is video rather than images?** Encode sampled frames and add temporal positions or temporal fusion so the model can distinguish motion and event order. Compress tokens across space and time, and evaluate short events that coarse sampling might miss.

**How would you reduce end-to-end latency on a mobile device?** Profile preprocessing, visual encoding, LLM prefill, token decoding, and memory transfers separately. Reduce resolution or visual tokens, use supported quantization and smaller models, and benchmark actual-device latency, memory, thermals, and grounded quality.

**Video understanding and video quality**

<a id="cv-q41"></a>

**102. How would you build a video classification model?**

Begin with sampled frames, a pretrained frame encoder, temporal pooling, and a classification head. Pooling gives a cheap baseline, while temporal models can capture motion and order; split by source video to avoid related clips leaking across sets.

$$
I\in\mathbb R^{B\times T\times C\times H\times W}
\xrightarrow{\text{frame encoder}}F\in\mathbb R^{B\times T\times d}
\xrightarrow{\text{temporal mean}}h=\tfrac1T\sum_{t=1}^TF_t
\in\mathbb R^{B\times d}
\xrightarrow{\text{head}}z\in\mathbb R^{B\times K}.
$$

<a id="cv-q42"></a>

**103. What is the difference between a 2D CNN and a 3D CNN?**

A 2D CNN uses spatial kernels independently on images or frames, whereas a 3D CNN uses kernels spanning time and space. A kernel of size $k_t\times k_h\times k_w$ learns local spatiotemporal patterns, but increases computation and depends on the clip's temporal sampling.

<a id="cv-q43"></a>

**104. How do you efficiently sample frames from long videos?**

Uniform sampling gives temporal coverage; keyframe, motion-aware, or adaptive sampling can spend more compute on informative intervals. Sparse sampling can entirely miss a brief harmful event, and motion alone does not establish semantic importance. Consider overlapping clips or a cheap dense first pass followed by targeted high-resolution analysis, measuring event recall against compute.

<a id="cv-q44"></a>

**105. How would you model temporal dependencies in video?**

Temporal pooling ignores order, RNNs maintain sequential state, 3D convolutions model local motion, and temporal attention connects more distant observations. Choose clip length, sampling rate, and model capacity according to event duration and required ordering information. Include temporal positions for attention and test order-sensitive cases explicitly.

<a id="cv-q45"></a>

**106. How would you build a perceptual video quality prediction model?**

First determine whether the task has a pristine reference: full-reference and no-reference prediction have different available inputs. Combine spatial artifacts with temporal degradation, motion, viewing conditions, and source characteristics, then train against subjective scores or pairwise judgments. MOS commonly increases with perceived quality, whereas DMOS often increases with impairment, but verify each dataset's scale and sign; report rank correlation and calibrated prediction error on source-separated splits.

**Production CV and model optimization**

<a id="cv-q46"></a>

**107. How would you deploy a large vision model under a strict latency budget?**

Profile the complete request path, including preprocessing, queueing, inference, and transfers, using realistic hardware and p95/p99 latency. Compare smaller backbones, supported quantization, distillation, caching, and input reduction; batching can improve throughput while increasing waiting time. Asynchronous execution changes scheduling, not necessarily a user's time to result, so measure the actual service objective.

<a id="cv-q47"></a>

**108. Knowledge distillation vs quantization vs pruning?**

Distillation trains a smaller student to reproduce teacher predictions or representations. Quantization reduces numerical precision, while pruning removes weights, channels, tokens, or other structures. Their training needs and quality effects differ, and unstructured sparsity or lower precision only accelerates inference when the runtime and hardware exploit it.

<a id="cv-q48"></a>

**109. What is the trade-off between model accuracy and inference latency?**

Compare quality, tail latency, throughput, memory, and cost to identify a Pareto frontier rather than choosing the highest offline accuracy. A design is dominated if another meets or improves every relevant objective. Select among remaining designs using latency constraints and the actual costs of false positives and false negatives.

<a id="cv-q49"></a>

**110. How would you monitor a deployed CV model?**

Monitor input and embedding drift, missing or corrupted inputs, score distributions, failure rates, and p95/p99 latency. Collect delayed labels and audited samples to measure per-class and subgroup quality, because drift and confidence alone do not prove an accuracy change. Version models and data transformations so regressions can be investigated and rolled back.

<a id="cv-q50"></a>

**111. How would you design a cost-efficient moderation system at large scale?**

Use a cascade in which a cheap model resolves validated low-risk cases and routes uncertain or high-risk cases to stronger models or human reviewers. Tune class-specific decisions against recall, false-positive costs, review capacity, and latency; high confidence alone does not establish safety, particularly under shift. Audit auto-approved content and measure the whole cascade, since an early false negative cannot be repaired by a later model that never sees it.

**Five modeling design cases**

<a id="cv-case01"></a>

**112. How would you design a computer vision system to detect unsafe images uploaded to Roblox?**

Start by defining the unsafe categories, the costs of each error, and whether the output is image-level labels or object locations. Multiple categories can coexist, so a multi-label classifier is often a more appropriate starting point than a mutually exclusive softmax classifier.

1. **Problem formulation.** “First, I would clarify which unsafe categories matter, whether several can occur in one image, and whether reviewers need an image-level decision or localized evidence.” Define latency, upload volume, review capacity, and the consequences of false negatives and false positives before selecting a model.
2. **Data and labels.** “I would use reviewed UGC images, consistent annotation guidelines, and splits that account for creators, near-duplicates, and time.” Include benign lookalikes, rare classes, low-quality inputs, adversarial examples, and different rendering styles. Resolve ambiguous labels and evaluate reviewer agreement.
3. **Baseline model.** “I would start with a pretrained ViT or CLIP image encoder and a lightweight multi-label head.” For an embedding $h\in\mathbb R^d$ and $K$ safety categories, use $z=Wh+b\in\mathbb R^K$ and independent probabilities $p_k=\sigma(z_k)$. Add a larger VLM only if complex semantic reasoning measurably improves grounded decisions.
4. **Loss and optimization.** “I would train with binary cross-entropy, test class weighting for rare categories, and mine visually similar benign hard negatives.” With $y_{ik}\in\{0,1\}$, the basic objective is below; implement it with a numerically stable logits-based loss. Focal loss is an option, not a mandatory extra.

$$
\mathcal L=-\frac{1}{NK}\sum_{i=1}^{N}\sum_{k=1}^{K}
\left[y_{ik}\log p_{ik}+(1-y_{ik})\log(1-p_{ik})\right].
$$

5. **Evaluation.** “I would report per-category precision and recall, average precision, and recall at a constrained false-positive rate.” Select thresholds on validation data, then evaluate once on a held-out test set. Inspect new game styles and other distribution shifts; aggregate accuracy alone can hide rare-category failures.
6. **Deployment.** “I would use a cascade: a small model handles validated low-risk cases, while uncertain or high-risk examples go to a stronger model or a reviewer.” Monitor end-to-end safety quality, review volume, cost, and p95 latency. Audit auto-approved examples too: downstream review cannot repair an item that the first stage incorrectly filters out.

**Follow-up: If false negatives are extremely costly, how would you choose the decision threshold?** Choose per-class thresholds on validation data to satisfy a recall requirement or minimize an explicit error-cost objective, subject to review capacity. Lowering a threshold generally raises recall and false positives; if no threshold meets both safety and capacity constraints, improve the model or review process rather than claiming the conflict is solved.

<a id="cv-case02"></a>

**113. How would you retrieve Roblox assets using a text query or reference image?**

Use a CLIP-style dual encoder: embed a text query or reference image and each candidate asset in a shared space. For a 3D asset, begin with multiple rendered views and useful metadata; an image encoder does not directly consume a mesh.

1. **Training data.** Collect relevant query–asset pairs, captions, and human relevance judgments. Fine-tune with a contrastive objective such as InfoNCE, mixing random and hard negatives while checking for false negatives. Split related assets and creators carefully.
2. **Retrieval.** Normalize embeddings and rank by $s(q,a)=e_q^\top e_a$. Precompute asset embeddings, store them in an approximate-nearest-neighbor index, retrieve a candidate set, and optionally rerank it with a more expensive cross-modal model. An image query uses the image encoder; a text query uses the text encoder.
3. **Evaluation and serving.** Report Recall@K for candidate retrieval, and a relevance-aware ranking metric such as NDCG when graded judgments exist. Measure query latency, index freshness, and quality on new styles. Version the encoder and index together because embeddings from incompatible model versions should not be mixed blindly.

**Follow-up: How would you handle new assets or text queries not represented in training data?** Encode new assets from their content and update the index without requiring interaction history. A pretrained text encoder can represent unseen queries, but representation does not guarantee relevance: evaluate held-out concepts, new rendering styles, and difficult vocabulary, then collect domain-specific pairs for adaptation.

**Follow-up: Why choose CLIP instead of a ViT classifier, and when would it fail?** CLIP supports open-vocabulary text queries and retrieval without a fixed classification head. A supervised ViT classifier may be better for a stable, well-labeled category set; CLIP can fail on unusual rendering styles, fine-grained distinctions, or misleading captions. Compare both on held-out domain data, improve positives and hard negatives, and profile the actual serving path before choosing.

<a id="cv-case03"></a>

**114. How would you detect harmful events in short gameplay videos?**

Clarify whether the product needs a single clip label or the start and end of each harmful event. Begin with frame sampling, a pretrained spatial encoder, temporal pooling, and a classification head; add a temporal model when motion, ordering, or localization matters.

$$
I\in\mathbb R^{B\times T\times C\times H\times W}
\longrightarrow F\in\mathbb R^{B\times T\times d}
\longrightarrow \text{temporal model}
\longrightarrow Z\in\mathbb R^{B\times T\times K}.
$$

Per-frame scores can be pooled for a clip-level decision or decoded into event intervals. Collect event boundaries when localization is required; clip-level labels alone give weaker temporal supervision. Use source-video-separated splits, account for rare events with suitable sampling or loss weighting, and evaluate event recall, false alarms, temporal localization, and inference cost rather than frame accuracy alone.

**Follow-up: What if the harmful event appears in only 0.5 seconds of a 60-second video?** Sparse uniform frames can miss it entirely. Use denser sampling or overlapping short windows, possibly with a cheap high-coverage first pass and a stronger second pass. Motion-guided sampling is useful but not sufficient—a brief harmful image can be static. Measure recall specifically for short events and ensure a missed first-stage event is included in the end-to-end error analysis.

<a id="cv-case04"></a>

**115. How would you adapt a pretrained VLM using 100K labeled image-text examples from Roblox?**

First establish whether the target is retrieval, captioning, visual question answering, or another task: the same image-text pairs do not imply the same training objective. Inspect label quality and deduplicate before constructing creator-, style-, and time-aware evaluation sets.

For question answering, format each example as an image, instruction, and grounded answer. Start with the vision encoder frozen and train the projector and/or LoRA adapters; compare broader fine-tuning only when validation failures justify it. Use teacher forcing and answer-only cross-entropy, masking prompt, visual, and padding positions. For retrieval, use a contrastive objective instead of assuming answer-generation SFT is appropriate.

Evaluate task quality, hallucination, refusal or abstention behavior, general-capability forgetting, and unfamiliar game styles. Select checkpoints and hyperparameters on validation data, then measure latency, memory, and token count on the intended deployment hardware.

**Follow-up: The model performs well on training data but poorly on new Roblox game styles. What would you do?** Separate ordinary overfitting from a genuine domain shift using style-held-out evaluation. Add representative styles and hard cases, rebalance coverage, apply semantics-preserving augmentation, and compare lower learning rates, stronger regularization, or selective unfreezing. Keep a truly unseen-style test set so improvements are not just repeated tuning to the same examples.

<a id="cv-case05"></a>

**116. How would you build a low-latency VLM that answers questions about gameplay screenshots on mobile?**

Define whether the latency budget means time to first token or time to a complete answer, and specify image size, answer length, target device, privacy constraints, and expected load. Profile the complete path: capture and preprocessing, visual encoder, projector, LLM prefill, autoregressive decoding, memory transfers, and network time if inference is remote.

Use an efficient visual backbone, reduce visual tokens with pooling or a resampler, and compare a distilled or quantized language model on supported hardware. Cache unchanged visual features only when their freshness is valid. Compare on-device inference with cloud or hybrid execution using measured latency, battery use, thermal behavior, memory, connectivity, and grounded answer quality.

**Follow-up: What if the VLM takes 500 ms but the product requires 100 ms?** Break down the 500 ms first and optimize the dominant stage: fewer visual tokens, a smaller encoder or decoder, supported quantization, shorter outputs, or carefully valid caching. Verify each change on the target device. Streaming can improve time to first visible output but does not make a 500 ms complete answer finish in 100 ms; if no configuration meets both quality and latency constraints, explicitly renegotiate the scope or budget.

<details>
<summary>Additional prompts from the conversation</summary>

These are alternative wordings and combined follow-ups from the same conversation; the linked answers above cover their underlying concepts.

- **Why does ViT need positional embeddings?** Without spatial positions, patch self-attention treats patches as an unordered set. See [the positional-encoding answer](#cv-q06).
- **CNN vs ViT: which would you choose for a small image dataset?** Consider pretrained features and domain fit; a CNN's local priors can help with limited training data, while a pretrained ViT can still be strong. See [the comparison](#cv-q07).
- **How would you improve detection of small objects?** Preserve resolution, use appropriate multiscale features, and consider tiled inference. See [small-object detection](#cv-q26).
- **Why use focal loss instead of cross-entropy?** It downweights easy examples when they dominate training; validate the trade-off rather than using it automatically. See [focal loss](#cv-q12).
- **How does InfoNCE work and what is the role of temperature?** Contrast a positive against negatives; temperature controls the sharpness of similarity logits and the training gradients. See [InfoNCE and its follow-ups](#cv-q17).
- **How would you address overfitting and severe class imbalance?** Diagnose leakage and label quality, regularize, and compare sampling or weighting using class-specific held-out metrics. See [overfitting](#cv-q16) and [imbalance](#cv-q20).
- **How would you fine-tune CLIP for a domain-specific retrieval task?** Use representative paired data, a contrastive objective, and carefully chosen negatives; evaluate held-out retrieval quality. See [CLIP training](#cv-q31) and [the retrieval case](#cv-case02).
- **How would you reduce visual tokens without losing key information?** Compare pooling, resampling, or pruning while explicitly testing small objects and text. See [visual token reduction](#cv-q37).
- **How would you build a video understanding model for rare events?** Match temporal coverage to event duration and evaluate event-level recall. See [the video case](#cv-case03).
- **Design an unsafe-image detection system for Roblox.** See [the worked moderation case](#cv-case01), including its data, loss, evaluation, and deployment decisions.
- **Design an image-text retrieval system for 3D assets.** See [the retrieval case](#cv-case02), including how rendered views connect 3D assets to image encoders.
- **Design a low-latency VLM for gameplay understanding.** See [the mobile case](#cv-case05), including the 500 ms to 100 ms follow-up.

</details>


---

<a id="4-the-modern-transformer-lm"></a>
# 3. Transformer: From Input to Output

One sentence, followed through every equation of the original encoder–decoder Transformer ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)). Each step states its input, the operation, and its output, with the concrete shapes of the example next to the general ones. The code is kept out of the book: `transformer_encoder_decoder.py` transcribes these equations line by line, and `transformer_encoder_decoder.ipynb` runs it block by block on this example, printing the shape written here at every step.

<a id="7-transformer--attention"></a>
<a id="transformer-math"></a>

**The example.** Source sentence `we watch a short video`, target sentence `nous regardons video`, one tiny vocabulary per language with `PAD = 0`, `BOS = 1`, `EOS = 2`:

$$
\begin{aligned}
\text{source ids}\quad \mathbf{s} &= (\underbrace{3}_{\text{we}},\ \underbrace{4}_{\text{watch}},\ \underbrace{5}_{\text{a}},\ \underbrace{6}_{\text{short}},\ \underbrace{7}_{\text{video}}) && n = 5 \\[6pt]
\text{decoder input}\quad \mathbf{t} &= (\underbrace{1}_{\text{BOS}},\ \underbrace{3}_{\text{nous}},\ \underbrace{4}_{\text{regardons}},\ \underbrace{5}_{\text{video}}) && m = 4 \\[6pt]
\text{labels}\quad \mathbf{y} &= (\underbrace{3}_{\text{nous}},\ \underbrace{4}_{\text{regardons}},\ \underbrace{5}_{\text{video}},\ \underbrace{2}_{\text{EOS}}) && y_j = t_{j+1}
\end{aligned}
$$

An id is nothing but a row number. The source vocabulary is the list `<pad>, <bos>, <eos>, we, watch, a, short, video`, so `we` is $3$ only because it is entry $3$ of that list; the target vocabulary is a separate list (`<pad>, <bos>, <eos>, nous, regardons, video, ...`) with its own embedding table, so target id $3$ (`nous`) has nothing to do with source id $3$. The number carries no meaning and no order; its only use is to pick a row of the embedding table in step 1.

The decoder input is the target shifted right by one token, and every position predicts the token after it:

| position $j$ | decoder input $t_j$ | may read (mask of step 6) | label $y_j$ |
|---|---|---|---|
| 0 | BOS | BOS | nous |
| 1 | nous | BOS, nous | regardons |
| 2 | regardons | BOS, nous, regardons | video |
| 3 | video | BOS, nous, regardons, video | EOS |

The answer of position $j$ is always one step to its right, and the mask hides everything to the right, so no position is shown its own answer. Feeding the whole known target at once is teacher forcing: all $m$ predictions are trained in one forward pass instead of $m$ sequential ones, and each position sees exactly the information it would have during generation. Feeding the true prefix rather than the model's own guesses keeps the supervision clean early in training; at inference (step 11) the prefix comes from the model's previous outputs instead. Sizes used throughout: model width $d = 8$, heads $h = 2$, width per head $d_k = d/h = 4$, feed-forward width $d_{ff} = 32$, vocabularies $V_{\text{src}} = 8$ and $V_{\text{tgt}} = 10$. Every matrix below has one row per token. PyTorch puts a batch axis $B$ in front of every shape; for this single sentence $B = 1$.

<a id="73-positional-encoding"></a>

**Step 1 · Token ids → embeddings + positions.** $\mathbf{s} \in \{0,\dots,V_{\text{src}}-1\}^{n} \;\longrightarrow\; X \in \mathbb{R}^{n \times d}$

$$
X = \sqrt{d}\; E_{\text{src}}[\mathbf{s}] + P_{1:n}, \qquad E_{\text{src}} \in \mathbb{R}^{V_{\text{src}} \times d} \ \text{(learned)}, \qquad P \in \mathbb{R}^{L_{\max} \times d} \ \text{(fixed)}
$$

$$
P_{p,\,2i} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad P_{p,\,2i+1} = \cos\!\left(\frac{p}{10000^{2i/d}}\right), \qquad p = \text{position},\quad i = 0,\dots,\tfrac{d}{2}-1
$$

$E_{\text{src}}[\mathbf{s}]$ stacks rows $3, 4, 5, 6, 7$ of the embedding table: a lookup, not a multiplication by the id. Without $P$, self-attention is permutation-equivariant and `we watch` is indistinguishable from `watch we`. Shapes: $[5] \to [5, 8]$.

The $\sqrt{d}$ is the paper's convention. Its embedding rows are initialized at scale $1/\sqrt{d}$ because the same matrix is reused as the output projection, and the factor lifts them to the $O(1)$ scale of $P$, whose rows have norm $\sqrt{d/2}$: token and position signals must be comparable when they are added. With PyTorch's default $\mathcal{N}(0, 1)$ initialization the factor is not needed, and the first LayerNorm rescales everything anyway. GPT-2 omits it because its token and position tables are both learned and start at a comparable small scale, LLaMA because it adds no position vector at all; Gemma keeps it. Most current LLMs drop the addition of $P$ altogether and rotate queries and keys inside attention instead: see [Rotary positions (RoPE)](#rotary-positions-rope) after the code.

<a id="step-1-images"></a>

**The same step for an image.** $I \in \mathbb{R}^{H \times W \times C} \;\longrightarrow\; Z_0 \in \mathbb{R}^{N \times d_v}$

For an image the tokens are patches (steps 1–2 of [Chapter 4](#4-vlm-from-image-and-question-to-answer)): $P \times P$ squares of pixels in a grid with $H_P = H/P$ rows and $W_P = W/P$ columns, $N = H_P W_P$ patches in all. The grid is read row after row, so the patch in row $h$ and column $w$ is token number $k = h\, W_P + w$. The embedding is a linear layer instead of a lookup, because pixels are continuous, and the position vector is added exactly as for words:

$$
Z_0[k] = X_p[k]\, W^{E} + b^{E} + P^{\text{vis}}_{(h,w)}, \qquad k = h\, W_P + w
$$

$P^{\text{vis}}_{(h,w)}$ is built in one of two ways:

1. **A learned table with one row per grid cell** (ViT, CLIP, SigLIP, and the toy of Chapter 4). $P^{\text{vis}} \in \mathbb{R}^{N \times d_v}$ is a free parameter and row $k$ is used for patch $(h, w)$. The table knows nothing about rows and columns; the model learns from data that rows $k$ and $k + W_P$ are neighbours. Because the table has exactly $N$ rows, another resolution needs it reshaped to $[H_P, W_P, d_v]$, interpolated to the new grid, and flattened again.
2. **Fixed two-dimensional sine and cosine.** The $d_v$ features are split into two halves: the first half encodes the row with the formula of step 1, the second half the column, each with $d_v / 4$ frequencies $\omega_i = 10000^{-4i/d_v}$:

$$
\begin{aligned}
P^{\text{vis}}_{(h,w)} = \big[\ &\sin(h\,\omega_0),\ \cos(h\,\omega_0),\ \dots,\ \sin(h\,\omega_{d_v/4-1}),\ \cos(h\,\omega_{d_v/4-1}), && \text{row half} \\[2pt]
&\sin(w\,\omega_0),\ \cos(w\,\omega_0),\ \dots,\ \sin(w\,\omega_{d_v/4-1}),\ \cos(w\,\omega_{d_v/4-1})\ \big] && \text{column half}
\end{aligned}
$$

The dot product of two such vectors depends on the row distance and on the column distance, separately, as in the 1D case:

$$
P^{\text{vis}}_{(h,w)} \cdot P^{\text{vis}}_{(h',w')} = \sum_{i=0}^{d_v/4-1} \cos\big((h - h')\,\omega_i\big) + \sum_{i=0}^{d_v/4-1} \cos\big((w - w')\,\omega_i\big)
$$

With the $2 \times 2$ grid of Chapter 4 and $d_v = 8$ there are two frequencies per half, $\omega_0 = 1$ and $\omega_1 = 10000^{-4/8} = 0.01$:

<a id="step-1-image-table"></a>

| patch $k$ | $(h, w)$ | row half: $\sin h\omega_0,\ \cos h\omega_0,\ \sin h\omega_1,\ \cos h\omega_1$ | column half: the same with $w$ |
|---|---|---|---|
| $0$ | $(0, 0)$ | $0,\ 1,\ 0,\ 1$ | $0,\ 1,\ 0,\ 1$ |
| $1$ | $(0, 1)$ | $0,\ 1,\ 0,\ 1$ | $0.841,\ 0.540,\ 0.010,\ 1.000$ |
| $2$ | $(1, 0)$ | $0.841,\ 0.540,\ 0.010,\ 1.000$ | $0,\ 1,\ 0,\ 1$ |
| $3$ | $(1, 1)$ | $0.841,\ 0.540,\ 0.010,\ 1.000$ | $0.841,\ 0.540,\ 0.010,\ 1.000$ |

Patches in the same row share the row half and patches in the same column share the column half. The dot product of patch $3$ with patch $1$, one row apart, is $\cos 1 + \cos 0.01 + 2 = 3.540$, the same as with patch $2$, one column apart. Both ways add the position to the content once, at the input, like step 1 for words. [RoPE](#rope-2d) instead turns the queries and keys by $(h, w)$ inside attention.

<a id="71-self-attention"></a>
<a id="72-multi-head-attention"></a>
<a id="coding-scaled-attention"></a>
<a id="coding-mha"></a>

**Step 2 · Encoder multi-head self-attention.** $X \in \mathbb{R}^{n \times d} \;\longrightarrow\; A \in \mathbb{R}^{n \times d}$

For each head $i = 1, \dots, h$, project the same $X$ three times:

$$
Q_i = X W_i^{Q}, \qquad K_i = X W_i^{K}, \qquad V_i = X W_i^{V} \qquad \in \mathbb{R}^{n \times d_k}, \qquad W_i^{Q}, W_i^{K}, W_i^{V} \in \mathbb{R}^{d \times d_k}
$$

Score every query against every key, normalize each row over the keys, and average the values with those weights:

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}} + M\right) V_i \;\in \mathbb{R}^{n \times d_k}, \qquad \operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_{k'} e^{z_{k'}}}
$$

Concatenate the heads along the feature axis and mix them once more:

$$
A = \operatorname{MHA}(X, X, X;\, M) = \big[\, \text{head}_1 \,\Vert\, \cdots \,\Vert\, \text{head}_h \,\big]\, W^{O} \;\in \mathbb{R}^{n \times d}, \qquad W^{O} \in \mathbb{R}^{d \times d}
$$

$\operatorname{MHA}(\cdot, \cdot, \cdot;\, M)$ names the inputs that produce $Q$, $K$, $V$ and the mask; in self-attention all three are $X$. $M \in \mathbb{R}^{n \times n}$ has one row per query and one column per key: $0$ where query $j$ may read key $k$, $-\infty$ where it may not. Added before the softmax, $e^{-\infty} = 0$ makes a blocked key's weight exactly $0$ and the allowed keys renormalize among themselves: scores $(2.0, 1.0, 0.5, 1.7, 3.0)$ with $M = (0, 0, 0, -\infty, -\infty)$ give weights $(0.63, 0.23, 0.14, 0, 0)$. A $0$ in place of $-\infty$ would not work, because $e^{0} = 1$ still receives weight. The encoder's $M$ only blocks `PAD` keys, so for this single sentence it is all zeros and changes nothing; it matters when a batch pads shorter sentences, and it always blocks columns (keys), never rows (queries). Every head uses the same $M$. Per head: $Q_i K_i^{\top}$ is $[5, 4] \times [4, 5] = [5, 5]$, one row of weights per query that sums to $1$; weights $\times\, V_i$ is $[5, 5] \times [5, 4] = [5, 4]$; two heads concatenate to $[5, 8]$. The scale is $\sqrt{d_k} = 2$, not $\sqrt{d}$. In code the $h$ matrices $W_i^{Q}$ sit side by side in one `Linear(d, d)`, and the head split is a reshape of the projected features.

**Step 3 · Add & Norm.** $X, A \in \mathbb{R}^{n \times d} \;\longrightarrow\; X_1 \in \mathbb{R}^{n \times d}$

$$
X_1 = \operatorname{LN}(X + A), \qquad \operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \frac{1}{d}\sum_{k=1}^{d} x_k, \quad \sigma^2 = \frac{1}{d}\sum_{k=1}^{d} (x_k - \mu)^2
$$

The addition is the residual path: the sublayer only has to learn a correction to $X$. LayerNorm treats each row $x \in \mathbb{R}^{d}$ (one token) on its own, so the $5$ tokens get $5$ separate means and variances; $\gamma, \beta \in \mathbb{R}^{d}$ are learned and shared by all tokens. Nothing is averaged across tokens or across the batch (with a batch there are $B \times n$ separate statistics), which is why variable lengths and batch size $1$ are no problem, unlike BatchNorm. RMSNorm, used by most current LLMs, normalizes the same axis without subtracting the mean.

**Step 4 · Feed-forward, then Add & Norm.** $X_1 \in \mathbb{R}^{n \times d} \;\longrightarrow\; H \in \mathbb{R}^{n \times d}$

$$
\operatorname{FFN}(x) = \operatorname{ReLU}(x W_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d \times d_{ff}}, \quad W_2 \in \mathbb{R}^{d_{ff} \times d}
$$

$$
H = \operatorname{LN}\big(X_1 + \operatorname{FFN}(X_1)\big) \;\in \mathbb{R}^{n \times d}
$$

Lower-case $x$ is one token's vector, a single row of $X_1$: the FFN is a function of one row, and the same $W_1, b_1, W_2, b_2$ are applied to each of the $n$ rows, so nothing moves between positions here (only attention does that). Shapes: $[5, 8] \to [5, 32] \to [5, 8]$: widen, apply the nonlinearity, come back to $d$ so the residual addition works. Attention decides which tokens to read; the FFN transforms what was read, and with $2\,d\,d_{ff}$ weights it holds about two thirds of a layer's parameters. $H$ is the encoder output, one contextual vector per source token; the code calls it `memory`. With $N$ stacked encoder layers, steps 2–4 repeat $N$ times with separate weights and $H$ is the output of the last one.

**Step 5 · Decoder input.** $\mathbf{t} \in \{0,\dots,V_{\text{tgt}}-1\}^{m} \;\longrightarrow\; Y \in \mathbb{R}^{m \times d}$

$$
Y = \sqrt{d}\; E_{\text{tgt}}[\mathbf{t}] + P_{1:m}, \qquad E_{\text{tgt}} \in \mathbb{R}^{V_{\text{tgt}} \times d}
$$

Step 1 again, with a separate table for the target language and the same fixed $P$. Shapes: $[4] \to [4, 8]$.

<a id="74-causal-masking"></a>
<a id="coding-causal-mask"></a>

**Step 6 · Masked self-attention + Add & Norm.** $Y \in \mathbb{R}^{m \times d} \;\longrightarrow\; Y_1 \in \mathbb{R}^{m \times d}$

$$
Y_1 = \operatorname{LN}\big(Y + \operatorname{MHA}(Y, Y, Y;\, M^{\text{causal}})\big), \qquad M^{\text{causal}}_{jk} = \begin{cases} 0 & k \le j \\ -\infty & k > j \end{cases}
$$

$$
M^{\text{causal}} = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}
$$

Rows are the queries `BOS, nous, regardons, video`; columns are the same tokens as keys. Same MHA as step 2 with its own weights; scores are $[4, 4]$ per head. Row $j$ may read keys $k \le j$ only, so the prediction at position $j$ depends on $t_1, \dots, t_j$ and never on $t_{j+1}$, which is its label. That is what lets one forward pass train all $m$ positions at once (teacher forcing) while matching left-to-right generation.

**Step 7 · Cross-attention + Add & Norm.** $Y_1 \in \mathbb{R}^{m \times d},\ H \in \mathbb{R}^{n \times d} \;\longrightarrow\; Y_2 \in \mathbb{R}^{m \times d}$

$$
Y_2 = \operatorname{LN}\big(Y_1 + \operatorname{MHA}(Y_1, H, H;\, M^{\text{pad}})\big), \qquad Q_i = Y_1 W_i^{Q}, \quad K_i = H W_i^{K}, \quad V_i = H W_i^{V}
$$

Queries come from the target, keys and values from the source: this is the only place where the decoder reads the sentence it is translating. Per head, $Q_i K_i^{\top}$ is $[4, 4] \times [4, 5] = [4, 5]$ (target rows, source columns), weights $\times\, V_i$ is $[4, 5] \times [5, 4] = [4, 4]$, and the heads concatenate to $[4, 8]$: the output length follows the queries. $M^{\text{pad}}$ blocks `PAD` source keys; there is no causal mask here because the whole source is known.

**Step 8 · Feed-forward + Add & Norm.** $Y_2 \in \mathbb{R}^{m \times d} \;\longrightarrow\; Y_3 \in \mathbb{R}^{m \times d}$

$$
Y_3 = \operatorname{LN}\big(Y_2 + \operatorname{FFN}(Y_2)\big)
$$

Same form as step 4 with its own $W_1, W_2$: $[4, 8] \to [4, 32] \to [4, 8]$. Steps 6–8 form one decoder layer; $N$ layers stack the same way as in the encoder, every layer reading the same $H$.

**Step 9 · Logits.** $Y_3 \in \mathbb{R}^{m \times d} \;\longrightarrow\; Z \in \mathbb{R}^{m \times V_{\text{tgt}}}$

$$
Z = Y_3 W^{\text{vocab}} + b^{\text{vocab}}, \qquad W^{\text{vocab}} \in \mathbb{R}^{d \times V_{\text{tgt}}}, \qquad p\big(t_{j+1} = v \,\big|\, t_{1:j}, \mathbf{s}\big) = \operatorname{softmax}(Z_j)_v
$$

Row $j$ scores every vocabulary entry as the token that follows the prefix $t_{1:j}$. Shapes: $[4, 8] \to [4, 10]$; the last axis now indexes words, not features (with a real vocabulary it would be $[4, 32000]$).

**Step 10 · Loss (training).** $Z \in \mathbb{R}^{m \times V_{\text{tgt}}},\ \mathbf{y} \in \{0,\dots,V_{\text{tgt}}-1\}^{m} \;\longrightarrow\; \mathcal{L} \in \mathbb{R}$

$$
\mathcal{L} = -\frac{1}{m} \sum_{j=1}^{m} \log \operatorname{softmax}(Z_j)_{y_j} = -\frac{1}{m} \sum_{j=1}^{m} \Big( Z_{j,\, y_j} - \log \sum_{v=1}^{V_{\text{tgt}}} e^{Z_{j,\, v}} \Big)
$$

Cross-entropy of the next token at every position; positions whose label is `PAD` are dropped from the average. Backpropagation gives $\partial \mathcal{L} / \partial \theta$ for every weight of steps 1–9, encoder included, and one optimizer step updates them all. `F.cross_entropy` takes the raw $Z$ and applies the log-softmax itself.

**Step 11 · Generation (inference).** $\mathbf{s} \;\longrightarrow\; \hat{\mathbf{t}}$

$$
H = \operatorname{Encoder}(\mathbf{s}) \ \text{once}; \qquad \hat t_1 = \text{BOS}, \qquad \hat t_{j+1} = \operatorname*{arg\,max}_{v}\ Z_j\big(\hat t_{1:j},\, H\big)_v, \qquad \text{stop at EOS or at a length limit}
$$

Each step reruns steps 5–9 on the current prefix, so the prefix grows $[1] \to [2] \to [3] \to \cdots$ while $H$ stays $[5, 8]$, and only the last row $Z_j$ is used. Recomputing the prefix costs $O(j)$ per step; the KV cache of [§7](#7-inference) stores each layer's $K_i, V_i$ for past positions instead. Sampling (temperature, top-k, top-p) replaces the $\arg\max$.

<a id="transformer-shapes"></a>
<a id="49-the-complete-tensor-ledger"></a>

**Every shape in one table** ($n = 5$, $m = 4$, $d = 8$, $h = 2$, $d_k = 4$, $d_{ff} = 32$, $V_{\text{tgt}} = 10$; put $B$ in front of each shape for PyTorch).

| Step | Input | Output |
|---|---|---|
| 1 · source embedding + position | $\mathbf{s}$: $[5]$ | $X$: $[5, 8]$ |
| 2 · $Q_i, K_i, V_i$ per head | $X$: $[5, 8]$ | $[5, 4]$ each |
| 2 · scores $Q_i K_i^{\top} / \sqrt{d_k}$ | $[5, 4] \times [4, 5]$ | $[5, 5]$ per head |
| 2 · weights $\times\, V_i$ | $[5, 5] \times [5, 4]$ | $[5, 4]$ per head |
| 2 · concatenate heads, $W^{O}$ | $[5, 8]$ | $A$: $[5, 8]$ |
| 3 · Add & Norm | $X, A$: $[5, 8]$ | $X_1$: $[5, 8]$ |
| 4 · FFN + Add & Norm | $X_1$: $[5, 8] \to [5, 32]$ | $H$: $[5, 8]$ |
| 5 · target embedding + position | $\mathbf{t}$: $[4]$ | $Y$: $[4, 8]$ |
| 6 · masked self-attention (scores $[4, 4]$) + Add & Norm | $Y$: $[4, 8]$ | $Y_1$: $[4, 8]$ |
| 7 · cross-attention (scores $[4, 5]$) + Add & Norm | $Y_1$: $[4, 8]$, $H$: $[5, 8]$ | $Y_2$: $[4, 8]$ |
| 8 · FFN + Add & Norm | $Y_2$: $[4, 8] \to [4, 32]$ | $Y_3$: $[4, 8]$ |
| 9 · vocabulary projection | $Y_3$: $[4, 8]$ | $Z$: $[4, 10]$ |
| 10 · cross-entropy | $Z$: $[4, 10]$, $\mathbf{y}$: $[4]$ | $\mathcal{L}$: scalar |

Weights per layer: an encoder layer holds $4d^2$ attention weights ($W^{Q}, W^{K}, W^{V}, W^{O}$ over all heads) and $2\,d\,d_{ff}$ FFN weights, plus small bias and LayerNorm vectors; a decoder layer has two attention blocks, $8d^2 + 2\,d\,d_{ff}$. Compute per layer is those weights times the number of tokens, plus the attention scores, $O(h\,n^2 d_k) = O(n^2 d)$, the only term quadratic in the length.

<a id="410-complete-runnable-code"></a>
<a id="pytorch-implementation"></a>

**Code.** The implementation is read in VS Code, not here. `transformer_encoder_decoder.py` implements the eleven steps with plain `nn.Embedding`, `nn.Linear` and `nn.LayerNorm`: boolean masks with `True` = this key may be read, the paper's post-norm order, no dropout. `transformer_encoder_decoder.ipynb` runs it block by block on this example with a check after every step, and `test_transformer_encoder_decoder.py` compares the attention with a per-head loop and with `F.scaled_dot_product_attention`.

<a id="coding-rope"></a>
## Rotary positions (RoPE)

Step 1 tells the model where a token stands by adding a position vector to its embedding, once, at the input. Most current LLMs do it differently ([RoFormer](https://arxiv.org/abs/2104.09864)): nothing is added in step 1, and inside every attention layer the query and key vectors are turned by an angle that depends on the position of their token. This section says where that happens and what every symbol means, then gives the rule and the reason it works, and then the three cases in use: 1D for text, 2D for images, 3D for video.

**Where it happens.** Two places of the walkthrough change; everything else stays.

<a id="rope-where"></a>

| | walkthrough | with RoPE |
|---|---|---|
| step 1 | $X = \sqrt{d}\, E[\mathbf{s}] + P$ | $X = E[\mathbf{s}]$, no position is added |
| step 2, projections | $Q_i = X W_i^{Q}$, $K_i = X W_i^{K}$, $V_i = X W_i^{V}$ | the same |
| step 2, new | nothing | every row of $Q_i$ and of $K_i$ is turned by its position; $V_i$ is left alone |
| step 2, scores | $Q_i K_i^{\top} / \sqrt{d_k} + M$ | $\tilde Q_i \tilde K_i^{\top} / \sqrt{d_k} + M$ |
| softmax, values, $W^{O}$, steps 3–11 | as written | the same, with the same shapes |

The turn is repeated in every layer and every head, in the self-attention of steps 2 and 6. Cross-attention (step 7) usually has none, because its queries and keys stand in two different sequences.

**Notation.** Everything below is about one head of one attention layer.

<a id="rope-notation"></a>

| symbol | meaning |
|---|---|
| $p$, $p'$ | position of the query token and of the key token, counted from $0$ |
| $q_p \in \mathbb{R}^{d_k}$ | row $p$ of $Q_i$: the query vector of the token at position $p$ |
| $k_{p'} \in \mathbb{R}^{d_k}$ | row $p'$ of $K_i$: the key vector of the token at position $p'$ |
| $q_p^{\top} k_{p'}$ | their score, the entry $(p, p')$ of $Q_i K_i^{\top}$ |
| pair $j$ | features $2j$ and $2j+1$ of a vector, read as one point in the plane; $j = 0, \dots, d_k/2 - 1$ |
| $\theta_j$ | speed of pair $j$ in radians per position, $\theta_j = 10000^{-2j/d_k}$: the frequencies of step 1 |
| $R(\alpha)$ | the $2 \times 2$ matrix that turns a point of the plane by the angle $\alpha$ |
| $\tilde q_p$, $\tilde k_{p'}$ | the same vectors after the turn |

$$
R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}, \qquad R(\alpha)^{\top} = R(-\alpha), \qquad R(\alpha)\, R(\beta) = R(\alpha + \beta)
$$

**The rule.** $q_p \in \mathbb{R}^{d_k} \;\longrightarrow\; \tilde q_p \in \mathbb{R}^{d_k}$, and the same for $k_{p'}$

Pair $j$ of the vector of the token at position $p$ is turned by the angle $p\,\theta_j$:

$$
\begin{pmatrix} \tilde q_{p,\,2j} \\ \tilde q_{p,\,2j+1} \end{pmatrix} = R(p\,\theta_j) \begin{pmatrix} q_{p,\,2j} \\ q_{p,\,2j+1} \end{pmatrix} = \begin{pmatrix} q_{p,\,2j} \cos(p\,\theta_j) - q_{p,\,2j+1} \sin(p\,\theta_j) \\ q_{p,\,2j} \sin(p\,\theta_j) + q_{p,\,2j+1} \cos(p\,\theta_j) \end{pmatrix}
$$

The token at position $0$ is not turned, the token at position $1$ is turned by $\theta_j$, the token at position $2$ by $2\,\theta_j$, and so on. A turn changes the direction of a pair and not its length, so $\lVert \tilde q_p \rVert = \lVert q_p \rVert$. All pairs together form one block matrix, which is never built in practice because each pair costs only four multiplications:

$$
\tilde q_p = R_p\, q_p, \qquad \tilde k_{p'} = R_{p'}\, k_{p'}, \qquad R_p = \begin{pmatrix} R(p\,\theta_0) & & \\ & \ddots & \\ & & R(p\,\theta_{d_k/2-1}) \end{pmatrix} \in \mathbb{R}^{d_k \times d_k}
$$

**Why the score sees only the distance.** $\tilde q_p,\ \tilde k_{p'} \in \mathbb{R}^{d_k} \;\longrightarrow\; \tilde q_p^{\top}\, \tilde k_{p'} \in \mathbb{R}$

Take pair $j$ of both vectors, $a_j = (q_{p,\,2j},\ q_{p,\,2j+1})$ and $b_j = (k_{p',\,2j},\ k_{p',\,2j+1})$: two points of the plane with lengths $\lVert a_j \rVert$, $\lVert b_j \rVert$ and directions $\phi_{a_j}$, $\phi_{b_j}$. Their dot product is the product of the lengths times the cosine of the angle between them. Turning $a_j$ by $p\,\theta_j$ and $b_j$ by $p'\theta_j$ changes that angle by the difference of the two turns:

$$
\big(R(p\,\theta_j)\, a_j\big)^{\top} \big(R(p'\theta_j)\, b_j\big) = a_j^{\top}\, R\big((p' - p)\,\theta_j\big)\, b_j = \lVert a_j \rVert\, \lVert b_j \rVert \cos\big(\phi_{a_j} - \phi_{b_j} + (p - p')\,\theta_j\big)
$$

The score is the sum over the pairs:

$$
\tilde q_p^{\top}\, \tilde k_{p'} = q_p^{\top}\, R_{p'-p}\, k_{p'} = \sum_{j=0}^{d_k/2-1} \lVert a_j \rVert\, \lVert b_j \rVert \cos\big(\phi_{a_j} - \phi_{b_j} + (p - p')\,\theta_j\big)
$$

The lengths and directions are content. The positions enter only through $p - p'$: the two absolute positions have disappeared, and a pair of tokens at positions $(3, 1)$ scores exactly like the same pair at $(103, 101)$.

<a id="rope-1d"></a>

**1D: text.** The position of a token is one number, its index $p$ in the sequence, and every pair is turned by $p\,\theta_j$. This is the case described so far and the one inside an LLM. With the sentence of this chapter and $d_k = 4$ there are two pairs, with $\theta_0 = 1$ and $\theta_1 = 10000^{-2/4} = 0.01$:

<a id="rope-angles"></a>

| position $p$ | token | pair 0 is turned by $p\,\theta_0$ | pair 1 is turned by $p\,\theta_1$ |
|---|---|---|---|
| 0 | we | $0$ | $0$ |
| 1 | watch | $1$ rad $= 57.3^{\circ}$ | $0.01$ rad $= 0.6^{\circ}$ |
| 2 | a | $2$ rad $= 114.6^{\circ}$ | $0.02$ rad $= 1.1^{\circ}$ |
| 3 | short | $3$ rad $= 171.9^{\circ}$ | $0.03$ rad $= 1.7^{\circ}$ |
| 4 | video | $4$ rad $= 229.2^{\circ}$ | $0.04$ rad $= 2.3^{\circ}$ |

To see the positions alone, give every token the same query and key, $q = k = (1, 0, 1, 0)$. Without the turn every score would be $q^{\top} k = 2$. With it, the score is $\cos\big((p - p')\,\theta_0\big) + \cos\big((p - p')\,\theta_1\big)$:

<a id="rope-scores"></a>

| query reads key | offset $p - p'$ | score |
|---|---|---|
| `short` reads `short` | $0$ | $\cos 0 + \cos 0 = 2.000$ |
| `a` reads `watch` | $1$ | $\cos 1 + \cos 0.01 = 1.540$ |
| `short` reads `watch` | $2$ | $\cos 2 + \cos 0.02 = 0.584$ |
| `video` reads `a` | $2$ | $\cos 2 + \cos 0.02 = 0.584$ |
| `video` reads `we` | $4$ | $\cos 4 + \cos 0.04 = 0.346$ |

Equal offsets give equal scores. The fast pair 0 tells neighbours apart. The slow pair 1 has hardly moved within five tokens; it needs $2\pi / 0.01 \approx 628$ positions for a full turn and tells distant positions apart.

<a id="rope-2d"></a>

**2D: images.** $q_{(h,w)},\ k_{(h',w')} \in \mathbb{R}^{d_k} \;\longrightarrow\; \tilde q_{(h,w)}^{\top}\, \tilde k_{(h',w')} \in \mathbb{R}$, a function of $(h - h',\ w - w')$

An image of $H \times W$ pixels cut into patches of $P \times P$ pixels is a grid of $H/P$ rows and $W/P$ columns, and the tokens of steps 1–3 of [Chapter 4](#4-vlm-from-image-and-question-to-answer) are the patches of that grid, read row after row. The position of a patch is therefore two numbers, while the vectors keep their $d_k$ features:

<a id="rope-2d-notation"></a>

| symbol | meaning |
|---|---|
| $(h, w)$ | row and column of the query patch in the grid, $h \in \{0, \dots, H/P - 1\}$, $w \in \{0, \dots, W/P - 1\}$ |
| $(h', w')$ | row and column of the key patch |
| $q_{(h,w)}$, $k_{(h',w')} \in \mathbb{R}^{d_k}$ | the query vector of the patch at $(h, w)$ and the key vector of the patch at $(h', w')$ |
| $m = d_k / 4$ | pairs per coordinate: the $d_k / 2$ pairs are split into a row half and a column half of $m$ pairs each |
| $\theta_j = 10000^{-j/m}$ | speed of the $j$-th pair of a half, $j = 0, \dots, m - 1$; both halves use the same list, from fast to slow |
| row pair $j$ | features $2j$ and $2j + 1$, turned by the row |
| column pair $j$ | features $2(m + j)$ and $2(m + j) + 1$, turned by the column |

The rule. Row pairs are turned by $h$ times their speed and column pairs by $w$ times their speed; the key is treated the same way with $h'$ and $w'$:

$$
\begin{pmatrix} \tilde q_{2j} \\ \tilde q_{2j+1} \end{pmatrix} = R(h\,\theta_j) \begin{pmatrix} q_{2j} \\ q_{2j+1} \end{pmatrix}, \qquad \begin{pmatrix} \tilde q_{2(m+j)} \\ \tilde q_{2(m+j)+1} \end{pmatrix} = R(w\,\theta_j) \begin{pmatrix} q_{2(m+j)} \\ q_{2(m+j)+1} \end{pmatrix}, \qquad j = 0, \dots, m - 1
$$

As one block matrix, $\tilde q_{(h,w)} = R_{(h,w)}\, q_{(h,w)}$ with

$$
R_{(h,w)} = \operatorname{diag}\big( R(h\,\theta_0), \dots, R(h\,\theta_{m-1}),\ R(w\,\theta_0), \dots, R(w\,\theta_{m-1}) \big) \in \mathbb{R}^{d_k \times d_k}
$$

The score. Call $a_j, b_j$ the row pair $j$ of $q$ and of $k$, and $\bar a_j, \bar b_j$ their column pair $j$. The argument of the 1D case applies to every pair on its own, with $h - h'$ for the row pairs and $w - w'$ for the column pairs:

$$
\begin{aligned}
\tilde q_{(h,w)}^{\top}\, \tilde k_{(h',w')} &= q_{(h,w)}^{\top}\, R_{(h'-h,\ w'-w)}\, k_{(h',w')} \\[4pt]
&= \sum_{j=0}^{m-1} \lVert a_j \rVert\, \lVert b_j \rVert \cos\big(\phi_{a_j} - \phi_{b_j} + (h - h')\,\theta_j\big) \\[4pt]
&\quad + \sum_{j=0}^{m-1} \lVert \bar a_j \rVert\, \lVert \bar b_j \rVert \cos\big(\phi_{\bar a_j} - \phi_{\bar b_j} + (w - w')\,\theta_j\big)
\end{aligned}
$$

The first sum sees only the row offset, the second only the column offset. Both are visible and they are kept apart: moving the key one row up changes the first sum, moving it one column to the left changes the second, and the model can weight the two directions differently by putting different content into row pairs and column pairs. Each half has its own speeds from fast to slow, so each direction can tell neighbours apart as well as far patches.

Worked example: the $2 \times 2$ grid of Chapter 4 with $d_k = 4$, so $m = 1$, one row pair and one column pair, both with $\theta_0 = 1$:

<a id="rope-grid"></a>

| patch | $(h, w)$ | row pair, turned by $h\,\theta_0$ | column pair, turned by $w\,\theta_0$ |
|---|---|---|---|
| $0$ | $(0, 0)$ | $0$ | $0$ |
| $1$ | $(0, 1)$ | $0$ | $1$ rad |
| $2$ | $(1, 0)$ | $1$ rad | $0$ |
| $3$ | $(1, 1)$ | $1$ rad | $1$ rad |

With $q = k = (1, 0, 1, 0)$ for every patch, the score is $\cos(h - h') + \cos(w - w')$:

<a id="rope-2d-scores"></a>

| query patch reads key patch | $(h - h',\ w - w')$ | in the image | score |
|---|---|---|---|
| $3$ at $(1, 1)$ reads $3$ | $(0, 0)$ | itself | $\cos 0 + \cos 0 = 2.000$ |
| $3$ at $(1, 1)$ reads $1$ at $(0, 1)$ | $(1, 0)$ | the patch above | $\cos 1 + \cos 0 = 1.540$ |
| $3$ at $(1, 1)$ reads $2$ at $(1, 0)$ | $(0, 1)$ | the patch to the left | $\cos 0 + \cos 1 = 1.540$ |
| $2$ at $(1, 0)$ reads $1$ at $(0, 1)$ | $(1, -1)$ | above and to the right | $\cos 1 + \cos(-1) = 1.081$ |
| $3$ at $(1, 1)$ reads $0$ at $(0, 0)$ | $(1, 1)$ | above and to the left | $\cos 1 + \cos 1 = 1.081$ |

With the same content in both halves the score cannot tell "above" from "to the left"; a model separates the two by giving its row pairs and column pairs different content. One index is not enough for an image. Written row after row, the grid has the 1D indices $0, 1, 2, 3$:

<a id="rope-flat"></a>

| query patch | key patch | in the image | difference of 1D indices | $(h - h',\ w - w')$ |
|---|---|---|---|---|
| $1$ at $(0, 1)$ | $0$ at $(0, 0)$ | neighbours, left and right | $1$ | $(0, 1)$ |
| $2$ at $(1, 0)$ | $1$ at $(0, 1)$ | diagonal | $1$ | $(1, -1)$ |
| $2$ at $(1, 0)$ | $0$ at $(0, 0)$ | neighbours, above and below | $2$ | $(1, 0)$ |
| $3$ at $(1, 1)$ | $1$ at $(0, 1)$ | neighbours, above and below | $2$ | $(1, 0)$ |

The 1D difference gives the same number to a neighbour and to a diagonal, and its number for "one row below" is the width of the grid, which changes with the resolution. The pair of offsets says what the image says, at any resolution. This is what replaces the learned position table inside the vision encoder from Qwen2-VL on: an image of any size or aspect ratio only changes the range of $h$ and $w$, so no interpolation of positions is needed.

<a id="rope-3d"></a>

**3D: video, and words and images in one sequence.** The position has three numbers: the frame $t$, the row $h$ and the column $w$. Every pair belongs to one of the three coordinates and is turned by that coordinate times its speed, so the score depends on $(t - t',\ h - h',\ w - w')$. This is M-RoPE, used inside the LLM of a vision-language model, where words, image patches and video patches stand in one sequence:

<a id="rope-tokens"></a>

| token | $t$ | $h$ | $w$ |
|---|---|---|---|
| word at position $p$ | $p$ | $p$ | $p$ |
| patch of an image | one value for the whole image | its row | its column |
| patch of a video frame | index of the frame | its row | its column |

For a word the three coordinates are equal, every pair is turned by $p\,\theta_j$, and the rule is exactly the 1D rule: a text-only input behaves like an ordinary LLM. Which pair belongs to which coordinate matters, because the speeds run from fast to slow. With six pairs, $\theta_j = 10000^{-j/6}$:

<a id="rope-pairs"></a>

| pair $j$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| speed $\theta_j$ | $1$ | $0.215$ | $0.046$ | $0.01$ | $0.002$ | $0.0005$ |
| in blocks (Qwen2-VL) | $t$ | $t$ | $h$ | $h$ | $w$ | $w$ |
| interleaved (Qwen3-VL) | $t$ | $h$ | $w$ | $t$ | $h$ | $w$ |

In blocks, the frame index owns only fast pairs. It is a watch with a second hand and no hour hand: good at telling neighbouring frames apart, unable to tell frames that are far apart, which hurts on long video. Interleaving gives every coordinate both fast and slow pairs.

**The three cases side by side.**

<a id="rope-cases"></a>

| | 1D | 2D | 3D |
|---|---|---|---|
| tokens | words | patches of an image | patches of video frames, also mixed with words |
| position of a token | $p$ | $(h, w)$ | $(t, h, w)$ |
| pair $j$ is turned by | $p\,\theta_j$ | $h\,\theta_j$ or $w\,\theta_j$ | $t\,\theta_j$, $h\,\theta_j$ or $w\,\theta_j$ |
| the score depends on | $p - p'$ | $(h - h',\ w - w')$ | $(t - t',\ h - h',\ w - w')$ |
| used in | the layers of an LLM | the layers of a vision encoder | the LLM of a vision-language model |

**Against adding $P$ in step 1.**

<a id="rope-compare"></a>

| | add $P$ (step 1) | rotate (RoPE) |
|---|---|---|
| where | once, at the input | in every attention layer, on $Q$ and $K$ |
| what the score sees | absolute positions, mixed into the content | the offset $p - p'$ only |
| content vector | changed by the addition | length unchanged, $\lVert R_p\, x \rVert = \lVert x \rVert$ |
| parameters | none (sin/cos) or a table with $L_{\max}$ rows | none |
| longer than in training | poor, and impossible with a table | better; very long contexts rescale $\theta_j$ (NTK-aware scaling, YaRN) |

Nothing here is learned: the angles are fixed by the position and by $\theta_j$. What the model learns is where to put the energy of $q$ and $k$, on fast pairs to be sensitive to near offsets and on slow pairs for far ones. RoPE also fits the KV cache of [§7](#7-inference): a key is turned once by its own position and stored, a new query is turned by its position, and their product contains the offset.

**In code.** `transformer_encoder_decoder.py` has one function per case that returns the angle of every pair of every token, `angles_1d`, `angles_2d` and `angles_3d`, and one `rotate` shared by all three; the RoPE part of the notebook runs them. The tests check `rotate` against the explicit matrices $R_p$, the scores of the tables above, and that shifting both tokens by the same amount along any coordinate leaves the score unchanged.

<a id="9-llm-internals"></a>
<a id="91-architecture-decoder-only-the-2026-default"></a>
<a id="decoder-only-llms"></a>

## Decoder-only LLMs: GPT-2, step by step

A decoder-only language model keeps one half of the walkthrough. Drop the encoder (steps 1–4) and the cross-attention of step 7, and what is left is the decoder: embeddings, masked self-attention, feed-forward, logits, loss and generation. The prompt and its continuation form one sequence in one causal stream, and every position predicts the token after it. [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) is the reference design that modern LLMs refine: LLaMA, Qwen and Mistral keep its skeleton ([the comparison](#gpt2-vs-modern) below). Compared with the walkthrough, GPT-2 also moves every LayerNorm to the input of its sub-block. In this section, "step $k$" alone refers to the walkthrough and "GPT-2 step $k$" to the steps below.

**The example.** The source sentence of the walkthrough, `we watch a short video`, is now a prompt for GPT-2 small. GPT-2's byte-level BPE starts from the 256 byte values, so no string is out of vocabulary. Its 50,000 learned merges and one end-of-text token bring the vocabulary to $V = 50257$. Each word is one token, the last four with their leading space, so $n = 5$. The ids are just the tokenizer's row numbers for these five strings. There is no separate decoder input: the labels are the same tokens shifted left by one.

$$
\begin{aligned}
\mathbf{t} &= (t_1,\ t_2,\ t_3,\ t_4,\ t_5) && \text{input, } n = 5 \\[4pt]
\mathbf{y} &= (t_2,\ t_3,\ t_4,\ t_5) && y_j = t_{j+1}
\end{aligned}
$$

Sizes of GPT-2 small: width $d = 768$, $L = 12$ blocks, $h = 12$ heads of $d_k = d/h = 64$, MLP width $d_{ff} = 4d = 3072$, context $n_{\max} = 1024$. Matrices have one row per token and the batch axis is left out. Where the heads are separate, the head axis comes first, as in $[12, 5, 64]$.

**GPT-2 step 1 · Token ids → embeddings + learned positions.** $\mathbf{t} \in \{0,\dots,V-1\}^{n} \;\longrightarrow\; X_0 \in \mathbb{R}^{n \times d}$

$$
\begin{aligned}
X_0 &= E[\mathbf{t}] + P_{1:n} \\[4pt]
E &\in \mathbb{R}^{V \times d} \\[4pt]
P &\in \mathbb{R}^{n_{\max} \times d}
\end{aligned}
$$

$E[\mathbf{t}]$ is a lookup of five rows, as in step 1. $P$ is now learned as well, one free row per position, like the position table of a ViT. There is no row for position $1025$, so the model cannot take a longer input. Both tables start as small random values of comparable scale, so the $\sqrt{d}$ factor is gone. Shapes: $[5] \to [5, 768]$. $X_0$ starts the residual stream, which every later sub-block reads and adds to.

**GPT-2 step 2 · Pre-norm causal self-attention.** $X \in \mathbb{R}^{n \times d} \;\longrightarrow\; X + A \in \mathbb{R}^{n \times d}$

$$
\begin{aligned}
A &= \operatorname{MHA}(\tilde X, \tilde X, \tilde X;\ M^{\text{causal}}), \quad \tilde X = \operatorname{LN}_1(X) \\[4pt]
X &\leftarrow X + A
\end{aligned}
$$

One matrix produces queries, keys and values together, and each is cut into $h$ heads (here $V$ is the value matrix, not the vocabulary size, and $M = M^{\text{causal}}$):

$$
\begin{aligned}
[Q \,\Vert\, K \,\Vert\, V] &= \operatorname{LN}_1(X)\, W^{QKV} + b^{QKV} \\[4pt]
Q &= [\,Q_1 \Vert \cdots \Vert Q_h\,], \quad \text{same for } K, V \\[4pt]
S_i &= Q_i K_i^{\top} / \sqrt{d_k} + M \\[4pt]
\text{head}_i &= \operatorname{softmax}(S_i)\, V_i \\[4pt]
A &= [\,\text{head}_1 \Vert \cdots \Vert \text{head}_h\,]\, W^{O} + b^{O}
\end{aligned}
$$

$W^{QKV} \in \mathbb{R}^{d \times 3d}$ holds the three projections of step 2 side by side, so $[5, 768] \to [5, 2304]$. Each third is reshaped into $12$ heads of $64$ columns, $[12, 5, 64]$. Per head the scores $S_i$ are $[5, 64] \times [64, 5] = [5, 5]$, scaled by $\sqrt{64} = 8$; all heads together hold $[12, 5, 5]$. The causal mask is that of step 6, now $5 \times 5$: `we` reads only itself and `video` reads all five tokens. The weighted values are $[12, 5, 64]$, the heads concatenate back to $[5, 768]$, and $W^{O} \in \mathbb{R}^{d \times d}$ mixes them. $\operatorname{LN}_1$ normalizes each row as in step 3, but only the copy that enters the attention. The stream $X$ itself is never normalized inside a block; it only receives the sum.

**GPT-2 step 3 · Pre-norm MLP.** $X \in \mathbb{R}^{n \times d} \;\longrightarrow\; X + \operatorname{MLP}(\operatorname{LN}_2(X)) \in \mathbb{R}^{n \times d}$

$$
\begin{aligned}
X &\leftarrow X + \operatorname{MLP}\big(\operatorname{LN}_2(X)\big) \\[4pt]
\operatorname{MLP}(x) &= \operatorname{GELU}(x W_1 + b_1)\, W_2 + b_2 \\[4pt]
\operatorname{GELU}(u) &= u\,\Phi(u) \\[4pt]
&\approx \tfrac{u}{2}\big(1 + \tanh(\sqrt{2/\pi}\,(u + 0.044715\,u^3))\big)
\end{aligned}
$$

This is the feed-forward of step 4 with GELU in place of ReLU, applied to each row on its own: $W_1 \in \mathbb{R}^{d \times 4d}$ and $W_2 \in \mathbb{R}^{4d \times d}$ take $[5, 768] \to [5, 3072] \to [5, 768]$. $\Phi$ is the standard normal cumulative distribution function, so GELU is a smooth ReLU that lets small negative inputs through a little. GPT-2 evaluates it with the $\tanh$ approximation of the last line.

**GPT-2 step 4 · Twelve blocks, then a final LayerNorm.** $X_0 \in \mathbb{R}^{n \times d} \;\longrightarrow\; X_f \in \mathbb{R}^{n \times d}$

$$
\begin{aligned}
X_\ell &= \operatorname{Block}_\ell(X_{\ell-1}) && \ell = 1, \dots, L \\[4pt]
X_f &= \operatorname{LN}_f(X_L)
\end{aligned}
$$

A block is GPT-2 steps 2 and 3 with its own weights, and each of the $12$ maps $[5, 768]$ to $[5, 768]$. The extra $\operatorname{LN}_f$ is required by pre-norm. After the last block the stream is the embedding plus $24$ sub-block outputs. It has never been normalized as a whole, so its scale grows with depth. $\operatorname{LN}_f$ normalizes each row once more, with its own $\gamma$ and $\beta$, before the vocabulary projection. A post-norm model needs no final norm, because its last sub-block already ends with a LayerNorm.

**GPT-2 step 5 · Logits from the tied token table.** $X_f \in \mathbb{R}^{n \times d} \;\longrightarrow\; Z \in \mathbb{R}^{n \times V}$

$$
\begin{aligned}
Z &= \operatorname{LN}_f(X_L)\, E^{\top} = X_f\, E^{\top} \\[4pt]
p(t_{j+1} = v \mid t_{1:j}) &= \operatorname{softmax}(Z_j)_v
\end{aligned}
$$

The LM head is the token table of GPT-2 step 1, transposed: no new matrix and no bias. $Z_{j,v}$ scores token $v$ by how well its embedding matches the final vector of position $j$. Shapes: $[5, 768] \times [768, 50257] = [5, 50257]$, the largest single product in the model. Tying saves $Vd = 38.6$M parameters; untied, GPT-2 small would have 163.0M.

**GPT-2 step 6 · Loss: the same tokens, shifted left.** $Z \in \mathbb{R}^{n \times V},\ \mathbf{y} \in \{0,\dots,V-1\}^{n-1} \;\longrightarrow\; \mathcal{L} \in \mathbb{R}$

$$
\begin{aligned}
\mathcal{L} &= -\frac{1}{n-1} \sum_{j=1}^{n-1} \log p_j(y_j) \\[4pt]
p_j &= \operatorname{softmax}(Z_j)
\end{aligned}
$$

Row $1$, which has read only `we`, is scored on `watch`; row $4$, which has read `we watch a short`, is scored on `video`. Row $5$ has no label inside these five tokens. The usual pretraining recipe cuts a window of $n_{\max} + 1$ tokens from one long stream of documents joined by the end-of-text token. All $1024$ positions then have a label, and one forward pass trains $1024$ predictions; the causal mask makes this legal, as with teacher forcing in step 6. The prompt tokens are targets too; instruction tuning usually drops them from the average. Shapes: $Z_{1:4}$ is $[4, 50257]$, $\mathbf{y}$ is $[4]$, and $\mathcal{L}$ is a scalar.

**GPT-2 step 7 · Generation with the KV cache.** $\mathbf{t} \;\longrightarrow\; t_6, t_7, \dots$

$$
\begin{aligned}
t_{n+1} &= \operatorname*{arg\,max}_{v}\ Z_{n,v} && \text{greedy} \\[4pt]
t_{n+1} &\sim \operatorname{softmax}(Z_n / \tau) && \text{sampling}
\end{aligned}
$$

Only the last row of $Z$ is used: the chosen token is appended and the model runs again. Greedy decoding takes the arg max; sampling draws from the softmax at temperature $\tau$. The KV cache stores every layer's $K_i, V_i$ for the positions already seen. After the prompt has been processed once (prefill), each new token goes through the $12$ blocks alone. Its query, $[12, 1, 64]$ per layer, reads the cached keys plus its own, $[12, 6, 64]$ for the first new token. The cache grows by $2 L d = 18{,}432$ numbers per token, 36 KiB in 16-bit floats. Generation stops at end-of-text or at $1024$ positions, where the position table ends.

<a id="coding-transformer-block"></a>

**Post-norm vs pre-norm.** The 2017 Transformer and GPT-1 put the LayerNorm after the residual addition. GPT-2 moved it to the input of each sub-block and added the final $\operatorname{LN}_f$. Count sub-blocks rather than blocks, $\ell = 0, \dots, N - 1$ with $N = 2L = 24$. Let $x_\ell$ be the stream that enters sub-block $\ell$ and $F_\ell$ its attention or MLP:

$$
\begin{aligned}
x_{\ell+1} &= \operatorname{LN}(x_\ell + F_\ell(x_\ell)) && \text{post-norm} \\[4pt]
x_{\ell+1} &= x_\ell + F_\ell(\operatorname{LN}(x_\ell)) && \text{pre-norm}
\end{aligned}
$$

Write $G_k = F_k(\operatorname{LN}(x_k))$ for what pre-norm sub-block $k$ adds. Pre-norm unrolls into a sum, so the Jacobian from any sub-block to the top contains the identity:

$$
\begin{aligned}
x_N &= x_\ell + \sum_{k=\ell}^{N-1} G_k \\[4pt]
\frac{\partial x_N}{\partial x_\ell} &= I + \sum_{k=\ell}^{N-1} \frac{\partial G_k}{\partial x_\ell}
\end{aligned}
$$

With $\ell = 0$ the top is $x_N = x_0 + \sum_{k=0}^{N-1} G_k$: the embeddings plus one correction per sub-block. The gradient $\partial \mathcal{L} / \partial x_N$, which has crossed only the head and $\operatorname{LN}_f$, reaches every sub-block and the embeddings unchanged through $I$. The sub-blocks above only add their own terms to it. Post-norm has no such path: the gradient crosses every LayerNorm Jacobian on the way down.

$$
\begin{aligned}
\frac{\partial x_N}{\partial x_\ell} &= J_{N-1}(I + F'_{N-1}) \cdots J_\ell (I + F'_\ell) \\[4pt]
J_k &= \tfrac{1}{\sigma_k} \operatorname{diag}(\gamma)\, \Pi_k \\[4pt]
\Pi_k &= I - \tfrac{1}{d}\big(\mathbf{1}\mathbf{1}^\top + \hat z_k \hat z_k^\top\big)
\end{aligned}
$$

Here $F'_k = \partial F_k / \partial x_k$, and $J_k$ is the Jacobian of the LayerNorm of sub-block $k$, whose input is $u_k = x_k + F_k(x_k)$. It acts on each token's row separately, and the formula is that of one row: $\sigma_k$ is the standard deviation of that row of $u_k$ and $\hat z_k$ its normalized version. Since $\hat z_k$ has zero mean and squared norm close to $d$, $\Pi_k$ projects out two directions, $\mathbf{1}$ and $\hat z_k$. $J_k$ then rescales by $\gamma / \sigma_k$. No part of the gradient passes through unchanged, so its size at sub-block $\ell$ depends on every layer above it.

[Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) show that at initialization the post-norm gradients of the parameters near the output are large. A large learning rate at the start then makes training unstable, so post-norm needs a warmup. Later work on deeper models (Liu et al. 2020, Wang et al. 2022) found that deep post-norm stacks can diverge even with one. With pre-norm these gradients shrink like $1/\sqrt{L}$, and warmup can be shortened or dropped. Pre-norm has a price. Every sub-block adds to a stream that is never renormalized, so the norm of $x_\ell$ grows with depth and a late sub-block changes it relatively less. The final norm becomes necessary, and a post-norm model that does train is often slightly better (the [interview answer](#interview-pre-norm) lists the compromises).

The same growth sets GPT-2's initialization. If $x_0$ and the $N$ sub-block outputs are uncorrelated, each output with variance $s^2$, the top of the stream has variance $\operatorname{Var}(x_0) + N s^2$, which grows with depth. The paper scales the weights of residual layers by $1/\sqrt{N}$, with $N$ the number of residual layers. Common implementations apply it to the two projections that write into the stream, $W^{O}$ and $W_2$:

$$
\begin{aligned}
W^{O},\ W_2 &\sim \mathcal{N}\big(0,\ \sigma_0^2 / N\big) \\[4pt]
\sigma_0 &= 0.02, \quad N = 2L
\end{aligned}
$$

Then $s^2$ falls like $1/N$ and $N s^2$ stays the same at any depth. For GPT-2 small the standard deviation is $0.02 / \sqrt{24} \approx 0.0041$ instead of $0.02$.

**Parameter count.** Every weight of GPT-2 small, from the shapes above:

<a id="gpt2-params"></a>

| Part | Weights | Count |
|---|---|---|
| token table $E$ | $V d$ | 38,597,376 |
| position table $P$ | $n_{\max} d$ | 786,432 |
| block: $W^{QKV}, b^{QKV}$ | $3d^2 + 3d$ | 1,771,776 |
| block: $W^{O}, b^{O}$ | $d^2 + d$ | 590,592 |
| block: $W_1, b_1$ | $4d^2 + 4d$ | 2,362,368 |
| block: $W_2, b_2$ | $4d^2 + d$ | 2,360,064 |
| block: $\operatorname{LN}_1, \operatorname{LN}_2$ ($\gamma, \beta$) | $4d$ | 3,072 |
| one block | $12d^2 + 13d$ | 7,087,872 |
| all $L = 12$ blocks | $L\,(12d^2 + 13d)$ | 85,054,464 |
| final $\operatorname{LN}_f$ | $2d$ | 1,536 |
| LM head | tied to $E$ | 0 |
| **total** | | **124,439,808** |

The paper lists this size as 117M; the released weights have exactly the count above, 124.4M. The two embedding tables are 31.6% of it. Without biases and norms, a block holds $12d^2$ weights, $4d^2$ in attention and $8d^2$ in the MLP, so $12 L d^2 \approx 84.9$M is the usual estimate of the block stack.

**GPT-2 and a modern decoder.** One row per component that changed:

<a id="gpt2-vs-modern"></a>

| Component | GPT-2 (2019) | LLaMA-style today |
|---|---|---|
| positions | learned table, added at the input | RoPE on $Q_i, K_i$ in every layer |
| norm | LayerNorm, pre-norm | RMSNorm, pre-norm |
| FFN | GELU, width $4d$ | SwiGLU, width $\tfrac{8}{3}d$ to $3.5d$ |
| attention heads | MHA, one $K, V$ per head | GQA, one $K, V$ per group of heads |
| biases | in every linear layer and norm | none |
| LM head | tied to $E$ | often untied |
| vocabulary | 50,257 | 32k to over 128k |
| context | 1,024 | 8k to over 128k |

The skeleton is GPT-2's: a token table, a stack of pre-norm blocks in which causal self-attention and a row-wise MLP add to one residual stream, a final norm, a vocabulary projection, the shifted cross-entropy and cached decoding. The rows only swap parts, so the seven steps above still apply, with [RoPE](#rotary-positions-rope) turning $Q_i$ and $K_i$ in GPT-2 step 2 instead of adding $P$ in GPT-2 step 1.

**Every GPT-2 shape in one table** ($n = 5$, $d = 768$, $h = 12$, $d_k = 64$, $d_{ff} = 3072$, $V = 50257$; the rows of GPT-2 steps 2 and 3 repeat in each of the $12$ blocks; compare the [walkthrough's table](#transformer-shapes)).

<a id="gpt2-shapes"></a>

| Step | Input | Output |
|---|---|---|
| 1 · token + position embedding | $\mathbf{t}$: $[5]$ | $X_0$: $[5, 768]$ |
| 2 · $\operatorname{LN}_1$, fused $W^{QKV}$ | $[5, 768]$ | $[5, 2304]$ |
| 2 · split into heads | $[5, 2304]$ | $Q, K, V$: $[12, 5, 64]$ each |
| 2 · scores $Q_i K_i^{\top} / 8$ + causal mask | $[12, 5, 64] \times [12, 64, 5]$ | $[12, 5, 5]$ |
| 2 · weights $\times\, V_i$ | $[12, 5, 5] \times [12, 5, 64]$ | $[12, 5, 64]$ |
| 2 · concatenate, $W^{O}$, add to stream | $[5, 768]$ | $[5, 768]$ |
| 3 · $\operatorname{LN}_2$, $W_1$, GELU | $[5, 768]$ | $[5, 3072]$ |
| 3 · $W_2$, add to stream | $[5, 3072]$ | $[5, 768]$ |
| 4 · 12 blocks, then $\operatorname{LN}_f$ | $X_0$: $[5, 768]$ | $X_f$: $[5, 768]$ |
| 5 · tied LM head $E^{\top}$ | $[5, 768] \times [768, 50257]$ | $Z$: $[5, 50257]$ |
| 6 · cross-entropy | $Z_{1:4}$: $[4, 50257]$, $\mathbf{y}$: $[4]$ | $\mathcal{L}$: scalar |
| 7 · one cached decode step | new token: $[1]$ | $Z$: $[1, 50257]$ |

<a id="vision-transformers"></a>
## Vision Transformers, from ViT to today

The encoder of this chapter does not care what its tokens are. Give it image patches instead of words and it is a Vision Transformer (ViT): [the same step for an image](#step-1-images) above and steps 1–3 of [Chapter 4](#4-vlm-from-image-and-question-to-answer) follow one tiny image through it. This section is the family that grew out of that model. Each member changes some of five choices, and the [model line](#vit-model-line) at the end compares them on these choices.

1. **Tokens.** How pixels become tokens: square patches, a pyramid of grids, tubes of video, or patches of any aspect ratio.
2. **Attention range.** Which keys a query reads: every token, a local window, a pooled set of keys, or one axis of space-time at a time.
3. **Shape.** One grid from the first layer to the last, like ViT, or a pyramid that halves the grid and widens the channels at each stage, like a CNN.
4. **Positions.** A learned table added to the tokens, a relative bias inside the softmax, or the [2D RoPE](#rotary-positions-rope) above.
5. **Training signal.** Labels, a teacher's labels, masked reconstruction, self-distillation, or captions ([CLIP and SigLIP](#vlm-model-line) in Chapter 4).

<a id="vit-example"></a>

**The example.** One RGB image, to be classified into the $K = 1000$ ImageNet classes. A plain ViT cuts it into $16 \times 16$ patches; the hierarchical models start four times finer:

$$
\begin{aligned}
I &\in \mathbb{R}^{224 \times 224 \times C}, \quad C = 3 \text{ channels} \\[4pt]
P = 16: \quad N &= 14 \times 14 = 196 \ \text{patches of}\ 16 \cdot 16 \cdot 3 = 768 \ \text{numbers} \\[4pt]
P = 4: \quad N &= 56 \times 56 = 3136 \ \text{patches of}\ 4 \cdot 4 \cdot 3 = 48 \ \text{numbers}
\end{aligned}
$$

Where a width is needed, the sizes are those of ViT-B/16: $D = 768$, $L = 12$ layers, $12$ heads, 86M parameters. The code of this section, `vision_transformers.py` ([note at the end](#coding-vit)), implements the core of ViT, DeiT, Swin, PVT, MAE, DINO, TimeSformer, ViViT, NaViT and DiT at these sizes; ViTDet, SAM, BEiT and the later DINO versions are described but not coded.

**ViT · patches, a class token, a learned position table.** $I \in \mathbb{R}^{224 \times 224 \times 3} \;\longrightarrow\; \hat y \in \mathbb{R}^{1000}$

$$
\begin{aligned}
z_0 &= \big[\, x_{\text{cls}};\ x_p^1 E;\ \dots;\ x_p^N E \,\big] + E_{\text{pos}}, && E \in \mathbb{R}^{P^2 C \times D},\ \ E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D} \\[4pt]
z_\ell &= \operatorname{Block}(z_{\ell-1}), && \ell = 1, \dots, L \\[4pt]
\hat y &= \operatorname{LN}\big(z_L^0\big)\, W_{\text{head}}, && W_{\text{head}} \in \mathbb{R}^{D \times K}
\end{aligned}
$$

$x_p^i$ is patch $i$ flattened, and Block is the pre-norm block of Chapter 4, step 3, with no mask. The class token $x_{\text{cls}}$ is a learned row in front of the patches. It has no pixels of its own, so its output $z_L^0$ holds only what attention gathered from the patches, and the classifier reads that one row; the paper finds that averaging the patch outputs works as well once the learning rate is tuned. Shapes: $[224, 224, 3] \to [196, 768] \to [197, 768] \to [1000]$. Two facts that interviews probe:

- **Data.** Trained on ImageNet-1k alone, ViT-B/16 reaches 77.9%, a few points below ResNets of its size. Pretrained on JFT-300M, it matches or beats the best CNNs (BiT-L 87.5%, Noisy Student 88.5%) for less pretraining compute, up to 88.55% for ViT-H/14; pretrained on ImageNet-21k it comes close, 85.3% for ViT-L/16. The paper's explanation: ViT has much less image-specific inductive bias than a CNN, so locality and translation equivariance must be learned from data ([CNN vs Transformer](#76-cnn-vs-transformer)).
- **Resolution.** Pretraining is at 224 px and fine-tuning at 384 px with the same $P = 16$, so $N$ grows to $24 \times 24 = 576$ and the learned table no longer fits. It is resized as a $14 \times 14$ image of vectors, and the class-token row is kept:

$$
E_{\text{pos}}^{\,24 \times 24} = \operatorname{interp}\big(E_{\text{pos}}^{\,14 \times 14}\big) \qquad \text{linear in the ViT code, bicubic in DeiT}
$$

The patching and this resize are the only places where ViT is told that the image is two-dimensional.

**The cost of one attention sublayer.** $(N, D) \;\longrightarrow\; \Omega$, in multiply–adds

$$
\Omega_{\text{global}} = 4ND^2 + 2N^2 D, \qquad \Omega_{\text{window}} = 4ND^2 + 2M^2 N D
$$

The first term is the four projections $W_Q, W_K, W_V, W_O$ and is linear in $N$. The second is $QK^\top$ plus the weighted sum of $V$. It is quadratic in $N$ for global attention and becomes linear when every query reads only the $M \times M$ tokens of its window. These are equations 1 and 2 of the Swin paper, which leave out the softmax.

<a id="vit-cost"></a>

| Layer | $N$ | $D$ | $4ND^2$ | Second term | Its share |
|---|---|---|---|---|---|
| ViT-B/16 at 224 px | 197 | 768 | 465M | 60M | 11% |
| ViT-B/16 at 1024 px | 4,096 | 768 | 9.66G | 25.8G | 73% |
| Swin stage 1, global | 3,136 | 96 | 116M | 1.89G | 94% |
| Swin stage 1, $7 \times 7$ windows | 3,136 | 96 | 116M | 29.5M | 20% |

At 224 px the attention matrix of ViT is cheap. It dominates when the tokens are many: high resolution, small patches, video. Every variant below that limits the attention range does so to shrink the second term: windows (Swin, ViTDet) make it linear in $N$, reduced keys (PVT) divide it by $R^2$, and divided space-time attention (TimeSformer) divides it by $TN/(T + N)$.

**DeiT · ImageNet only, plus a distillation token.** $I \;\longrightarrow\; (Z_{\text{cls}},\ Z_{\text{dist}})$, two vectors in $\mathbb{R}^{1000}$

$$
\begin{aligned}
z_0 &= \big[\, x_{\text{cls}};\ x_{\text{dist}};\ x_p^1 E;\ \dots;\ x_p^N E \,\big] + E_{\text{pos}}, && E_{\text{pos}} \in \mathbb{R}^{(N+2) \times D} \\[4pt]
\mathcal{L} &= \tfrac12\, \operatorname{CE}\big(\psi(Z_{\text{cls}}),\ y\big) + \tfrac12\, \operatorname{CE}\big(\psi(Z_{\text{dist}}),\ y_t\big), && y_t = \arg\max_c Z_t(c)
\end{aligned}
$$

$\psi$ is the softmax, $y$ the true label, $Z_t$ the logits of the teacher, a CNN (RegNetY-16GF, 82.9%), and $Z_{\text{cls}}, Z_{\text{dist}}$ are linear heads on the outputs of the two tokens. The distillation token is a second learned row trained to reproduce the teacher's hard decision instead of the label; at test time the softmax outputs of the two heads are added. Otherwise the network is ViT-B. What changed is the recipe: ImageNet-1k alone (1.28M images), 300 epochs, strong augmentation (RandAugment, Mixup, CutMix, random erasing), stochastic depth instead of dropout, AdamW. DeiT-B reaches 81.8% at 224 px after 53 hours on one 8-GPU node, and 83.4% with the distillation token, above its own teacher. A CNN teacher works better than a transformer teacher, plausibly because the student inherits some of its inductive bias.

**Swin · windows, shifted windows, patch merging.** $I \;\longrightarrow\; 56^2 \times 96,\ \ 28^2 \times 192,\ \ 14^2 \times 384,\ \ 7^2 \times 768$

Tokens are $4 \times 4$ patches, $48$ numbers each, embedded to a width $C = 96$; from here to PVT, $C$ is the token width, not the 3 image channels. Attention runs inside non-overlapping $M \times M$ windows, $M = 7$, and adds a learned bias for the relative position of the two tokens:

$$
\begin{aligned}
\operatorname{Attn}(Q, K, V) &= \operatorname{softmax}\big(QK^\top / \sqrt{d_k} + B\big)\, V && \text{per head, over the } M^2 \text{ tokens of one window} \\[4pt]
B_{ij} &= \hat B\big[\, y_i - y_j + M - 1,\ \ x_i - x_j + M - 1 \,\big], && \hat B \in \mathbb{R}^{(2M-1) \times (2M-1)}
\end{aligned}
$$

$d_k = C/h$ is the head width, $32$ in Swin-T. On the $56 \times 56$ grid that is $64$ windows of $49$ tokens; per head, $B$ is a $49 \times 49$ matrix read from a $13 \times 13$ table. Swin has no position table at all. Windows alone would never exchange information, so consecutive blocks alternate between the regular partition (W-MSA) and one displaced by $\lfloor M/2 \rfloor = 3$ tokens in both directions (SW-MSA):

$$
\begin{aligned}
\hat z^{\ell} &= \text{W-MSA}\big(\operatorname{LN}(z^{\ell-1})\big) + z^{\ell-1}, & z^{\ell} &= \operatorname{MLP}\big(\operatorname{LN}(\hat z^{\ell})\big) + \hat z^{\ell} \\[4pt]
\hat z^{\ell+1} &= \text{SW-MSA}\big(\operatorname{LN}(z^{\ell})\big) + z^{\ell}, & z^{\ell+1} &= \operatorname{MLP}\big(\operatorname{LN}(\hat z^{\ell+1})\big) + \hat z^{\ell+1}
\end{aligned}
$$

Displacing the partition would cut extra, smaller windows at the border. Instead the map is rolled by $-3$ along both axes and the same $64$ windows are used. A window can then hold pieces of up to four regions $r$ that were not neighbours in the image, so their pairs are masked before the softmax, and the map is rolled back by $+3$:

$$
\operatorname{mask}_{ij} = \begin{cases} 0 & r(i) = r(j) \\ -100 & \text{otherwise: a weight below } 10^{-43} \text{ after the softmax} \end{cases}
$$

$-100$ is the official code's value; the code below masks with $-\infty$, which gives exact zeros. The cost is linear because $M$ is fixed; the shift adds connections across windows, not cost. Between stages, patch merging halves the grid and doubles the width:

$$
x'_{u,v} = \operatorname{LN}\big(\big[\, x_{2u,2v};\ x_{2u+1,2v};\ x_{2u,2v+1};\ x_{2u+1,2v+1} \,\big]\big)\, W_{\text{merge}}, \qquad W_{\text{merge}} \in \mathbb{R}^{4C \times 2C}
$$

Swin-T: $C = 96$, blocks $(2, 2, 6, 2)$ per stage, heads $(3, 6, 12, 24)$, 29M parameters, 4.5 GFLOPs, 81.3%. The four maps have strides 4, 8, 16 and 32, those of a ResNet, so Swin drops into FPN, Mask R-CNN or UPerNet unchanged. That, more than ImageNet accuracy, made it the default backbone for detection and segmentation in 2021. At stage 4 the $7 \times 7$ map is a single window, so the shift is switched off and attention is global.

**PVT · keys and values reduced by $R$.** $Z \in \mathbb{R}^{N \times C} \;\longrightarrow\; Z' \in \mathbb{R}^{N \times C}$

$$
\begin{aligned}
\operatorname{SRA}(Q, K, V) &= \operatorname{Attn}\big(Q W^Q,\ \operatorname{SR}(K)\, W^K,\ \operatorname{SR}(V)\, W^V\big) \\[4pt]
\operatorname{SR}(x) &= \operatorname{LN}\big(\operatorname{Reshape}(x, R)\, W^S\big), \qquad W^S \in \mathbb{R}^{R^2 C \times C}
\end{aligned}
$$

$\operatorname{Reshape}(x, R)$ turns each $R \times R$ block of tokens into one row of $R^2 C$ numbers and $W^S$ maps it back to $C$, a learned stride-$R$ projection rather than a pooling, so the $N$ queries read $N / R^2$ keys and values. Stage 1 of the example has $3136$ queries and $R = 8$, so $49$ keys; with $R = 8, 4, 2, 1$ over the four stages, every stage reads a $7 \times 7$ grid of keys at 224 px. PVT is a pyramid like Swin, with widths $64, 128, 320, 512$, but makes the opposite trade: every query still sees the whole image, coarsely, where a Swin query sees a small region exactly. The cost falls by $R^2$ yet stays quadratic in $N$; the linear variant of PVTv2 pools the keys to a fixed $7 \times 7$ grid instead.

**MAE · mask 75% of the patches, encode only the rest.** $I \;\longrightarrow\; \hat x \in \mathbb{R}^{N \times P^2 C}$, pixels for all $N$ patches, of which only the hidden ones are scored

$$
\begin{aligned}
\mathcal{V} &\subset \{1, \dots, N\}, \quad |\mathcal{V}| = (1 - 0.75)\, N = 49 && \text{uniform, without replacement} \\[4pt]
h &= \operatorname{Enc}\big(\{\, x_p^i E + E_{\text{pos}}^i : i \in \mathcal{V} \,\}\big) && \text{49 tokens, no mask tokens} \\[4pt]
\hat x &= \operatorname{Dec}\big(\operatorname{unshuffle}[\, \operatorname{LN}(h)\, W_{\text{dec}};\ m, \dots, m \,] + E_{\text{pos}}^{\text{dec}}\big) && W_{\text{dec}} \in \mathbb{R}^{D \times 512},\ m \in \mathbb{R}^{512} \\[4pt]
\mathcal{L} &= \frac{1}{|\mathcal{M}|} \sum_{j \in \mathcal{M}} \frac{\big\lVert \hat x_j - \tilde x_j \big\rVert^2}{P^2 C}, \qquad \tilde x_j = \frac{x_p^j - \mu_j}{\sigma_j} && \mathcal{M} = \text{the hidden patches}
\end{aligned}
$$

The encoder, a full ViT, sees only the $49$ visible patches; there are no mask tokens in it. A light decoder (8 blocks of width 512, 9% of ViT-L's compute per token and about 30% of ViT-B's) receives the encoded patches, projected to its width, plus one shared learned vector $m$ for every hidden patch, put back in image order; its last linear layer predicts $P^2 C = 768$ pixel values per token. The loss counts only the hidden patches, and the target is each patch normalized by its own mean $\mu_j$ and standard deviation $\sigma_j$. After pretraining the decoder is thrown away and the encoder is fine-tuned on all patches. Three reasons it works. Images are redundant, so with 75% hidden the task cannot be solved by copying neighbours and needs a model of objects. The encoder runs on a quarter of the tokens, so pretraining is 3× or more faster, which makes huge models affordable. And cropping and flipping are all the augmentation it needs. ImageNet-1k fine-tuned: ViT-B 83.6%, ViT-L 85.9%, ViT-H 86.9%, 87.8% at 448 px, from ImageNet-1k alone. Its predecessor BEiT kept all 196 tokens in the encoder, replaced about 40% of them by a mask embedding, and classified each into one of 8,192 visual tokens produced by DALL-E's discrete VAE, a relative of the [VQ-VAE](#621-vq-vae); VideoMAE hides 90% of a video's space-time tubes.

**DINO · self-distillation without labels.** crops of $I$ $\;\longrightarrow\;$ a loss, no label

$$
\begin{aligned}
P_s(x) &= \operatorname{softmax}\big(g_{\theta_s}(x) / \tau_s\big), \qquad P_t(x) = \operatorname{softmax}\big((g_{\theta_t}(x) - c) / \tau_t\big) \\[4pt]
\mathcal{L} &= \sum_{x \in \{x^g_1,\, x^g_2\}} \ \sum_{x' \in V,\ x' \neq x} -P_t(x)^\top \log P_s(x') \\[4pt]
\theta_t &\leftarrow \lambda\, \theta_t + (1 - \lambda)\, \theta_s, \qquad c \leftarrow m\, c + (1 - m)\, \tfrac{1}{B} \textstyle\sum_{b} g_{\theta_t}(x_b)
\end{aligned}
$$

One image gives a set $V$ of crops: two global crops $x^g_1, x^g_2$ at 224 px and several local crops at 96 px. The student $g_{\theta_s}$, a ViT plus a projection head with $K = 65{,}536$ outputs, sees all crops; the teacher $g_{\theta_t}$ sees only the global ones. The student is trained to give, from a small local crop, the same distribution the teacher gives from a large view of the same image: local-to-global correspondence. The teacher is not trained; it is an exponential moving average of the student, with $\lambda$ from $0.996$ to $1$. Two operations on the teacher's output prevent collapse to a constant: centering by $c$, a running mean of the teacher's outputs over batches of $B$ images with momentum $m = 0.9$ (not MAE's mask token), stops one dimension from dominating, and sharpening with a low temperature ($\tau_t = 0.04$ to $0.07$ against $\tau_s = 0.1$) stops the output from becoming uniform. Each pushes the opposite way, and either one alone collapses. No labels and no negative pairs. The code below, like the official one, divides the sum by the number of crop pairs. The self-attention of the class token in the last layer turns out to segment the main object without any supervision. DINOv2 scales this to a curated set of 142M images and a 1.1B-parameter ViT-g/14, adds a masked-patch loss (iBOT), and gives frozen features good enough for classification, segmentation and depth without fine-tuning. Large ViTs trained long, with or without labels (DINOv2, DeiT-III, OpenCLIP), grow a few high-norm tokens in empty background patches; they hold global information and spoil the attention maps. Four learned register tokens, appended to the input and dropped at the output, give that information a place to go. DINOv3 grows to a 6.7B-parameter ViT/16 with RoPE. Its dense features degrade during very long training, so a late refinement phase adds Gram anchoring: a loss $\lVert X_S X_S^\top - X_G X_G^\top \rVert_F^2$, where $X_S$ and $X_G$ hold the L2-normalized patch features of the student and of an early teacher checkpoint, which pulls the student's patch-to-patch cosine-similarity matrix back to the early one and repairs the dense features.

**ViTDet and SAM · a plain backbone for detection and segmentation.** $I \in \mathbb{R}^{1024 \times 1024 \times 3} \;\longrightarrow\; F \in \mathbb{R}^{64 \times 64 \times D} \;\longrightarrow\; P_2, \dots, P_5$

The opposite answer to Swin: keep the pretrained ViT plain, one grid at stride 16, and build the pyramid only at the end, from the last map $F$:

$$
P_5 = \operatorname{MaxPool}_{2 \times 2}(F), \qquad P_4 = F, \qquad P_3 = \operatorname{Deconv}_{2 \times 2}(F), \qquad P_2 = \operatorname{Deconv}_{2 \times 2}\big(\operatorname{Deconv}_{2 \times 2}(F)\big)
$$

Each level then passes a $1 \times 1$ and a $3 \times 3$ convolution to 256 channels, and a LayerNorm and GELU sit between the two deconvolutions of $P_2$. At 1024 px the backbone carries $4096$ tokens, the row of the cost table where global attention is most of the work. ViTDet therefore uses $14 \times 14$ windows, the grid size of pretraining and not shifted, in most blocks and global attention in four blocks spread evenly through the depth (blocks 3, 6, 9 and 12 of ViT-B). Windows, the four global blocks and the pyramid are added only for fine-tuning; the backbone is pretrained as a plain ViT with MAE, which adds 3.1 box AP for ViT-B and 4.6 for ViT-L over random initialization, while supervised ImageNet-1k pretraining is slightly worse than random initialization (−0.5 and −0.4). Mask R-CNN on COCO reaches 51.6, 55.6 and 56.7 box AP with ViT-B, L and H. SAM's image encoder is this design: an MAE-pretrained ViT-H/16 at 1024 px with $14 \times 14$ windows in 28 blocks and global attention in 4, and a neck to a $256 \times 64 \times 64$ embedding. It runs once per image; only the light prompt encoder and mask decoder run for each click.

**TimeSformer and ViViT · attention over space and time.** $X \in \mathbb{R}^{T \times H \times W \times 3} \;\longrightarrow\; z \in \mathbb{R}^{T \times N \times D}$

With $T = 8$ frames of the example, each cut into $N = 196$ patches, joint attention over all $TN = 1568$ tokens costs $(TN)^2 \approx 2.46$M pairs per head. TimeSformer's divided attention splits every block into a temporal and a spatial step, each with its own weights:

$$
\begin{aligned}
z' &= z + \operatorname{MSA}_{\text{time}}\big(\operatorname{LN}(z)\big) && \text{token } (t, p) \text{ reads } (t', p) \text{ for all frames } t' \\[4pt]
z'' &= z' + \operatorname{MSA}_{\text{space}}\big(\operatorname{LN}(z')\big) && \text{token } (t, p) \text{ reads } (t, p') \text{ for all patches } p' \\[4pt]
z''' &= z'' + \operatorname{MLP}\big(\operatorname{LN}(z'')\big)
\end{aligned}
$$

Each query now reads $T + N = 204$ keys instead of $TN = 1568$, $TN(T + N) \approx 0.32$M pairs per head, about $7.7$ times fewer, and two steps still connect every pair of tokens. The cost is parameters: separate temporal weights take ViT-B from 86M to 121M. Order matters a little: time first, then space, is 0.5% better on Kinetics than the reverse. ViViT changes the tokens: a tubelet of $t \times P \times P$ pixels, a 3D convolution with kernel and stride $(2, 16, 16)$, makes one token from two frames, which halves $T$ before attention starts; initializing it with the 2D patch filter on frame $\lfloor t/2 \rfloor$ and zeros elsewhere, which the paper calls central-frame initialization, works best in its ablation (79.2% against 77.6% for copying the filter to every frame, with $t = 4$). ViViT compares four models: joint attention and three factorizations, from a separate spatial and temporal encoder to splitting the heads between space and time. The same factorization appears in step 5 of the [Cosmos tokenizer](#cosmos-tokenizer).

**NaViT · images of any shape in one packed sequence.** images $I_k \in \mathbb{R}^{H_k \times W_k \times 3}$ $\;\longrightarrow\;$ one sequence of $\sum_k N_k$ tokens

$$
\begin{aligned}
N_k &= \frac{H_k}{P} \cdot \frac{W_k}{P}, \qquad E_{\text{pos}}(y, x) = \phi_y(y) + \phi_x(x) \\[4pt]
\operatorname{mask}_{ij} &= \begin{cases} 0 & \text{tokens } i \text{ and } j \text{ come from the same image} \\ -\infty & \text{otherwise} \end{cases}
\end{aligned}
$$

ViT resizes every image to one square. NaViT keeps each image's aspect ratio, patches it, and concatenates the patches of several images into one sequence of fixed length, as LLM training packs documents. The block-diagonal mask keeps each image to itself, pooling is per image, and positions come from one table per axis, so any grid has positions. Example: a $224 \times 160$ and a $96 \times 224$ image give $14 \cdot 10 = 140$ and $6 \cdot 14 = 84$ tokens, one sequence of $224$. During training the resolution is sampled per image and a random fraction of tokens is dropped, so the model sees about five times more images for the same compute, and NaViT matches the best ViT with four times less compute. Qwen2-VL's encoder in Chapter 4 takes the whole idea: it cites NaViT and packs images of different sizes into one sequence, with 2D RoPE in place of a position table. SigLIP 2's NaFlex variant takes only the native aspect ratio: one padded image per sequence, with its learned position table resized to the patch grid.

**DiT · a transformer as the diffusion denoiser.** $x_t \in \mathbb{R}^{32 \times 32 \times 4},\ t,\ y \;\longrightarrow\; (\hat\epsilon, \hat\Sigma)$, both $32 \times 32 \times 4$

The noisy latent of a 256 px image, from the Stable Diffusion VAE, is cut into $p \times p = 2 \times 2$ patches, $T = (32/2)^2 = 256$ tokens. The timestep $t$ and the class $y$ steer every block through adaLN-Zero:

$$
\begin{aligned}
(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2) &= W \operatorname{SiLU}(e_t + e_y), && W \in \mathbb{R}^{6D \times D},\ W = 0 \text{ at first} \\[4pt]
x &\leftarrow x + \alpha_1 \odot \operatorname{MSA}\big((1 + \gamma_1) \odot \operatorname{LN}(x) + \beta_1\big) \\[4pt]
x &\leftarrow x + \alpha_2 \odot \operatorname{MLP}\big((1 + \gamma_2) \odot \operatorname{LN}(x) + \beta_2\big)
\end{aligned}
$$

$e_t$ is a sinusoidal embedding of $t$ passed through an MLP and $e_y$ a learned class embedding, both in $\mathbb{R}^D$. The LayerNorms have no learned scale and shift of their own; $\gamma$ and $\beta$ come from the condition, one value per channel. With $W = 0$ every $\alpha$ is $0$, so each block starts as the identity. On ImageNet 256 after 400K steps, adaLN-Zero reaches FID 19.5 against 25.2 for adaLN, 26.1 for cross-attention and 35.2 for appending the condition as tokens. The last layer maps each token to $p \times p \times 2 \cdot 4$ numbers, the noise and a diagonal covariance. DiT-XL/2 (28 layers, $D = 1152$, 119 GFLOPs) reaches FID 2.27 with classifier-free guidance, and FID falls steadily as GFLOPs grow, whether by a larger configuration (S to XL, deeper and wider together) or by smaller patches. It is the denoiser of [diffusion](#632-diffusion) written as a transformer, the design that later video and image generators build on.

**The model line.** Ordered by the first arXiv version.

<a id="vit-model-line"></a>

| Model | Tokens and shape | Attention | Trained with | What it changed |
|---|---|---|---|---|
| [ViT](https://arxiv.org/abs/2010.11929), 2020 | $16 \times 16$ patches, one grid | global | labels on JFT-300M or ImageNet-21k | A plain Transformer on patches matches the best CNNs, given enough data. |
| [DeiT](https://arxiv.org/abs/2012.12877), 2020 | as ViT, plus a distillation token | global | ImageNet-1k and a CNN teacher | ViT without extra data, in about two days on one 8-GPU node: augmentation, regularization, distillation. |
| [TimeSformer](https://arxiv.org/abs/2102.05095), 2021 | 8 frames of $16 \times 16$ patches | time, then space | ImageNet weights, then video labels | Video at a fraction of joint attention's cost. |
| [PVT](https://arxiv.org/abs/2102.12122), 2021 | pyramid, strides 4 to 32 | keys and values reduced by $R$ | labels | The first pyramid ViT, a backbone for dense prediction. |
| [Swin](https://arxiv.org/abs/2103.14030), 2021 | pyramid by patch merging | shifted $7 \times 7$ windows, relative bias | labels | Linear cost and ResNet strides: the 2021 detection and segmentation backbone. |
| [ViViT](https://arxiv.org/abs/2103.15691), 2021 | $2 \times 16 \times 16$ tubelets | joint and three factorizations compared | ImageNet or JFT weights, then video labels | Tokens that span time, and a map of how to factorize space-time. |
| [DINO](https://arxiv.org/abs/2104.14294), 2021 | one grid | global | self-distillation, no labels | Label-free features; attention maps segment objects. |
| [BEiT](https://arxiv.org/abs/2106.08254), 2021 | one grid, about 40% masked | global | masked patches to dVAE token ids | BERT's masked prediction for images. |
| [MAE](https://arxiv.org/abs/2111.06377), 2021 | the encoder sees 25% of the patches | global | masked pixel reconstruction | Simple, fast self-supervised pretraining that scales to ViT-H. |
| [ViTDet](https://arxiv.org/abs/2203.16527), 2022 | one grid, pyramid built at the end | $14 \times 14$ windows plus 4 global blocks | MAE, then detection | A plain ViT is enough for detection. |
| [DiT](https://arxiv.org/abs/2212.09748), 2022 | $2 \times 2$ patches of a VAE latent | global, conditioned by adaLN-Zero | diffusion denoising | The transformer replaces the U-Net in diffusion; quality follows GFLOPs. |
| [SAM](https://arxiv.org/abs/2304.02643), 2023 | ViT-H/16 at 1024 px | windows plus 4 global blocks | MAE, then 1.1B masks | Promptable segmentation on the ViTDet design. |
| [DINOv2](https://arxiv.org/abs/2304.07193), 2023 | $14 \times 14$ patches | global | DINO and iBOT on 142M curated images | Frozen general-purpose features for classification, segmentation and depth. |
| [NaViT](https://arxiv.org/abs/2307.06304), 2023 | any aspect ratio, packed | global within each image | labels or image–text pairs | Native resolution, the ancestor of dynamic-resolution VLM encoders. |
| [Registers](https://arxiv.org/abs/2309.16588), 2023 | plus 4 register tokens | global | as the base model | Removes the high-norm artifact tokens of large ViTs. |
| [DINOv3](https://arxiv.org/abs/2508.10104), 2025 | ViT-7B/16 with RoPE and registers | global | DINO and iBOT, then Gram anchoring | Repairs dense features that degrade during very long training. |

**What the line shows.**

- **Tokens**: square patches on one grid (ViT), then pyramids like a CNN's (PVT, Swin) and tubes of video (TimeSformer, ViViT), then one grid again with the pyramid built at the end (ViTDet), and finally any aspect ratio (NaViT).
- **Attention range**: global attention is restricted only where the tokens are many: windows for high-resolution detection and segmentation (Swin, ViTDet, SAM), spatially reduced keys (PVT), one axis at a time for video (TimeSformer). At 224 px plain global attention stayed, and [FlashAttention](#75-efficient-attention) made it cheaper still.
- **Training signal**: labels on 300M images (ViT), then ImageNet with a teacher (DeiT), then no labels at all (DINO, BEiT, MAE), and image–text pairs (CLIP, SigLIP). Image–text pretraining is still the usual route for VLM encoders, and SigLIP 2 adds masked prediction and self-distillation to it as extra losses; label-free self-distillation (DINOv2, DINOv3) gives the strongest frozen dense features.
- **What won**: the plain ViT. It is the encoder of CLIP, SigLIP and every VLM in Chapter 4, the denoiser of DiT and the backbone of SAM; hierarchical designs survive mostly as detection and segmentation backbones.

<a id="coding-vit"></a>

**Code.** Read it in VS Code: `vision_transformers.py` writes the mechanisms of ViT, DeiT, registers, Swin, PVT, MAE, DINO, TimeSformer, ViViT, NaViT and DiT as small modules, in the order above, and `test_vision_transformers.py` checks the fast implementations against direct ones, such as shifted windows against masked global attention in image coordinates, and pins each formula of this section.

---

<a id="10-vlm--mllm"></a>
# 4. VLM: From Image and Question to Answer

One image and one question, followed through every equation of the classic projector-based vision-language model: a ViT vision encoder, a small MLP projector, and a decoder-only LLM ([LLaVA](https://arxiv.org/abs/2304.08485), [LLaVA-1.5](https://arxiv.org/abs/2310.03744)). The format is that of [Chapter 3](#3-transformer-from-input-to-output): each step states its input, the operation, and its output, with the toy shapes next to the general ones and the real LLaVA-1.5 sizes in the final table. Attention ($\operatorname{MHA}$), $\operatorname{FFN}$, $\operatorname{LN}$, the causal mask, and cross-entropy are those of Chapter 3 and are not derived again. The [PyTorch implementation](#pytorch-implementation-vit--projector--llm) at the end transcribes these equations and trains the toy model.

This is your TikTok and AquaVLM territory — expect the deepest questioning here.

<a id="vlm-math"></a>

**The example.** A tiny RGB image $I$ of a cat, the question `what is this ?`, and the answer `a cat`. The text is one token sequence with a placeholder `<image>` where the picture goes:

$$
\begin{aligned}
\text{image}\quad I &\in \mathbb{R}^{H \times W \times C} && H = W = 8,\ C = 3 \\[6pt]
\text{text ids}\quad \mathbf{t} &= (\underbrace{1}_{\text{BOS}},\ \underbrace{3}_{\langle\text{image}\rangle},\ \underbrace{4}_{\text{what}},\ \underbrace{5}_{\text{is}},\ \underbrace{6}_{\text{this}},\ \underbrace{7}_{?},\ \underbrace{8}_{\text{a}},\ \underbrace{9}_{\text{cat}}) && m = 8 \\[6pt]
\text{answer}\quad \mathbf{a} &= (\underbrace{8}_{\text{a}},\ \underbrace{9}_{\text{cat}},\ \underbrace{2}_{\text{EOS}}) && \text{the only tokens trained on}
\end{aligned}
$$

Sizes used throughout: patch size $P = 4$, so $N = HW / P^2 = 4$ patches; vision width $d_v = 8$; LLM width $d = 16$; $h = 2$ heads in both; vocabulary $V = 13$ (`<pad>, <bos>, <eos>, <image>, what, is, this, ?, a, cat, dog, photo, of`). Once the placeholder is replaced by the $N$ image tokens, the LLM sees $L = m - 1 + N = 11$ positions. PyTorch puts a batch axis $B$ in front of every shape and stores images channel-first, $[B, C, H, W]$.

**Step 1 · Image → patches.** $I \in \mathbb{R}^{H \times W \times C} \;\longrightarrow\; X_p \in \mathbb{R}^{N \times P^2 C}$

$$
X_p[k] = \operatorname{vec}\big(I[\,rP : (r+1)P,\ \ cP : (c+1)P,\ \ :\,]\big), \qquad k = r\,\tfrac{W}{P} + c, \qquad N = \tfrac{H}{P} \cdot \tfrac{W}{P}
$$

Cut the image into a grid of non-overlapping $P \times P$ squares and flatten each one into a row. A patch plays the role of a word, and reading the grid left to right, top to bottom gives the "sentence". Shapes: $[8, 8, 3] \to [4, 48]$.

**Step 2 · Patches → patch embeddings + positions.** $X_p \in \mathbb{R}^{N \times P^2 C} \;\longrightarrow\; Z_0 \in \mathbb{R}^{N \times d_v}$

$$
Z_0 = X_p W^{E} + b^{E} + P^{\text{vis}}, \qquad W^{E} \in \mathbb{R}^{P^2 C \times d_v}, \qquad P^{\text{vis}} \in \mathbb{R}^{N \times d_v} \ \text{(learned)}
$$

Pixels are continuous, so there is no vocabulary and no lookup: a linear layer takes the place of the embedding table of Chapter 3, and [step 1 of Chapter 3](#step-1-images) shows the two usual ways to build $P^{\text{vis}}$. Steps 1–2 together equal one convolution with kernel $P$ and stride $P$, which is how ViTs implement them. Without $P^{\text{vis}}$ the encoder could not tell where a patch came from. CLIP's ViT also prepends a learned `[CLS]` token that feeds its contrastive loss; LLaVA drops it and keeps the $N$ patch tokens. Shapes: $[4, 48] \to [4, 8]$.

**Step 3 · Vision encoder (ViT layers).** $Z_0 \in \mathbb{R}^{N \times d_v} \;\longrightarrow\; Z \in \mathbb{R}^{N \times d_v}$

$$
Z'_{\ell} = Z_{\ell-1} + \operatorname{MHA}\big(\operatorname{LN}(Z_{\ell-1}),\ \operatorname{LN}(Z_{\ell-1}),\ \operatorname{LN}(Z_{\ell-1})\big), \qquad Z_{\ell} = Z'_{\ell} + \operatorname{FFN}\big(\operatorname{LN}(Z'_{\ell})\big), \qquad \ell = 1, \dots, N_v
$$

The block of Chapter 3 in pre-norm order: $\operatorname{LN}$ is applied to the input of each sublayer and the residual path is left untouched. There is no mask, so every patch reads every patch in both directions; an image has no future to hide. Scores are $[4, 4]$ per head, and the FFN uses GELU instead of ReLU. $Z = Z_{N_v}$ is one contextual vector per patch. LLaVA takes the output of the second-to-last layer, because the last one is specialized for CLIP's contrastive objective.

**Step 4 · Projector.** $Z \in \mathbb{R}^{N \times d_v} \;\longrightarrow\; U \in \mathbb{R}^{N \times d}$

$$
U = \operatorname{GELU}(Z W_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d_v \times d}, \quad W_2 \in \mathbb{R}^{d \times d}
$$

A two-layer MLP applied to each patch vector on its own, like an FFN. It changes the width from the vision encoder's $d_v$ to the LLM's $d$ and, once trained, the meaning: each row of $U$ becomes a vector the LLM can read as if it were a word embedding. It is the only new module; the two big models on either side are pretrained separately. The number of tokens stays $N$. Shapes: $[4, 8] \to [4, 16]$.

**Step 5 · Text ids → embeddings.** $\mathbf{t} \in \{0, \dots, V-1\}^{m} \;\longrightarrow\; T \in \mathbb{R}^{m \times d}$

$$
T = E[\mathbf{t}], \qquad E \in \mathbb{R}^{V \times d}
$$

The LLM's own embedding table, a lookup as in Chapter 3. The row that belongs to `<image>` is a placeholder and is thrown away in the next step. Shapes: $[8] \to [8, 16]$.

**Step 6 · Splice the image tokens into the text.** $T \in \mathbb{R}^{m \times d},\ U \in \mathbb{R}^{N \times d} \;\longrightarrow\; X_0 \in \mathbb{R}^{L \times d}$

$$
X_0 = \big[\, T_{\text{BOS}};\ \ U_1;\ U_2;\ U_3;\ U_4;\ \ T_{\text{what}};\ T_{\text{is}};\ T_{\text{this}};\ T_{?};\ T_{\text{a}};\ T_{\text{cat}} \,\big] + P^{\text{txt}}_{1:L}, \qquad L = m - 1 + N
$$

Stack the rows, with the $N$ rows of $U$ in place of the single placeholder row. From here on the LLM cannot tell an image token from a word: both are rows in $\mathbb{R}^{d}$. The toy adds learned positions $P^{\text{txt}}$; real LLMs rotate queries and keys inside attention instead ([RoPE, §3](#rotary-positions-rope)). Shapes: $[8, 16]$ and $[4, 16] \to [11, 16]$.

**Step 7 · LLM layers with causal self-attention.** $X_0 \in \mathbb{R}^{L \times d} \;\longrightarrow\; X \in \mathbb{R}^{L \times d}$

$$
X'_{\ell} = X_{\ell-1} + \operatorname{MHA}\big(\operatorname{LN}(X_{\ell-1}),\ \operatorname{LN}(X_{\ell-1}),\ \operatorname{LN}(X_{\ell-1});\ M^{\text{causal}}\big), \qquad X_{\ell} = X'_{\ell} + \operatorname{FFN}\big(\operatorname{LN}(X'_{\ell})\big)
$$

The block of step 3 plus the causal mask of Chapter 3, now $L \times L = 11 \times 11$. There is no cross-attention and no encoder–decoder split: the word `what` at position 5 reads the image tokens at positions 1–4 through ordinary self-attention, because they stand earlier in the same sequence. That is why the image is placed before the question. Scores are $[11, 11]$ per head.

**Step 8 · Logits.** $X \in \mathbb{R}^{L \times d} \;\longrightarrow\; Z^{\text{out}} \in \mathbb{R}^{L \times V}$

$$
Z^{\text{out}} = \operatorname{LN}(X)\, W^{\text{vocab}}, \qquad W^{\text{vocab}} \in \mathbb{R}^{d \times V}, \qquad p\big(\text{next} = v \,\big|\, \text{positions} \le j\big) = \operatorname{softmax}(Z^{\text{out}}_j)_v
$$

The output is text only: every row scores the word vocabulary, including the rows at image positions, which nobody reads. Shapes: $[11, 16] \to [11, 13]$.

**Step 9 · Loss on the answer only.** $Z^{\text{out}} \in \mathbb{R}^{L \times V},\ \mathbf{y} \;\longrightarrow\; \mathcal{L} \in \mathbb{R}$

| position $j$ | input at $j$ | label $y_j$ (the next token) |
|---|---|---|
| 0 | BOS | ignore |
| 1–4 | image tokens $U_1, \dots, U_4$ | ignore |
| 5–7 | what, is, this | ignore |
| 8 | ? | a |
| 9 | a | cat |
| 10 | cat | EOS |

$$
\mathcal{L} = -\frac{1}{|A|} \sum_{j \in A} \log \operatorname{softmax}(Z^{\text{out}}_j)_{y_j}, \qquad A = \{8, 9, 10\}
$$

The cross-entropy of Chapter 3, averaged over the answer positions $A$ only. An image position has no label because what follows it is a continuous vector, not a word; the question is given, not something to learn to produce. With several turns, $A$ holds the positions of every assistant answer.

**Step 10 · What is trained.** $\mathcal{L} \;\longrightarrow\; \nabla_{\theta}\, \mathcal{L}$ for the trainable part of $\theta = (\theta_{\text{vis}},\ \theta_{\text{proj}},\ \theta_{\text{LLM}})$

The weights fall into three groups with three different histories:

| group | contains | toy / LLaVA-1.5 | comes from |
|---|---|---|---|
| $\theta_{\text{vis}}$, steps 1–3 | $W^{E}, b^{E}, P^{\text{vis}}$, ViT layers | $2{,}104$ / $304$M | CLIP contrastive pretraining |
| $\theta_{\text{proj}}$, step 4 | $W_1, b_1, W_2, b_2$ | $416$ / $21$M | nothing: random start |
| $\theta_{\text{LLM}}$, steps 5–8 | $E$, LLM layers, final $\operatorname{LN}$, $W^{\text{vocab}}$ | $7{,}904$ / $7$B | language pretraining |

The vision encoder and the LLM are downloaded already trained, each on its own task: CLIP learned to match images with their captions, the LLM learned to predict the next word. Only the projector between them starts from noise. Training the VLM then takes two stages that minimize the same loss $\mathcal{L}$ of step 9 over different sets of weights:

$$
\text{stage 1 (alignment):}\quad \min_{\theta_{\text{proj}}} \mathcal{L} \qquad\qquad \text{stage 2 (instruction tuning):}\quad \min_{\theta_{\text{proj}},\ \theta_{\text{LLM}}} \mathcal{L}
$$

| | stage 1: alignment | stage 2: instruction tuning |
|---|---|---|
| data | image + caption, $558$K pairs | image + question + answer, $665$K |
| updated | $\theta_{\text{proj}}$ | $\theta_{\text{proj}}$, $\theta_{\text{LLM}}$ (full or LoRA) |
| frozen | $\theta_{\text{vis}}$, $\theta_{\text{LLM}}$ | $\theta_{\text{vis}}$ |
| learning rate | $10^{-3}$ | $2 \times 10^{-5}$ |
| result | the frozen LLM can read $U$ | the LLM follows instructions about the image |

Every update is ordinary gradient descent on the trainable set. In stage 1:

$$
\theta_{\text{proj}} \leftarrow \theta_{\text{proj}} - \eta\, \nabla_{\theta_{\text{proj}}} \mathcal{L}, \qquad \nabla_{\theta_{\text{proj}}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial U} \cdot \frac{\partial U}{\partial \theta_{\text{proj}}}
$$

Frozen means not updated, not skipped. The forward pass still runs through the frozen LLM, and $\partial \mathcal{L} / \partial U$ is obtained by backpropagating from the loss through every frozen LLM layer down to the image positions of $X_0$; the optimizer simply holds no weights of the LLM.

The order of the stages matters. At the start the projector emits noise, and letting that noise update a pretrained LLM would damage its language ability. So the small new module is trained first, with a large learning rate, until what it writes looks like input the LLM already understands; only then is the LLM allowed to move, with a learning rate fifty times smaller. The vision encoder stays frozen because its features are already good and updating it is expensive; later models unfreeze it with a small learning rate to gain fine detail and OCR.

**Step 11 · Generation.** $I,\ \text{question} \;\longrightarrow\; \hat{\mathbf{a}}$

$$
U = \operatorname{Proj}\big(\operatorname{ViT}(I)\big) \ \text{once}; \qquad \hat t_{j+1} = \operatorname*{arg\,max}_{v}\ Z^{\text{out}}_j\big([\,\text{BOS};\ U;\ \text{question};\ \hat t_{\le j}\,]\big)_v, \qquad \text{stop at EOS}
$$

The image is encoded once and its tokens sit in the prefix, so with a KV cache ([Chapter 7, step 2](#51-kv-cache--the-thing-that-dominates-inference)) they are paid for in prefill only. The prefix has $1 + N$ positions plus the question, and $N$ dominates: $576$ image tokens against a question of perhaps $20$. That makes the image resolution the main cost lever of a VLM.

<a id="vlm-shapes"></a>

**Every shape in one table.** Toy sizes as above; the last column is LLaVA-1.5 (CLIP ViT-L/14 at 336 px, Vicuna-7B).

| Step | Input | Output | LLaVA-1.5 output |
|---|---|---|---|
| 1 · patches | $I$: $[8, 8, 3]$ | $X_p$: $[4, 48]$ | $[576, 588]$ from $[336, 336, 3]$, $P = 14$ |
| 2 · patch embedding + position | $X_p$: $[4, 48]$ | $Z_0$: $[4, 8]$ | $[576, 1024]$ |
| 3 · ViT layers (scores $[4, 4]$) | $Z_0$: $[4, 8]$ | $Z$: $[4, 8]$ | $[576, 1024]$, layer 23 of 24 |
| 4 · projector | $Z$: $[4, 8]$ | $U$: $[4, 16]$ | $[576, 4096]$ |
| 5 · text embeddings | $\mathbf{t}$: $[8]$ | $T$: $[8, 16]$ | $[m, 4096]$ |
| 6 · splice + position | $T$: $[8, 16]$, $U$: $[4, 16]$ | $X_0$: $[11, 16]$ | $[m - 1 + 576,\ 4096]$ |
| 7 · LLM layers (scores $[11, 11]$) | $X_0$: $[11, 16]$ | $X$: $[11, 16]$ | $[L, 4096]$, 32 layers |
| 8 · logits | $X$: $[11, 16]$ | $Z^{\text{out}}$: $[11, 13]$ | $[L, 32000]$ |
| 9 · loss over $A$ | $Z^{\text{out}}$: $[11, 13]$, $\mathbf{y}$: $[11]$ | $\mathcal{L}$: scalar | scalar |

Attention in the LLM costs $O(L^2 d)$ with $L \approx N + m$. Doubling the side of the image gives $4N$ patches, so about $16\times$ the attention work in prefill and $4\times$ the KV cache, before a single word of the answer is produced.

<a id="vlm-models"></a>
## Classic models, from CLIP to today

Every model in the line answers the same four questions, and they are the steps of the walkthrough: which vision encoder (steps 1–3), how the image reaches the language model (steps 4 and 6), how many image tokens that costs (the $N$ of step 1), and what is trained (step 10). The history below is the history of those four choices.

<a id="102-vlm-architecture-families"></a>
<a id="132-vlm-architecture-families"></a>

**Three ways for the image to reach the language model.** With $Z \in \mathbb{R}^{N \times d_v}$ the patch features of step 3:

1. **Contrastive dual encoder.** No language model at all: an image encoder $f$ and a text encoder $g$ are trained so that matching pairs score high, $s(I, T) = \cos\big(f(I),\ g(T)\big) / \tau$. It can rank and retrieve but cannot write text. CLIP, SigLIP.
2. **Projection into the input** (this chapter). $U = \operatorname{MLP}(Z) \in \mathbb{R}^{N \times d}$ is spliced into the token sequence and read by ordinary self-attention. Every patch is kept, so the token count follows the resolution. Simple, cheap, dominant: LLaVA, InternVL, Qwen2-VL and its successors, Gemma 3, MobileVLM.
3. **Resampler.** $U = \operatorname{MHA}(Q, Z, Z) \in \mathbb{R}^{K \times d}$: $K$ learned query vectors $Q$ cross-attend to the patches, so the output has $K$ tokens whatever $N$ is. It compresses hard and loses fine detail. BLIP-2's Q-Former with $K = 32$, Qwen-VL with $K = 256$.

**The model line.** The column "way in" uses the numbers of the list above.

<a id="vlm-model-line"></a>

| Model | Vision encoder | Way in | Image tokens | What it changed |
|---|---|---|---|---|
| [CLIP](https://arxiv.org/abs/2103.00020), 2021 | ViT or ResNet | contrastive (1) | one vector per image | Image and text in one embedding space, from 400M web pairs. Its ViT became the vision encoder of most later models. |
| [BLIP-2](https://arxiv.org/abs/2301.12597), 2023 | EVA-CLIP ViT-g, frozen | Q-Former (3) | 32 | A small trainable bridge between two frozen models. |
| [LLaVA](https://arxiv.org/abs/2304.08485), 2023 | CLIP ViT-L/14 at 224 px | one linear layer (2) | 256 | The simplest connector, plus instruction data written by GPT-4. |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744), 2023 | CLIP ViT-L/14 at 336 px | two-layer MLP (2) | 576 | The model of this chapter: a strong baseline from 1.2M samples in about a day of training. |
| [Qwen-VL](https://arxiv.org/abs/2308.12966), 2023 | OpenCLIP ViT-bigG at 448 px | one cross-attention layer (3) | 256 | Boxes written as text, which gives grounding and text reading. |
| [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/), 2024 | CLIP ViT-L/14, 336 px tiles | MLP (2) | up to 2,880 | AnyRes: tiles at native resolution plus one downsampled global view. |
| [InternVL 1.5](https://arxiv.org/abs/2404.16821), 2024 | InternViT-6B, 448 px tiles | pixel shuffle, then MLP (2) | 256 per tile, up to 40 tiles | A 6B vision encoder, and 4× fewer tokens per tile. |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191), 2024 | ViT 675M with 2D-RoPE | 2×2 merge, then MLP (2) | $HW / 28^2$ | Native dynamic resolution without tiles. M-RoPE gives time, height and width their own position components. |
| [PaliGemma](https://arxiv.org/abs/2407.07726), 2024 | SigLIP-So400m | one linear layer (2) | 256, 1,024 or 4,096 | SigLIP in place of CLIP. Image and prompt attend to each other in both directions. |
| [SigLIP 2](https://arxiv.org/abs/2502.14786), 2025 | is the encoder | contrastive (1) | one vector per image | The sigmoid loss of [SigLIP](https://arxiv.org/abs/2303.15343) plus captioning, self-distillation and masked prediction. The usual vision encoder of recent models. |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), 2025 | ViT from scratch, window attention | 2×2 merge, then MLP (2) | follows the resolution | Encoder cost linear in the number of patches. Time ids tied to absolute time for long video. |
| [Gemma 3](https://arxiv.org/abs/2503.19786), 2025 | SigLIP 400M at 896 px, frozen | average pooling, then projection (2) | 256 | A fixed budget; Pan & Scan crops at inference for other aspect ratios. |
| [InternVL3](https://arxiv.org/abs/2504.10479), 2025 | InternViT, 448 px tiles | pixel shuffle, then MLP (2) | 256 per tile | Native multimodal pretraining: text and multimodal data in one stage, instead of adapting a text-only LLM. |
| [Qwen3-VL](https://arxiv.org/abs/2511.21631), 2025 | SigLIP-2 | 2×2 merge, then MLP (2), plus DeepStack | follows the resolution | ViT features from three depths are added to the first three LLM layers. Video timestamps as text. 256K context. |
| [Gemma 4](https://arxiv.org/abs/2607.02770), 2026 | ViT with 2D positions and RoPE, native aspect ratio | projection (2) | 70, 140, 280, 560 or 1,120 | Native aspect ratio, and the token budget is a setting chosen per image. |

**What the line shows.**

- **Vision encoder**: CLIP, then SigLIP and SigLIP 2, then encoders built for variable resolution (2D-RoPE, window attention, native aspect ratio).
- **Way in**: the resampler was the early answer because it keeps the sequence short. Plain projection won, because keeping every patch preserves the detail that OCR and grounding need; compression moved to a cheap 2×2 merge in front of the MLP.
- **Image tokens**: one fixed low resolution, then tiles, then native dynamic resolution. The token count has become a budget that the user sets.
- **Training**: two frozen models and a small bridge (BLIP-2), then the two stages of step 10 (LLaVA), then joint training of all parts at scale (Qwen2-VL and later), and finally multimodal data in pretraining itself (InternVL3).

<a id="coding-vlm"></a>
## PyTorch implementation: ViT + projector + LLM

[vlm_vit_projector_llm.py](./vlm_vit_projector_llm.py) implements the eleven steps and reuses the attention module of Chapter 3. It is a toy with random initial weights, so the demo also plays the part of pretraining. [The tests](./test_vlm_vit_projector_llm.py) check that patching equals a strided convolution, that the vision encoder is bidirectional while the LLM is causal, that only answer positions carry loss, and that each stage updates only its own parameters.

```bash
.venv-transformer/bin/python vlm_vit_projector_llm.py                   # runs the example below
.venv-transformer/bin/python -m unittest -v test_vlm_vit_projector_llm  # 10 tests
```

<!-- vlm-code: vlm-imports -->

**Steps 1 and 2 — patches and patch embeddings.** `patchify` is the $\operatorname{vec}$ of step 1 written as reshape and permute; the linear layer is $W^{E}$ and the parameter `positions` is $P^{\text{vis}}$.

<!-- vlm-code: vlm-patches -->

**Steps 3 and 7 — one pre-norm block, two uses.** Without a mask it is a ViT layer; with the causal mask it is an LLM layer.

<!-- vlm-code: vlm-blocks -->

**Steps 3 and 4 — vision encoder and projector.**

<!-- vlm-code: vlm-vision -->

**Steps 5 to 11 — the whole model, the answer loss, the stages, and generation.** `splice` finds the placeholder and puts the image tokens in its place; `splice_labels` gives the image positions the label `IGNORE`; `set_stage` switches `requires_grad`; `generate` encodes the image once.

<!-- vlm-code: vlm-model -->

**The example, step by step, then trained.** `main()` prints the shape of every step for the cat example. It then trains on two examples with the same question, a cat image answered `a cat` and a dog image answered `a dog`, so the answer can only come from the image. Stage 0 stands in for a pretrained LLM: the image slot holds the four words of a caption (`photo of a cat`), and the LLM learns to answer from them. Stage 1 puts the real image tokens in the slot and trains the projector alone; stage 2 trains projector and LLM together. The toy vision encoder is never trained: its random features are already enough to tell the two images apart, which is the part CLIP pretraining plays in a real model.

<!-- vlm-code: vlm-demo -->

Its output:

```text
t = [1, 3, 4, 5, 6, 7, 8, 9] = <bos> <image> what is this ? a cat 

step 0   I (channel-first)                  [1, 3, 8, 8]
step 1   X_p = patchify(I)                  [1, 4, 48]
step 2   Z0 = X_p W_E + b_E + P_vis         [1, 4, 8]
step 3   Z = ViT(Z0)                        [1, 4, 8]
step 3       attention weights, per head    [1, 2, 4, 4]
step 4   U = projector(Z)                   [1, 4, 16]
step 5   T = E[t]                           [1, 8, 16]
step 6   X0 = splice(T, U) + P_txt          [1, 11, 16]
step 7       causal weights, per head       [1, 2, 11, 11]
step 8   Z_out = LN(X) W_vocab              [1, 11, 13]

labels y = [-100, -100, -100, -100, -100, -100, -100, -100, 8, 9, 2]
step 9   loss over positions 8, 9, 10 = 2.1354  (untrained)

step 10  stage 0 (LLM, caption words in the image slot): 7904 of 10424 parameters, loss 2.357 -> 0.001
step 10  stage 1 (projector only, real image tokens): 416 of 10424 parameters, loss 1.371 -> 0.001
step 10  stage 2 (projector + LLM): 8320 of 10424 parameters, loss 0.001 -> 0.000
step 11  cat image: <bos> <image> what is this ? a cat <eos>
step 11  dog image: <bos> <image> what is this ? a dog <eos>
```

With an untrained projector the frozen LLM finds noise in the image slot and the loss jumps from $0.001$ to $1.371$. Training $416$ projector parameters brings it back to $0.001$: the projector has learned to write the image in a form the LLM already reads. The projected tokens need not coincide with the embedding of any word; they only need to have the same effect on the LLM.


---

<a id="11-post-training-sft-rlhf-dpo-grpo"></a>
# 5. Post-Training: LoRA

Post-training is everything done to a pretrained model to shape its behaviour: supervised fine-tuning on demonstrations, then preference optimization such as DPO or GRPO. Each stage minimizes a loss over the weights exactly as in steps 9–10 of [Chapter 3](#3-transformer-from-input-to-output); what differs between stages is the data and the loss. The question they all share is which weights are allowed to move and what that costs, and the standard answer is LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)): keep every pretrained matrix frozen and learn a small low-rank correction next to it. This chapter follows one weight matrix through LoRA, with the input and output of every step, and ends with a PyTorch implementation that post-trains the toy Transformer of Chapter 3.

Your resume says "SFT & RL post-training": expect to be asked what exactly is trained, why it fits in memory, and what it cannot do.

<a id="lora-math"></a>

**The example.** The toy Transformer of Chapter 3 ($d = 8$, $h = 2$, $d_{ff} = 32$) is first trained on two sentence pairs, which stands in for pretraining, and then post-trained on a third pair, `we watch video` → `nous regardons video`, that it gets wrong. Every attention module holds four matrices $W^{Q}, W^{K}, W^{V}, W^{O} \in \mathbb{R}^{8 \times 8}$, and LoRA with $r = 2$ is put on all of them. The real-size reference is LLaMA-7B, where the same matrices are $4096 \times 4096$ and $r = 16$ is typical.

**Where LoRA enters Chapter 3.** Nothing in the walkthrough is rewritten; a few of its matrices get a correction next to them:

<a id="lora-where"></a>

| Chapter 3 step | matrices | in the toy | with LoRA |
|---|---|---|---|
| step 1, embeddings $E$ and positions $P$ | frozen | | unchanged |
| step 2, projections $W^{Q}, W^{K}, W^{V}, W^{O}$ | adapted | $[8, 8]$ each | $W^{Q} + \tfrac{\alpha}{r} W^{Q}_{\text{down}} W^{Q}_{\text{up}}$, likewise $K$, $V$, $O$ |
| step 3, LayerNorm $\gamma, \beta$ | frozen | | unchanged |
| step 4, FFN $W_1, W_2$ | frozen in the toy, adapted in most LLM recipes | $[8, 32]$, $[32, 8]$ | $W_1 + \tfrac{\alpha}{r} W_{1,\text{down}} W_{1,\text{up}}$ |
| step 6, decoder self-attention projections | adapted | $[8, 8]$ each | as in step 2 |
| step 7, cross-attention projections | adapted | $[8, 8]$ each | as in step 2 |
| step 9, $W^{\text{vocab}}$ | frozen | | unchanged |

Everything below is written for $W^{Q}$ of step 2 and holds word for word for $W^{K}$, $W^{V}$, $W^{O}$ and for the projections of steps 6 and 7.

**Notation.** All symbols are those of Chapter 3, step 2.

<a id="lora-notation"></a>

| symbol | meaning |
|---|---|
| $X \in \mathbb{R}^{n \times d}$ | the input of the attention block, one row per token (step 2) |
| $W^{Q} \in \mathbb{R}^{d \times d}$ | the query projection, frozen: the `Linear(d, d)` of step 2 that holds $W_1^{Q}, \dots, W_h^{Q}$ side by side |
| $Q = X W^{Q} \in \mathbb{R}^{n \times d}$ | the queries of all heads, split into $Q_1, \dots, Q_h$ afterwards exactly as in step 2 |
| $r$ | the rank of the correction, $r \ll d$ |
| $W^{Q}_{\text{down}} \in \mathbb{R}^{d \times r}$, $W^{Q}_{\text{up}} \in \mathbb{R}^{r \times d}$ | the two trainable matrices added next to $W^{Q}$; the LoRA paper calls them $A$ and $B$, letters that Chapter 3 already uses |
| $\alpha$ | a scale hyperparameter; the correction is multiplied by $\alpha / r$ |
| $\Delta W^{Q} = \frac{\alpha}{r} W^{Q}_{\text{down}} W^{Q}_{\text{up}} \in \mathbb{R}^{d \times d}$ | the correction, of rank at most $r$ |
| $\theta_{\text{LoRA}}$ | every $W_{\text{down}}$ and $W_{\text{up}}$ of the model; $\theta_{\text{frozen}}$ is everything else |
| $G = \partial \mathcal{L} / \partial Q \in \mathbb{R}^{n \times d}$ | the gradient of the loss arriving at $Q$ |

**Step 1 · Which weights.** $\theta \;\longrightarrow\; (\theta_{\text{frozen}},\ \theta_{\text{LoRA}})$

Every pretrained parameter is frozen. For each adapted projection a fresh pair $(W_{\text{down}}, W_{\text{up}})$ is added, as in the table above. In the toy the three attention modules give $12$ projections, so $\theta_{\text{LoRA}}$ has $12 \cdot 2 \cdot (8 + 8) = 384$ parameters out of $2{,}578$. In LLaMA-7B, all linear layers with $r = 16$ give about $40$M trainable parameters out of $6.7$B, that is $0.6\%$.

**Step 2 · The frozen projection of Chapter 3, step 2.** $X \in \mathbb{R}^{n \times d} \;\longrightarrow\; Q_0 \in \mathbb{R}^{n \times d}$

$$
Q_0 = X\, W^{Q}
$$

This is the first line of step 2, unchanged: $[5, 8] \times [8, 8] \to [5, 8]$. $Q_0$ is the query matrix the pretrained model would have used.

**Step 3 · The low-rank correction.** $X \in \mathbb{R}^{n \times d} \;\longrightarrow\; \Delta Q \in \mathbb{R}^{n \times d}$

$$
\Delta Q = \frac{\alpha}{r}\, \big(X\, W^{Q}_{\text{down}}\big)\, W^{Q}_{\text{up}}, \qquad X\, W^{Q}_{\text{down}} \in \mathbb{R}^{n \times r}
$$

Multiply by $W^{Q}_{\text{down}}$ first, then by $W^{Q}_{\text{up}}$: $[5, 8] \times [8, 2] = [5, 2]$, then $[5, 2] \times [2, 8] = [5, 8]$. The product $\Delta W^{Q} = \frac{\alpha}{r} W^{Q}_{\text{down}} W^{Q}_{\text{up}}$ is an $8 \times 8$ matrix whose rows are all combinations of the $r = 2$ rows of $W^{Q}_{\text{up}}$, so its rank is at most $r$: that is the whole restriction, and the whole saving. $\Delta W^{Q}$ is never formed during training; the correction costs $2rd$ multiplications per token against $d^2$ for the frozen projection.

**Step 4 · The queries that the rest of step 2 sees.** $Q_0,\ \Delta Q \;\longrightarrow\; Q \in \mathbb{R}^{n \times d}$

$$
Q = Q_0 + \Delta Q = X\,\big(W^{Q} + \tfrac{\alpha}{r} W^{Q}_{\text{down}} W^{Q}_{\text{up}}\big)
$$

From here step 2 continues exactly as written in Chapter 3: split $Q$ into the heads $Q_i$, score them against $K_i$, softmax, weighted sum of $V_i$, concatenate, multiply by $W^{O}$. $K$, $V$ and $W^{O}$ get the same treatment with their own pairs. $W_{\text{up}}$ is initialized to zero and $W_{\text{down}}$ at random, so at the start $\Delta W^{Q} = 0$ and $Q = Q_0$: post-training begins exactly at the pretrained model.

**Step 5 · The loss of Chapter 3, step 10.** $Z \;\longrightarrow\; \mathcal{L}$

$$
\mathcal{L} = -\frac{1}{m} \sum_{j=1}^{m} \log \operatorname{softmax}(Z_j)_{y_j}
$$

The loss is the one of the stage being run: here the next-token cross-entropy over the target tokens, `PAD` labels dropped; for preference optimization it is a preference loss on pairs of answers. LoRA changes neither; it only changes which parameters the gradient of $\mathcal{L}$ is allowed to reach.

**Step 6 · Gradients: only $W_{\text{down}}$ and $W_{\text{up}}$ receive them.** $G = \partial \mathcal{L} / \partial Q \;\longrightarrow\; \partial \mathcal{L} / \partial W^{Q}_{\text{down}},\ \partial \mathcal{L} / \partial W^{Q}_{\text{up}}$

Backpropagation from step 10 down to step 2 delivers $G$ at $Q$, exactly as in Chapter 3. From $Q = X W^{Q} + \frac{\alpha}{r} (X W^{Q}_{\text{down}}) W^{Q}_{\text{up}}$ and the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial W^{Q}_{\text{up}}} = \frac{\alpha}{r}\, \big(X W^{Q}_{\text{down}}\big)^{\top} G \in \mathbb{R}^{r \times d}, \qquad \frac{\partial \mathcal{L}}{\partial W^{Q}_{\text{down}}} = \frac{\alpha}{r}\, X^{\top} \big( G\, W^{Q\top}_{\text{up}} \big) \in \mathbb{R}^{d \times r}
$$

$$
\frac{\partial \mathcal{L}}{\partial X} = G\, W^{Q\top} + \frac{\alpha}{r}\, \big( G\, W^{Q\top}_{\text{up}} \big) W^{Q\top}_{\text{down}} \qquad \text{plus the same terms through } W^{K} \text{ and } W^{V}
$$

No gradient is computed for $W^{Q}$, but $W^{Q}$ still appears in $\partial \mathcal{L} / \partial X$, which carries the signal down to step 1 and to the layers below: frozen means not updated, not skipped. At initialization $W^{Q}_{\text{up}} = 0$ makes $\partial \mathcal{L} / \partial W^{Q}_{\text{down}} = 0$ while $\partial \mathcal{L} / \partial W^{Q}_{\text{up}} = \frac{\alpha}{r} (X W^{Q}_{\text{down}})^{\top} G \neq 0$, so $W_{\text{up}}$ moves first and $W_{\text{down}}$ follows; had both started at zero, both gradients would stay zero forever, which is why $W_{\text{down}}$ is random.

**Step 7 · The update and what it costs in memory.** $\partial \mathcal{L} / \partial \theta_{\text{LoRA}} \;\longrightarrow\; \theta_{\text{LoRA}}$

$$
W^{Q}_{\text{down}} \leftarrow W^{Q}_{\text{down}} - \eta\, \frac{\partial \mathcal{L}}{\partial W^{Q}_{\text{down}}}, \qquad W^{Q}_{\text{up}} \leftarrow W^{Q}_{\text{up}} - \eta\, \frac{\partial \mathcal{L}}{\partial W^{Q}_{\text{up}}} \qquad \text{(or the Adam version of the same)}
$$

The optimizer keeps state only for $\theta_{\text{LoRA}}$. That is where the memory goes in full fine-tuning: with AdamW in mixed precision, every trainable parameter needs its bf16 weight, a bf16 gradient, an fp32 master copy and two fp32 moments, about $16$ bytes.

<a id="lora-memory"></a>

| LLaMA-7B, before activations | full fine-tuning | LoRA, $r = 16$ on all linear layers |
|---|---|---|
| trainable parameters | $6.7$B | about $40$M |
| gradients, master copy, Adam moments | about $110$ GB | about $0.6$ GB |
| frozen weights | none | $13.5$ GB in bf16, $3.4$ GB in 4-bit (QLoRA) |

**Step 8 · Merge for inference.** $W^{Q},\ W^{Q}_{\text{down}},\ W^{Q}_{\text{up}} \;\longrightarrow\; W^{Q\prime} \in \mathbb{R}^{d \times d}$

$$
W^{Q\prime} = W^{Q} + \frac{\alpha}{r} W^{Q}_{\text{down}} W^{Q}_{\text{up}}, \qquad Q = X\, W^{Q\prime}
$$

Done once after training, step 2 is again the plain $Q = X W^{Q\prime}$ of Chapter 3 with a matrix of the original shape: no extra parameters, no extra latency. Subtracting the same term restores $W^{Q}$, and one frozen model can serve several tasks by swapping the small pairs.

**Step 9 · Rank, scale, and where.** $(r,\ \alpha,\ \text{which matrices}) \;\longrightarrow\; \text{capacity and cost}$

The rank bounds how much can be learned: $r = 8$ to $16$ is enough to change format or style, $r = 64$ to $128$ is used when a genuinely new capability is taught. $\alpha$ is usually set to $2r$, which keeps the size of $\Delta Q$ comparable when $r$ changes. Adapting the FFN matrices $W_1, W_2$ of step 4 as well as the projections of step 2 is closer to full fine-tuning than adapting attention only. The usual recipe for a vision-language model, step 10 of [Chapter 4](#4-vlm-from-image-and-question-to-answer), is to freeze the vision encoder, put LoRA on the LLM, and train the projector in full.

**Step 10 · What LoRA does not do.** $\theta_{\text{frozen}} \;\longrightarrow\; \theta_{\text{frozen}}$

LoRA matches full fine-tuning on narrow adaptation and, because the base weights are untouched, forgets less; it underperforms when a task needs substantial new knowledge, and a task where LoRA fails badly is a task that needs full fine-tuning. Less forgetting is not none: in the toy below, training the adapters on the new pair alone makes the model answer the old pairs with the new sentence, and mixing the old pairs into the post-training data is what keeps them. QLoRA stores the frozen weights in 4 bits and trains the adapters in bf16; DoRA splits the correction into a magnitude and a direction.

<a id="lora-shapes"></a>

**Every shape in one table.** Toy: $W^{Q}$ of step 2 with $d = 8$, $r = 2$ and $n = 5$ tokens; LLaMA-7B: $d = 4096$ and $r = 16$.

| Step | Input | Output | LLaMA-7B output |
|---|---|---|---|
| 2 · $Q_0 = X W^{Q}$ | $X$: $[5, 8]$, $W^{Q}$: $[8, 8]$ | $Q_0$: $[5, 8]$ | $[n, 4096]$ |
| 3 · $X W^{Q}_{\text{down}}$ | $X$: $[5, 8]$, $W^{Q}_{\text{down}}$: $[8, 2]$ | $[5, 2]$ | $[n, 16]$ |
| 3 · $(X W^{Q}_{\text{down}})\, W^{Q}_{\text{up}}$ | $[5, 2]$, $W^{Q}_{\text{up}}$: $[2, 8]$ | $\Delta Q$: $[5, 8]$ | $[n, 4096]$ |
| 4 · $Q = Q_0 + \Delta Q$ | $[5, 8]$, $[5, 8]$ | $Q$: $[5, 8]$, then $h = 2$ heads of $[5, 4]$ as in step 2 | $[n, 4096]$ |
| 6 · $\partial \mathcal{L} / \partial W^{Q}_{\text{up}}$ | $(X W^{Q}_{\text{down}})^{\top}$: $[2, 5]$, $G$: $[5, 8]$ | $[2, 8]$ | $[16, 4096]$ |
| 6 · $\partial \mathcal{L} / \partial W^{Q}_{\text{down}}$ | $X^{\top}$: $[8, 5]$, $G W^{Q\top}_{\text{up}}$: $[5, 2]$ | $[8, 2]$ | $[4096, 16]$ |
| 7 · trainable parameters per projection | $2 r d$ | $32$ of $64$ | $131{,}072$ of $16{,}777{,}216$ |
| 8 · merged $W^{Q\prime}$ | $W^{Q} + \frac{\alpha}{r} W^{Q}_{\text{down}} W^{Q}_{\text{up}}$ | $[8, 8]$ | $[4096, 4096]$ |

At $d = 8$ the adapter is half the size of the projection; the saving only appears at real widths, where $2rd$ is under one percent of $d^2$.

<a id="116-lora-and-peft"></a>
<a id="coding-lora"></a>
## PyTorch implementation: LoRA on the toy Transformer

[lora_linear.py](./lora_linear.py) wraps any `nn.Linear` in the layer of steps 2–4 and applies it to the projections $W^{Q}, W^{K}, W^{V}, W^{O}$ of every attention module of the model of Chapter 3. [The tests](./test_lora_linear.py) check that the layer starts exactly at the pretrained one, that only $W_{\text{down}}$ and $W_{\text{up}}$ receive gradients and that those gradients match the formulas of step 6, that $\Delta W^{Q}$ has rank at most $r$, that merging is exact, and that post-training the toy model leaves every frozen weight bitwise unchanged.

```bash
.venv-transformer/bin/python lora_linear.py                   # runs the example below
.venv-transformer/bin/python -m unittest -v test_lora_linear  # 8 tests
```

<!-- lora-code: lora-imports -->

**Steps 2 to 4 and 8 — the layer.** `down` is $W_{\text{down}}$ (random) and `up` is $W_{\text{up}}$ (zero); `forward` multiplies by `down` before `up`, and `merge` folds $\Delta W^{Q}$ into the stored weight, which PyTorch keeps transposed as `[d_out, d_in]`.

<!-- lora-code: lora-linear -->

**Step 1 — which weights.** `add_lora` freezes the whole model and replaces every Linear called `w_q`, `w_k`, `w_v` or `w_o` by the wrapped layer; the two helpers merge all adapters and count parameters.

<!-- lora-code: lora-apply -->

**The example.** `main()` trains the toy model in full on two pairs, which stands in for pretraining, adds LoRA with $r = 2$, and post-trains only $W_{\text{down}}$ and $W_{\text{up}}$ on the new pair mixed with the two old ones. It then checks that the frozen weights did not change and that merging does not change the logits.

<!-- lora-code: lora-demo -->

Its output:

```text
pretrained on two pairs (all parameters), loss 0.001
  we watch a short video   -> nous regardons une courte video <eos>
  you watch video          -> tu regardes video <eos>
  we watch video           -> nous regardons une courte video <eos>
LoRA r=2 on 12 matrices: 384 of 2578 parameters trainable (15%); each 8x8 matrix gets W_down [8,2] and W_up [2,8]
  before training Delta W = 0, so the new pair still gives: nous regardons une courte video <eos>
LoRA trained on the new pair plus the two old ones: loss 0.002, frozen weights unchanged: True
  we watch video           -> nous regardons video <eos>
  we watch a short video   -> nous regardons une courte video <eos>
  you watch video          -> tu regardes video <eos>
merged: same logits True; one attention matrix is again a plain (8, 8) weight
```

---
# 6. Generative Models & Learned Compression

> **Chapter guide:** Every generative model written the same way: what one call takes and returns, the loss it is trained with, one worked example. First the autoencoder and the VAE, then the discrete tokenizers from VQ-VAE to FSQ, then GANs and diffusion, then learned compression, where the latent becomes bits, and finally the Cosmos tokenizer step by step. Chapter 3 supplies attention and the Transformer block; [01_video_domain.md](01_video_domain.md) supplies the classical codecs.

$x$ is the input, $z$ the latent, $\hat x$ the reconstruction; $f_\phi$ is the encoder and $g_\theta$ the decoder; $K$ is a codebook size and $d$ a latent width; $D$ is distortion (in 6.2.2 and 6.3.1 only, $D$ is also the GAN discriminator), $R$ bits, and $\lambda$ the price of a bit. $\operatorname{sg}[\cdot]$ is stop-gradient: the value passes forward, no gradient passes back.

## 6.1 Autoencoders & VAEs

<a id="81-autoencoder"></a>

### 6.1.1 Autoencoder

**One call.** $x \in \mathbb{R}^{H \times W \times 3} \;\longrightarrow\; z = f_\phi(x) \in \mathbb{R}^{h \times w \times d} \;\longrightarrow\; \hat x = g_\theta(z) \in \mathbb{R}^{H \times W \times 3}$

$$
\mathcal{L}_{\text{AE}}(\phi, \theta) = \mathbb{E}_x \big\lVert x - g_\theta(f_\phi(x)) \big\rVert_2^2
$$

The only thing that makes this more than the identity is the **bottleneck**: $h w d \ll 3 H W$, so the encoder has to choose what to keep. With a linear encoder and decoder and the squared loss, the optimum is the principal subspace of the data, PCA; the nonlinear version keeps the same idea with a curved subspace. Example: a $256 \times 256$ RGB image is $196{,}608$ numbers; a $16 \times 16 \times 256$ latent is $65{,}536$, three times fewer, but they are still floats, so nothing has been turned into bits yet.

Two things an autoencoder does not give you. There is no distribution on $z$, so a random $z$ decodes to garbage: it is not a generative model. And there is no rate: the latent is as expensive as its float count. 6.4 adds quantization and an entropy model to turn $z$ into bits, and the objective becomes $D + \lambda R$; 6.2 makes $z$ discrete so that it can be transmitted or fed to a transformer.

<a id="82-vae"></a>

### 6.1.2 VAE

**The model.** A prior $p(z) = \mathcal{N}(0, I)$ and a decoder $p_\theta(x \mid z)$, usually $\mathcal{N}(g_\theta(z), \sigma_x^2 I)$ for images. The likelihood of a data point is an integral over every latent that could have produced it, and it is intractable:

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz
$$

**The encoder as an approximate posterior.** $x \;\longrightarrow\; q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x),\ \operatorname{diag}\,\sigma_\phi^2(x)\big)$: the encoder returns two vectors of width $d$, a mean and a log-variance, not a point.

**The ELBO.** For any $q$, the log-likelihood splits exactly into three terms:

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - D_{\text{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{ELBO}(\phi, \theta;\, x)} + \underbrace{D_{\text{KL}}\big(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\big)}_{\ge 0,\ \text{the gap}}
$$

The gap is how far the encoder's posterior is from the true one. Maximizing the ELBO therefore both raises the likelihood and tightens the bound; it cannot exceed $\log p_\theta(x)$. The training loss is the negative ELBO per sample:

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[-\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{regularization}}
$$

With the Gaussian decoder, $-\log p_\theta(x \mid z) = \lVert x - g_\theta(z) \rVert^2 / (2\sigma_x^2) + \text{const}$, so the reconstruction term is an MSE whose weight relative to the KL is set by $\sigma_x^2$. With a Bernoulli decoder it is binary cross-entropy.

**The KL in closed form.** With both distributions diagonal Gaussians:

$$
D_{\text{KL}}\big(\mathcal{N}(\mu, \operatorname{diag}\sigma^2)\,\|\,\mathcal{N}(0, I)\big) = \frac{1}{2} \sum_{j=1}^{d} \big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\big)
$$

| Term | Penalizes | Zero when |
|---|---|---|
| $\mu_j^2$ | a mean away from $0$ | $\mu_j = 0$ |
| $\sigma_j^2 - 1 - \log \sigma_j^2$ | a variance away from $1$; it is $\ge 0$ because $t - 1 - \log t \ge 0$ | $\sigma_j = 1$ |

Derivation: $D_{\text{KL}} = \mathbb{E}_q[\log q(z) - \log p(z)]$, and under $q$ the expectation of $(z_j - \mu_j)^2 / \sigma_j^2$ is $1$ while that of $z_j^2$ is $\mu_j^2 + \sigma_j^2$, which gives $-\tfrac12 \log \sigma_j^2 - \tfrac12 + \tfrac12(\mu_j^2 + \sigma_j^2)$ per dimension. Example with $d = 2$, $\mu = (0.5, 0)$ and $\sigma = (1, 0.5)$: the first dimension contributes $\tfrac12 \cdot 0.25 = 0.125$ nats, the second $\tfrac12 (0.25 - 1 + 1.386) = 0.318$ nats, together $0.443$ nats, or $0.64$ bits.

**The reparameterization trick.** The reconstruction term needs a sample $z \sim q_\phi(z \mid x)$, and a sample has no gradient with respect to $\mu$ and $\sigma$. Write it as

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

Now $\mu$ and $\sigma$ enter through ordinary arithmetic and the randomness sits in $\varepsilon$, which has no parameters. One sample per data point is enough in practice; the KL term needs none, because it has the closed form above.

**The tension, and the two failure modes.** The reconstruction term wants $z$ to carry everything about $x$; the KL wants every $q(z \mid x)$ to equal $\mathcal{N}(0, I)$, which carries nothing. The balance is a latent that is compact but informative, and the KL is what makes the space smooth, so that a $z$ between two encodings decodes to something in between. If the KL wins completely, **posterior collapse**: $q(z \mid x) = p(z)$ for every $x$, the decoder ignores $z$, which a powerful autoregressive decoder can afford. Fixes: KL warm-up, free bits (a floor on the per-dimension KL), $\beta < 1$, or a weaker decoder. If the reconstruction wins completely, the latent has no structure and sampling gives garbage.

**$\beta$-VAE is a rate–distortion objective.** Weighting the KL,

$$
\mathcal{L}_\beta = \mathbb{E}_q\big[-\log p_\theta(x \mid z)\big] + \beta\, D_{\text{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)
$$

is $D + \lambda R$ with the KL as the rate: the KL in bits is the cost of transmitting $z$ to a receiver who knows the prior, and $\beta$ is the exchange rate between a bit and a unit of distortion. $\beta > 1$ gives a more disentangled, blurrier model; $\beta < 1$ a sharper, less structured one. 6.4 makes this literal: a learned codec is a VAE whose prior is the entropy model and whose KL is counted by an arithmetic coder.

<a id="coding-vae-vq"></a>

#### Coding: VAE reparameterization and diagonal-Gaussian KL `[READY]`

- **Prompt:** implement the pathwise sample and the closed-form KL; say why the KL needs no sample and the reconstruction term needs one.
- **Shape / cost:** `mu, logvar: [B, d]`; both functions are `O(B d)`.
- **Pitfalls:** the encoder usually returns `logvar`, so `std = exp(0.5 * logvar)`; sum the KL over dimensions and average over the batch, and state which; reduce the reconstruction term the same way or $\beta$ silently changes; the closed form holds only for the diagonal-Gaussian prior and posterior.

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)                    # sigma, from the log-variance the encoder returns
    return mu + std * torch.randn_like(std)          # z = mu + sigma * eps

def diagonal_gaussian_kl(mu, logvar):
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)   # nats, per sample
```

These two functions are the whole VAE-specific machinery; the rest of a VAE is an encoder, a decoder and a reconstruction loss. No local training script exists; the [learned-compression scaffold](./learned_compression.py#L11) is an autoencoder with noisy scalar quantization, not a VAE.

---

## 6.2 Discrete Tokenizers: VQ-VAE, VQGAN, FSQ and Friends

A continuous latent is a grid of floats. A **tokenizer** turns each position into an integer, an index into a finite set of codes that encoder and decoder share, so the latent can be sent as bits or read by a transformer as a vocabulary. The models in this chapter differ in where the codes come from and in how they are trained; they share one example.

**The example.** A $256 \times 256$ RGB image through an encoder with downsampling factor $f = 16$ gives a $16 \times 16$ grid, $N = 256$ positions. With a codebook of $K = 1024$ codes each position costs $\log_2 1024 = 10$ bits, so the image costs $2{,}560$ bits before any entropy coding, about $0.04$ bits per pixel. The latent width is $d = 256$ unless stated.

<a id="83-vq-vae"></a>

### 6.2.1 VQ-VAE

**One call.** $x \;\longrightarrow\; z_e = f_\phi(x) \in \mathbb{R}^{N \times d} \;\longrightarrow\; k \in \{1, \dots, K\}^{N} \;\longrightarrow\; z_q \in \mathbb{R}^{N \times d} \;\longrightarrow\; \hat x = g_\theta(z_q)$

The codebook is a learned table $E = \{e_1, \dots, e_K\} \subset \mathbb{R}^d$. Each encoder vector is replaced by its nearest code:

$$
k_i = \underset{j}{\arg\min}\ \lVert z_{e,i} - e_j \rVert_2, \qquad z_{q,i} = e_{k_i}
$$

Only the indices $k$ are transmitted or modeled; the decoder looks them up in the same table.

**The loss has three terms**, because the nearest-code step has zero gradient almost everywhere and the table would otherwise never learn:

$$
\mathcal{L}_{\text{VQ-VAE}} = \underbrace{\big\lVert x - g_\theta(z_q) \big\rVert_2^2}_{\text{reconstruction}} + \underbrace{\big\lVert \operatorname{sg}[z_e] - e_k \big\rVert_2^2}_{\text{codebook}} + \beta\, \underbrace{\big\lVert z_e - \operatorname{sg}[e_k] \big\rVert_2^2}_{\text{commitment}}, \qquad \beta \approx 0.25
$$

| Term | Gradient reaches | What it does |
|---|---|---|
| reconstruction, through the straight-through estimator $z_q \leftarrow z_e + \operatorname{sg}[z_q - z_e]$ | decoder, and the encoder as if $z_q$ were $z_e$ | the usual autoencoder signal; the table gets nothing from it |
| codebook | the chosen code $e_k$ only | pulls $e_k$ toward the encoder outputs assigned to it |
| commitment | the encoder only | keeps $z_e$ near the code it chose, so the straight-through copy stays a good approximation |

The straight-through estimator is the identity in the backward pass: the decoder's gradient at $z_q$ is handed to $z_e$ unchanged. It is a biased estimate, and the commitment term is what keeps the bias small by keeping $\lVert z_e - e_k \rVert$ small.

**Why two extra terms, and why each has its own stop-gradient.** Start from what breaks without them. The nearest-code step has zero gradient: moving $z_e$ inside its cell does not change $z_q$, so the encoder receives nothing. The straight-through copy repairs that, but the way it is written, $z_q = z_e + \operatorname{sg}[e_k - z_e]$, routes the decoder's gradient to $z_e$ and gives the table nothing, so the codes would stay at their random initialization forever. The **codebook term** fixes exactly that: its gradient $2(e_k - z_e)$ moves the chosen code toward the encoder outputs assigned to it, a k-means step, and the $\operatorname{sg}$ on $z_e$ keeps it from moving the encoder. The **commitment term** fixes a second problem: the straight-through copy is only a good approximation while $z_q \approx z_e$, because it applies a gradient computed at $e_k$ to a point that may be far from $e_k$. Without it the encoder outputs drift and grow, jump between cells, and the codes chase a moving target. The commitment gradient $2\beta(z_e - e_k)$ pulls $z_e$ back toward the code it chose, and the $\operatorname{sg}$ on $e_k$ keeps it from moving the table.

A single term $\lVert z_e - e_k \rVert^2$ without stop-gradients would push both sides with the same force, and the two sides should not move alike: the table should follow the encoder fully (weight $1$), while the encoder, which is already steered by the reconstruction loss, should only be held lightly (weight $\beta = 0.25$), or it would snap onto the current codes and stop exploring. The two stop-gradients are what make the two weights independent: *the codebook chases the encoder, the encoder commits to the codebook*. In the example above, with $z_e = (0.7, -0.2)$ and $e_1 = (0.9, 0.1)$, the codebook term moves $e_1$ by a gradient of $(0.4, 0.6)$ toward $z_e$, the commitment term moves $z_e$ by $(-0.1, -0.15)$ toward $e_1$, a quarter of the force, and the reconstruction gradient at $e_1$ is copied to $z_e$ unchanged. The EMA variant below makes the division explicit: the table is updated by running averages instead of a loss, and only the commitment term remains.

**EMA codebook.** The codebook term can be replaced by a running k-means: with the assignment counts $n_k$ and sums of the batch,

$$
N_k \leftarrow \gamma N_k + (1 - \gamma)\, n_k, \qquad m_k \leftarrow \gamma\, m_k + (1 - \gamma) \sum_{i:\, k_i = k} z_{e,i}, \qquad e_k \leftarrow \frac{m_k}{N_k}, \qquad \gamma \approx 0.99
$$

Each code moves to the running mean of the vectors assigned to it; only the commitment term remains in the loss. It is more stable than the gradient version and is what most implementations use.

**A two-dimensional example.** $d = 2$, four codes, one encoder vector $z_e = (0.7, -0.2)$:

$$
\begin{aligned}
e_{1:4} &= (0.9, 0.1),\ (-0.5, 0.8),\ (-0.6, -0.7),\ (2, 2) \\
\lVert z_e - e_j \rVert^2 &= 0.13,\ 2.44,\ 1.94,\ 6.53 \;\Rightarrow\; k = 1,\quad z_q = (0.9, 0.1)
\end{aligned}
$$

The codebook term pulls $e_1$ toward $(0.7, -0.2)$; the commitment term pushes $z_e$ toward $(0.9, 0.1)$; $e_4$ receives nothing, and if no vector ever lands nearest to it, it never will.

**Codebook collapse.** That last sentence is the standard failure: a code that is never the nearest gets no gradient, stays where it was initialized, and is dead; large codebooks end up with a few hundred live codes. Measure it with the usage histogram $p_k$ over a dataset and its perplexity $\exp(-\sum_k p_k \log p_k)$, which equals $K$ only when every code is used equally. Fixes: EMA updates, re-initializing dead codes to random encoder outputs, k-means initialization, a lower code dimension with a projection and $\ell_2$-normalized codes (ViT-VQGAN runs $K = 8192$ this way), or replacing the learned table by a fixed grid, which is FSQ (6.2.3). On underwater images AquaScope found $519$ of VQGAN's $1{,}024$ codes never used and merged the table to $64$ codes with usage-weighted k-means, $10 \to 6$ bits per index.

<a id="vqgan"></a>

### 6.2.2 VQGAN

The same quantizer with two more losses on the decoder, so that reconstructions are sharp instead of blurry:

$$
\mathcal{L}_{\text{VQGAN}} = \underbrace{\lVert x - \hat x \rVert_1 + \mathcal{L}_{\text{LPIPS}}(x, \hat x)}_{\mathcal{L}_{\text{rec}}} + \mathcal{L}_{\text{codebook}} + \beta\, \mathcal{L}_{\text{commit}} + \lambda_{\text{adv}}\, \mathcal{L}_G
$$

**The perceptual term** compares features of a frozen VGG rather than pixels, layer by layer, with learned channel weights $w_l$:

$$
\mathcal{L}_{\text{LPIPS}}(x, \hat x) = \sum_l \frac{1}{H_l W_l} \sum_{h, w} \big\lVert w_l \odot \big(\varphi_l(x)_{hw} - \varphi_l(\hat x)_{hw}\big) \big\rVert_2^2
$$

$\varphi_l(x)_{hw} \in \mathbb{R}^{C_l}$ is the layer-$l$ activation at position $(h, w)$, unit-normalized along the channel axis, so with $w_l = 1$ each term is a cosine distance. A squared pixel loss is minimized by the conditional mean of all images consistent with the codes, which is blurry; a feature loss is minimized by images with the right texture statistics.

**The adversarial term** uses a patch discriminator $D$ that classifies $P \times P$ patches, $P \ll H, W$, rather than the whole image, with the hinge loss:

$$
\mathcal{L}_D = \mathbb{E}_x\big[\max(0,\ 1 - D(x))\big] + \mathbb{E}_x\big[\max(0,\ 1 + D(\hat x))\big], \qquad \mathcal{L}_G = -\,\mathbb{E}_x\big[D(\hat x)\big]
$$

$D$ is trained to push real patches above $+1$ and reconstructions below $-1$; the decoder is trained to raise $D(\hat x)$. The adversarial loss is switched on only after a warm-up, and its weight is set adaptively from the gradients at the decoder's last layer $G_L$:

$$
\lambda_{\text{adv}} = \frac{\lVert \nabla_{G_L} \mathcal{L}_{\text{rec}} \rVert}{\lVert \nabla_{G_L} \mathcal{L}_G \rVert + \delta}, \qquad \delta = 10^{-6}
$$

so that the two losses pull on the decoder with comparable force whatever their scales; if the reconstruction gradient has norm $0.8$ and the adversarial one $0.05$, the weight is $16$. The published models use $f = 16$, $K = 1024$ or $16{,}384$, $d = 256$; a second stage trains a transformer on the token grid, $p(k_i \mid k_{<i})$, for generation, which is also exactly the entropy model of 6.2.8. The GAN loss buys sharpness at the cost of pixel fidelity: a reconstruction can contain detail that was never in the input, the perception–distortion trade-off of 6.4.5. Every generative codec in this book, GenStream's and AquaScope's included, is a VQGAN-trained decoder at heart.

<a id="fsq"></a>

### 6.2.3 FSQ: finite scalar quantization

**One call.** $z \in \mathbb{R}^{N \times d} \;\longrightarrow\; u = z W \in \mathbb{R}^{N \times d'} \;\longrightarrow\; \hat z \in \prod_i \mathcal{G}_i,\ k \in \{0, \dots, \prod_i L_i - 1\}$, with $W \in \mathbb{R}^{d \times d'}$ a learned projection, $d'$ small ($3$ to $6$ in the paper's recommended configurations, fewer than $10$ in general) and $\mathcal{G}_i$ the $L_i$ grid values of coordinate $i$

There is no table. Each of the $d'$ coordinates is squashed into a bounded range and rounded to one of $L_i$ levels:

$$
\begin{aligned}
\bar u_i &= \tanh(u_i + \beta_i)\,\eta_i - o_i, & \eta_i &= \frac{(L_i - 1)(1 + \varepsilon)}{2}, & o_i &= \tfrac12 \text{ if } L_i \text{ is even, else } 0, & \beta_i &= \operatorname{artanh}\frac{o_i}{\eta_i} \\[4pt]
\hat z_i &= \frac{\operatorname{round}(\bar u_i)}{\lfloor L_i / 2 \rfloor}, & k &= \sum_{i=1}^{d'} \lfloor L_i / 2 \rfloor\,(\hat z_i + 1)\, b_i, & b_i &= \prod_{j < i} L_j
\end{aligned}
$$

$\eta_i$ makes exactly $L_i$ integers reachable, $o_i$ centres an even number of levels, $\beta_i$ puts $u_i = 0$ on the integer $0$, and $\varepsilon = 10^{-3}$ is a numerical margin (this $(1 + \varepsilon)$, $\operatorname{artanh}$ form is the Cosmos code's; the FSQ paper's reference code writes $(L_i - 1)(1 - \varepsilon)/2$ and $\tan(o_i / \eta_i)$, and both reach every level). The codebook is implicit: the grid of $\prod_i L_i$ points. The FSQ paper's recommended levels for a $2^{16}$ vocabulary are $L = (8, 8, 8, 5, 5, 5)$, $64{,}000$ codes and $15.97$ bits per token, and the $k$ formula packs the six digits in mixed radix. Training uses the straight-through estimator for the rounding, $\operatorname{round}_{\text{STE}}(\bar u) = \bar u + \operatorname{sg}[\operatorname{round}(\bar u) - \bar u]$, and nothing else: no codebook loss, no commitment loss, because there is no table to pull and the $\tanh$ already keeps $u$ inside the grid. Every code is reachable by moving $d'$ scalars, so there is no collapse: on $2^{14}$ codes FSQ uses almost all of them where VQ uses fewer than half. Example: $u = (0, 0.3, -1.2, 2.5, -0.4, 0.9)$ gives $\hat z = (0, 0.25, -0.75, 1, -0.5, 0.5)$, digits $(4, 5, 1, 4, 1, 3)$ and $k = 43{,}116$. The price is a uniform, axis-aligned grid in few dimensions; the encoder and decoder adapt to it, and the paper reports quality on par with VQ. FSQ is the quantizer of the Cosmos discrete tokenizer, whose step 7 in 6.5 traces it on real shapes, with a full side-by-side against VQ in the note under that step.

<a id="lfq"></a>

### 6.2.4 LFQ: lookup-free quantization

**One call.** $z \in \mathbb{R}^{N \times d} \;\longrightarrow\; q = \operatorname{sign}(z) \in \{-1, +1\}^{N \times d},\quad k_i = \sum_{j=1}^{d} \mathbf{1}[z_{ij} > 0]\, 2^{j-1}$

Each coordinate becomes one bit, so the vocabulary is $K = 2^d$ with no table at all: MAGVIT-v2 uses $d = 18$, $262{,}144$ codes, far beyond what a learned table could keep alive. Example: $z = (0.7, -0.2, 1.3) \to q = (+1, -1, +1)$, bits $(1, 0, 1)$, $k = 5$. The loss adds two terms to the reconstruction (and the adversarial terms of a VQGAN-style decoder):

$$
\mathcal{L}_{\text{LFQ}} = \mathcal{L}_{\text{rec}} + \beta \lVert z - \operatorname{sg}[q] \rVert_2^2 + \gamma \Big( \mathbb{E}_x\big[H(p(k \mid x))\big] - H\big(\mathbb{E}_x[p(k \mid x)]\big) \Big)
$$

$p(k \mid x)$ is a softmax over $-\lVert z - c_k \rVert^2$ for the codes $c_k$, which factorizes per dimension because the codes are the corners of a hypercube. The first entropy term asks each input to be confident about its code; the second, subtracted, asks the codes to be used uniformly across the data. Together they are the entropy penalty that keeps a huge vocabulary from collapsing onto a few corners. Gradients again pass by the straight-through estimator.

<a id="rvq"></a>

### 6.2.5 RVQ: residual vector quantization

**One call.** $z \;\longrightarrow\; (k^{(1)}, \dots, k^{(S)})$, $S$ indices per position

$$
r_0 = z, \qquad q_s = \underset{e \in E_s}{\arg\min}\ \lVert r_{s-1} - e \rVert_2, \qquad r_s = r_{s-1} - q_s, \qquad \hat z = \sum_{s=1}^{S} q_s
$$

Stage $s$ quantizes what the earlier stages left over, with its own codebook $E_s$, so $S$ stages of $K$ codes reach a precision that one codebook would need $K^S$ codes for, at $S \log_2 K$ bits per position. Toy with scalars and codebooks $\{0, 2, 4\}$, $\{-1, 0, 1\}$, $\{-0.5, 0, 0.5\}$: $z = 3.7 \to q = (4, 0, -0.5)$, $\hat z = 3.5$, residual $0.2$. The stages are ordered by importance, so dropping the last ones gives a coarser reconstruction without re-encoding: SoundStream streams audio this way with $4$ to $24$ stages of $K = 1024$, $3$ to $18$ kbps at $75$ frames per second, and EnCodec with $2$ to $32$; the same idea gives a token stream that can shed refinement under bandwidth pressure. Training is VQ-VAE training per stage, with the straight-through sum and commitment terms on each residual.

<a id="titok"></a>

### 6.2.6 TiTok: a one-dimensional token sequence

Every tokenizer above keeps a grid: one index per spatial position, $N = (H/f)(W/f)$. TiTok drops the grid. A ViT encoder reads the patch embeddings together with $N$ learned **latent tokens** appended to the sequence, and only the outputs at the latent positions are kept and quantized, with $K = 4096$ in the released models; the decoder reads the quantized latent tokens plus one mask token per patch and produces the image, in the paper through a frozen VQGAN decoder during a two-stage training. $N$ is a free choice, $32$ to $256$ per image, far fewer than any grid, and the tokens spend capacity where the image needs it rather than one per patch. AquaScope's setting is $N = 64$ tokens of $12$ bits, $768$ bits for a $256 \times 256$ image, which is what fits an underwater acoustic link; with a grid tokenizer the same image was $256$ indices of $6$ bits after clustering, $1{,}536$ bits. The cost is that a wrong token no longer maps to one patch: the error spreads over the image, which is why AquaScope fine-tunes the decoder under injected token errors (the $2\%$ BER figure of 6.4.5).

<a id="quantizers-side-by-side"></a>

### 6.2.7 The quantizers side by side

| Quantizer | Codes | Trained by | Extra losses | Collapse | Bits per position | Used by |
|---|---|---|---|---|---|---|
| VQ-VAE | learned table of $K$ vectors | codebook term, or EMA | codebook, commitment | yes, dead codes | $\log_2 K$ | VQ-VAE, VQ-VAE-2 |
| VQGAN | the same table | as above, plus LPIPS and a patch discriminator | perceptual, adversarial | yes | $\log_2 K$ | VQGAN, MaskGIT, most generative codecs |
| FSQ | fixed grid of $\prod_i L_i$ points | nothing to train | none | no | $\log_2 \prod_i L_i$ | Cosmos, MaskGIT-FSQ |
| LFQ | corners of a $d$-cube, $2^d$ | nothing to train | commitment, entropy penalty | no, by the penalty | $d$ | MAGVIT-v2 |
| RVQ | $S$ tables of $K$ | per stage, usually EMA | per stage | per stage | $S \log_2 K$ | SoundStream, EnCodec |
| TiTok | a table, $K = 4096$ | VQGAN-style | perceptual, adversarial | yes | $\log_2 K$, but only $N = 32$ to $256$ positions | AquaScope |

<a id="bits-per-token"></a>

### 6.2.8 Bits per token: the fixed baseline and a learned prior

Transmitting the indices as they are costs $N \log_2 K$ bits, $2{,}560$ for the example, and that figure is blind to the content. A **prior** over the token sequence turns it into a real rate: with a causal transformer giving $p(k_i \mid k_{<i})$ and an arithmetic coder,

$$
R = \sum_{i=1}^{N} -\log_2 p_\theta(k_i \mid k_{<i}), \qquad \mathbb{E}[R] = N \big( H(k) + D_{\text{KL}}(p_{\text{data}} \,\|\, p_\theta) \big), \qquad H(k) \le \log_2 K
$$

The entropy $H(k)$ of the true token distribution is at most $\log_2 K$ per token; the prior pays that plus a KL term for how wrong it is, so a prior beats the fixed length only when it is better than uniform on the data it is run on, and under domain shift it can cost more than $N \log_2 K$, one reason the fixed-length stream is the safe fallback. The saving comes from the structure of real images: neighbouring tokens are correlated, a sky token predicts more sky, so the conditional distribution is far from uniform. For $K = 1024$ this typically takes $10$ bits per token down to $4$ to $6$. Encoder and decoder run the same prior on the same context, so decoding is serial, one token after another, which is the latency cost of 6.4; and because the arithmetic coder needs bit-identical probabilities on both ends, the prior has to run deterministically. Without a prior, the fixed length is also what makes a token stream loss-tolerant: any index can be read on its own, which is the property GenStream and AquaScope rely on (6.4.4, 6.4.7).

<a id="coding-vq-ste"></a>

#### Coding: VQ with a straight-through estimator `[READY]`

**Likely prompt:** *"Implement vector quantization without killing the encoder gradient. How would you turn the indices into fewer than $\log_2 K$ bits per token?"*

- **Shapes:** flattened encoder tokens `z_e: [B, N, D]`; codebook `[K, D]`; indices `[B, N]`; quantized vectors `[B, N, D]`.
- **Cost:** the distance matrix is `O(B N K D)` time and `O(B N K)` memory; a prior for the rate is a causal transformer over `N` tokens, `O(N^2 D)` per image and serial at decode time.
- **Pitfalls:** detach the right side of each loss; monitor usage and perplexity for collapse; `N log2 K` is only the fixed-length baseline, the real rate comes from the prior; the EMA variant drops the codebook loss and updates the table outside the optimizer. The original [VQ-VAE](https://arxiv.org/abs/1711.00937) and [VQGAN](https://arxiv.org/abs/2012.09841) papers define the terms.

```python
def vector_quantize(z_e, codebook, beta=0.25):
    # z_e: [B, N, D], codebook: [K, D]
    distance = (z_e.square().sum(-1, keepdim=True)
                + codebook.square().sum(-1)[None, None]
                - 2 * z_e @ codebook.t())                   # ||z_e - e_k||^2 for every pair, [B, N, K]
    indices = distance.argmin(dim=-1)                     # k_i:  [B, N]
    z_q = F.embedding(indices, codebook)                  # e_k:  [B, N, D]
    z_st = z_e + (z_q - z_e).detach()                     # straight-through: forward z_q, backward z_e
    codebook_loss = (z_q - z_e.detach()).square().mean()  # moves e_k toward z_e
    commitment = (z_e - z_q.detach()).square().mean()     # keeps z_e near e_k
    return z_st, indices, codebook_loss + beta * commitment
```

`[READY]` for the quantizer itself; the causal token prior with a shared integer CDF is `[TODO]`, and **`[READY]`** [`arithmetic_coding.py`](arithmetic_coding.py) supplies the static arithmetic-coder round trip it would plug into.

---
## 6.3 GANs & Diffusion

Two ways to train a decoder to produce images that look real rather than images that minimize a pixel error. The GAN is what makes the VQGAN decoders of 6.2 sharp; diffusion is the generative family behind today's image and video models, and the Cosmos world models of 6.5 run it in a tokenizer's latent.

<a id="84-gan"></a>

### 6.3.1 GAN

**One call.** $z \sim p(z) \;\longrightarrow\; G(z) \in \mathbb{R}^{H \times W \times 3}$, judged by $D: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}$

$$
\min_G \max_D\ \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]
$$

For a fixed $G$ the best discriminator is $D^*(x) = p_{\text{data}}(x) / (p_{\text{data}}(x) + p_G(x))$, and plugging it back in makes the generator's objective $2\,\text{JS}(p_{\text{data}} \,\|\, p_G) - \log 4$: a GAN minimizes the Jensen–Shannon divergence between the data and what it generates, without ever writing down a density. Two things go wrong in practice:

- **Vanishing gradient.** When $D$ wins, $\log(1 - D(G(z)))$ saturates and $G$ stops learning. The fix is the non-saturating loss $-\log D(G(z))$, or a loss that is linear in $D$: the hinge loss, used by VQGAN, and the Wasserstein loss with a gradient penalty,

$$
\begin{aligned}
\mathcal{L}_D^{\text{hinge}} &= \mathbb{E}\big[\max(0, 1 - D(x))\big] + \mathbb{E}\big[\max(0, 1 + D(G(z)))\big] \\[4pt]
\mathcal{L}_D^{\text{WGAN-GP}} &= \mathbb{E}\big[D(G(z))\big] - \mathbb{E}\big[D(x)\big] + \gamma\, \mathbb{E}_{\tilde x}\big[(\lVert \nabla_{\tilde x} D(\tilde x) \rVert_2 - 1)^2\big]
\end{aligned}
$$

with $\tilde x$ on random segments between real and generated samples, and in both cases $\mathcal{L}_G = -\mathbb{E}[D(G(z))]$. Spectral normalization of $D$ and the R1 penalty $\tfrac{\gamma}{2}\mathbb{E}_x \lVert \nabla_x D(x) \rVert^2$ on real samples serve the same purpose: keep $D$ smooth so that its gradient is informative.
- **Mode collapse.** $G$ finds a few outputs that fool $D$ and stops covering the data; nothing in the objective rewards diversity directly.

**For codecs and restoration** the GAN is an auxiliary loss, not the model: a decoder trained with reconstruction plus $\lambda_{\text{adv}} \mathcal{L}_G$ produces texture instead of blur, with the adaptive weight and patch discriminator of 6.2.2. A **patch discriminator** scores every $P \times P$ patch and averages, so it judges texture statistics rather than global layout, has few parameters and works at any resolution; it is the standard choice in pix2pix, VQGAN and neural codecs. What it buys is realism; what it costs is fidelity, which is the trade-off of 6.4.5.

<a id="85-diffusion"></a>

### 6.3.2 Diffusion

**Forward process.** $x_0 \;\longrightarrow\; x_1 \;\longrightarrow\; \dots \;\longrightarrow\; x_T \approx \mathcal{N}(0, I)$, Gaussian noise added in $T$ small steps with a variance schedule $\beta_t$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\big), \qquad \alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s
$$

Because Gaussians compose, any $x_t$ can be sampled from $x_0$ in one line, which is what makes training cheap:

$$
q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\big), \qquad x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

With DDPM's linear schedule, $\beta_t$ from $10^{-4}$ to $0.02$ over $T = 1000$ steps:

| $t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$, the signal | $\sqrt{1 - \bar\alpha_t}$, the noise | signal-to-noise ratio |
|---|---|---|---|---|
| 100 | 0.897 | 0.947 | 0.321 | 8.7 |
| 250 | 0.524 | 0.724 | 0.690 | 1.1 |
| 500 | 0.079 | 0.280 | 0.960 | 0.085 |
| 1000 | 0.00004 | 0.006 | 1.000 | 0.00004 |

By $t = 500$ the image is mostly noise; by $t = T$ nothing of $x_0$ is left, which is what lets sampling start from pure noise.

**Reverse process.** $x_T \sim \mathcal{N}(0, I) \;\longrightarrow\; \dots \;\longrightarrow\; x_0$, one learned Gaussian step at a time, $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$. The step is learnable because, given $x_0$, the true reverse step is also Gaussian with a known mean:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\big), \qquad \tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t, \qquad \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t
$$

**The loss.** Substituting $x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\varepsilon) / \sqrt{\bar\alpha_t}$ into $\tilde\mu_t$ shows that predicting the mean is the same as predicting the noise, so the network is a noise predictor $\varepsilon_\theta(x_t, t)$:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \varepsilon_\theta(x_t, t) \Big), \qquad \mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \varepsilon} \Big\lVert \varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon,\ t\big) \Big\rVert_2^2
$$

This is the variational bound of the VAE in 6.1.2, written for a chain of latents $x_1, \dots, x_T$, with the per-step KL terms simplified to this unweighted regression; the weighting is what the original ELBO would add. Every training step samples one $t$, one $\varepsilon$, computes $x_t$ in one line and regresses $\varepsilon$: a supervised problem with a fixed target, which is why diffusion training is stable where GAN training is a game.

**Sampling.** DDPM runs the chain down from $T$:

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t\, \xi, \qquad \xi \sim \mathcal{N}(0, I), \qquad \sigma_t^2 \in \{\beta_t,\ \tilde\beta_t\}
$$

a thousand network evaluations. DDIM uses the same network as a deterministic map by first estimating $x_0$ and then re-noising it to the next level:

$$
\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \varepsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}, \qquad x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{t-1}}\, \varepsilon_\theta(x_t, t)
$$

which can skip levels, $20$ to $50$ steps, and is deterministic given $x_T$. Distillation (progressive, consistency models, LCM) trains a student to take the chain in $1$ to $4$ steps. The number of steps is the practical cost of diffusion.

**Classifier-free guidance.** Train one network with and without the condition $c$ (drop it about $10\%$ of the time) and extrapolate at sampling time:

$$
\hat\varepsilon = \varepsilon_\theta(x_t, t, \varnothing) + w \big( \varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \varnothing) \big)
$$

$w = 1$ is the conditional model; $w > 1$ sharpens adherence to $c$ at the cost of diversity, and the FID of $2.27$ for DiT-XL/2 quoted in [§3](#vision-transformers) is obtained at $w = 1.5$.

**Latent diffusion.** Run the whole chain on the latent of an autoencoder, the Stable Diffusion VAE with $f = 8$ and $4$ channels, so a $256 \times 256$ image is a $32 \times 32 \times 4$ tensor: $48\times$ fewer numbers, and the only reason diffusion is affordable. The denoiser was a U-Net; the DiT in [§3](#vision-transformers) replaces it with a transformer over $2 \times 2$ latent patches, conditioned by adaLN-Zero.

<a id="flow-matching"></a>

### 6.3.3 Flow matching

Most models since SD3 and Flux use a simpler parameterization of the same idea. Draw a data point $x_0$ and a noise sample $x_1 \sim \mathcal{N}(0, I)$, connect them by a straight line and train a network to predict the line's velocity:

$$
x_t = (1 - t)\, x_0 + t\, x_1, \qquad t \sim \mathcal{U}(0, 1), \qquad \mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\, x_0,\, x_1} \big\lVert v_\theta(x_t, t) - (x_1 - x_0) \big\rVert_2^2
$$

Sampling integrates the ODE $dx/dt = v_\theta(x, t)$ from $t = 1$ (noise) to $t = 0$ (data), with Euler steps $x_{t - \Delta} = x_t - \Delta\, v_\theta(x_t, t)$. The straight-line path replaces the variance schedule, there is no $\bar\alpha_t$ table, and the velocity target is the same for every point on the line, so the learned field is close to straight and few steps suffice. DDPM is the same family with a curved path and the noise $\varepsilon$ as the target instead of the velocity; the two are related by a change of variables, not by a different principle.

<a id="video-diffusion"></a>

### 6.3.4 Video diffusion and world models

A video model adds the time axis to the denoiser: 3D attention over all space-time tokens, or the factorized spatial-then-temporal attention of TimeSformer in [§3](#vision-transformers), and the same chain runs on a clip. Two costs decide the design. Memory, because a $16$-frame clip has $16\times$ the tokens of an image, which is why video models run in a video tokenizer's latent, $8\times$ down in time and $8$ to $16\times$ in space (6.5). And temporal consistency, which the per-frame losses of this chapter do not enforce; the fix is attention across frames inside the denoiser and losses on flow-warped neighbours in the tokenizer. The Cosmos world models are this design at scale: a diffusion model on the continuous $8 \times 8 \times 8$ latent and an autoregressive model on the discrete $8 \times 16 \times 16$ tokens, both of 6.5.

<a id="coding-diffusion"></a>

#### Coding: diffusion forward noising `[TOY]`

- **Prompt:** sample $x_t$ directly from $x_0$, then form the noise-prediction target; explain why no loop over earlier timesteps is required.
- **Shape / cost:** `x0`, `noise` and the output share `[B, C, H, W]`; `alpha_bar_t` is gathered per sample as `[B, 1, 1, 1]`; elementwise, `O(BCHW)`.
- **Pitfalls:** use the cumulative $\bar\alpha_t$, not the per-step $\alpha_t$; broadcast one schedule value per sample; the loss compares the model's prediction to the *same* `noise` that produced `x_t`; do not detach the prediction.

```python
def q_sample(x0, alpha_bar_t, noise):
    return alpha_bar_t.sqrt() * x0 + (1.0 - alpha_bar_t).sqrt() * noise   # x_t in one line

def noise_prediction_loss(model, x0, t, alpha_bar):
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, alpha_bar[t].view(-1, 1, 1, 1), noise)
    return (model(x_t, t) - noise).square().mean()                          # L_simple
```

The two functions are the training step; the schedule, the sampler (DDPM or DDIM), timestep sampling and classifier-free guidance are `[TODO]`.

---
## 6.4 Learned & Generative Compression

<a id="1-learned--generative-compression"></a>

The models of 6.1 to 6.3 become a codec once the latent is turned into bits. This section is that step: the objective, the quantizer, the entropy model that decides the rate, the two shapes a learned codec can take, and what a decoder that hallucinates is allowed to do.

<a id="11-the-core-idea"></a>

### 6.4.1 The core idea

Replace the hand-crafted transform, quantizer and entropy model with networks trained **end-to-end on the rate–distortion objective**.

**One call** (this section keeps Ballé's letters: $y$ is the latent and $z$ the hyper-latent of 6.4.3). $x \;\longrightarrow\; y = g_a(x) \;\longrightarrow\; \hat y = Q(y) \;\longrightarrow\; \text{bits} \;\longrightarrow\; \hat x = g_s(\hat y)$

$$
\begin{aligned}
y &= g_a(x) && \text{analysis transform, the encoder} \\[2pt]
\hat y &= Q(y) = \operatorname{round}(y) && \text{scalar quantization, one integer per latent element} \\[2pt]
\text{bits} &\leftarrow \operatorname{AC}\big(\hat y,\ p_\psi(\hat y)\big) && \text{arithmetic coding under a learned prior} \\[2pt]
\hat x &= g_s(\hat y) && \text{synthesis transform, the decoder}
\end{aligned}
$$

**Training objective**, in one common convention:

$$
\mathcal{L} = \underbrace{D(x, \hat x)}_{\text{distortion}} + \lambda\, \underbrace{\mathbb{E}\Big[\sum_i -\log_2 p_\psi(\hat y_i)\Big]}_{R,\ \text{expected bits}}
$$

Some papers instead write `L = R + λD`; then larger `λ` means *more* quality / rate, whereas in
`D + λR` it means *more* rate pressure. State the convention before interpreting a sweep. For bpp,
divide the summed code length (including side latents) by the number of input pixels.

**Why `−log p(ŷ)` *is* the bitrate**: Shannon's source coding theorem says the optimal code length for a symbol is `−log₂ p`. An arithmetic coder achieves that up to a negligible constant. So minimizing the negative log-likelihood of the latent under the model *is* minimizing the compressed size. Say this crisply — it's the one-line justification for the whole field.

**Note the structural identity with the VAE** ([6.1.2](#612-vae)): $D + \lambda R$ is the negative ELBO with a Lagrangian weight. Learned compression is a VAE whose prior is used as an entropy model and whose KL is a real bit count.

#### Coding: learned codec forward + RD loss

**Likely prompt:** *"Write the train/eval forward pass of a learned codec. Where does the gradient
cross quantization, and what exactly does `λ` control?"*

- **Local code:** **`[TOY]`** [`learned_compression.py`](learned_compression.py) —
  `LearnedCompressionModel.forward`, `rd_loss`, and `train_step` show the complete control flow.
  **`[READY]`** [`ml_coding_torch.py`](ml_coding_torch.py) — `rate_distortion_loss` is the clean
  algebra-only drill.
- **Shapes:** `x:[B,3,H,W] → y:[B,M,H/8,W/8] → x_hat:[B,3,H,W]` for the current three stride-2
  layers. A real model must define padding/cropping for dimensions not divisible by 8.
- **Cost:** convolutional work and activation memory are linear in `B·H·W`; the decoder usually owns
  the largest activation maps.
- **Pitfalls:** training uses uniform noise while evaluation rounds; `D+λR` and `R+λD` reverse the
  meaning of increasing `λ`; the returned `likelihood` is not automatically a real bpp.

```python
def forward(self, x):
    y = self.g_a(x)
    if self.training:
        y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
    else:
        y_hat = torch.round(y)
    x_hat = self.g_s(y_hat)
    likelihood = self.entropy_model(y_hat)
    return x_hat, y_hat, likelihood

def rd_objective(x, x_hat, rate_bpp, lam):
    distortion = (x - x_hat).square().mean()
    return distortion + lam * rate_bpp
```

<a id="12-the-quantization-problem"></a>

### 6.4.2 The quantization problem

`round()` has zero gradient almost everywhere, so you can't backprop through it. Three standard surrogates:

| Method | How | Notes |
|--------|-----|-------|
| **Additive uniform noise** (Ballé) | `ỹ = y + u`, `u ~ U(−½, ½)` during training | Makes the density a continuous relaxation; the resulting model is a proper VAE with a uniform posterior. Train/test mismatch is the drawback. |
| **Straight-through estimator** | forward: round; backward: identity | Exact forward pass, biased gradient |
| **Mixed / "universal" quantization** | noise for the rate term, STE for the distortion term | What most modern codecs actually do; the rate term is still fitted to the noisy latent, so the estimated bits keep a train/test mismatch against the rounded symbols that are coded |

**The train/test mismatch** is a real, askable weakness: noise-based training optimizes for a *noisy* latent, but inference uses a *rounded* one. Later work (soft-to-hard annealing, quantization-aware fine-tuning of the last epochs) closes the gap.

<a id="13-entropy-models--where-the-gains-actually-come-from"></a>

### 6.4.3 Entropy models — where the gains actually come from

The transform matters less than most people assume; **the entropy model is where learned codecs win.**

**Factorized prior** (baseline): each latent channel gets its own learned 1-D CDF, independent across space. Cheap, weak.

**Hyperprior** (Ballé 2018), the key idea. A second, smaller latent $z = h_a(y)$ is quantized and sent as side information, a few percent of the bits, and the decoder expands it into a scale for every element of $\hat y$; the mean-and-scale form written here is Minnen 2018's extension:

$$
\begin{aligned}
\mathcal{L} &= D(x, \hat x) + \lambda \Big( \mathbb{E}\big[-\log_2 p(\hat y \mid \hat z)\big] + \mathbb{E}\big[-\log_2 p(\hat z)\big] \Big) \\[4pt]
(\mu, \sigma) &= h_s(\hat z), \qquad p(\hat y_i \mid \hat z) = \int_{\hat y_i - 1/2}^{\hat y_i + 1/2} \mathcal{N}(u;\ \mu_i, \sigma_i^2)\, du = \Phi\Big(\tfrac{\hat y_i + 1/2 - \mu_i}{\sigma_i}\Big) - \Phi\Big(\tfrac{\hat y_i - 1/2 - \mu_i}{\sigma_i}\Big)
\end{aligned}
$$

The probability of an integer symbol is the Gaussian mass of its unit bin, not the density at its centre; $\hat z$ itself is coded under a learned factorized prior.
**Why it works**: the latent is *not* stationary — a latent element over a textured region has high variance, one over the sky has low variance. A single global distribution must be wide enough to cover both, so it wastes bits everywhere. The hyperprior transmits a tiny description of the *local* scale so each element gets a tight, well-matched distribution. Spending 3% of your bits to halve the entropy of the other 97% is a huge win. This "spend side information to sharpen the prior" argument is the single most important concept in the area — and it's structurally identical to HEVC signalling per-CTU quantization/SAO parameters.

**Autoregressive context model** (Minnen 2018): additionally condition on already-decoded neighbors, `p(ŷ_i | ŷ_{<i}, ẑ)`. Another ~10% rate. **But it serializes decoding** — each element waits for its neighbors, so decode time explodes. This is the classic rate/latency tradeoff of the field.

**Checkerboard / channel-wise context** (checkerboard: He 2021; channel groups: Minnen & Singh 2020 and ELIC, He 2022): split the latent into two interleaved halves or into channel groups; decode group 1 in parallel, use it as context for group 2. Recovers most of the autoregressive gain with 2–10 passes instead of N. **This is exactly the kind of "make it parallel" answer interviewers want.**

#### Coding: quantized likelihood, estimated bpp, and real bits

**Likely prompt:** *"Your network returned a Gaussian likelihood tensor. Turn it into bpp, then tell
me why that number can still differ from the file size."*

- **Local code:** **`[TOY]`** [`learned_compression.py`](learned_compression.py) currently evaluates a
  Gaussian **PDF** and averages `−log₂` over latent elements. That is a useful warning example, not a
  correct actual-bpp implementation. **`[READY]`** [`arithmetic_coding.py`](arithmetic_coding.py)
  round-trips symbols under a static frequency model; learned integer-CDF integration is **`[TODO]`**.
- **Shapes:** `y_hat, mu, scale:[B,M,H',W']`; optional hyper-latent `z_hat:[B,M_z,H_z,W_z]`;
  `bits_y` and `bits_z` are scalars; normalize by `B·H·W`, not by the number of latent elements.
- **Cost:** likelihood evaluation is `O(B·M·H'·W')`; autoregressive arithmetic coding is linear in
  symbol count but serial, while checkerboard/channel groups trade a few sequential passes for parallelism.
- **Pitfalls:** a density value is not the probability of an integer symbol; count hyperlatents,
  headers/CDF tables, padding, and coder termination; encoder and decoder need the same quantized CDF
  bit-for-bit. See the [scale-hyperprior paper](https://arxiv.org/abs/1802.01436).

```python
def gaussian_bin_prob(y_hat, mu, scale, eps=1e-9):
    scale = scale.clamp_min(1e-6)
    normal = torch.distributions.Normal(mu, scale)
    # Probability mass assigned to the integer-centered quantization bin.
    prob = normal.cdf(y_hat + 0.5) - normal.cdf(y_hat - 0.5)
    return prob.clamp_min(eps)

def estimated_bpp(prob_y, input_shape, prob_z=None):
    batch, _, height, width = input_shape
    bits = -torch.log2(prob_y).sum()
    if prob_z is not None:
        bits = bits - torch.log2(prob_z).sum()
    return bits / (batch * height * width)
```

<a id="14-two-paradigms"></a>

### 6.4.4 Two paradigms

#### Paradigm 1 — Hybrid neural codec (H.264-shaped)

Keep predict → residual → entropy-code, replace each module with a network. DVC, FVC, DCVC / DCVC-DC / DCVC-FM.

```
Input frame
   ↓ optical flow network                 (motion estimation)
   ↓ compress the flow field itself       (an autoencoder + entropy model)
   ↓ warp reference → predicted frame     (motion compensation)
   ↓ residual = input − predicted   ... OR ...
   ↓ [DCVC] condition the encoder on the predicted *feature*, don't subtract
   ↓ autoencoder → latent y → quantize → arithmetic code with hyperprior
```

**The DCVC insight worth knowing**: "Deep **Contextual** Video Compression" replaces *subtraction*
with *conditioning*. Residual coding hard-codes the reference's role as `r = x − pred`; conditional
coding exposes the context and lets the network learn how to use it. The precise information-theory
statement is `H(X|C) ≤ H(X)`. If `pred` is determined by `C`, then `X ↔ R=X−pred(C)` is bijective for
fixed `C`, so `H(X|C) = H(R|C)`; the practical advantage comes from a richer conditional model, not
from an unconditional "conditioning is always strictly better" inequality. DCVC-FM and successors
report gains over VVC (VTM) on standard test sets in BD-rate terms.

- Latent is **continuous**, scalar-quantized, arithmetic-coded.
- Optimized for `D + λ·R` in this book's convention (the DVC and DCVC papers write `λ·D + R`, so their λ runs the other way), with D = MSE or MS-SSIM.
- Reconstruction is **deterministic** and high-fidelity.

#### Paradigm 2 — VQ / generative codec

```
Input image/video
   ↓ CNN/ViT encoder → continuous latent z_e
   ↓ vector quantization → nearest codebook entry
   ↓ token grid (e.g. 16×16 indices from a codebook of K=1024)
   ↓ entropy-code the token sequence
   ↓ [transmitted]
   ↓ decoder (often GAN- or diffusion-trained) → reconstruction
```

**Naive cost**: `16×16 × log₂(1024) = 2560 bits`, fixed and statistics-blind.

**Why put a prior on the tokens**: token indices are far from uniform, and neighbors are highly correlated (a sky token predicts more sky). Modeling `p(k_i | k_<i)` with an autoregressive transformer and arithmetic-coding against it typically takes K=1024 tokens from 10 bits down to **4–6 bits/token**.

```
Encode token i:  feed k_<i to the transformer → p(·|k_<i) over K codes
                 arithmetic-encode the true k_i under that distribution
Decode token i:  same transformer, same context → same distribution
                 arithmetic-decode k_i, append, continue
```
Both sides share the pre-trained transformer weights out-of-band, like a dictionary. Only the arithmetic-coded output goes on the wire.

**VQGAN** ([6.2.2](#622-vqgan)) adds LPIPS and patch-discriminator losses to VQ-VAE, making reconstructions sharp instead of blurry; it is the basis for most generative codecs. The token prior that takes $K = 1024$ from $10$ bits to $4$ to $6$ per token is worked through in [6.2.8](#628-bits-per-token-the-fixed-baseline-and-a-learned-prior).

**Core differences:**

| | Hybrid (H.264-shaped) | VQ / generative |
|---|---|---|
| Latent | continuous → scalar quantized | discrete codebook indices |
| Pipeline | predict + residual/context + entropy code | tokenize → entropy code indices |
| Entropy model | hyperprior / spatial context on `p(ŷ)` | autoregressive transformer on `p(k_i\|k_<i)` |
| Objective | rate-distortion (MSE/MS-SSIM) | perceptual + adversarial |
| Reconstruction | deterministic, faithful | can hallucinate plausible detail |
| Sweet spot | mid-to-high bitrate, objective metrics | **very low bitrate**, visual quality |
| Error behavior | graceful-ish | a corrupted token can change semantics |

#### Coding: closed-loop references, random access, and error propagation

**Likely prompt:** *"Extend a single-image learned codec to video. Which reference does the encoder
use, when do you reset state, and what happens after one corrupted P-frame?"*

- **Local code:** **`[TOY]`** [`video_encoding_pipeline.py`](video_encoding_pipeline.py) —
  `encode_video` inserts periodic I-frames and feeds the **reconstructed** prior frame back as the
  reference. It is encoder-only and classical, but it demonstrates the invariant a learned codec must keep.
- **Shapes:** frame `x_t:[B,3,H,W]`; reconstructed reference or feature state
  `s_{t-1}`; an intra frame uses `None`/reset state. Multiple reference features multiply memory by
  DPB depth.
- **Cost:** coding remains `O(T·H·W)` at fixed architecture, but latency and memory depend on reference
  count and whether future frames are allowed.
- **Pitfalls:** conditioning on the original previous frame during training causes train/decode drift;
  long GOPs improve rate but amplify loss propagation and seek latency; scene cuts need intra refresh;
  packet loss, model-version metadata, motion bits, and hyperlatents all belong in the rate/system story.
  Compare the [DVC](https://arxiv.org/abs/1812.00101) and
  [DCVC](https://arxiv.org/abs/2109.15047) predictive structures.

```python
def encode_sequence(frames, gop, encode_one, is_scene_cut):
    state, packets = None, []
    for t, frame in enumerate(frames):
        intra = (t % gop == 0) or is_scene_cut(frame)
        if intra:
            state = None
        packet, reconstruction, next_state = encode_one(
            frame, reference_state=state, intra=intra
        )
        packets.append(packet)
        # Never feed the unavailable original frame into the next prediction loop.
        state = next_state if next_state is not None else reconstruction
    return packets
```

<a id="15-the-rate-distortion-perception-tradeoff"></a>

### 6.4.5 The rate-distortion-perception tradeoff

**The theorem** (Blau & Michaeli): at a fixed rate, you cannot simultaneously minimize distortion and maximize perceptual quality. Forcing the output distribution to match the source distribution (perfect "realism") costs you at most a 2× increase in MSE — but it *does* cost you.

Practical reading: below ~0.1 bpp, hybrid codecs go blurry and blocky, while generative codecs still produce convincing images — that are **not the same image**. Which you want depends entirely on the application:

| Application | Want |
|-------------|------|
| Medical, forensic, scientific, machine vision input | fidelity — hallucination is a correctness bug |
| Consumer streaming, video calls, photo storage | perception — nobody diffs pixels |
| Your AquaScope case (situational awareness underwater) | **the interesting middle** — you need the *semantics* to survive, not the pixels |

> **Say this in an interview**: "The right question isn't distortion vs perception, it's what the downstream consumer is. For underwater situational awareness, a reconstruction that preserves object identity and position at 2% BER beats a pixel-faithful one that doesn't arrive." That's the kind of framing that turns a metric argument into a systems argument.

**Diffusion as decoder** (ultra-low bitrate): transmit a very coarse latent (a few hundred bits), then sample a plausible image with a conditional diffusion model. The diffusion model doesn't compress — it *synthesizes* the missing high frequencies. Representative: CDC, PerCo (which goes down to 0.003 bpp). Costs: many denoising steps, and stochastic sampling may produce a different reconstruction unless the seed and sampling path are fixed.

<a id="16-learned-vs-traditional--the-honest-tradeoff"></a>

### 6.4.6 Learned vs traditional — the honest tradeoff

**Better:**
- Low-bitrate visual quality — no fixed block grid, so no blocking artifacts.
- Can optimize a perceptual loss (LPIPS) or an adversarial loss *directly*.
- Joint optimization with downstream tasks (compression + super-resolution, or compression optimized for a detector's accuracy).
- Adapts to a content domain by fine-tuning — a codec specialized for video calls, or for underwater imagery.

**Worse:**
- **Decode speed**: neural inference, 10–100× slower, wants a GPU. Traditional decode is a fixed-function block that costs milliwatts.
- **No hardware acceleration**: every phone has an H.264/HEVC/AV1 decoder in silicon. None has a neural-codec decoder.
- **High bitrate**: the advantage disappears; traditional codecs are already near the fidelity ceiling.
- **No standard / interop**: the decoder *is* the model weights. Version skew between encoder and decoder is catastrophic, and shipping a 100 MB decoder is its own problem.
- **Determinism**: floating-point non-determinism across platforms breaks the entropy decoder — arithmetic coding needs bit-exact probabilities on both sides. This is a real, underappreciated blocker; the fix is integer/deterministic entropy-model arithmetic.
- **Rate control**: hitting an exact bitrate target is easy with QP, hard with a λ-parameterized model (you need multiple models or a conditional/variable-rate one).

**The realistic 2026 position to state**: neural codecs win decisively at very low bitrate and in domain-specific settings; traditional codecs own general-purpose deployment because of hardware, standards, and determinism. The interesting near-term deployments are hybrids — neural pre/post-processing around a standard codec, or neural tools *inside* a standard (which is what JVET's NNVC exploration is).

<a id="17-neural-video-tokenizers--your-genstream--aquascope-space"></a>

### 6.4.7 Neural video tokenizers — your GenStream / AquaScope space

| Model | Idea | Relevance |
|-------|------|-----------|
| **VQGAN** ([6.2.2](#622-vqgan)) | VQ + LPIPS + GAN | the baseline generative image tokenizer |
| **TiTok** ([6.2.6](#626-titok-a-one-dimensional-token-sequence)) | 1-D token sequence instead of a 2-D grid; an image → as few as 32 tokens | extreme token efficiency — your AquaScope choice |
| **MAGVIT-v2** ([6.2.4](#624-lfq-lookup-free-quantization)) | **LFQ** (lookup-free quantization), huge vocabulary, joint image+video | showed a good tokenizer beats diffusion on video generation benchmarks |
| **Cosmos Tokenizer** ([6.5](#65-the-cosmos-tokenizer-step-by-step)) | continuous *and* discrete, causal spatiotemporal, high compression ratios | your GenStream backbone — **and Nvidia will ask about it** |
| **FSQ** ([6.2.3](#623-fsq-finite-scalar-quantization)) | scalar quantization to a small level set; no codebook loss | the simple, collapse-free alternative |
| **EnCodec / SoundStream** ([6.2.5](#625-rvq-residual-vector-quantization)) | RVQ, streaming, scalable | the audio analogue; RVQ's scalability is the transferable idea |

**Causal vs non-causal tokenizers** — the point that matters for streaming: a causal temporal tokenizer only looks at past frames, so you can encode and emit as frames arrive. A non-causal one needs the whole clip. Cosmos is causal by design for exactly this reason (every released Cosmos video tokenizer is). If you're doing *real-time* streaming with a tokenizer, causality is non-negotiable, and you should say so.

**Token adaptation for network dynamics** (your GenStream contribution): the natural knob in a token-based codec is *how many tokens* you send, not a QP. That maps cleanly onto bandwidth adaptation — but unlike QP, token count changes the semantic content that survives, so the quality/rate curve is content-dependent in a different way. Exploiting "video context diversity" — that different regions/scenes tolerate token reduction very differently — is the interesting part. Expect: *"how is this different from just lowering the bitrate?"* Answer: QP degrades everything uniformly and predictably; token dropping degrades *selectively*, so you can preserve what matters if you know what matters.

---
## 6.5 The Cosmos Tokenizer, Step by Step

<a id="cosmos-tokenizer"></a>

NVIDIA's Cosmos tokenizer turns a video into a short grid of latent vectors or integer tokens and back. It is the autoencoder of [6.1.1](#611-autoencoder) with three choices that fit it to video: a fixed wavelet transform does the first factor of 4 along every axis, every layer is causal in time so frames are never read before they exist, and the discrete version quantizes with FSQ ([6.2.3](#623-fsq-finite-scalar-quantization)) instead of a learned codebook. Only the decoder's output is supervised, with pixel, perceptual, motion and adversarial losses ([6.3.1](#631-gan)). The tokenizer is the latent space of the Cosmos world models: their diffusion models generate CV8×8×8 latents and their autoregressive models predict DV8×16×16 tokens. Each step states its input and output; the shapes of the example were traced through the released network.

<a id="cosmos-example"></a>

**The example.** The discrete video tokenizer DV8×16×16 ($s_T = 8$ frames and $s_{HW} = 16$ pixels per latent position) on a $720$p clip of $17$ frames, the window the reference code tokenizes at a time, with RGB values scaled to $[-1, 1]$:

$$
x_{0:T} \in [-1, 1]^{(1 + T) \times H \times W \times 3}, \qquad T = 16,\quad H = 720,\quad W = 1280
$$

The count starts at $0$ on purpose. Frame $0$ is treated on its own and the other $T$ frames in groups, so the same network tokenizes an image, the case $T = 0$, and a video.

<a id="cosmos-notation"></a>

| Symbol | Meaning | In the example |
|---|---|---|
| $x_t \in [-1,1]^{H \times W \times 3}$ | frame $t$, for $t = 0, \dots, T$ | $17$ frames of $720 \times 1280$ |
| $s_T,\ s_{HW}$ | frames per latent frame, pixels per latent row and column | $8,\ 16$ |
| $g_j$ | group $j$ of four frames fed to the wavelet | $5$ groups |
| $\nu = (\nu_t, \nu_y, \nu_x) \in \{0,1\}^3$ | Haar sub-band of one level: on each axis, $0$ = mean and $1$ = difference | $8$ per level, $64$ after two |
| $w_j$ | wavelet coefficients of group $j$ | $180 \times 320 \times 192$ each |
| $\mathcal E,\ \mathcal D$ | encoder and decoder | $46$M and $66$M parameters |
| $z$ | encoder output, $16$ channels per position | $3 \times 45 \times 80 \times 16$ |
| $u$ | projected latent, $6$ numbers per position | $3 \times 45 \times 80 \times 6$ |
| $L_i$ | number of levels of dimension $i$ | $(8, 8, 8, 5, 5, 5)$ |
| $\hat z$ | quantized code, each value on a grid in $[-1, 1]$ | |
| $k$ | token index | $0, \dots, 63{,}999$ |
| $\hat x_{0:T}$ | reconstruction | $17$ frames of $720 \times 1280$ |

**Step 1 · Group the frames.** $x_{0:T} \;\longrightarrow\; \{g_0, g_1, \dots, g_{T/4}\}$

$$
g_0 = (x_0,\ x_0,\ x_0,\ x_0), \qquad g_j = (x_{4j-3},\ x_{4j-2},\ x_{4j-1},\ x_{4j}), \quad j = 1, \dots, \tfrac{T}{4}
$$

The first frame is copied four times to make a full group of its own; all later frames go in fours. Example: $17$ frames give $1 + 16/4 = 5$ groups. A single image gives one group of four identical copies.

**Step 2 · Two-level Haar wavelet.** $g_j \in \mathbb{R}^{4 \times H \times W \times 3} \;\longrightarrow\; w_j \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times 192}$

Write the pixels of a group as $g_j[\tau, y, x, c]$: frame $\tau \in \{0, 1, 2, 3\}$ inside the group, row $y$, column $x$, colour $c$. Each colour is transformed on its own, and the same step is applied twice.

*One level along one axis.* Two neighbouring samples $a_0, a_1$ are replaced by their mean and their half-difference:

$$
s = \frac{a_0 + a_1}{2}, \qquad d = \frac{a_0 - a_1}{2}
$$

The code uses the orthonormal Haar filters $\big(\tfrac{1}{\sqrt 2}, \pm\tfrac{1}{\sqrt 2}\big)$ on each axis and divides the 3D result by $2\sqrt 2$; together that is exactly this factor $\tfrac12$ per axis. Example: $(100, 104) \to s = 102,\ d = -2$.

*One level along all three axes.* Doing this along time, height and width turns each $2 \times 2 \times 2$ block into $8$ numbers, one per sub-band $\nu = (\nu_t, \nu_y, \nu_x) \in \{0, 1\}^3$, where on each axis $0$ takes the mean and $1$ the difference. With $\delta = (\delta_t, \delta_y, \delta_x) \in \{0, 1\}^3$ the position inside the block:

$$
w^{(1)}_{\nu}[t', y', x', c] = \frac{1}{8} \sum_{\delta \in \{0, 1\}^3} (-1)^{\,\nu \cdot \delta}\; g_j[2t' + \delta_t,\ 2y' + \delta_y,\ 2x' + \delta_x,\ c]
$$

$\nu = (0, 0, 0)$ has only plus signs, so it is the block mean. The other seven are signed differences: along time, along height, along width, or along several at once. Each axis becomes half as long.

*The second level.* The same step is applied again to all $8$ sub-bands, not only to the mean as a wavelet pyramid would do. A level-2 coefficient therefore covers a $4 \times 4 \times 4$ block. Write the offset inside it as $\delta \in \{0, 1, 2, 3\}^3$: its low bit, $\delta \bmod 2$, is the position inside the level-1 block, and its high bit, $\lfloor \delta / 2 \rfloor$, the position at level 2:

$$
w_j[y', x', (c, \nu^{(1)}, \nu^{(2)})] = \frac{1}{64} \sum_{\delta \in \{0, 1, 2, 3\}^3} (-1)^{\,\nu^{(1)} \cdot (\delta \bmod 2) \,+\, \nu^{(2)} \cdot \lfloor \delta / 2 \rfloor}\; g_j[\delta_t,\ 4y' + \delta_y,\ 4x' + \delta_x,\ c]
$$

With both labels $(0, 0, 0)$ this is the mean of the $4 \times 4 \times 4$ block; the other $63$ are signed differences. $8 \times 8 = 64$ coefficients per colour make $192$ channels for RGB. The four frames of the group are consumed by the sum over $\delta_t$, so each group becomes a single time step, which is why $w_j$ has no time axis. Nothing here is learned, and the transform is exactly invertible.

Smooth video makes most of the $63$ differences close to zero, so the transform removes pixel-level redundancy at no cost and leaves the network to compress structure. For $g_0$, the four copies of $x_0$ are identical, so every sub-band with a time difference ($\nu_t = 1$ at either level) is exactly zero: only $16$ of its $64$ channels per colour carry information. Example: stacking the $5$ groups gives $w \in \mathbb{R}^{5 \times 180 \times 320 \times 192}$.

**Step 3 · Causal convolutions and residual blocks.** $w \;\longrightarrow\; h$, the same grid with $128 \to 256 \to 512$ channels

Every 3D convolution is split into a spatial $1 \times 3 \times 3$ kernel and a temporal $3 \times 1 \times 1$ kernel, and the temporal one only looks back:

$$
\text{Conv}_s:\ y_j(p) = \sum_{\delta \in \{-1, 0, 1\}^2} \mathsf{W}_\delta\, h_j(p + \delta), \qquad
\text{Conv}_t:\ y_j = \sum_{\tau = 0}^{2} \mathsf{W}_\tau\, h_{j - \tau}, \quad h_{-1} = h_{-2} = h_0
$$

The left side of time is padded with two copies of the first frame (kernel size $3$, so $3 - 1 = 2$), never with a later one. A residual block applies the pair twice with a per-frame LayerNorm and the Swish activation:

$$
h \leftarrow h + \text{Conv}_t\,\text{Conv}_s\,\text{Swish}\,\text{LN}\Big(\text{Conv}_t\,\text{Conv}_s\,\text{Swish}\,\text{LN}(h)\Big), \qquad \text{Swish}(a) = a\, \sigma(a)
$$

LN normalizes each latent frame over its height, width and channels on its own, so even normalization statistics never mix a frame with a later one. Nothing looks forward in $j$, so output $j$ depends on $g_0, \dots, g_j$ only. An input convolution maps $192$ wavelet channels to $128$, and three levels of two residual blocks each widen them to $256$, $512$ and $512$.

**Step 4 · Downsample to the latent grid.** $h \in \mathbb{R}^{5 \times 180 \times 320 \times 256} \;\longrightarrow\; h \in \mathbb{R}^{3 \times 45 \times 80 \times 512}$

The wavelet already divided every axis by $4$. The rest comes from learned factor-$2$ reductions, $\log_2 s_{HW} - 2$ in space and $\log_2 s_T - 2$ in time, here $2$ and $1$, each a strided convolution plus a fixed average:

$$
\text{Down}_2(h) = \text{Conv}^{\text{stride } 2}(h) + \text{AvgPool}_2(h)
$$

In time, the first latent frame is again padded with itself, so $2m + 1$ groups become $m + 1$ latent frames. Level $0$ halves time, height and width ($5 \times 180 \times 320 \to 3 \times 90 \times 160$), level $1$ halves height and width ($\to 3 \times 45 \times 80$), and level $2$ keeps the grid:

$$
1 + \frac{T}{s_T} = 1 + \frac{16}{8} = 3 \text{ latent frames}, \qquad \frac{H}{s_{HW}} \times \frac{W}{s_{HW}} = 45 \times 80 \text{ positions each}
$$

**Step 5 · Attention at the bottleneck.** $h \in \mathbb{R}^{3 \times 45 \times 80 \times 512} \;\longrightarrow\; z \in \mathbb{R}^{3 \times 45 \times 80 \times 16}$

Only at the smallest grid, between two residual blocks, sit one spatial self-attention within each latent frame and one causal temporal self-attention at each position $p$, both single-head with $c = 512$ channels and a residual connection:

$$
\begin{aligned}
\text{spatial:}\quad & a_j = \operatorname{softmax}\!\Big(\frac{Q_j K_j^{\top}}{\sqrt c}\Big) V_j \qquad \text{over the } 45 \cdot 80 \text{ positions of frame } j \\[6pt]
\text{temporal:}\quad & a_j(p) = \sum_{i \le j} \frac{\exp\big(q_j(p)^{\top} k_i(p) / \sqrt c\big)}{\sum_{i' \le j} \exp\big(q_j(p)^{\top} k_{i'}(p) / \sqrt c\big)}\, v_i(p)
\end{aligned}
$$

The temporal mask is the causal mask of [Chapter 3](#3-transformer-from-input-to-output), step 6, laid along time instead of along a sentence. A LayerNorm, Swish and a $1 \times 3 \times 3$ plus $3 \times 1 \times 1$ convolution then bring the channels to $16$. Running the released architecture confirms the causality: changing frames $9$ to $16$ changes latent frame $2$ only, and latent frames $0$ and $1$ stay bit-identical. Latent frame $j \ge 1$ summarizes frames $8j - 7$ to $8j$ and everything before them.

**Step 6 · Project to six numbers per position.** $z \in \mathbb{R}^{3 \times 45 \times 80 \times 16} \;\longrightarrow\; u \in \mathbb{R}^{3 \times 45 \times 80 \times 6}$

$$
u = z\, W_q, \qquad W_q \in \mathbb{R}^{16 \times 6} \quad (\text{a } 1 \times 1 \times 1 \text{ convolution})
$$

One scalar per quantizer dimension. The continuous tokenizer stops before this step: its latent is $z$ itself, $16$ numbers per position, with no sampling and no KL term. It is the plain autoencoder of 6.1.1, not the VAE of 6.1.2.

**Step 7 · Finite scalar quantization.** $u \;\longrightarrow\; \hat z$ on a grid, and $k \in \{0, \dots, 63{,}999\}$

Each of the $6$ dimensions is squashed into a bounded range and rounded to one of $L_i$ levels, with $L = (8, 8, 8, 5, 5, 5)$:

$$
\begin{aligned}
\bar u_i &= \tanh(u_i + \beta_i)\, \eta_i - o_i, \qquad \eta_i = \frac{(L_i - 1)(1 + \varepsilon)}{2}, \qquad \beta_i = \operatorname{artanh}\frac{o_i}{\eta_i} \\[6pt]
\hat z_i &= \frac{\operatorname{round}(\bar u_i)}{\lfloor L_i / 2 \rfloor}, \qquad o_i = \tfrac12 \text{ if } L_i \text{ is even, else } 0, \qquad \varepsilon = 10^{-3}
\end{aligned}
$$

The offset $o_i$ centres an even number of levels, $\beta_i$ makes $u_i = 0$ land on $0$, and $\varepsilon$ keeps the outermost levels reachable. For $L_i = 8$ the range is $(-4.004, 3.004)$, the rounded values are $-4, \dots, 3$ and $\hat z_i \in \{-1, -0.75, \dots, 0.75\}$; for $L_i = 5$, $\hat z_i \in \{-1, -0.5, 0, 0.5, 1\}$. The six digits are packed into one integer in mixed radix, and the decoder unpacks it:

$$
k = \sum_{i=1}^{6} \lfloor L_i / 2 \rfloor (\hat z_i + 1)\, b_i, \qquad b = (1, 8, 64, 512, 2560, 12800), \qquad \hat z_i = \frac{\lfloor k / b_i \rfloor \bmod L_i}{\lfloor L_i / 2 \rfloor} - 1
$$

The vocabulary is $8^3 \cdot 5^3 = 64{,}000$ codes, so a token carries $\log_2 64{,}000 = 15.97$ bits. Rounding has zero gradient almost everywhere, so training uses the straight-through estimator of 6.2.1:

$$
\operatorname{round}_{\text{STE}}(\bar u) = \bar u + \operatorname{sg}\big[\operatorname{round}(\bar u) - \bar u\big], \qquad \frac{\partial \hat z_i}{\partial \bar u_i} = \frac{1}{\lfloor L_i / 2 \rfloor}
$$

Example: $3 \times 45 \times 80 = 10{,}800$ tokens, $10{,}800 \times 15.97 \approx 172{,}000$ bits, about $21$ KiB for the $17$ frames.

<a id="cosmos-vq-vs-fsq"></a>

> **Note · How FSQ differs from VQ.** Both turn a vector into one of finitely many codes and send only the code's index. They differ in where the codes come from. VQ ([6.2.1](#621-vq-vae)) keeps a learned table of $K$ vectors and picks the nearest one. FSQ fixes the codes to a grid and rounds each coordinate on its own:
>
> $$
> \begin{aligned}
> \text{VQ:}\quad & k = \arg\min_{j \in \{1, \dots, K\}} \lVert u - e_j \rVert^2, \qquad \hat z = e_k, \qquad e_1, \dots, e_K \in \mathbb{R}^d \text{ learned} \\[6pt]
> \text{FSQ:}\quad & \hat z_i = \frac{\operatorname{round}(\bar u_i)}{\lfloor L_i / 2 \rfloor} \text{ for each } i, \qquad k = \text{the digits of } \hat z \text{ in mixed radix}, \qquad \text{no parameters}
> \end{aligned}
> $$
>
> **A two-dimensional example.** Take $u = (0.7, -0.2)$. VQ has $K = 4$ learned entries. FSQ has $L = (3, 3)$, whose $9$ codes are the fixed points $\{-1, 0, 1\}^2$:
>
> $$
> \begin{aligned}
> \text{VQ:}\quad & e_{1:4} = (0.9, 0.1),\ (-0.5, 0.8),\ (-0.6, -0.7),\ (2, 2) \\
> & \lVert u - e_j \rVert^2 = 0.13,\ 2.44,\ 1.94,\ 6.53 \;\Rightarrow\; k = 1,\quad \hat z = (0.9, 0.1) \\[6pt]
> \text{FSQ:}\quad & \bar u = 1.001 \tanh u = (0.60, -0.20) \;\Rightarrow\; \hat z = (1, 0) \\
> & \text{digits } (2, 1) \;\Rightarrow\; k = 2 + 3 \cdot 1 = 5
> \end{aligned}
> $$
>
> VQ cuts the plane into the Voronoi cells of the $e_j$, and the cells move as the entries learn. FSQ cuts it into fixed squares.
>
> **Why VQ needs two extra losses.** The straight-through estimator writes $\hat z = u + \operatorname{sg}[e_k - u]$, so the reconstruction gradient reaches $u$ but never the table. The table needs a loss of its own, and the encoder needs to be held near the table:
>
> $$
> \mathcal{L}_{\text{VQ}} = \mathcal{L}_{\text{rec}} + \underbrace{\lVert \operatorname{sg}[u] - e_k \rVert^2}_{\text{codebook: } e_k \text{ moves toward } u} + \beta\, \underbrace{\lVert u - \operatorname{sg}[e_k] \rVert^2}_{\text{commitment: } u \text{ stays near } e_k}, \qquad \beta \approx 0.25
> $$
>
> Without the commitment term, $u$ can drift far from every entry. The straight-through copy then ignores a large gap $e_k - u$ and the gradient becomes a poor guide. The EMA variant replaces the codebook term by setting each $e_k$ to the running mean of the $u$ assigned to it, a k-means step per batch.
>
> **Why VQ collapses.** Both extra terms touch only the chosen entry $e_k$. An entry that is never the nearest to any $u$, like $e_4 = (2, 2)$ above, gets no gradient and stays where it was initialized, unused. If only $300$ of $K = 1024$ entries are ever used, a token carries at most $\log_2 300 = 8.2$ bits instead of $10$. The FSQ paper measures this in its MaskGIT experiments: VQ uses fewer than half its codes once $K > 2^{11}$ and no more than about $2^{10}$ codes even in larger codebooks, while FSQ uses almost all of $2^{14}$. The fixes in 6.2.1, such as dead-code resets, EMA and k-means initialization, all patch this failure.
>
> **Why FSQ needs none of it.** There is no table, so nothing has to be pulled and nothing can be left behind. $\hat z$ is a fixed function of $u$, and the only trained parts are the encoder, which learns where to put $u$, and the decoder, which learns what each grid point means. Every one of the $\prod_i L_i$ codes is reached by moving each scalar into the right interval. However large $u$ grows, $\tanh$ keeps $\bar u$ inside the grid, which does the job of the commitment loss in VQ. The whole objective is the reconstruction loss of step 9.
>
> **What FSQ gives up.** VQ can place its codes anywhere in $\mathbb{R}^d$, densely where the data are dense. FSQ's codes form a uniform, axis-aligned grid in a few dimensions: the paper notes $d < 10$ for FSQ against $d \ge 512$ for typical VQ. The learned layers around the grid make up for it. The projection of step 6 and the encoder bend the data to fit the grid, and the decoder reads it back. With this design the FSQ paper reports performance competitive with VQ on image generation, depth estimation, colorization and panoptic segmentation.
>
> | | VQ | FSQ |
> |---|---|---|
> | Codebook | $K$ learned vectors in $\mathbb{R}^d$, $K \cdot d$ parameters | fixed grid of $\prod_i L_i$ points, no parameters |
> | Quantize one position | distances to all $K$ entries, $K \cdot d$ multiply-adds | round $d$ scalars |
> | Loss | reconstruction + codebook + $\beta \cdot$ commitment | reconstruction only |
> | Gradient | encoder by straight-through, table by its own loss or EMA | encoder by straight-through, no table |
> | Failure mode | collapse: unused entries never move | none in the codebook |
> | Shape of the codes | free, follows the data | uniform box, the encoder adapts instead |

**Step 8 · Decode.** $k \;\longrightarrow\; \hat x_{0:T} \in [-1, 1]^{17 \times 720 \times 1280 \times 3}$

The indices become codes, a $1 \times 1 \times 1$ convolution maps $6$ channels back to $16$, and the encoder runs in mirror image: three residual blocks per level instead of two, the same bottleneck attention, and upsampling instead of downsampling:

$$
\begin{aligned}
\text{Up}^{s}_2(h) &= \text{Conv}_s\big(\text{NN}_2(h)\big) + \text{NN}_2(h) \qquad \text{(nearest-neighbour doubling of height and width)} \\[6pt]
\text{Up}^{t}_2(h) &:\ (h_0, h_1, \dots, h_m) \mapsto (h_0, h_1, h_1, \dots, h_m, h_m), \quad \text{then } + \text{Conv}_t
\end{aligned}
$$

The temporal step repeats every latent frame except the first, so $m + 1$ frames become $2m + 1$, the inverse of step 4. Example: $3 \times 45 \times 80$, then $3 \times 90 \times 160$ (spatial), then $5 \times 180 \times 320$ (spatial and temporal); an output convolution to $192$ channels; the inverse Haar transform gives $20$ frames of $720 \times 1280$, and the $3$ extra copies of frame $0$ are dropped: $17$ frames.

**Step 9 · Training losses.** $(x_{0:T},\ \hat x_{0:T}) \;\longrightarrow\; \mathcal{L}$

Two stages, and only the decoder output is supervised. Stage $1$:

$$
\mathcal{L}_1 = \big\lVert \hat x_{0:T} - x_{0:T} \big\rVert_1, \qquad
\mathcal{L}_{\text{Perc}} = \frac{1}{L} \sum_{l=1}^{L} \sum_{t} \alpha_l \big\lVert \text{VGG}_l(\hat x_t) - \text{VGG}_l(x_t) \big\rVert_1
$$

Stage $2$ fine-tunes with an optical-flow loss, a Gram-matrix loss and an adversarial loss ([6.3.1](#631-gan)):

$$
\begin{aligned}
\mathcal{L}_{\text{Flow}} &= \frac{1}{T} \sum_{t=1}^{T} \big\lVert \text{OF}(\hat x_t, \hat x_{t-1}) - \text{OF}(x_t, x_{t-1}) \big\rVert_1 + \frac{1}{T} \sum_{t=0}^{T-1} \big\lVert \text{OF}(\hat x_t, \hat x_{t+1}) - \text{OF}(x_t, x_{t+1}) \big\rVert_1 \\[6pt]
\mathcal{L}_{\text{Gram}} &= \frac{1}{L} \sum_{l=1}^{L} \sum_{t} \alpha_l \big\lVert \text{GM}_l(\hat x_t) - \text{GM}_l(x_t) \big\rVert_1
\end{aligned}
$$

$\text{VGG}_l$ is layer $l$ of a pretrained VGG-19, $\text{OF}$ an optical-flow estimator (the paper cites RAFT), and $\text{GM}_l$ the Gram matrix of a layer's features, which measures texture statistics. Each term fixes a different failure: $\mathcal{L}_1$ alone gives blur, the perceptual loss makes features match, the flow loss makes the motion between reconstructed frames match the real motion so the video does not flicker, the Gram loss restores sharp texture, and the adversarial loss adds detail where compression is heaviest. There is no commitment loss, because FSQ needs none, and no KL term, because the latent is a plain autoencoder's. Images and videos alternate in the mini-batches, which works because an image is a one-frame video under the causal design. The paper does not give the loss weights.

**Step 10 · Tokens, bitrate and delay.** $(s_T,\ s_{HW},\ H,\ W,\ \text{fps}) \;\longrightarrow\; N_{\text{tok}},\ R,\ \text{delay}$

$$
N_{\text{tok}} = \Big(1 + \frac{T}{s_T}\Big) \frac{H}{s_{HW}} \frac{W}{s_{HW}}, \qquad
R = \frac{H\, W}{s_{HW}^2} \cdot \log_2 64{,}000 \cdot \frac{\text{fps}}{s_T}, \qquad
\text{delay} \ge \frac{s_T}{\text{fps}}
$$

$R$ is the raw rate of sending every index as a fixed $15.97$-bit word, with no entropy coding of the tokens. For $720$p at $30$ fps:

<a id="cosmos-variants"></a>

| Video tokenizer | Tokens per latent frame | Raw rate | One latent frame waits for | DAVIS PSNR, discrete / continuous |
|---|---|---|---|---|
| $4 \times 8 \times 8$ | $14{,}400$ | $1.72$ Mbit/s | $4$ frames, $133$ ms | $28.81$ / $32.80$ dB |
| $8 \times 8 \times 8$ | $14{,}400$ | $862$ kbit/s | $8$ frames, $267$ ms | $27.51$ / $30.61$ dB |
| $8 \times 16 \times 16$ | $3{,}600$ | $216$ kbit/s | $8$ frames, $267$ ms | $25.09$ / $27.60$ dB |

The PSNR values are the paper's Cosmos-0.1 models on DAVIS; the later Cosmos-Tokenize1 checkpoints share the architecture and were trained on longer clips. Two readings matter. First, the discrete tokens cost $2.5$ to $4$ dB against the continuous latent at the same ratio, the price of $15.97$ bits per position against $16$ floating-point numbers. Second, causal does not mean frame by frame. Latent frame $j \ge 1$ needs frames up to $s_T j$, so the encoder emits one latent frame per $s_T$ input frames. What causality buys is that it never waits for more than that and never needs the rest of the clip, whereas a non-causal tokenizer must hold the whole window before emitting anything. The image tokenizers, CI and DI, have one-frame delay and no temporal compression. For streaming, $s_T$ is the trade between rate and delay.

<a id="cosmos-shapes"></a>

| Step | Input | Output | DV8×16×16, $17$ frames of $720$p |
|---|---|---|---|
| 1 · Group | $1 + T$ frames | $1 + T/4$ groups | $17 \to 5$ |
| 2 · Wavelet | group of $4 \times H \times W \times 3$ | $\frac{H}{4} \times \frac{W}{4} \times 192$ | $5 \times 180 \times 320 \times 192$ |
| 3 · Causal blocks | $w$ | $h$, $128 \to 512$ channels | $5 \times 180 \times 320 \times 256$ |
| 4 · Downsample | $h$ | $1 + T/s_T$ latent frames | $3 \times 45 \times 80 \times 512$ |
| 5 · Attention, output | $h$ | $z$, $16$ channels | $3 \times 45 \times 80 \times 16$ |
| 6 · Project | $z$ | $u$, $6$ channels | $3 \times 45 \times 80 \times 6$ |
| 7 · FSQ | $u$ | $\hat z$ and $k$ | $10{,}800$ tokens, $21$ KiB |
| 8 · Decode | $k$ | $\hat x_{0:T}$ | $17 \times 720 \times 1280 \times 3$ |
| 9 · Losses | $x, \hat x$ | $\mathcal{L}$ | L1, VGG, flow, Gram, adversarial |
| 10 · Budget | $s_T, s_{HW}$, fps | $N_{\text{tok}}, R$, delay | $216$ kbit/s, $267$ ms |

**Primary sources:** [Cosmos World Foundation Model Platform, section 4](https://arxiv.org/abs/2501.03575) · [NVIDIA/Cosmos-Tokenizer code and configurations](https://github.com/NVIDIA/Cosmos-Tokenizer) · [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)

---


# 7. Inference

One request, followed through everything a serving system does with the decoder-only model of [Chapter 3](#3-transformer-from-input-to-output): the prompt is pushed through the model once, every later token is produced one step at a time against a cache, and tokens are drawn from the logits. Each step states its input, the operation and its output, with the cost next to it; the recurring question is not *what* is computed, which Chapter 3 settled, but *how many bytes and FLOPs it takes and which of the two is the limit*.

<a id="inference-math"></a>

**The example.** LLaMA-3-8B in BF16 on one H100: $L = 32$ layers, $d = 4096$, $h = 32$ query heads, $h_{kv} = 8$ key/value heads (GQA), $d_k = 128$, $d_{\text{ff}} = 14{,}336$, vocabulary $V = 128{,}256$, $N = 8.03 \times 10^{9}$ parameters, $16$ GB in BF16. Of these, $N_{\text{mat}} = 7.50 \times 10^{9}$ sit in the matrices that multiply activations (the rest is the input embedding table, which is only looked up), so the bytes a step must stream are $W_{\text{bytes}} = 2 N_{\text{mat}} \approx 15\,\text{GB}$. A request with a prompt of $n = 1{,}000$ tokens asks for $m = 200$ new tokens. The GPU moves $\beta = 3.35\,\text{TB/s}$ from its memory and performs $\pi = 989\,\text{TFLOP/s}$ in BF16, so its ridge point is $\pi / \beta \approx 295$ FLOP per byte: a kernel that does fewer FLOPs per byte it reads is waiting on memory, one that does more is waiting on arithmetic. The runnable code at the end uses a toy model instead, so that every identity below can be checked.

<a id="inference-notation"></a>

| Symbol | Meaning | In the example |
|---|---|---|
| $L,\ d,\ h,\ h_{kv},\ d_k$ | layers, model width, query heads, key/value heads, head width $d / h$ | $32,\ 4096,\ 32,\ 8,\ 128$ |
| $N$, $N_{\text{mat}}$, $W_{\text{bytes}} = 2N_{\text{mat}}$ | all parameters, the matrix parameters that are multiplied, and the BF16 bytes of the latter | $8.03\,\text{B}$, $7.50\,\text{B}$, $15\,\text{GB}$ |
| $n,\ m,\ t$ | prompt length, tokens to generate, current sequence length | $1{,}000,\ 200,\ n \le t < n + m$ |
| $B$ | sequences decoded together | $1$ to a few hundred |
| $K_\ell, V_\ell \in \mathbb{R}^{t \times h_{kv} d_k}$ | keys and values of layer $\ell$ for every token so far | the KV cache |
| $c = 2 \cdot L \cdot h_{kv} \cdot d_k \cdot 2\,\text{bytes}$ | cache bytes per token | $131{,}072$ B $= 128$ KiB |
| $\beta,\ \pi$ | memory bandwidth and peak compute of the accelerator | $3.35\,\text{TB/s}$, $989\,\text{TFLOP/s}$ |
| $I = \text{FLOPs} / \text{bytes}$ | arithmetic intensity, compared with the ridge $\pi / \beta$ | ridge $\approx 295$ |
| $\mathbf z \in \mathbb{R}^{V}$, $p$, $T$ | logits of one position, the distribution drawn from, temperature | |

**Step 1 · Prefill.** $\mathbf s \in \{0, \dots, V-1\}^{n} \;\longrightarrow\; \mathbf z_n \in \mathbb{R}^{V},\ \{K_\ell, V_\ell\}_{\ell = 1}^{L}$

The prompt is one forward pass of Chapter 3 in its decoder-only form: all $n$ positions at once, causal mask, and only the last row of the logits is needed, because it is the distribution of the first new token. Every layer's keys and values are kept (step 2). The cost counts the two FLOPs of every multiply-add against every weight, plus attention:

$$
\text{FLOPs}_{\text{prefill}} \approx 2 N_{\text{mat}} n + 4 n^{2} d L, \qquad
\text{bytes}_{\text{prefill}} \approx W_{\text{bytes}} + c\, n, \qquad
I_{\text{prefill}} \approx \frac{2 N_{\text{mat}} n}{2 N_{\text{mat}}} = n
$$

With $n = 1{,}000$: $15.0$ TFLOP for the weights and $0.5$ TFLOP for attention, $15.5$ TFLOP in all, against $15$ GB read once, so $I \approx 1{,}000 \gg 295$: prefill is **compute-bound**, and its time is $15.5 \times 10^{12} / \pi \approx 16$ ms at peak, two to three times that in practice. This is the time to first token, TTFT. The $n^{2}$ term is $3\%$ here and overtakes the weights only when $n > 2N_{\text{mat}} / (4 d L) \approx 29{,}000$ tokens.

<a id="93-kv-cache--the-thing-that-dominates-inference"></a>
<a id="51-kv-cache--the-thing-that-dominates-inference"></a>

**Step 2 · The KV cache.** $\{K_\ell, V_\ell\}$ of the first $t$ tokens $\;\longrightarrow\;$ kept in memory, $c\,t\,B$ bytes

In Chapter 3, step 2, the key of token $i$ is $k_i = x_i W^{K}$: it depends only on token $i$'s own input to that layer, and with a causal mask that input depends only on tokens $\le i$. So when token $t + 1$ arrives, every $k_i$ and $v_i$ with $i \le t$ is exactly what it was when it was first computed. Recomputing them would cost a full prefill of $t$ tokens per step, $\sum_{t = n}^{n + m - 1} 2 N_{\text{mat}}\, t \approx 2 N_{\text{mat}}\, m\, (n + m/2)$, quadratic in the length; storing them costs memory instead:

$$
\text{cache bytes} = \underbrace{2}_{K \text{ and } V} \cdot L \cdot h_{kv} \cdot d_k \cdot t \cdot B \cdot \underbrace{2}_{\text{bytes}} = c\,t\,B,
\qquad c = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 \text{ B}
$$

One token costs $128$ KiB; $8{,}192$ tokens cost $1.07$ GB per sequence, $16$ sequences at that length cost as much as the weights. $h_{kv}$ is in the formula and $h$ is not: GQA shares each key/value head among $h / h_{kv} = 4$ query heads, which is why LLaMA-3-8B's cache is $4\times$ smaller than LLaMA-2-7B's ($h_{kv} = 32$, $512$ KiB per token) at the same $d$. Everything else about serving follows from this formula: long context is expensive in memory before it is expensive in compute, the cache is stored in fixed-size blocks so that sequences of unknown final length do not fragment memory (paged attention), prompts shared by many requests are cached once (prefix caching), and the cache is quantized to 8 or 4 bits when memory is the constraint.

<a id="94-prefill-vs-decode"></a>
<a id="52-prefill-vs-decode"></a>

**Step 3 · One decode step.** $x_t \in \{0, \dots, V-1\},\ \{K_\ell, V_\ell\}_{1:t-1} \;\longrightarrow\; \mathbf z_t \in \mathbb{R}^{V},\ \{K_\ell, V_\ell\}_{1:t}$

The new token is embedded at position $t$, and in every layer its single query reads the whole cache:

$$
\begin{aligned}
q_t &= x_t W^{Q}, \qquad k_t = x_t W^{K}, \qquad v_t = x_t W^{V} \qquad (k_t, v_t \text{ appended to the cache}) \\[6pt]
a_t &= \operatorname{softmax}\!\Big( \frac{q_t K_{1:t}^{\top}}{\sqrt{d_k}} \Big) V_{1:t} \qquad \text{per head}
\end{aligned}
$$

The arithmetic is tiny and the traffic is not: every weight matrix is read once to produce one token, and so is the cache; only the embedding table is spared, because a lookup gathers one $8$ KB row and multiplies nothing.

$$
\text{FLOPs}_{\text{step}} \approx 2 N_{\text{mat}} + 4 t d L \approx 15.5\,\text{GFLOP}, \qquad
\text{bytes}_{\text{step}} \approx W_{\text{bytes}} + c\,t \approx 15.1\,\text{GB}, \qquad
I_{\text{step}} \approx 1
$$

$$
\text{time}_{\text{step}} \ge \max\!\Big( \frac{\text{FLOPs}}{\pi},\ \frac{\text{bytes}}{\beta} \Big) = \max(0.016\,\text{ms},\ 4.5\,\text{ms}) = 4.5\,\text{ms}
$$

One sequence cannot be decoded faster than about $220$ tokens per second on this GPU no matter how fast its arithmetic is; the step is **memory-bound** by a factor of nearly $300$. This asymmetry is the whole subject of inference optimization. Anything that shrinks the bytes helps decode nearly in proportion: weights in INT4 ($W_{\text{bytes}} \approx 3.9$ GB with scales) bring the step to $1.2$ ms, and a smaller cache (GQA, quantized cache) matters once $c\,t$ is comparable with $W_{\text{bytes}}$. Anything that only adds FLOPs is free until $I$ reaches the ridge. The time per output token, TPOT, is this number.

<a id="54-batching-packing-and-scheduling"></a>

**Step 4 · Batching.** $B$ sequences, each at its own length $t_b$ $\;\longrightarrow\;$ one step for all of them

The weights are read once per step whatever $B$ is; the caches are not shared. So

$$
\text{time}_{\text{step}}(B) \approx \max\!\Big( \frac{2 N_{\text{mat}} B}{\pi},\ \frac{W_{\text{bytes}} + c \sum_b t_b}{\beta} \Big), \qquad
\text{throughput} = \frac{B}{\text{time}_{\text{step}}(B)}
$$

At $t = 1{,}000$: $B = 8$ takes $4.8$ ms for $1{,}700$ tokens/s, $B = 64$ takes $7.0$ ms for $9{,}200$ tokens/s, $B = 256$ takes $14.5$ ms for $17{,}700$ tokens/s. Batching amortizes the weight read, which is why it raises decode throughput by an order of magnitude at a modest cost in latency, and why it barely helps prefill, which was compute-bound already. It does not amortize the cache read: per generated token the cache costs $c\,t$ bytes against $2 N_{\text{mat}}$ FLOPs, so decode stays memory-bound at any batch size once $t > 2 N_{\text{mat}} / (c \cdot \pi / \beta) \approx 400$ tokens. Two more limits bound $B$. Memory: $B_{\max}(t) = (\text{HBM} - 2N) / (c\,t)$ with all $16$ GB of weights resident, about $120$ sequences of $4{,}096$ tokens or $60$ of $8{,}192$ on $80$ GB. Arrivals: requests do not start and finish together, so the batch is rebuilt every step (continuous batching), a finishing sequence is replaced by a waiting one, and long prompts are prefilled in chunks interleaved with decode steps so that one $n = 30{,}000$ arrival does not freeze everyone else's tokens for a second.

<a id="95-sampling"></a>
<a id="53-sampling"></a>

**Step 5 · Sampling.** $\mathbf z_t \in \mathbb{R}^{V} \;\longrightarrow\; x_{t+1} \sim p$

The logits become a distribution through a temperature, the distribution is truncated, and the remainder is renormalized; the token is drawn from the result. Chapter 3's step 9 is the $T = 1$, no-truncation case.

$$
p_T(v) = \frac{e^{z_v / T}}{\sum_{u} e^{z_u / T}}, \qquad
T \to 0 \Rightarrow p_T \to \text{one-hot at } \arg\max_v z_v \ (\text{greedy}), \qquad
T > 1 \Rightarrow \text{flatter}
$$

$$
\begin{aligned}
\text{top-}k:&\quad \mathcal S = \text{the } k \text{ largest } p_T(v) \\[4pt]
\text{top-}p:&\quad \mathcal S = \text{the smallest prefix of the sorted list with } \textstyle\sum_{v \in \mathcal S} p_T(v) \ge p \\[4pt]
\text{min-}p:&\quad \mathcal S = \{ v : p_T(v) \ge p_{\min} \max_u p_T(u) \}
\end{aligned}
$$

$$
\begin{aligned}
\tilde p(v) &= \frac{p_T(v)\, \mathbf 1[v \in \mathcal S]}{\sum_{u \in \mathcal S} p_T(u)} \\[6pt]
\text{repetition penalty:}&\quad z_v \leftarrow z_v / \rho \ \ (z_v > 0) \quad \text{or} \quad z_v\, \rho \ \ (z_v \le 0) \quad \text{for every } v \text{ already generated},\ \rho > 1
\end{aligned}
$$

Example, five logits $\mathbf z = (2.0, 1.0, 0.5, 0.0, -1.0)$:

| rule | distribution drawn from |
|---|---|
| $T = 1$ | $(0.563,\ 0.207,\ 0.126,\ 0.076,\ 0.028)$ |
| $T = 0.5$ | $(0.829,\ 0.112,\ 0.041,\ 0.015,\ 0.002)$ |
| top-$k$, $k = 2$ | $(0.731,\ 0.269,\ 0,\ 0,\ 0)$ |
| top-$p$, $p = 0.9$ | $(0.579,\ 0.213,\ 0.129,\ 0.078,\ 0)$: the mass reaches $0.9$ only with the fourth token |

Temperature changes the shape, truncation removes the tail where a single draw of a $10^{-4}$ token derails the text; top-$p$ adapts the cut to the shape (one token when the model is sure, many when it is not), which fixed $k$ does not. The order matters: temperature first, then truncation, then renormalization. For anything that is scored or compared, including a quality model's outputs, use $T = 0$: the result is then a deterministic function of the prompt.

<a id="inference-shapes"></a>

| Step | Input | Output | LLaMA-3-8B on one H100 |
|---|---|---|---|
| 1 · Prefill | $n$ prompt tokens | $\mathbf z_n$, caches of $L$ layers | $15.5$ TFLOP, compute-bound, $\approx 16$ ms |
| 2 · KV cache | $K_\ell, V_\ell$ of $t$ tokens | $c\,t\,B$ bytes kept | $128$ KiB per token, $1.07$ GB at $8{,}192$ |
| 3 · Decode step | one token + cache | $\mathbf z_t$, cache of $t$ tokens | $15$ GB read for $15$ GFLOP, $\ge 4.5$ ms |
| 4 · Batching | $B$ sequences | one step for all | $B = 64$: $7.0$ ms, $9{,}200$ tokens/s |
| 5 · Sampling | $\mathbf z_t$ | $x_{t+1} \sim \tilde p$ | $T$, top-$k$, top-$p$, min-$p$ |

<a id="coding-inference"></a>
<a id="coding-kv-cache"></a>
<a id="coding-llm-sampling"></a>

## PyTorch implementation: KV cache and sampling

[inference_kv_cache.py](./inference_kv_cache.py) implements the steps on a toy decoder-only model: two layers with $d = 32$, trained on four sentences over a nine-word vocabulary. [test_inference_kv_cache.py](./test_inference_kv_cache.py) checks the claims: decoding against the cache reproduces the full forward pass to $10^{-9}$, the cache-size formula gives $128$ KiB per token for LLaMA-3-8B, the sampling rules give the numbers of step 5 and their limits ($T \to 0$, $k = 1$, $p = 1$), and seeded sampling is reproducible and never leaves the truncated set.

```bash
.venv-transformer/bin/python inference_kv_cache.py
.venv-transformer/bin/python -m unittest -v test_inference_kv_cache
```

**The model.** Step 2 of Chapter 3 with a causal mask, written so that past keys and values are reused: `cache` holds $K_{1:t-1}$ and $V_{1:t-1}$ of the layer, the new keys and values are concatenated, and the mask is shifted by `t_past` so that the new rows see everything before them. `Block` and `CausalLM` use the pre-norm order of Chapter 4, $x \leftarrow x + \operatorname{MHA}(\operatorname{LN}(x))$ then $x \leftarrow x + \operatorname{FFN}(\operatorname{LN}(x))$, with GELU and a final LayerNorm before the head, rather than the post-norm block of Chapter 3, steps 3 and 4; the cache argument does not depend on that choice.

<!-- inference-code: inference-imports -->

<!-- inference-code: inference-model -->

**Steps 1 to 3.** `prefill` runs the prompt once and keeps every layer's cache; `decode_step` feeds one token at position `t_past`; `cache_bytes` is the formula of step 2; `generate` is the generation loop, prefill once and then one decode step per token, with `use_cache=False` as the quadratic alternative that recomputes the whole prefix.

<!-- inference-code: inference-cache -->

**Step 5.** `filtered_probabilities` applies the temperature, then the truncations, then renormalizes; `temperature=0` returns the one-hot of the arg max. The top-$p$ mask keeps a token while the mass *before* it is below $p$, so the token that crosses $p$ survives and at least one token always does.

<!-- inference-code: inference-sampling -->

**The demo** trains the model, verifies the cache against a full pass, prints the table of step 5, compares greedy decoding with and without the cache, and draws one sampled continuation:

<!-- inference-code: inference-demo -->

```text
model: 2 layers, d=32, trained to loss 0.000; vocabulary of 9 words
cache matches full recompute: True (max |diff| 1.0e-05 over 28 decode steps)
KV cache of LLaMA-3-8B: 128 KiB per token, 1.07 GB at 8192 tokens
sampling on logits [2.0, 1.0, 0.5, 0.0, -1.0]:
  T=1.0     -> [0.563 0.207 0.126 0.076 0.028]
  T=0.5     -> [0.829 0.112 0.041 0.015 0.002]
  top-k=2   -> [0.731 0.269 0.000 0.000 0.000]
  top-p=0.9 -> [0.579 0.213 0.129 0.078 0.000]
greedy from "the cat": sat on the mat . the dog sat on the log .
greedy with cache == greedy without cache: True
sampled at T=1, top-p=0.9: sat on the mat . the dog sat on the log .
```

**Primary sources:** [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [Transformer inference accounting](https://jax-ml.github.io/scaling-book/inference/)

---

# 8. Scaling Laws & Transformer Accounting

> **Chapter guide:** Start system-design answers with order-of-magnitude estimates: whether parameters fit, how many FLOPs training requires, and how the KV cache grows with context and concurrency. Discuss frameworks only after that.

## 8.1 Count parameters, FLOPs, and memory

A matrix multiplication `[M,K] @ [K,N]` performs approximately `2MKN` FLOPs, counting one multiply
and one add. Let a dense model have `P` parameters in its major matmuls and process `T` training tokens:

```
forward FLOPs  ≈ 2 · P · T
training FLOPs ≈ 6 · P · T       # forward + two backward matmuls
wall time      ≈ total_FLOPs / (num_devices · peak_FLOPs/s · MFU)
```

This first-order rule ignores attention `O(L·T_context²·d)`, embeddings, MoE routing, and
recomputation. Attention can no longer be ignored at long context lengths. For MoE, estimate FLOPs
using **active parameters** per token, while memory must still hold all experts.

**First divide accelerator memory into four buckets:**

1. Weights: BF16/FP16 usually use 2 bytes per parameter; INT8 uses about 1 and INT4 about 0.5, plus scale metadata.
2. Gradients: commonly 2 or 4 bytes per parameter.
3. Optimizer states: Adam's FP32 `m,v` together use 8 bytes per parameter; some implementations also keep an FP32 master weight.
4. Activations: grow with microbatch size, sequence length, hidden size, and layer count; checkpointing trades recomputation for storage.

A 7B model therefore needs about 14 GB for BF16 weights alone. A common Adam configuration may
already require roughly 12–16 bytes per parameter **before activations**. Always state assumptions
about dtypes, master weights, gradients, and sharding. Inference has no optimizer or gradient states,
but it adds the KV cache of Chapter 7, which can dominate with long contexts and high concurrency.

## 8.2 Scaling laws: what they do and do not promise

Empirical scaling laws commonly model reducible loss as a power-law decline with parameter count,
data, or compute. Their purpose is not to claim that larger models must exhibit emergence; they help
allocate a fixed budget among more parameters, more independent data, and more training steps.

- Early fits from Kaplan et al. emphasized the sample efficiency of larger models.
- Within its experimental regime, Chinchilla found that compute-optimal training scales model size
  and training tokens at roughly the same rate as budget grows. Many earlier large models were therefore undertrained relative to their parameter count.
- The familiar "about 20 tokens per parameter" is an empirical point for a particular model family, dataset, and budget range, not a physical constant.
- **Training-optimal is not product-optimal**: a smaller model trained longer can have lower inference latency and serving cost. Data quality, deduplication, repeated epochs, target domain, and downstream post-training all shift the optimum.

## 8.3 A 60-second budgeting checklist

1. Write down `P, tokens, dtype, context, batch/concurrency`; do not report only "model size."
2. Estimate training compute with `6PT`, then use MFU to turn peak FLOPs into achievable time.
3. Estimate weights, optimizer states, gradients, activations, and KV separately; identify whether capacity or bandwidth is the constraint.
4. Add checkpointing, evaluation, failed runs, and data-pipeline overhead. A theoretical lower bound is not a project schedule.

**Primary sources:** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) ·
[Chinchilla / compute-optimal training](https://arxiv.org/abs/2203.15556) ·
[Transformer math and the `6PT` approximation](https://jax-ml.github.io/scaling-book/transformers/)

---

# 9. GPUs & the Roofline Model

> **Chapter guide:** FLOPs tell you how much arithmetic is required; the roofline model tells you whether the hardware is computing or waiting for data. This is why fewer FLOPs do not guarantee lower latency.

## 9.1 The three hard limits

A kernel is constrained by at least three resources: compute throughput, memory or communication bandwidth, and memory capacity.

```
arithmetic_intensity = FLOPs / bytes_moved
achievable_FLOPs/s <= min(peak_compute,
                          memory_bandwidth * arithmetic_intensity)
T_lower_bound >= max(FLOPs / peak_compute,
                     bytes_moved / memory_bandwidth)
```

Low arithmetic intensity lies in the bandwidth-bound region; high intensity is required to saturate
Tensor Cores. Large matmuls raise intensity by reusing tiles on chip. Elementwise operations,
normalization, and small-batch decode repeatedly read and write large tensors while doing little
arithmetic, making them more likely to be limited by bandwidth or launch overhead.

## 9.2 Memory hierarchy and practical optimization

A useful memory-hierarchy model is: registers/shared memory (small and fast) → L2 → HBM (large but
distant) → host/network. Optimization is not blindly reducing arithmetic; it is keeping data close and reusing it:

- **Tiling**: repeatedly use blocks that fit in on-chip storage for matmul or attention. Reducing HBM IO is the central idea behind FlashAttention.
- **Kernel fusion**: combine adjacent operations such as bias, activation, and normalization so intermediate tensors are written back less often.
- **Batching**: let multiple tokens reuse loaded weights, usually improving throughput while increasing queueing and tail latency.
- **Layout / shape**: Tensor-Core-friendly dimensions and contiguous access determine whether an operation actually uses the fast path.
- **Zero-copy pipeline**: avoid CPU↔GPU round trips. Video deployment should keep decode→preprocess→model on the device in particular.

## 9.3 What to measure

Use a profiler first to separate kernel time, memory copies, CPU launch gaps, and collectives; then
inspect achieved bandwidth, Tensor Core utilization, occupancy, and MFU. The roofline is an upper-bound
model that omits dependencies, synchronization, scheduling, and small-kernel launch costs, so the final answer must return to wall-clock measurements.

**Primary sources:** [Google DeepMind scaling book: rooflines](https://jax-ml.github.io/scaling-book/roofline/) ·
[NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) ·
[FlashAttention](https://arxiv.org/abs/2205.14135)

---
<a id="12-efficiency--deployment"></a>
# 10. Precision, Efficiency & Deployment

> **Chapter guide:** Low precision, pruning, and distillation count as optimizations only when the target hardware uses the corresponding kernels and end-to-end quality still meets the bar. Report parameters, FLOPs, bandwidth, and latency separately.


Your resume claims "on-device inference optimization" and a 4-bit iOS deployment — this section must be airtight.

<a id="121-quantization"></a>
## 10.1 Quantization

**The basic mapping**:
```
Symmetric:   q = round(x / s),            s = max|x| / (2^{b−1} − 1)
Asymmetric:  q = round(x / s) + z,        s = (max−min)/(2^b − 1),  z = zero-point
```

**Granularity** (biggest quality lever): per-tensor → per-channel → per-group (e.g. 128 weights share a scale). Finer = better quality, more metadata.

**What goes wrong with INT8, and the fixes:**
- **Outliers**: one large value forces a big scale, crushing the resolution of everything else. In LLMs, specific activation channels have persistent huge outliers (the "emergent outlier features" phenomenon above ~6B params). → **SmoothQuant** migrates the difficulty from activations into weights via a per-channel scaling that's mathematically absorbed into the previous layer; **LLM.int8()** keeps outlier channels in fp16.
- **Asymmetric activations**: ReLU/GELU outputs are mostly positive, so symmetric INT8 wastes half the range → use asymmetric.
- **Layer sensitivity**: embeddings, the LM head, the first and last layers, and attention logits are far more sensitive → keep them at higher precision (mixed precision).
- **Error accumulation** across layers, especially in a *recurrent* or *residual-heavy* structure.

**Methods:**
| Method | Type | Notes |
|--------|------|-------|
| **PTQ** | post-training | fast, needs a small calibration set |
| **QAT** | during training | simulate quantization in the forward pass (fake-quant + straight-through estimator), best quality, expensive |
| **GPTQ** | weight-only PTQ | layer-wise, uses second-order (Hessian) information to compensate quantization error across remaining weights |
| **AWQ** | weight-only PTQ | protects the ~1% of weight channels that matter most (identified by *activation* magnitude) via per-channel scaling |
| **GGUF K-quants** | weight-only | llama.cpp's format: mixed block sizes, Q4_K_M is the popular quality/size point |
| **NF4** | weight-only | QLoRA's 4-bit "normal float," information-theoretically optimal for normally-distributed weights |

**For image/video models specifically**: after quantizing, don't only check PSNR — check for *visual* artifacts and **temporal flicker**, because quantization error is spatially structured and can vary frame to frame, producing shimmer that a per-frame metric completely misses. That's a strong, domain-specific point to make.

**On-device (your iOS/LLMFarm case)**: 4-bit weight-only quantization on a 3B model → ~1.7 GB, which fits in the per-app memory budget. The real constraints on a phone are (1) memory ceiling and OOM-kill, (2) memory *bandwidth* — decode is bandwidth-bound, (3) sustained thermal throttling — the first 30 s of benchmarking lies, and (4) energy per token. Expect "how did you measure it?" — the answer should mention sustained load, not a single cold run.

<a id="122-pruning-and-distillation"></a>
## 10.2 Pruning and distillation

**Pruning**: remove weights (unstructured) or whole channels/heads/layers (structured).
- Unstructured pruning cuts parameter count but gives **no speedup** without sparse-tensor hardware support. Say this — it's a common misconception the interviewer may be testing.
- Structured pruning (channels, heads, layers) gives real speedup. Layer dropping is surprisingly effective for LLMs (the middle layers are the most redundant).

**Knowledge distillation**: train a small student to match a large teacher.
```
L = α·L_task + (1−α)·L_KD
L_KD = T² · KL( softmax(z_teacher/T) ‖ softmax(z_student/T) )
```
Higher temperature T softens the teacher's distribution, exposing the "dark knowledge" of relative class similarities. The `T²` factor keeps gradient magnitudes comparable across temperatures.
- **Feature/intermediate distillation** (match hidden states, attention maps) usually beats logit-only distillation.
- **For generative models**: sequence-level distillation — train the student on the teacher's *generated outputs*. This is how most small instruct models are made.

**Prefer distillation or structured pruning over unstructured pruning** if you actually need the model to run faster.

<a id="coding-quant-distill"></a>

#### Coding: Quantize/dequantize + distillation KL `[TODO · P1]`

- **Prompt:** derive symmetric/asymmetric affine qparams, quantize/dequantize per tensor and per channel, then implement softened teacher→student distillation.
- **Shape / cost:** quantization is elementwise `O(P)` plus reductions over the chosen scale axis; logit KD is `O(NC)`.
- **Pitfalls:** handle zero dynamic range, integer clipping, zero-point rounding, and scale broadcasting. In PyTorch, `kl_div(log_student, teacher_probs)` represents `KL(teacher || student)`—argument order is easy to reverse.

```python
def quantize_dequantize(x, bits=8, axis=None, asymmetric=False):
    """TODO: choose scale/zero-point, clamp integer q, then dequantize."""
    raise NotImplementedError

def distillation_loss(student_logits, teacher_logits, temperature):
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * temperature**2
```

No local quantization or KD helper exists. The KD snippet is the checked loss direction; adding both implementations and edge-case tests remains `[TODO]`.

<a id="123-parameters-vs-flops-vs-latency"></a>
## 10.3 Parameters vs FLOPs vs latency

- **Parameters** → memory footprint.
- **FLOPs** → theoretical compute.
- **Latency** → what the user experiences.

**Lower FLOPs ≠ lower latency.** The gap comes from memory bandwidth, arithmetic intensity, kernel launch overhead, operator fusion, and how well the shape maps onto the hardware's tiling. Depthwise convolutions and attention are the canonical examples of low-FLOP, bandwidth-bound operations.

**Roofline framing** (worth having): `arithmetic intensity = FLOPs / bytes moved`. Below the machine balance point you're bandwidth-bound and only reducing *memory traffic* helps; above it you're compute-bound and only reducing FLOPs helps. LLM decode sits far to the left; prefill and convolutions sit further right. **Always profile on the target device.**

> **Go deeper**: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (DeepMind). Part 1 is the cleanest treatment of rooflines and arithmetic intensity; the transformer and inference chapters derive the FLOP/byte accounting for attention, FFN, and KV cache from first principles. The worked exercises ("at what batch size does a matmul become compute-bound on this chip?") are exactly the shape of efficiency-round questions.

<a id="124-latency-reduction-checklist"></a>
## 10.4 Latency-reduction checklist

1. **Profile first** — find the actual bottleneck, don't guess from parameter count.
2. Quantize (FP16 → INT8 → INT4 weight-only), checking quality at each step.
3. Operator fusion (conv+BN+ReLU → one kernel; fused attention).
4. Structured pruning or distillation to a smaller architecture.
5. Reduce input resolution, or tile and process regions.
6. Replace global attention with windowed/local attention; use FlashAttention.
7. Cache and reuse temporal features across frames (video-specific, big win).
8. Depthwise separable convolutions — but verify actual latency, not FLOPs.
9. Avoid dynamic shapes and unsupported ops; they force fallback to CPU.
10. Batching (throughput) vs single-stream (latency) — know which you're optimizing.
11. For LLMs: KV cache, continuous batching, speculative decoding, paged attention.

<a id="125-deployment-stacks"></a>
## 10.5 Deployment stacks

| Target | Stack | Gotchas |
|--------|-------|---------|
| Nvidia GPU | TensorRT, Triton, TensorRT-LLM | op coverage, plugin authoring, engine build time; INT8 calibration |
| Apple | CoreML, ANE, MPS, MLX | ANE supports a restricted op set — unsupported ops silently fall back to GPU/CPU; fixed shapes preferred; fp16 native |
| Android | NNAPI, TFLite, QNN, ExecuTorch | huge device fragmentation |
| Server LLM | vLLM, SGLang, TensorRT-LLM | continuous batching, paged attention, prefix caching |
| On-device LLM | llama.cpp / GGUF, MLC, MLX, LLMFarm | memory ceiling, thermal throttling |
| Cross-platform | ONNX Runtime | operator-set version mismatches |

**"Works offline but fails on device" — the checklist:**
- Unsupported operators → silent CPU fallback → 10× slowdown.
- Precision mismatch (FP32 offline vs FP16/INT8 on device).
- Memory pressure / OOM at the target resolution or context length.
- Thermal throttling — sustained performance is often 50–70% of peak.
- Preprocessing mismatch: RGB vs YUV, normalization constants, bit depth, resize interpolation, color space.
- Different batch size, sequence length, or temporal-window assumptions.
- Camera-pipeline differences: color space, HDR tone mapping, compression artifacts on the *input*.
- Dynamic shapes triggering recompilation every call.

---

# 11. Parallelism

> **Chapter guide:** Parallelism is not a switch; it partitions tensors across memory, compute, and communication. A complete answer first states what each device retains, then identifies the required collective.

## 11.1 Collective operations

| Collective | What each rank receives | Common use |
|------------|------------------------|----------|
| **AllReduce** | The reduction of all inputs, identical on every device | DDP gradient sum/average |
| **ReduceScatter** | Its assigned shard of the reduced result | FSDP/ZeRO sharded gradients |
| **AllGather** | The concatenation of every rank's shards | Temporarily reconstruct sharded parameters/activations |
| **AllToAll** | A distinct shard from every other rank | MoE token dispatch |
| **Send/Recv** | A point-to-point tensor from an adjacent stage | Pipeline parallelism |

`ReduceScatter + AllGather` is semantically equivalent to AllReduce, but separating them allows the
intermediate result to remain sharded. Every participating rank must call a collective with matching
count and dtype; a missing rank can cause the operation to hang.

## 11.2 Data parallelism and FSDP / ZeRO

- **DDP**: every device holds the complete model and receives a different data shard; gradients are
  AllReduced during backward. Scaling the batch is simple, but weights, gradients, and optimizer states
  are replicated, so one device must hold the full state.
- **ZeRO/FSDP**: shard optimizer states, gradients, and parameters by stage. Parameters are AllGathered
  before computation and gradients are ReduceScattered afterward. This saves memory but shifts the
  problem toward communication, prefetching, and scheduling.
- Gradient accumulation increases global batch size and reduces optimizer-step frequency, but does not
  magically shrink the model or activations for one microbatch. Activation checkpointing is a separate
  axis that exchanges recomputation for memory.

## 11.3 Tensor, pipeline, sequence, and expert parallelism

- **Tensor parallelism (TP)**: split matrices or heads within a layer. Every layer requires collectives,
  so TP belongs inside a node with fast NVLink; frequent communication performs poorly across a slow network.
- **Pipeline parallelism (PP)**: split by layers into stages, then use microbatches to fill the pipeline.
  Communication is less frequent, at the cost of pipeline bubbles, stage imbalance, and a more complex schedule.
- **Sequence/context parallelism**: split activations along the token dimension to reduce activation
  memory for long contexts; attention must still exchange K/V or partial results.
- **Expert parallelism**: distribute MoE experts and route tokens with AllToAll; load imbalance and
  cross-node bandwidth are the main hazards.

Real systems commonly combine three or four parallel dimensions. First satisfy capacity, then keep
high-frequency collectives inside the fastest interconnect domain, and finally tune microbatches so
compute can overlap communication. Near the strong-scaling limit, adding devices can make training slower.

**Primary sources:** [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) ·
[Megatron-LM tensor parallelism](https://arxiv.org/abs/1909.08053) ·
[GPipe](https://arxiv.org/abs/1811.06965) · [ZeRO](https://arxiv.org/abs/1910.02054)

---

<a id="13-metrics-evaluation--debugging"></a>
# 12. Metrics, Evaluation & Debugging

> **Chapter guide:** Metrics must correspond to the product decision and the independent-sample unit. Rule out data leakage and pipeline bugs before discussing model capacity.


<a id="131-classification-metrics"></a>
## 12.1 Classification metrics

```
Precision = TP/(TP+FP)     of predicted positives, how many are right
Recall    = TP/(TP+FN)     of actual positives, how many did we catch
F1        = 2PR/(P+R)      harmonic mean — punishes a low value in either
```
- **AUC-ROC**: TPR vs FPR across thresholds. 0.5 random, 1.0 perfect. Fine when classes are roughly balanced.
- **AUC-PR**: better for **imbalanced** data — ROC-AUC looks deceptively high when negatives dominate, because FPR barely moves.
- **Calibration**: is a predicted 0.8 actually right 80% of the time? Measure with ECE / reliability diagrams. Matters whenever a downstream system thresholds your score — e.g. a brand-conflict check.

<a id="coding-classification-metrics"></a>

#### Coding: Confusion matrix / precision / recall / F1 / ROC-AUC `[READY · P1]`

- **Prompt:** compute metrics without sklearn and explain how the answer changes with threshold, class imbalance, and ties.
- **Shape / cost:** labels/scores `[N]`; confusion/P-R-F1 are `O(N)`. The teaching ROC implementation checks all unique thresholds and is up to `O(N²)` after sorting, rather than the optimal cumulative `O(N log N)` approach.
- **Pitfalls:** choose the positive label explicitly; define zero-denominator behavior; include both ROC endpoints; group equal scores rather than making tie order affect AUC; prefer PR-AUC for extreme imbalance.

```python
tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) \
    if precision + recall > 0 else 0.0
```

[Full implementation (NumPy, including ROC-AUC)](./ml_coding.py#L407) · [Torch confusion/P-R-F1](./ml_coding_torch.py#L498). PR-AUC and calibration/ECE are `[TODO]`.

<a id="132-regression--ranking-metrics-for-quality-models"></a>
## 12.2 Regression / ranking metrics (for quality models)

For anything predicting a continuous human score:
- **PLCC** (Pearson) after a monotonic logistic fit → accuracy of predicted value.
- **SRCC** (Spearman) → do you rank correctly?
- **KRCC** (Kendall) → rank, robust to ties.
- **RMSE / MAE** in the score's own units.

**SRCC vs PLCC** is a real decision: SRCC is what you want for "which encode is better"; PLCC for "does this clear our absolute quality bar." Report both. See the UGC-VQA drill in [04 §6](04_resume_drill.md#6-tiktok--ugc-short-video-quality-assessment-via-mllm).

<a id="coding-quality-metrics"></a>

#### Coding: PSNR / SSIM / temporal consistency `[READY + TOY · P1]`

- **Prompt:** implement PSNR from MSE, outline windowed SSIM, and warp adjacent frames with flow before measuring temporal error.
- **Shape / cost:** PSNR is linear in pixels; local global-SSIM approximation is also `O(HWC)`; bilinear warping is `O(THWC)` excluding flow estimation.
- **Pitfalls:** state `max_val`, color space, bit depth, crop, and aggregation convention. Global SSIM is not standard windowed SSIM. The Torch SSIM mixes unbiased variance with population covariance, and its temporal helper creates grid tensors on CPU and has a flow-normalization/sign contract that needs repair.

```python
def psnr(img1, img2, max_val=255.0):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)
```

[Full PSNR plus teaching versions of global SSIM/warp (NumPy)](./ml_coding.py#L269) · [Torch version](./ml_coding_torch.py#L378). PSNR is `[READY]`; SSIM and temporal warp are `[TOY]` until the boundaries above are fixed. PLCC/SRCC/KRCC implementations are `[TODO]`.

<a id="133-bias-variance"></a>
## 12.3 Bias-variance

```
Expected error = Bias² + Variance + Irreducible noise
```
- High bias, low variance → too simple, consistently wrong → more capacity, better features.
- Low bias, high variance → too complex, unstable → regularize, more data, less capacity.

Regularization trades a little bias for a lot of variance reduction — usually a net win.

<a id="134-debugging-playbook"></a>
## 12.4 Debugging playbook

**Training loss falls, validation is bad:**
1. Overfitting (model too large / data too small).
2. **Data leakage** — random frame split instead of scene/clip split for video; duplicate items across splits; a feature that encodes the label.
3. Train/val domain mismatch (different camera, compression level, scene type, device).
4. Augmentation applied at train but not val, or differently.
5. Metric computed inconsistently (PSNR on YUV vs displayed RGB; different crop/border handling).
6. Label or preprocessing inconsistency between the two paths.
7. Temporal leakage — future frames visible during training.

**Loss doesn't decrease at all:**
- LR too high (diverging) or too low (stuck); check by overfitting a single batch to ~zero loss — if you can't, the bug is in the model or data pipeline, not the optimizer.
- Labels misaligned/shuffled relative to inputs.
- Missing/incorrect normalization.
- Dead ReLUs / bad init.
- Wrong loss reduction (mean vs sum) interacting with LR.
- Gradients not flowing (a detach, a `no_grad`, a frozen module you meant to train).

**Good PSNR, bad visual quality**: L2 predicts the conditional mean → blur. PSNR is blind to sharpness, texture realism, and temporal stability. Fix with perceptual/adversarial/diffusion losses; always look at the images.

**Loss spikes / NaN mid-training**: LR too high, bad batch, fp16 overflow (switch to bf16), division by near-zero in a normalization, or exploding attention logits (add QK-norm or logit soft-capping). Standard mitigation: gradient clipping, skip batches with non-finite loss, checkpoint and resume from before the spike.

<a id="135-evaluating-generative-and-llm-outputs"></a>
## 12.5 Evaluating generative and LLM outputs

- **Automatic**: perplexity (only meaningful within a fixed tokenizer), task benchmarks, exact match, pass@k for code.
- **LLM-as-judge**: cheap and scalable, but biased toward length, position (first option), and its own style. Mitigate with pairwise comparison, position swapping, and a rubric. Always validate against human labels on a subset.
- **Human eval**: the ground truth. Same rigor as subjective video testing — multiple raters, inter-rater agreement (Krippendorff's α / Cohen's κ), confidence intervals.
- **Benchmark contamination**: assume any public benchmark is partly in the pretraining data. Prefer held-out, freshly collected, or private evaluation sets — which is exactly the argument for building your own benchmark, as you did.

<a id="coding-classic-ml"></a>

#### Coding: Classic ML + preprocessing `[TOY · P2]`

- **Prompt:** vectorize KNN, implement K-means with an empty-cluster rule, normalize from training statistics only, and shuffle/split reproducibly.
- **Shape / cost:** brute-force KNN is `O(MND)` for `M` queries and `N` training points; K-means is `O(iterations·N·K·D)`.
- **Pitfalls:** avoid leakage by fitting normalization on train only; video splits must be by scene/clip rather than adjacent frames; recompute final K-means labels if the iteration budget ends immediately after a centroid update.

```python
dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # [N, K]
labels = np.argmin(dists, axis=1)
new_centroids = np.array([
    X[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
    for i in range(k)
])
```

[Existing implementations and data utilities (NumPy)](./ml_coding.py#L358) · [Torch subset](./ml_coding_torch.py#L449). KNN and the basic utilities are useful interview baselines, but K-means can return labels from the previous centroids when `max_iter` expires immediately after an update, so the grouped entry remains `[TOY]` until fixed.

---

<a id="14-rapid-fire-drill--ml"></a>
# 13. Rapid-Fire Drill — ML

> **Chapter guide:** Hide the answer and respond aloud in 30–60 seconds. If you cannot answer, return to the relevant chapter instead of memorizing isolated sentences on this page.


**Fundamentals**
1. Why does gradient checkpointing save memory, and what does it cost?
2. Why He init for ReLU and Xavier for tanh?
3. Why does L2 loss blur images? What actually fixes it?
4. Write numerically stable log-softmax and explain each term.
5. Why label smoothing?

**Optimization / normalization**
6. Adam's update rule from memory; why bias correction?
7. Why does AdamW's decoupled decay matter?
8. Why do transformers need warmup?
9. BatchNorm vs LayerNorm vs GroupNorm vs RMSNorm — pick one for a batch-size-2 video model and justify.
10. Pre-norm vs post-norm.
11. FP16 vs BF16 vs FP8; when do you need loss scaling?

**CNN**
12. Why do two 3×3 convs beat one 5×5?
13. Why does depthwise separable conv save 8× FLOPs but not 8× latency?
14. Why do transposed convs produce checkerboards? Two alternatives.
15. Why avoid pooling in restoration networks?

**Transformer**
16. Why divide by √d_k?
17. MHA vs MQA vs GQA — what problem does GQA solve?
18. What is RoPE and why does it give relative position for free?
19. How would you extend a model's context beyond its trained length?
20. Is FlashAttention an approximation? Why is it faster?
21. Compute the KV cache size for a 7B model at 32k context.
22. Why is decode memory-bandwidth-bound and prefill compute-bound? Two consequences.
23. How does speculative decoding preserve the output distribution?

**Generative models & compression**
24. Why is $\mathbb{E}[-\log_2 p(\hat y)]$ the bitrate?
25. Three ways to backprop through quantization; what does each get wrong?
26. Explain the hyperprior. Why is spending 3% of bits on side information a win?
27. Why does an autoregressive context model hurt, and how do checkerboard/channel-wise contexts fix it?
28. DCVC's conditional coding vs residual coding — when are their conditional entropies equal, and where does the practical gain come from?
29. Hybrid vs VQ-based learned codecs: latent, entropy model, objective, sweet spot.
30. State the rate-distortion-perception tradeoff and give an application on each side.
31. Give five concrete reasons neural codecs aren't deployed on phones.
32. Why does floating-point non-determinism break a neural codec specifically?
33. How do you hit an exact bitrate target with a λ-trained model?
34. Causal vs non-causal tokenizer — why does streaming care?
35. Token dropping vs raising QP — how do they degrade differently?
36. Derive the ELBO; what is the gap between it and $\log p(x)$?
37. Why is the reparameterization trick necessary, and why does the KL term need no sample?
38. What is posterior collapse and how do you fix it?
39. Why is a β-VAE a rate–distortion objective? What plays the role of the rate?
40. Why is diffusion training more stable than GAN training?
41. What is classifier-free guidance and what does raising $w$ cost you?
42. Why latent diffusion instead of pixel diffusion?
43. Flow matching vs DDPM: what is the path, what is the target, why fewer sampling steps?
44. VQ-VAE's three loss terms and what each gradient reaches.
45. What is codebook collapse? How do you measure it, and name two quantizers that cannot collapse.
46. VQGAN's adaptive adversarial weight: write it down and say what it balances.
47. FSQ: why does $\tanh$ replace the commitment loss, and how are the digits packed into one index?
48. LFQ's entropy penalty: what do the two entropy terms push toward?
49. RVQ: why can you drop the last stages, and what does that cost in bits?
50. TiTok's 1-D tokens: what do you gain over a grid, and what does a wrong token do?
51. How does a learned prior take 10 bits per token down to 4 to 6, and what does it cost at decode time?

**VLM / LLM**
52. Draw the LLaVA architecture and describe both training stages.
53. Q-Former vs MLP projector — what do you trade?
54. How do modern VLMs handle high resolution, and what does it cost?
55. Why are video LLMs weak at motion? How would you fix that?
56. Why do VLMs hallucinate objects, and what reduces it?
57. Derive DPO's intuition: why is there no reward model?
58. GRPO vs PPO — what does the group replace, and why does that matter?
59. When would you pick RLVR over DPO?
60. What is reward hacking? Give a concrete example from your own setting.

**Efficiency**
61. Why is B initialized to zero in LoRA? What is α/r for?
62. QLoRA — what are NF4, double quantization, and paged optimizers for?
63. GPTQ vs AWQ — different core idea in each.
64. Why do LLM activations have outliers, and how does SmoothQuant handle them?
65. Unstructured pruning cut my parameters 50% and latency 0%. Why?
66. Distillation loss — why the T² factor?
67. Give three reasons a model is fast in the lab and slow on a phone.
68. How would you measure on-device latency honestly?

**Evaluation**
69. When is AUC-PR right and AUC-ROC misleading?
70. SRCC vs PLCC for a quality model — which do you optimize and why?
71. Your validation metric is great and production is bad. Walk the checklist.
72. Name three failure modes of LLM-as-judge and how you'd control each.

---

<a id="ml-coding-companion"></a>
<a id="15-coding-reverse-index--ml"></a>
# 14. Coding Reverse Index — ML

> **Chapter guide:** Use an interview prompt to find the corresponding implementation. Status labels describe the current boundary, not production readiness.


Status is scoped: **`[READY]`** means the linked implementation is a sound answer within the stated
shape; **`[PARTIAL]`** has reusable pieces; **`[TOY]`** has a documented correctness or
production boundary; **`[TODO]`** is only a checked contract. P0 should be writable cold, P1 after
one review, and P2 is lower-frequency.

| Theme | Inline drill | Status | Full local code |
|---|---|---|---|
| Backprop + activations | [Linear backward](#coding-linear-backward) · [activations](#coding-activations) | READY | [NumPy](./ml_coding.py) · [Torch](./ml_coding_torch.py) |
| Losses | [stable softmax/CE](#coding-stable-ce) · [focal/InfoNCE](#coding-loss-suite) | READY + TOY + PARTIAL | [NumPy](./ml_coding.py#L21) · [Torch](./ml_coding_torch.py#L30) |
| Optimization | [SGD/Adam/AdamW](#coding-optimizers) · [schedule/clipping](#coding-training-utils) | READY + TOY + TODO | [NumPy](./ml_coding.py#L123) · [Torch](./ml_coding_torch.py#L121) |
| Normalization | [BN/LN/RMSNorm](#coding-normalization) | READY + TODO | [NumPy](./ml_coding.py#L176) · [Torch](./ml_coding_torch.py#L191) |
| CNN | [2-D convolution](#coding-convolution) | TOY | [Scratch implementation](./interview_practice.py#L82) |
| Complete Encoder–Decoder | [Step-by-step math](#transformer-math) · [shape table](#transformer-shapes) · [PyTorch implementation](#pytorch-implementation) | TOY: complete one-layer model, untrained | [Runnable script](./transformer_encoder_decoder.py) · [Tests](./test_transformer_encoder_decoder.py) |
| Vision transformers | [ViT to DINOv3](#vision-transformers) · [attention cost](#vit-cost) · [model line](#vit-model-line) · [PyTorch implementation](#coding-vit) | TOY: one small module per mechanism, each checked against a direct implementation | [Runnable script](./vision_transformers.py) · [Tests](./test_vision_transformers.py) |
| Vision-language model | [Step-by-step math](#vlm-math) · [shape table](#vlm-shapes) · [PyTorch implementation](#coding-vlm) | TOY: ViT + projector + LLM, trained on two examples | [Runnable script](./vlm_vit_projector_llm.py) · [Tests](./test_vlm_vit_projector_llm.py) |
| Attention core | [scaled attention](#coding-scaled-attention) · [MHA](#coding-mha) · [causal mask](#coding-causal-mask) | Implemented and tested in the one-layer model | [Encoder–Decoder script](./transformer_encoder_decoder.py) · [NumPy attention](./ml_coding.py#L226) |
| Modern extensions | [RoPE](#coding-rope) · [pre-norm/SwiGLU block](#coding-transformer-block) · [KV cache](#coding-kv-cache) | RoPE: `rotate` implemented and tested. Pre-norm block: in the VLM script. SwiGLU: TODO. KV cache: READY in the inference script | [RoPE in the Encoder–Decoder script](./transformer_encoder_decoder.py) · [pre-norm block](./vlm_vit_projector_llm.py) · building blocks in [Torch](./ml_coding_torch.py#L220) |
| Generative models | [VAE](#coding-vae-vq) · [VQ](#coding-vq-ste) · [diffusion](#coding-diffusion) | READY (VAE, VQ) + TOY (diffusion) | inline in Chapter 6 |
| Learned compression | [RD forward](#641-the-core-idea) · [bpp from likelihoods](#643-entropy-models--where-the-gains-actually-come-from) · [closed-loop video](#644-two-paradigms) | TOY + READY | [learned_compression.py](./learned_compression.py) · [arithmetic_coding.py](./arithmetic_coding.py) · [video_encoding_pipeline.py](./video_encoding_pipeline.py) |
| LLM decoding | [temperature/top-k/top-p](#coding-llm-sampling) · [KV cache](#coding-kv-cache) | READY: `filtered_probabilities`, `prefill` / `decode_step`, tested on the toy model | [Runnable script](./inference_kv_cache.py) · [Tests](./test_inference_kv_cache.py) |
| Post-training | [LoRA](#coding-lora) | READY: `LoRALinear`, `add_lora`, merge, tested on the toy model | [Runnable script](./lora_linear.py) · [Tests](./test_lora_linear.py) · [older Torch version](./ml_coding_torch.py#L312) |
| Efficiency | [quantization/distillation](#coding-quant-distill) | TODO | No local implementation |
| Metrics | [classification](#coding-classification-metrics) · [quality/temporal](#coding-quality-metrics) | READY + TOY | [NumPy](./ml_coding.py#L269) · [Torch](./ml_coding_torch.py#L378) |
| Classic ML | [KNN/K-means/preprocessing](#coding-classic-ml) | TOY · P2 | [NumPy](./ml_coding.py#L358) · [Torch](./ml_coding_torch.py#L449) |

Check answers against the primary [Transformer paper](https://arxiv.org/abs/1706.03762),
[PyTorch SDPA docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[AdamW](https://arxiv.org/abs/1711.05101), [RMSNorm](https://arxiv.org/abs/1910.07467),
[RoPE](https://arxiv.org/abs/2104.09864), [GQA](https://arxiv.org/abs/2305.13245),
[nucleus sampling](https://arxiv.org/abs/1904.09751), and [LoRA](https://arxiv.org/abs/2106.09685).

---

*Continue to [04_resume_drill.md](04_resume_drill.md) for the project drills.*
