"""
ML Coding Questions — PyTorch Version
Apple Video ML Engineer Interview Prep
======================================
Same structure as ml_coding.py but using PyTorch:
  - nn.Module for all layers
  - autograd-compatible (backward works out of the box)
  - Custom autograd Functions where instructive

  1. Loss Functions
  2. Activation Functions
  3. Optimizers (manual step loop)
  4. Neural Network Components
  5. Image / Video Quality Metrics
  6. Classic ML Algorithms
  7. Data Processing Utilities
  8. LoRA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. LOSS FUNCTIONS
# =============================================================================

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()


def binary_cross_entropy(pred: Tensor, target: Tensor,
                         eps: float = 1e-9) -> Tensor:
    """pred: probabilities in (0,1), target: {0,1}"""
    pred = pred.clamp(eps, 1 - eps)
    return -(target * pred.log() + (1 - target) * (1 - pred).log()).mean()


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """
    From raw logits — applies log-softmax internally (numerically stable).
    logits: [N, C],  labels: [N] integer class indices.
    """
    # log-sum-exp trick for numerical stability
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_sum_exp = shifted.exp().sum(dim=1, keepdim=True).log()
    log_probs = shifted - log_sum_exp                       # [N, C]
    return -log_probs[torch.arange(len(labels)), labels].mean()


def rate_distortion_loss(distortion: Tensor, rate: Tensor,
                         lam: float) -> Tensor:
    """
    Standard learned compression objective.
    distortion: reconstruction error (e.g. MSE)
    rate:       estimated bits (entropy of quantised latent)
    lam:        larger → penalise bits more → lower bitrate
    """
    return distortion + lam * rate


# =============================================================================
# 2. ACTIVATION FUNCTIONS  (as autograd-compatible Functions)
# =============================================================================

class SigmoidFunction(torch.autograd.Function):
    """Sigmoid with manual backward — useful to show in interviews."""
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        s = 1 / (1 + (-x).exp())
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (s,) = ctx.saved_tensors
        return grad_out * s * (1 - s)


sigmoid = SigmoidFunction.apply


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (x,) = ctx.saved_tensors
        return grad_out * (x > 0).float()


relu = ReLUFunction.apply


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax."""
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit — standard in transformers."""
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5
                                      * (x + 0.044715 * x ** 3)))


# =============================================================================
# 3. OPTIMIZERS  (manual implementations — no torch.optim)
# =============================================================================

class SGD:
    def __init__(self, params: list, lr: float = 0.01,
                 momentum: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            v.mul_(self.momentum).add_(p.grad, alpha=-self.lr)
            p.add_(v)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class Adam:
    """
    Adam: adaptive per-parameter learning rates.
    m = first moment (gradient mean)
    v = second moment (gradient variance)
    Bias-corrected estimates are used for the update.
    """
    def __init__(self, params: list, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p          # AdamW-style decoupling

            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# =============================================================================
# 4. NEURAL NETWORK COMPONENTS
# =============================================================================

class BatchNorm1d(nn.Module):
    """
    BatchNorm for [N, C]: normalises each feature across the batch.
    Maintains running mean/var for inference (no batch dependency at test time).
    """
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        """x: [N, C]"""
        if self.training:
            mean = x.mean(dim=0)
            var  = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_hat + self.beta


class LayerNorm(nn.Module):
    """
    LayerNorm: normalises each sample across its feature dimension.
    No batch dependency — identical behaviour at train and inference.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        """x: [..., C] — normalises over last dimension."""
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_hat + self.beta


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor,
                mask: Tensor = None) -> Tensor:
        """
        Q, K: [..., seq, d_k]
        V:    [..., seq, d_v]
        mask: additive mask (e.g. -inf for padding)
        """
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / d_k ** 0.5  # [..., seq, seq]
        if mask is not None:
            scores = scores + mask
        weights = self.drop(softmax(scores, dim=-1))
        return weights @ V                               # [..., seq, d_v]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self/cross attention.
    Splits d_model into h heads of size d_k = d_model // h.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, S, d_model] → [B, h, S, d_k]"""
        B, S, _ = x.shape
        return x.view(B, S, self.h, self.d_k).transpose(1, 2)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor,
                mask: Tensor = None) -> Tensor:
        B = Q.size(0)
        Q = self._split_heads(self.W_q(Q))
        K = self._split_heads(self.W_k(K))
        V = self._split_heads(self.W_v(V))
        out = self.attn(Q, K, V, mask)                  # [B, h, S, d_k]
        out = out.transpose(1, 2).reshape(B, -1, self.h * self.d_k)
        return self.W_o(out)


class Linear(nn.Module):
    """Fully connected layer with He initialisation."""
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


# =============================================================================
# 5. LoRA
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA wrapper around a frozen Linear layer.

    Forward:  h = W·x + (B·A)·x  =  (W + B·A)·x
    Only A and B are trained; W is frozen.

    Merge at inference:  call .merge() to fold B·A into W,
    making the layer identical in cost to a plain Linear.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 4, alpha: float = 1.0, bias: bool = True):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank               # ΔW is multiplied by scale

        # frozen base weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2 / in_features) ** 0.5,
            requires_grad=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # trainable low-rank matrices
        # A: Gaussian init, B: zero init → ΔW = B·A = 0 at start
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        self._merged = False

    def forward(self, x: Tensor) -> Tensor:
        if self._merged:
            return F.linear(x, self.weight, self.bias)
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.A), self.B) * self.scale
        return base + lora

    def merge(self):
        """
        Fold ΔW = scale * B·A into W (in-place).
        After merging: zero inference overhead, identical to plain Linear.
        """
        if self._merged:
            return
        with torch.no_grad():
            self.weight.add_(self.scale * self.B @ self.A)
        self._merged = True

    def unmerge(self):
        if not self._merged:
            return
        with torch.no_grad():
            self.weight.sub_(self.scale * self.B @ self.A)
        self._merged = False

    def trainable_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def total_params(self) -> int:
        return self.weight.numel() + self.trainable_params()


# =============================================================================
# 6. IMAGE / VIDEO QUALITY METRICS
# =============================================================================

def psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> Tensor:
    """
    Peak Signal-to-Noise Ratio (dB). Higher = better.
    pred, target: any shape, values in [0, max_val].
    """
    mse = ((pred - target) ** 2).mean()
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)


def ssim(pred: Tensor, target: Tensor, max_val: float = 1.0) -> Tensor:
    """
    Global SSIM (no sliding window — for illustration).
    pred, target: [N, C, H, W] or flattened.
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1, mu2 = pred.mean(), target.mean()
    sigma1_sq = pred.var()
    sigma2_sq = target.var()
    sigma12   = ((pred - mu1) * (target - mu2)).mean()

    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return num / den


def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Cosine similarity between two vectors (1D tensors)."""
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def temporal_warping_error(frames: Tensor, flows: Tensor) -> Tensor:
    """
    Temporal consistency metric.
    frames: [T, C, H, W]
    flows:  [T-1, 2, H, W]  (flow[t] warps frame t to t+1)
    Returns mean squared warping error across time.
    """
    T, C, H, W = frames.shape
    total = torch.tensor(0.0)

    for t in range(T - 1):
        flow = flows[t]                                   # [2, H, W]
        prev = frames[t].unsqueeze(0)                     # [1, C, H, W]
        curr = frames[t + 1]

        # build sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij',
        )
        # normalise flow to [-1, 1] range
        dx = flow[0] / (W / 2)
        dy = flow[1] / (H / 2)
        grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)

        warped = F.grid_sample(prev, grid, align_corners=True,
                               mode='bilinear', padding_mode='border')
        total = total + ((curr - warped.squeeze(0)) ** 2).mean()

    return total / (T - 1)


# =============================================================================
# 7. CLASSIC ML ALGORITHMS
# =============================================================================

def knn_predict(X_train: Tensor, y_train: Tensor,
                X_test: Tensor, k: int = 3) -> Tensor:
    """
    K-Nearest Neighbours (brute-force L2).
    X_train: [N, D],  y_train: [N],  X_test: [M, D]
    Returns predicted labels [M].
    """
    # pairwise L2: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
    a_sq = (X_test ** 2).sum(dim=1, keepdim=True)           # [M, 1]
    b_sq = (X_train ** 2).sum(dim=1, keepdim=True).T        # [1, N]
    dists = (a_sq + b_sq - 2 * X_test @ X_train.T).clamp(min=0).sqrt()

    nn_idx  = dists.topk(k, dim=1, largest=False).indices   # [M, k]
    nn_lbls = y_train[nn_idx]                               # [M, k]
    # majority vote
    predictions = nn_lbls.mode(dim=1).values
    return predictions


def kmeans(X: Tensor, k: int, max_iter: int = 100) -> tuple:
    """
    K-Means clustering.
    Returns (centroids [k, D], labels [N]).
    """
    idx = torch.randperm(len(X))[:k]
    centroids = X[idx].clone()

    for _ in range(max_iter):
        # [N, k] pairwise distances
        dists = torch.cdist(X, centroids)
        labels = dists.argmin(dim=1)                        # [N]

        new_centroids = torch.stack([
            X[labels == i].mean(dim=0) if (labels == i).any()
            else centroids[i]
            for i in range(k)
        ])

        if torch.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# =============================================================================
# 8. CLASSIFICATION METRICS
# =============================================================================

def confusion_matrix(y_true: Tensor, y_pred: Tensor,
                     num_classes: int) -> Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_true: Tensor, y_pred: Tensor,
                         pos_label: int = 1) -> tuple:
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum().float()
    fp = ((y_pred == pos_label) & (y_true != pos_label)).sum().float()
    fn = ((y_pred != pos_label) & (y_true == pos_label)).sum().float()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# =============================================================================
# 9. DATA PROCESSING UTILITIES
# =============================================================================

def min_max_normalize(x: Tensor) -> Tensor:
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    denom = (x_max - x_min).clamp(min=1e-9)
    return (x - x_min) / denom


def z_score_normalize(x: Tensor) -> Tensor:
    mean = x.mean(dim=0)
    std  = x.std(dim=0).clamp(min=1e-9)
    return (x - mean) / std


def shuffle_dataset(X: Tensor, y: Tensor) -> tuple:
    idx = torch.randperm(len(X))
    return X[idx], y[idx]


# =============================================================================
# SMOKE TESTS
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # --- losses ---
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])
    print(f"cross_entropy:       {cross_entropy_loss(logits, labels):.4f}")
    print(f"torch CE (ref):      {F.cross_entropy(logits, labels):.4f}")

    d = torch.tensor(0.02)
    r = torch.tensor(1.5)
    print(f"rate_distortion:     {rate_distortion_loss(d, r, lam=0.01):.4f}")

    # --- activations ---
    x = torch.randn(3, requires_grad=True)
    s = sigmoid(x)
    s.sum().backward()
    print(f"sigmoid grad ok:     {x.grad is not None}")

    # --- attention ---
    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 5, 8)
    V = torch.randn(2, 5, 16)
    attn = ScaledDotProductAttention()
    out = attn(Q, K, V)
    print(f"attention shape:     {tuple(out.shape)}")   # (2, 5, 16)

    # --- MHA ---
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    x_seq = torch.randn(2, 10, 32)
    print(f"MHA output shape:    {tuple(mha(x_seq, x_seq, x_seq).shape)}")

    # --- norms ---
    x_2d = torch.randn(8, 16)
    bn = BatchNorm1d(16)
    ln = LayerNorm(16)
    print(f"BN output mean≈0:    {bn(x_2d).mean().item():.4f}")
    print(f"LN output mean≈0:    {ln(x_2d).mean().item():.4f}")

    # --- LoRA ---
    lora_layer = LoRALinear(64, 128, rank=4, alpha=4.0)
    x_in = torch.randn(8, 64)
    out_before = lora_layer(x_in).detach().clone()

    trainable = lora_layer.trainable_params()
    total     = lora_layer.total_params()
    print(f"LoRA trainable:      {trainable} / {total} params "
          f"({100*trainable/total:.1f}%)")

    lora_layer.merge()
    out_after = lora_layer(x_in)
    print(f"LoRA merge correct:  {torch.allclose(out_before, out_after, atol=1e-6)}")

    # --- image metrics ---
    img1 = torch.rand(1, 3, 64, 64)
    img2 = img1 + torch.randn_like(img1) * 0.1
    print(f"PSNR:                {psnr(img1, img2):.2f} dB")
    print(f"SSIM:                {ssim(img1, img2):.4f}")

    # --- optimizers ---
    model = nn.Linear(4, 2)
    opt = Adam(list(model.parameters()), lr=1e-3)
    loss = model(torch.randn(8, 4)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Adam step ok:        grad zeroed = {model.weight.grad is None or model.weight.grad.abs().max() == 0}")

    # --- KNN / KMeans ---
    X = torch.randn(50, 4)
    y = torch.randint(0, 3, (50,))
    preds = knn_predict(X, y, X[:5], k=3)
    print(f"KNN preds:           {preds.tolist()}")

    centroids, lbls = kmeans(X, k=3)
    print(f"KMeans centroids:    {tuple(centroids.shape)}")

    # --- classification metrics ---
    y_true = torch.tensor([0, 1, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 0, 1])
    p, r, f = precision_recall_f1(y_true, y_pred)
    print(f"Precision={p:.2f}  Recall={r:.2f}  F1={f:.2f}")
