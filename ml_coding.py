"""
ML Coding Questions — Apple Video ML Engineer Interview Prep
============================================================
Topics:
  1. Loss Functions
  2. Activation Functions
  3. Optimizers
  4. Neural Network Components (BatchNorm, LayerNorm, Attention)
  5. Image / Video Quality Metrics
  6. Classic ML Algorithms
  7. Data Processing Utilities
"""

import numpy as np


# =============================================================================
# 1. LOSS FUNCTIONS
# =============================================================================

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error: mean over all elements of (pred - true)^2."""
    return np.mean((y_pred - y_true) ** 2)


def mae_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error: more robust to outliers than MSE."""
    return np.mean(np.abs(y_pred - y_true))


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray,
                         eps: float = 1e-9) -> float:
    """
    Binary cross-entropy for sigmoid output.
    y_pred: probabilities in (0, 1)
    y_true: binary labels {0, 1}
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Categorical cross-entropy from raw logits (applies softmax internally).
    logits: shape [N, C]
    labels: integer class indices, shape [N]
    """
    # numerically stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    print(f"shifted logits:\n{shifted}")
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    print(f"log_sum_exp:\n{log_sum_exp}")
    log_probs = shifted - log_sum_exp                           # [N, C]
    print(f"log_probs:\n{log_probs}")
    return -np.mean(log_probs[np.arange(len(labels)), labels])

## logits: NxC

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    log_sum_exp = np.log(np.sum(np.exp(logits), axis=1, keepdims=True))

    log_probs = logits - log_sum_exp                           # [N, C]
    cross_entropy = 0
    for i in range(len(labels)):
        cross_entropy += -log_probs[i, labels[i]]
    return cross_entropy / len(labels)


def rate_distortion_loss(distortion: float, rate: float, lam: float) -> float:
    """
    Standard learned compression objective.
    distortion: e.g. MSE between input and reconstruction
    rate:       estimated bits (entropy of latent)
    lam:        lagrange multiplier controlling bitrate-quality tradeoff
                larger lam → penalise bits more → lower bitrate
    """
    return distortion + lam * rate


# =============================================================================
# 2. ACTIVATION FUNCTIONS (forward + backward)
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    x: shape [N, C] or [C]
    """
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit — common in transformers.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


# =============================================================================
# 3. OPTIMIZERS
# =============================================================================

class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, params: dict, grads: dict):
        for name in params:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])
            self.velocity[name] = (self.momentum * self.velocity[name]
                                   - self.lr * grads[name])
            params[name] += self.velocity[name]


class Adam:
    """
    Adam optimiser: adaptive per-parameter learning rates.
    m = first moment (mean of gradients)
    v = second moment (mean of squared gradients)
    bias-corrected estimates used for update.
    """
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2

            # bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# 4. NEURAL NETWORK COMPONENTS
# =============================================================================

class BatchNorm1d:
    """
    BatchNorm for shape [N, C].
    Normalises each feature (column) across the batch.
    Maintains running stats for inference.
    """
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        # cache for backward
        self._cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: [N, C]"""
        if training:
            mean = x.mean(axis=0)           # [C]
            var = x.var(axis=0)             # [C]
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            self._cache = (x, x_hat, mean, var)
            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta


class LayerNorm:
    """
    LayerNorm: normalises each sample across its feature dimension.
    No batch dependency — same behaviour at train and inference.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [..., C] — normalises over last dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                  V: np.ndarray,
                                  mask: np.ndarray = None) -> np.ndarray:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    Q, K, V: shape [batch, seq_len, d_k] (or [seq_len, d_k] for single head)
    mask:    additive mask (e.g. -inf for padding positions)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)    # [batch, seq, seq]
    if mask is not None:
        scores += mask
    weights = softmax(scores)                           # [batch, seq, seq]
    return weights @ V                                  # [batch, seq, d_v]


class Linear:
    """Fully connected layer with optional bias."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # He initialisation
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features) if bias else None
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        out = x @ self.W
        if self.b is not None:
            out += self.b
        return out

    def backward(self, grad_out: np.ndarray):
        x = self._cache
        grad_W = x.T @ grad_out
        grad_b = grad_out.sum(axis=0) if self.b is not None else None
        grad_x = grad_out @ self.W.T
        return grad_x, grad_W, grad_b


# =============================================================================
# 5. IMAGE / VIDEO QUALITY METRICS
# =============================================================================

def psnr(img1: np.ndarray, img2: np.ndarray,
         max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher = better quality. Typical range: 30–50 dB for video.
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def ssim(img1: np.ndarray, img2: np.ndarray,
         max_val: float = 255.0) -> float:
    """
    Structural Similarity Index (scalar approximation, no sliding window).
    Compares luminance, contrast, and structure.
    Range: [-1, 1], 1 = identical.

    Full SSIM uses local patches; this is the global version for simplicity.
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator   = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return numerator / denominator


def temporal_warping_error(frames_out: list, optical_flows: list) -> float:
    """
    Measures temporal consistency: how much the output changes relative
    to what optical flow predicts between consecutive frames.

    warping_error_t = || output_t - warp(output_{t-1}, flow_{t-1->t}) ||^2

    frames_out:   list of output frames [T, H, W, C]
    optical_flows: list of flows [T-1, H, W, 2] (flow[t] maps frame t to t+1)

    Note: real warp uses bilinear sampling. Here we use a simplified version
    for illustration; in practice use cv2.remap or torch grid_sample.
    """
    from scipy.ndimage import map_coordinates

    total_error = 0.0
    for t in range(1, len(frames_out)):
        prev = frames_out[t - 1].astype(float)
        curr = frames_out[t].astype(float)
        flow = optical_flows[t - 1]             # [H, W, 2]
        H, W = prev.shape[:2]

        grid_y, grid_x = np.mgrid[0:H, 0:W]
        # flow gives displacement: new_x = x + flow_x
        sample_y = grid_y - flow[..., 1]
        sample_x = grid_x - flow[..., 0]

        warped = np.stack([
            map_coordinates(prev[..., c], [sample_y, sample_x], order=1)
            for c in range(prev.shape[2])
        ], axis=-1)

        total_error += np.mean((curr - warped) ** 2)

    return total_error / (len(frames_out) - 1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Range: [-1, 1]. 1 = same direction, 0 = orthogonal.
    Used in embedding comparison, feature matching.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# 6. CLASSIC ML ALGORITHMS
# =============================================================================

def knn_predict(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, k: int = 3) -> np.ndarray:
    """
    K-Nearest Neighbours (brute force, L2 distance).
    X_train: [N, D], y_train: [N], X_test: [M, D]
    Returns predicted labels [M].
    """
    predictions = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)    # [N]
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        # majority vote
        counts = np.bincount(nn_labels.astype(int))
        predictions.append(np.argmax(counts))
    return np.array(predictions)


def kmeans(X: np.ndarray, k: int, max_iter: int = 100,
           seed: int = 0) -> tuple:
    """
    K-Means clustering.
    Returns (centroids [k, D], labels [N]).
    """
    rng = np.random.default_rng(seed)
    # random initialisation
    centroids = X[rng.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        # assign each point to nearest centroid
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # [N, k]
        labels = np.argmin(dists, axis=1)                               # [N]

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# =============================================================================
# 7. CLASSIFICATION METRICS
# =============================================================================

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Returns [num_classes, num_classes] confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                         pos_label: int = 1) -> tuple:
    """
    Binary precision, recall, F1.
    precision = TP / (TP + FP)  — of all predicted positive, how many correct
    recall    = TP / (TP + FN)  — of all actual positive, how many caught
    F1        = harmonic mean of precision and recall
    """
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute ROC curve (fpr, tpr) and AUC via trapezoidal rule.
    y_scores: predicted probability for the positive class.
    """
    thresholds = np.sort(np.unique(y_scores))[::-1]
    fprs, tprs = [0.0], [0.0]

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    for thresh in thresholds:
        pred = (y_scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fprs.append(fp / total_neg if total_neg > 0 else 0)
        tprs.append(tp / total_pos if total_pos > 0 else 0)

    fprs.append(1.0)
    tprs.append(1.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    auc = np.trapz(tprs, fprs)
    return fprs, tprs, auc


# =============================================================================
# 8. DATA PROCESSING UTILITIES
# =============================================================================

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] per feature."""
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1          # avoid divide-by-zero for constant features
    return (x - x_min) / denom


def z_score_normalize(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per feature."""
    mean = x.mean(axis=0)
    std  = x.std(axis=0)
    std[std == 0] = 1
    return (x - mean) / std


def shuffle_dataset(X: np.ndarray, y: np.ndarray,
                    seed: int = 0) -> tuple:
    """Shuffle X and y in unison without built-in shuffle."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_val_split(X: np.ndarray, y: np.ndarray,
                    val_ratio: float = 0.2, seed: int = 0) -> tuple:
    """
    Split dataset into train and validation sets.
    For video data: split by clip/scene index, not by random frame,
    to avoid temporal leakage between adjacent frames.
    """
    X, y = shuffle_dataset(X, y, seed)
    n_val = int(len(X) * val_ratio)
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


# =============================================================================
# QUICK SMOKE TESTS
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- loss ---
    pred = rng.random((4, 3))
    print(f"pred:\n{pred}")
    labels = np.array([0, 1, 2, 1])
    print(f"labels: {labels}")
    print(f"cross_entropy:  {cross_entropy_loss(pred, labels):.4f}")

    #print(f"rate_distortion: {rate_distortion_loss(0.02, 1.5, lam=0.01):.4f}")

    # --- softmax / attention ---
    x = rng.standard_normal((2, 4))
    print(f"softmax sums to 1: {softmax(x).sum(axis=1)}")

    Q = rng.standard_normal((2, 5, 8))
    K = rng.standard_normal((2, 5, 8))
    V = rng.standard_normal((2, 5, 16))
    out = scaled_dot_product_attention(Q, K, V)
    print(f"attention output shape: {out.shape}")   # (2, 5, 16)

    # --- norms ---
    x_2d = rng.standard_normal((8, 16))
    bn = BatchNorm1d(16)
    ln = LayerNorm(16)
    print(f"BN output mean≈0: {bn.forward(x_2d).mean():.4f}")
    print(f"LN output mean≈0: {ln.forward(x_2d).mean():.4f}")

    # --- image metrics ---
    img1 = rng.integers(0, 256, (64, 64, 3)).astype(float)
    img2 = img1 + rng.standard_normal((64, 64, 3)) * 10
    print(f"PSNR: {psnr(img1, img2):.2f} dB")
    print(f"SSIM: {ssim(img1, img2):.4f}")

    # --- classification metrics ---
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    p, r, f = precision_recall_f1(y_true, y_pred)
    print(f"Precision={p:.2f}  Recall={r:.2f}  F1={f:.2f}")

    # --- adam ---
    params = {"W": rng.standard_normal((4, 4))}
    grads  = {"W": rng.standard_normal((4, 4))}
    opt = Adam()
    opt.step(params, grads)
    print(f"Adam step done, W norm: {np.linalg.norm(params['W']):.4f}")
