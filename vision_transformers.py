"""Vision transformers: the mechanisms of Chapter 2, "Vision Transformers, from ViT to today".

One small module per idea, each written to match its equation in the book: ViT and DeiT
(class, distillation and register tokens), Swin (windows, shifted windows, relative position
bias, patch merging), PVT (spatial-reduction attention), MAE (random masking), DINO
(self-distillation loss, EMA teacher), TimeSformer and ViViT (tubelets, divided space-time
attention), NaViT (packed sequences) and DiT (adaLN-Zero).
Run:   .venv-transformer/bin/python vision_transformers.py
Tests: .venv-transformer/bin/python -m unittest -v test_vision_transformers

Tokens are batch-first, [B, N, D]; images are channel-first, [B, C, H, W]. Masks are boolean
and True means "this key may be read", as in Chapter 2.
"""

# BEGIN vit-imports
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vlm_vit_projector_llm import patchify  # Chapter 3, step 1: [B, C, H, W] -> [B, N, P*P*C]
# END vit-imports


# BEGIN vit-attention
class Attention(nn.Module):
    """head_i = softmax(Q_i K_i^T / sqrt(d_k) + bias, masked) V_i;  out = [head_1 || ... || head_h] W_O.

    Chapter 2's attention plus an additive bias (Swin's relative positions). The queries come
    from x; keys and values from `context`, which is x itself unless given (PVT's pooled keys).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h, self.d_k = heads, dim // heads
        self.q = nn.Linear(dim, dim)              # W_Q for all heads
        self.kv = nn.Linear(dim, 2 * dim)         # W_K and W_V side by side
        self.out = nn.Linear(dim, dim)            # W_O

    def forward(self, x: Tensor, context: Tensor = None, bias: Tensor = None, mask: Tensor = None) -> Tensor:
        context = x if context is None else context
        B, Lq, D = x.shape
        Lk = context.shape[1]
        q = self.q(x).view(B, Lq, self.h, self.d_k).transpose(1, 2)                   # [B, h, Lq, d_k]
        k, v = self.kv(context).view(B, Lk, 2, self.h, self.d_k).permute(2, 0, 3, 1, 4)  # 2 x [B, h, Lk, d_k]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)                         # [B, h, Lq, Lk]
        if bias is not None:
            scores = scores + bias                                                     # broadcasts over B
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        heads = scores.softmax(dim=-1) @ v                                             # [B, h, Lq, d_k]
        return self.out(heads.transpose(1, 2).reshape(B, Lq, D))


def mlp(dim: int, ratio: int = 4) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Linear(ratio * dim, dim))


class Block(nn.Module):
    """Pre-norm encoder block (Chapter 3, step 3):  x' = x + MSA(LN(x));  x = x' + MLP(LN(x'))."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim), mlp(dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.mlp(self.norm2(x))
# END vit-attention


# BEGIN vit-classifier
class PatchEmbed(nn.Module):
    """x_p^i E for every patch at once. Conv2d with kernel = stride = P equals the Linear layer
    applied to the flattened patches of patchify(); the output is in row-major patch order."""

    def __init__(self, patch: int, channels: int, dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)

    def forward(self, images: Tensor) -> Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)          # [B, C, H, W] -> [B, N, D]


def resize_positions(positions: Tensor, lead: int, grid: tuple) -> Tensor:
    """Another resolution at fine-tuning: 2D interpolation of the patch rows of E_pos. Bicubic,
    as in DeiT, which notes that bilinear shrinks the vectors; the ViT code interpolates linearly.

    positions: [1, lead + n*n, D]. The `lead` rows (class and distillation tokens) are kept.
    """
    head, table = positions[:, :lead], positions[:, lead:]
    side = math.isqrt(table.shape[1])
    if (side, side) == tuple(grid):
        return positions
    D = table.shape[-1]
    table = table.reshape(1, side, side, D).permute(0, 3, 1, 2)                  # [1, D, n, n]
    table = F.interpolate(table, size=tuple(grid), mode="bicubic", align_corners=False)
    return torch.cat([head, table.permute(0, 2, 3, 1).reshape(1, -1, D)], dim=1)


class ViT(nn.Module):
    """z0 = [x_cls; (x_dist;) x_p^1 E; ...; x_p^N E] + E_pos  ->  L blocks  ->  LN(z_L^0) W_head.

    distill=True is DeiT: a second learned token with its own head, trained on the teacher's
    labels. registers=r adds r learned tokens without positions that join attention and are
    dropped at the output (Vision Transformers Need Registers).
    """

    def __init__(self, image_size: int = 224, patch: int = 16, channels: int = 3, dim: int = 768,
                 depth: int = 12, heads: int = 12, classes: int = 1000, distill: bool = False,
                 registers: int = 0):
        super().__init__()
        self.embed = PatchEmbed(patch, channels, dim)
        self.lead = 1 + int(distill)                                              # tokens before the patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))                           # x_cls
        self.dist = nn.Parameter(torch.zeros(1, 1, dim)) if distill else None     # x_dist (DeiT)
        self.registers = nn.Parameter(torch.zeros(1, registers, dim)) if registers else None
        grid = image_size // patch
        self.positions = nn.Parameter(0.02 * torch.randn(1, self.lead + grid * grid, dim))  # E_pos
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)                                       # W_head
        self.head_dist = nn.Linear(dim, classes) if distill else None

    def forward(self, images: Tensor):
        B, _, H, W = images.shape
        P = self.embed.patch
        x = self.embed(images)                                                    # [B, N, D]
        lead = [self.cls.expand(B, -1, -1)]
        if self.dist is not None:
            lead.append(self.dist.expand(B, -1, -1))
        x = torch.cat(lead + [x], dim=1) + resize_positions(self.positions, self.lead, (H // P, W // P))
        if self.registers is not None:
            r = self.registers.expand(B, -1, -1)
            x = torch.cat([x[:, :self.lead], r, x[:, self.lead:]], dim=1)
        for block in self.blocks:
            x = block(x)                                                          # z_1 ... z_L
        x = self.norm(x)
        if self.head_dist is None:
            return self.head(x[:, 0])                                             # [B, K] from z_L^0
        return self.head(x[:, 0]), self.head_dist(x[:, 1])                        # class and distillation heads


def deit_loss(logits_cls: Tensor, logits_dist: Tensor, labels: Tensor, teacher_logits: Tensor) -> Tensor:
    """Hard distillation:  L = 1/2 CE(psi(Z_cls), y) + 1/2 CE(psi(Z_dist), y_t),  y_t = argmax_c Z_t(c)."""
    y_t = teacher_logits.argmax(dim=-1)
    return 0.5 * F.cross_entropy(logits_cls, labels) + 0.5 * F.cross_entropy(logits_dist, y_t)


def deit_predict(model: ViT, images: Tensor) -> Tensor:
    """At test time the paper adds the softmax outputs of the two heads (here: their mean, which
    ranks the classes the same way); the official code averages the logits instead."""
    logits_cls, logits_dist = model(images)
    return (logits_cls.softmax(dim=-1) + logits_dist.softmax(dim=-1)) / 2
# END vit-classifier


# BEGIN vit-swin
def window_partition(x: Tensor, M: int) -> Tensor:
    """[B, H, W, C] -> [B * (H/M) * (W/M), M*M, C]: each M x M window becomes one short sequence."""
    B, H, W, C = x.shape
    x = x.view(B, H // M, M, W // M, M, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, M * M, C)


def window_reverse(windows: Tensor, M: int, H: int, W: int) -> Tensor:
    """Inverse of window_partition: [B * (H/M) * (W/M), M*M, C] -> [B, H, W, C]."""
    C = windows.shape[-1]
    x = windows.view(-1, H // M, W // M, M, M, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, H, W, C)


def relative_position_index(M: int) -> Tensor:
    """[M*M, M*M] indices into the flattened (2M-1) x (2M-1) table:
    B_ij = B_hat[y_i - y_j + M - 1, x_i - x_j + M - 1]."""
    ys, xs = torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij")
    y, x = ys.flatten(), xs.flatten()                                  # row-major order inside a window
    dy = y[:, None] - y[None, :] + M - 1                               # 0 ... 2M-2
    dx = x[:, None] - x[None, :] + M - 1
    return dy * (2 * M - 1) + dx


def shifted_window_mask(H: int, W: int, M: int, s: int) -> Tensor:
    """[windows, M*M, M*M], True = may read. After the roll by -s, a window can hold pieces of
    up to four regions that were not neighbours in the image; only same-region pairs attend.
    The official code adds -100 to the other pairs (the book's mask); Attention here uses -inf,
    which gives exact zeros after the softmax."""
    region = torch.zeros(1, H, W, 1, dtype=torch.long)
    label = 0
    for rows in (slice(0, -M), slice(-M, -s), slice(-s, None)):
        for cols in (slice(0, -M), slice(-M, -s), slice(-s, None)):
            region[:, rows, cols] = label
            label += 1
    r = window_partition(region, M).squeeze(-1)                         # [windows, M*M]
    return r[:, :, None] == r[:, None, :]


class SwinBlock(nn.Module):
    """z_hat = (S)W-MSA(LN(z)) + z;  z = MLP(LN(z_hat)) + z_hat.  shift = 0 is W-MSA, M // 2 is SW-MSA."""

    def __init__(self, dim: int, heads: int, M: int = 7, shift: int = 0):
        super().__init__()
        self.M, self.shift = M, shift
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.table = nn.Parameter(0.02 * torch.randn(heads, (2 * M - 1) ** 2))     # B_hat, one per head
        self.register_buffer("index", relative_position_index(M), persistent=False)
        self.norm2, self.mlp = nn.LayerNorm(dim), mlp(dim)

    def mix(self, h: Tensor, H: int, W: int) -> Tensor:
        """The attention sublayer on a normalized map h: [B, H*W, C] -> [B, H*W, C]."""
        B, N, C = h.shape
        M = self.M
        s = self.shift if min(H, W) > M else 0                         # one window covers the map: no shift
        h = h.view(B, H, W, C)
        if s:
            h = torch.roll(h, shifts=(-s, -s), dims=(1, 2))            # cyclic shift up and left
        windows = window_partition(h, M)                               # [B * windows, M*M, C]
        bias = self.table[:, self.index]                               # B: [heads, M*M, M*M]
        mask = None
        if s:
            mask = shifted_window_mask(H, W, M, s).to(h.device).repeat(B, 1, 1)[:, None]
        h = window_reverse(self.attn(windows, bias=bias, mask=mask), M, H, W)
        if s:
            h = torch.roll(h, shifts=(s, s), dims=(1, 2))              # undo the shift
        return h.reshape(B, N, C)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = x + self.mix(self.norm1(x), H, W)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """x'_{u,v} = LN([x_{2u,2v}; x_{2u+1,2v}; x_{2u,2v+1}; x_{2u+1,2v+1}]) W_merge:
    [B, H*W, C] -> [B, (H/2)*(W/2), 2C]. Half the grid, twice the width."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)       # W_merge

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x.reshape(B, -1, 4 * C)))


class Swin(nn.Module):
    """4 x 4 patches -> C, then four stages of (W-MSA, SW-MSA) block pairs with merging in between.
    Returns the four feature maps, at strides 4, 8, 16 and 32 like a ResNet."""

    def __init__(self, channels: int = 3, dim: int = 96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                 M: int = 7, patch: int = 4):
        super().__init__()
        self.embed = PatchEmbed(patch, channels, dim)                  # 4*4*3 = 48 numbers -> C
        self.stages = nn.ModuleList(
            nn.ModuleList(SwinBlock(dim * 2 ** i, h, M, shift=0 if j % 2 == 0 else M // 2) for j in range(n))
            for i, (n, h) in enumerate(zip(depths, heads)))
        self.merges = nn.ModuleList(PatchMerging(dim * 2 ** i) for i in range(len(depths) - 1))

    def forward(self, images: Tensor) -> list:
        x = self.embed(images)
        H, W = images.shape[2] // self.embed.patch, images.shape[3] // self.embed.patch
        features = []
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, H, W)
            features.append(x)                                         # [B, H*W, C * 2^i]
            if i < len(self.merges):
                x, H, W = self.merges[i](x, H, W), H // 2, W // 2
        return features
# END vit-swin


# BEGIN vit-sra
class SpatialReductionAttention(nn.Module):
    """PVT:  SRA(Q, K, V) = Attn(Q W_Q, SR(K) W_K, SR(V) W_V),  SR(x) = LN(Reshape(x, R) W_S).

    Reshape(x, R) turns every R x R block of tokens into one row of R*R*C numbers and W_S maps it
    back to C, so N queries read N / R^2 keys. Conv2d(kernel = stride = R) is exactly that.
    """

    def __init__(self, dim: int, heads: int, R: int):
        super().__init__()
        self.R = R
        self.attn = Attention(dim, heads)
        if R > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=R, stride=R)     # Reshape(x, R) W_S
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        if self.R == 1:
            return self.attn(x)
        B, N, C = x.shape
        reduced = self.sr(x.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2)  # [B, N/R^2, C]
        return self.attn(x, context=self.norm(reduced))
# END vit-sra


# BEGIN vit-mae
def random_masking(x: Tensor, ratio: float, generator: torch.Generator = None):
    """Keep a random (1 - ratio) of the N tokens of every image, sampled without replacement.

    Returns the kept tokens [B, N_keep, D], the mask [B, N] in image order (1 = removed) and
    ids_restore [B, N], the permutation that puts a shuffled sequence back in image order.
    """
    B, N, D = x.shape
    keep = int(N * (1 - ratio))
    noise = torch.rand(B, N, generator=generator).to(x.device)
    ids_shuffle = noise.argsort(dim=1)                                 # the smallest noise is kept
    ids_restore = ids_shuffle.argsort(dim=1)
    kept = torch.gather(x, 1, ids_shuffle[:, :keep, None].expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :keep] = 0
    return kept, torch.gather(mask, 1, ids_restore), ids_restore


def unshuffle(h: Tensor, mask_token: Tensor, ids_restore: Tensor) -> Tensor:
    """Decoder input: the encoded visible tokens plus the shared mask token m for every removed
    patch, back in image order. [B, N_keep, D] -> [B, N, D]."""
    B, keep, D = h.shape
    N = ids_restore.shape[1]
    full = torch.cat([h, mask_token.expand(B, N - keep, D)], dim=1)   # still in shuffled order
    return torch.gather(full, 1, ids_restore[:, :, None].expand(-1, -1, D))


def mae_loss(pred: Tensor, patches: Tensor, mask: Tensor) -> Tensor:
    """L = mean over removed patches j of ||x_hat_j - norm(x_j)||^2 / (P*P*C), with per-patch
    normalized pixels as the target. Visible patches do not count."""
    mean, var = patches.mean(dim=-1, keepdim=True), patches.var(dim=-1, keepdim=True)
    target = (patches - mean) / (var + 1e-6) ** 0.5
    per_patch = ((pred - target) ** 2).mean(dim=-1)                    # [B, N]
    return (per_patch * mask).sum() / mask.sum()


class MAE(nn.Module):
    """Encoder on the visible quarter only, a small decoder on all N positions, pixels out.
    Defaults: a ViT-B encoder (the paper's ablations use ViT-L/16) with the paper's 8-block,
    512-wide decoder, 75% masked. The paper also carries a class token and uses fixed 2D
    sine-cosine positions; this sketch learns them."""

    def __init__(self, image_size: int = 224, patch: int = 16, channels: int = 3, dim: int = 768,
                 depth: int = 12, heads: int = 12, dec_dim: int = 512, dec_depth: int = 8,
                 dec_heads: int = 16, ratio: float = 0.75):
        super().__init__()
        N = (image_size // patch) ** 2
        self.ratio = ratio
        self.embed = PatchEmbed(patch, channels, dim)
        self.positions = nn.Parameter(0.02 * torch.randn(1, N, dim))
        self.encoder = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.to_decoder = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))                 # m
        self.dec_positions = nn.Parameter(0.02 * torch.randn(1, N, dec_dim))
        self.decoder = nn.ModuleList(Block(dec_dim, dec_heads) for _ in range(dec_depth))
        self.to_pixels = nn.Linear(dec_dim, patch * patch * channels)

    def forward(self, images: Tensor, generator: torch.Generator = None):
        x = self.embed(images) + self.positions                        # positions before masking
        h, mask, ids_restore = random_masking(x, self.ratio, generator)
        for block in self.encoder:
            h = block(h)                                               # [B, N/4, D] only
        h = unshuffle(self.to_decoder(self.norm(h)), self.mask_token, ids_restore) + self.dec_positions
        for block in self.decoder:
            h = block(h)                                               # [B, N, D_dec]
        pred = self.to_pixels(h)                                       # [B, N, P*P*C]
        return mae_loss(pred, patchify(images, self.embed.patch), mask), pred, mask
# END vit-mae


# BEGIN vit-dino
class DINOLoss(nn.Module):
    """L = sum over teacher crops x and student crops x' != x of  -P_t(x)^T log P_s(x').

    P_s(x) = softmax(g_s(x) / tau_s);  P_t(x) = softmax((g_t(x) - c) / tau_t), centred and sharpened.
    The centre c is an EMA of the teacher's batch mean. The sum is averaged over the pairs.
    """

    def __init__(self, out_dim: int, tau_s: float = 0.1, tau_t: float = 0.04, center_momentum: float = 0.9):
        super().__init__()
        self.tau_s, self.tau_t, self.m = tau_s, tau_t, center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))       # c

    def forward(self, student: list, teacher: list) -> Tensor:
        """student: [B, K] outputs for every crop, global crops first; teacher: one per global crop."""
        targets = [F.softmax((t - self.center) / self.tau_t, dim=-1).detach() for t in teacher]
        total, pairs = 0.0, 0
        for i, p_t in enumerate(targets):
            for j, s in enumerate(student):
                if j == i:                                             # the same view on both sides
                    continue
                total = total + (-p_t * F.log_softmax(s / self.tau_s, dim=-1)).sum(dim=-1).mean()
                pairs += 1
        self.update_center(torch.cat(teacher))
        return total / pairs

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """c <- m c + (1 - m) * mean of the teacher outputs in the batch."""
        self.center.mul_(self.m).add_((1 - self.m) * teacher_out.mean(dim=0, keepdim=True))


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, lam: float) -> None:
    """theta_t <- lam * theta_t + (1 - lam) * theta_s. The teacher never receives gradients."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(lam).add_((1 - lam) * s.detach())
# END vit-dino


# BEGIN vit-video
class TubeletEmbed(nn.Module):
    """ViViT: Conv3d with kernel = stride = (t, P, P), one token per t x P x P tube.
    [B, C, T, H, W] -> [B, T/t, (H/P)*(W/P), D]."""

    def __init__(self, t: int, patch: int, channels: int, dim: int):
        super().__init__()
        self.proj = nn.Conv3d(channels, dim, kernel_size=(t, patch, patch), stride=(t, patch, patch))

    def forward(self, video: Tensor) -> Tensor:
        x = self.proj(video)                                           # [B, D, T', H', W']
        B, D, T, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, D)


class DividedSpaceTimeBlock(nn.Module):
    """TimeSformer: attention over time (one patch position, all frames), then over space (one
    frame, all patches), then the MLP, each pre-norm with a residual. x: [B, T, N, D]."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_t, self.time = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm_s, self.space = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm_m, self.mlp = nn.LayerNorm(dim), mlp(dim)

    def mix_time(self, x: Tensor) -> Tensor:
        B, T, N, D = x.shape
        seqs = self.norm_t(x).transpose(1, 2).reshape(B * N, T, D)    # B*N sequences of length T
        return x + self.time(seqs).view(B, N, T, D).transpose(1, 2)

    def mix_space(self, x: Tensor) -> Tensor:
        B, T, N, D = x.shape
        seqs = self.norm_s(x).reshape(B * T, N, D)                     # B*T sequences of length N
        return x + self.space(seqs).view(B, T, N, D)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mix_space(self.mix_time(x))
        return x + self.mlp(self.norm_m(x))
# END vit-video


# BEGIN vit-navit
def packing_mask(lengths: list) -> Tensor:
    """NaViT: images of different sizes share one sequence of sum(lengths) tokens; the mask is
    block-diagonal, so a token reads only the tokens of its own image. [L, L], True = may read."""
    image = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    return image[:, None] == image[None, :]


class FactorizedPositions(nn.Module):
    """E_pos(y, x) = phi_y[y] + phi_x[x]: one table per axis, so any grid up to max_side on each
    side and any aspect ratio has positions."""

    def __init__(self, max_side: int, dim: int):
        super().__init__()
        self.phi_y = nn.Parameter(0.02 * torch.randn(max_side, dim))
        self.phi_x = nn.Parameter(0.02 * torch.randn(max_side, dim))

    def forward(self, rows: int, cols: int) -> Tensor:
        return (self.phi_y[:rows, None] + self.phi_x[None, :cols]).reshape(rows * cols, -1)  # [rows*cols, D]
# END vit-navit


# BEGIN vit-dit
class DiTBlock(nn.Module):
    """adaLN-Zero.  (beta1, gamma1, alpha1, beta2, gamma2, alpha2) = Linear(SiLU(c)), c = e_t + e_y;
    x <- x + alpha1 * MSA((1 + gamma1) * LN(x) + beta1);  x <- x + alpha2 * MLP((1 + gamma2) * LN(x) + beta2).
    The linear layer starts at zero, so every alpha is 0 and the block starts as the identity."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = mlp(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:                # x: [B, T, D], c: [B, D]
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = self.ada(c)[:, None].chunk(6, dim=-1)
        x = x + alpha1 * self.attn(self.norm1(x) * (1 + gamma1) + beta1)
        return x + alpha2 * self.mlp(self.norm2(x) * (1 + gamma2) + beta2)
# END vit-dit


# BEGIN vit-demo
def attention_cost(N: int, D: int, M: int = None) -> tuple:
    """Multiply-adds of one attention sublayer: 4 N D^2 for the projections, plus 2 N^2 D for the
    scores and the weighted sum, or 2 M^2 N D when every query reads only its M x M window."""
    return 4 * N * D * D, 2 * N * N * D if M is None else 2 * M * M * N * D


def main() -> None:
    torch.manual_seed(0)
    I = torch.randn(1, 3, 224, 224)                                    # the example: one 224 x 224 RGB image

    def show(name: str, value) -> None:
        print(f"{name:<48} {list(value.shape) if isinstance(value, Tensor) else value}")

    vit = ViT(dim=64, depth=2, heads=4).eval()                         # ViT-B/16 patches; D = 64, 2 blocks, 4 heads
    show("ViT     I", I)
    show("ViT     x_p E, P = 16", vit.embed(I))
    show("ViT     y_hat = LN(z_L^0) W_head", vit(I))
    show("ViT     E_pos for 384 px (14x14 -> 24x24)", resize_positions(vit.positions, 1, (24, 24)))
    deit = ViT(dim=64, depth=2, heads=4, distill=True).eval()
    show("DeiT    fused prediction of the two heads", deit_predict(deit, I))

    print("\ncost of one attention sublayer in multiply-adds: projections + scores and mixing")
    for name, N, D, M in (("ViT-B/16, 224 px", 197, 768, None), ("ViT-B/16, 1024 px (ViTDet)", 4096, 768, None),
                          ("Swin stage 1, global", 3136, 96, None), ("Swin stage 1, 7x7 windows", 3136, 96, 7)):
        linear, quadratic = attention_cost(N, D, M)
        print(f"  {name:<27} N = {N:>4}, D = {D:>3}: {linear / 1e6:7.1f}M + {quadratic / 1e6:7.1f}M"
              f"  ({100 * quadratic / (linear + quadratic):.0f}% in the second term)")

    print()
    swin = Swin(dim=24, depths=(2, 2, 2, 2)).eval()                    # Swin-T grids and heads; C = 24, 2 blocks in stage 3
    with torch.no_grad():
        for i, feature in enumerate(swin(I)):
            show(f"Swin    stage {i + 1}, {56 >> i}x{56 >> i} grid", feature)
    show("PVT     R = 8: 3136 queries read 49 keys", SpatialReductionAttention(24, 1, R=8)(torch.randn(1, 3136, 24), 56, 56))

    print()
    mae = MAE(dim=64, depth=2, heads=4, dec_dim=32, dec_depth=1, dec_heads=4)
    loss, pred, mask = mae(I, torch.Generator().manual_seed(0))
    show("MAE     patches seen by the encoder", f"{int((mask == 0).sum())} of {mask.shape[1]}")
    show("MAE     decoder output, pixels per patch", pred)
    show("MAE     loss over the removed patches only", f"{loss.item():.3f} ({int(mask.sum())} patches)")
    head = nn.Linear(64, 256)
    crops = [torch.randn(4, 64) for _ in range(2 + 4)]                 # 2 global + 4 local crops
    loss = DINOLoss(256)([head(c) for c in crops], [head(c) for c in crops[:2]])
    show("DINO    loss, 2 teacher crops x 6 student crops", f"{loss.item():.3f}")

    print()
    video = torch.randn(1, 3, 8, 224, 224)                             # [B, C, T, H, W]
    tokens = TubeletEmbed(1, 16, 3, 64)(video)
    show("TimeSformer  8 frames -> [B, T, N, D]", tokens)
    show("TimeSformer  divided space-time block", DividedSpaceTimeBlock(64, 4)(tokens))
    show("ViViT   2 x 16 x 16 tubelets", TubeletEmbed(2, 16, 3, 64)(video))

    print()
    embed, attn = PatchEmbed(16, 3, 64), Attention(64, 4)
    images = [torch.randn(1, 3, 224, 160), torch.randn(1, 3, 96, 224)]   # two shapes, no resizing
    tokens = [embed(image) for image in images]                        # 140 and 84 tokens
    lengths = [t.shape[1] for t in tokens]
    packed = attn(torch.cat(tokens, dim=1), mask=packing_mask(lengths)[None, None])
    show("NaViT   224x160 and 96x224 in one sequence", f"{lengths[0]} + {lengths[1]} = {sum(lengths)} tokens")
    show("NaViT   packed equals one image at a time", torch.allclose(packed, torch.cat([attn(t) for t in tokens], dim=1), atol=1e-5))
    x, c = torch.randn(1, (32 // 2) ** 2, 64), torch.randn(1, 64)
    show("DiT     latent 32x32x4 with p = 2", x)
    show("DiT     block at initialization is the identity", torch.equal(DiTBlock(64, 4)(x, c), x))


if __name__ == "__main__":
    main()
# END vit-demo
