"""A minimal decoder-only LLM in the style of current open models (LLaMA, Qwen, Mistral).

One causal stream of tokens, pre-norm blocks with RMSNorm, rotary positions (RoPE) on queries and
keys, grouped-query attention (GQA), a SwiGLU feed-forward, no biases, a final RMSNorm and an untied
LM head; the next-token loss; and generation that prefills the prompt once and then decodes one token
at a time from a KV cache.

Run:      .venv-transformer/bin/python decoder_only_llm.py
Tests:    .venv-transformer/bin/python -m unittest -v test_decoder_only_llm
Notebook: decoder_only_llm.ipynb walks through it block by block.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 11
    d_model: int = 16          # d: width of every token vector
    n_layers: int = 2          # L: number of blocks
    n_heads: int = 4           # h: query heads
    n_kv_heads: int = 2        # h_kv: key/value heads; h_kv = h is MHA, h_kv = 1 is MQA
    d_ff: int = 48             # SwiGLU width: 8/3 d = 42.7, rounded up to a multiple of 8
    max_seq_len: int = 16      # rows of the RoPE tables
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g: rescale each token, no mean subtraction, no bias."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))   # g, learned, one per feature

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()                               # like LLaMA, compute the norm in float32 even for fp16/bf16 inputs
        return (x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_angles(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """cos and sin of p * theta_j for every position p and pair j: two tables of shape [max_seq_len, head_dim/2]."""
    theta = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)   # [head_dim/2]: 1, ..., small
    angle = torch.arange(max_seq_len, dtype=torch.float32)[:, None] * theta           # [max_seq_len, head_dim/2]
    return angle.cos(), angle.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Turn each pair (x_2j, x_2j+1) of each row by its angle. x: [..., n, head_dim]; cos, sin: [n, head_dim/2]."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Causal self-attention with RoPE and grouped-query attention; can read and extend a KV cache."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "query heads must split evenly into key/value groups"
        self.h, self.h_kv, self.d_h = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, self.h * self.d_h, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.h_kv * self.d_h, bias=False)   # fewer K, V heads than Q heads
        self.wv = nn.Linear(cfg.d_model, self.h_kv * self.d_h, bias=False)
        self.wo = nn.Linear(self.h * self.d_h, cfg.d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                cache: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # x [B, n, d] at the positions given by cos, sin [n, d_h/2]; cache: (K, V) of the earlier positions.
        B, n, _ = x.shape
        q = self.wq(x).view(B, n, self.h, self.d_h).transpose(1, 2)        # [B, h, n, d_h]
        k = self.wk(x).view(B, n, self.h_kv, self.d_h).transpose(1, 2)     # [B, h_kv, n, d_h]
        v = self.wv(x).view(B, n, self.h_kv, self.d_h).transpose(1, 2)     # [B, h_kv, n, d_h]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)            # positions enter here; V is not turned
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)                            # [B, h_kv, T, d_h], T = past + n
            v = torch.cat([cache[1], v], dim=2)
        cache = (k, v)
        group = self.h // self.h_kv
        k = k.repeat_interleave(group, dim=1)                              # [B, h, T, d_h]: K/V head g serves
        v = v.repeat_interleave(group, dim=1)                              # query heads g*group .. g*group+group-1
        T = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)            # [B, h, n, T]
        causal = torch.ones(n, T, dtype=torch.bool, device=x.device).tril(diagonal=T - n)  # query i reads keys <= its position
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, n, self.h * self.d_h)   # concatenate the heads
        return self.wo(out), cache


class SwiGLU(nn.Module):
    """FFN(x) = (SiLU(x W1) * x W3) W2: the SiLU branch gates the linear branch, element by element."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)   # gate branch
        self.w3 = nn.Linear(d, d_ff, bias=False)   # linear ("up") branch
        self.w2 = nn.Linear(d_ff, d, bias=False)   # back to d ("down")

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))   # [B, n, d] -> [B, n, d_ff] -> [B, n, d]


class Block(nn.Module):
    """Pre-norm: x <- x + Attention(RMSNorm(x)); x <- x + SwiGLU(RMSNorm(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                cache: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        a, cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + a                                     # the residual stream itself is never normalized
        x = x + self.ffn(self.ffn_norm(x))
        return x, cache


class DecoderLM(nn.Module):
    """Token embedding -> L pre-norm blocks -> final RMSNorm -> LM head. No position vector is added."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight   # one [V, d] matrix for input and output
        cos, sin = rope_angles(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, ids: Tensor, caches: Optional[list] = None, start: int = 0) -> tuple[Tensor, list]:
        """ids [B, n] at positions start .. start+n-1 -> logits [B, n, V] and every layer's (K, V)."""
        n = ids.shape[1]
        assert start + n <= self.cfg.max_seq_len, "longer than the RoPE tables"
        cos, sin = self.rope_cos[start:start + n], self.rope_sin[start:start + n]
        x = self.embed(ids)                                            # [B, n, d]
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, None if caches is None else caches[i])
            new_caches.append(cache)
        return self.lm_head(self.norm(x)), new_caches                  # [B, n, V]


def next_token_loss(logits: Tensor, ids: Tensor, pad_id: Optional[int] = None) -> Tensor:
    """Position j predicts token j+1: drop the last row of logits and the first token of ids."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                           ignore_index=-100 if pad_id is None else pad_id)


@torch.no_grad()
def generate(model: DecoderLM, prompt: Tensor, max_new_tokens: int, temperature: float = 0.0,
             eos_id: Optional[int] = None) -> Tensor:
    """Prefill the prompt once, then feed one new token per step and reuse every layer's K and V."""
    assert temperature >= 0, "temperature 0 is greedy; a negative one would invert the distribution"
    logits, caches = model(prompt)                                     # prefill: [B, n, V]
    ids = prompt
    done = torch.zeros(prompt.shape[0], 1, dtype=torch.bool, device=prompt.device)   # rows that emitted eos
    steps = min(max_new_tokens, model.cfg.max_seq_len - prompt.shape[1])   # never run past the RoPE tables
    for step in range(steps):
        last = logits[:, -1]                                           # only the last position predicts
        if temperature == 0:
            next_id = last.argmax(dim=-1, keepdim=True)                # greedy
        else:
            next_id = torch.multinomial(F.softmax(last / temperature, dim=-1), 1)
        if eos_id is not None:
            next_id = next_id.masked_fill(done, eos_id)                # a finished row keeps emitting eos
            done = done | (next_id == eos_id)
        ids = torch.cat([ids, next_id], dim=1)
        if step == steps - 1 or bool(done.all()):
            break                                                      # no forward pass whose logits go unused
        logits, caches = model(next_id, caches, start=ids.shape[1] - 1)   # decode: [B, 1, V], at its own position
    return ids


def kv_cache_numbers(cfg: Config, tokens: int) -> int:
    """Numbers stored in the KV cache: 2 (K and V) x layers x tokens x h_kv x d_h."""
    return 2 * cfg.n_layers * tokens * cfg.n_kv_heads * cfg.head_dim


VOCAB = ["<pad>", "<bos>", "<eos>", "we", "watch", "a", "short", "video", "you", "read", "book"]
PAD, BOS, EOS = 0, 1, 2
SENTENCES = [["we", "watch", "a", "short", "video"], ["you", "read", "a", "short", "book"]]


def encode(words: list) -> Tensor:
    return torch.tensor([[BOS] + [VOCAB.index(w) for w in words] + [EOS]])


def decode_ids(ids: Tensor) -> str:
    return " ".join(VOCAB[i] for i in ids.tolist())


def train(model: DecoderLM, steps: int = 300, lr: float = 1e-2) -> list:
    """Fit the two sentences; returns the loss every 50 steps."""
    batch = torch.cat([encode(s) for s in SENTENCES])                  # [2, 7]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    for step in range(steps + 1):
        logits, _ = model(batch)
        loss = next_token_loss(logits, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            history.append(round(loss.item(), 4))
    return history


def main() -> None:
    torch.manual_seed(0)
    cfg = Config()
    model = DecoderLM(cfg)
    ids = encode(SENTENCES[0])                                         # BOS we watch a short video EOS: [1, 7]
    logits, caches = model(ids)
    print("ids", list(ids.shape), "->", decode_ids(ids[0]))
    print("logits", list(logits.shape), "  K per layer", list(caches[0][0].shape), "  V per layer", list(caches[0][1].shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("loss before training:", round(next_token_loss(logits, ids).item(), 4), " log V =", round(math.log(cfg.vocab_size), 4))
    print("loss every 50 steps:", train(model))
    for first in ["we", "you"]:
        prompt = torch.tensor([[BOS, VOCAB.index(first)]])
        print("generate from", decode_ids(prompt[0]), "->", decode_ids(generate(model, prompt, 8, eos_id=EOS)[0]))
    print("KV cache for 7 tokens:", kv_cache_numbers(cfg, 7), "numbers; with MHA it would be",
          kv_cache_numbers(Config(n_kv_heads=cfg.n_heads), 7))


if __name__ == "__main__":
    main()
