"""Chapter 5 of the ML book: serving a decoder-only Transformer token by token.

Prefill, the KV cache, one decode step and sampling, on a toy model trained on a few sentences so
that every claim of the chapter can be checked.

Run:   .venv-transformer/bin/python inference_kv_cache.py
Tests: .venv-transformer/bin/python -m unittest -v test_inference_kv_cache
"""
# BEGIN inference-imports
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
# END inference-imports


# BEGIN inference-model
class CausalSelfAttention(nn.Module):
    """Step 2 of Chapter 2 with a causal mask, written so that past keys and values are reused."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads, self.d_k = num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        batch, t_new, d_model = x.shape
        heads = lambda z: z.view(batch, t_new, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, t_new, d_k]
        q, k, v = heads(self.w_q(x)), heads(self.w_k(x)), heads(self.w_v(x))
        if cache is not None:                        # step 3: the past is read, never recomputed
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        t_past = k.shape[2] - t_new
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)      # [B, h, t_new, t_past + t_new]
        causal = torch.ones(t_new, t_past + t_new, dtype=torch.bool, device=x.device).tril(diagonal=t_past)
        scores = scores.masked_fill(~causal, float("-inf"))
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, t_new, d_model)
        return self.w_o(out), {"k": k, "v": v}


class Block(nn.Module):
    """Pre-norm block:  x' = x + Attention(LN(x));  x'' = x' + FFN(LN(x'))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        a, cache = self.attention(self.norm1(x), cache)
        x = x + a
        return x + self.ffn(self.norm2(x)), cache


def sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    """The fixed table P of Chapter 2, step 1."""
    position = torch.arange(max_len, dtype=torch.float64)[:, None]
    frequency = 10000.0 ** (-torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
    table = torch.zeros(max_len, d_model, dtype=torch.float64)
    table[:, 0::2], table[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
    return table


class CausalLM(nn.Module):
    """A decoder-only Transformer: embeddings + positions, L pre-norm blocks, final norm, LM head."""

    def __init__(self, vocab_size: int, d_model: int = 32, num_heads: int = 4, d_ff: int = 64,
                 num_layers: int = 2, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("positions", sinusoidal_positions(max_len, d_model).float())
        self.blocks = nn.ModuleList(Block(d_model, num_heads, d_ff) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: Tensor, caches: list | None = None, start: int = 0) -> tuple[Tensor, list]:
        """ids: [B, t_new] token ids that sit at positions start .. start + t_new - 1.
        Returns the logits of every new position, [B, t_new, V], and the updated cache of every layer."""
        x = self.embedding(ids) * math.sqrt(self.d_model) + self.positions[start:start + ids.shape[1]].to(ids.device)
        updated = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[i])
            updated.append(cache)
        return self.lm_head(self.norm(x)), updated
# END inference-model


# BEGIN inference-cache
def prefill(model: CausalLM, ids: Tensor) -> tuple[Tensor, list]:
    """Step 1: the whole prompt in one pass. Returns the logits of the last position and every layer's cache."""
    logits, caches = model(ids)
    return logits[:, -1], caches


def decode_step(model: CausalLM, token: Tensor, caches: list) -> tuple[Tensor, list]:
    """Step 3: one new token reads the cache and appends its own key and value. token: [B]."""
    start = caches[0]["k"].shape[2]
    logits, caches = model(token[:, None], caches, start=start)
    return logits[:, -1], caches


def cache_bytes(num_layers: int, kv_heads: int, d_head: int, tokens: int, batch: int = 1, bytes_per_value: int = 2) -> int:
    """Step 2: 2 (K and V) x layers x KV heads x d_head x tokens x batch x bytes per value."""
    return 2 * num_layers * kv_heads * d_head * tokens * batch * bytes_per_value


@torch.no_grad()
def generate(model: CausalLM, prompt: Tensor, max_new_tokens: int, sample, use_cache: bool = True) -> Tensor:
    """Steps 1 and 3 in a loop: prefill once, then one decode step per token. `sample` maps logits [B, V] to tokens [B]."""
    ids = prompt
    logits, caches = prefill(model, ids)
    for _ in range(max_new_tokens):
        token = sample(logits)
        ids = torch.cat([ids, token[:, None]], dim=1)
        if use_cache:
            logits, caches = decode_step(model, token, caches)
        else:                                        # the quadratic alternative: rerun the whole prefix
            logits = model(ids)[0][:, -1]
    return ids
# END inference-cache


# BEGIN inference-sampling
def filtered_probabilities(logits: Tensor, temperature: float = 1.0, top_k: int | None = None,
                           top_p: float | None = None, min_p: float | None = None) -> Tensor:
    """Step 5: temperature, then truncation, then renormalisation. logits [B, V] -> probabilities [B, V]."""
    if temperature <= 0:                             # the T -> 0 limit: all mass on the arg max (greedy)
        return nn.functional.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)
    probs = (logits / temperature).softmax(dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if top_k is not None:                            # the k largest probabilities
        keep &= probs >= probs.topk(top_k, dim=-1).values[..., -1:]
    if top_p is not None:                            # the smallest prefix of the sorted list whose mass reaches p
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        keep &= torch.zeros_like(keep).scatter(-1, order, mass_before < top_p)
    if min_p is not None:                            # at least a fraction min_p of the largest probability
        keep &= probs >= min_p * probs.amax(dim=-1, keepdim=True)
    probs = probs * keep
    return probs / probs.sum(dim=-1, keepdim=True)


def repetition_penalty(logits: Tensor, context: Tensor, penalty: float) -> Tensor:
    """Logits of tokens already in the context are divided by `penalty` when positive and multiplied when negative."""
    logits = logits.clone()
    for b in range(logits.shape[0]):
        seen = context[b].unique()
        z = logits[b, seen]
        logits[b, seen] = torch.where(z > 0, z / penalty, z * penalty)
    return logits


def sample_next(logits: Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                min_p: float | None = None, generator: torch.Generator | None = None) -> Tensor:
    probs = filtered_probabilities(logits, temperature, top_k, top_p, min_p)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
# END inference-sampling



# BEGIN inference-demo
CORPUS = ("the cat sat on the mat . the dog sat on the log . "
          "the cat saw the dog on the log . the dog saw the cat on the mat .")
WORDS = sorted(set(CORPUS.split()))
IDS = torch.tensor([[WORDS.index(w) for w in CORPUS.split()]])


def train(model: CausalLM, steps: int, lr: float = 5e-3) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        logits, _ = model(IDS[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), IDS[:, 1:].reshape(-1))
        optimizer.zero_grad(), loss.backward(), optimizer.step()
    return loss.item()


def words(ids: Tensor) -> str:
    return " ".join(WORDS[i] for i in ids[0].tolist())


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CausalLM(len(WORDS), d_model=32, num_heads=4, d_ff=64, num_layers=2)
    print(f"model: 2 layers, d=32, trained to loss {train(model, 400):.3f}; vocabulary of {len(WORDS)} words")

    with torch.no_grad():
        full, _ = model(IDS)                                       # every position in one pass
        logits, caches = prefill(model, IDS[:, :4])
        cached = [logits]
        for t in range(4, IDS.shape[1]):
            logits, caches = decode_step(model, IDS[:, t], caches)
            cached.append(logits)
        gap = (torch.stack(cached, dim=1) - full[:, 3:]).abs().max().item()
    print(f"cache matches full recompute: {gap < 1e-4} (max |diff| {gap:.1e} over {IDS.shape[1] - 4} decode steps)")
    print(f"KV cache of LLaMA-3-8B: {cache_bytes(32, 8, 128, 1) // 1024} KiB per token, "
          f"{cache_bytes(32, 8, 128, 8192) / 1e9:.2f} GB at 8192 tokens")

    z = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    show = lambda p: "[" + " ".join(f"{v:.3f}" for v in p[0].tolist()) + "]"
    print(f"sampling on logits {z[0].tolist()}:")
    print(f"  T=1.0     -> {show(filtered_probabilities(z))}")
    print(f"  T=0.5     -> {show(filtered_probabilities(z, temperature=0.5))}")
    print(f"  top-k=2   -> {show(filtered_probabilities(z, top_k=2))}")
    print(f"  top-p=0.9 -> {show(filtered_probabilities(z, top_p=0.9))}")

    prompt = IDS[:, :2]
    greedy = lambda logits: logits.argmax(dim=-1)
    with_cache = generate(model, prompt, 12, greedy)
    without = generate(model, prompt, 12, greedy, use_cache=False)
    print(f'greedy from "{words(prompt)}": {words(with_cache[:, 2:])}')
    print(f"greedy with cache == greedy without cache: {torch.equal(with_cache, without)}")
    generator = torch.Generator().manual_seed(0)
    sampled = generate(model, prompt, 12, lambda logits: sample_next(logits, temperature=1.0, top_p=0.9, generator=generator))
    print(f'sampled at T=1, top-p=0.9: {words(sampled[:, 2:])}')
# END inference-demo
