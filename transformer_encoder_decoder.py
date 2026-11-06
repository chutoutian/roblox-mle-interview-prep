"""A minimal encoder-decoder Transformer that follows Chapter 4 equation by equation.

Run:   .venv-transformer/bin/python transformer_encoder_decoder.py
Tests: .venv-transformer/bin/python -m unittest -v test_transformer_encoder_decoder

Post-norm, as in "Attention Is All You Need": x = LayerNorm(x + Sublayer(x)).
Dropout and weight tying are left out so that every line maps to one equation.
Masks are boolean and True means "this key may be read".
"""

# BEGIN transformer-imports
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
# END transformer-imports


# BEGIN transformer-embedding
def sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    """P[p, 2i] = sin(p / 10000^(2i/d)),  P[p, 2i+1] = cos(p / 10000^(2i/d))."""
    assert d_model % 2 == 0, "sin/cos pairs need an even d_model"
    position = torch.arange(max_len, dtype=torch.float32)[:, None]      # [L, 1]
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)            # [d/2]: 0, 2, 4, ...
    angle = position / (10000 ** (two_i / d_model))                     # [L, d/2]
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe                                                           # [L, d]


class TokenEmbedding(nn.Module):
    """Steps 1 and 5:  X = sqrt(d) * E[ids] + P[:n]."""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # E: [V, d], learned
        self.scale = math.sqrt(d_model)
        self.register_buffer("positions", sinusoidal_positions(max_len, d_model))  # P: [L, d], fixed

    def forward(self, ids: Tensor) -> Tensor:
        n = ids.shape[1]
        return self.embedding(ids) * self.scale + self.positions[:n]    # [B, n] -> [B, n, d]
# END transformer-embedding


# BEGIN transformer-attention
class MultiHeadAttention(nn.Module):
    """Steps 2, 6 and 7.

    head_i = softmax(Q_i K_i^T / sqrt(d_k) + M) V_i
    MHA(query, key, value) = [head_1 || ... || head_h] W_O
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must split evenly into heads"
        self.h, self.d_k = num_heads, d_model // num_heads
        # One Linear(d, d) holds the h per-head matrices W_i^Q side by side; same for K, V.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attention_weights = None  # [B, h, Lq, Lk] of the last call, kept for inspection

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        # query [B, Lq, d]; key, value [B, Lk, d]; mask broadcasts to [B, h, Lq, Lk], True = may read.
        B, Lq, d = query.shape
        Lk = key.shape[1]
        q = self.w_q(query).view(B, Lq, self.h, self.d_k).transpose(1, 2)   # [B, h, Lq, d_k]
        k = self.w_k(key).view(B, Lk, self.h, self.d_k).transpose(1, 2)     # [B, h, Lk, d_k]
        v = self.w_v(value).view(B, Lk, self.h, self.d_k).transpose(1, 2)   # [B, h, Lk, d_k]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)              # [B, h, Lq, Lk]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))               # blocked key -> weight 0
        weights = F.softmax(scores, dim=-1)                                 # each query row sums to 1
        self.attention_weights = weights
        heads = weights @ v                                                 # [B, h, Lq, d_k]
        concat = heads.transpose(1, 2).reshape(B, Lq, d)                    # [B, Lq, h * d_k] = [B, Lq, d]
        return self.w_o(concat)                                             # [B, Lq, d]
# END transformer-attention


# BEGIN transformer-rope
def speeds(m: int, base: float = 10000.0) -> Tensor:
    """theta_0 .. theta_{m-1} = base^(-j/m): from 1 radian per position down to almost 0."""
    return base ** (-torch.arange(m, dtype=torch.float64) / m)


def angles_1d(p: Tensor, d_k: int) -> Tensor:
    """Text. Token at position p: every pair j turns by p * theta_j.  [L] -> [L, d_k/2]"""
    return p[:, None] * speeds(d_k // 2)


def angles_2d(h: Tensor, w: Tensor, d_k: int) -> Tensor:
    """Image. Patch in row h, column w: half of the pairs turn with h, the other half with w."""
    theta = speeds(d_k // 4)
    return torch.cat([h[:, None] * theta, w[:, None] * theta], dim=1)


def angles_3d(t: Tensor, h: Tensor, w: Tensor, d_k: int) -> Tensor:
    """Video. Patch in frame t, row h, column w: the pairs are dealt out t, h, w, t, h, w, ..."""
    position = torch.stack([t, h, w], dim=1)          # [L, 3]
    owner = torch.arange(d_k // 2) % 3                # pair j belongs to coordinate j mod 3
    return position[:, owner] * speeds(d_k // 2)


def rotate(x: Tensor, angle: Tensor) -> Tensor:
    """Turn pair j of row p of x by angle[p, j].  x: [..., L, d_k], angle: [L, d_k/2].

    In attention this is applied to the rows of Q and K after the head split and before the
    scores; V is left alone.
    """
    cos, sin = angle.cos().to(x), angle.sin().to(x)
    even, odd = x[..., 0::2], x[..., 1::2]            # x_2j and x_2j+1
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def rotary_scores(q: Tensor, k: Tensor, q_angle: Tensor, k_angle: Tensor) -> Tensor:
    """Scores of step 2 with RoPE: q [..., Lq, d_k], k [..., Lk, d_k] -> [..., Lq, Lk]."""
    return rotate(q, q_angle) @ rotate(k, k_angle).transpose(-2, -1) / math.sqrt(q.shape[-1])
# END transformer-rope


# BEGIN transformer-blocks
class FeedForward(nn.Module):
    """FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2, applied to every position independently."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # W_1: d -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)   # W_2: d_ff -> d

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))   # [B, L, d] -> [B, L, d_ff] -> [B, L, d]


class EncoderLayer(nn.Module):
    """Steps 2-4:  X1 = LN(X + MHA(X, X, X));  H = LN(X1 + FFN(X1))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)   # normalizes the d features of each token separately
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.norm1(x + self.self_attention(x, x, x, src_mask))
        return self.norm2(x + self.ffn(x))


class DecoderLayer(nn.Module):
    """Steps 6-8:  Y1 = LN(Y + MHA(Y, Y, Y; causal));  Y2 = LN(Y1 + MHA(Y1, H, H; pad));
    Y3 = LN(Y2 + FFN(Y2))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y: Tensor, memory: Tensor, tgt_mask: Tensor, src_mask: Tensor) -> Tensor:
        y = self.norm1(y + self.self_attention(y, y, y, tgt_mask))
        y = self.norm2(y + self.cross_attention(y, memory, memory, src_mask))  # Q from y; K, V from H
        return self.norm3(y + self.ffn(y))
# END transformer-blocks


# BEGIN transformer-model
class Transformer(nn.Module):
    """Encoder(s) -> memory H;  Decoder(t, H) -> logits Z.  Steps 1-9 end to end."""

    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 8, num_heads: int = 2,
                 d_ff: int = 32, num_layers: int = 1, pad_id: int = 0, max_len: int = 512):
        super().__init__()
        self.pad_id = pad_id
        self.src_embedding = TokenEmbedding(src_vocab, d_model, pad_id, max_len)
        self.tgt_embedding = TokenEmbedding(tgt_vocab, d_model, pad_id, max_len)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers))
        self.vocab_projection = nn.Linear(d_model, tgt_vocab)   # W_vocab: d -> V_tgt

    def source_mask(self, src_ids: Tensor) -> Tensor:
        return (src_ids != self.pad_id)[:, None, None, :]        # [B, 1, 1, S]: PAD keys are hidden

    def target_mask(self, tgt_ids: Tensor) -> Tensor:
        T = tgt_ids.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=tgt_ids.device).tril()  # key k <= query j
        return causal[None, None] & (tgt_ids != self.pad_id)[:, None, None, :]     # [B, 1, T, T]

    def encode(self, src_ids: Tensor) -> Tensor:
        src_mask = self.source_mask(src_ids)
        x = self.src_embedding(src_ids)                          # step 1:    [B, S] -> [B, S, d]
        for layer in self.encoder:
            x = layer(x, src_mask)                               # steps 2-4: [B, S, d]
        return x                                                 # memory H:  [B, S, d]

    def decode(self, tgt_ids: Tensor, memory: Tensor, src_mask: Tensor) -> Tensor:
        tgt_mask = self.target_mask(tgt_ids)
        y = self.tgt_embedding(tgt_ids)                          # step 5:    [B, T] -> [B, T, d]
        for layer in self.decoder:
            y = layer(y, memory, tgt_mask, src_mask)             # steps 6-8: [B, T, d]
        return self.vocab_projection(y)                          # step 9:    logits Z [B, T, V_tgt]

    def forward(self, src_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        return self.decode(tgt_ids, self.encode(src_ids), self.source_mask(src_ids))


@torch.no_grad()
def greedy_decode(model: Transformer, src_ids: Tensor, bos_id: int = 1, eos_id: int = 2,
                  max_new_tokens: int = 8) -> Tensor:
    """Step 11: encode once, then append argmax_v Z_j until every row has produced EOS."""
    memory, src_mask = model.encode(src_ids), model.source_mask(src_ids)
    ids = torch.full((src_ids.shape[0], 1), bos_id, dtype=torch.long, device=src_ids.device)
    finished = torch.zeros(src_ids.shape[0], dtype=torch.bool, device=src_ids.device)
    for _ in range(max_new_tokens):
        next_id = model.decode(ids, memory, src_mask)[:, -1].argmax(dim=-1)  # last row of Z only
        next_id = torch.where(finished, model.pad_id, next_id)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        finished |= next_id == eos_id
        if finished.all():
            break
    return ids
# END transformer-model


# BEGIN transformer-demo
def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=2, sci_mode=False)
    src_vocab = ["<pad>", "<bos>", "<eos>", "we", "watch", "a", "short", "video"]
    tgt_vocab = ["<pad>", "<bos>", "<eos>", "nous", "regardons", "video", "une", "courte", "tu", "regardes"]
    s = torch.tensor([[3, 4, 5, 6, 7]])   # we watch a short video     [B=1, n=5]
    t = torch.tensor([[1, 3, 4, 5]])      # BOS nous regardons video   [B=1, m=4]
    y = torch.tensor([[3, 4, 5, 2]])      # nous regardons video EOS   [B=1, m=4]
    print("s =", s[0].tolist(), "=", " ".join(src_vocab[i] for i in s[0]))
    print("t =", t[0].tolist(), "=", " ".join(tgt_vocab[i] for i in t[0]))
    print("y =", y[0].tolist(), "=", " ".join(tgt_vocab[i] for i in y[0]), "\n")

    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=8, num_heads=2, d_ff=32)
    enc, dec = model.encoder[0], model.decoder[0]
    src_mask, tgt_mask = model.source_mask(s), model.target_mask(t)

    def show(step: str, name: str, tensor: Tensor) -> Tensor:
        print(f"step {step:<3} {name:<38} {list(tensor.shape)}")
        return tensor

    X = show("1", "X = sqrt(d) E_src[s] + P", model.src_embedding(s))
    A = show("2", "A = MHA(X, X, X)", enc.self_attention(X, X, X, src_mask))
    show("2", "    softmax weights, per head", enc.self_attention.attention_weights)
    X1 = show("3", "X1 = LN(X + A)", enc.norm1(X + A))
    H = show("4", "H = LN(X1 + FFN(X1))", enc.norm2(X1 + enc.ffn(X1)))
    Y = show("5", "Y = sqrt(d) E_tgt[t] + P", model.tgt_embedding(t))
    Y1 = show("6", "Y1 = LN(Y + MHA(Y, Y, Y; causal))", dec.norm1(Y + dec.self_attention(Y, Y, Y, tgt_mask)))
    show("6", "    causal weights, per head", dec.self_attention.attention_weights)
    Y2 = show("7", "Y2 = LN(Y1 + MHA(Y1, H, H; pad))", dec.norm2(Y1 + dec.cross_attention(Y1, H, H, src_mask)))
    show("7", "    cross weights, per head", dec.cross_attention.attention_weights)
    Y3 = show("8", "Y3 = LN(Y2 + FFN(Y2))", dec.norm3(Y2 + dec.ffn(Y2)))
    Z = show("9", "Z = Y3 W_vocab + b", model.vocab_projection(Y3))
    assert torch.allclose(Z, model(s, t))  # the step-by-step path is exactly model.forward

    print("\ncausal mask, rows = queries, columns = keys (1 = may read):")
    print(tgt_mask[0, 0].int())
    print("\ndecoder self-attention weights, head 0 (rows sum to 1; zeros above the diagonal):")
    print(dec.self_attention.attention_weights[0, 0].detach())

    loss = F.cross_entropy(Z.reshape(-1, len(tgt_vocab)), y.reshape(-1), ignore_index=model.pad_id)
    print(f"\nstep 10  loss = -mean_j log p(y_j | t_1..t_j, s) = {loss.item():.4f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss.backward()    # dL/dtheta for every parameter, encoder included
    optimizer.step()   # one update; a real model needs many thousands

    model.eval()
    ids = greedy_decode(model, s, max_new_tokens=6)
    print("step 11  greedy decode:", ids[0].tolist(), "=",
          " ".join(tgt_vocab[i] for i in ids[0]), "(untrained, so the words are arbitrary)")


if __name__ == "__main__":
    main()
# END transformer-demo
