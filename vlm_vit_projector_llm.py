"""A minimal vision-language model that follows Chapter 3 equation by equation.

Layout (LLaVA): ViT vision encoder -> MLP projector -> decoder-only LLM.
Run:   .venv-transformer/bin/python vlm_vit_projector_llm.py
Tests: .venv-transformer/bin/python -m unittest -v test_vlm_vit_projector_llm

Attention is the module of Chapter 2 (transformer_encoder_decoder.py); masks are
boolean and True means "this key may be read". One image per row, no padding.
"""

# BEGIN vlm-imports
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from transformer_encoder_decoder import MultiHeadAttention  # Chapter 2, step 2

IGNORE = -100  # label of a position that is not trained on (F.cross_entropy's ignore_index)
# END vlm-imports


# BEGIN vlm-patches
def patchify(images: Tensor, patch: int) -> Tensor:
    """Step 1: [B, C, H, W] -> [B, N, P*P*C], patches in row-major order."""
    B, C, H, W = images.shape                                     # PyTorch stores images channel-first
    x = images.reshape(B, C, H // patch, patch, W // patch, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)                               # [B, H/P, W/P, P, P, C]
    return x.reshape(B, (H // patch) * (W // patch), patch * patch * C)


class PatchEmbedding(nn.Module):
    """Step 2:  Z0 = X_p W_E + b_E + P_vis.  Equal to Conv2d(C, d_v, kernel=P, stride=P)."""

    def __init__(self, image_size: int, patch: int, channels: int, d_vision: int):
        super().__init__()
        self.patch = patch
        self.linear = nn.Linear(patch * patch * channels, d_vision)                        # W_E, b_E
        self.positions = nn.Parameter(0.02 * torch.randn((image_size // patch) ** 2, d_vision))  # P_vis

    def forward(self, images: Tensor) -> Tensor:
        return self.linear(patchify(images, self.patch)) + self.positions                  # [B, N, d_v]
# END vlm-patches


# BEGIN vlm-blocks
class Block(nn.Module):
    """Steps 3 and 7, pre-norm:  x' = x + MHA(LN(x));  x = x' + FFN(LN(x'))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attention(h, h, h, mask)   # mask=None: every token reads every token
        return x + self.ffn(self.norm2(x))
# END vlm-blocks


# BEGIN vlm-vision
class VisionEncoder(nn.Module):
    """Steps 1-3: image -> one contextual vector per patch. No mask, so it is bidirectional."""

    def __init__(self, image_size: int = 8, patch: int = 4, channels: int = 3,
                 d_vision: int = 8, num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        self.d_vision = d_vision
        self.embed = PatchEmbedding(image_size, patch, channels, d_vision)
        self.layers = nn.ModuleList(Block(d_vision, num_heads, 4 * d_vision) for _ in range(num_layers))

    def forward(self, images: Tensor) -> Tensor:
        z = self.embed(images)        # steps 1-2: [B, C, H, W] -> [B, N, d_v]
        for layer in self.layers:
            z = layer(z)              # step 3
        return z                      # Z: [B, N, d_v]


class Projector(nn.Module):
    """Step 4:  U = GELU(Z W_1 + b_1) W_2 + b_2, from the vision width d_v to the LLM width d."""

    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(d_vision, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear2(F.gelu(self.linear1(z)))   # [B, N, d_v] -> [B, N, d]
# END vlm-vision


# BEGIN vlm-model
class VLM(nn.Module):
    """Steps 1-8: image + text ids -> next-token logits over the vocabulary."""

    def __init__(self, vocab_size: int, image_token_id: int, d_model: int = 16, num_heads: int = 2,
                 num_layers: int = 2, max_len: int = 64, vision: VisionEncoder = None):
        super().__init__()
        self.image_token_id = image_token_id
        self.vision = vision if vision is not None else VisionEncoder()
        self.projector = Projector(self.vision.d_vision, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)                      # E
        self.positions = nn.Parameter(0.02 * torch.randn(max_len, d_model))     # P_txt (real LLMs: RoPE)
        self.layers = nn.ModuleList(Block(d_model, num_heads, 4 * d_model) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.vocab_projection = nn.Linear(d_model, vocab_size, bias=False)      # W_vocab

    def encode_image(self, images: Tensor) -> Tensor:
        return self.projector(self.vision(images))                 # steps 1-4: [B, C, H, W] -> [B, N, d]

    def splice(self, text_ids: Tensor, image_tokens: Tensor) -> Tensor:
        """Steps 5-6: embed the text, then put the N image tokens where <image> stands."""
        text = self.embedding(text_ids)                                         # T: [B, m, d]
        rows = []
        for b in range(text_ids.shape[0]):
            k = (text_ids[b] == self.image_token_id).nonzero()[0, 0]            # position of <image>
            rows.append(torch.cat([text[b, :k], image_tokens[b], text[b, k + 1:]]))
        x = torch.stack(rows)                                                   # [B, L, d], L = m - 1 + N
        return x + self.positions[:x.shape[1]]

    def splice_labels(self, text_ids: Tensor, text_labels: Tensor, num_image_tokens: int) -> Tensor:
        """Labels for the spliced sequence: image positions are never trained on."""
        ignore = torch.full((num_image_tokens,), IGNORE, dtype=text_labels.dtype, device=text_labels.device)
        rows = []
        for b in range(text_ids.shape[0]):
            k = (text_ids[b] == self.image_token_id).nonzero()[0, 0]
            rows.append(torch.cat([text_labels[b, :k], ignore, text_labels[b, k + 1:]]))
        return torch.stack(rows)                                                # [B, L]

    def forward(self, images: Tensor, text_ids: Tensor, image_tokens: Tensor = None) -> Tensor:
        if image_tokens is None:
            image_tokens = self.encode_image(images)
        x = self.splice(text_ids, image_tokens)                                 # X0: [B, L, d]
        L = x.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()[None, None]  # [1, 1, L, L]
        for layer in self.layers:
            x = layer(x, causal)                                                # step 7
        return self.vocab_projection(self.norm(x))                              # step 8: [B, L, V]


def answer_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Step 9: cross-entropy averaged over the answer positions only."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE)


def set_stage(model: VLM, stage: int) -> None:
    """Step 10: which parameters are updated. The vision encoder stays frozen throughout.

    0 = the LLM alone (stands in for language pretraining), 1 = the projector alone (alignment),
    2 = projector and LLM (instruction tuning).
    """
    for parameter in model.parameters():
        parameter.requires_grad = stage in (0, 2)
    for parameter in model.vision.parameters():
        parameter.requires_grad = False
    for parameter in model.projector.parameters():
        parameter.requires_grad = stage in (1, 2)


@torch.no_grad()
def generate(model: VLM, images: Tensor, prompt_ids: Tensor, eos_id: int = 2, pad_id: int = 0,
             max_new_tokens: int = 8) -> Tensor:
    """Step 11: encode the image once, then append argmax tokens until every row produced EOS."""
    image_tokens = model.encode_image(images)
    ids = prompt_ids
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        next_id = model(images, ids, image_tokens)[:, -1].argmax(dim=-1)        # last row of Z only
        next_id = torch.where(finished, pad_id, next_id)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        finished |= next_id == eos_id
        if finished.all():
            break
    return ids
# END vlm-model


# BEGIN vlm-demo
VOCAB = ["<pad>", "<bos>", "<eos>", "<image>", "what", "is", "this", "?", "a", "cat", "dog", "photo", "of"]


def toy_batch():
    """Two 8x8 images that differ in which patch is bright, the same question, different answers."""
    cat, dog = torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)
    cat[:, :4, :4] = 1.0                                   # top-left patch bright     -> "a cat"
    dog[:, 4:, 4:] = 1.0                                   # bottom-right patch bright -> "a dog"
    full = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 2],      # BOS <image> what is this ? a cat EOS
                         [1, 3, 4, 5, 6, 7, 8, 10, 2]])    # BOS <image> what is this ? a dog EOS
    text_ids, next_ids = full[:, :-1], full[:, 1:].clone() # input and next token at every position
    next_ids[:, :-3] = IGNORE                              # only the answer (a, cat/dog, EOS) is a label
    captions = torch.tensor([[11, 12, 8, 9],               # photo of a cat
                             [11, 12, 8, 10]])             # photo of a dog
    return torch.stack([cat, dog]), text_ids, next_ids, captions


def words(ids) -> str:
    return " ".join(VOCAB[i] for i in ids)


def train(model: VLM, stage: int, forward, labels: Tensor, steps: int, lr: float = 1e-2):
    """Minimize the answer loss over the parameters that the stage leaves trainable."""
    set_stage(model, stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    before = answer_loss(forward(), labels).item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = answer_loss(forward(), labels)
        loss.backward()                                    # gradients pass through the frozen parts
        optimizer.step()
    return sum(parameter.numel() for parameter in trainable), before, loss.item()


def main() -> None:
    torch.manual_seed(0)
    images, text_ids, text_labels, captions = toy_batch()
    model = VLM(vocab_size=len(VOCAB), image_token_id=3)
    I, t = images[:1], text_ids[:1]                        # the cat example, B = 1
    print("t =", t[0].tolist(), "=", words(t[0]), "\n")

    def show(step: str, name: str, tensor: Tensor) -> Tensor:
        print(f"step {step:<3} {name:<34} {list(tensor.shape)}")
        return tensor

    show("0", "I (channel-first)", I)
    show("1", "X_p = patchify(I)", patchify(I, 4))
    show("2", "Z0 = X_p W_E + b_E + P_vis", model.vision.embed(I))
    Z = show("3", "Z = ViT(Z0)", model.vision(I))
    show("3", "    attention weights, per head", model.vision.layers[-1].attention.attention_weights)
    U = show("4", "U = projector(Z)", model.projector(Z))
    show("5", "T = E[t]", model.embedding(t))
    show("6", "X0 = splice(T, U) + P_txt", model.splice(t, U))
    logits = model(I, t)
    show("7", "    causal weights, per head", model.layers[-1].attention.attention_weights)
    show("8", "Z_out = LN(X) W_vocab", logits)

    labels = model.splice_labels(text_ids, text_labels, U.shape[1])             # both examples: [2, 11]
    print("\nlabels y =", labels[0].tolist())
    print(f"step 9   loss over positions 8, 9, 10 = {answer_loss(logits, labels[:1]).item():.4f}  (untrained)\n")

    total = sum(parameter.numel() for parameter in model.parameters())
    with_words = lambda: model(None, text_ids, model.embedding(captions))       # a caption fills the image slot
    with_images = lambda: model(images, text_ids)                               # projected patches fill it
    for stage, forward, steps, note in ((0, with_words, 300, "LLM, caption words in the image slot"),
                                        (1, with_images, 300, "projector only, real image tokens"),
                                        (2, with_images, 100, "projector + LLM")):
        count, before, after = train(model, stage, forward, labels, steps)
        print(f"step 10  stage {stage} ({note}): {count} of {total} parameters, loss {before:.3f} -> {after:.3f}")

    model.eval()
    prompt = text_ids[:, :6]                               # BOS <image> what is this ?
    for name, ids in zip(("cat image", "dog image"), generate(model, images, prompt)):
        print(f"step 11  {name}: {words(ids.tolist())}")


if __name__ == "__main__":
    main()
# END vlm-demo
