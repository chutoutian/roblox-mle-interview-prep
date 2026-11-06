"""LoRA: a trainable low-rank correction next to a frozen weight, following Chapter 4 equation by equation.

Run:   .venv-transformer/bin/python lora_linear.py
Tests: .venv-transformer/bin/python -m unittest -v test_lora_linear

Notation of Chapter 2: a projection is Q = X W_Q with W_Q of shape [d, d]; with LoRA it becomes
Q = X (W_Q + (alpha / r) W_down W_up). PyTorch's nn.Linear stores W_Q transposed, as weight[d_out, d_in],
which only matters inside merge(). The LoRA paper calls W_down "A" and W_up "B".
"""

# BEGIN lora-imports
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from transformer_encoder_decoder import Transformer, greedy_decode   # the toy model of Chapter 2
# END lora-imports


# BEGIN lora-linear
class LoRALinear(nn.Module):
    """Q = X W_Q + (alpha / r) (X W_down) W_up, with W_Q frozen and only W_down, W_up trainable."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad = False                            # step 2: W_Q (and a bias) frozen
        d_in, d_out = base.in_features, base.out_features
        self.scale = alpha / r                                         # alpha / r
        self.down = nn.Parameter(torch.randn(d_in, r) / math.sqrt(d_in))   # W_down: random, so that W_up gets a gradient
        self.up = nn.Parameter(torch.zeros(r, d_out))                       # W_up: zero, so that Delta W = 0 at the start
        self.merged = False

    def delta_w(self) -> Tensor:
        return self.scale * self.down @ self.up                        # Delta W: [d, d], rank <= r

    def forward(self, x: Tensor) -> Tensor:
        q = self.base(x)                                               # step 2: X W_Q
        if self.merged:
            return q
        return q + self.scale * (x @ self.down) @ self.up              # steps 3-4: X W_down first, then W_up

    @torch.no_grad()
    def merge(self) -> None:
        """Step 8: W_Q <- W_Q + Delta W, once. The layer then costs exactly what a plain Linear costs."""
        if not self.merged:
            self.base.weight += self.delta_w().T                       # weight is stored as [d_out, d_in]
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_w().T
            self.merged = False
# END lora-linear


# BEGIN lora-apply
def add_lora(model: nn.Module, r: int, alpha: float, names=("w_q", "w_k", "w_v", "w_o")) -> list:
    """Step 1: freeze the whole model, then wrap every Linear called `names` (W_Q, W_K, W_V, W_O) in a LoRALinear."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapters = []
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in names:
                adapter = LoRALinear(child, r, alpha)
                setattr(module, name, adapter)
                adapters.append(adapter)
    return adapters


def merge_all(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def count_parameters(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())
# END lora-apply


# BEGIN lora-demo
SRC = ["<pad>", "<bos>", "<eos>", "we", "watch", "a", "short", "video", "you"]
TGT = ["<pad>", "<bos>", "<eos>", "nous", "regardons", "video", "une", "courte", "tu", "regardes"]


def example(source: str, target: str) -> tuple:
    """Source ids, teacher-forcing decoder input (BOS + target) and next-token labels (target + EOS)."""
    s = torch.tensor([[SRC.index(word) for word in source.split()]])
    full = torch.tensor([[1] + [TGT.index(word) for word in target.split()] + [2]])
    return s, full[:, :-1], full[:, 1:]


def train(model: nn.Module, examples: list, steps: int, lr: float = 1e-2) -> float:
    """Minimize the next-token loss of Chapter 2 over whatever parameters are trainable."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = sum(F.cross_entropy(model(s, t).reshape(-1, len(TGT)), y.reshape(-1), ignore_index=0)
                   for s, t, y in examples) / len(examples)
        loss.backward()
        optimizer.step()
    return loss.item()


def translate(model: nn.Module, s: Tensor) -> str:
    return " ".join(TGT[i] for i in greedy_decode(model.eval(), s)[0].tolist()[1:])


def main() -> None:
    torch.manual_seed(2)
    model = Transformer(len(SRC), len(TGT))                             # d = 8, h = 2, d_ff = 32
    old = [example("we watch a short video", "nous regardons une courte video"),
           example("you watch video", "tu regardes video")]
    new = example("we watch video", "nous regardons video")

    loss = train(model.train(), old, 400)                               # stands in for pretraining
    print(f"pretrained on two pairs (all parameters), loss {loss:.3f}")
    for s, _, _ in old + [new]:
        print(f"  {' '.join(SRC[i] for i in s[0].tolist()):24s} -> {translate(model, s)}")

    adapters = add_lora(model, r=2, alpha=4.0)                          # step 1: W_Q, W_K, W_V, W_O of every attention
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    trainable, total = count_parameters(model)
    print(f"LoRA r=2 on {len(adapters)} matrices: {trainable} of {total} parameters trainable "
          f"({100 * trainable / total:.0f}%); each 8x8 matrix gets W_down [8,2] and W_up [2,8]")
    print(f"  before training Delta W = 0, so the new pair still gives: {translate(model, new[0])}")

    loss = train(model.train(), [new] + old, 300)                       # steps 5-7: only W_down and W_up move
    unchanged = all(torch.equal(p, frozen[name]) for name, p in model.named_parameters() if name in frozen)
    print(f"LoRA trained on the new pair plus the two old ones: loss {loss:.3f}, frozen weights unchanged: {unchanged}")
    for s, _, _ in [new] + old:
        print(f"  {' '.join(SRC[i] for i in s[0].tolist()):24s} -> {translate(model, s)}")

    before = model.eval()(new[0], new[1])
    merge_all(model)                                                    # step 8
    print(f"merged: same logits {torch.allclose(before, model(new[0], new[1]), atol=1e-6)}; "
          f"one attention matrix is again a plain {tuple(adapters[0].base.weight.shape)} weight")


if __name__ == "__main__":
    main()
# END lora-demo
