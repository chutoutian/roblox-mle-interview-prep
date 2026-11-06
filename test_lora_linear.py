"""Deterministic checks for lora_linear.py, the Chapter 4 walkthrough.

Run: .venv-transformer/bin/python -m unittest -v test_lora_linear
"""

import unittest

import torch
from torch import nn

from lora_linear import SRC, TGT, LoRALinear, add_lora, count_parameters, example, merge_all, train
from transformer_encoder_decoder import Transformer


class LoRALinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = nn.Linear(8, 6, bias=False).double()
        self.layer = LoRALinear(self.base, r=2, alpha=4.0).double()
        self.x = torch.randn(5, 8, dtype=torch.float64)

    def test_starts_exactly_at_the_pretrained_layer(self):
        self.assertTrue(torch.equal(self.layer.up, torch.zeros(2, 6, dtype=torch.float64)))
        torch.testing.assert_close(self.layer(self.x), self.base(self.x))
        self.assertEqual(self.layer.scale, 2.0)

    def test_only_down_and_up_receive_gradients_and_up_moves_first(self):
        self.layer(self.x).sum().backward()
        self.assertIsNone(self.base.weight.grad)
        self.assertFalse(self.base.weight.requires_grad)
        self.assertTrue(torch.equal(self.layer.down.grad, torch.zeros_like(self.layer.down)))   # W_up = 0 kills it
        self.assertGreater(self.layer.up.grad.abs().sum().item(), 0)

    def test_gradients_match_the_formulas_of_step_6(self):
        with torch.no_grad():
            self.layer.up.copy_(torch.randn(2, 6, dtype=torch.float64))
        x = self.x.clone().requires_grad_(True)
        G = torch.randn(5, 6, dtype=torch.float64)                    # dL/dy for L = sum(y * G)
        (self.layer(x) * G).sum().backward()
        s, down, up, W = self.layer.scale, self.layer.down, self.layer.up, self.base.weight.T   # W_Q as [d, d]
        torch.testing.assert_close(up.grad, s * (x @ down).T.detach() @ G)
        torch.testing.assert_close(down.grad, s * x.T.detach() @ (G @ up.T.detach()))
        torch.testing.assert_close(x.grad, G @ W.T + s * (G @ up.T.detach()) @ down.T.detach())

    def test_the_update_has_rank_at_most_r(self):
        with torch.no_grad():
            self.layer.up.copy_(torch.randn(2, 6, dtype=torch.float64))
        self.assertEqual(tuple(self.layer.delta_w().shape), (8, 6))
        self.assertLessEqual(torch.linalg.matrix_rank(self.layer.delta_w()).item(), 2)

    def test_merge_is_exact_once_and_unmerge_restores_the_weight(self):
        with torch.no_grad():
            self.layer.up.copy_(torch.randn(2, 6, dtype=torch.float64))
        expected, weight = self.layer(self.x), self.base.weight.clone()
        self.layer.merge()
        self.layer.merge()                                            # a second merge must not add again
        torch.testing.assert_close(self.layer(self.x), expected)
        torch.testing.assert_close(self.base.weight, weight + self.layer.delta_w().T)
        self.layer.unmerge()
        torch.testing.assert_close(self.base.weight, weight)
        torch.testing.assert_close(self.layer(self.x), expected)

    def test_parameter_count_is_r_times_two_d(self):
        trainable, total = count_parameters(self.layer)
        self.assertEqual(trainable, 2 * (8 + 6))
        self.assertEqual(total, 8 * 6 + 2 * (8 + 6))
        big = LoRALinear(nn.Linear(4096, 4096, bias=False), r=16, alpha=32.0)
        self.assertEqual(count_parameters(big)[0], 16 * (4096 + 4096))            # 131,072 of 16.8M


class ToyModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = Transformer(len(SRC), len(TGT))
        self.old = [example("we watch a short video", "nous regardons une courte video"),
                    example("you watch video", "tu regardes video")]
        self.new = example("we watch video", "nous regardons video")

    def test_add_lora_freezes_everything_else_and_wraps_the_attention_projections(self):
        adapters = add_lora(self.model, r=2, alpha=4.0)
        self.assertEqual(len(adapters), 12)                           # 3 attention modules x (Q, K, V, O)
        trainable, total = count_parameters(self.model)
        self.assertEqual(trainable, 12 * 2 * (8 + 8))
        self.assertEqual(total, 2194 + trainable)
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.endswith((".down", ".up")), name)

    def test_lora_learns_the_new_pair_while_the_frozen_weights_stay_bitwise_equal(self):
        train(self.model.train(), self.old, 400)
        add_lora(self.model, r=2, alpha=4.0)
        frozen = {name: p.clone() for name, p in self.model.named_parameters() if not p.requires_grad}
        s, t, y = self.new
        before = self.model.eval()(s, t).detach()
        loss = train(self.model.train(), [self.new] + self.old, 300)
        self.assertLess(loss, 0.05)
        self.assertTrue(all(torch.equal(p, frozen[name]) for name, p in self.model.named_parameters() if name in frozen))
        after = self.model.eval()(s, t)
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(after[0].argmax(-1).tolist(), y[0].tolist())  # teacher-forced next tokens right
        merge_all(self.model)
        torch.testing.assert_close(self.model(s, t), after)


if __name__ == "__main__":
    unittest.main()
