"""Deterministic checks for transformer_encoder_decoder.py, the Chapter 4 model.

Run: .venv-transformer/bin/python -m unittest -v test_transformer_encoder_decoder
"""

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from transformer_encoder_decoder import (
    MultiHeadAttention,
    Transformer,
    angles_1d,
    angles_2d,
    angles_3d,
    greedy_decode,
    rotary_scores,
    rotate,
    sinusoidal_positions,
    speeds,
)

PAD, BOS, EOS = 0, 1, 2


def make_model(**options) -> Transformer:
    return Transformer(src_vocab=8, tgt_vocab=10, d_model=8, num_heads=2, d_ff=32, **options).eval()


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_a_per_head_loop_written_from_the_equations(self):
        attention = MultiHeadAttention(8, 2).double()
        query = torch.randn(2, 3, 8, dtype=torch.float64)
        key = torch.randn(2, 5, 8, dtype=torch.float64)
        value = torch.randn(2, 5, 8, dtype=torch.float64)
        mask = torch.ones(2, 1, 3, 5, dtype=torch.bool)
        mask[0, :, :, 4] = False   # example 0: the last key is padding
        mask[1, :, 0, 1:] = False  # example 1: query 0 may read key 0 only
        actual = attention(query, key, value, mask)

        # head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i, written with loops and dot products.
        q, k, v = attention.w_q(query), attention.w_k(key), attention.w_v(value)
        expected = torch.zeros(2, 3, 8, dtype=torch.float64)
        for b in range(2):
            for j in range(3):
                for i in range(2):
                    f = slice(4 * i, 4 * i + 4)  # head i owns projected features 4i .. 4i+3
                    allowed = [kk for kk in range(5) if mask[b, 0, j, kk]]
                    scores = torch.stack([q[b, j, f] @ k[b, kk, f] for kk in allowed]) / math.sqrt(4)
                    weights = torch.softmax(scores, dim=0)
                    expected[b, j, f] = sum(w * v[b, kk, f] for w, kk in zip(weights, allowed))
        torch.testing.assert_close(actual, attention.w_o(expected))

        weights = attention.attention_weights
        self.assertEqual(tuple(weights.shape), (2, 2, 3, 5))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 2, 3, dtype=torch.float64))
        self.assertTrue((weights[~mask.expand_as(weights)] == 0).all())

    def test_matches_torch_scaled_dot_product_attention(self):
        attention = MultiHeadAttention(8, 2)
        query, memory = torch.randn(2, 4, 8), torch.randn(2, 5, 8)
        mask = torch.ones(2, 1, 4, 5, dtype=torch.bool)
        mask[1, :, :, 3:] = False
        actual = attention(query, memory, memory, mask)
        q = attention.w_q(query).view(2, 4, 2, 4).transpose(1, 2)
        k = attention.w_k(memory).view(2, 5, 2, 4).transpose(1, 2)
        v = attention.w_v(memory).view(2, 5, 2, 4).transpose(1, 2)
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # True = take part
        torch.testing.assert_close(actual, attention.w_o(reference.transpose(1, 2).reshape(2, 4, 8)))
        self.assertEqual(tuple(attention.attention_weights.shape), (2, 2, 4, 5))

    def test_output_length_follows_the_queries(self):
        attention = MultiHeadAttention(8, 2)
        output = attention(torch.randn(1, 4, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 8))
        self.assertEqual(tuple(output.shape), (1, 4, 8))
        self.assertEqual(tuple(attention.attention_weights.shape), (1, 2, 4, 5))


class RotaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def block_rotation(self, angles) -> torch.Tensor:
        """The matrix with one 2x2 rotation per feature pair on its diagonal."""
        R = torch.zeros(2 * len(angles), 2 * len(angles), dtype=torch.float64)
        for j, angle in enumerate(angles):
            R[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=torch.float64)
        return R

    def score(self, q, k, q_angle, k_angle) -> float:
        return (rotate(q, q_angle) @ rotate(k, k_angle).T).item()

    def test_speeds_are_the_frequencies_of_the_sinusoidal_encoding(self):
        torch.testing.assert_close(speeds(4), torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float64))
        for d_k in (4, 8, 64):
            expected = torch.tensor([10000 ** (-2 * j / d_k) for j in range(d_k // 2)], dtype=torch.float64)
            torch.testing.assert_close(speeds(d_k // 2), expected)

    def test_1d_matches_the_block_rotation_matrices(self):
        x = torch.randn(6, 8, dtype=torch.float64)
        rotated = rotate(x, angles_1d(torch.arange(6), 8))
        for p in range(6):
            R_p = self.block_rotation([p * 10000 ** (-2 * j / 8) for j in range(4)])
            torch.testing.assert_close(rotated[p], R_p @ x[p])

    def test_keeps_the_length_and_leaves_position_zero_alone(self):
        x = torch.randn(2, 3, 5, 4, dtype=torch.float64)          # [B, h, L, d_k]
        rotated = rotate(x, angles_1d(torch.arange(5), 4))
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
        torch.testing.assert_close(rotated[..., 0, :], x[..., 0, :])
        self.assertFalse(torch.allclose(rotated[..., 1, :], x[..., 1, :]))

    def test_1d_score_depends_on_the_offset_only(self):
        q, k = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
        at = lambda p, p2: self.score(q, k, angles_1d(torch.tensor([p]), 8), angles_1d(torch.tensor([p2]), 8))
        self.assertAlmostEqual(at(3, 1), at(103, 101), places=9)           # same offset 2
        self.assertAlmostEqual(at(3, 1), at(4, 2), places=9)
        self.assertNotAlmostEqual(at(3, 1), at(3, 2), places=3)            # another offset
        R = self.block_rotation([(1 - 3) * 10000 ** (-2 * j / 8) for j in range(4)])
        self.assertAlmostEqual(at(3, 1), (q[0] @ R @ k[0]).item(), places=9)   # q^T R_{p'-p} k

    def test_one_pair_contributes_a_cosine_of_the_offset(self):
        a = torch.tensor([[1.5, 0.5]], dtype=torch.float64)
        b = torch.tensor([[-0.3, 2.0]], dtype=torch.float64)
        phi_a, phi_b = math.atan2(0.5, 1.5), math.atan2(2.0, -0.3)
        for p, p2 in ((0, 0), (3, 1), (2, 7)):
            actual = self.score(a, b, angles_1d(torch.tensor([p]), 2), angles_1d(torch.tensor([p2]), 2))
            expected = a.norm().item() * b.norm().item() * math.cos(phi_a - phi_b + (p - p2) * 1.0)
            self.assertAlmostEqual(actual, expected, places=9)

    def test_worked_example_of_the_chapter(self):
        angle = angles_1d(torch.arange(5), 4)                      # we watch a short video
        torch.testing.assert_close(angle, torch.tensor(
            [[0, 0], [1, 0.01], [2, 0.02], [3, 0.03], [4, 0.04]], dtype=torch.float64))
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)      # same content for q and k
        for (p, p2), expected in (((3, 3), 2.0), ((2, 1), 1.5403), ((3, 1), 0.5837), ((4, 2), 0.5837)):
            self.assertAlmostEqual(self.score(x, x, angle[p:p + 1], angle[p2:p2 + 1]), expected, places=4)

    def test_2d_angles_of_the_patch_grid(self):
        h, w = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])      # the 2x2 grid of Chapter 3
        torch.testing.assert_close(angles_2d(h, w, 4), torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64))
        angle = angles_2d(torch.tensor([2, 2]), torch.tensor([3, 5]), 8)   # same row, other column
        torch.testing.assert_close(angle[0, :2], angle[1, :2])             # the row pairs did not move
        self.assertFalse(torch.allclose(angle[0, 2:], angle[1, 2:]))

    def test_2d_worked_example_of_the_chapter(self):
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)      # same content for every patch
        at = lambda h, w: angles_2d(torch.tensor([h]), torch.tensor([w]), 4)
        for (query, key), expected in ((((1, 1), (1, 1)), 2.0), (((1, 1), (0, 1)), 1.5403), (((1, 1), (1, 0)), 1.5403),
                                       (((1, 0), (0, 1)), 1.0806), (((1, 1), (0, 0)), 1.0806)):
            self.assertAlmostEqual(self.score(x, x, at(*query), at(*key)), expected, places=4)

    def test_2d_score_depends_on_the_row_and_column_offsets(self):
        q, k = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
        def at(query, key):
            (h, w), (h2, w2) = query, key
            return self.score(q, k, angles_2d(torch.tensor([h]), torch.tensor([w]), 8),
                              angles_2d(torch.tensor([h2]), torch.tensor([w2]), 8))
        self.assertAlmostEqual(at((1, 0), (0, 0)), at((8, 5), (7, 5)), places=9)     # one row down
        self.assertAlmostEqual(at((1, 0), (0, 1)), at((4, 6), (3, 7)), places=9)     # the same diagonal
        self.assertNotAlmostEqual(at((0, 1), (0, 0)), at((1, 0), (0, 1)), places=3)  # neighbour vs diagonal
        self.assertNotAlmostEqual(at((1, 0), (0, 0)), at((0, 1), (0, 0)), places=3)  # down vs right

    def test_3d_text_tokens_are_rotated_as_in_1d(self):
        p = torch.arange(7)
        torch.testing.assert_close(angles_3d(p, p, p, 12), angles_1d(p, 12))

    def test_3d_every_coordinate_gets_a_fast_and_a_slow_pair(self):
        theta = speeds(6)
        angle = angles_3d(torch.tensor([2]), torch.tensor([1]), torch.tensor([0]), 12)   # frame 2, row 1, column 0
        expected = torch.stack([2 * theta[0], 1 * theta[1], 0 * theta[2], 2 * theta[3], 1 * theta[4], 0 * theta[5]])
        torch.testing.assert_close(angle[0], expected)
        self.assertAlmostEqual(theta[0].item(), 1.0)       # frame coordinate: fast pair 0 ...
        self.assertAlmostEqual(theta[3].item(), 0.01)      # ... and slow pair 3

    def test_3d_score_depends_on_the_three_offsets(self):
        q, k = torch.randn(1, 12, dtype=torch.float64), torch.randn(1, 12, dtype=torch.float64)
        def at(query, key):
            one = lambda v: torch.tensor([v])
            return self.score(q, k, angles_3d(*map(one, query), 12), angles_3d(*map(one, key), 12))
        self.assertAlmostEqual(at((5, 1, 2), (3, 0, 2)), at((9, 4, 7), (7, 3, 7)), places=9)
        self.assertNotAlmostEqual(at((5, 1, 2), (3, 0, 2)), at((5, 1, 2), (4, 0, 2)), places=3)

    def test_rotary_scores_are_shift_invariant_for_equal_content(self):
        q = torch.randn(1, 2, 1, 4).expand(1, 2, 5, 4)             # the same query content at 5 positions
        k = torch.randn(1, 2, 1, 4).expand(1, 2, 6, 4)             # the same key content at 6 positions
        scores = rotary_scores(q, k, angles_1d(torch.arange(5), 4), angles_1d(torch.arange(6), 4))
        self.assertEqual(tuple(scores.shape), (1, 2, 5, 6))
        torch.testing.assert_close(scores[..., 1:, 1:], scores[..., :-1, :-1], atol=1e-6, rtol=1e-5)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_follow_the_chapter_table(self):
        model = make_model()
        s, t = torch.tensor([[3, 4, 5, 6, 7]]), torch.tensor([[BOS, 3, 4, 5]])
        memory = model.encode(s)
        logits = model.decode(t, memory, model.source_mask(s))
        self.assertEqual(tuple(model.src_embedding(s).shape), (1, 5, 8))
        self.assertEqual(tuple(model.encoder[0].self_attention.attention_weights.shape), (1, 2, 5, 5))
        self.assertEqual(tuple(model.encoder[0].ffn.linear1(memory).shape), (1, 5, 32))
        self.assertEqual(tuple(memory.shape), (1, 5, 8))
        self.assertEqual(tuple(model.tgt_embedding(t).shape), (1, 4, 8))
        self.assertEqual(tuple(model.decoder[0].self_attention.attention_weights.shape), (1, 2, 4, 4))
        self.assertEqual(tuple(model.decoder[0].cross_attention.attention_weights.shape), (1, 2, 4, 5))
        self.assertEqual(tuple(logits.shape), (1, 4, 10))
        torch.testing.assert_close(model(s, t), logits)

    def test_masks_hide_padding_keys_and_future_keys(self):
        model = make_model()
        src_mask = model.source_mask(torch.tensor([[3, 4, 5, PAD, PAD]]))
        self.assertEqual(tuple(src_mask.shape), (1, 1, 1, 5))
        self.assertEqual(src_mask[0, 0, 0].tolist(), [True, True, True, False, False])
        tgt_mask = model.target_mask(torch.tensor([[BOS, 3, 4, PAD]]))
        self.assertEqual(tuple(tgt_mask.shape), (1, 1, 4, 4))
        self.assertEqual(
            tgt_mask[0, 0].int().tolist(),
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
        )

    def test_future_target_tokens_cannot_change_earlier_logits(self):
        model = make_model()
        s = torch.tensor([[3, 4, 5]])
        short = torch.tensor([[BOS, 3]])
        long = torch.tensor([[BOS, 3, 4, 5]])
        other_future = torch.tensor([[BOS, 3, 7, 8]])
        torch.testing.assert_close(model(s, short), model(s, long)[:, :2])
        torch.testing.assert_close(model(s, long)[:, :2], model(s, other_future)[:, :2])

    def test_source_padding_does_not_change_real_outputs(self):
        model = make_model()
        s = torch.tensor([[3, 4, 5], [6, 7, 3]])
        padded = F.pad(s, (0, 2), value=PAD)
        t = torch.tensor([[BOS, 3, 4], [BOS, 5, 6]])
        torch.testing.assert_close(model.encode(s), model.encode(padded)[:, :3])
        torch.testing.assert_close(model(s, t), model(padded, t))

    def test_cross_attention_ignores_padded_memory_positions(self):
        model = make_model()
        s, t = torch.tensor([[3, 4, PAD, PAD]]), torch.tensor([[BOS, 3, 4]])
        memory, src_mask = model.encode(s), model.source_mask(s)
        altered = memory.clone()
        altered[:, 2:] = 1000 * torch.randn_like(altered[:, 2:])
        torch.testing.assert_close(model.decode(t, memory, src_mask), model.decode(t, altered, src_mask))

    def test_layer_norm_normalizes_each_token_over_its_features(self):
        model = make_model()
        norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]
        self.assertEqual(len(norms), 5)  # encoder: 2, decoder: 3
        x = torch.randn(2, 3, 8)
        x[0, 1] = 30 + 5 * x[0, 1]  # one token with very different statistics
        for norm in norms:
            with self.subTest(norm=norm):
                self.assertEqual(tuple(norm.normalized_shape), (8,))
                mean = x.mean(dim=-1, keepdim=True)
                variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                expected = (x - mean) / torch.sqrt(variance + norm.eps) * norm.weight + norm.bias
                torch.testing.assert_close(norm(x), expected)

    def test_sinusoidal_positions_match_the_formula_and_separate_repeated_tokens(self):
        P = sinusoidal_positions(6, 8)
        for p in range(6):
            for i in range(4):
                angle = p / 10000 ** (2 * i / 8)
                self.assertAlmostEqual(P[p, 2 * i].item(), math.sin(angle), places=5)
                self.assertAlmostEqual(P[p, 2 * i + 1].item(), math.cos(angle), places=5)
        memory = make_model().encode(torch.tensor([[3, 3, 3]]))
        self.assertFalse(torch.allclose(memory[0, 0], memory[0, 1]))

    def test_stacked_layers_and_batches_keep_the_shapes(self):
        model = make_model(num_layers=2)
        self.assertEqual((len(model.encoder), len(model.decoder)), (2, 2))
        s = torch.tensor([[3, 4, 5, 6, 7], [3, 4, PAD, PAD, PAD]])
        t = torch.tensor([[BOS, 3, 4, 5], [BOS, 6, 7, PAD]])
        logits = model(s, t)
        self.assertEqual(tuple(logits.shape), (2, 4, 10))
        self.assertTrue(torch.isfinite(logits).all())

    def test_loss_has_finite_gradients_for_every_parameter(self):
        model = make_model().train()
        s = torch.tensor([[3, 4, 5, 6, 7], [3, 4, PAD, PAD, PAD]])
        t = torch.tensor([[BOS, 3, 4, 5], [BOS, 6, 7, EOS]])
        y = torch.tensor([[3, 4, 5, EOS], [6, 7, EOS, PAD]])
        loss = F.cross_entropy(model(s, t).reshape(-1, 10), y.reshape(-1), ignore_index=PAD)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


class ScriptedModel(nn.Module):
    """Fixes the next-token choice so decoding tests never depend on random weights."""

    pad_id = PAD

    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.encode_calls = 0
        self.prefixes = []
        self.grad_enabled = []

    def source_mask(self, src_ids):
        return (src_ids != PAD)[:, None, None, :]

    def encode(self, src_ids):
        self.encode_calls += 1
        return src_ids.float()[..., None]

    def decode(self, ids, memory, src_mask):
        self.prefixes.append(ids.clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        step = min(len(self.prefixes) - 1, len(self.predictions) - 1)
        logits = torch.full((*ids.shape, 10), -1000.0)
        for row, token in enumerate(self.predictions[step]):
            logits[row, -1, token] = 1000.0
        return logits


class GreedyDecodeTests(unittest.TestCase):
    def test_encodes_once_and_pads_finished_rows_until_every_row_reaches_eos(self):
        model = ScriptedModel([[EOS, 4], [9, 5], [9, EOS]])
        ids = greedy_decode(model, torch.tensor([[3, 4], [5, 6]]), max_new_tokens=6)
        self.assertEqual(ids.tolist(), [[BOS, EOS, PAD, PAD], [BOS, 4, 5, EOS]])
        self.assertEqual(model.encode_calls, 1)
        self.assertEqual([prefix.shape[1] for prefix in model.prefixes], [1, 2, 3])
        for step, prefix in enumerate(model.prefixes, start=1):
            self.assertTrue(torch.equal(prefix, ids[:, :step]))
        self.assertEqual(model.grad_enabled, [False, False, False])

    def test_without_eos_it_stops_at_max_new_tokens(self):
        for limit in (0, 3):
            with self.subTest(limit=limit):
                model = ScriptedModel([[4]])
                ids = greedy_decode(model, torch.tensor([[3, 4]]), max_new_tokens=limit)
                self.assertEqual(ids.tolist(), [[BOS] + [4] * limit])

    def test_real_model_generates_a_bos_started_sequence(self):
        torch.manual_seed(0)
        ids = greedy_decode(make_model(), torch.tensor([[3, 4, 5, 6, 7]]), max_new_tokens=5)
        self.assertEqual(ids[0, 0].item(), BOS)
        self.assertLessEqual(ids.shape[1], 6)
        self.assertFalse(ids.requires_grad)


if __name__ == "__main__":
    unittest.main()
