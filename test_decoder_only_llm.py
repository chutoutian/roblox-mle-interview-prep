"""Tests for decoder_only_llm.py: every component against its equation, plus causality and the KV cache."""
import math
import unittest

import torch
from torch.nn import functional as F

from decoder_only_llm import (BOS, EOS, SENTENCES, VOCAB, Config, DecoderLM, RMSNorm, SwiGLU, apply_rope,
                              encode, generate, kv_cache_numbers, next_token_loss, rope_angles, train)


class TestComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_rmsnorm_matches_formula_and_ignores_scale(self):
        norm = RMSNorm(16)
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
        x = torch.randn(2, 3, 16)
        expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
        self.assertTrue(torch.allclose(norm(x), expected, atol=1e-6))
        self.assertTrue(torch.allclose(norm(5 * x), norm(x), atol=1e-5))          # scale-invariant per token
        self.assertFalse(torch.allclose(norm(x + 3), norm(x) + 0, atol=1e-3))      # but not shift-invariant

    def test_rope_is_relative_and_keeps_length(self):
        cos, sin = rope_angles(8, 64)
        q, k = torch.randn(8), torch.randn(8)
        def score(p, p2):
            return (apply_rope(q[None], cos[p:p + 1], sin[p:p + 1])[0] @ apply_rope(k[None], cos[p2:p2 + 1], sin[p2:p2 + 1])[0]).item()
        self.assertAlmostEqual(score(2, 5), score(30, 33), places=4)               # same offset, same score
        self.assertNotAlmostEqual(score(2, 5), score(2, 6), places=3)
        x = torch.randn(10, 8)
        rotated = apply_rope(x, cos[:10], sin[:10])
        self.assertTrue(torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(rotated[0], x[0]))                          # position 0 turns by 0
        cos4, sin4 = rope_angles(4, 8, base=100.0)                                 # theta_j = 100^(-2j/4) = 1, 0.1
        self.assertTrue(torch.allclose(cos4[3], torch.cos(3 * torch.tensor([1.0, 0.1]))))
        self.assertTrue(torch.allclose(sin4[3], torch.sin(3 * torch.tensor([1.0, 0.1]))))
        turned = apply_rope(torch.tensor([[1.0, 0.0, 0.0, 1.0]]), cos4[3:4], sin4[3:4])   # counter-clockwise by 3 and 0.3
        self.assertTrue(torch.allclose(turned, torch.tensor([[math.cos(3), math.sin(3), -math.sin(0.3), math.cos(0.3)]]), atol=1e-6))

    def test_swiglu_matches_formula(self):
        ffn = SwiGLU(16, 48)
        x = torch.randn(2, 3, 16)
        expected = (F.silu(x @ ffn.w1.weight.T) * (x @ ffn.w3.weight.T)) @ ffn.w2.weight.T
        self.assertTrue(torch.allclose(ffn(x), expected, atol=1e-6))
        self.assertEqual(sum(p.numel() for p in ffn.parameters()), 3 * 16 * 48)


class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = Config()
        self.attn = DecoderLM(self.cfg).blocks[0].attn
        self.x = torch.randn(2, 6, self.cfg.d_model)
        cos, sin = rope_angles(self.cfg.head_dim, self.cfg.max_seq_len)
        self.cos, self.sin = cos[:6], sin[:6]

    def per_head_loop(self):
        """GQA written head by head from the equations: query head i reads key/value head i // group."""
        a, cfg = self.attn, self.cfg
        group, d_h = cfg.n_heads // cfg.n_kv_heads, cfg.head_dim
        outs = []
        for i in range(cfg.n_heads):
            g = i // group
            q = apply_rope(self.x @ a.wq.weight[i * d_h:(i + 1) * d_h].T, self.cos, self.sin)
            k = apply_rope(self.x @ a.wk.weight[g * d_h:(g + 1) * d_h].T, self.cos, self.sin)
            v = self.x @ a.wv.weight[g * d_h:(g + 1) * d_h].T
            s = q @ k.transpose(-2, -1) / math.sqrt(d_h)
            mask = torch.ones(6, 6, dtype=torch.bool).tril()
            outs.append(F.softmax(s.masked_fill(~mask, float("-inf")), dim=-1) @ v)
        return torch.cat(outs, dim=-1) @ a.wo.weight.T

    def test_matches_per_head_loop(self):
        out, _ = self.attn(self.x, self.cos, self.sin)
        self.assertTrue(torch.allclose(out, self.per_head_loop(), atol=1e-5))

    def test_matches_scaled_dot_product_attention(self):
        a, cfg = self.attn, self.cfg
        B, n = self.x.shape[:2]
        q = apply_rope(a.wq(self.x).view(B, n, cfg.n_heads, -1).transpose(1, 2), self.cos, self.sin)
        k = apply_rope(a.wk(self.x).view(B, n, cfg.n_kv_heads, -1).transpose(1, 2), self.cos, self.sin)
        v = a.wv(self.x).view(B, n, cfg.n_kv_heads, -1).transpose(1, 2)
        group = cfg.n_heads // cfg.n_kv_heads
        ref = F.scaled_dot_product_attention(q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1), is_causal=True)
        out, _ = a(self.x, self.cos, self.sin)
        self.assertTrue(torch.allclose(out, a.wo(ref.transpose(1, 2).reshape(B, n, -1)), atol=1e-5))

    def test_mha_is_the_special_case_h_kv_equal_h(self):
        cfg = Config(n_kv_heads=4)
        attn = DecoderLM(cfg).blocks[0].attn
        self.assertEqual(attn.wk.weight.shape, (16, 16))                          # one K head per query head
        out, (k, v) = attn(self.x, self.cos, self.sin)
        self.assertEqual(k.shape, (2, 4, 6, 4))

    def test_kv_cache_is_smaller_by_h_over_h_kv(self):
        self.assertEqual(kv_cache_numbers(Config(n_kv_heads=4), 7) // kv_cache_numbers(self.cfg, 7), 2)


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = Config()
        self.model = DecoderLM(self.cfg)
        self.ids = encode(SENTENCES[0])

    def test_shapes_and_parameter_count(self):
        logits, caches = self.model(self.ids)
        self.assertEqual(logits.shape, (1, 7, len(VOCAB)))
        self.assertEqual(len(caches), self.cfg.n_layers)
        self.assertEqual(caches[0][0].shape, (1, self.cfg.n_kv_heads, 7, self.cfg.head_dim))
        c, d, dh = self.cfg, self.cfg.d_model, self.cfg.head_dim
        per_block = 2 * d + d * c.n_heads * dh + 2 * d * c.n_kv_heads * dh + c.n_heads * dh * d + 3 * d * c.d_ff
        expected = c.vocab_size * d + c.n_layers * per_block + d + d * c.vocab_size
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), expected)

    def test_causal_future_tokens_do_not_change_the_past(self):
        changed = self.ids.clone()
        changed[0, 5] = VOCAB.index("book")
        a, _ = self.model(self.ids)
        b, _ = self.model(changed)
        self.assertTrue(torch.allclose(a[:, :5], b[:, :5], atol=1e-6))
        self.assertFalse(torch.allclose(a[:, 5:], b[:, 5:], atol=1e-4))

    def test_kv_cache_matches_full_recompute(self):
        full, _ = self.model(self.ids)
        logits, caches = self.model(self.ids[:, :3])                               # prefill three tokens
        steps = [logits]
        logits, caches = self.model(self.ids[:, 3:5], caches, start=3)             # a 2-token chunk: needs the T - n offset
        steps.append(logits)
        for pos in range(5, 7):                                                    # then one token at a time
            logits, caches = self.model(self.ids[:, pos:pos + 1], caches, start=pos)
            steps.append(logits)
        self.assertTrue(torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5))

    def test_loss_is_shifted_cross_entropy(self):
        logits, _ = self.model(self.ids)
        manual = -F.log_softmax(logits[0, :-1], dim=-1)[torch.arange(6), self.ids[0, 1:]].mean()
        self.assertTrue(torch.allclose(next_token_loss(logits, self.ids), manual))

    def test_training_learns_both_sentences_and_generation_uses_context(self):
        history = train(self.model, steps=300)
        self.assertLess(history[-1], 0.13)                       # the floor is ln 2 / 6: "we" or "you" after BOS
        self.assertAlmostEqual(history[-1], math.log(2) / 6, delta=0.01)
        for sentence in SENTENCES:
            prompt = torch.tensor([[BOS, VOCAB.index(sentence[0])]])
            out = generate(self.model, prompt, 8, eos_id=EOS)
            self.assertEqual(out[0].tolist(), encode(sentence)[0].tolist())

    def test_generate_with_cache_equals_greedy_without_cache(self):
        prompt = torch.tensor([[BOS, VOCAB.index("we")]])
        cached = generate(self.model, prompt, 6)
        ids = prompt
        for _ in range(6):
            logits, _ = self.model(ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        self.assertEqual(cached.tolist(), ids.tolist())

    def test_forward_is_embedding_pre_norm_blocks_final_norm_head(self):
        m = self.model
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, RMSNorm):
                    mod.weight.uniform_(0.5, 1.5)          # make attn_norm, ffn_norm and the final norm distinguishable
            n = self.ids.shape[1]
            cos, sin = m.rope_cos[:n], m.rope_sin[:n]
            x = m.embed.weight[self.ids]                    # no position vector, no sqrt(d) scale
            for blk in m.blocks:
                x = x + blk.attn(blk.attn_norm(x), cos, sin)[0]
                x = x + blk.ffn(blk.ffn_norm(x))
            logits, _ = m(self.ids)
        self.assertTrue(torch.allclose(logits, m.lm_head(m.norm(x)), atol=1e-5))

    def test_generate_feeds_each_new_token_at_its_own_position(self):
        calls, forward = [], self.model.forward
        def spy(ids, caches=None, start=0):
            calls.append((ids.shape[1], start))
            return forward(ids, caches, start)
        self.model.forward = spy
        generate(self.model, torch.tensor([[BOS, VOCAB.index("we")]]), 3)
        self.assertEqual(calls, [(2, 0), (1, 2), (1, 3)])   # prefill at 0, then positions 2 and 3; no unused last call

    def test_generate_stops_finished_rows_in_a_batch(self):
        train(self.model, steps=300)
        prompts = torch.tensor([[BOS, VOCAB.index("we"), VOCAB.index("watch"), VOCAB.index("a"), VOCAB.index("short")],
                                [VOCAB.index("we"), VOCAB.index("watch"), VOCAB.index("a"), VOCAB.index("short"), VOCAB.index("video")]])
        out = generate(self.model, prompts, 8, eos_id=EOS)
        for row in out[:, prompts.shape[1]:].tolist():
            self.assertIn(EOS, row)
            self.assertTrue(all(token == EOS for token in row[row.index(EOS):]))   # nothing but eos after the first eos
        self.assertLess(out.shape[1], prompts.shape[1] + 8)                         # stops once every row has finished

    def test_kv_cache_numbers_counts_the_real_cache(self):
        _, caches = self.model(self.ids)
        self.assertEqual(kv_cache_numbers(self.cfg, 7), sum(k.numel() + v.numel() for k, v in caches))

    def test_loss_skips_pad_labels(self):
        logits, _ = self.model(self.ids)
        padded = self.ids.clone()
        padded[0, -1] = 0                                   # replace <eos> by <pad>
        self.assertTrue(torch.allclose(next_token_loss(logits, padded, pad_id=0), next_token_loss(logits[:, :-1], self.ids[:, :-1])))

    def test_tied_embeddings_share_one_matrix(self):
        tied = DecoderLM(Config(tie_embeddings=True))
        self.assertIs(tied.lm_head.weight, tied.embed.weight)
        self.assertEqual(sum(p.numel() for p in tied.parameters()), sum(p.numel() for p in self.model.parameters()) - 16 * len(VOCAB))


if __name__ == "__main__":
    unittest.main()
