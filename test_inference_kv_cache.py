"""Deterministic checks for inference_kv_cache.py, the Chapter 5 walkthrough.

Run: .venv-transformer/bin/python -m unittest -v test_inference_kv_cache
"""

import unittest

import torch

from inference_kv_cache import (CausalLM, cache_bytes, decode_step, filtered_probabilities, generate, prefill,
                                repetition_penalty, sample_next)


def tiny(seed: int, num_layers: int = 2) -> CausalLM:
    torch.manual_seed(seed)
    return CausalLM(vocab_size=11, d_model=16, num_heads=2, d_ff=32, num_layers=num_layers).double().eval()


class CacheTests(unittest.TestCase):
    def test_decode_with_cache_matches_full_recompute(self):
        model, ids = tiny(0), torch.randint(0, 11, (2, 9))
        with torch.no_grad():
            full, _ = model(ids)
            logits, caches = prefill(model, ids[:, :3])
            rows = [logits]
            for t in range(3, 9):
                logits, caches = decode_step(model, ids[:, t], caches)
                rows.append(logits)
        torch.testing.assert_close(torch.stack(rows, dim=1), full[:, 2:], atol=1e-9, rtol=0)
        self.assertEqual(caches[0]["k"].shape, (2, 2, 9, 8))

    def test_generate_with_cache_equals_without(self):
        model, prompt = tiny(1), torch.randint(0, 11, (1, 3))
        greedy = lambda logits: logits.argmax(dim=-1)
        self.assertTrue(torch.equal(generate(model, prompt, 8, greedy), generate(model, prompt, 8, greedy, use_cache=False)))

    def test_llama3_8b_cache_size(self):
        self.assertEqual(cache_bytes(32, 8, 128, 1), 131072)               # 128 KiB per token
        self.assertEqual(cache_bytes(32, 8, 128, 8192), 2 ** 30)           # 1 GiB at 8192 tokens
        self.assertEqual(cache_bytes(32, 32, 128, 1), 4 * cache_bytes(32, 8, 128, 1))  # LLaMA-2-7B: no GQA


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.z = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])

    def test_limits(self):
        one_hot = torch.tensor([[1.0, 0, 0, 0, 0]])
        self.assertTrue(torch.equal(filtered_probabilities(self.z, temperature=0), one_hot))
        self.assertTrue(torch.equal(filtered_probabilities(self.z, top_k=1), one_hot))
        torch.testing.assert_close(filtered_probabilities(self.z, top_p=1.0), self.z.softmax(-1))
        torch.testing.assert_close(filtered_probabilities(self.z), self.z.softmax(-1))

    def test_worked_example(self):
        torch.testing.assert_close(filtered_probabilities(self.z, temperature=0.5), torch.tensor([[0.829, 0.112, 0.041, 0.015, 0.002]]), atol=1e-3, rtol=0)
        torch.testing.assert_close(filtered_probabilities(self.z, top_k=2), torch.tensor([[0.731, 0.269, 0, 0, 0]]), atol=1e-3, rtol=0)
        torch.testing.assert_close(filtered_probabilities(self.z, top_p=0.9), torch.tensor([[0.579, 0.213, 0.129, 0.078, 0]]), atol=1e-3, rtol=0)
        self.assertEqual(int((filtered_probabilities(self.z, min_p=0.2) > 0).sum()), 3)

    def test_top_p_keeps_the_crossing_token_and_at_least_one(self):
        probs = filtered_probabilities(self.z, top_p=0.6)   # 0.563 < 0.6: the second token is needed to reach p
        self.assertEqual(int((probs > 0).sum()), 2)
        self.assertEqual(int((filtered_probabilities(self.z, top_p=1e-6) > 0).sum()), 1)

    def test_repetition_penalty(self):
        penalised = repetition_penalty(self.z, torch.tensor([[0, 4]]), 2.0)
        torch.testing.assert_close(penalised, torch.tensor([[1.0, 1.0, 0.5, 0.0, -2.0]]))


class SampleNextTests(unittest.TestCase):
    def test_seeded_sampling_is_reproducible_and_respects_truncation(self):
        z = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]]).repeat(200, 1)
        first = sample_next(z, top_k=2, generator=torch.Generator().manual_seed(0))
        second = sample_next(z, top_k=2, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool((first <= 1).all()), 'top-k=2 only ever draws the two largest logits')
        self.assertTrue(torch.equal(sample_next(z, temperature=0), torch.zeros(200, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
