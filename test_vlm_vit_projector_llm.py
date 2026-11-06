"""Deterministic checks for vlm_vit_projector_llm.py, the Chapter 3 model.

Run: .venv-transformer/bin/python -m unittest -v test_vlm_vit_projector_llm
"""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from vlm_vit_projector_llm import (
    IGNORE,
    VLM,
    VOCAB,
    PatchEmbedding,
    answer_loss,
    generate,
    patchify,
    set_stage,
    toy_batch,
    train,
)

BOS, EOS, IMAGE = 1, 2, 3
CAT_ANSWER = [1, 3, 4, 5, 6, 7, 8, 9, 2]    # BOS <image> what is this ? a cat EOS
DOG_ANSWER = [1, 3, 4, 5, 6, 7, 8, 10, 2]   # BOS <image> what is this ? a dog EOS


def make_model() -> VLM:
    return VLM(vocab_size=len(VOCAB), image_token_id=IMAGE)


def groups(model: VLM):
    llm = [p for name, p in model.named_parameters() if not name.startswith(("vision.", "projector."))]
    return list(model.vision.parameters()), list(model.projector.parameters()), llm


class PatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_patchify_matches_manual_slicing_in_row_major_order(self):
        images = torch.randn(2, 3, 8, 8)
        patches = patchify(images, 4)
        self.assertEqual(tuple(patches.shape), (2, 4, 48))
        for b in range(2):
            for r in range(2):
                for c in range(2):
                    block = images[b, :, 4 * r:4 * r + 4, 4 * c:4 * c + 4]          # [C, P, P]
                    torch.testing.assert_close(patches[b, 2 * r + c], block.permute(1, 2, 0).reshape(-1))

    def test_patch_embedding_equals_a_convolution_with_kernel_and_stride_p(self):
        embed = PatchEmbedding(image_size=8, patch=4, channels=3, d_vision=8)
        conv = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        with torch.no_grad():
            conv.weight.copy_(embed.linear.weight.view(8, 4, 4, 3).permute(0, 3, 1, 2))
            conv.bias.copy_(embed.linear.bias)
        images = torch.randn(2, 3, 8, 8)
        expected = conv(images).flatten(2).transpose(1, 2) + embed.positions        # [B, N, d_v]
        torch.testing.assert_close(embed(images), expected, atol=1e-6, rtol=1e-5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.images, self.text_ids, self.text_labels, self.captions = toy_batch()
        self.model = make_model().eval()

    def test_shapes_follow_the_chapter_table(self):
        I, t = self.images[:1], self.text_ids[:1]
        Z = self.model.vision(I)
        U = self.model.projector(Z)
        logits = self.model(I, t)
        self.assertEqual(tuple(patchify(I, 4).shape), (1, 4, 48))
        self.assertEqual(tuple(Z.shape), (1, 4, 8))
        self.assertEqual(tuple(self.model.vision.layers[-1].attention.attention_weights.shape), (1, 2, 4, 4))
        self.assertEqual(tuple(U.shape), (1, 4, 16))
        self.assertEqual(tuple(self.model.embedding(t).shape), (1, 8, 16))
        self.assertEqual(tuple(self.model.splice(t, U).shape), (1, 11, 16))
        self.assertEqual(tuple(self.model.layers[-1].attention.attention_weights.shape), (1, 2, 11, 11))
        self.assertEqual(tuple(logits.shape), (1, 11, 13))

    def test_splice_puts_the_image_tokens_where_the_placeholder_stands(self):
        t = self.text_ids[:1]
        U = self.model.encode_image(self.images[:1])
        X0 = self.model.splice(t, U) - self.model.positions[:11]
        E = self.model.embedding(t)
        torch.testing.assert_close(X0[0, 0], E[0, 0])          # BOS
        torch.testing.assert_close(X0[0, 1:5], U[0])           # the four image tokens
        torch.testing.assert_close(X0[0, 5:], E[0, 2:])        # what is this ? a cat
        self.assertFalse(torch.allclose(X0[0, 1], E[0, 1]))    # the <image> row itself is never used

    def test_vision_encoder_is_bidirectional(self):
        changed = self.images[:1].clone()
        changed[:, :, 4:, 4:] += 1.0                           # alter the last patch only
        before, after = self.model.vision(self.images[:1]), self.model.vision(changed)
        self.assertFalse(torch.allclose(before[0, 0], after[0, 0]))   # the first patch saw the change

    def test_llm_is_causal_and_text_reads_the_image_tokens(self):
        t = self.text_ids[:1]
        other = t.clone()
        other[0, 6:] = torch.tensor([11, 12])                  # change the two tokens after '?'
        base = self.model(self.images[:1], t)
        torch.testing.assert_close(self.model(self.images[:1], other)[:, :9], base[:, :9])
        with_dog = self.model(self.images[1:], t)              # same text, other image
        torch.testing.assert_close(with_dog[:, 0], base[:, 0])          # BOS precedes the image
        self.assertFalse(torch.allclose(with_dog[:, 5:], base[:, 5:]))  # every word after it changed

    def test_only_the_answer_positions_are_trained_on(self):
        labels = self.model.splice_labels(self.text_ids, self.text_labels, 4)
        self.assertEqual(labels[0].tolist(), [IGNORE] * 8 + [8, 9, 2])
        self.assertEqual(labels[1].tolist(), [IGNORE] * 8 + [8, 10, 2])
        logits = self.model(self.images, self.text_ids)
        log_p = F.log_softmax(logits, dim=-1)
        manual = -torch.stack([log_p[b, j, labels[b, j]] for b in range(2) for j in (8, 9, 10)]).mean()
        torch.testing.assert_close(answer_loss(logits, labels), manual)
        noisy = logits.clone()
        noisy[:, :8] += torch.randn_like(noisy[:, :8])         # image and question positions
        torch.testing.assert_close(answer_loss(noisy, labels), manual)

    def test_each_stage_updates_only_its_own_parameters(self):
        labels = self.model.splice_labels(self.text_ids, self.text_labels, 4)
        vision, projector, llm = groups(self.model)
        expected = {0: (False, False, True), 1: (False, True, False), 2: (False, True, True)}
        for stage, trains in expected.items():
            with self.subTest(stage=stage):
                set_stage(self.model, stage)
                self.model.zero_grad()
                for parameter in self.model.parameters():
                    parameter.grad = None
                answer_loss(self.model(self.images, self.text_ids), labels).backward()
                for group, trained in zip((vision, projector, llm), trains):
                    self.assertTrue(all((p.grad is not None) == trained for p in group))
                if trains[1]:   # the gradient reached the projector through the LLM layers
                    self.assertTrue(any(p.grad.abs().sum() > 0 for p in projector))

    def test_generate_encodes_the_image_once_and_stops_at_eos(self):
        calls = []
        self.model.vision.register_forward_hook(lambda *args: calls.append(1))
        prompt = self.text_ids[:, :6]
        ids = generate(self.model, self.images, prompt, max_new_tokens=3)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(ids[:, :6], prompt))
        self.assertEqual(ids.shape[1], 9)
        self.assertFalse(ids.requires_grad)


class TrainingTests(unittest.TestCase):
    def test_alignment_lets_a_frozen_llm_read_the_image(self):
        torch.manual_seed(0)
        images, text_ids, text_labels, captions = toy_batch()
        model = make_model()
        labels = model.splice_labels(text_ids, text_labels, 4)
        vision, projector, llm = groups(model)
        snapshot = lambda group: [p.detach().clone() for p in group]
        same = lambda group, saved: all(torch.equal(p, q) for p, q in zip(group, saved))
        vision_start = snapshot(vision)

        # Stage 0: language only. The LLM learns to read the answer from words in the image slot.
        _, _, after = train(model, 0, lambda: model(None, text_ids, model.embedding(captions)), labels, 300)
        self.assertLess(after, 0.05)
        llm_pretrained, projector_start = snapshot(llm), snapshot(projector)
        self.assertTrue(same(projector, projector_start))

        # Stage 1: the projector alone makes the image readable for the frozen LLM.
        count, before, after = train(model, 1, lambda: model(images, text_ids), labels, 300)
        self.assertEqual(count, sum(p.numel() for p in projector))
        self.assertGreater(before, 0.3)
        self.assertLess(after, 0.05)
        self.assertTrue(same(llm, llm_pretrained))
        self.assertFalse(same(projector, projector_start))
        model.eval()
        self.assertEqual(generate(model, images, text_ids[:, :6]).tolist(), [CAT_ANSWER, DOG_ANSWER])
        swapped = generate(model, images.flip(0), text_ids[:, :6]).tolist()
        self.assertEqual(swapped, [DOG_ANSWER, CAT_ANSWER])     # the answer follows the image, not the row

        # Stage 2: projector and LLM together; the vision encoder never moved.
        model.train()
        _, _, after = train(model, 2, lambda: model(images, text_ids), labels, 100)
        self.assertLess(after, 0.01)
        self.assertFalse(same(llm, llm_pretrained))
        self.assertTrue(same(vision, vision_start))


if __name__ == "__main__":
    unittest.main()
