"""Deterministic checks for vision_transformers.py, the vision transformer section of Chapter 2.

Run: .venv-transformer/bin/python -m unittest -v test_vision_transformers

Every efficient implementation is compared with a direct one: the convolution with a linear
layer on patches, shifted windows with masked global attention in image coordinates, the pooled
keys with an explicit reshape, packed sequences with separate images.
"""

import contextlib
import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from vision_transformers import (
    MAE,
    Attention,
    DINOLoss,
    DiTBlock,
    DividedSpaceTimeBlock,
    FactorizedPositions,
    PatchEmbed,
    PatchMerging,
    SpatialReductionAttention,
    Swin,
    SwinBlock,
    TubeletEmbed,
    ViT,
    attention_cost,
    deit_loss,
    deit_predict,
    ema_update,
    main,
    mae_loss,
    packing_mask,
    random_masking,
    relative_position_index,
    resize_positions,
    unshuffle,
    window_partition,
    window_reverse,
)
from vlm_vit_projector_llm import patchify


def close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol)


class ViTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_patch_embedding_is_a_linear_layer_on_patches(self):
        embed = PatchEmbed(patch=4, channels=3, dim=8)
        images = torch.randn(2, 3, 16, 16)
        # Conv weight [D, C, P, P] -> the matrix E acting on patchify's (row, col, channel) order.
        E = embed.proj.weight.permute(0, 2, 3, 1).reshape(8, -1)
        expected = patchify(images, 4) @ E.T + embed.proj.bias
        self.assertTrue(close(embed(images), expected))

    def test_class_token_and_shapes(self):
        model = ViT(image_size=32, patch=8, dim=16, depth=1, heads=2, classes=5).eval()
        self.assertEqual(list(model.embed(torch.randn(1, 3, 32, 32)).shape), [1, 16, 16])
        self.assertEqual(list(model.positions.shape), [1, 17, 16])                 # N + 1 rows
        self.assertEqual(list(model(torch.randn(3, 3, 32, 32)).shape), [3, 5])

    def test_position_resize(self):
        positions = torch.randn(1, 1 + 14 * 14, 8)
        self.assertIs(resize_positions(positions, 1, (14, 14)), positions)          # same grid: untouched
        bigger = resize_positions(positions, 1, (24, 24))
        self.assertEqual(list(bigger.shape), [1, 1 + 24 * 24, 8])
        self.assertTrue(torch.equal(bigger[:, 0], positions[:, 0]))                  # class row kept
        constant = torch.ones(1, 1 + 4, 3)                                           # a flat table stays flat
        self.assertTrue(close(resize_positions(constant, 1, (6, 6)), torch.ones(1, 37, 3)))
        model = ViT(image_size=32, patch=8, dim=16, depth=1, heads=2, classes=5).eval()
        self.assertEqual(list(model(torch.randn(1, 3, 48, 48)).shape), [1, 5])     # fine-tune resolution

    def test_deit_two_heads_loss_and_prediction(self):
        model = ViT(image_size=32, patch=8, dim=16, depth=1, heads=2, classes=5, distill=True).eval()
        self.assertEqual(list(model.positions.shape), [1, 18, 16])                 # N + 2 rows
        logits_cls, logits_dist = model(torch.randn(2, 3, 32, 32))
        labels, teacher = torch.tensor([1, 2]), torch.randn(2, 5)
        expected = 0.5 * F.cross_entropy(logits_cls, labels) + 0.5 * F.cross_entropy(logits_dist, teacher.argmax(-1))
        self.assertTrue(close(deit_loss(logits_cls, logits_dist, labels, teacher), expected))
        images = torch.randn(2, 3, 32, 32)
        logits_cls, logits_dist = model(images)
        expected = (logits_cls.softmax(-1) + logits_dist.softmax(-1)) / 2                # both heads count
        self.assertTrue(close(deit_predict(model, images), expected))

    def test_registers_join_attention_but_not_the_output(self):
        torch.manual_seed(1)
        model = ViT(image_size=32, patch=8, dim=16, depth=1, heads=2, classes=5, registers=4).eval()
        images = torch.randn(1, 3, 32, 32)
        before = model(images)
        with torch.no_grad():
            model.registers.add_(torch.randn_like(model.registers))                  # registers change attention
        self.assertFalse(close(model(images), before))
        self.assertEqual(list(model.positions.shape), [1, 17, 16])                 # no positions for registers
        with torch.no_grad():
            z = torch.cat([model.cls, model.embed(images)], dim=1) + model.positions
            z = torch.cat([z[:, :1], model.registers, z[:, 1:]], dim=1)            # after x_cls, no positions
            for block in model.blocks:
                z = block(z)
            self.assertTrue(close(model(images), model.head(model.norm(z)[:, 0])))  # the head reads x_cls only


def reference_window_mix(block: SwinBlock, h, H, W):
    """(S)W-MSA without roll or window_partition: global attention over all H*W tokens where a
    pair may attend only if both tokens lie in the same (shifted) window of the image, and the
    bias is looked up from their true offset. Window of row y: floor((y - s) / M)."""
    M = block.M
    s = block.shift if min(H, W) > M else 0
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    y, x = ys.flatten(), xs.flatten()
    wy, wx = torch.div(y - s, M, rounding_mode="floor"), torch.div(x - s, M, rounding_mode="floor")
    same = (wy[:, None] == wy[None, :]) & (wx[:, None] == wx[None, :])
    dy = (y[:, None] - y[None, :]).clamp(1 - M, M - 1) + M - 1                       # clamped pairs are masked
    dx = (x[:, None] - x[None, :]).clamp(1 - M, M - 1) + M - 1
    bias = block.table[:, dy * (2 * M - 1) + dx]                                       # [heads, N, N]
    return block.attn(h, bias=bias, mask=same[None, None])


class SwinTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_partition_round_trip(self):
        x = torch.randn(2, 14, 21, 5)
        windows = window_partition(x, 7)
        self.assertEqual(list(windows.shape), [2 * 2 * 3, 49, 5])
        self.assertTrue(torch.equal(window_reverse(windows, 7, 14, 21), x))
        self.assertTrue(torch.equal(windows[1].view(7, 7, 5), x[0, :7, 7:14]))       # window (0, 1) of image 0

    def test_relative_position_index(self):
        M = 7
        index = relative_position_index(M)
        self.assertEqual(list(index.shape), [49, 49])
        self.assertEqual((int(index.min()), int(index.max())), (0, (2 * M - 1) ** 2 - 1))
        centre = (M - 1) * (2 * M - 1) + (M - 1)                                       # offset (0, 0)
        self.assertTrue(torch.equal(index.diagonal(), torch.full((49,), centre)))
        self.assertTrue(torch.equal(index + index.T, torch.full((49, 49), 2 * centre)))  # offset -> -offset
        self.assertEqual(int(index[0, 1]), centre - 1)                                 # (0,0) vs (0,1): dx = -1

    def test_window_attention_matches_masked_global_attention(self):
        for shift in (0, 3):
            block = SwinBlock(dim=8, heads=2, M=7, shift=shift)
            h = torch.randn(2, 14 * 14, 8)
            self.assertTrue(close(block.mix(h, 14, 14), reference_window_mix(block, h, 14, 14)), f"shift {shift}")

    def test_shift_lets_windows_exchange_information(self):
        plain, shifted = SwinBlock(8, 2, 7, 0), SwinBlock(8, 2, 7, 3)
        shifted.load_state_dict(plain.state_dict())
        h = torch.randn(1, 14 * 14, 8)
        bumped = h.clone()
        bumped[0, 6 * 14 + 6] += torch.randn(8)                                       # token (6, 6): last of window (0, 0)
        reached = lambda block: (block.mix(bumped, 14, 14) - block.mix(h, 14, 14)).abs().sum(-1)[0] > 1e-6
        self.assertFalse(bool(reached(plain)[7 * 14 + 7]))                            # (7, 7) is in window (1, 1)
        self.assertTrue(bool(reached(shifted)[7 * 14 + 7]))                           # the shifted window holds both

    def test_one_window_covering_the_map_is_global_attention(self):
        block = SwinBlock(dim=8, heads=2, M=7, shift=3)                               # shift dropped at 7 x 7
        h = torch.randn(1, 49, 8)
        bias = block.table[:, block.index]
        self.assertTrue(close(block.mix(h, 7, 7), block.attn(h, bias=bias)))

    def test_patch_merging_order(self):
        merge = PatchMerging(dim=2)
        x = torch.randn(1, 4 * 4, 2)
        grid = x.view(1, 4, 4, 2)
        u, v = 1, 0
        concat = torch.cat([grid[0, 2 * u, 2 * v], grid[0, 2 * u + 1, 2 * v], grid[0, 2 * u, 2 * v + 1], grid[0, 2 * u + 1, 2 * v + 1]])
        expected = merge.reduction(merge.norm(concat))
        out = merge(x, 4, 4)
        self.assertEqual(list(out.shape), [1, 4, 4])
        self.assertTrue(close(out[0, u * 2 + v], expected))

    def test_pyramid(self):
        features = Swin(dim=8, depths=(2, 2, 2, 2), heads=(1, 2, 4, 8))(torch.randn(1, 3, 224, 224))
        self.assertEqual([list(f.shape) for f in features], [[1, 3136, 8], [1, 784, 16], [1, 196, 32], [1, 49, 64]])


class PVTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_no_reduction_is_plain_attention(self):
        sra = SpatialReductionAttention(dim=8, heads=2, R=1)
        x = torch.randn(1, 16, 8)
        self.assertTrue(close(sra(x, 4, 4), sra.attn(x)))

    def test_reduction_is_reshape_then_linear(self):
        R, C, H, W = 2, 4, 4, 6
        sra = SpatialReductionAttention(dim=C, heads=1, R=R)
        x = torch.randn(1, H * W, C)
        blocks = x.view(1, H // R, R, W // R, R, C).permute(0, 1, 3, 2, 4, 5)          # [1, H/R, W/R, R, R, C]
        rows = blocks.reshape(1, (H // R) * (W // R), R * R * C)                       # Reshape(x, R)
        W_S = sra.sr.weight.permute(0, 2, 3, 1).reshape(C, R * R * C)                  # conv weight as a matrix
        keys = sra.norm(rows @ W_S.T + sra.sr.bias)                                    # SR(x) = LN(Reshape(x, R) W_S)
        self.assertEqual(keys.shape[1], H * W // R ** 2)
        self.assertTrue(close(sra(x, H, W), sra.attn(x, context=keys)))


class MAETests(unittest.TestCase):
    def test_masking_keeps_a_quarter_and_restores_order(self):
        x = torch.arange(2 * 196, dtype=torch.float).view(2, 196, 1).expand(-1, -1, 3).contiguous()
        kept, mask, ids_restore = random_masking(x, 0.75, torch.Generator().manual_seed(0))
        self.assertEqual(list(kept.shape), [2, 49, 3])
        self.assertTrue(torch.equal(mask.sum(1), torch.tensor([147.0, 147.0])))
        token = torch.zeros(1, 1, 3)
        full = unshuffle(kept, token, ids_restore)
        visible = mask == 0
        self.assertTrue(torch.equal(full[visible], x[visible]))                        # kept tokens back in place
        self.assertTrue(torch.equal(full[~visible], torch.zeros(int((~visible).sum()), 3)))  # mask tokens elsewhere

    def test_loss_ignores_visible_patches(self):
        torch.manual_seed(0)
        patches, pred = torch.randn(1, 8, 12), torch.randn(1, 8, 12)
        mask = torch.tensor([[1.0, 0, 1, 0, 0, 1, 0, 0]])
        changed = pred.clone()
        changed[0, 1] += 5.0                                                           # a visible patch
        self.assertTrue(close(mae_loss(pred, patches, mask), mae_loss(changed, patches, mask)))
        target = (patches - patches.mean(-1, keepdim=True)) / (patches.var(-1, keepdim=True) + 1e-6) ** 0.5
        expected = ((pred - target) ** 2).mean(-1)[mask.bool()].mean()
        self.assertTrue(close(mae_loss(pred, patches, mask), expected))

    def test_model_trains_on_the_masked_patches(self):
        torch.manual_seed(0)
        model = MAE(image_size=32, patch=8, dim=16, depth=1, heads=2, dec_dim=8, dec_depth=1, dec_heads=2)
        loss, pred, mask = model(torch.randn(2, 3, 32, 32), torch.Generator().manual_seed(0))
        self.assertEqual(list(pred.shape), [2, 16, 8 * 8 * 3])
        self.assertEqual(int(mask.sum()), 2 * 12)
        loss.backward()
        self.assertIsNotNone(model.mask_token.grad)


class DINOTests(unittest.TestCase):
    def test_loss_formula_and_centre(self):
        torch.manual_seed(0)
        student = [torch.randn(3, 5) for _ in range(4)]                                # 2 global + 2 local crops
        teacher = [torch.randn(3, 5) for _ in range(2)]
        loss = DINOLoss(5, tau_s=0.1, tau_t=0.04, center_momentum=0.9)
        value = loss(student, teacher)
        terms = []
        for i, t in enumerate(teacher):
            p_t = F.softmax(t / 0.04, dim=-1)                                          # centre starts at 0
            for j, s in enumerate(student):
                if j != i:
                    terms.append(-(p_t * F.log_softmax(s / 0.1, dim=-1)).sum(-1).mean())
        self.assertEqual(len(terms), 2 * 3)
        self.assertTrue(close(value, torch.stack(terms).mean()))
        self.assertTrue(close(loss.center, 0.1 * torch.cat(teacher).mean(0, keepdim=True)))
        c = torch.randn(1, 5)
        loss.center.copy_(c)                                                           # a non-zero centre is subtracted
        value = loss(student, teacher)
        terms = [-(F.softmax((t - c) / 0.04, dim=-1) * F.log_softmax(s / 0.1, dim=-1)).sum(-1).mean()
                 for i, t in enumerate(teacher) for j, s in enumerate(student) if j != i]
        self.assertTrue(close(value, torch.stack(terms).mean()))
        self.assertTrue(close(loss.center, 0.9 * c + 0.1 * torch.cat(teacher).mean(0, keepdim=True)))

    def test_only_the_student_gets_gradients_and_the_teacher_is_an_ema(self):
        torch.manual_seed(0)
        student, teacher = nn.Linear(4, 6), nn.Linear(4, 6)
        crops = [torch.randn(2, 4) for _ in range(3)]
        DINOLoss(6)([student(c) for c in crops], [teacher(c) for c in crops[:2]]).backward()
        self.assertIsNotNone(student.weight.grad)
        self.assertIsNone(teacher.weight.grad)
        before = teacher.weight.detach().clone()
        ema_update(teacher, student, 0.996)
        self.assertTrue(close(teacher.weight, 0.996 * before + 0.004 * student.weight.detach()))


class VideoTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_tubelets(self):
        video = torch.randn(2, 3, 8, 32, 48)                                          # B = 2, non-square frames
        tube, frame = TubeletEmbed(1, 16, 3, 8), PatchEmbed(16, 3, 8)
        with torch.no_grad():
            frame.proj.weight.copy_(tube.proj.weight[:, :, 0])
            frame.proj.bias.copy_(tube.proj.bias)
        expected = torch.stack([frame(video[:, :, t]) for t in range(8)], dim=1)        # [B, T, N, D], row-major patches
        self.assertTrue(close(tube(video), expected))
        self.assertEqual(list(TubeletEmbed(2, 16, 3, 8)(video).shape), [2, 4, 6, 8])

    def test_time_attention_stays_on_one_patch_position(self):
        block = DividedSpaceTimeBlock(dim=8, heads=2)
        x = torch.randn(1, 4, 6, 8)                                                    # T = 4 frames, N = 6 patches
        bumped = x.clone()
        bumped[0, 0, 2] += torch.randn(8)                                              # frame 0, patch 2 (not a constant: LN removes those)
        changed = (block.mix_time(bumped) - block.mix_time(x)).abs().sum(-1)[0] > 1e-6
        self.assertTrue(bool(changed[:, 2].all()))                                    # every frame of patch 2
        self.assertFalse(bool(changed[:, [0, 1, 3, 4, 5]].any()))                     # no other patch
        changed = (block.mix_space(bumped) - block.mix_space(x)).abs().sum(-1)[0] > 1e-6
        self.assertTrue(bool(changed[0].all()))                                       # every patch of frame 0
        self.assertFalse(bool(changed[1:].any()))                                     # no other frame

    def test_time_then_space_then_mlp(self):
        block = DividedSpaceTimeBlock(dim=8, heads=2)
        x = torch.randn(2, 4, 6, 8)
        z = block.mix_space(block.mix_time(x))                                         # time first, as in TimeSformer
        self.assertTrue(close(block(x), z + block.mlp(block.norm_m(z))))

    def test_divided_attention_costs(self):
        T, N = 8, 196
        self.assertEqual(((T * N) ** 2, T * N * (T + N)), (2_458_624, 319_872))          # the book's 2.46M and 0.32M pairs
        self.assertEqual(round((T * N) ** 2 / (T * N * (T + N)), 1), 7.7)


class NaViTTests(unittest.TestCase):
    def test_packed_sequence_equals_separate_images(self):
        torch.manual_seed(0)
        attn = Attention(dim=8, heads=2)
        a, b = torch.randn(1, 5, 8), torch.randn(1, 3, 8)
        packed = attn(torch.cat([a, b], dim=1), mask=packing_mask([5, 3])[None, None])
        self.assertTrue(close(packed, torch.cat([attn(a), attn(b)], dim=1)))

    def test_factorized_positions(self):
        positions = FactorizedPositions(max_side=16, dim=4)
        table = positions(3, 5)
        self.assertEqual(list(table.shape), [15, 4])
        self.assertTrue(close(table[1 * 5 + 2], positions.phi_y[1] + positions.phi_x[2]))


class DiTTests(unittest.TestCase):
    def test_adaln_zero_starts_as_identity_and_then_learns(self):
        torch.manual_seed(0)
        block = DiTBlock(dim=8, heads=2)
        x, c = torch.randn(2, 16, 8), torch.randn(2, 8)
        self.assertTrue(torch.equal(block(x, c), x))
        block(x, c).pow(2).sum().backward()
        self.assertGreater(float(block.ada[1].weight.grad.abs().sum()), 0.0)          # alpha can move away from 0

    def test_adaln_zero_modulation(self):
        torch.manual_seed(0)
        D = 8
        block = DiTBlock(dim=D, heads=2)
        with torch.no_grad():
            block.ada[1].weight.normal_()
            block.ada[1].bias.normal_()                                                # leave the zero start
        x, c = torch.randn(2, 16, D), torch.randn(2, D)
        p = block.ada(c)[:, None]                                                      # DiT order: shift, scale, gate
        beta1, gamma1, alpha1 = p[..., :D], p[..., D:2 * D], p[..., 2 * D:3 * D]
        beta2, gamma2, alpha2 = p[..., 3 * D:4 * D], p[..., 4 * D:5 * D], p[..., 5 * D:]
        ln = lambda t: F.layer_norm(t, (D,), eps=1e-6)
        h = x + alpha1 * block.attn((1 + gamma1) * ln(x) + beta1)
        self.assertTrue(close(block(x, c), h + alpha2 * block.mlp((1 + gamma2) * ln(h) + beta2)))


class DemoTests(unittest.TestCase):
    def test_cost_formula(self):
        self.assertEqual(attention_cost(3136, 96), (4 * 3136 * 96 ** 2, 2 * 3136 ** 2 * 96))
        self.assertEqual(attention_cost(3136, 96, 7)[1], 2 * 49 * 3136 * 96)

    def test_demo_runs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main()
        text = out.getvalue()
        self.assertIn("Swin    stage 4, 7x7 grid", text)
        self.assertIn("[1, 49, 192]", text)
        self.assertIn("49 of 196", text)
        self.assertRegex(text, r"block at initialization is the identity\s+True")
        self.assertRegex(text, r"packed equals one image at a time\s+True")


if __name__ == "__main__":
    unittest.main()
