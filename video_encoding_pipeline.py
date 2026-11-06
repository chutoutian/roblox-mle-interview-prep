"""
Interview-style video encoding pipeline (encoder only).

Scope:
- RGB -> YUV420 preprocess
- Block split
- Intra / Inter prediction (simple block matching)
- Residual transform + quantization
- Entropy coding (arithmetic coding on syntax tokens)
- In-loop reconstruction + simple deblock filter
- Basic rate control over QP

This is intentionally simplified and not standard-compliant H.264/HEVC/AV1.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from arithmetic_coding import arithmetic_encode


TOK_FRAME_Y = 10**9 + 1
TOK_FRAME_U = 10**9 + 2
TOK_FRAME_V = 10**9 + 3
TOK_BLOCK_START = 10**9 + 10
TOK_EOB = 10**9 + 11


@dataclass
class EncodedFrame:
    index: int
    frame_type: str
    qp: int
    bits: int
    payload: Dict[str, object]
    mv_count: int


def rgb_to_yuv420(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected frame shape (H, W, 3)")
    x = rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    h, w = y.shape
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        u = np.pad(u, ((0, pad_h), (0, pad_w)), mode="edge")
        v = np.pad(v, ((0, pad_h), (0, pad_w)), mode="edge")
    u420 = 0.25 * (u[0::2, 0::2] + u[0::2, 1::2] + u[1::2, 0::2] + u[1::2, 1::2])
    v420 = 0.25 * (v[0::2, 0::2] + v[0::2, 1::2] + v[1::2, 0::2] + v[1::2, 1::2])
    return y.astype(np.float32), u420.astype(np.float32), v420.astype(np.float32)


def pad_to_block(plane: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = plane.shape
    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    if pad_h == 0 and pad_w == 0:
        return plane, (0, 0)
    return np.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge"), (pad_h, pad_w)


def unpad(plane: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = pad
    if pad_h:
        plane = plane[:-pad_h, :]
    if pad_w:
        plane = plane[:, :-pad_w]
    return plane


_DCT8 = None
_ZZ = None


def dct_mat_8() -> np.ndarray:
    global _DCT8
    if _DCT8 is not None:
        return _DCT8
    n = 8
    c = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        for i in range(n):
            c[k, i] = alpha * np.cos(np.pi * (i + 0.5) * k / n)
    _DCT8 = c
    return c


def dct2(block: np.ndarray) -> np.ndarray:
    c = dct_mat_8()
    return c @ block @ c.T


def idct2(coeff: np.ndarray) -> np.ndarray:
    c = dct_mat_8()
    return c.T @ coeff @ c


def zigzag_indices() -> np.ndarray:
    global _ZZ
    if _ZZ is not None:
        return _ZZ
    order = []
    for s in range(15):
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < 8 and j < 8:
                    order.append(i * 8 + j)
        else:
            for j in range(s + 1):
                i = s - j
                if i < 8 and j < 8:
                    order.append(i * 8 + j)
    _ZZ = np.array(order, dtype=np.int32)
    return _ZZ


def zigzag_scan(block8: np.ndarray) -> np.ndarray:
    return block8.reshape(-1)[zigzag_indices()]


def quant_step_from_qp(qp: int) -> float:
    # Qstep grows roughly exponentially with QP in practical codecs.
    return float(2 ** ((qp - 4) / 6.0))


def intra_predict(recon_plane: np.ndarray, by: int, bx: int, block_size: int) -> np.ndarray:
    if by == 0 and bx == 0:
        return np.full((block_size, block_size), 128.0, dtype=np.float32)
    vals = []
    if bx > 0:
        vals.append(float(np.mean(recon_plane[by : by + block_size, bx - 1])))
    if by > 0:
        vals.append(float(np.mean(recon_plane[by - 1, bx : bx + block_size])))
    pred_val = float(np.mean(vals)) if vals else 128.0
    return np.full((block_size, block_size), pred_val, dtype=np.float32)


def inter_predict(
    cur_block: np.ndarray,
    ref_plane: np.ndarray,
    by: int,
    bx: int,
    block_size: int,
    search_range: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = ref_plane.shape
    y0 = max(0, by - search_range)
    y1 = min(h - block_size, by + search_range)
    x0 = max(0, bx - search_range)
    x1 = min(w - block_size, bx + search_range)

    best_cost = float("inf")
    best_pos = (by, bx)
    for ry in range(y0, y1 + 1):
        for rx in range(x0, x1 + 1):
            cand = ref_plane[ry : ry + block_size, rx : rx + block_size]
            cost = float(np.sum(np.abs(cur_block - cand)))
            if cost < best_cost:
                best_cost = cost
                best_pos = (ry, rx)

    ry, rx = best_pos
    pred = ref_plane[ry : ry + block_size, rx : rx + block_size].astype(np.float32)
    mv = (ry - by, rx - bx)
    return pred, mv


def coeffs_to_tokens(zz: np.ndarray) -> List[int]:
    tokens = [int(zz[0])]  # DC
    run = 0
    for v in zz[1:]:
        value = int(v)
        if value == 0:
            run += 1
            continue
        tokens.append(run)
        tokens.append(value)
        run = 0
    tokens.append(TOK_EOB)
    return tokens


def deblock_filter(plane: np.ndarray, block_size: int) -> np.ndarray:
    out = plane.copy()
    h, w = out.shape
    for x in range(block_size, w, block_size):
        avg = 0.5 * (out[:, x - 1] + out[:, x])
        out[:, x - 1] = avg
        out[:, x] = avg
    for y in range(block_size, h, block_size):
        avg = 0.5 * (out[y - 1, :] + out[y, :])
        out[y - 1, :] = avg
        out[y, :] = avg
    return out


def encode_plane(
    plane: np.ndarray,
    ref_plane: Optional[np.ndarray],
    frame_type: str,
    qp: int,
    block_size: int,
    search_range: int,
) -> Tuple[np.ndarray, List[int], int]:
    padded, pad = pad_to_block(plane, block_size)
    if ref_plane is not None:
        ref_padded, _ = pad_to_block(ref_plane, block_size)
    else:
        ref_padded = None

    h, w = padded.shape
    recon = np.zeros((h, w), dtype=np.float32)
    qstep = quant_step_from_qp(qp)
    tokens: List[int] = []
    mv_count = 0

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            cur = padded[by : by + block_size, bx : bx + block_size].astype(np.float32)

            if frame_type == "P" and ref_padded is not None:
                pred, mv = inter_predict(cur, ref_padded, by, bx, block_size, search_range)
                mode = 1
                mv_count += 1
            else:
                pred = intra_predict(recon, by, bx, block_size)
                mv = (0, 0)
                mode = 0

            residual = cur - pred
            coeff = dct2(residual)
            qcoeff = np.rint(coeff / qstep).astype(np.int32)
            zz = zigzag_scan(qcoeff)

            tokens.extend([TOK_BLOCK_START, mode, int(mv[0]), int(mv[1])])
            tokens.extend(coeffs_to_tokens(zz))

            recon_residual = idct2(qcoeff.astype(np.float32) * qstep)
            recon_block = np.clip(np.rint(pred + recon_residual), 0, 255)
            recon[by : by + block_size, bx : bx + block_size] = recon_block

    recon = deblock_filter(recon, block_size)
    recon = unpad(recon, pad).astype(np.float32)
    return recon, tokens, mv_count


def update_qp(prev_qp: int, prev_bits: Optional[int], target_bits_per_frame: float) -> int:
    if prev_bits is None:
        return prev_qp
    qp = prev_qp
    if prev_bits > target_bits_per_frame * 1.10:
        qp += 2
    elif prev_bits < target_bits_per_frame * 0.90:
        qp -= 2
    return int(np.clip(qp, 12, 42))


def encode_frame(
    frame_rgb: np.ndarray,
    frame_idx: int,
    frame_type: str,
    qp: int,
    ref_yuv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    block_size: int = 8,
    search_range: int = 4,
) -> Tuple[EncodedFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    y, u, v = rgb_to_yuv420(frame_rgb)
    ref_y = ref_u = ref_v = None
    if ref_yuv is not None:
        ref_y, ref_u, ref_v = ref_yuv

    rec_y, tok_y, mv_y = encode_plane(y, ref_y, frame_type, qp, block_size, search_range)
    rec_u, tok_u, mv_u = encode_plane(u, ref_u, frame_type, qp, block_size, search_range)
    rec_v, tok_v, mv_v = encode_plane(v, ref_v, frame_type, qp, block_size, search_range)

    syntax_tokens: List[int] = [TOK_FRAME_Y] + tok_y + [TOK_FRAME_U] + tok_u + [TOK_FRAME_V] + tok_v
    bitstream, freqs = arithmetic_encode(syntax_tokens)
    payload = {
        "bitstream": bitstream,
        "freqs": freqs,
        "length": len(syntax_tokens),
    }
    bits = len(bitstream)

    encoded = EncodedFrame(
        index=frame_idx,
        frame_type=frame_type,
        qp=qp,
        bits=bits,
        payload=payload,
        mv_count=(mv_y + mv_u + mv_v),
    )
    recon_yuv = (rec_y, rec_u, rec_v)
    return encoded, recon_yuv


def encode_video(
    frames_rgb: Sequence[np.ndarray],
    gop: int = 8,
    target_bpp: float = 0.20,
    start_qp: int = 24,
    block_size: int = 8,
    search_range: int = 4,
) -> Dict[str, object]:
    if not frames_rgb:
        return {"frames": [], "total_bits": 0, "avg_bpp": 0.0}

    h, w = frames_rgb[0].shape[:2]
    target_bits_per_frame = target_bpp * h * w

    encoded_frames: List[EncodedFrame] = []
    prev_bits = None
    qp = int(start_qp)
    ref_yuv = None

    for i, frame in enumerate(frames_rgb):
        qp = update_qp(qp, prev_bits, target_bits_per_frame)
        frame_type = "I" if (i % gop == 0) else "P"
        encoded, recon_yuv = encode_frame(
            frame_rgb=frame,
            frame_idx=i,
            frame_type=frame_type,
            qp=qp,
            ref_yuv=ref_yuv,
            block_size=block_size,
            search_range=search_range,
        )
        encoded_frames.append(encoded)
        prev_bits = encoded.bits
        ref_yuv = recon_yuv

    total_bits = int(sum(f.bits for f in encoded_frames))
    avg_bpp = float(total_bits / (len(encoded_frames) * h * w))
    return {"frames": encoded_frames, "total_bits": total_bits, "avg_bpp": avg_bpp}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, size=(144, 256, 3), dtype=np.uint8) for _ in range(10)]
    out = encode_video(frames, gop=4, target_bpp=0.15, start_qp=24)
    print("total_bits:", out["total_bits"])
    print("avg_bpp:", round(out["avg_bpp"], 4))
    for f in out["frames"][:3]:
        print(f"frame={f.index} type={f.frame_type} qp={f.qp} bits={f.bits} mv={f.mv_count}")
