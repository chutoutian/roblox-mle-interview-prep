"""
Simplified, interview-friendly JPEG-like codec (not a standards-compliant .jpg writer).

Implements the core ideas end-to-end:
- RGB <-> YCbCr transform (BT.601-ish)
- 8x8 block DCT / IDCT
- Quantization / dequantization (quality scaling)
- Zigzag scan
- DC differential coding
- AC run-length coding (EOB / ZRL)
- Huffman entropy coding

The output is a custom Python bitstream container that can be decoded by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Color transform
# -----------------------------


def rgb_to_ycbcr(rgb_u8: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 image to YCbCr float32 image.
    Shape: (H, W, 3) -> (H, W, 3).
    """
    rgb = rgb_u8.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # ITU-R BT.601 (full-range-ish). For interview practice, this is fine.
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr float image to RGB uint8 image.
    Shape: (H, W, 3) -> (H, W, 3).
    """
    y = ycbcr[..., 0].astype(np.float32)
    cb = ycbcr[..., 1].astype(np.float32) - 128.0
    cr = ycbcr[..., 2].astype(np.float32) - 128.0

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def chroma_subsample_420(ycbcr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Very simple 4:2:0 subsampling:
    - Y stays HxW
    - Cb/Cr are averaged over 2x2 -> (H/2)x(W/2)
    """
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]

    h, w = y.shape
    # Pad chroma to even H/W by edge replication (avoids dropping last row/col).
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        cb = np.pad(cb, ((0, pad_h), (0, pad_w)), mode="edge")
        cr = np.pad(cr, ((0, pad_h), (0, pad_w)), mode="edge")

    cb2 = 0.25 * (cb[0::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 0::2] + cb[1::2, 1::2])
    cr2 = 0.25 * (cr[0::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 0::2] + cr[1::2, 1::2])
    return y, cb2.astype(np.float32), cr2.astype(np.float32)


def chroma_upsample_420(y: np.ndarray, cb2: np.ndarray, cr2: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Simple nearest-neighbor 4:2:0 upsampling back to HxW.
    """
    cb = np.repeat(np.repeat(cb2, 2, axis=0), 2, axis=1)
    cr = np.repeat(np.repeat(cr2, 2, axis=0), 2, axis=1)
    cb = cb[:out_h, :out_w]
    cr = cr[:out_h, :out_w]
    y = y[:out_h, :out_w]
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


# -----------------------------
# Block ops, DCT/IDCT
# -----------------------------


def pad_to_multiple(x: np.ndarray, block: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = x.shape

    ## we want  (h + pad_h ) % block == 0 =>  pad_h = (-h) % block
    pad_h = (-h) % block
    pad_w = (-w) % block
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return np.pad(x, ((0, pad_h), (0, pad_w)), mode="edge"), (pad_h, pad_w)


def unpad(x: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = pad
    if pad_h:
        x = x[:-pad_h, :]
    if pad_w:
        x = x[:, :-pad_w]
    return x


def iter_blocks(x: np.ndarray, block: int = 8) -> Iterable[Tuple[int, int, np.ndarray]]:
    h, w = x.shape
    for by in range(0, h, block):
        for bx in range(0, w, block):
            yield by, bx, x[by : by + block, bx : bx + block]


def put_block(dst: np.ndarray, by: int, bx: int, block_arr: np.ndarray) -> None:
    b = block_arr.shape[0]
    dst[by : by + b, bx : bx + b] = block_arr


_DCT_MAT_8: Optional[np.ndarray] = None


def dct_matrix(n: int = 8) -> np.ndarray:
    global _DCT_MAT_8
    if n == 8 and _DCT_MAT_8 is not None:
        return _DCT_MAT_8

    mat = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = alpha * np.cos((np.pi / n) * (i + 0.5) * k)
    if n == 8:
        _DCT_MAT_8 = mat
    return mat


def dct2(block8: np.ndarray) -> np.ndarray:
    c = dct_matrix(8)
    x = block8.astype(np.float32)
    return (c @ x @ c.T).astype(np.float32)


def idct2(coeff8: np.ndarray) -> np.ndarray:
    c = dct_matrix(8)
    x = coeff8.astype(np.float32)
    return (c.T @ x @ c).astype(np.float32)


# -----------------------------
# Quantization + zigzag
# -----------------------------


LUMA_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


CHROMA_Q = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def quality_scale_table(qtable: np.ndarray, quality: int) -> np.ndarray:
    """
    JPEG-like quality scaling. quality in [1,100], 50 means unchanged-ish.
    """
    q = int(np.clip(quality, 1, 100))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    scaled = np.floor((qtable * scale + 50) / 100)
    scaled = np.clip(scaled, 1, 255)
    return scaled.astype(np.float32)


_ZZ_IDX: Optional[np.ndarray] = None
_IZZ_POS: Optional[np.ndarray] = None


def zigzag_indices() -> np.ndarray:
    global _ZZ_IDX, _IZZ_POS
    if _ZZ_IDX is not None:
        return _ZZ_IDX

    idx = []
    for s in range(15):  # sums 0..14
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < 8 and j < 8:
                    idx.append((i, j))
        else:
            for j in range(s + 1):
                i = s - j
                if i < 8 and j < 8:
                    idx.append((i, j))
    _ZZ_IDX = np.array([i * 8 + j for i, j in idx], dtype=np.int32)

    pos = np.zeros((64,), dtype=np.int32)
    for k, flat in enumerate(_ZZ_IDX):
        pos[flat] = k
    _IZZ_POS = pos
    return _ZZ_IDX


def zigzag_scan(block8: np.ndarray) -> np.ndarray:
    zz = zigzag_indices()
    flat = block8.reshape(-1)
    return flat[zz].astype(np.int32)


def inverse_zigzag(coeffs64: np.ndarray) -> np.ndarray:
    zz = zigzag_indices()
    flat = np.zeros((64,), dtype=np.int32)
    flat[zz] = coeffs64.astype(np.int32)
    return flat.reshape(8, 8)


def quantize(dct8: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    return np.rint(dct8 / qtable).astype(np.int32)


def dequantize(qcoeff8: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    return (qcoeff8.astype(np.float32) * qtable).astype(np.float32)


# -----------------------------
# JPEG-like symbolization (category + amplitude bits)
# -----------------------------


def magnitude_category(v: int) -> int:
    a = abs(int(v))
    if a == 0:
        return 0
    return int(np.floor(np.log2(a))) + 1


def amplitude_to_bits(v: int, size: int) -> int:
    """
    JPEG "additional bits" mapping.
    For negative values, bits are in ones-complement form within 'size' bits.
    """
    if size == 0:
        return 0
    v = int(v)
    if v >= 0:
        return v
    return ((1 << size) - 1) + v


def bits_to_amplitude(bits: int, size: int) -> int:
    if size == 0:
        return 0
    bits = int(bits)
    if bits >= (1 << (size - 1)):
        return bits
    return bits - ((1 << size) - 1)


def rle_ac_symbols(ac_coeffs: Sequence[int]) -> List[Tuple[int, int]]:
    """
    AC coefficients (63 values) -> list of (RS_symbol, amplitude_value).
    RS_symbol packs: (run_of_zeros << 4) | size_category.
    Uses:
    - EOB: 0x00
    - ZRL: 0xF0 (16 zeros)
    """
    out: List[Tuple[int, int]] = []
    run = 0
    for v in ac_coeffs:
        v = int(v)
        if v == 0:
            run += 1
            continue

        while run > 15:
            out.append((0xF0, 0))  # ZRL
            run -= 16

        size = magnitude_category(v)
        rs = (run << 4) | size
        out.append((rs, v))
        run = 0

    # Only emit EOB when there are remaining trailing zeros (i.e., last coefficients are 0).
    # If the final AC coefficient is non-zero, the block is "fully specified" and EOB is redundant.
    if run > 0:
        out.append((0x00, 0))  # EOB
    return out


# -----------------------------
# Bit I/O
# -----------------------------


class BitWriter:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0
        self._bitlen = 0

    @property
    def bit_length(self) -> int:
        return self._bitlen

    def write_bits(self, bits: int, n: int) -> None:
        if n == 0:
            return
        bits &= (1 << n) - 1
        self._bitlen += n
        while n > 0:
            take = min(8 - self._nbits, n)
            shift = n - take
            chunk = (bits >> shift) & ((1 << take) - 1)
            self._cur = (self._cur << take) | chunk
            self._nbits += take
            n -= take
            if self._nbits == 8:
                self._buf.append(self._cur & 0xFF)
                self._cur = 0
                self._nbits = 0

    def finish(self) -> bytes:
        if self._nbits:
            self._cur <<= 8 - self._nbits
            self._buf.append(self._cur & 0xFF)
            self._cur = 0
            self._nbits = 0
        return bytes(self._buf)


class BitReader:
    def __init__(self, data: bytes, bit_length: int) -> None:
        self._data = data
        self._bit_length = int(bit_length)
        self._pos = 0  # bits read
        self._byte_i = 0
        self._cur = 0
        self._nbits = 0

    def read_bits(self, n: int) -> int:
        if n == 0:
            return 0
        if self._pos + n > self._bit_length:
            raise ValueError("Bitstream underflow")
        out = 0
        while n > 0:
            if self._nbits == 0:
                self._cur = self._data[self._byte_i]
                self._byte_i += 1
                self._nbits = 8
            take = min(self._nbits, n)
            shift = self._nbits - take
            chunk = (self._cur >> shift) & ((1 << take) - 1)
            out = (out << take) | chunk
            self._nbits -= take
            self._cur &= (1 << shift) - 1 if shift else 0
            n -= take
            self._pos += take
        return out


# -----------------------------
# Huffman coding
# -----------------------------


@dataclass(frozen=True)
class HuffmanTable:
    # symbol -> (code_bits, code_length)
    codes: Dict[int, Tuple[int, int]]

    def build_decoder(self) -> "HuffmanDecoder":
        return HuffmanDecoder.from_codes(self.codes)


class HuffmanDecoder:
    def __init__(self) -> None:
        self._next: Dict[int, "HuffmanDecoder"] = {}
        self._symbol: Optional[int] = None

    @staticmethod
    def from_codes(codes: Dict[int, Tuple[int, int]]) -> "HuffmanDecoder":
        root = HuffmanDecoder()
        for sym, (bits, n) in codes.items():
            node = root
            for i in range(n - 1, -1, -1):
                b = (bits >> i) & 1
                if b not in node._next:
                    node._next[b] = HuffmanDecoder()
                node = node._next[b]
            node._symbol = int(sym)
        return root

    def decode_symbol(self, br: BitReader) -> int:
        node = self
        while node._symbol is None:
            b = br.read_bits(1)
            if b not in node._next:
                raise ValueError("Invalid Huffman code")
            node = node._next[b]
        return int(node._symbol)


def _build_huffman_codes(symbol_counts: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """
    Build Huffman codes using a classic heap-based tree.
    Returns symbol -> (bits, length).
    """
    # Ensure non-empty: add a dummy symbol if needed.
    if not symbol_counts:
        symbol_counts = {0: 1}

    # Heap nodes: (freq, tie, node)
    # node: int (symbol) or (left, right)
    heap: List[Tuple[int, int, object]] = []
    tie = 0
    for sym, c in symbol_counts.items():
        heappush(heap, (int(c), tie, int(sym)))
        tie += 1

    if len(heap) == 1:
        _, _, only = heap[0]
        return {int(only): (0b0, 1)}

    while len(heap) > 1:
        f1, _, n1 = heappop(heap)
        f2, _, n2 = heappop(heap)
        heappush(heap, (f1 + f2, tie, (n1, n2)))
        tie += 1

    _, _, root = heap[0]
    codes: Dict[int, Tuple[int, int]] = {}

    def walk(node: object, bits: int, n: int) -> None:
        if isinstance(node, int):
            codes[int(node)] = (bits, n)
            return
        left, right = node  # type: ignore[misc]
        walk(left, (bits << 1) | 0, n + 1)
        walk(right, (bits << 1) | 1, n + 1)

    walk(root, 0, 0)

    # Guard: no zero-length code.
    for sym, (bits, n) in list(codes.items()):
        if n == 0:
            codes[sym] = (0, 1)
    return codes


def build_huffman_table(symbols: Iterable[int]) -> HuffmanTable:
    counts: Dict[int, int] = {}
    for s in symbols:
        counts[int(s)] = counts.get(int(s), 0) + 1
    return HuffmanTable(codes=_build_huffman_codes(counts))


# -----------------------------
# Container format
# -----------------------------


@dataclass(frozen=True)
class JPEGLikeBitstream:
    height: int
    width: int
    subsampling: str  # "444" or "420"
    quality: int
    y_pad: Tuple[int, int]
    cb_pad: Tuple[int, int]
    cr_pad: Tuple[int, int]
    y_shape_padded: Tuple[int, int]
    cb_shape_padded: Tuple[int, int]
    cr_shape_padded: Tuple[int, int]
    q_luma: np.ndarray  # (8,8) float32
    q_chroma: np.ndarray  # (8,8) float32
    dc_luma: HuffmanTable
    ac_luma: HuffmanTable
    dc_chroma: HuffmanTable
    ac_chroma: HuffmanTable
    data: bytes
    bit_length: int


# -----------------------------
# Encode / decode
# -----------------------------


def _encode_channel_symbols(
    channel: np.ndarray, qtable: np.ndarray
) -> Tuple[List[Tuple[int, int, List[Tuple[int, int]]]], List[int], List[int]]:
    """
    Returns:
    - per-block symbols: [(dc_size, dc_val, [(rs, ac_val), ...]), ...]
    - dc_size_symbols (for Huffman table building)
    - ac_rs_symbols (for Huffman table building)
    """
    blocks: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    dc_sizes: List[int] = []
    ac_rs: List[int] = []

    prev_dc = 0
    for _, _, b in iter_blocks(channel, 8):
        shifted = b.astype(np.float32) - 128.0
        coeff = dct2(shifted)
        qcoeff = quantize(coeff, qtable)
        zz = zigzag_scan(qcoeff)

        dc = int(zz[0])
        dcdiff = dc - prev_dc
        prev_dc = dc
        dc_size = magnitude_category(dcdiff)
        dc_sizes.append(dc_size)

        ac_pairs = rle_ac_symbols(zz[1:].tolist())
        ac_rs.extend([rs for rs, _ in ac_pairs])

        blocks.append((dc_size, dcdiff, ac_pairs))

    return blocks, dc_sizes, ac_rs


def _write_channel_bits(
    bw: BitWriter,
    blocks: Sequence[Tuple[int, int, List[Tuple[int, int]]]],
    dc_table: HuffmanTable,
    ac_table: HuffmanTable,
) -> None:
    dc_codes = dc_table.codes
    ac_codes = ac_table.codes
    for dc_size, dc_val, ac_pairs in blocks:
        bits, n = dc_codes[int(dc_size)]
        bw.write_bits(bits, n)
        bw.write_bits(amplitude_to_bits(dc_val, dc_size), dc_size)

        for rs, v in ac_pairs:
            bits, n = ac_codes[int(rs)]
            bw.write_bits(bits, n)
            size = int(rs) & 0x0F
            bw.write_bits(amplitude_to_bits(v, size), size)


def jpeg_like_encode(rgb_u8: np.ndarray, quality: int = 50, subsampling: str = "420") -> JPEGLikeBitstream:
    """
    Encode an (H,W,3) uint8 RGB image into a simplified JPEG-like bitstream.
    subsampling: "444" or "420"
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("Expected input shape (H, W, 3)")
    if rgb_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB input")
    if subsampling not in ("444", "420"):
        raise ValueError("subsampling must be '444' or '420'")
    ### 444: no subsampling
    ### 422: H x (W/2)
    h, w, _ = rgb_u8.shape
    ycbcr = rgb_to_ycbcr(rgb_u8)

    if subsampling == "420": # average 2x2
        y, cb, cr = chroma_subsample_420(ycbcr)
    else:
        y = ycbcr[..., 0]
        cb = ycbcr[..., 1]
        cr = ycbcr[..., 2]

    y, y_pad = pad_to_multiple(y, 8)
    cb, cb_pad = pad_to_multiple(cb, 8)
    cr, cr_pad = pad_to_multiple(cr, 8)

    q_luma = quality_scale_table(LUMA_Q, quality)
    q_chroma = quality_scale_table(CHROMA_Q, quality)

    y_blocks, y_dc_sizes, y_ac_rs = _encode_channel_symbols(y, q_luma)
    cb_blocks, cb_dc_sizes, cb_ac_rs = _encode_channel_symbols(cb, q_chroma)
    cr_blocks, cr_dc_sizes, cr_ac_rs = _encode_channel_symbols(cr, q_chroma)

    dc_luma = build_huffman_table(y_dc_sizes)
    ac_luma = build_huffman_table(y_ac_rs)
    dc_chroma = build_huffman_table(cb_dc_sizes + cr_dc_sizes)
    ac_chroma = build_huffman_table(cb_ac_rs + cr_ac_rs)

    bw = BitWriter()
    _write_channel_bits(bw, y_blocks, dc_luma, ac_luma)
    _write_channel_bits(bw, cb_blocks, dc_chroma, ac_chroma)
    _write_channel_bits(bw, cr_blocks, dc_chroma, ac_chroma)
    data = bw.finish()

    return JPEGLikeBitstream(
        height=h,
        width=w,
        subsampling=subsampling,
        quality=int(np.clip(quality, 1, 100)),
        y_pad=y_pad,
        cb_pad=cb_pad,
        cr_pad=cr_pad,
        y_shape_padded=tuple(int(x) for x in y.shape),
        cb_shape_padded=tuple(int(x) for x in cb.shape),
        cr_shape_padded=tuple(int(x) for x in cr.shape),
        q_luma=q_luma.astype(np.float32),
        q_chroma=q_chroma.astype(np.float32),
        dc_luma=dc_luma,
        ac_luma=ac_luma,
        dc_chroma=dc_chroma,
        ac_chroma=ac_chroma,
        data=data,
        bit_length=bw.bit_length,
    )


def _decode_channel(
    br: BitReader,
    shape_padded: Tuple[int, int],
    pad: Tuple[int, int],
    qtable: np.ndarray,
    dc_table: HuffmanTable,
    ac_table: HuffmanTable,
) -> np.ndarray:
    h, w = shape_padded
    out = np.zeros((h, w), dtype=np.float32)

    dc_dec = dc_table.build_decoder()
    ac_dec = ac_table.build_decoder()

    prev_dc = 0
    for by in range(0, h, 8):
        for bx in range(0, w, 8):
            dc_size = dc_dec.decode_symbol(br)
            dc_bits = br.read_bits(dc_size)
            dcdiff = bits_to_amplitude(dc_bits, dc_size)
            dc = prev_dc + dcdiff
            prev_dc = dc

            zz = np.zeros((64,), dtype=np.int32)
            zz[0] = int(dc)

            k = 1
            while k < 64:
                rs = ac_dec.decode_symbol(br)
                if rs == 0x00:
                    break  # EOB
                if rs == 0xF0:
                    k += 16
                    continue

                run = (rs >> 4) & 0x0F
                size = rs & 0x0F
                k += run
                if k >= 64:
                    raise ValueError("Invalid RLE (overrun)")
                amp_bits = br.read_bits(size)
                v = bits_to_amplitude(amp_bits, size)
                zz[k] = int(v)
                k += 1

            qcoeff = inverse_zigzag(zz)
            coeff = dequantize(qcoeff, qtable)
            block = idct2(coeff) + 128.0
            put_block(out, by, bx, block)

    out = np.clip(out, 0, 255)
    out = unpad(out, pad)
    return out


def jpeg_like_decode(stream: JPEGLikeBitstream) -> np.ndarray:
    """
    Decode a JPEGLikeBitstream back to an (H,W,3) uint8 RGB image.
    """
    br = BitReader(stream.data, stream.bit_length)

    y = _decode_channel(
        br,
        stream.y_shape_padded,
        stream.y_pad,
        stream.q_luma,
        stream.dc_luma,
        stream.ac_luma,
    )
    cb = _decode_channel(
        br,
        stream.cb_shape_padded,
        stream.cb_pad,
        stream.q_chroma,
        stream.dc_chroma,
        stream.ac_chroma,
    )
    cr = _decode_channel(
        br,
        stream.cr_shape_padded,
        stream.cr_pad,
        stream.q_chroma,
        stream.dc_chroma,
        stream.ac_chroma,
    )

    if stream.subsampling == "420":
        ycbcr = chroma_upsample_420(y, cb, cr, stream.height, stream.width)
    else:
        ycbcr = np.stack([y, cb, cr], axis=-1).astype(np.float32)
        ycbcr = ycbcr[: stream.height, : stream.width, :]

    return ycbcr_to_rgb(ycbcr)


# -----------------------------
# Metrics / quick demo
# -----------------------------


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(m))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 192, 3), dtype=np.uint8)
    stream = jpeg_like_encode(img, quality=50, subsampling="420")
    rec = jpeg_like_decode(stream)
    print("PSNR:", psnr(img, rec))
