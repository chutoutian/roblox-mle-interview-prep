# Part I — Video Domain Knowledge

> Image and video codec knowledge, from pixels to the encode pipeline and rate control. Networking, streaming delivery,
> and real-time transport live in [05_networking_streaming.md](05_networking_streaming.md). Anything
> that needs a neural network lives in [02_ml_domain.md#6-generative-models--learned-compression](02_ml_domain.md#6-generative-models--learned-compression). General ML lives in
> [02_ml_domain.md](02_ml_domain.md).

**Contents**

| § | Topic | Who asks it |
|---|-------|-------------|
| 1 | Human vision & color | everyone, warm-up |
| 2 | Image codecs | Apple, Google |
| 3 | Video codecs | Netflix, Meta, Nvidia, Apple |

---

# 1. Human Vision & Color

## 1.1 Why YCbCr instead of RGB

**Q: Why convert RGB to YCbCr? Why is Y more important?**

Two independent reasons, and you should give both:

1. **Decorrelation.** R, G, B are highly correlated (a bright pixel is bright in all three). A codec compresses each plane independently, so correlated planes mean you pay for the same information three times. Y/Cb/Cr are much closer to statistically independent, so per-plane coding is near-optimal.
2. **Perceptual separation.** The transform isolates *luminance* (Y) from *chrominance* (Cb, Cr). The human visual system has ~10× more luminance-sensitive receptors in the fovea than color-sensitive ones, and its contrast sensitivity function falls off far faster for chroma than for luma. Once brightness is in its own channel, you can spend fewer bits on the two you can't see well.

```
Y  = 0.299R + 0.587G + 0.114B          (BT.601 — SD)
Y  = 0.2126R + 0.7152G + 0.0722B       (BT.709 — HD)     ← green dominates: eye is most sensitive to green
Cb = (B − Y) / 1.772   (scaled)
Cr = (R − Y) / 1.402   (scaled)
```

## 1.2 Chroma subsampling

**Q: What is 4:2:0 / 4:2:2 / 4:4:4? Why is 4:2:0 the default?**

The `J:a:b` notation is defined over a 4-pixel-wide × 2-row reference block:
- `J` = reference width (always 4)
- `a` = number of Cb/Cr samples in row 1
- `b` = number of Cb/Cr samples in row 2 (0 = row 2 reuses row 1's samples)

| Format | Horizontal | Vertical | Bytes/pixel (8-bit) | Where used |
|--------|-----------|----------|------|------|
| 4:4:4  | full      | full     | 3.0  | Mastering, screen content, ProRes 4444 |
| 4:2:2  | ½         | full     | 2.0  | Broadcast contribution, ProRes 422, chroma keying |
| 4:2:0  | ½         | ½        | 1.5  | Everything consumer: streaming, phones, Blu-ray |
| 4:0:0  | —         | —        | 1.0  | Monochrome |

**Encoder-side downsampling** — each 2×2 block of Cb (and Cr) collapses to one sample, ideally low-pass filtered first to avoid aliasing:

```
Full-res pixels (each has Y, Cb, Cr):     After 4:2:0:

  P00  P01  P02  P03                       Y plane (W×H, unchanged)
  P10  P11  P12  P13       Y →               Y00 Y01 Y02 Y03
  P20  P21  P22  P23                          Y10 Y11 Y12 Y13  ...
  P30  P31  P32  P33
                                            Cb plane (W/2 × H/2):
                            Cb →              CbA  CbB
                                              CbC  CbD

  CbA = filter(Cb00, Cb01, Cb10, Cb11)
```

**Sampling rate & antialiasing:** Downsampling reduces the spatial sampling rate, so detail above the new Nyquist limit (half that rate) can alias into false patterns or flicker. Antialiasing suppresses those frequencies with a low-pass filter before downsampling; for 4:2:0, this filtering applies to Cb/Cr while Y retains its original sampling grid.

**Decoder-side upsampling**: nearest-neighbor (fast, blocky), bilinear (most decoders), bicubic/Lanczos (post-production).

**Memory layout (YUV420P planar, 1920×1080)**:
```
Y  plane: 1920 × 1080 = 2,073,600 bytes
Cb plane:  960 ×  540 =   518,400 bytes
Cr plane:  960 ×  540 =   518,400 bytes
─────────────────────────────────────────
Total:               3,110,400 bytes = 1.5 bytes/pixel
RGB24:     1920×1080×3 = 6,220,800 bytes = 3.0 bytes/pixel
```
4:2:0 halves the data *before* the codec does anything.

**Follow-ups you should expect:**

- *"When does 4:2:0 hurt?"*

  1. **Colored text and UI:** Sharp color edges, especially small red/blue text on contrasting backgrounds, can blur or bleed. Screen captures and remote displays benefit from 4:4:4 when supported.
  2. **Green-screen keying:** Reduced color detail makes hair and fine edges harder to separate from the background. 4:2:2 or 4:4:4 helps produce a cleaner transparency mask (matte).
  3. **Later color grading:** Increasing saturation or selectively adjusting colors can reveal chroma bleeding and blurred color boundaries. Preserving more chroma detail gives post-production more flexibility.

- *"Chroma siting?"* — Where the subsampled chroma sample sits relative to the luma grid. Left/co-sited and center are common conventions; H.264/HEVC can signal sample location in VUI rather than imposing one universal layout. JPEG/MPEG-1 commonly use center siting. Getting the convention wrong produces a half-pixel chroma shift — a classic "colors look slightly offset" bug.
- *"Planar vs semi-planar?"* — I420/YV12 are three separate planes; **NV12** interleaves Cb and Cr in one plane (`Y plane, then CbCrCbCr...`). NV12 is what GPU/hardware encoders want (NVENC, VideoToolbox, Media Foundation) because it's two surfaces instead of three.

#### Coding: RGB ↔ YCbCr and 4:2:0 `[READY]`

**Task:** Implement the color transform; keep Y at `H×W`, average each 2×2 Cb/Cr region into one sample, and support odd dimensions.  
**Complexity:** `O(HW)` time and `O(HW)` auxiliary space beyond the output for vectorized intermediates.  
**Pitfalls:** Do not downsample Y; state the BT.601/709, full/limited-range, and chroma-siting conventions first; do not silently drop odd boundary pixels.

```python
import numpy as np

def rgb_to_ycbcr(rgb):
    x = rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y  =  0.299000*r + 0.587000*g + 0.114000*b
    cb = -0.168736*r - 0.331264*g + 0.500000*b + 128
    cr =  0.500000*r - 0.418688*g - 0.081312*b + 128
    return np.stack([y, cb, cr], axis=-1)

def chroma_420(ycbcr):
    y, cb, cr = np.moveaxis(ycbcr, -1, 0)
    ph, pw = y.shape[0] % 2, y.shape[1] % 2
    cb = np.pad(cb, ((0, ph), (0, pw)), mode="edge")
    cr = np.pad(cr, ((0, ph), (0, pw)), mode="edge")

    def avg2x2(p):
        return (p[0::2, 0::2] + p[0::2, 1::2]
                + p[1::2, 0::2] + p[1::2, 1::2]) * 0.25

    return y, avg2x2(cb), avg2x2(cr)

def upsample_420_nn(y, cb, cr):
    h, w = y.shape
    cb = np.repeat(np.repeat(cb, 2, 0), 2, 1)[:h, :w]
    cr = np.repeat(np.repeat(cr, 2, 0), 2, 1)[:h, :w]
    return np.stack([y, cb, cr], axis=-1)
```

[Full implementation](./image_codec.py) — forward/inverse color transforms, 4:2:0 down/up-sampling, and odd-size padding are all in the same file.

#### Coding: bilinear resize `[TOY]`

**Task:** Resize an `H×W×C` image to arbitrary dimensions without OpenCV; explicitly define the half-pixel coordinate and boundary policies.  
**Complexity:** `O(H_out W_out C)` time and `O(H_out W_out C)` output space.  
**Pitfalls:** Mixing `(x,y)` with `(row,col)`; inconsistent `align_corners`/half-pixel conventions; downscaling also requires low-pass antialiasing.

```python
import numpy as np

def bilinear_resize(img, out_h, out_w):
    h, w, channels = img.shape
    out = np.empty((out_h, out_w, channels), dtype=np.float32)
    for oy in range(out_h):
        sy = (oy + 0.5) * h / out_h - 0.5       # half-pixel mapping
        iy, wy = int(np.floor(sy)), sy - np.floor(sy)
        y0, y1 = max(0, min(iy, h - 1)), max(0, min(iy + 1, h - 1))
        for ox in range(out_w):
            sx = (ox + 0.5) * w / out_w - 0.5
            ix, wx = int(np.floor(sx)), sx - np.floor(sx)
            x0, x1 = max(0, min(ix, w - 1)), max(0, min(ix + 1, w - 1))
            top = (1 - wx) * img[y0, x0] + wx * img[y0, x1]
            bot = (1 - wx) * img[y1, x0] + wx * img[y1, x1]
            out[oy, ox] = (1 - wy) * top + wy * bot
    return out
```

**Worked diagram: 8×8 → 2×2.** Half-pixel mapping places the four output samples at source coordinates `(1.5, 1.5)`, `(1.5, 5.5)`, `(5.5, 1.5)`, and `(5.5, 5.5)`, written as `(row, col)`. Each output uses four surrounding samples with weight `0.25` each—not the full corresponding 4×4 area. The interactive HTML diagram lets you inspect each output.

<!-- bilinear-8-to-2 -->

[Full implementation](./sampling.py) — the file contains a fixed 2×, three-channel version; its `downsample` also shrinks Y and mutates the input, so it is not a standard 4:2:0 solution.

#### Coding: I420 → NV12 (stride-aware) `[TODO]`

**Task:** Interleave the U/V planes of three-plane I420 into an NV12 UV plane without assuming tightly packed rows.  
**Complexity:** `O(HW)` time and `O(HW)` output space.  
**Pitfalls:** Stride differing from width, chroma rounding for odd dimensions, and reversing the U/V order.

```python
def i420_to_nv12(y, u, v, y_stride, u_stride, v_stride):
    """TODO: copy visible Y rows, then interleave visible U/V samples."""
    raise NotImplementedError
```

Full implementation: TODO (suggested file: `pixel_formats.py`).

## 1.3 Bit depth and range

- **8-bit** = 256 levels/channel. **10-bit** = 1024. Even for SDR content, 10-bit encoding reduces **banding** and gives ~5–10% BD-rate gain, because internal precision is higher and rounding error accumulates less across prediction loops. This is why "10-bit for 8-bit content" is a real production trick (x264 High10, HEVC Main10).
- **Limited (TV) range**: Y ∈ [16, 235], Cb/Cr ∈ [16, 240], for 8-bit. **Full (PC) range**: [0, 255]. Mismatching them is the classic "washed-out or crushed blacks" bug. Signaled in the bitstream (`video_full_range_flag`).

---

# 2. Image Codecs

## 2.1 JPEG: overall pipeline

<!-- jpeg-encoding-pipeline -->

## 2.2 Color conversion and chroma subsampling

**Steps 1–2: RGB pixels → Y, Cb, Cr planes.**

Start with an `H × W × 3` RGB image. Color conversion produces three component planes:
Y for luma, Cb and Cr for chroma. Conversion alone does not reduce the number of samples.

Optionally low-pass filter Cb/Cr, then downsample them. With **4:2:0**, each chroma plane
has half the width and half the height; Y keeps its original resolution.
For a `16 × 16` input, the planes are `Y: 16 × 16`, `Cb: 8 × 8`, `Cr: 8 × 8`.

**4:2:0 is a choice, not a JPEG requirement.** With 4:4:4, keep all three planes full resolution.
The color convention here is the common [JFIF YCbCr path](https://www.w3.org/Graphics/JPEG/jfif3.pdf).
The conversion and downsampling code is in [§1.1](#11-why-ycbcr-instead-of-rgb) and [§1.2](#12-chroma-subsampling).

## 2.3 Blocks and level shift

**Step 3: each component plane → 8 × 8 blocks → signed samples.**

Split each plane separately. In the `16 × 16`, 4:2:0 example, one interleaved
**MCU (minimum coded unit)** contains four Y blocks, one Cb block, and one Cr block.
Each block contains samples from only one component.

Pad incomplete edge blocks as needed, for example by repeating edge samples.
Keep the original image dimensions so the decoder can discard the padding.

Before the DCT, subtract `128` from every 8-bit sample:
`0…255 → −128…127`. For example, `[128, 140, 120] → [0, 12, −8]`.
This centers the values around zero; it does not discard information.

**Q: Why split into 8×8 blocks?**

Two sub-questions: (1) why blocks at all? (2) why 8×8?

**Why blocks at all — local stationarity.** The DCT implicitly assumes the signal is *stationary*: uniform statistics across the input. A whole image obviously isn't — smooth sky, sharp edges, and texture all live in different places. Within a small patch the content is roughly uniform, so the assumption approximately holds. Each block gets its own coefficients, so smooth and textured regions are represented separately.

**Why 8×8 — three competing forces:**

```
Block size:    Smaller ←────────────────────────────→ Larger
                 4×4           8×8           16×16

Stationarity:    ✓ very local    ✓ good       ✗ crosses more edges
Freq resolution: ✗ only 4 freqs  ✓ 8 freqs    ✓ 16 freqs
Blocking:        mild seams      ✓ acceptable ✗ large visible blocks
Block count:     many blocks     moderate     fewer blocks
```

- **Too small**: fewer frequency bins and more blocks to process. Baseline JPEG does not transmit video-style modes or motion vectors.
- **Too large**: a single block spans visually diverse content, energy compaction fails, and coarse quantization shows up as one big uniform patch.
- **8 = 2³** permits efficient fast-DCT implementations. Baseline JPEG fixes this block size; the encoder does not choose it per region.

## 2.4 DCT

**Step 4: 64 spatial samples → 64 frequency coefficients.**

Apply a 2-D DCT to each shifted block. The top-left **DC** coefficient represents the
block's average level (with a scale factor); the other 63 **AC** coefficients describe variations.
There are still 64 numbers: the DCT reorganizes the information rather than reducing the sample count.

**Q: Why does the DCT concentrate energy?**

Natural images are dominated by low-frequency content. The DCT basis functions are ordered from smooth to oscillatory, so projecting a smooth block onto them puts almost all the energy in the top-left coefficients:

```
[nz  nz  nz   0   0   0   0   0
 nz  nz   0   0   0   0   0   0
 nz   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0  ...]
```

Formally, DCT-II approaches the **KLT** (the optimal decorrelating transform) for a first-order Markov source with correlation ρ→1, which natural images approximate well. Unlike the KLT, DCT is signal-independent, so it needs no transmitted basis and has a fast algorithm.

#### Coding: 2-D DCT / IDCT `[READY]`

**Task:** Construct an orthonormal DCT-II matrix, implement 2-D DCT/IDCT with separable matrix multiplication, and verify `IDCT(DCT(x)) ≈ x`.  
**Complexity:** Direct matrix multiplication takes `O(N³)` time and `O(N²)` matrix space; both are constant for a fixed 8×8 block.  
**Pitfalls:** The directions of `C @ x @ C.T` and the inverse transform; different normalization for the DC row; JPEG's `-128` level shift is not part of the DCT itself; standard video codecs generally use integer approximations.

```python
import numpy as np

def dct_matrix(n=8):
    C = np.empty((n, n), dtype=np.float32)
    for k in range(n):
        alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        for i in range(n):
            C[k, i] = alpha * np.cos(np.pi * (i + 0.5) * k / n)
    return C

def dct2(block):
    C = dct_matrix(block.shape[0])
    return C @ block @ C.T

def idct2(coeff):
    C = dct_matrix(coeff.shape[0])
    return C.T @ coeff @ C
```

[Full implementation](./image_codec.py)

## 2.5 Quantization

**Step 5: DCT coefficients → rounded integer coefficients.**

**Q: How does quantization work?**

For each DCT position, divide by the matching quantization-table entry and round to an integer.

```
c'(x,y)    = round(c(x,y) / q(x,y))      ← encode: divide and round
c_rec(x,y) = c'(x,y) × q(x,y)            ← decode: multiply back
```

Error is bounded by `±q/2`. Larger step → larger error → fewer bits.

**The JPEG quantization table** (luma, quality 50) — step size per frequency:

```
  16  11  10  16  24  40  51  61
  12  12  14  19  26  58  60  55
  14  13  16  24  40  57  69  56
  14  17  22  29  51  87  80  62
  18  22  37  56  68 109 103  77
  24  35  55  64  81 104 113  92
  49  64  78  87 103 121 120 101
  72  92  95  98 112 100 103  99
   ↑ low freq top-left (fine)      high freq bottom-right (coarse) ↑
```

The table encodes the human contrast sensitivity function: we barely see quantization error at high spatial frequencies, so those get big steps.

**Why zeros appear**: if `|c| < q/2` then `round(c/q) = 0`. High-frequency coefficients are already small (energy compaction) *and* get large q → the vast majority vanish. That's what makes zigzag + RLE effective.

```
c = 3,  q = 16  →  round(0.19) = 0    ← gone
c = 52, q = 16  →  round(3.25) = 3    ← kept, error 52−48 = 4
```

#### Coding: quantization / dequantization `[READY]`

**Task:** Scale a JPEG quality table and implement quantization/dequantization; demonstrate that a larger step produces both more zeros and greater error.  
**Complexity:** `O(N²)` time and output space per block.  
**Pitfalls:** Encoding must round rather than truncate; quality must be clamped; dequantization cannot recover information lost to rounding; integer overflow.

```python
import numpy as np

def scale_qtable(base, quality):
    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((base * scale + 50) / 100), 1, 255)

def quantize(coeff, qtable):
    return np.rint(coeff / qtable).astype(np.int32)

def dequantize(qcoeff, qtable):
    return qcoeff.astype(np.float32) * qtable
```

[Full implementation](./image_codec.py)

## 2.6 Zigzag and entropy coding

**Steps 6–8: quantized block → ordered coefficients → symbols → bits.**

Zigzag orders the 64 coefficients starting with DC. Encode DC as a difference, and encode the
63 AC coefficients using zero runs. Huffman-code the resulting categories/symbols, then append
the amplitude bits. These steps preserve the quantized coefficients exactly.

**Q: Why do Zigzag + DC-DPCM + RLE + Huffman work together? Why encode a size category instead of the value?**

- **Zigzag scan** traverses the 8×8 block from low to high frequency, so the sequence naturally ends in a long run of zeros → RLE collapses it, and one **EOB** symbol ends the block.
- **RLE for AC coefficients**: emit `(run, size)` — number of preceding zeros, plus the bit-length category of the next non-zero value.
- **Why `(run, size)` and not the value**: AC values span a huge range, so Huffman-coding values directly means a giant, poorly-trained table. The size category has ~11 values, so the joint `(run, size)` alphabet is small and its statistics are stable. The signed amplitude is represented by additional bits appended after the Huffman code.
  - Example: `0, 0, 0, −6` → Huffman symbol for `(3, 3)` + 3 raw bits for −6.
- **DC coefficients** use DPCM against the previous block's quantized DC **of the same component**. Keep separate predictors for Y, Cb, and Cr.

#### Coding: standard zigzag and inverse `[TOY]`

**Task:** Scan any `m×n` matrix along anti-diagonals and reconstruct it reversibly; a 3×3 input must yield `[1,2,4,7,5,3,6,8,9]`.  
**Complexity:** `O(mn)` time and `O(1)` auxiliary space beyond the returned result.  
**Pitfalls:** Every diagonal satisfies `row+col=d`; reversing the even/odd directions gives the transposed convention; handle 1×N, M×1, empty, and nonrectangular inputs, and validate the length when decoding.

```python
def zigzag_encode(a):
    m, n = len(a), len(a[0])
    out = []
    for d in range(m + n - 1):
        if d % 2 == 0:                    # bottom-left -> top-right
            r, c = min(d, m - 1), d - min(d, m - 1)
            while r >= 0 and c < n:
                out.append(a[r][c]); r -= 1; c += 1
        else:                             # top-right -> bottom-left
            c, r = min(d, n - 1), d - min(d, n - 1)
            while c >= 0 and r < m:
                out.append(a[r][c]); r += 1; c -= 1
    return out

def zigzag_decode(seq, m, n):
    if len(seq) != m * n:
        raise ValueError("wrong coefficient count")
    out, k = [[0] * n for _ in range(m)], 0
    for d in range(m + n - 1):
        if d % 2 == 0:
            r, c = min(d, m - 1), d - min(d, m - 1)
            while r >= 0 and c < n:
                out[r][c] = seq[k]; k += 1; r -= 1; c += 1
        else:
            c, r = min(d, n - 1), d - min(d, n - 1)
            while c >= 0 and r < m:
                out[r][c] = seq[k]; k += 1; r += 1; c -= 1
    return out
```

[Full implementation](./video_codec_components.py) — the scan function itself is readable, but this scratch module's import-time Huffman demo still fails; for a fixed 8×8 codec version, see [image_codec.py](./image_codec.py), whose diagonal direction follows the transposed convention.

#### Coding: DC-DPCM + AC-RLE (EOB/ZRL) `[READY]`

**Task:** Difference each DC coefficient against the previous block; emit JPEG-style `(run,size,value)` tuples for AC, use ZRL for each run of 16 zeros, and EOB for trailing zeros.  
**Complexity:** `O(64)` time per 8×8 block and `O(64)` output in the worst case.  
**Pitfalls:** The additional bits for a negative amplitude are not ordinary two's complement; handle `run == 16`; EOB is unnecessary when the final AC coefficient is nonzero; a restart marker resets the DC predictor.

```python
def magnitude_category(v):
    return 0 if v == 0 else abs(int(v)).bit_length()

def dc_dpcm(dc_values):
    out, prev = [], 0
    for dc in dc_values:
        out.append(dc - prev)
        prev = dc
    return out

def rle_ac(ac):
    out, run = [], 0
    for value in ac:
        if value == 0:
            run += 1
            continue
        while run >= 16:
            out.append((0xF0, 0))         # ZRL
            run -= 16
        size = magnitude_category(value)
        out.append(((run << 4) | size, value))
        run = 0
    if run:
        out.append((0x00, 0))             # EOB
    return out
```

[Full implementation](./image_codec.py)

### Huffman coding

Huffman coding builds a binary prefix tree from symbol frequencies: frequent symbols receive short codewords, rare symbols receive longer ones, and no codeword is a prefix of another. That prefix property lets the decoder consume bits unambiguously without separators. Baseline JPEG Huffman-codes the compact DC categories and AC `(run,size)` symbols, then appends the coefficient-amplitude bits described above.

#### Coding: Huffman encode / decode `[READY]`

**Task:** Build a tree with a min-heap and generate prefix codes; handle tied frequencies and a single-symbol alphabet, then recover the symbols from a bit string.  
**Complexity:** For `N` inputs and `K` distinct symbols, tree construction is `O(N + K log K)` and encoding/decoding is `O(total bits)`.  
**Pitfalls:** Heap ties must not compare heterogeneous tree nodes; the only symbol cannot receive an empty code; the decoder must reject nonbinary characters and an incomplete suffix.

```python
from collections import Counter
from heapq import heapify, heappop, heappush
from itertools import count

def huffman_codes(symbols):
    tie = count()
    heap = [(freq, next(tie), (sym,))
            for sym, freq in Counter(symbols).items()]
    heapify(heap)
    if not heap:
        return {}
    if len(heap) == 1:
        return {heap[0][2][0]: "0"}
    while len(heap) > 1:
        f1, _, left = heappop(heap)
        f2, _, right = heappop(heap)
        heappush(heap, (f1 + f2, next(tie), (left, right)))

    codes = {}
    def walk(node, prefix):
        if len(node) == 1:                # leaf = (symbol,)
            codes[node[0]] = prefix
            return
        walk(node[0], prefix + "0")
        walk(node[1], prefix + "1")
    walk(heap[0][2], "")
    return codes

def huffman_decode(bits, codes):
    reverse, cur, out = {v: k for k, v in codes.items()}, "", []
    for bit in bits:
        if bit not in "01":
            raise ValueError("non-binary input")
        cur += bit
        if cur in reverse:
            out.append(reverse[cur]); cur = ""
    if cur:
        raise ValueError("truncated Huffman suffix")
    return out
```

[Full implementation](./image_codec.py) — uses a bit-level table/decoder; it is a dynamically constructed teaching table, not a default JPEG table.

## 2.7 JPEG file and decoding

**Step 9: compressed block data + decoding information → a JPEG file.**

A common single-scan baseline file has this structure (some table/metadata ordering can vary):

```text
SOI → APP metadata → DQT → SOF0 → DHT → SOS → encoded scan → EOI
                    │      │      │      │
                    │      │      │      └─ scan components / table choices
                    │      │      └──────── Huffman tables
                    │      └─────────────── dimensions / sampling factors
                    └────────────────────── quantization tables
```

SOI/EOI mark the beginning/end. In the encoded scan, a data byte `FF` is followed by
a stuffed `00` so it cannot be mistaken for a marker. The file stores the actual
quantization tables—not just an encoder's “quality 80” setting.

**Decoding reverses the stages:**

```text
Read headers / tables
  ↓
Huffman decode + amplitude bits
  ↓
Restore DC values + expand AC zero runs
  ↓
Inverse zigzag → multiply by quantization table
  ↓
IDCT → add 128 → round / clip to 0…255
  ↓
Reassemble planes / discard padding
  ↓
Upsample chroma if needed → YCbCr to RGB
```

Dequantization and chroma upsampling cannot recover the detail discarded during encoding.
See [JPEG T.81, Annexes A, B and F](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) for the block, file and baseline-coding rules.

#### Coding: JPEG-like block pipeline `[READY]`

**Task:** Connect level shift → DCT → quantization → zigzag → DC-DPCM/AC-RLE, and explain how the decoder reverses the process.  
**Complexity:** Constant work per fixed 8×8 block; `O(HW)` time and coefficient space for the full image.  
**Pitfalls:** Padding/original size and the quantization/Huffman tables must be stored in the header; maintain a DC predictor per component; the length of the teaching container must not be presented as the size of a real `.jpg` file.

```python
import numpy as np

def encode_block(block, qtable, prev_dc):
    coeff = dct2(block.astype(np.float32) - 128.0)
    qcoeff = quantize(coeff, qtable)
    zz = zigzag_scan(qcoeff)

    dc = int(zz[0])
    dc_diff = dc - prev_dc
    symbols = {
        "dc": (magnitude_category(dc_diff), dc_diff),
        "ac": rle_ac_symbols(zz[1:]),
    }
    return symbols, dc

def reconstruct_block(qcoeff, qtable):
    return np.clip(idct2(dequantize(qcoeff, qtable)) + 128, 0, 255)
```

[Full implementation](./image_codec.py) — supports an encode/decode round trip, but its output is a custom Python container rather than a standards-compliant JPEG file.

### JPEG follow-ups

- **Progressive JPEG:** sends coefficients across multiple scans, selecting frequency bands and/or refining precision so a preview can appear early.
- **Restart markers:** introduce byte-aligned restart points and reset DC predictors; Huffman tables are retained.
- **Blocking artifacts:** neighboring 8×8 blocks are quantized independently; baseline JPEG has no in-loop deblocking filter.
- **Color bleed:** chroma downsampling and later upsampling can soften color edges separately from DCT blocking.

## 2.8 Modern image codecs

| Codec | Basis | Notes |
|-------|-------|-------|
| **WebP** | VP8 intra | ~25–30% over JPEG; lossless mode; wide browser support |
| **HEIC/HEIF** | HEVC intra | Apple default since iOS 11; container also holds depth maps, bursts, Live Photos |
| **AVIF** | AV1 intra | ~50% over JPEG at low bitrate; royalty-free; 10/12-bit + HDR; now default-ish on the web |
| **JPEG XL** | new (modular + VarDCT) | Lossless JPEG *transcode* (recompress existing JPEGs ~20% smaller, reversible), progressive, wide gamut. Adoption stalled after Chrome pulled it; Apple ships it |

**Key insight to state:** all of these are essentially "the intra path of a video codec, wrapped in a container." That's why AVIF beats JPEG so decisively — it inherits 25 years of intra-prediction and entropy-coding research. And it's why they're slower to encode.

---

# 3. Video Codecs

Video adds **temporal** redundancy removal on top of spatial. That's the whole story.

## 3.1 The encode pipeline

**Q: Walk me through a video encoder.**

One block of one P-frame, followed through every stage of a hybrid block-based encoder, in the style of the [Transformer walkthrough](./02_ml_domain.html#3-transformer-from-input-to-output). Each step states its input, the operation and its output, with the concrete numbers of the example next to it. H.264, HEVC, VVC and AV1 are all this skeleton with more candidates per step; sections 3.2–3.10 go deeper into each stage, and the coding drill at the end of this section implements steps 4–9 and 11 on 8×8 blocks.

<a id="codec-example"></a>

**The example.** An 8-bit 4:2:0 sequence. Frame $t$ is coded as a P-frame whose only reference is $\tilde F_{t-1}$, the *reconstruction* of the previous frame. We follow one $4 \times 4$ luma block $B$ at rows $8$–$11$, columns $16$–$19$ ($4 \times 4$ keeps every matrix printable; the same equations run on $8 \times 8$ up to $128 \times 128$). The block sits on a bright band that moved one pixel to the right since the previous frame, and the whole picture got brighter by about $4$:

$$
B = \begin{pmatrix} 94 & 95 & 124 & 125 \\ 94 & 96 & 125 & 126 \\ 95 & 96 & 125 & 127 \\ 95 & 97 & 126 & 127 \end{pmatrix},
\qquad
\tilde F_{t-1}[7{:}12,\ 14{:}21] = \begin{pmatrix}
86 & 88 & 89 & 90 & 91 & 92 & 92 & 93 \\
88 & 90 & 91 & 119 & 121 & 122 & 122 & 123 \\
89 & 90 & 92 & 120 & 121 & 123 & 123 & 124 \\
89 & 91 & 92 & 121 & 122 & 123 & 124 & 124 \\
90 & 91 & 93 & 121 & 122 & 123 & 124 & 125 \\
86 & 88 & 89 & 90 & 91 & 92 & 92 & 93
\end{pmatrix}
$$

The reference window shows rows $7$–$12$ and columns $14$–$21$ of $\tilde F_{t-1}$, so the block's own position is its rows $2$–$5$ and columns $3$–$6$; the band is one pixel further left there. The already-coded neighbours of $B$ in the current frame's reconstruction $\tilde F_t$ are the row above, $\mathbf t = (92, 93, 94, 95)$, and the column to the left, $\mathbf l = (92, 93, 93, 94)$. $\mathrm{QP} = 28$.

<a id="codec-notation"></a>

| Symbol | Meaning | In the example |
|---|---|---|
| $F_t \in \{0,\dots,255\}^{H \times W}$ | original luma of frame $t$; the chroma planes $C^b_t, C^r_t$ are $\tfrac{H}{2} \times \tfrac{W}{2}$ | $1920 \times 1080$ |
| $\tilde F_t$ | reconstruction of frame $t$: the picture the decoder will have. References are always reconstructions | |
| $B \in \{0,\dots,255\}^{N \times N}$ | the current block, top-left corner $(r_0, c_0)$ | $N = 4$, $(r_0, c_0) = (8, 16)$ |
| $\mathbf t, \mathbf l \in \{0,\dots,255\}^{N}$ | reconstructed row above and column to the left of $B$ | $(92, 93, 94, 95)$, $(92, 93, 93, 94)$ |
| $P \in \{0,\dots,255\}^{N \times N}$ | prediction of $B$, from an intra mode $m$ or a motion vector $\mathbf v$ | |
| $\mathbf v = (v_y, v_x)$, $\mathbf v^{p}$ | motion vector in pixels (may be fractional) and its predictor from neighbouring blocks | $\mathbf v^{*} = (0, -1)$ |
| $E = B - P$ | residual | |
| $T \in \mathbb{R}^{N \times N}$, $C = T E T^{\top}$ | transform matrix (DCT) and transform coefficients | |
| $\mathrm{QP}$, $\Delta = 2^{(\mathrm{QP} - 4)/6}$, $f$ | quantization parameter, step size, rounding offset | $\mathrm{QP} = 28$, $\Delta = 16$ |
| $\ell \in \mathbb{Z}^{N \times N}$ | quantized levels: the only block data that is transmitted | |
| $\hat C, \hat E, \hat B$ | dequantized coefficients, reconstructed residual, reconstructed block | |
| $D = \lVert B - \hat B \rVert_2^2$, $R$, $\lambda$, $J = D + \lambda R$ | distortion (SSE), bits, Lagrange multiplier, rate–distortion cost | $\lambda = 23$ |

<a id="codec-overview"></a>

**The pipeline at a glance.** The figure is the whole encoder for one block. Each box carries its step number and links to it, and each arrow carries the symbol that step's input–output line uses. Read it in three loops. The **forward path** along the top turns the block $B$ into bits: subtract the prediction $P$ (step 5), transform the residual $E$ to $C$ (6), quantize $C$ to the levels $\ell$ (7) and entropy-code them (8). The **reconstruction loop** down the right repeats what the decoder will do: dequantize and inverse-transform $\ell$ to $\hat C$ and $\hat E$, then add $P$ back and clip to get $\hat B$ (all step 9). The reconstructed frame $\tilde F_t^{\text{raw}}$ passes the in-loop filters (11) and enters the reference buffer as $\tilde F_t$. The **prediction** on the left builds $P$ from those reconstructions, never from the originals. Intra prediction reads the neighbours $\mathbf t, \mathbf l$ of the current frame (3). Inter prediction searches $\tilde F_{t-1}$ for $\mathbf v^{*}$ and fetches the block it points to (4). Mode decision keeps the candidate with the smallest $D_k + \lambda R_k$ (10). The dotted line is that comparison: every candidate is trial-encoded through steps 5–9, which give its distortion $D_k$ and its bit count $R_k$. Rate control (12) turns the bits $R$ already spent into the QP of step 7. The dashed lines are the side information that joins the levels in the bitstream: the mode $m$ and the motion-vector difference $\mathbf v^{*} - \mathbf v^{p}$. The green modules are exactly what the decoder runs. The white ones are decisions only the encoder makes, which is why encoding costs one to two orders of magnitude more than decoding.

<!-- codec-encode-pipeline -->

<a id="codec-step-1"></a>

**Step 1 · Color conversion + chroma subsampling.** $\text{RGB} \in \{0,\dots,255\}^{H \times W \times 3} \;\longrightarrow\; Y \in \{0,\dots,255\}^{H \times W},\ C^b, C^r \in \{0,\dots,255\}^{\frac{H}{2} \times \frac{W}{2}}$

$$
Y = 0.2126\,R' + 0.7152\,G' + 0.0722\,B', \qquad
C^b = 128 + \frac{B' - Y}{1.8556}, \qquad
C^r = 128 + \frac{R' - Y}{1.5748}
$$

$$
C^b_{\downarrow}[i, j] = \Big\lfloor \tfrac{1}{4} \sum_{a=0}^{1} \sum_{b=0}^{1} C^b[2i + a,\ 2j + b] + \tfrac12 \Big\rfloor
\qquad \text{(4:2:0; the same for } C^r\text{)}
$$

BT.709 primaries, full range; the primes mark gamma-encoded channels. The eye resolves brightness far better than colour (Chapter 1), so the two chroma planes are averaged down to one sample per $2 \times 2$ luma samples: $1.5$ samples per pixel instead of $3$ before a single bit is coded. Scaling and denoising happen here too; sensor noise is high-entropy and incompressible, so removing it is a real bitrate win. One pixel: $(R', G', B') = (200, 120, 80) \to (Y, C^b, C^r) = (134, 99, 170)$. From here on each plane is coded on its own, and we follow luma.

<a id="codec-step-2"></a>

**Step 2 · Partition the frame into blocks.** $F_t \;\longrightarrow\; \{B_k\}$, a quadtree of square blocks

$$
\text{split}(B) = \begin{cases}
\text{yes} & \text{if } \displaystyle\sum_{i=1}^{4} J\big(B^{(i)}\big) + \lambda R_{\text{split}} < J(B) \\[4pt]
\text{no} & \text{otherwise}
\end{cases}
$$

$J(B)$ is the rate–distortion cost of coding $B$ as one block (steps 3–10) and $B^{(1)}, \dots, B^{(4)}$ are its four quadrants, each put through the same test again. H.264 uses $16 \times 16$ macroblocks with $8 \times 8$ and $4 \times 4$ sub-blocks; HEVC splits a $64 \times 64$ coding tree unit down to $8 \times 8$; AV1 starts at $128 \times 128$; VVC adds binary and ternary splits (§3.6). Big blocks are cheap where the prediction is good (sky, static background); small blocks follow edges and motion boundaries. The example block is a leaf of this tree: $N = 4$ at $(r_0, c_0) = (8, 16)$.

<a id="codec-step-3"></a>

**Step 3 · Intra prediction candidates.** $(\mathbf t, \mathbf l) \;\longrightarrow\; P^{\text{intra}}_{m} \in \{0,\dots,255\}^{N \times N}$ for every mode $m$

$$
P^{\text{DC}}[i,j] = \Big\lfloor \frac{\sum_{k} t_k + \sum_{k} l_k + N}{2N} \Big\rfloor, \qquad
P^{\text{V}}[i,j] = t_j, \qquad
P^{\text{H}}[i,j] = l_i
$$

$$
P^{\text{ang}}_{\theta}[i,j] = (1 - w)\, \rho[j + k] + w\, \rho[j + k + 1], \qquad
k + w = (i + 1) \tan\theta,\quad k \in \mathbb{Z},\ 0 \le w < 1
$$

$\rho$ is the reference line ($\mathbf t$ extended to the right, or $\mathbf l$ for the horizontal family): every pixel copies the reference sample found by walking back along the direction $\theta$, with linear interpolation between the two nearest samples. HEVC has 33 such angles plus DC and planar (the average of a vertical and a horizontal linear interpolation), VVC 65, AV1 56 plus smooth and Paeth modes (§3.4). The neighbours come from $\tilde F_t$, never from $F_t$: the decoder has only reconstructions.

Example: $P^{\text{DC}} = \lfloor (374 + 372 + 4)/8 \rfloor = 93$ everywhere, and $P^{\text{V}}$ repeats $(92, 93, 94, 95)$ in every row. Neither neighbour contains the bright right half of $B$, so every intra mode misses it: $\text{SAD}(B, P^{\text{DC}}) = 279$ and $\text{SAD}(B, P^{\text{V}}) = 271$. On a P-frame, intra is the fallback for content that motion cannot explain: new objects, uncovered background, scene cuts.

<a id="codec-step-4"></a>

**Step 4 · Inter prediction: motion estimation + compensation.** $(B,\ \tilde F_{t-1}) \;\longrightarrow\; (\mathbf v^{*},\ P^{\text{inter}})$

$$
P_{\mathbf v}[i,j] = \tilde F_{t-1}[r_0 + i + v_y,\ c_0 + j + v_x], \qquad
\mathbf v^{*} = \underset{\mathbf v \in \mathcal W}{\arg\min}\ \underbrace{\sum_{i,j} \big| B[i,j] - P_{\mathbf v}[i,j] \big|}_{\text{SAD}(\mathbf v)} + \lambda_{\text{me}}\, R(\mathbf v - \mathbf v^{p})
$$

Motion estimation is a search. Full search over the window $\mathcal W = \{-S, \dots, S\}^2$ costs $(2S + 1)^2$ SADs of $N^2$ pixels each; fast searches (diamond, hexagon, hierarchical; §3.5) try a few dozen candidates. SAD is used because it is cheap; SATD (the Hadamard transform of the residual) tracks the bits after step 6 better and is used to refine. Only $\mathbf v - \mathbf v^{p}$ is transmitted, where $\mathbf v^{p}$ is predicted from the vectors of the left, top and top-right neighbours (H.264: their median; HEVC and AV1: a candidate list plus an index), so a vector that agrees with its neighbours is nearly free, and merge or skip modes copy $\mathbf v^{p}$ with no vector bits at all. A fractional $\mathbf v$ reads an interpolated reference, in H.264 through the half-pel filter

$$
\tilde F\big[r, c + \tfrac12\big] = \Big\lfloor \frac{\tilde F[r, c-2] - 5\,\tilde F[r, c-1] + 20\,\tilde F[r, c] + 20\,\tilde F[r, c+1] - 5\,\tilde F[r, c+2] + \tilde F[r, c+3] + 16}{32} \Big\rfloor
$$

(quarter-pel by averaging two neighbours; HEVC uses 8 taps). B-frames average two such predictions, $P = \lfloor (P^{(0)}_{\mathbf v_0} + P^{(1)}_{\mathbf v_1} + 1)/2 \rfloor$ (§3.3).

Example: full search with $S = 2$ horizontally and $1$ vertically, $15$ candidates (a real window is $\pm 64$ or more):

| $\text{SAD}(v_y, v_x)$ | $v_x = -2$ | $v_x = -1$ | $v_x = 0$ | $v_x = 1$ | $v_x = 2$ |
|---|---|---|---|---|---|
| $v_y = -1$ | 242 | 139 | 180 | 243 | 245 |
| $v_y = 0$ | 200 | **70** | 135 | 226 | 231 |
| $v_y = 1$ | 235 | 133 | 182 | 249 | 249 |

$\mathbf v^{*} = (0, -1)$: the band moved one pixel to the right, so the block is found one pixel to the *left* in the previous frame. The neighbours moved the same way, so $\mathbf v^{p} = (0, -1)$ and the transmitted difference is $(0, 0)$.

$$
P^{\text{inter}} = P_{(0,-1)} = \tilde F_{t-1}[8{:}11,\ 15{:}18] = \begin{pmatrix} 90 & 91 & 119 & 121 \\ 90 & 92 & 120 & 121 \\ 91 & 92 & 121 & 122 \\ 91 & 93 & 121 & 122 \end{pmatrix}
$$

<a id="codec-step-5"></a>

**Step 5 · Residual.** $(B, P) \;\longrightarrow\; E = B - P \in \{-255, \dots, 255\}^{N \times N}$

The codec never transmits pixels, only what the prediction got wrong. Steps 5–9 run for every candidate of steps 3–4; the matrices below are those of the inter candidate.

$$
E = B - P^{\text{inter}} = \begin{pmatrix} 4 & 4 & 5 & 4 \\ 4 & 4 & 5 & 5 \\ 4 & 4 & 4 & 5 \\ 4 & 4 & 5 & 5 \end{pmatrix}, \qquad \sum_{i,j} E[i,j]^2 = 310
$$

Motion explained the edge; what is left is the $+4$ brightness change and one-pixel noise.

<a id="codec-step-6"></a>

**Step 6 · Transform.** $E \;\longrightarrow\; C = T\, E\, T^{\top} \in \mathbb{R}^{N \times N}$

$$
T[u, x] = \sqrt{\tfrac{2}{N}}\; \kappa_u \cos\!\Big(\frac{(2x + 1)\, u\, \pi}{2N}\Big), \quad \kappa_0 = \tfrac{1}{\sqrt 2},\ \kappa_{u > 0} = 1;
\qquad
T_{N=4} = \begin{pmatrix} 0.500 & 0.500 & 0.500 & 0.500 \\ 0.653 & 0.271 & -0.271 & -0.653 \\ 0.500 & -0.500 & -0.500 & 0.500 \\ 0.271 & -0.653 & 0.653 & -0.271 \end{pmatrix}
$$

Row $u$ of $T$ is a cosine with $u$ half-periods across the block, and $C[u, v]$ measures how much of the residual looks like cosine $u$ vertically times cosine $v$ horizontally. $T$ is orthonormal, $T^{\top} T = I$, so the transform loses nothing ($\sum C^2 = \sum E^2$) and is undone by $E = T^{\top} C\, T$; its job is to gather the energy of a smooth residual into a few coefficients so that the next step can drop the rest. Standards use integer approximations, in H.264 the rows $(1,1,1,1)$, $(2,1,-1,-2)$, $(1,-1,-1,1)$, $(1,-2,2,-1)$ with the scaling folded into the quantizer, so that encoder and decoder are bit-exact on any hardware (§3.7).

$$
C = \begin{pmatrix} 17.50 & -1.39 & 0.00 & 0.57 \\ -0.19 & 0.35 & -0.46 & 0.35 \\ 0.00 & 0.19 & -0.50 & 0.46 \\ -0.46 & 0.35 & 0.19 & -0.35 \end{pmatrix}, \qquad C[0,0] = \tfrac{1}{N} \sum_{i,j} E[i,j] = \tfrac{70}{4} = 17.5
$$

The DC coefficient alone carries $17.5^2 = 306$ of the $310$ units of energy: sixteen residual pixels became one number that matters and fifteen that do not.

<a id="codec-step-7"></a>

**Step 7 · Quantization.** $C \;\longrightarrow\; \ell = \operatorname{sgn}(C)\, \Big\lfloor \frac{|C|}{\Delta} + f \Big\rfloor \in \mathbb{Z}^{N \times N}$

$$
\Delta = 2^{(\mathrm{QP} - 4)/6}: \qquad \mathrm{QP} + 6 \ \Rightarrow\ \Delta \times 2 \ \Rightarrow\ \text{roughly half the bits}, \qquad \mathrm{QP} + 1 \ \Rightarrow\ \Delta \times 1.12
$$

This floor is the only place in the whole pipeline where information is destroyed, and $\mathrm{QP}$ is the knob that sets how much; prediction, transform and entropy coding are all reversible. $f = \tfrac12$ would be rounding to nearest; encoders use $f = \tfrac13$ for intra and $\tfrac16$ for inter blocks (a dead zone), because a small coefficient costs more bits than the distortion it removes, and RDOQ (§3.9) makes that choice per coefficient with an explicit $D + \lambda R$. Real quantizers fold the integer transform's scaling into $\Delta$ and may weight frequencies with a scaling list (§3.7).

Example: $\mathrm{QP} = 28 \Rightarrow \Delta = 2^{4} = 16$. $C/\Delta$ is $1.09$ at DC and nowhere else above $0.09$, so with $f = \tfrac16$

$$
\ell = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
$$

Sixteen real numbers became one integer. The AC coefficients were noise and are gone: that is the compression.

<a id="codec-step-8"></a>

**Step 8 · Entropy coding.** $(\text{syntax: split, mode or } \mathbf v - \mathbf v^{p}, \text{ coded-block flag},\ \ell) \;\longrightarrow\; \text{bits}, \qquad R = \sum_{k} -\log_2 p_k(b_k)$

CABAC (§3.8) does this in three sub-steps. *Binarize*: scan $\ell$ in zigzag order and turn everything into binary decisions (bins): a coded-block flag, then per position a significance bin, and for each significant coefficient a last-in-block bin, a greater-than-one bin, the remaining magnitude in unary or Exp-Golomb, and a sign. *Model*: each bin is coded with a context, a probability estimate chosen from what was already coded (position in the block, the neighbouring blocks' flags) and updated after every bin:

$$
p \leftarrow p + \frac{b - p}{2^{w}} \qquad (w \approx 4 \text{ to } 7;\ \text{H.264's 64-state table is this recursion in disguise, VVC keeps two rates})
$$

*Arithmetic-code*: a bin with probability $p$ costs $-\log_2 p$ bits, so a bin that was $90\%$ predictable costs $0.15$ bit and a $50\%$ bin costs $1$ bit. Bit counts are fractional, and there is no code table: the probabilities are the code. Sign and remainder bins use a bypass mode at $p = \tfrac12$. AV1 codes multi-symbol alphabets directly instead of binarizing.

Example, with probabilities fixed for the example ($p(\text{sig} = 1) = 0.2$, $p(\text{last} = 1) = 0.4$, $p(\text{gt1} = 1) = 0.3$, coded-block flag and bypass bins at $0.5$): the one level of the inter block costs

$$
R_{\ell} = \underbrace{1.00}_{\text{cbf} = 1} + \underbrace{2.32}_{\text{sig} = 1} + \underbrace{1.32}_{\text{last} = 1} + \underbrace{0.51}_{\text{gt1} = 0} + \underbrace{1.00}_{\text{sign}} = 6.16 \text{ bits}
$$

plus about $2$ bits for the mode flags and $\mathbf v - \mathbf v^{p} = (0, 0)$: $R_{\text{inter}} \approx 8.2$ bits for $16$ pixels. The intra-DC candidate's levels, $(4, -3, 0, 0, 0, 0, 1)$ in scan order, cost $24$ bits: $R_{\text{DC}} \approx 26$.

<a id="codec-step-9"></a>

**Step 9 · Reconstruction inside the encoder (the closed loop).** $(\ell, P) \;\longrightarrow\; \hat C = \Delta\, \ell \;\longrightarrow\; \hat E = T^{\top} \hat C\, T \;\longrightarrow\; \hat B = \operatorname{clip}_{[0, 255]}\big(P + \operatorname{round}(\hat E)\big), \qquad D = \lVert B - \hat B \rVert_2^2$

This is the decoder's arithmetic, run inside the encoder, and it is $\hat B$, not $B$, that is written into $\tilde F_t$ and becomes the reference of frame $t + 1$. The decoder never sees $B$. If the encoder predicted frame $t + 1$ from $F_t$ while the decoder predicts it from $\tilde F_t$, the two would disagree by $P(F_t) - P(\tilde F_t)$, frame $t + 2$ would be predicted from that, and the mismatch would add up frame after frame until the next I-frame: drift. The same rule fixes the order of everything downstream: the in-loop filters of step 11 must run before the picture is used as a reference, exactly as in the decoder.

$$
\hat C[0,0] = 16 \cdot 1 = 16, \qquad \hat E = T^{\top} \hat C\, T = \begin{pmatrix} 4 & 4 & 4 & 4 \\ 4 & 4 & 4 & 4 \\ 4 & 4 & 4 & 4 \\ 4 & 4 & 4 & 4 \end{pmatrix}
$$

$$
\hat B = P^{\text{inter}} + 4 = \begin{pmatrix} 94 & 95 & 123 & 125 \\ 94 & 96 & 124 & 125 \\ 95 & 96 & 125 & 126 \\ 95 & 97 & 125 & 126 \end{pmatrix}, \qquad D = \lVert B - \hat B \rVert_2^2 = 6
$$

The brightness change was coded exactly (a DC of $16$ spreads as $16 / N = 4$ per pixel) and six pixels are off by one: the noise the quantizer threw away is the loss.

<a id="codec-step-10"></a>

**Step 10 · Mode decision (RDO).** $\{(D_k, R_k)\}_{k \in \text{candidates}} \;\longrightarrow\; k^{*} = \underset{k}{\arg\min}\ D_k + \lambda R_k, \qquad \lambda = 0.57 \cdot 2^{(\mathrm{QP} - 12)/3}$

Every decision (split or not, intra mode, motion vector, merge or skip, transform type, whether to zero a coefficient) is the same comparison: run steps 5–9 for each candidate, measure its $D$ and $R$, keep the cheapest $J$ (§3.9). $\lambda$ is the exchange rate between one unit of squared error and one bit; it is tied to QP so that at high QP, where bits are scarce, the encoder prefers cheap modes.

Example, $\mathrm{QP} = 28$, $\lambda = 0.57 \cdot 2^{16/3} = 23$:

| Candidate | $D$ | $R$ (bits) | $J = D + 23 R$ |
|---|---|---|---|
| inter, $\mathbf v = (0, -1)$, residual coded | 6 | 8.2 | **194** |
| skip: $\mathbf v = \mathbf v^{p}$, no residual, $\hat B = P^{\text{inter}}$ | 310 | 1 | 333 |
| intra DC | 119 | 26 | 717 |

The coded inter block wins. At $\mathrm{QP} = 34$ the same block gives $\Delta = 32$ and $\ell = \lfloor 17.5 / 32 + \tfrac16 \rfloor = 0$: the brightness change no longer survives the quantizer, the candidate collapses into skip, and the block costs one bit with $D = 310$. Same pixels, different QP, different decision.

<a id="codec-step-11"></a>

**Step 11 · In-loop filters.** $\tilde F_t^{\text{raw}} \;\longrightarrow\; \tilde F_t = \text{ALF}\big(\text{SAO}\big(\text{deblock}(\tilde F_t^{\text{raw}})\big)\big)$, then $\tilde F_t$ enters the reference buffer

Each block was quantized on its own, so block edges show seams (§3.10). The deblocking filter looks at the two pixels on each side of every block edge, $p_1, p_0 \mid q_0, q_1$, and smooths the edge only where it looks like a coding artifact rather than a real edge, with a strength from $bS \in \{0, \dots, 4\}$ (intra $>$ coded coefficients $>$ different motion $>$ same motion):

$$
\begin{aligned}
&\text{filter if } |p_0 - q_0| < \alpha(\mathrm{QP}),\quad |p_1 - p_0| < \beta(\mathrm{QP}),\quad |q_1 - q_0| < \beta(\mathrm{QP}) \\[6pt]
&\delta = \operatorname{clip}_{[-t_c,\, t_c]}\Big( \Big\lfloor \frac{4 (q_0 - p_0) + (p_1 - q_1) + 4}{8} \Big\rfloor \Big), \qquad p_0' = p_0 + \delta,\quad q_0' = q_0 - \delta
\end{aligned}
$$

(H.264's normal filter; $\alpha$, $\beta$ and $t_c$ are tables that grow with QP, so heavier quantization is filtered harder). SAO (HEVC) then adds a transmitted offset per pixel class, $\tilde F[i,j] \mathrel{+}= o_{\text{class}(i,j)}$, the classes being edge shapes (local minimum, maximum, slope) or intensity bands; ALF (VVC) is a transmitted Wiener filter; AV1 uses CDEF and loop restoration. All are *in-loop*: the decoder applies the same filters, so the filtered picture is the reference, and the encoder must filter identically before step 4 of the next frame.

Example: the left edge of $\hat B$ in row $8$ reads $(p_1, p_0 \mid q_0, q_1) = (91, 92 \mid 94, 95)$; $bS = 2$ (coded coefficients), and the thresholds at $\mathrm{QP} = 28$ (about $20$ and $7$) let all three tests pass: $\delta = \lfloor (8 - 4 + 4)/8 \rfloor = 1$, the edge becomes $(91, 93 \mid 93, 95)$, and the step of $2$ across it is gone.

<a id="codec-step-12"></a>

**Step 12 · Rate control.** $(R_{\text{target}},\ \text{buffer},\ \text{bits spent so far}) \;\longrightarrow\;$ a bit budget, then $\mathrm{QP}$ of the next frame or block, which feeds step 7.

One example is an R–$\lambda$ model used in HEVC encoder rate control; it is not a requirement of the codec:

$$
\text{bpp} = \frac{R_{\text{frame}}}{H\, W}, \qquad \lambda = a \cdot \text{bpp}^{\,b}, \qquad \mathrm{QP} = \big\lfloor 4.2005 \ln \lambda + 13.7122 + \tfrac12 \big\rfloor
$$

Here $R_{\text{frame}}$ is the frame's budget **in bits**, not bits/s. With illustrative parameters $a = 3.2003$ and $b = -1.367$, a 200 kbit budget for a $1920 \times 1080$ frame gives $\text{bpp} = 0.0965$, $\lambda \approx 78$, and $\mathrm{QP} \approx 32$. Actual bits feed back into later budgets and model updates; frame type, complexity, and block-level adjustments can change the chosen QP.

[§3.2: CBR, VBR, CQP, and buffers](#32-rate-control-cbr-vbr-cqp-and-buffers) explains how that budget is chosen and why it need not be the same for every frame.

<a id="codec-decoder"></a>

**The decoder.** $\text{bits} \;\longrightarrow\; \tilde F_t$

$$
\text{bits} \xrightarrow{\ \text{step 8}^{-1}\ } \big(m \text{ or } \mathbf v,\ \ell\big) \xrightarrow{\ \text{step 3 or 4 on } \tilde F\ } P \xrightarrow{\ \text{step 9}\ } \hat B = \operatorname{clip}\big(P + T^{\top} (\Delta\, \ell)\, T\big) \xrightarrow{\ \text{step 11}\ } \tilde F_t
$$

The decoder runs the inverse of step 8 and then steps 3 or 4, 9 and 11, with the same probability updates, the same $\tilde F_{t-1}$ and the same integer arithmetic, so it reproduces $\tilde F_t$ bit for bit. It never searches: no motion estimation, no mode decision, no rate control. That is why decoding is one to two orders of magnitude cheaper than encoding, and why a standard specifies only the decoder: steps 6–9 and 11 are fixed by the spec, while steps 2, 3, 4, 10 and 12 are the encoder's freedom and the reason two conformant encoders of the same standard differ by tens of percent in bitrate at equal quality.

<a id="codec-shapes"></a>

| Step | Input | Output | Example |
|---|---|---|---|
| 1 · Color | RGB frame, $H \times W \times 3$ | $Y$ ($H \times W$), $C^b, C^r$ ($\tfrac{H}{2} \times \tfrac{W}{2}$) | $(200, 120, 80) \to (134, 99, 170)$ |
| 2 · Partition | $F_t$ | blocks $\{B_k\}$ | $4 \times 4$ at $(8, 16)$ |
| 3 · Intra | $\mathbf t, \mathbf l$ | $P^{\text{intra}}_m$ | $P^{\text{DC}} = 93$ |
| 4 · Inter | $B, \tilde F_{t-1}$ | $\mathbf v^{*}, P^{\text{inter}}$ | $\mathbf v^{*} = (0, -1)$, SAD $70$ |
| 5 · Residual | $B, P$ | $E$ | $\approx 4$ everywhere |
| 6 · Transform | $E$ | $C$ | $C[0,0] = 17.5$ |
| 7 · Quantize | $C$, QP | $\ell$ | $\ell[0,0] = 1$, rest $0$ |
| 8 · Entropy code | syntax, $\ell$ | bits | $8.2$ bits |
| 9 · Reconstruct | $\ell, P$ | $\hat B$, $D$ | $D = 6$ |
| 10 · RDO | $\{(D_k, R_k)\}$ | $k^{*}$ | $J = 194$, inter |
| 11 · Filters | $\tilde F_t^{\text{raw}}$ | $\tilde F_t$ | $92, 94 \to 93, 93$ |
| 12 · Rate control | target, buffer | QP | $6\,\text{Mbit/s} \to \mathrm{QP}\ 32$ |

#### Coding: closed-loop I/P encoder `[TOY]`

**Task:** Compare intra/inter candidates for each block, encode the residual, and reconstruct the reference inside the encoder through inverse quantization. Explain why referencing the original frame causes drift.  
**Complexity:** For `B×B` blocks and search radius `S`, the leading full-search term is approximately `O(HW(2S+1)²)` pixel comparisons.  
**Pitfalls:** The next frame may reference only a reconstruction that the decoder also has; in-loop filter ordering must match; the true bit cost includes the mode, MV, headers, and probability tables.

```python
import numpy as np

def code_one_block(block, candidates, qstep, lam, estimate_bits):
    best = None
    for syntax, pred in candidates:          # intra mode or (ref, MV)
        coeff = dct2(block - pred)
        qcoeff = np.rint(coeff / qstep).astype(np.int32)
        recon_residual = idct2(qcoeff * qstep)
        recon = np.clip(np.rint(pred + recon_residual), 0, 255)
        distortion = np.sum((block - recon) ** 2)
        rate = estimate_bits(syntax, qcoeff)
        candidate = (distortion + lam * rate, syntax, qcoeff, recon)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1:]                           # syntax, qcoeff, reconstructed block

# Frame loop stores `recon`, not the original block, into the next reference.
```

[Full implementation](./video_encoding_pipeline.py) — includes I/P coding, in-loop reconstruction, toy deblocking, and an entropy payload; it has no decoder, B-frames, sub-pel prediction, or full RDO, and its DCT is fixed at 8×8.

## 3.2 Rate control: CBR, VBR, CQP, and buffers

**Rate control chooses how many bits to spend, then adjusts QP to approach that budget.** Higher QP usually means fewer bits and more distortion; lower QP usually means more bits and better quality. A controller can choose a frame-level QP and refine it for individual blocks.

<a id="codec-rc-modes"></a>

**CBR, VBR, and CQP: what is being controlled?**

| Mode | Example setting | What the encoder tries to do |
|---|---|---|
| **CBR — constant bitrate** | Target: 6 Mb/s | Keep the rate steady;<br>raise QP when bits are scarce. |
| **VBR — variable bitrate** | Average: 6 Mb/s<br>Maximum: 10 Mb/s | Spend more on complex scenes,<br>within average and peak limits. |
| **CQP — constant QP** | Base QP: 28 | Fixed quantization policy;<br>bitrate and visual quality vary. |

These are encoder policies, not different video codecs; exact controls depend on the encoder. CQP does not adjust QP to meet a bitrate target, but frame-type QP offsets may still apply. See [NVENC's rate-control modes](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/#rate-control).

**CBR does not mean every frame has the same size.** At 6 Mb/s and 30 fps:

$$
\text{average frame budget} = \frac{6{,}000{,}000}{30} = 200{,}000\ \text{bits} = 200\ \text{kbit}.
$$

Frames of **260, 170, and 170 kbit** still average 200 kbit. A larger frame can borrow some buffer headroom; later smaller frames make up for it. CBR constrains the rate over time, not each frame to exactly 200 kbit.

**Then how is VBR different?** VBR deliberately allows the rate to follow scene complexity over longer intervals. For example, an easy five-second scene at 4 Mb/s followed by a difficult five-second scene at 8 Mb/s averages 6 Mb/s. This can fit a VBR setup with sufficient peak allowance and buffer headroom. A tightly buffered 6 Mb/s CBR setup cannot sustain that 8 Mb/s scene for five seconds: it would accumulate 10 Mbit of excess data.

VBR does **not** require a minimum bitrate. A maximum is a rate constraint interpreted with a buffer and timing model, not a universal per-frame size limit. Setting the same tight rate and buffer limits for both modes makes their behavior more similar.

<a id="codec-vbv-notation"></a>

**Bitrate is a speed; buffer fullness is an amount.**

| Quantity | Unit | Meaning |
|---|---|---|
| Target bitrate | bits/s | Desired average, e.g. 6 Mb/s |
| Maximum bitrate / model rate | bits/s | Rate used to constrain bursts;<br>may differ from the average target |
| Buffer capacity | bits | Maximum amount, e.g. 600 kbit |
| Current fullness | bits | Changing state, e.g. 300 kbit now |
| Desired fullness, if used | bits | Internal control target,<br>not a mandatory CBR/VBR setting |

Setting a 6 Mb/s target does **not** mean keeping the buffer at a fixed water level. A desired fullness of 300 kbit also does not define a minimum or maximum bitrate. An **initial** fullness sets the starting state, not a level that must remain constant. See [x265's VBV settings](https://x265.readthedocs.io/en/stable/cli.html#cmdoption-vbv-bufsize). Here, kbit means 1,000 bits, not bytes.

<a id="codec-vbv-model"></a>

**The simplest buffer picture: bits waiting to leave.**

Imagine a sender-side virtual queue: encoding adds bits; delivery removes bits. For one frame interval, use this simplified accounting:

$$
B_{\text{next}} = \max\!\left(0,\ B_{\text{current}} + \text{frame bits} - \frac{C}{\text{fps}}\right).
$$

Here $B$ is the backlog in bits, and $C$ is the assumed drain rate in bits/s. A larger frame increases the backlog; a smaller frame lets it drain. This is a **teaching model**, not a full decoder-side VBV/HRD simulation: it ignores when bits arrive and leave within each interval.

<a id="codec-vbv-example"></a>

With $C = 6$ Mb/s, 30 fps, and an initial backlog of 300 kbit, each interval drains 200 kbit:

| Frame size | Backlog calculation | Backlog afterward |
|---|---|---|
| 260 kbit | 300 + 260 − 200 | 360 kbit |
| 170 kbit | 360 + 170 − 200 | 330 kbit |
| 170 kbit | 330 + 170 − 200 | 300 kbit |

The buffer moves even in CBR. It records the accumulated mismatch between producing and draining bits.

**Where does $C$ come from?** It is a configured model rate, not bandwidth the encoder magically knows. A live application's transport controller can estimate the available sending rate and update the encoder's budget. In VBR, the average target, the VBV rate limit, and actual network throughput need not be equal.

**How does this change QP?**

1. Estimate the next frame's complexity and start with a bit budget.
2. Adjust that budget using bits already spent and available buffer headroom.
3. Choose QP from a model of how many bits this frame will need.
4. Encode, measure the actual bits, update the buffer/model, and repeat. Some encoders also adjust QP within the frame.

A frame of 260 kbit against a 200 kbit average is **not automatically a mistake**. The question is whether that extra spending is affordable:

- **Backlog = 100 kbit:** the example model predicts $100 + 260 - 200 = 160$ kbit. Plenty of room in a 600 kbit buffer; the controller can recover the extra spending later.
- **Backlog = 550 kbit:** it predicts $550 + 260 - 200 = 610$ kbit, beyond capacity. The controller must tighten the frame budget before committing those bits, usually by raising QP. In this simplified calculation, the budget must be at most 250 kbit.

A finite buffer is a hard limit; the average bitrate is a goal approached over time. Without a tight buffer limit, there is more freedom to compensate later—but consistently oversized frames still miss the bitrate target. There is no universal "260 kbit means QP + 1" rule. If QP limits cannot make a real-time stream fit, the application may also need lower resolution, lower frame rate, or frame dropping.

**What VBV adds.** VBV means *Video Buffering Verifier*: the encoder checks that its output fits a specified delivery/buffering model, rather than only achieving a good average. CBR and constrained VBR can both use it. The formal decoder-buffer view fills when encoded bits arrive and empties when pictures are decoded—the opposite direction from the sender-backlog example. Neither buffer is automatically the actual network queue. A larger buffer permits bigger bursts; buffer capacity divided by rate is a buffer timescale, not the entire end-to-end latency.

**One related mode: CRF.** CRF targets roughly consistent perceptual quality by varying QP; it is not CQP and does not specify an average bitrate. Adding maximum-rate and buffer constraints gives *capped CRF*. See [x265 rate-control options](https://x265.readthedocs.io/en/stable/cli.html#cmdoption-crf).

**Remember:** CBR regulates the rate closely, VBR gives scene complexity more freedom to change the rate, and CQP holds the quantization policy. A buffer limits how far short-term bit production can get ahead of delivery.

## 3.3 Frame types, GOP, and reference structures

**Q: What are I, P, B frames?**

| Type | Prediction | Size (rough) | Notes |
|------|-----------|--------------|-------|
| **I** | intra only | 1× (largest) | Self-contained. **IDR** additionally flushes the reference buffer → a true random-access point. A non-IDR I-frame does not. |
| **P** | one direction, ≥1 refs | ~0.3× | H.264+ allows multiple reference frames (up to 16) |
| **B** | bi-predictive, past + future | ~0.1× | Averages two predictions → lower residual. Requires reordering. |

**GOP** = the pattern, e.g. `IBBPBBP...` with GOP length 2 s (60 frames at 30 fps is typical for streaming).

- **Closed GOP**: no frame references anything outside the GOP → each GOP is independently decodable → the straightforward choice at ABR switch/splice/random-access boundaries.
- **Open GOP**: leading B-frames may reference the previous GOP → ~2–5% better compression, but complicates clean splicing and independent segment decode. Streaming can carry open GOPs when dependencies are controlled, but independently switchable segment boundaries are normally closed/IDR-aligned.

**Hierarchical B / pyramid** — the structure real encoders use:

```
Display order:   0    1    2    3    4    5    6    7    8
Type:            I    b    B    b    P?   ...            P
Layer:           0    2    1    2    0
Coding order:    0    8    4    2    1    3    ...
                 ↑    ↑    ↑
                 I    P    B(ref)   → then lower-importance b's

Layer 0 frames: lowest QP (best quality) — everything depends on them
Layer 2 frames: highest QP — nothing references them, errors don't propagate
```

**QP offsets by pyramid level** are one of the biggest single wins in modern rate control: spend bits where they propagate.

**Follow-ups:**
- *"Why do B-frames add latency?"* — Frame 4 must be encoded before frames 1–3, so the encoder must buffer future frames. Coding order ≠ display order → that's the whole reason **DTS** exists separately from **PTS**. In real-time (< 100 ms) you disable B-frames or use "low-delay B" (both refs in the past).
- *"What's a reference frame buffer / DPB?"* — Decoded Picture Buffer. Its size is capped by the level, and it's the memory that limits how many refs you can use. Long-term reference pictures can be pinned (useful for error resilience: "resend from the last known-good long-term ref").
- *"What's a scene cut?"* — Encoders detect it (SAD/histogram between frames) and force an I-frame. Placing an I-frame *at* the cut rather than 2 frames after saves substantial bits.

## 3.4 Intra prediction

Predict the current block from the reconstructed row above and column to the left, then code only the residual.

| Standard | Modes |
|----------|-------|
| H.264 | 4×4 and 8×8: 9 modes; 16×16: 4 modes (V, H, DC, Plane); chroma 8×8: 4 modes |
| HEVC | **35**: DC, Planar, 33 angular |
| VVC | **67** angular + wide-angle for non-square, plus **MIP** (matrix intra prediction — a tiny learned/tabulated linear predictor), **CCLM** (predict chroma from reconstructed luma) |
| AV1 | 8 directional with ±3 delta angles (56 total), plus DC / Paeth / 4 SMOOTH modes, **CFL** (chroma from luma), **intra block copy** (screen content), palette mode |

- **DC mode**: fill with the average of the neighbors. Good for flat regions.
- **Planar / SMOOTH**: bilinear interpolation between edges. Good for gradients — this is what stops banding on sky.
- **Angular**: extrapolate along a direction. Good for edges.

**Interview point:** intra prediction is why intra-frame video codecs beat JPEG so badly — JPEG has *no* spatial prediction at all beyond the DC delta.

#### Coding: DC / planar / angular intra prediction `[READY]`

**Task:** Predict an `N×N` block from reconstructed top/left references, compute the residual/SAD, and choose a mode; as a follow-up, use `D+λR`.  
**Complexity:** `O(N²)` time and output per mode; `O(MN²)` for `M` modes.  
**Pitfalls:** References must come from the reconstruction; top-left/top-right/bottom-left extension and unavailable neighbors; fixed-point interpolation, clipping, and tie-breaking.

```python
def dc_predict(top, left):
    n = len(top)
    dc = (sum(top) + sum(left) + n) // (2 * n)
    return [[dc] * n for _ in range(n)]

def planar_predict(top, left, top_right, bottom_left):
    n, out = len(top), [[0] * len(top) for _ in top]
    for y in range(n):
        for x in range(n):
            value = ((n - 1 - x) * left[y] + (x + 1) * top_right
                     + (n - 1 - y) * top[x] + (y + 1) * bottom_left + n)
            out[y][x] = value // (2 * n)
    return out

def sad(block, pred):
    return sum(abs(a - b) for ra, rp in zip(block, pred)
               for a, b in zip(ra, rp))
```

[Full implementation](./intra_prediction_mode.py) — also includes a simplified fixed-point angular predictor and SAD mode selection; it is not an HEVC bit-exact implementation.

## 3.5 Inter prediction and motion estimation

**Motion estimation (ME)** finds, for each block, the displacement into a reference frame that minimizes prediction cost. **Motion compensation (MC)** applies it.

**Cost function**: SAD (sum of absolute differences) for speed; SATD (Hadamard-transformed SAD) correlates better with post-transform bits; SSE for final RDO. Real encoders use SAD for the coarse search and SATD/RDO for refinement.

**Search strategies** (this is a classic "how would you make it faster" question):
- **Full search**: exhaustive over the search window. Optimal, prohibitively slow.
- **Three-step / diamond / hexagon search**: coarse-to-fine local descent. x264's `umh` (uneven multi-hexagon) is the practical default.
- **Predictive search**: start from the median of neighboring MVs, the collocated MV in the previous frame, and (0,0). Because motion fields are smooth, the predictor is usually nearly right — this is a bigger speedup than any search-pattern trick.
- **Hierarchical / pyramid ME**: estimate on a downsampled image, refine at full resolution. Handles large motion cheaply.
- **Early termination**: stop when cost drops below a threshold.

**Sub-pixel motion**: real motion isn't an integer number of pixels.
- H.264: **quarter-pel** luma. Half-pel via a 6-tap Wiener filter `(1,−5,20,20,−5,1)/32`, quarter-pel by bilinear averaging. Chroma goes to 1/8-pel.
- HEVC: quarter-pel with **8-tap** luma / 4-tap chroma filters (better, no two-stage rounding).
- AV1/VVC: eighth-pel modes, switchable interpolation filters.

Sub-pel is worth a lot — typically 20%+ bitrate at the same quality versus integer-pel only.

**Motion vector coding**: MVs are correlated with neighbors, so you code a *difference* from a predictor.
- H.264: median of left/top/top-right MVs; `mvd` is coded.
- HEVC: **AMVP** (advanced MVP — a candidate list, you signal the index) and **Merge mode** (inherit a neighbor's full motion info, code no MVD at all). **Skip** = merge with zero residual — the cheapest possible block, and why static scenes cost almost nothing.
- AV1: dynamic reference MV list, ranked by how often each candidate appears among neighbors.

**Advanced inter tools to name-drop:**
- **Bi-prediction / weighted prediction**: average two references; weighted prediction handles fades and flashes (huge win on cross-dissolves).
- **Affine motion (VVC, AV1 warped motion)**: 4- or 6-parameter model handles zoom, rotation, shear — translational MVs can't.
- **OBMC (AV1)**: overlapped block MC, blends predictions from neighbors' MVs to soften block edges.
- **DMVR / BDOF (VVC)**: decoder-side MV refinement and bi-directional optical flow — the decoder refines motion itself, so no extra bits.
- **Global motion (AV1)**: one warp per frame for camera pan/zoom.
- **Compound prediction / GPM (VVC)**: non-rectangular partitions along a geometric line, for objects whose boundary isn't block-aligned.

#### Coding: full-search block motion estimation `[READY]`

**Task:** Use SAD to find `(dy,dx)` for the current block within a reference search window, and also return the motion-compensated prediction.  
**Complexity:** `O((2S+1)²B²)` for one `B×B` block; approximately `O(HW(2S+1)²)` for a full frame.  
**Pitfalls:** Define the MV direction; candidate upper bounds must be inclusive; cost must be a scalar; handle partial blocks/padding; use deterministic tie-breaking for equal costs.

```python
import numpy as np

def motion_search(cur, ref, by, bx, block_size, search_range):
    h, w = ref.shape
    best = (float("inf"), by, bx)
    y0, y1 = max(0, by-search_range), min(h-block_size, by+search_range)
    x0, x1 = max(0, bx-search_range), min(w-block_size, bx+search_range)
    for ry in range(y0, y1 + 1):
        for rx in range(x0, x1 + 1):
            candidate = ref[ry:ry+block_size, rx:rx+block_size]
            cost = float(np.sum(np.abs(cur - candidate)))
            if cost < best[0]:
                best = (cost, ry, rx)
    _, ry, rx = best
    return ref[ry:ry+block_size, rx:rx+block_size], (ry-by, rx-bx)
```

[Full implementation](./video_encoding_pipeline.py) — uses `inter_predict`; the scalar cost and bounds in the standalone [motion_vector.py](./motion_vector.py) are not yet fixed, so do not use it as the primary answer.

#### Coding: sub-pel motion compensation + SATD `[TODO]`

**Task:** First implement half/quarter-pel prediction with bilinear sampling, then compare candidates using 4×4 Hadamard SATD.  
**Complexity:** Each candidate remains `O(B²)`; total search complexity is multiplied by the number of fractional positions.  
**Pitfalls:** Fixed-point MV units, border extension, forward/backward warp conventions, and filter rounding must match between encoder and decoder.

```python
def subpel_mc(reference, x_qpel, y_qpel, block_size):
    """TODO: sample a B×B block at quarter-pixel coordinates."""
    raise NotImplementedError

def satd4x4(residual):
    """TODO: separable 4×4 Hadamard transform, then sum abs(coefficients)."""
    raise NotImplementedError
```

Full implementation: TODO (suggested file: `prediction.py`).

## 3.6 Block partitioning across standards

```
H.264:  Macroblock 16×16
          → 16×16, 16×8, 8×16, 8×8
          → sub-8×8: 8×4, 4×8, 4×4

HEVC:   CTU (up to 64×64)
          → quadtree → CU (64→8)
          → PU (prediction): 2Nx2N, 2NxN, Nx2N, NxN, + AMP (asymmetric: nLx2N etc.)
          → TU (transform): its own quadtree, decoupled from PU

VVC:    CTU 128×128 → QTMTT (quadtree + binary + ternary splits) → CU is also PU and TU
          Ternary split (1:2:1) is the distinctive VVC tool

AV1:    Superblock 128×128 → recursive 10-way partition (incl. T-shaped) down to 4×4
```

**Why bigger blocks help at high resolution:** a 4K frame has 4× the pixels of 1080p but roughly the same *number of objects*. Signaling a mode + MV per 16×16 block costs 4× more headers for the same content. Larger CTUs amortize that. This is the single biggest reason HEVC beats H.264 more at 4K than at 480p — and a great answer to *"why is HEVC's gain content-dependent?"*

## 3.7 Residual transforms and quantization

After prediction, video codecs transform the **residual** rather than the original pixels, then quantize the resulting coefficients. The transform family and quantizer must be specified precisely enough for encoder and decoder to reconstruct the same reference samples.

**Modern transform additions you should name:**
- **Integer transforms**: H.264 onward use integer approximations of the DCT so encoder and decoder produce *bit-exact* identical results. Floating-point DCT would drift between implementations and break the prediction loop.
- **DST-VII**: HEVC/VVC use it for small intra luma blocks. Intra prediction copies from the top/left edge, so residual error *grows* with distance from the edge — DST's basis functions start near zero and grow, matching that profile better than DCT.
- **Multiple Transform Selection (MTS)** in VVC and **ADST/flipADST/identity** in AV1: the encoder picks the transform per block by RDO. The identity transform is for screen content, where a "residual" is often already sparse.

**Video quantization** is parameterized by **QP** rather than a single JPEG-style table:
```
step size ≈ 2^((QP − 4) / 6)      (H.264/HEVC convention)
QP + 6  →  step doubles  →  roughly half the bitrate
QP + 1  →  step × 2^(1/6) ≈ ×1.12  →  ~12% step increase
```
Codecs may also use a **scaling list / quantization matrix** for per-frequency weighting. It is off by default in many streaming encodes because it can hurt PSNR while helping perception.

**Dead zone**: real encoders use a wider quantization interval around zero than elsewhere (`round(|c|/q + f)` with f < 0.5, e.g. f = 1/3 for intra, 1/6 for inter in the H.264 reference encoder and x264). Sacrificing near-zero coefficients is cheap in distortion and very profitable in rate.

## 3.8 Entropy coding and CABAC

### Huffman vs arithmetic coding

**Huffman assigns an integer number of bits per symbol — minimum 1.** The information-theoretic optimum is `−log₂(p)`:

```
p = 0.50 → optimal 1.00 bit  → Huffman 1 bit  ✓
p = 0.25 → optimal 2.00 bits → Huffman 2 bits ✓
p = 0.90 → optimal 0.15 bits → Huffman 1 bit  ✗ 6.7× overhead
```

Huffman is optimal only when all probabilities are powers of ½. It breaks on **skewed** distributions — exactly what codec syntax elements look like (`skip_flag` is true ~95% of the time in static scenes).

**Concrete: {A: 0.9, B: 0.1}**
```
H = −0.9 log₂(0.9) − 0.1 log₂(0.1) ≈ 0.469 bits/symbol
Huffman: A→0, B→1 (forced) → 1.0 bit/symbol → 113% excess

100 A's:  Huffman 100 bits   vs   optimal ≈ 15 bits
```

**Arithmetic coding** encodes the *entire sequence* as one number in [0,1), narrowing an interval by each symbol's probability:

```
Encode "A A B" with P(A)=0.9, P(B)=0.1:

Start:     [0.000, 1.000)   A: [0.0, 0.9)    B: [0.9, 1.0)
After A:   [0.000, 0.900)   A: [0.0, 0.81)   B: [0.81, 0.9)
After A:   [0.000, 0.810)   A: [0.0, 0.729)  B: [0.729, 0.81)
After B:   [0.729, 0.810)   width = 0.081

Cost = −log₂(0.081) ≈ 3.6 bits = −log₂(0.9·0.9·0.1)   ← exactly the sum of optimal lengths
```

Fractional bits per symbol, amortized across the sequence. For long sequences the rate approaches the true entropy.

| | Huffman | Arithmetic |
|---|---|---|
| Unit | per symbol | whole sequence |
| Bits/symbol | integer ≥ 1 | fractional (−log₂ p) |
| Reaches entropy? | only if p = 2^−k | yes, asymptotically |
| Skewed p | wastes bits | handles efficiently |
| Adaptivity | needs table retransmit | context models update online |
| Used in | JPEG baseline, CAVLC | CABAC (H.264/HEVC/VVC), AV1 range coder, learned codecs |

#### Coding: integer arithmetic coder `[READY]`

**Task:** Implement range updates and E1/E2/E3 renormalization, and test `decode(encode(x)) == x`.  
**Complexity:** `O(N + output_bits)` time and `O(output_bits + alphabet)` space.  
**Pitfalls:** The `+1/-1` at interval endpoints; E3 pending bits; termination; the decoder needs the same model, precision, and symbol count; this is not CABAC.

```python
def narrow(low, high, cum_lo, cum_hi, total):
    width = high - low + 1
    new_high = low + width * cum_hi // total - 1
    new_low  = low + width * cum_lo // total
    return new_low, new_high

def renormalize(low, high, pending, bits, precision=32):
    half, quarter = 1 << (precision - 1), 1 << (precision - 2)
    three_quarters = 3 * quarter

    def emit(bit):
        nonlocal pending
        bits.append(str(bit))
        bits.extend(str(1 - bit) for _ in range(pending))
        pending = 0

    while True:
        if high < half:                    # E1: interval in lower half
            emit(0)
        elif low >= half:                  # E2: interval in upper half
            emit(1); low -= half; high -= half
        elif low >= quarter and high < three_quarters:  # E3
            pending += 1; low -= quarter; high -= quarter
        else:
            break
        low, high = low << 1, (high << 1) + 1
    return low, high, pending
```

[Full implementation](./arithmetic_coding.py) — includes a matching encoder/decoder, termination procedure, and static frequency model.

### CABAC — the answer they actually want

**Context-Adaptive Binary Arithmetic Coding** (H.264 High profile onward, all of HEVC/VVC). Three stages:

1. **Binarization** — every syntax element becomes a bin string (unary, truncated unary, Exp-Golomb, or fixed). Now the alphabet is binary, so the arithmetic coder only needs one probability per context.
2. **Context modeling** — each bin is coded with a context selected from neighboring already-decoded information (left and above block's mode, current depth, coefficient position). Each context holds an adaptively updated probability state (H.264: 64 states via a lookup table; VVC: multi-hypothesis with variable adaptation rate).
3. **Binary arithmetic coding** — a multiplication-free, table-driven range coder. There's also a **bypass mode** at fixed p = 0.5 for sign bits and suffix bits, which is much faster.

**Why it wins**: ~9–14% bitrate saving over CAVLC at the same quality. **Why it hurts**: it is inherently serial (each bin depends on the previous state), so it's the hardest part of a decoder to parallelize. HEVC's WPP and tiles exist partly to break CABAC dependencies.

**AV1** uses a **multi-symbol arithmetic coder** (Daala-derived) instead: it codes an M-ary symbol directly rather than binarizing, which cuts the number of coder invocations and improves throughput.

> **Question you'll get:** *"Why doesn't JPEG use arithmetic coding?"* — It's actually in the standard (Annex), but patents (IBM/Mitsubishi QM-coder) killed adoption, and Huffman was cheap enough in 1992. Those patents expired; JPEG XL and AVIF now use modern entropy coders (ANS / range coding).
>
> *"What is ANS?"* — Asymmetric Numeral Systems (Duda). Same compression as arithmetic coding but with table-driven, near-Huffman speed. Used in Zstandard, JPEG XL, AV1's earlier drafts. Good name-drop.

## 3.9 RDO — rate-distortion optimization

Every encoder decision (mode, partition, MV, whether to zero a coefficient) minimizes:

```
J = D + λ·R
```

- `D` = distortion (SSE, or SATD in fast modes)
- `R` = estimated bits, from the actual entropy coder's context states
- `λ` = Lagrange multiplier tying rate to distortion

**λ is derived from QP.** In HEVC's reference encoder:
```
λ ≈ 0.57 · 2^((QP − 12)/3)        (scaled per slice type / pyramid level)
```
Intuition: at high QP (low quality), bits are precious, so λ is large and the encoder aggressively picks cheap modes. At low QP, λ is small and it will spend bits for small quality gains.

**Interview-grade insight:** the reason RDO works at all is that D and R for a *single* decision are approximately independent of other blocks' decisions, so greedy per-block optimization is near-optimal. Where that breaks (a block that will be referenced by 100 future blocks) is exactly what **mb-tree / cu-tree** fixes: a lookahead that lowers the QP of blocks that many future blocks will reference.

**Trellis quantization / RDOQ**: instead of `round(c/q)`, search over candidate quantized values per coefficient (including forcing to zero) with a Viterbi trellis, minimizing `D + λR` for the whole block. Costs encode time, buys 5–10% bitrate. Turn it off for real-time.

#### Coding: RDO and exact bit allocation under a budget `[TOY]`

**Task:** First use `argmin(D+λR)` for single-block mode selection; then treat each block's `(D,R)` options as a multiple-choice knapsack problem and minimize total distortion under a global budget.  
**Complexity:** For `N` blocks, at most `M` modes per block, and budget `B`, the DP takes `O(NBM)` time and `O(NB)` space (or `O(B)` when only the cost is required).  
**Pitfalls:** Rate must be a nonnegative integer for this DP; distinguish `≤B` from `==B`; report an error when no feasible path exists; a λ sweep is the more common engineering approach for a continuous budget.

```python
def choose_rd(modes, lam):                 # mode = (distortion, bits, payload)
    return min(modes, key=lambda m: m[0] + lam * m[1])

def allocate_bits(modes_per_block, budget):
    inf = float("inf")
    dp, parents = [inf] * (budget + 1), []
    dp[0] = 0
    for modes in modes_per_block:
        nxt, parent = [inf] * (budget + 1), [None] * (budget + 1)
        for used, old_dist in enumerate(dp):
            if old_dist == inf:
                continue
            for mode_i, (dist, bits) in enumerate(modes):
                new_bits = used + bits
                if new_bits <= budget and old_dist + dist < nxt[new_bits]:
                    nxt[new_bits] = old_dist + dist
                    parent[new_bits] = (used, mode_i)
        dp, parents = nxt, parents + [parent]

    used = min(range(budget + 1), key=dp.__getitem__)
    if dp[used] == inf:
        raise ValueError("budget is infeasible")
    picked = []
    for parent in reversed(parents):
        used, mode_i = parent[used]
        picked.append(mode_i)
    return picked[::-1]
```

[Full implementation](./video_codec_components.py) — the recurrence exists, but the function has an extraneous `self`, and the module's Huffman demo causes import failure; it is therefore labeled only `[TOY]`.

## 3.10 Compression artifacts and in-loop filters

Quantization error isn't random — it's structured, which is why artifacts have names.

**Artifact 1 — Blocking.** Each block quantized independently → discontinuities at boundaries, worst in smooth regions.
```
Original smooth gradient:          After heavy quantization:
░░░▒▒▒▓▓▓███                      ░░░░│▒▒▒▒│▓▓▓▓│████
                                        ↑ visible seams
```
→ **Deblocking filter** (H.264/HEVC/AV1/VVC, in-loop). Adaptive by QP and *boundary strength*: a boundary between two intra blocks with different MVs gets filtered hard; a boundary that is a real object edge is left alone. This is the classic "don't blur real edges" heuristic.

**Artifact 2 — Ringing (Gibbs).** Zeroing high-frequency coefficients means reconstructing a sharp edge from low frequencies only → oscillations near the edge.
```
Original edge:     |‾‾‾‾‾‾
After truncation:  ~‾~‾‾‾‾   ← ripples
```
→ **SAO Edge Offset** (HEVC): classify each pixel by its relation to neighbors (local min, local max, slope) and add a signalled offset per class — lift the dips, suppress the peaks. → **CDEF** (AV1): detects the dominant edge direction per 8×8 and applies a constrained low-pass *along* it, keeping the edge sharp. → **ALF** (VVC): a Wiener-style adaptive loop filter with coefficients signalled per frame, chosen by classifying local gradients.

**Artifact 3 — Banding / contouring.** Smooth gradients (sky, skin) round to a few discrete levels → visible stair-steps.
```
Smooth:   ░░░░▒▒▒▒▓▓▓▓████
Banded:   ░░░░░░░▒▒▒▒▒▒████
```
→ **SAO Band Offset** (HEVC): split the value range into 32 bands, signal offsets for the 4 most-populated bands per CTU. → **Higher bit depth** (10-bit encoding of 8-bit content). → **Adaptive quantization**: give flat regions *lower* QP because that's where banding shows. → **Dithering / grain** at the display end.

**Artifact 4 — Mosquito noise.** High-frequency noise around sharp edges that *flickers* frame to frame, because ME picks slightly different predictors each frame. Deblocking + SAO reduce it; stable references matter more.

**Artifact 5 — Flicker / pumping.** Quality oscillates with the GOP: the I-frame is crisp, the frames just before the next I-frame are soft. Caused by rate control draining bits. Fixed with better bit allocation and hierarchical QP offsets.

**Artifact 6 — Film grain loss.** Grain is high-entropy and expensive; encoders kill it, and the result looks plasticky. → **AV1 film grain synthesis**: strip grain before encoding, transmit a parametric grain model, re-synthesize at the decoder *out of loop*. Netflix pushed hard for this. Great thing to mention in a Netflix interview.

**Where filters live:**
```
Encode loop per frame:
  original block
      ↓ prediction
      ↓ transform + quantization         ← lossy, artifacts born here
      ↓ inverse quant + inverse transform
      ↓ add prediction back
      ↓ Deblocking          ── block boundaries      (H.264/HEVC/AV1/VVC)
      ↓ CDEF                ── ringing, directional  (AV1)
      ↓ SAO                 ── ringing + banding     (HEVC)
      ↓ ALF                 ── adaptive Wiener       (VVC)
      ↓ Loop Restoration    ── Wiener / self-guided  (AV1)
      ↓ store as reference frame
                            ── Film grain synthesis is OUT of loop (AV1)
```

**All in-loop filters must run inside the encoder loop** so the encoder's reference matches the decoder's exactly. Otherwise prediction diverges and the picture visibly corrupts over the GOP. That "why in-loop?" question is asked constantly — the answer is always *drift*.

| Filter | H.264 | HEVC | AV1 | VVC |
|--------|-------|------|-----|-----|
| Deblocking | ✓ | ✓ | ✓ | ✓ |
| SAO | ✗ | ✓ | ✗ | ✗ |
| CDEF | ✗ | ✗ | ✓ | ✗ |
| Loop Restoration (Wiener/SGR) | ✗ | ✗ | ✓ | ✗ |
| ALF / CC-ALF | ✗ | ✗ | ✗ | ✓ |
| LMCS (luma mapping + chroma scaling) | ✗ | ✗ | ✗ | ✓ |
| Film grain synthesis (out of loop) | ✗ | SEI only | ✓ | SEI |


---

*Continue to [02_ml_domain.md](02_ml_domain.md) for ML and generative compression, and [05_networking_streaming.md](05_networking_streaming.md) for delivery.*
