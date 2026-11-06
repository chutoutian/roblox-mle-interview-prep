#######

from collections import Counter
from heapq import heappop, heappush
from unittest import result

import numpy as np


## first calculate PSNR

raw_image = np.array([[1,2,3], [3,4,5], [7,8,9] ], dtype = np.float32)


def psnr(raw_image, processed_image):
    

    # psnr = 10 * log10(MAX^2 / MSE)

    mse = np.mean( (raw_image - processed_image) ** 2 )
    max_pixel = 255.0

    if mse == 0:
        return float('inf')
    else:
        psnr = 10 * np.log10(max_pixel ** 2 / mse)
    
    return psnr

raw_image = np.array([[1,2,3], [3,4,5], [7,8,9] ], dtype = np.float32)


###. [1,2,3]
###. [3,4,5]
###. [7,8,9]

# kernel
# [1/4, 1/4, 1/4]
# [1/4, 1/4, 1/4]
# [1/4, 1/4, 1/4]

def convlution(image, kernel, stride = 1):
    h, w = image.shape
    kh, kw = kernel.shape
    
    ## version 1: no padding
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    output = np.zeros( (out_h, out_w), dtype = np.float32)
    for i in range(out_h):
        for j in range(out_w):

            output[i,j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)
    
    
    ### version 2: with padding

    pad_h = kh // 2 #
    pad_w = kw // 2 #

    padded_image = np.zeros( (h + 2*pad_h, w+ 2*pad_w), dtype = np.float32)
    
    padded_image[pad_h:pad_h+h, pad_w:pad_w+w] = image
    # padding should be the value of the border pixel, not zero
    padded_image[:pad_h, pad_w:pad_w+w] = image[0:pad_h, :]
    padded_image[pad_h+h:, pad_w:pad_w+w] = image[h-pad_h:, :]
    padded_image[:, :pad_w] = padded_image[:, pad_w:pad_w+1]
    padded_image[:, pad_w+w:] = padded_image[:, pad_w+w-1:pad_w+w]

    out_h = (h + 2*pad_h - kh) // stride + 1
    out_w = (w + 2*pad_w - kw) // stride + 1

    output = np.zeros( (out_h, out_w), dtype = np.float32)
    for i in range(out_h):
        for j in range(out_w):

            output[i,j] = np.sum(padded_image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)

    return output

def convolve(
    image:   List[List[float]],
    kernel:  List[List[float]],
    stride:  int = 1,
    padding: str = "same",    # "valid" | "same"
) -> List[List[float]]:
    H,  W  = len(image),  len(image[0])
    kH, kW = len(kernel), len(kernel[0])

    # --- padding ---
    if padding == "same":
        pad_h, pad_w = (kH - 1) // 2, (kW - 1) // 2
        padded = [[0.0] * (W + 2 * pad_w) for _ in range(H + 2 * pad_h)]
        for i in range(H):
            for j in range(W):
                padded[i + pad_h][j + pad_w] = image[i][j]
    else:                          # "valid": no padding
        pad_h, pad_w = 0, 0
        padded = image

    pH, pW = len(padded), len(padded[0])

    # --- output size ---
    out_H = (pH - kH) // stride + 1
    out_W = (pW - kW) // stride + 1

    result = [[0.0] * out_W for _ in range(out_H)]

    # --- slide kernel over image ---
    for i in range(out_H):
        for j in range(out_W):
            total = 0.0
            for ki in range(kH):
                for kj in range(kW):
                    total += padded[i * stride + ki][j * stride + kj] * kernel[ki][kj]
            result[i][j] = total

    return result

####.  [1,2,3]
####   [3,4,5]
####   [7,8,9]

#res 90 degree rotation
###. [7,3,1]
###. [8,4,2]
###. [9,5,3]


# first step: transpose the matrix
# [1,3,7]
# [2,4,8]
# [3,5,9]

# second step: reverse each row

def rotation(image, degree):
    
    h, w = image.shape

    # step 1: transpose
    for i in range(h):
        for j in range(i+1, w):
            image[i,j], image[j,i] = image[j,i], image[i,j]
    
    # step 2: reverse each row
    for i in range(h):
        image[i] = image[i][::-1]
    
    return image


def rotation_any_degree(image, degree):
    # this is more complicated, we can use inverse mapping to find the source pixel for each output pixel
    h, w = image.shape
    rad = np.radians(degree)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    new_h = int(abs(h * cos_a) + abs(w * sin_a))
    new_w = int(abs(h * sin_a) + abs(w * cos_a))

    cx_in, cy_in = (w - 1) / 2.0, (h - 1) / 2.0
    cx_out, cy_out = (new_w - 1) / 2.0, (new_h - 1) / 2.0

    res = np.zeros( (new_h, new_w), dtype = np.float32)

    for i in range(new_h):
        for j in range(new_w):
            x = j - cx_out
            y = i - cy_out

            src_x = int(x * cos_a + y * sin_a + cx_in)
            src_y = int(-x * sin_a + y * cos_a + cy_in)

            if 0 <= src_x < w and 0 <= src_y < h:
                res[i,j] = image[src_y, src_x]
    return result



    

def motion_vector(reference, image, block_size):
    # this is a very complex problem, we can use block matching algorithm to find the motion vector for each block
    w, h = image.shape
    res = np.zeros( (w//block_size, h//block_size, 2), dtype = np.float32)
    for i in range(0,w,block_size):
        for j in range(0, h ,block_size):
            best_asd = float('inf')
            best_mv = (0,0)
            for mv_x in range(-block_size, block_size+1):
                for mv_y in range(-block_size, block_size+1):
                    asd = 0.0
                    for k in range(block_size):
                        for l in range(block_size):
                            if 0<= i+mv_x < w and 0 <= j+mv_y <h:
                                reference_block = reference[i+mv_x+k, j+mv_y+l]
                                current_block = image[i+k, j+l]
                                asd += abs(reference_block - current_block)
                    if asd < best_asd:
                        best_asd = asd
                        best_mv = (mv_x, mv_y)  
            res[i//block_size, j//block_size] = best_mv
    return res


###. symbols = [0,0,0,0,0,1,1,2,3]
###   0: 5; 1: 2; 2: 1; 3: 1
###   0: 1, 1: 01, 2: 000, 3: 001

def huffman_encoding(symbols):
    
    symbol_count = Counter(symbols)
    heap = []
    order = 0
    for sym, c in symbol_count.items():
        heappush(heap, (int(c), order, int(sym)))
        order += 1
    
    while len(heap) > 1:
        f1, _, s1 = heappop(heap)
        f2, _, s2 = heappop(heap)
        heappush(heap, (f1+ f2, order, (s1,s2)))
        order += 1
    codebook = {}
    def dfs(node, path):
        if isinstance(node, int):
            codebook[node] = path
            return
        left, right = node
        dfs(left, path + "0")
        dfs(right, path + "1")
        return codebook
    root = heap[0][2] if heap else None
    dfs(root, "")



#. 50 51 51 0 0 0 
#. 50 51 0  0 0 0
#. 49 0  0  0 0 0
## 50 51 51 0 0 0
## 50 51 0  0 0 0
## 49 0  0  0 0 0
## even: from left bottom to right top; odd: from right top to left bottom
# 50 -> 51 -> 50 -> 49 -> 50

def zig_zag_encode(arr):
    m = len(arr)
    n = len(arr[0])
    lay = 0
    result = []
    for lay in range(m + n - 1):
        if lay % 2 == 0:
            # Even layer: traverse from bottom-left to top-right
            r = min(lay, m - 1)
            c = lay - r
            while r >= 0 and c < n:
                result.append(arr[r][c])
                r -= 1
                c += 1
        else:
            # Odd layer: traverse from top-right to bottom-left
            c = min(lay, n - 1)
            r = lay - c
            while c >= 0 and r < m:
                result.append(arr[r][c])
                r += 1
                c -= 1
    return result


