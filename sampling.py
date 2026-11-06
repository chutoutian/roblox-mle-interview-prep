


def downsample(arr):
    
    m = len(arr)
    n = len(arr[0])

    

    pad_m = m % 2
    pad_n = n % 2

    if pad_m != 0:
        last_row = arr[-1]
        arr.append(last_row)
        m+=1

    if pad_n != 0:
        for i in range(len(arr)):
            arr[i].append(arr[i][-1])
        n+=1
    

    yuv = [[[0, 0, 0] for _ in range(n//2)] for _ in range(m//2)]

    for i in range(0, m, 2):
        for j in range(0, n, 2):
            r1, g1, b1 = arr[i][j]
            r2, g2, b2 = arr[i][j+1]
            r3, g3, b3 = arr[i+1][j]
            r4, g4, b4 = arr[i+1][j+1]

            y1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
            y2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
            y3 = 0.299 * r3 + 0.587 * g3 + 0.114 * b3
            y4 = 0.299 * r4 + 0.587 * g4 + 0.114 * b4

            u = -0.14713 * (r1 + r2 + r3 + r4)/4 - 0.28886 * (g1 + g2 + g3 + g4)/4 + 0.436 * (b1 + b2 + b3 + b4)/4
            v = 0.615 * (r1 + r2 + r3 + r4)/4 - 0.51499 * (g1 + g2 + g3 + g4)/4 - 0.10001 * (b1 + b2 + b3 + b4)/4

            yuv[i//2][j//2] = [ (y1+y2+y3+y4)/4 , u, v]
    return yuv

def upsample(arr):
    m = len(arr)
    n = len(arr[0])

    rgb = [[[0, 0, 0] for _ in range(n*2)] for _ in range(m*2)]

    for i in range(m):
        for j in range(n):
            y, u, v = arr[i][j]

            r = y + 1.13983 * v
            g = y - 0.39465 * u - 0.58060 * v
            b = y + 2.03211 * u

            rgb[i*2][j*2] = [r, g, b]
            rgb[i*2][j*2+1] = [r, g, b]
            rgb[i*2+1][j*2] = [r, g, b]
            rgb[i*2+1][j*2+1] = [r, g, b]

    return rgb
def upsample_bilinear_2x(arr):
    m = len(arr)
    if m == 0:
        return []
    n = len(arr[0])
    if n == 0:
        return [[]]

    out_h, out_w = m * 2, n * 2
    out = [[[0.0, 0.0, 0.0] for _ in range(out_w)] for _ in range(out_h)]

    # 
    for i2 in range(out_h):
        # Map output y to source y
        y = i2 / 2.0
        y0 = int(y)
        y1 = min(y0 + 1, m - 1)
        wy = y - y0  # in [0,1]

        for j2 in range(out_w):
            # Map output x to source x
            x = j2 / 2.0
            x0 = int(x)
            x1 = min(x0 + 1, n - 1)
            wx = x - x0  # in [0,1]

            # Bilinear for each channel
            for c in range(3):
                v00 = arr[y0][x0][c]
                v01 = arr[y0][x1][c]
                v10 = arr[y1][x0][c]
                v11 = arr[y1][x1][c]

                top = v00 * (1.0 - wx) + v01 * wx
                bot = v10 * (1.0 - wx) + v11 * wx
                out[i2][j2][c] = top * (1.0 - wy) + bot * wy

    return out
    
