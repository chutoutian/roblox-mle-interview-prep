import math
import copy
from typing import List


def rotate_90(matrix: List[List[int]]) -> List[List[int]]:
    """Exact in-place 90° clockwise for square matrices (no padding needed)."""
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix


def rotate(matrix: List[List[int]], degrees: float, pad: int = 0) -> List[List[int]]:
    rows, cols = len(matrix), len(matrix[0])
    rad = math.radians(degrees % 360)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    # Output canvas large enough to hold every rotated pixel.
    # Subtract EPS before ceil to prevent floating-point drift (e.g. sin(π) ≈ 1.2e-16)
    # from inflating the canvas by 1 pixel on axis-aligned rotations.
    EPS = 1e-9
    new_cols = math.ceil(abs(cols * cos_a) + abs(rows * sin_a) - EPS)
    new_rows = math.ceil(abs(cols * sin_a) + abs(rows * cos_a) - EPS)

    cx_in,  cy_in  = (cols - 1) / 2.0, (rows - 1) / 2.0
    cx_out, cy_out = (new_cols - 1) / 2.0, (new_rows - 1) / 2.0

    result = [[pad] * new_cols for _ in range(new_rows)]

    for i in range(new_rows):
        for j in range(new_cols):
            # Offset from output center
            x = j - cx_out
            y = i - cy_out

            # Inverse rotation: where did this output pixel come from?
            src_c = round(x * cos_a + y * sin_a + cx_in)
            src_r = round(-x * sin_a + y * cos_a + cy_in)

            if 0 <= src_r < rows and 0 <= src_c < cols:
                result[i][j] = matrix[src_r][src_c]

    return result


if __name__ == "__main__":
    m = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

    for deg in [45, 90, 135, 180, 270, 360]:
        result = rotate(copy.deepcopy(m), deg)
        print(f"{deg}°:")
        for row in result:
            print(" ", row)
