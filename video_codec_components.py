from collections import Counter
from heapq import heappush, heappop
from typing import List


_LAST_CODEBOOK = None



from typing import List, Tuple


# dp[i][b]: minimum total distortion for the first i blocks (0..i-1)
# using exactly b bits.
# For every candidate (D, R) of block i, if the previous state is
# dp[i-1][b-R], transition with:
# dp[i][b] = min(dp[i][b], dp[i-1][b-R] + D).

# Initialization

# dp[0][0] = 0; every other state is infinity.
def rd_allocate_dp(self, modes: List[List[Tuple[int, int]]], B: int):
    # modes[i] = [(D, R), ...] for block i
    n = len(modes)
    INF = 10**18

    dp = [[INF] * (B + 1) for _ in range(n + 1)]
    parent = [[(-1, -1)] * (B + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for b in range(B + 1):
            for m_idx, (D, R) in enumerate(modes[i - 1]):
                if b >= R and dp[i - 1][b - R] != INF:
                    val = dp[i - 1][b - R] + D
                    if val < dp[i][b]:
                        dp[i][b] = val
                        parent[i][b] = (b - R, m_idx)

    best_b = min(range(B + 1), key=lambda b: dp[n][b])
    best_D = dp[n][best_b]
    pick = [0] * n
    b = best_b
    for i in range(n, 0, -1):
        pb, m_idx = parent[i][b]
        pick[i - 1] = m_idx
        b = pb
    return pick, best_D, best_b
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

def huffman_encoding(symbols):
    """
    Build a Huffman codebook and the corresponding binary tree.

    Interview-style simplification:
      - returns only the codebook {symbol -> bitstring}
      - stores it in a module global so huffman_decoding(bits) can decode without
        shipping a separate tree/codebook argument.
    """
    symbol_count = Counter(symbols)
    heap = []
    tie = 0
    for sym, c in symbol_count.items():
        # (freq, tie_breaker, node): tie_breaker avoids comparing int vs tuple when freqs tie.
        heappush(heap, (int(c), tie, int(sym)))
        tie += 1
    
    print("heap:", heap)
    while len(heap) > 1:
        f1, _, n1 = heappop(heap)
        f2, _, n2 = heappop(heap)
        heappush(heap, (f1 + f2, tie, (n1, n2)))
        tie += 1
    
    print("heap after merge:", heap)
    def dfs(node, path="", codebook=None):
        if node is None:
            return codebook
        if isinstance(node, int):
            codebook[node] = path
            return codebook
        left, right = node
        print("left:", left, "right:", right)
        dfs(left, path + "0", codebook)
        dfs(right, path + "1", codebook)
        return codebook

    root = heap[0][2] if heap else None
    if isinstance(root, int):
        # Single-symbol alphabet: ensure a non-empty code so encoding/decoding is well-defined.
        codebook = {root: "0"}
    else:
        codebook = dfs(root, {})

    global _LAST_CODEBOOK
    _LAST_CODEBOOK = codebook
    return codebook


def huffman_decoding(bits: str) -> List[int]:
    """
    Decode a Huffman bitstring back into symbols using the last codebook
    produced by huffman_encoding().

    bits: string of '0'/'1'
    """
    global _LAST_CODEBOOK
    if not _LAST_CODEBOOK:
        return []

    codebook = _LAST_CODEBOOK
    rev = {v: k for k, v in codebook.items()}

    res = []
    cur = ""
    for b in bits:
        cur += b
        if cur in rev:
            res.append(rev[cur])
            cur = ""

    return res



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



def zig_zag_decode(arr, m, n):
    result = [[0] * n for _ in range(m)]
    index = 0
    for lay in range(m + n - 1):
        if lay % 2 == 0:
            r = min(lay, m - 1)
            c = lay - r
            while r >= 0 and c < n:
                result[r][c] = arr[index]
                index += 1
                r -= 1
                c += 1
        else:
            c = min(lay, n - 1)
            r = lay - c
            while c >= 0 and r < m:
                result[r][c] = arr[index]
                index += 1
                r += 1
                c -= 1
    return result


symbols = [0,0,0,0,0,1,1,2,3]

codebook = huffman_encoding(symbols)
print("codebook:", codebook)
encoded_bits = "".join(codebook[s] for s in symbols)
decoded_symbols = huffman_decoding(encoded_bits)

print("encoded_bits:", encoded_bits)
print("decoded_symbols:", decoded_symbols)
