from typing import List, Tuple


class IntraPredictor:
    """
    Interview-style intra prediction demo for one NxN block.
    Modes:
      1) DC
      2) Planar (HEVC-like simplified integer form)
      3) Angular (simplified: vertical-like / horizontal-like with interpolation)

    Assumptions:
    - top and left have length N
    - pixel values are integers (typically 0..255)
    - for planar, tr and bl are provided (top-right / bottom-left extension)
    """

    @staticmethod
    def clip_pixel(v: int, lo: int = 0, hi: int = 255) -> int:
        return max(lo, min(hi, v))

    @staticmethod
    def dc_predict(top: List[int], left: List[int]) -> List[List[int]]:
        n = len(top)
        if len(left) != n:
            raise ValueError("top and left must have same length")

        # DC = round((sum(top)+sum(left)) / (2n)) with integer-style rounding
        dc = (sum(top) + sum(left) + n) // (2 * n)
        dc = IntraPredictor.clip_pixel(dc)

        return [[dc for _ in range(n)] for _ in range(n)]

    @staticmethod
    def planar_predict(
        top: List[int],
        left: List[int],
        tr: int,
        bl: int
    ) -> List[List[int]]:
        """
        HEVC-style planar intuition:
          P(x,y) = ((N-1-x)*L[y] + (x+1)*TR + (N-1-y)*T[x] + (y+1)*BL + N) // (2N)
        """
        n = len(top)
        if len(left) != n:
            raise ValueError("top and left must have same length")

        out = [[0] * n for _ in range(n)]
        denom = 2 * n

        for y in range(n):
            for x in range(n):
                num = (
                    (n - 1 - x) * left[y]
                    + (x + 1) * tr
                    + (n - 1 - y) * top[x]
                    + (y + 1) * bl
                    + n
                )
                out[y][x] = IntraPredictor.clip_pixel(num // denom)
        return out

    @staticmethod
    def _interp_1d(ref: List[int], pos_num: int, scale: int = 32) -> int:
        """
        Linear interpolation on ref at fixed-point position pos_num/scale.
        Example:
          pos_num = i*scale + w, w in [0, scale-1]
          val = ((scale-w)*ref[i] + w*ref[i+1] + scale//2) // scale
        with boundary clamping.
        """
        i = pos_num // scale
        w = pos_num % scale

        i0 = max(0, min(i, len(ref) - 1))
        i1 = max(0, min(i + 1, len(ref) - 1))

        val = ((scale - w) * ref[i0] + w * ref[i1] + scale // 2) // scale
        return IntraPredictor.clip_pixel(val)

    @staticmethod
    def angular_predict(
        top: List[int],
        left: List[int],
        mode: str = "vertical",
        d: int = 0
    ) -> List[List[int]]:
        """
        Simplified angular predictor with fixed-point interpolation.

        mode="vertical":
            project from top refs, with slope controlled by d
            pos = x + (y+1)*d/32
        mode="horizontal":
            project from left refs, with slope controlled by d
            pos = y + (x+1)*d/32

        d > 0: tilt one way, d < 0: tilt the other way, d=0: pure vertical/horizontal copy.
        """
        n = len(top)
        if len(left) != n:
            raise ValueError("top and left must have same length")
        if mode not in ("vertical", "horizontal"):
            raise ValueError("mode must be 'vertical' or 'horizontal'")

        out = [[0] * n for _ in range(n)]
        scale = 32

        if mode == "vertical":
            # Reference line for x direction
            ref = top[:]  # length n
            for y in range(n):
                for x in range(n):
                    # fixed-point position numerator: (x * 32 + (y+1)*d)
                    pos_num = x * scale + (y + 1) * d
                    out[y][x] = IntraPredictor._interp_1d(ref, pos_num, scale)
        else:
            # horizontal: use left refs
            ref = left[:]
            for y in range(n):
                for x in range(n):
                    # fixed-point position numerator: (y * 32 + (x+1)*d)
                    pos_num = y * scale + (x + 1) * d
                    out[y][x] = IntraPredictor._interp_1d(ref, pos_num, scale)

        return out

    @staticmethod
    def residual(block: List[List[int]], pred: List[List[int]]) -> List[List[int]]:
        n = len(block)
        if n == 0 or any(len(row) != n for row in block):
            raise ValueError("block must be NxN")
        if len(pred) != n or any(len(row) != n for row in pred):
            raise ValueError("pred must be NxN and same size as block")

        return [[block[y][x] - pred[y][x] for x in range(n)] for y in range(n)]

    @staticmethod
    def sad(block: List[List[int]], pred: List[List[int]]) -> int:
        r = IntraPredictor.residual(block, pred)
        return sum(abs(v) for row in r for v in row)

    @staticmethod
    def choose_mode_by_sad(
        block: List[List[int]],
        top: List[int],
        left: List[int],
        tr: int,
        bl: int
    ) -> Tuple[str, List[List[int]], int]:
        """
        Interview-friendly mode decision using SAD only (not full RD).
        Returns: (best_mode_name, best_prediction, best_sad)
        """
        candidates = {}

        p_dc = IntraPredictor.dc_predict(top, left)
        candidates["DC"] = (p_dc, IntraPredictor.sad(block, p_dc))

        p_planar = IntraPredictor.planar_predict(top, left, tr, bl)
        candidates["Planar"] = (p_planar, IntraPredictor.sad(block, p_planar))

        # Try a few angular candidates (you can add more)
        p_ang_v0 = IntraPredictor.angular_predict(top, left, mode="vertical", d=0)
        candidates["Angular-Vertical(d=0)"] = (p_ang_v0, IntraPredictor.sad(block, p_ang_v0))

        p_ang_v16 = IntraPredictor.angular_predict(top, left, mode="vertical", d=16)
        candidates["Angular-Vertical(d=16)"] = (p_ang_v16, IntraPredictor.sad(block, p_ang_v16))

        p_ang_h16 = IntraPredictor.angular_predict(top, left, mode="horizontal", d=16)
        candidates["Angular-Horizontal(d=16)"] = (p_ang_h16, IntraPredictor.sad(block, p_ang_h16))

        best_mode, (best_pred, best_cost) = min(candidates.items(), key=lambda kv: kv[1][1])
        return best_mode, best_pred, best_cost


if __name__ == "__main__":
    # Example
    X = [
        [52, 55, 61, 66],
        [53, 56, 62, 67],
        [54, 57, 63, 68],
        [55, 58, 64, 69],
    ]
    top = [50, 54, 60, 65]
    left = [51, 52, 53, 54]
    tr, bl = 68, 56

    mode, pred, cost = IntraPredictor.choose_mode_by_sad(X, top, left, tr, bl)
    print("Best mode:", mode)
    print("Best SAD:", cost)
    print("Prediction:")
    for row in pred:
        print(row)

    print("\nResidual:")
    res = IntraPredictor.residual(X, pred)
    for row in res:
        print(row)
