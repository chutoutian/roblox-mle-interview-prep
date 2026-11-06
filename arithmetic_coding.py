from collections import Counter
from bisect import bisect_right


def _build_model(freqs):
    """Return sorted symbols, cumulative counts, and total count."""
    symbols = sorted(freqs)
    cum = [0]
    for s in symbols:
        c = int(freqs[s])
        if c <= 0:
            raise ValueError("All frequencies must be positive.")
        cum.append(cum[-1] + c)
    total = cum[-1]
    if total <= 0:
        raise ValueError("Total frequency must be positive.")
    return symbols, cum, total


def arithmetic_encode(data, freqs=None, precision=32):
    """
    Static arithmetic encoder (integer range coder style).
    Returns: (bitstream, freqs)
    """
    if precision < 16:
        raise ValueError("precision must be >= 16")
    if not data:
        return "", (dict(freqs) if freqs is not None else {})

    if freqs is None:
        freqs = dict(Counter(data))

    symbols, cum, total = _build_model(freqs)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    FULL = (1 << precision) - 1
    HALF = 1 << (precision - 1)
    QTR = 1 << (precision - 2)
    THREE_QTR = 3 * QTR

    low, high = 0, FULL
    pending = 0
    bits = []

    def emit(bit):
        nonlocal pending
        bits.append("1" if bit else "0")
        # flush deferred opposite bits (E3 handling)
        fill = "0" if bit else "1"
        while pending > 0:
            bits.append(fill)
            pending -= 1

    for s in data:
        i = sym_to_idx[s]
        s_low, s_high = cum[i], cum[i + 1]

        rng = high - low + 1
        high = low + (rng * s_high // total) - 1
        low = low + (rng * s_low // total)

        while True:
            if high < HALF:               # E1
                emit(0)
            elif low >= HALF:            # E2
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QTR and high < THREE_QTR:  # E3
                pending += 1
                low -= QTR
                high -= QTR
            else:
                break

            low = low << 1
            high = (high << 1) + 1

    # finalize
    pending += 1
    emit(0 if low < QTR else 1)

    return "".join(bits), freqs


def arithmetic_decode(bitstream, freqs, n_symbols, precision=32):
    """
    Static arithmetic decoder matching arithmetic_encode.
    """
    if n_symbols == 0:
        return []
    if precision < 16:
        raise ValueError("precision must be >= 16")

    symbols, cum, total = _build_model(freqs)

    FULL = (1 << precision) - 1
    HALF = 1 << (precision - 1)
    QTR = 1 << (precision - 2)
    THREE_QTR = 3 * QTR

    idx = 0

    def read_bit():
        nonlocal idx
        if idx < len(bitstream):
            b = bitstream[idx]
            idx += 1
            if b == "0":
                return 0
            if b == "1":
                return 1
            raise ValueError("bitstream must contain only '0'/'1'")
        return 0  # implicit zero padding

    low, high = 0, FULL
    value = 0
    for _ in range(precision):
        value = (value << 1) | read_bit()

    out = []
    for _ in range(n_symbols):
        rng = high - low + 1
        scaled = ((value - low + 1) * total - 1) // rng

        # find i s.t. cum[i] <= scaled < cum[i+1]
        i = bisect_right(cum, scaled) - 1
        out.append(symbols[i])

        s_low, s_high = cum[i], cum[i + 1]
        high = low + (rng * s_high // total) - 1
        low = low + (rng * s_low // total)

        while True:
            if high < HALF:               # E1
                pass
            elif low >= HALF:            # E2
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QTR and high < THREE_QTR:  # E3
                low -= QTR
                high -= QTR
                value -= QTR
            else:
                break

            low = low << 1
            high = (high << 1) + 1
            value = (value << 1) | read_bit()

    return out


if __name__ == "__main__":
    data = [2, 2, 2, 1, 1, 0, 3, 3, 2, 1]

    bitstream, freqs = arithmetic_encode(data, precision=32)
    recovered = arithmetic_decode(bitstream, freqs, len(data), precision=32)

    print("bits:", bitstream)
    print("decoded:", recovered)
    print("match:", recovered == data)
