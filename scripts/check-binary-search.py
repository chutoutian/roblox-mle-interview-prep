#!/usr/bin/env python3
"""Run the actual Python examples in the binary-search study notes.

Usage: python3 scripts/check-binary-search.py [path/to/notes.md]
Only run this against trusted Markdown: its Python fences are executed.
"""

import bisect
import itertools
from pathlib import Path
import random
import re
import signal
import sys


REQUIRED = (
    "binary_search_exact",
    "lower_bound",
    "upper_bound",
    "search_range",
    "min_eating_speed",
)
PYTHON_LANGUAGES = {"python", "python3", "py", "py3"}
CURRENT_CASE = "loading examples"
CALLS = 0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def python_fences(source):
    """Yield (first code line, code), ignoring Python-looking text in other fences."""
    opening = None
    code = []
    for line_number, line in enumerate(source.splitlines(keepends=True), 1):
        if opening is None:
            match = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$", line)
            if match:
                fence, info = match.groups()
                language = info.strip().split(maxsplit=1)
                opening = (fence, bool(language and language[0].lower() in PYTHON_LANGUAGES), line_number + 1)
                code = []
            continue
        fence, is_python, first_line = opening
        if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*[\r\n]*", line):
            if is_python:
                yield first_line, "".join(code)
            opening = None
        elif is_python:
            code.append(line)
    require(opening is None, "Markdown contains an unclosed code fence")


def load_examples(path):
    global CURRENT_CASE
    source = path.read_text(encoding="utf-8")
    namespace = {"__name__": "binary_search_notes", "__file__": str(path)}
    fences = list(python_fences(source))
    require(fences, f"No Python code fences found in {path}")
    for first_line, code in fences:
        CURRENT_CASE = f"Python fence at {path}:{first_line}"
        # Padding keeps Python tracebacks aligned with the Markdown source.
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
    for name in REQUIRED:
        require(callable(namespace.get(name)), f"Missing callable example: {name}")
    return namespace, len(fences)


def call(namespace, name, *args):
    global CURRENT_CASE, CALLS
    CURRENT_CASE = f"{name}{args!r}"
    CALLS += 1
    result = namespace[name](*args)
    return result


def array_call(namespace, name, values, target):
    nums = list(values)
    result = call(namespace, name, nums, target)
    require(nums == list(values), f"{CURRENT_CASE} mutated its input to {nums!r}")
    return result


def require_integer(result):
    require(type(result) is int, f"{CURRENT_CASE} returned non-integer {result!r}")


def check_arrays(namespace):
    cases = 0
    for length in range(7):
        for values in itertools.combinations_with_replacement(range(-2, 4), length):
            for target in range(-3, 5):
                cases += 1
                left = bisect.bisect_left(values, target)
                right = bisect.bisect_right(values, target)
                present = left != right
                for name, expected in (
                    ("lower_bound", left),
                    ("upper_bound", right),
                ):
                    result = array_call(namespace, name, values, target)
                    require_integer(result)
                    require(result == expected, f"{CURRENT_CASE}: expected {expected}, got {result}")
                result = array_call(namespace, "binary_search_exact", values, target)
                require_integer(result)
                if present:
                    require(left <= result < right, f"{CURRENT_CASE}: expected an equal element in [{left}, {right}), got {result}")
                else:
                    require(result == -1, f"{CURRENT_CASE}: expected -1, got {result}")
                result = array_call(namespace, "search_range", values, target)
                expected = (left, right - 1) if present else (-1, -1)
                require(isinstance(result, (list, tuple)) and len(result) == 2 and all(type(index) is int for index in result),
                        f"{CURRENT_CASE}: expected two integer indices, got {result!r}")
                require(tuple(result) == expected, f"{CURRENT_CASE}: expected {expected}, got {result!r}")
    return cases


def eating_speed_oracle(piles, hours):
    # Exhaustive candidate enumeration is independent of binary-search updates.
    for speed in range(1, max(piles) + 1):
        if sum((pile + speed - 1) // speed for pile in piles) <= hours:
            return speed
    raise AssertionError("Invalid test fixture: no feasible eating speed")


def check_eating_speeds(namespace):
    cases = 0

    def check(piles, hours, expected=None):
        nonlocal cases
        if expected is None:
            expected = eating_speed_oracle(piles, hours)
        result = array_call(namespace, "min_eating_speed", piles, hours)
        require_integer(result)
        require(result == expected, f"{CURRENT_CASE}: expected {expected}, got {result}")
        require(sum((pile + result - 1) // result for pile in piles) <= hours,
                f"{CURRENT_CASE}: returned an infeasible speed")
        if result > 1:
            require(sum((pile + result - 2) // (result - 1) for pile in piles) > hours,
                    f"{CURRENT_CASE}: returned a feasible speed that is not minimal")
        cases += 1

    for length in range(1, 5):
        for piles in itertools.product(range(1, 5), repeat=length):
            for hours in range(length, sum(piles) + 2):
                check(piles, hours)
    generator = random.Random(875)
    for _ in range(250):
        piles = tuple(generator.randint(1, 35) for _ in range(generator.randint(1, 10)))
        check(piles, generator.randint(len(piles), sum(piles) + 10))
    check((3, 6, 7, 11), 8, 4)
    check((30, 11, 23, 4, 20), 5, 30)
    check((30, 11, 23, 4, 20), 6, 23)
    # These also catch floating-point ceil/division in integer predicates.
    check((10 ** 100,), 1, 10 ** 100)
    check((10 ** 100 + 1,), 2, (10 ** 100 + 2) // 2)
    check((10 ** 100, 10 ** 100), 2, 10 ** 100)
    return cases


def timeout_handler(signum, frame):
    raise TimeoutError(f"Examples exceeded the 20-second limit while checking {CURRENT_CASE}")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 scripts/check-binary-search.py [path/to/notes.md]")
    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent.parent / "06_leetcode_notes.md"
    # A malformed loop must fail the command instead of hanging the build.
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    try:
        namespace, fences = load_examples(path)
        arrays = check_arrays(namespace)
        speeds = check_eating_speeds(namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(f"Binary-search examples passed: {fences} Python fences, {arrays:,} array/target cases, {speeds:,} eating-speed cases; {CALLS:,} function calls.")


if __name__ == "__main__":
    main()
