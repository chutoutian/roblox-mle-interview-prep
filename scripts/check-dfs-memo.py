#!/usr/bin/env python3
"""Execute and check the actual Python fences in the DFS + Memoization chapter.

Usage: python3 scripts/check-dfs-memo.py [path/to/notes.md]
Only run this against trusted Markdown: its Python fences are executed.
"""

from itertools import combinations
import math
from pathlib import Path
import bisect
import random
import re
import signal
import sys


CHAPTER = "3. Recursion: Tree DFS, Backtracking, Memoization"
SECTIONS = (4, 4)  # DFS + Memoization: section 3.4 of the recursion chapter, with its 3.4.x subsections
REQUIRED = ("combination_sum4", "num_squares", "num_squares_table", "length_of_lis_take_skip", "length_of_lis", "word_break",
            "can_partition", "can_partition_table", "max_product", "max_product_loop")
CURRENT_CASE = "loading DFS + Memoization examples"
CALLS = 0


def require(condition, message):
    if not condition:
        raise AssertionError(f"{CURRENT_CASE}: {message}")


def chapter_fences(source):
    """Ignore headings inside fences; preserve original traceback line numbers."""
    opening, code, selected, in_chapter, chapters = None, [], False, False, 0
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        if opening is None:
            heading = re.match(r"^ {0,3}#\s+(.+?)\s*#*\s*$", line)
            if heading:
                in_chapter = heading.group(1) == CHAPTER
                chapters += int(in_chapter)
                selected = False
            section = re.match(r"^ {0,3}##\s+3\.(\d+)\s", line)
            if section:
                selected = in_chapter and SECTIONS[0] <= int(section.group(1)) <= SECTIONS[1]
            match = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$", line)
            if match:
                fence, info = match.groups()
                language = info.strip().split(maxsplit=1)
                keep = selected and bool(language and language[0].lower() in {"python", "python3", "py", "py3"})
                opening, code = (fence, keep, number + 1), []
            continue
        fence, keep, first_line = opening
        closing = r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*[\r\n]*"
        if re.fullmatch(closing, line):
            if keep:
                yield first_line, "".join(code)
            opening = None
        elif keep:
            code.append(line)
    require(opening is None, "unclosed Markdown code fence")
    require(chapters == 1, f"expected one '# {CHAPTER}' chapter, found {chapters}")


def load_examples(path):
    global CURRENT_CASE
    namespace = {"__name__": "dfs_memo_notes", "__file__": str(path)}
    fences = list(chapter_fences(path.read_text(encoding="utf-8")))
    require(fences, "no Python fences in the DFS + Memoization chapter")
    for first_line, code in fences:
        CURRENT_CASE = f"Python fence at {path}:{first_line}"
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
    for name in REQUIRED:
        require(callable(namespace.get(name)), f"missing callable example: {name}")
    return namespace, len(fences)


def call(namespace, name, args):
    global CURRENT_CASE, CALLS
    CURRENT_CASE = f"{name}{args!r}"[:200]
    snapshot = repr(args)
    result = namespace[name](*args)
    CALLS += 1
    require(repr(args) == snapshot, "input was mutated")
    return result


def ordered_ways(nums, target):
    """Bottom-up oracle, independent of the notes' recursion."""
    dp = [1] + [0] * target
    for r in range(1, target + 1):
        dp[r] = sum(dp[r - v] for v in nums if v <= r)
    return dp[target]


def check_combination_sum4(namespace):
    for args, expected in ((([1, 2, 3], 4), 7), (([9], 3), 0), (([2], 4), 1), (([4, 2, 1], 32), 39882198)):
        require(call(namespace, "combination_sum4", args) == expected, f"expected {expected}")
    rng = random.Random(377)
    cases = 0
    for _ in range(300):
        nums = rng.sample(range(1, 11), rng.randint(1, 5))
        target = rng.randint(0, 60)
        expected = ordered_ways(nums, target)
        require(call(namespace, "combination_sum4", (nums, target)) == expected, f"expected {expected}")
        cases += 1
    return cases


def fewest_squares(n):
    """Number theory oracle (Lagrange and Legendre), independent of the notes' dynamic programming."""
    if n == 0:
        return 0
    if math.isqrt(n) ** 2 == n:
        return 1
    if any(math.isqrt(n - a * a) ** 2 == n - a * a for a in range(1, math.isqrt(n) + 1)):
        return 2
    m = n
    while m % 4 == 0:
        m //= 4
    return 4 if m % 8 == 7 else 3


def check_perfect_squares(namespace):
    for n, expected in ((12, 3), (13, 2), (1, 1), (7, 4), (10000, 1), (9998, 3)):
        for name in ("num_squares", "num_squares_table"):
            require(call(namespace, name, (n,)) == expected, f"expected {expected}")
    rng = random.Random(279)
    values = list(range(1, 301)) + rng.sample(range(301, 10001), 25)
    for n in values:
        expected = fewest_squares(n)
        for name in ("num_squares", "num_squares_table"):
            require(call(namespace, name, (n,)) == expected, f"Lagrange/Legendre says {expected}")
    return len(values)


def lis_brute(nums):
    best = 0
    for size in range(len(nums) + 1):
        for idx in combinations(range(len(nums)), size):
            values = [nums[i] for i in idx]
            if all(a < b for a, b in zip(values, values[1:])):
                best = max(best, size)
    return best


def lis_patience(nums):
    tails = []
    for value in nums:
        k = bisect.bisect_left(tails, value)
        tails[k:k + 1] = [value]
    return len(tails)


def check_lis(namespace):
    examples = (([10, 9, 2, 5, 3, 7, 101, 18], 4), ([0, 1, 0, 3, 2, 3], 4), ([7, 7, 7, 7, 7, 7, 7], 1), ([5], 1))
    for nums, expected in examples:
        for name in ("length_of_lis_take_skip", "length_of_lis"):
            require(call(namespace, name, (nums,)) == expected, f"expected {expected}")
    rng = random.Random(300)
    cases = 0
    for _ in range(200):
        nums = [rng.randint(-5, 5) for _ in range(rng.randint(1, 10))]
        expected = lis_brute(nums)
        for name in ("length_of_lis_take_skip", "length_of_lis"):
            require(call(namespace, name, (nums,)) == expected, f"brute force says {expected}")
        cases += 1
    for _ in range(60):
        nums = [rng.randint(-10 ** 4, 10 ** 4) for _ in range(rng.randint(50, 300))]
        expected = lis_patience(nums)
        for name in ("length_of_lis_take_skip", "length_of_lis"):
            require(call(namespace, name, (nums,)) == expected, f"patience sorting says {expected}")
        cases += 1
    return cases


def word_break_bfs(s, words):
    """Reachable-index oracle, independent of the notes' recursion."""
    reachable, frontier = {0}, [0]
    while frontier:
        index = frontier.pop()
        for word in words:
            end = index + len(word)
            if s[index:end] == word and end not in reachable:
                reachable.add(end)
                frontier.append(end)
    return len(s) in reachable


def check_word_break(namespace):
    examples = (
        (("leetcode", ["leet", "code"]), True),
        (("applepenapple", ["apple", "pen"]), True),
        (("catsandog", ["cats", "dog", "sand", "and", "cat"]), False),
        (("a" * 30 + "b", ["a", "aa", "aaa"]), False),
        (("a" * 200, ["a", "aa", "aaa", "aaaa"]), True),
    )
    for args, expected in examples:
        require(call(namespace, "word_break", args) is expected, f"expected {expected}")
    rng = random.Random(139)
    cases = 0
    for _ in range(400):
        s = "".join(rng.choice("ab") for _ in range(rng.randint(1, 14)))
        words = sorted({"".join(rng.choice("ab") for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 5))})
        expected = word_break_bfs(s, words)
        require(call(namespace, "word_break", (s, words)) is expected, f"expected {expected}")
        cases += 1
    return cases


def partition_oracle(nums):
    """All subset sums as a set, independent of the notes' recursion and table."""
    total, sums = sum(nums), {0}
    for value in nums:
        sums |= {x + value for x in sums}
    return total % 2 == 0 and total // 2 in sums


def check_partition(namespace):
    names = ("can_partition", "can_partition_table")
    examples = (([1, 5, 11, 5], True), ([1, 2, 3, 5], False), ([1], False), ([2, 2], True), ([1, 1, 1, 1], True),
                ([3, 3, 3, 4, 5], True), ([1, 2, 5], False), ([100] * 200, True), ([99] * 199 + [1], False))
    for nums, expected in examples:
        for name in names:
            require(call(namespace, name, (nums,)) is expected, f"expected {expected}")
    rng = random.Random(416)
    cases = 0
    for _ in range(400):
        nums = [rng.randint(1, 20) for _ in range(rng.randint(1, 14))]
        expected = partition_oracle(nums)
        for name in names:
            require(call(namespace, name, (nums,)) is expected, f"subset sums say {expected}")
        cases += 1
    for _ in range(20):                       # LeetCode-sized for the table: 200 numbers up to 100
        nums = [rng.randint(1, 100) for _ in range(200)]
        require(call(namespace, "can_partition_table", (nums,)) is partition_oracle(nums), "large table case")
        cases += 1
    return cases


def max_product_brute(nums):
    """Every subarray, independent of the notes' two-value recursion."""
    best = None
    for i in range(len(nums)):
        product = 1
        for j in range(i, len(nums)):
            product *= nums[j]
            best = product if best is None else max(best, product)
    return best


def check_max_product(namespace):
    names = ("max_product", "max_product_loop")
    examples = (([2, 3, -2, 4], 6), ([-2, 0, -1], 0), ([-2, 3, -4], 24), ([-2], -2), ([0, 2], 2),
                ([-1, -2, -3, 0], 6), ([3, -1, 4], 4), ([-2, -3, 7], 42), ([0], 0), ([-3, 0, -2], 0))
    for nums, expected in examples:
        for name in names:
            require(call(namespace, name, (nums,)) == expected, f"expected {expected}")
    rng = random.Random(152)
    cases = 0
    for _ in range(500):
        nums = [rng.randint(-5, 5) for _ in range(rng.randint(1, 12))]
        expected = max_product_brute(nums)
        for name in names:
            require(call(namespace, name, (nums,)) == expected, f"brute force says {expected}")
        cases += 1
    # LeetCode's largest size: calling dfs(0), dfs(1), ... in order keeps the recursion one level deep.
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(200)
        for _ in range(3):
            nums = [rng.choice((-1, 0, 1, 1, -1)) for _ in range(20_000)]
            expected = call(namespace, "max_product_loop", (nums,))
            require(call(namespace, "max_product", (nums,)) == expected, "20,000 numbers with a 200-frame limit")
            cases += 1
    finally:
        sys.setrecursionlimit(limit)
    return cases


def timeout_handler(signum, frame):
    raise TimeoutError(f"DFS + Memoization examples exceeded the 20-second limit while checking {CURRENT_CASE}")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 scripts/check-dfs-memo.py [path/to/notes.md]")
    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent.parent / "06_leetcode_notes.md"
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    try:
        namespace, fences = load_examples(path)
        counts = check_combination_sum4(namespace)
        squares = check_perfect_squares(namespace)
        lis = check_lis(namespace)
        breaks = check_word_break(namespace)
        partitions = check_partition(namespace)
        products = check_max_product(namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(f"DFS + Memoization examples passed: {fences} Python fences, {counts} counting cases, {squares} perfect-square cases, {lis} LIS cases "
          f"(brute force and patience sorting), {breaks} word-break cases, {partitions} partition cases, {products} maximum-product cases "
          f"against independent oracles; {CALLS:,} function calls checked.")


if __name__ == "__main__":
    main()
