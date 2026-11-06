#!/usr/bin/env python3
"""Execute and check the actual Python fences in the Backtracking chapter.

Usage: python3 scripts/check-backtracking.py [path/to/notes.md]
Only run this against trusted Markdown: its Python fences are executed.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, factorial
from pathlib import Path
import re
import signal
import sys


CHAPTER = "3. Recursion: Tree DFS, Backtracking, Memoization"
SECTIONS = (3, 3)  # Backtracking: section 3.3 of the recursion chapter, with its 3.3.x subsections
REQUIRED = (
    "subsets", "combine", "permute", "combination_sum", "subsets_with_dup",
    "combination_sum2", "permute_unique",
)
CURRENT_CASE = "loading Backtracking examples"
CALLS = 0
ROWS = 0
SAVED = {}


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
    namespace = {"__name__": "backtracking_notes", "__file__": str(path)}
    fences = list(chapter_fences(path.read_text(encoding="utf-8")))
    require(fences, "no Python fences in the Backtracking chapter")
    for first_line, code in fences:
        CURRENT_CASE = f"Python fence at {path}:{first_line}"
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
    for name in REQUIRED:
        require(callable(namespace.get(name)), f"missing callable example: {name}")
    return namespace, len(fences)


def check(namespace, name, args, expected, unordered=False):
    global CURRENT_CASE, CALLS, ROWS
    CURRENT_CASE = f"{name}{args!r}"
    before = [arg[:] if isinstance(arg, list) else arg for arg in args]
    result = namespace[name](*args)
    CALLS += 1
    require(list(args) == before, "input was mutated")
    require(isinstance(result, list) and all(isinstance(row, list) for row in result),
            "expected a list of lists")
    require(len({id(row) for row in result}) == len(result), "result rows alias the same path list")
    require(all(row is not arg for row in result for arg in args if isinstance(arg, list)),
            "an output row aliases an input list")
    require(all(type(value) is int for row in result for value in row), "invalid output element")
    actual = [tuple(sorted(row)) if unordered else tuple(row) for row in result]
    require(len(actual) == len(set(actual)), "duplicate answers")
    require(set(actual) == expected, f"wrong answers: got {len(actual)}, expected {len(expected)}")
    ROWS += len(result)
    if name in SAVED:
        previous, snapshot = SAVED[name]
        require(previous == snapshot, "a later call changed a previous result")
    elif len(result) > 1:
        SAVED[name] = (result, [row[:] for row in result])
    return result


def subset_oracle(values):
    """Enumerate index subsets, canonicalizing values only after selection."""
    return {tuple(sorted(values[index] for index in indexes))
            for size in range(len(values) + 1)
            for indexes in combinations(range(len(values)), size)}


@lru_cache(maxsize=None)
def repeatable_sum_oracle(candidates, target):
    """Enumerate value-count vectors, independently of the notes' DFS."""
    result = set()
    for counts in product(*(range(target // value + 1) for value in candidates)):
        if sum(value * count for value, count in zip(candidates, counts)) == target:
            result.add(tuple(value for value, count in zip(candidates, counts) for _ in range(count)))
    return result


def check_distinct(namespace):
    # Every input ordering through size 5, then larger bounded factorial cases.
    fixtures = [list(values) for size in range(6)
                for values in permutations(range(size))]
    fixtures.extend(([6, -2, 0, 8, 1, -7], [3, 1, 7, 5, 2, 6, 4]))
    for values in fixtures:
        expected = subset_oracle(values)
        require(len(expected) == 2 ** len(values), "invalid distinct subset oracle")
        check(namespace, "subsets", (values,), expected, unordered=True)
        expected = set(permutations(values))
        require(len(expected) == factorial(len(values)), "invalid distinct permutation oracle")
        check(namespace, "permute", (values,), expected)

    for n in range(9):
        for k in range(-2, n + 3):
            expected = set(combinations(range(1, n + 1), k)) if 0 <= k <= n else set()
            require(len(expected) == (comb(n, k) if 0 <= k <= n else 0), "invalid combination oracle")
            check(namespace, "combine", (n, k), expected, unordered=True)
    return len(fixtures)


def check_repeatable_sums(namespace):
    fixtures = [list(values) for size in range(5)
                for values in permutations(range(1, 6), size)]
    for values in fixtures:
        for target in range(13):
            expected = repeatable_sum_oracle(tuple(sorted(values)), target)
            check(namespace, "combination_sum", (values, target), expected, unordered=True)
    return len(fixtures)


def check_duplicates(namespace):
    # All arrangements (not only sorted inputs) test same-depth skipping.
    fixtures = [list(values) for size in range(6)
                for values in product((1, 2, 3), repeat=size)]
    fixtures.extend(([1] * 6, [2, 1, 2, 1, 2, 1], [3, 2, 1, 3, 2, 1]))
    for values in fixtures:
        expected_subsets = subset_oracle(values)
        counts = Counter(values)
        expected_count = 1
        for count in counts.values():
            expected_count *= count + 1
        require(len(expected_subsets) == expected_count, "invalid duplicate subset oracle")
        check(namespace, "subsets_with_dup", (values,), expected_subsets, unordered=True)
        expected_permutations = set(permutations(values))
        expected_count = factorial(len(values))
        for count in counts.values():
            expected_count //= factorial(count)
        require(len(expected_permutations) == expected_count, "invalid unique permutation oracle")
        check(namespace, "permute_unique", (values,), expected_permutations)
        for target in range(13):
            expected = {row for row in expected_subsets if sum(row) == target}
            check(namespace, "combination_sum2", (values, target), expected, unordered=True)

    # Zeros and negative values are valid for subsets/permutations, but not sums.
    for values in ([0, 0, -1], [-2, 0, -2, 3], [0] * 6):
        check(namespace, "subsets_with_dup", (values,), subset_oracle(values), unordered=True)
        check(namespace, "permute_unique", (values,), set(permutations(values)))
    return len(fixtures)


def check_result_ownership(namespace):
    fixtures = (
        ("subsets", ([3, 1, 2],), subset_oracle([3, 1, 2]), True),
        ("combine", (5, 3), set(combinations(range(1, 6), 3)), True),
        ("permute", ([3, 1, 2],), set(permutations([3, 1, 2])), False),
        ("combination_sum", ([3, 2], 6), {(2, 2, 2), (3, 3)}, True),
        ("subsets_with_dup", ([2, 1, 2],), subset_oracle([2, 1, 2]), True),
        ("combination_sum2", ([2, 1, 2, 1], 4), {(1, 1, 2), (2, 2)}, True),
        ("permute_unique", ([2, 1, 2],), set(permutations([2, 1, 2])), False),
    )
    for name, args, expected, unordered in fixtures:
        first = check(namespace, name, args, expected, unordered)
        snapshot = [row[:] for row in first]
        first[0].append(999999)
        require(first[1:] == snapshot[1:], "editing one answer changed other answers")
        second = check(namespace, name, args, expected, unordered)
        require(first is not second, "separate calls share the outer result list")
        require(not ({id(row) for row in first} & {id(row) for row in second}),
                "separate calls share result rows")
        require(first[0] == snapshot[0] + [999999], "a later call rewrote an earlier returned row")


def timeout_handler(signum, frame):
    raise TimeoutError(f"Backtracking examples exceeded the 20-second limit while checking {CURRENT_CASE}")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 scripts/check-backtracking.py [path/to/notes.md]")
    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent.parent / "06_leetcode_notes.md"
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    try:
        namespace, fences = load_examples(path)
        distinct = check_distinct(namespace)
        repeatable = check_repeatable_sums(namespace)
        duplicates = check_duplicates(namespace)
        check_result_ownership(namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(f"Backtracking examples passed: {fences} Python fences, {distinct} distinct-input fixtures, "
          f"{repeatable} repeatable-sum fixtures, {duplicates} duplicate-input fixtures; "
          f"{CALLS:,} function calls, {ROWS:,} result rows checked.")


if __name__ == "__main__":
    main()
