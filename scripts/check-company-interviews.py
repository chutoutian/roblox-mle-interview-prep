#!/usr/bin/env python3
"""Execute and check the Python fences of the Roblox chapter's LeetCode section (1.1) in Company Interviews.

Usage: python3 scripts/check-company-interviews.py [path/to/08_company_interviews.md]
Only run this against trusted Markdown: its Python fences are executed.
"""
import itertools
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

CHAPTER = "1. Roblox — Machine Learning Engineer"
SECTION = "1.1"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_no_adjacent(ns, rng):
    calls = 0
    for _ in range(3000):
        items = [rng.choice('abcde'[:rng.randint(1, 5)]) for _ in range(rng.randint(0, 12))]
        possible = any(all(p[i] != p[i + 1] for i in range(len(p) - 1)) for p in set(itertools.permutations(items))) if len(items) <= 8 else None
        got = ns['arrange_no_adjacent'](list(items))
        calls += 1
        counts = Counter(items)
        check(ns['can_arrange'](counts) == (max(counts.values(), default=0) <= (len(items) + 1) // 2), f'can_arrange {items}')
        if possible is not None:
            check(ns['can_arrange'](counts) == possible, f'can_arrange vs brute force {items}')
        if got:
            check(Counter(got) == counts and all(got[i] != got[i + 1] for i in range(len(got) - 1)), f'bad arrangement {items} -> {got}')
        else:
            check(not ns['can_arrange'](counts) or not items, f'gave up although possible {items}')
    check(ns['arrange_no_adjacent']([]) == [], 'empty input')
    for _ in range(1500):
        tasks = [rng.choice('ABCD') for _ in range(rng.randint(1, 14))]
        n = rng.randint(0, 4)
        check(ns['least_interval'](tasks, n) == simulate_scheduler(tasks, n), f'least_interval {tasks} {n}')
        calls += 1
    check(ns['least_interval'](list('AAABBB'), 2) == 8 and ns['least_interval'](list('AAABBB'), 0) == 6, 'LeetCode 621 examples')
    return calls


def simulate_scheduler(tasks, n):
    """Greedy simulation: each slot runs the available task with the most copies left (known to be optimal)."""
    remaining = Counter(tasks)
    ready_at = {t: 0 for t in remaining}
    time = 0
    while remaining:
        avail = [t for t in remaining if ready_at[t] <= time]
        if avail:
            t = max(avail, key=lambda x: (remaining[x], x))
            remaining[t] -= 1
            if not remaining[t]:
                del remaining[t]
            ready_at[t] = time + n + 1
        time += 1
    return time


def random_trace(rng, threads=None):
    lines, stacks = [], {t: [] for t in (threads or [None])}
    for _ in range(rng.randint(1, 40)):
        t = rng.choice(list(stacks))
        st = stacks[t]
        if st and rng.random() < 0.45:
            name = st.pop()
            lines.append(((t + ' ') if t else '') + '<- ' + name)
        else:
            name = rng.choice(['main', 'f', 'g', 'h'])
            st.append(name)
            lines.append(((t + ' ') if t else '') + '-> ' + name)
    return lines


def oracle_call_path(lines, threaded):
    stacks, counts, reached = {}, {}, {}
    for step, line in enumerate(lines):
        parts = line.split()
        t, arrow, name = (parts if threaded else [None] + parts)
        st = stacks.setdefault(t, [])
        if arrow == '->':
            st.append(name)
            p = tuple(st)
            counts[p] = counts.get(p, 0) + 1
            reached[p] = step
        else:
            st.pop()
    if not counts:
        return None, 0
    best = max(counts, key=lambda p: (counts[p], len(p), -reached[p]))
    return '->'.join(best), counts[best]


def test_call_path(ns, rng):
    example = ['-> main', '-> eventClick', '-> eventClickAction', '<- eventClickAction',
               '-> eventClickAction', '<- eventClickAction', '<- eventClick', '<- main']
    check(ns['most_frequent_call_path'](example) == ('main->eventClick->eventClickAction', 2), 'reported example')
    calls = 1
    for _ in range(3000):
        lines = random_trace(rng)
        check(ns['most_frequent_call_path'](lines) == oracle_call_path(lines, False), f'single thread {lines}')
        lines = random_trace(rng, ['t1', 't2', 't3'])
        check(ns['most_frequent_call_path_threads'](lines) == oracle_call_path(lines, True), f'threads {lines}')
        calls += 2
    for bad in (['<- main'], ['-> main', '<- f']):
        try:
            ns['most_frequent_call_path'](bad)
        except ValueError:
            calls += 1
        else:
            raise AssertionError(f'unmatched return accepted: {bad}')
    return calls


def test_median(ns, rng):
    calls = 0
    for _ in range(600):
        finder, seen = ns['MedianFinder'](), []
        for _ in range(rng.randint(1, 60)):
            x = rng.randint(-50, 50)
            finder.add_num(x)
            seen.append(x)
            check(finder.find_median() == statistics.median(seen), f'median {seen}')
            calls += 2
    try:
        ns['MedianFinder']().find_median()
    except ValueError:
        pass
    else:
        raise AssertionError('median of nothing must raise')
    return calls


def test_windows(ns, rng):
    calls = 0
    for _ in range(4000):
        nums = [rng.randint(0, 4) for _ in range(rng.randint(0, 15))]
        k, target = rng.randint(1, 6), rng.randint(0, 4)
        windows = [nums[i:i + k] for i in range(len(nums) - k + 1)]
        check(ns['windows_with_target'](nums, k, target) == [w for w in windows if target in w], f'windows {nums} {k} {target}')
        best = max((w.count(target) for w in windows), default=0)
        expected = next((w for w in windows if w.count(target) == best), None) if best else None
        check(ns['first_window_with_most_target'](nums, k, target) == expected, f'first max window {nums} {k} {target}')
        calls += 2
    return calls


def test_rate_limiter(ns, rng):
    calls = 0
    for _ in range(800):
        limit, window = rng.randint(1, 4), rng.randint(1, 6)
        limiter, accepted = ns['SlidingWindowRateLimiter'](limit, window), []
        t = 0
        for _ in range(rng.randint(1, 50)):
            t += rng.choice([0, 0, 1, 1, 2, 5])
            key = rng.choice('xy')
            expected = sum(1 for k2, t2 in accepted if k2 == key and t2 > t - window) < limit
            got = limiter.allow(key, t)
            check(got == expected, f'limit={limit} window={window} at {key}@{t}')
            if got:
                accepted.append((key, t))
            calls += 1
    for _ in range(800):
        limits, window = {'user': rng.randint(1, 3), 'experience': rng.randint(1, 4)}, rng.randint(1, 6)
        limiter, accepted, t = ns['MultiCapRateLimiter'](limits, window), [], 0
        for _ in range(rng.randint(1, 50)):
            t += rng.choice([0, 0, 1, 2, 4])
            keys = {'user': rng.choice('ab'), 'experience': rng.choice('xyz')}
            expected = all(sum(1 for t2, k2 in accepted if k2[d] == keys[d] and t2 > t - window) < cap for d, cap in limits.items())
            got = limiter.allow(t, **keys)
            check(got == expected, f'multi-cap {limits} window={window} at {keys}@{t}')
            if got:
                accepted.append((t, keys))
            calls += 1
    return calls


def oracle_build_order(decls):
    pos = {d[0]: i for i, d in enumerate(decls)}
    req = {d[0]: set(d[1:]) for d in decls}
    if any(x not in pos for s in req.values() for x in s):
        return ['Error']
    depth, state = {}, {}

    def level(c):
        if state.get(c) == 'busy':
            raise RecursionError
        if c in depth:
            return depth[c]
        state[c] = 'busy'
        depth[c] = 1 + max((level(d) for d in req[c]), default=-1)
        state[c] = 'done'
        return depth[c]
    try:
        for c in req:
            level(c)
    except RecursionError:
        return ['Error']
    return sorted(req, key=lambda c: (depth[c], pos[c]))


def test_build_order(ns, rng):
    examples = [([["head"], ["torso"], ["leg"]], ["head", "torso", "leg"]),
                ([["head", "leg"], ["leg", "head"]], ["Error"]),
                ([["head"], ["leg", "torso"]], ["Error"]),
                ([["hair", "head"], ["torso"], ["leg", "torso"], ["head"]], ["torso", "head", "hair", "leg"])]
    for decls, expected in examples:
        check(ns['build_order'](decls) == expected, f'reported example {decls}')
    calls = len(examples)
    for _ in range(3000):
        names = rng.sample('abcdefgh', rng.randint(1, 8))
        decls = [[n] + rng.sample(names + ['zz'] * (rng.random() < 0.05), rng.randint(0, min(3, len(names)))) for n in names]
        decls = [[d[0]] + [x for x in d[1:] if rng.random() < 0.9 or x == 'zz'] for d in decls]
        check(ns['build_order'](decls) == oracle_build_order(decls), f'build order {decls}')
        calls += 1
    return calls


def oracle_crush(board):
    board = [row[:] for row in board]
    R, C = len(board), len(board[0])
    while True:
        marked = set()
        for r in range(R):
            for c in range(C):
                v = board[r][c]
                if not v:
                    continue
                for dr, dc in ((1, 0), (0, 1)):
                    cells = [(r + k * dr, c + k * dc) for k in range(3)]
                    if all(0 <= rr < R and 0 <= cc < C and board[rr][cc] == v for rr, cc in cells):
                        marked.update(cells)
        if not marked:
            return board
        for r, c in marked:
            board[r][c] = 0
        for c in range(C):
            col = [board[r][c] for r in range(R) if board[r][c]]
            col = [0] * (R - len(col)) + col
            for r in range(R):
                board[r][c] = col[r]


def test_match3(ns, rng):
    board = [[1, 1, 1, 2], [3, 2, 4, 2], [3, 5, 5, 2], [3, 4, 4, 1]]
    check(ns['find_matches'](board) == [(3, 3, (1, 0), 'V'), (2, 3, (0, 3), 'V'), (1, 3, (0, 0), 'H')], 'order of reported runs')
    calls = 1
    for _ in range(2500):
        R, C = rng.randint(1, 6), rng.randint(1, 6)
        board = [[rng.randint(1, 3) for _ in range(C)] for _ in range(R)]
        expected = oracle_crush(board)
        got = ns['candy_crush']([row[:] for row in board])
        check(got == expected, f'candy crush {board}')
        for value, length, (r, c), d in ns['find_matches'](board):
            cells = [(r + k, c) if d == 'V' else (r, c + k) for k in range(length)]
            check(length >= 3 and all(board[rr][cc] == value for rr, cc in cells), f'bad run {board}')
        calls += 2
    return calls


def oracle_exists(board, word):
    R, C = len(board), len(board[0])

    def dfs(r, c, i, used):
        if board[r][c] != word[i]:
            return False
        if i == len(word) - 1:
            return True
        used.add((r, c))
        ok = any(0 <= rr < R and 0 <= cc < C and (rr, cc) not in used and dfs(rr, cc, i + 1, used)
                 for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))
        used.discard((r, c))
        return ok
    return any(dfs(r, c, 0, set()) for r in range(R) for c in range(C))


def test_word_search(ns, rng):
    board = [list('oaan'), list('etae'), list('ihkr'), list('iflv')]
    check(set(ns['find_words'](board, ['oath', 'pea', 'eat', 'rain'])) == {'oath', 'eat'}, 'LeetCode 212 example')
    calls = 1
    for _ in range(1500):
        R, C = rng.randint(1, 4), rng.randint(1, 4)
        board = [[rng.choice('abc') for _ in range(C)] for _ in range(R)]
        words = list({''.join(rng.choice('abc') for _ in range(rng.randint(1, 6))) for _ in range(8)})
        found = ns['find_words'](board, words)
        check(set(found) == {w for w in words if oracle_exists(board, w)}, f'word set {board} {words}')
        for w, path in found.items():
            check(len(path) == len(w) == len(set(path)) and ''.join(board[r][c] for r, c in path) == w
                  and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:])), f'bad path {w} {path}')
        calls += 1
    return calls


TESTS = [('no-adjacent scheduling', ('can_arrange', 'arrange_no_adjacent', 'least_interval'), test_no_adjacent),
         ('call path', ('most_frequent_call_path', 'most_frequent_call_path_threads'), test_call_path),
         ('median', ('MedianFinder',), test_median),
         ('target windows', ('windows_with_target', 'first_window_with_most_target'), test_windows),
         ('rate limiter', ('SlidingWindowRateLimiter', 'MultiCapRateLimiter'), test_rate_limiter),
         ('build order', ('build_order',), test_build_order),
         ('match-3', ('find_matches', 'crush_once', 'candy_crush'), test_match3),
         ('word search', ('find_words',), test_word_search)]


def section_fences(source):
    """Python fences of section SECTION inside CHAPTER; headings inside fences are ignored."""
    fence, keep, code, first, in_chapter, in_section, chapters = None, False, [], 0, False, False, 0
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        if fence is None:
            heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading and len(heading.group(1)) == 1:
                in_chapter = heading.group(2) == CHAPTER
                chapters += int(in_chapter)
                in_section = False
            elif heading and len(heading.group(1)) == 2:
                in_section = in_chapter and heading.group(2).startswith(SECTION + " ")
            opening = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$", line)
            if opening:
                fence, info = opening.groups()
                language = info.strip().split(maxsplit=1)
                keep = in_section and bool(language) and language[0].lower() in {"python", "python3", "py"}
                code, first = [], number + 1
            continue
        if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*[\r\n]*", line):
            if keep:
                yield first, "".join(code)
            fence = None
        elif keep:
            code.append(line)
    check(fence is None, "unclosed Markdown code fence")
    check(chapters == 1, f"expected one '# {CHAPTER}' chapter, found {chapters}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "08_company_interviews.md"
    namespace, count = {}, 0
    for first_line, code in section_fences(path.read_text(encoding="utf-8")):
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
        count += 1
    rng, total = random.Random(20261015), 0
    for name, required, test in TESTS:
        missing = [r for r in required if r not in namespace]
        check(not missing, f"{name}: the chapter must define {missing}")
        total += test(namespace, rng)
    print(f"Roblox coding drills passed: {count} Python fences, {len(TESTS)} problems against independent oracles; {total:,} checks.")


if __name__ == "__main__":
    main()
