#!/usr/bin/env python3
"""Execute and check the actual Python fences in the Dijkstra chapter.

Usage: python3 scripts/check-dijkstra.py [path/to/notes.md]
Only run this against trusted Markdown: its Python fences are executed.
"""

from collections import deque
from pathlib import Path
import random
import re
import signal
import sys


CHAPTER = "4. Dijkstra Shortest Path"
REQUIRED = ("dijkstra", "dijkstra_visited", "network_delay_time", "network_delay_time_visited",
            "minimum_effort_path", "minimum_effort_path_visited")
CURRENT_CASE = "loading Dijkstra examples"
CALLS = 0
INF = float("inf")


def require(condition, message):
    if not condition:
        raise AssertionError(f"{CURRENT_CASE}: {message}")


def chapter_fences(source):
    """Ignore headings inside fences; preserve original traceback line numbers."""
    opening, code, selected, chapters = None, [], False, 0
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        if opening is None:
            heading = re.match(r"^ {0,3}#\s+(.+?)\s*#*\s*$", line)
            if heading:
                selected = heading.group(1) == CHAPTER
                chapters += int(selected)
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
    namespace = {"__name__": "dijkstra_notes", "__file__": str(path)}
    fences = list(chapter_fences(path.read_text(encoding="utf-8")))
    require(fences, "no Python fences in the Dijkstra chapter")
    for first_line, code in fences:
        CURRENT_CASE = f"Python fence at {path}:{first_line}"
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
    for name in REQUIRED:
        require(callable(namespace.get(name)), f"missing callable example: {name}")
    return namespace, len(fences)


def all_pairs(n, edges):
    """Floyd-Warshall oracle, independent of the notes' heap-based code."""
    dist = [[INF] * n for _ in range(n)]
    for u in range(n):
        dist[u][u] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


def call(namespace, name, args):
    global CURRENT_CASE, CALLS
    CURRENT_CASE = f"{name}{args!r}"
    snapshot = repr(args)
    result = namespace[name](*args)
    CALLS += 1
    require(repr(args) == snapshot, "input was mutated")
    return result


def check_dijkstra(namespace):
    rng = random.Random(1631)
    # The worked example of 5.1: s=0, a=1, b=2, c=3.
    graph = [[(1, 4), (2, 1)], [(3, 1)], [(1, 2), (3, 5)], []]
    for name in ("dijkstra", "dijkstra_visited"):
        require(call(namespace, name, (4, graph, 0)) == [0, 3, 1, 4], "worked example of 5.1")
        require(call(namespace, name, (1, [[]], 0)) == [0], "single node")
    cases = 0
    for _ in range(300):
        n = rng.randint(1, 7)
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(0, 9)) for _ in range(rng.randint(0, 14))]
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
        source = rng.randrange(n)
        expected = all_pairs(n, edges)[source]
        for name in ("dijkstra", "dijkstra_visited"):
            result = call(namespace, name, (n, graph, source))
            require(isinstance(result, list) and len(result) == n, "dist must list every node")
            require(result == expected, f"wrong distances: got {result}, expected {expected}")
            require(all(value == INF or type(value) is int for value in result), "distances must be ints or inf")
        cases += 1
    return cases


def check_network_delay(namespace):
    examples = (
        (([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2), 2),
        (([[1, 2, 1]], 2, 1), 1),
        (([[1, 2, 1]], 2, 2), -1),
        (([], 1, 1), 0),
        (([[1, 2, 3], [2, 3, 4], [1, 3, 10]], 3, 1), 7),
    )
    for args, expected in examples:
        for name in ("network_delay_time", "network_delay_time_visited"):
            require(call(namespace, name, args) == expected, f"expected {expected}")
    rng = random.Random(743)
    cases = 0
    for _ in range(300):
        n = rng.randint(1, 6)
        times = [[rng.randint(1, n), rng.randint(1, n), rng.randint(1, 9)] for _ in range(rng.randint(0, 12))]
        k = rng.randint(1, n)
        row = all_pairs(n + 1, [(u, v, w) for u, v, w in times])[k]
        slowest = max(row[1:])
        expected = -1 if slowest == INF else slowest
        for name in ("network_delay_time", "network_delay_time_visited"):
            require(call(namespace, name, (times, n, k)) == expected, f"expected {expected}")
        cases += 1
    return cases


def bottleneck_oracle(heights):
    """Binary search on the allowed step, with a BFS feasibility test: independent of Dijkstra."""
    rows, cols = len(heights), len(heights[0])

    def reachable(limit):
        seen = {(0, 0)}
        queue = deque([(0, 0)])
        while queue:
            r, c = queue.popleft()
            if (r, c) == (rows - 1, cols - 1):
                return True
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen \
                        and abs(heights[nr][nc] - heights[r][c]) <= limit:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        return False

    low, high = 0, max(map(max, heights)) - min(map(min, heights))
    while low < high:
        mid = (low + high) // 2
        if reachable(mid):
            high = mid
        else:
            low = mid + 1
    return low


def check_minimum_effort(namespace):
    examples = (
        ([[1, 2, 2], [3, 8, 2], [5, 3, 5]], 2),
        ([[1, 2, 3], [3, 8, 4], [5, 3, 5]], 1),
        ([[1, 2, 1, 1, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 1, 1, 2, 1]], 0),
        ([[3]], 0),
        ([[1, 10, 6, 7, 9, 10, 4, 9]], 9),
    )
    for heights, expected in examples:
        for name in ("minimum_effort_path", "minimum_effort_path_visited"):
            require(call(namespace, name, (heights,)) == expected, f"expected {expected}")
    rng = random.Random(2020)
    cases = 0
    for _ in range(300):
        rows, cols = rng.randint(1, 5), rng.randint(1, 5)
        top = rng.choice((3, 10, 1000000))
        heights = [[rng.randint(1, top) for _ in range(cols)] for _ in range(rows)]
        expected = bottleneck_oracle(heights)
        for name in ("minimum_effort_path", "minimum_effort_path_visited"):
            result = call(namespace, name, (heights,))
            require(result == expected, f"wrong effort: got {result}, expected {expected}")
        cases += 1
    return cases


def timeout_handler(signum, frame):
    raise TimeoutError(f"Dijkstra examples exceeded the 20-second limit while checking {CURRENT_CASE}")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 scripts/check-dijkstra.py [path/to/notes.md]")
    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent.parent / "06_leetcode_notes.md"
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    try:
        namespace, fences = load_examples(path)
        graphs = check_dijkstra(namespace)
        delays = check_network_delay(namespace)
        grids = check_minimum_effort(namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(f"Dijkstra examples passed: {fences} Python fences, {graphs} random graphs, {delays} network-delay cases, "
          f"{grids} effort grids against independent oracles; {CALLS:,} function calls checked.")


if __name__ == "__main__":
    main()
