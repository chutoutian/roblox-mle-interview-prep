#!/usr/bin/env python3
"""Check the actual Python fences in the Tree DFS chapter.

Usage: python3 scripts/check-tree-dfs.py [path/to/notes.md]
Only run this against trusted Markdown: its Python fences are executed.
"""

from collections import Counter, deque
from pathlib import Path
import random
import re
import signal
import sys


CHAPTER = "3. Recursion: Tree DFS, Backtracking, Memoization"
SECTIONS = (2, 2)  # Tree DFS: section 3.2 of the recursion chapter, with its 3.2.x subsections
REQUIRED = (
    "TreeNode", "max_depth", "same_tree", "is_valid_bst", "insert_bst",
    "delete_bst", "lowest_common_ancestor", "path_sum", "is_balanced", "diameter",
)
CURRENT_CASE = "loading Tree DFS examples"
CALLS = 0
STATES = 0


def require(condition, message):
    if not condition:
        raise AssertionError(f"{CURRENT_CASE}: {message}")


def chapter_fences(source):
    """Keep original line numbers and ignore headings inside any code fence."""
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
        if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*[\r\n]*", line):
            if keep:
                yield first_line, "".join(code)
            opening = None
        elif keep:
            code.append(line)
    require(opening is None, "unclosed Markdown code fence")
    require(chapters == 1, f"expected one '# {CHAPTER}' chapter, found {chapters}")


def load_examples(path):
    global CURRENT_CASE
    namespace = {"__name__": "tree_dfs_notes", "__file__": str(path)}
    fences = list(chapter_fences(path.read_text(encoding="utf-8")))
    require(fences, "no Python fences in the Tree DFS chapter")
    for first_line, code in fences:
        CURRENT_CASE = f"Python fence at {path}:{first_line}"
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
    for name in REQUIRED:
        require(callable(namespace.get(name)), f"missing callable example: {name}")
    return namespace, len(fences)


def call(namespace, name, *args):
    global CALLS
    CALLS += 1
    return namespace[name](*args)


def nodes(root):
    """Bound traversal and reject cycles or a child shared by two parents."""
    global STATES
    STATES += 1
    result, seen, pending = [], set(), [root] if root is not None else []
    while pending:
        node = pending.pop()
        require(id(node) not in seen, "tree contains a cycle or a shared child")
        seen.add(id(node))
        require(len(seen) <= 1000, "tree unexpectedly grew beyond 1,000 nodes")
        require(all(hasattr(node, field) for field in ("val", "left", "right")), "invalid tree node")
        result.append(node)
        pending.extend(child for child in (node.right, node.left) if child is not None)
    return result


def snapshot(root):
    return [(id(node), node.val, id(node.left), id(node.right)) for node in nodes(root)]


def query(namespace, name, root, *args):
    before = snapshot(root)
    result = call(namespace, name, root, *args)
    require(snapshot(root) == before, f"{name} mutated its input tree")
    return result


def level_tree(Node, values):
    if not values or values[0] is None:
        return None
    root = Node(values[0])
    pending, index = deque([root]), 1
    while pending and index < len(values):
        parent = pending.popleft()
        for side in ("left", "right"):
            if index < len(values) and values[index] is not None:
                child = Node(values[index])
                setattr(parent, side, child)
                pending.append(child)
            index += 1
    return root


def tree_paths(root):
    """Breadth-first paths provide independent depth, path-sum, and LCA oracles."""
    result, pending = {}, deque([(root, ())]) if root is not None else deque()
    while pending:
        node, prefix = pending.popleft()
        path = prefix + (node,)
        result[id(node)] = path
        pending.extend((child, path) for child in (node.left, node.right) if child is not None)
    return result


def height_oracle(root):
    return max((len(path) for path in tree_paths(root).values()), default=0)


def diameter_oracle(root):
    # Longest shortest path in the undirected tree; independent of height DFS.
    adjacency = {id(node): [] for node in nodes(root)}
    for node in nodes(root):
        for child in (node.left, node.right):
            if child is not None:
                adjacency[id(node)].append(id(child))
                adjacency[id(child)].append(id(node))
    longest = 0
    for start in adjacency:
        pending, seen = deque([(start, 0)]), {start}
        while pending:
            node_id, distance = pending.popleft()
            longest = max(longest, distance)
            for neighbor in adjacency[node_id]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append((neighbor, distance + 1))
    return longest


def shape(root):
    pending, result = [(root, "")], {}
    while pending:
        node, address = pending.pop()
        if node is not None:
            result[address] = node.val
            pending.extend(((node.left, address + "L"), (node.right, address + "R")))
    return result


def clone(Node, root):
    copies = {id(node): Node(node.val) for node in nodes(root)}
    for node in nodes(root):
        copies[id(node)].left = copies.get(id(node.left))
        copies[id(node)].right = copies.get(id(node.right))
    return copies.get(id(root))


def check_queries(namespace):
    global CURRENT_CASE
    Node, generator = namespace["TreeNode"], random.Random(236)
    CURRENT_CASE = "TreeNode defaults and child arguments"
    default = Node()
    require((default.val, default.left, default.right) == (0, None, None), "wrong TreeNode defaults")
    child = Node(-1)
    parent = Node(2, child, default)
    require(parent.left is child and parent.right is default, "constructor lost child identities")
    fixtures = [
        [], [1], [1, 2, 3], [1, 2, None], [1, None, 2],
        [1, 2, None, 3, None, 4],
        [0, 0, 0, 0, 0, 0, 0],
        [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1],
        [1, -2, -3, 1, 3, -2, None, -1],
        # Diameter stays below the root; the equal-height root hides imbalance.
        [0, 1, None, 2, 3, 4, None, None, 5, 6, None, None, 7],
        [0, 1, 2, 3, None, 4, None, 5, None, 6, None],
        [0, -(10 ** 100), 10 ** 100],
        [5, 1, 7, None, None, 4, 9],  # ancestor-bound violation
        [5, 3, 8, None, 5],  # duplicate separated from its equal ancestor
    ]
    trees = [level_tree(Node, values) for values in fixtures]
    for _ in range(160):
        root, slots = None, []
        for index in range(generator.randrange(36)):
            child = Node(generator.randint(-5, 5))
            if index == 0:
                root = child
            else:
                parent, side = slots.pop(generator.randrange(len(slots)))
                setattr(parent, side, child)
            slots.extend(((child, "left"), (child, "right")))
        trees.append(root)
    previous_results = []
    for case, root in enumerate(trees):
        CURRENT_CASE = f"query tree {case}"
        paths, live = tree_paths(root), nodes(root)
        depth = query(namespace, "max_depth", root)
        require(type(depth) is int and depth == height_oracle(root), f"incorrect max_depth: {depth!r}")
        balanced = all(abs(height_oracle(node.left) - height_oracle(node.right)) <= 1 for node in live)
        require(query(namespace, "is_balanced", root) is balanced, "incorrect is_balanced")
        length = query(namespace, "diameter", root)
        require(type(length) is int and length == diameter_oracle(root), f"incorrect diameter in edges: {length!r}")
        ordered = [value for _, value in sorted(shape(root).items(), key=lambda item: item[0] + "M")]
        # L sorts before the node's M marker, and R after it: an inorder oracle.
        valid = all(first < second for first, second in zip(ordered, ordered[1:]))
        require(query(namespace, "is_valid_bst", root) is valid, "incorrect strict BST validity")

        leaves = [tuple(node.val for node in path) for path in paths.values()
                  if path[-1].left is None and path[-1].right is None]
        for target in sorted({-10, -1, 0, 1, 10, 22} | {sum(path) for path in leaves}):
            CURRENT_CASE = f"query tree {case}, path_sum target {target}"
            result = query(namespace, "path_sum", root, target)
            require(isinstance(result, list) and all(isinstance(path, list) for path in result),
                    "path_sum must return a list of lists")
            require(Counter(map(tuple, result)) == Counter(path for path in leaves if sum(path) == target),
                    f"incorrect root-to-leaf paths: {result!r}")
            require(len({id(path) for path in result}) == len(result), "returned paths alias the same list")
            previous_results.append((result, [path[:] for path in result]))

        CURRENT_CASE = f"query tree {case}, same_tree"
        duplicate = clone(Node, root)
        for other in (root, duplicate, trees[(case + 1) % len(trees)], None):
            before = snapshot(other)
            require(query(namespace, "same_tree", root, other) is (shape(root) == shape(other)), "incorrect same_tree")
            require(snapshot(other) == before, "same_tree mutated its second tree")
        if duplicate is not None:
            nodes(duplicate)[-1].val += 100
            require(query(namespace, "same_tree", root, duplicate) is False, "same_tree missed a deep value change")

        if live:
            pairs = [(root, root), (live[-1], live[-1]), (root, live[-1])]
            pairs.extend((generator.choice(live), generator.choice(live)) for _ in range(12))
            for first, second in pairs:
                CURRENT_CASE = f"query tree {case}, LCA values ({first.val}, {second.val})"
                common = [a for a, b in zip(paths[id(first)], paths[id(second)]) if a is b][-1]
                require(query(namespace, "lowest_common_ancestor", root, first, second) is common,
                        "LCA must return the exact common ancestor node, even for repeated values")
    for actual, expected in previous_results:
        require(actual == expected, "a later path_sum call mutated an earlier answer")
    return len(trees)


def check_bst_state(namespace, root, expected):
    live = nodes(root)
    require(len(live) == len(expected), "BST has the wrong number of nodes")
    require({node.val for node in live} == expected, "BST values differ from the set oracle")
    # Independent subtree membership checks also detect misplaced descendants.
    for node in live:
        require(all(child.val < node.val for child in nodes(node.left)), "left subtree violates ancestor bound")
        require(all(child.val > node.val for child in nodes(node.right)), "right subtree violates ancestor bound")
    require(query(namespace, "is_valid_bst", root) is True, "validator rejected a valid BST")


def update_bst(namespace, root, operation, value, expected):
    before = snapshot(root)
    old_nodes = nodes(root)
    target = next((node for node in old_nodes if node.val == value), None)
    replacement = (target.left if target.left is not None else target.right) if target is not None else None
    successor = None
    if operation == "delete_bst" and target is not None and target.left is not None and target.right is not None:
        successor = min(nodes(target.right), key=lambda node: node.val)
        successor_value = successor.val
    result = call(namespace, operation, root, value)
    if operation == "insert_bst":
        expected.add(value)
        if root is not None:
            require(result is root, "insertion replaced an existing root")
        if target is not None:
            require(snapshot(result) == before, "duplicate insertion must leave the BST unchanged")
    else:
        expected.discard(value)
        if target is None:
            require(result is root and snapshot(result) == before, "deleting an absent key changed the tree")
        elif successor is not None:
            require(result is root, "two-child deletion replaced the root instead of copying successor value")
            require(target.val == successor_value, "two-child deletion did not copy the successor value")
            live_ids = {id(node) for node in nodes(result)}
            require(id(target) in live_ids and id(successor) not in live_ids,
                    "two-child deletion must retain the target node and remove its successor")
        elif target is root:
            require(result is replacement, "root deletion must return its only child, or None for a leaf")
    check_bst_state(namespace, result, expected)
    return result


def check_bst_updates(namespace):
    global CURRENT_CASE
    generator = random.Random(450)
    # Root leaf, each one-child case, immediate/deep successor, and successor
    # with a right child; then delete each internal/leaf position independently.
    insertion_orders = [[], [4], [4, 2], [4, 6], [4, 2, 6],
                        [8, 3, 12, 1, 6, 10, 14, 9, 11],
                        [8, 3, 12, 10, 14, 11],
                        [0, -(10 ** 100), 10 ** 100]]
    scenarios = 0
    for values in insertion_orders:
        for deletion in values + [999]:
            root, expected = None, set()
            for value in values + values[:1]:
                CURRENT_CASE = f"BST fixture {values}, insert {value}"
                root = update_bst(namespace, root, "insert_bst", value, expected)
            CURRENT_CASE = f"BST fixture {values}, delete {deletion}"
            root = update_bst(namespace, root, "delete_bst", deletion, expected)
            # Drain by repeatedly deleting the current root, exercising return values.
            while root is not None:
                CURRENT_CASE = f"BST fixture {values}, drain root {root.val}"
                root = update_bst(namespace, root, "delete_bst", root.val, expected)
            scenarios += 1
    for case in range(30):
        root, expected = None, set()
        for step in range(150):
            operation = "insert_bst" if generator.random() < 0.53 else "delete_bst"
            value = generator.randrange(-20, 21)
            CURRENT_CASE = f"random BST {case}, step {step}, {operation}({value})"
            root = update_bst(namespace, root, operation, value, expected)
        while root is not None:
            CURRENT_CASE = f"random BST {case}, drain root {root.val}"
            root = update_bst(namespace, root, "delete_bst", root.val, expected)
        scenarios += 1
    return scenarios


def timeout_handler(signum, frame):
    raise TimeoutError(f"Tree DFS examples exceeded the 20-second limit while checking {CURRENT_CASE}")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 scripts/check-tree-dfs.py [path/to/notes.md]")
    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent.parent / "06_leetcode_notes.md"
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)
    try:
        namespace, fences = load_examples(path)
        trees = check_queries(namespace)
        scenarios = check_bst_updates(namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(f"Tree DFS examples passed: {fences} Python fences, {trees} query trees, "
          f"{scenarios} BST update scenarios; {CALLS:,} function calls, {STATES:,} tree-invariant checks.")


if __name__ == "__main__":
    main()
