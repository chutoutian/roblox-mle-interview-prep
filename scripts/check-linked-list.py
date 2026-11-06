#!/usr/bin/env python3
"""Regression-check the actual linked-list Python examples in the study notes.

Usage: python3 scripts/check-linked-list.py [path/to/notes.md]
Only run this against trusted Markdown: its Python fences are executed.
"""

from collections import OrderedDict, deque
from pathlib import Path
import random
import re
import signal
import sys


CURRENT_CASE = "loading examples"
CALLS = 0
STATES = 0


def require(condition, message):
    if not condition:
        raise AssertionError(f"{CURRENT_CASE}: {message}")


def python_fences(source):
    opening = None
    code = []
    for line_number, line in enumerate(source.splitlines(keepends=True), 1):
        if opening is None:
            match = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$", line)
            if match:
                fence, info = match.groups()
                language = info.strip().split(maxsplit=1)
                opening = (fence, bool(language and language[0].lower() in {"python", "python3", "py", "py3"}), line_number + 1)
                code = []
            continue
        fence, is_python, first_line = opening
        if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*[\r\n]*", line):
            if is_python:
                yield first_line, "".join(code)
            opening = None
        elif is_python:
            code.append(line)
    require(opening is None, "unclosed Markdown code fence")


def load_examples(path):
    global CURRENT_CASE
    namespace = {"__name__": "linked_list_notes", "__file__": str(path)}
    fences = list(python_fences(path.read_text(encoding="utf-8")))
    require(fences, "no Python code fences found")
    for first_line, code in fences:
        CURRENT_CASE = f"Python fence at {path}:{first_line}"
        exec(compile("\n" * (first_line - 1) + code, str(path), "exec"), namespace)
    for name in ("Node", "DoublyLinkedList", "MyCircularQueue", "LRUCache"):
        require(callable(namespace.get(name)), f"missing example class {name}")
    return namespace, len(fences)


def call(method, *args):
    global CALLS
    CALLS += 1
    return method(*args)


def check_order(order, expected_nodes=None):
    """Validate both directions independently, with bounded traversal on cycles."""
    global STATES
    STATES += 1
    require(order.head is not order.tail, "head and tail must be distinct sentinels")
    require(order.head.prev is None, "head.prev must be None")
    require(order.tail.next is None, "tail.next must be None")
    nodes, seen = [], {id(order.head)}
    previous, node = order.head, order.head.next
    while node is not order.tail:
        require(node is not None, "forward chain does not reach the tail")
        require(id(node) not in seen, "cycle in forward pointers")
        seen.add(id(node))
        require(node.prev is previous, "broken backward pointer in forward chain")
        nodes.append(node)
        previous, node = node, node.next
    require(order.tail.prev is previous, "tail.prev does not match final real node")

    backward, seen = [], {id(order.tail)}
    following, node = order.tail, order.tail.prev
    while node is not order.head:
        require(node is not None, "backward chain does not reach the head")
        require(id(node) not in seen, "cycle in backward pointers")
        seen.add(id(node))
        require(node.next is following, "broken forward pointer in backward chain")
        backward.append(node)
        following, node = node, node.prev
    require(order.head.next is following, "head.next does not match first real node")
    require([id(node) for node in backward] == [id(node) for node in reversed(nodes)],
            "forward and backward chains contain different nodes")
    if expected_nodes is not None:
        require([id(node) for node in nodes] == [id(node) for node in expected_nodes],
                "linked-list node identities/order differ from expectation")
    return nodes


def check_helpers(namespace):
    global CURRENT_CASE
    CURRENT_CASE = "generic doubly linked list"
    Node, List = namespace["Node"], namespace["DoublyLinkedList"]
    default = Node()
    require((default.key, default.value, default.prev, default.next) == (0, 0, None, None),
            "Node defaults do not match the reusable template")
    order = List()
    check_order(order, [])
    nodes = [Node(key=index, value=index * 10) for index in range(12)]
    expected = []
    # Empty -> singleton -> empty, then first/interior/last removals.
    call(order.append, nodes[0])
    check_order(order, [nodes[0]])
    call(order.remove, nodes[0])
    require(nodes[0].prev is None and nodes[0].next is None, "removed node must be detached")
    check_order(order, [])
    for node in nodes[:4]:
        call(order.append, node)
        expected.append(node)
        check_order(order, expected)
    for node in (nodes[1], nodes[0], nodes[3], nodes[2]):
        call(order.remove, node)
        expected.remove(node)
        require(node.prev is None and node.next is None, "removed node must be detached")
        check_order(order, expected)
    generator = random.Random(206)
    for step in range(1500):
        CURRENT_CASE = f"generic doubly linked list, random step {step}"
        available = [node for node in nodes if node not in expected]
        if available and (not expected or generator.random() < 0.55):
            node = generator.choice(available)
            call(order.append, node)
            expected.append(node)
        else:
            node = generator.choice(expected)
            call(order.remove, node)
            expected.remove(node)
            require(node.prev is None and node.next is None, "removed node must be detached")
        check_order(order, expected)


def check_queue_state(queue, expected, capacity):
    require(queue.capacity == capacity, "queue capacity changed")
    require(queue.size == len(expected), "queue size differs from stored values")
    require(0 <= queue.size <= capacity, "queue exceeds capacity")
    require(call(queue.Front) == (expected[0] if expected else -1), "wrong Front")
    require(call(queue.Rear) == (expected[-1] if expected else -1), "wrong Rear")
    require(call(queue.isEmpty) is (not expected), "wrong isEmpty")
    require(call(queue.isFull) is (len(expected) == capacity), "wrong isFull")
    nodes = check_order(queue.order)
    require([node.value for node in nodes] == list(expected), "queue linked-list order differs from FIFO order")


def check_queues(namespace):
    global CURRENT_CASE
    generator = random.Random(622)
    for capacity in (1, 2, 3, 7, 31):
        queue = namespace["MyCircularQueue"](capacity)
        expected = deque()
        CURRENT_CASE = f"queue capacity {capacity}, initially empty"
        check_queue_state(queue, expected, capacity)
        # Fill, reject overflow without mutation, drain, reject underflow, reuse.
        operations = [("enQueue", index) for index in range(capacity + 2)]
        operations += [("deQueue", None)] * (capacity + 2)
        operations += [("enQueue", 73), ("deQueue", None)]
        operations += [("enQueue", generator.randrange(1001)) if generator.random() < 0.57
                       else ("deQueue", None) for _ in range(800)]
        for step, (operation, value) in enumerate(operations):
            CURRENT_CASE = f"queue capacity {capacity}, step {step}: {operation}({value})"
            before = check_order(queue.order)
            if operation == "enQueue":
                success = len(expected) < capacity
                result = call(queue.enQueue, value)
                if success:
                    expected.append(value)
            else:
                success = bool(expected)
                result = call(queue.deQueue)
                if success:
                    expected.popleft()
                    require(before[0].prev is None and before[0].next is None, "dequeued node remains linked")
            require(result is success, f"expected {success}, got {result!r}")
            if not success:
                check_order(queue.order, before)
            check_queue_state(queue, expected, capacity)


def check_cache_state(cache, expected, capacity):
    require(cache.capacity == capacity, "cache capacity changed")
    require(len(cache.cache) == len(expected) <= capacity, "cache size/capacity mismatch")
    nodes = check_order(cache.order)
    require([(node.key, node.value) for node in nodes] == list(expected.items()),
            "cache list differs from expected LRU -> MRU order")
    require(set(cache.cache) == set(expected), "dictionary keys differ from live list keys")
    for node in nodes:
        require(cache.cache[node.key] is node, "dictionary does not reference the exact live node")
    require(len(nodes) == len(cache.cache), "list and dictionary sizes differ")


def check_caches(namespace):
    global CURRENT_CASE
    generator = random.Random(146)
    for capacity in (1, 2, 3, 7, 31):
        cache = namespace["LRUCache"](capacity)
        expected = OrderedDict()
        CURRENT_CASE = f"LRU capacity {capacity}, initially empty"
        check_cache_state(cache, expected, capacity)
        operations = [("get", 99, None), ("put", 1, 1), ("put", 2, 2),
                      ("get", 1, None), ("put", 3, 3), ("get", 2, None),
                      ("put", 4, 4), ("get", 1, None), ("get", 3, None),
                      ("get", 4, None), ("put", 3, 30), ("get", 3, None),
                      ("put", 0, 0), ("get", 0, None), ("get", 99999, None)]
        # Explicitly refresh the least-recent entry by updating its value.
        operations += [("put", key, key + 100) for key in range(capacity)]
        operations += [("put", 0, 789), ("put", capacity, 456), ("get", 0, None)]
        for _ in range(1200):
            key = generator.randrange(capacity * 2 + 3)
            operations.append(("put", key, generator.randrange(1001)) if generator.random() < 0.58
                              else ("get", key, None))
        for step, (operation, key, value) in enumerate(operations):
            CURRENT_CASE = f"LRU capacity {capacity}, step {step}: {operation}({key}, {value})"
            before = check_order(cache.order)
            if operation == "put":
                call(cache.put, key, value)
                expected[key] = value
                expected.move_to_end(key)
                if len(expected) > capacity:
                    expected.popitem(last=False)
            else:
                result = call(cache.get, key)
                require(result == expected.get(key, -1), f"wrong get result {result!r}")
                if key in expected:
                    expected.move_to_end(key)
                else:
                    check_order(cache.order, before)
            live_ids = {id(node) for node in check_order(cache.order)}
            for node in before:
                if id(node) not in live_ids:
                    require(node.prev is None and node.next is None, "evicted/replaced node remains linked")
            check_cache_state(cache, expected, capacity)


def timeout_handler(signum, frame):
    raise TimeoutError(f"Examples exceeded the 10-second limit while checking {CURRENT_CASE}")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 scripts/check-linked-list.py [path/to/notes.md]")
    path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent.parent / "06_leetcode_notes.md"
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        namespace, fences = load_examples(path)
        check_helpers(namespace)
        check_queues(namespace)
        check_caches(namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(f"Linked-list examples passed: {fences} Python fences, reusable pointer operations, "
          f"FIFO queues and LRU caches at 5 capacities; {CALLS:,} method calls, {STATES:,} pointer-invariant checks.")


if __name__ == "__main__":
    main()
