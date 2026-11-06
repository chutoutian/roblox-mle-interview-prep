# Part VI — LeetCode Notes

> Reusable templates, common mistakes, and worked examples.

# 1. Binary Search

Start with two templates: **find an exact value**, or **find the first value that meets a condition**.
The bounds, loop condition, and updates work as a set—do not mix lines from different templates.
All array examples below assume ascending order.

## 1.1 Template A: find an exact value

Use a closed interval `[left, right]`: both endpoints are included.

- `right = len(nums) - 1`: the last valid index.
- `while left <= right`: when the pointers meet, one element still needs checking.
- On a mismatch, use `mid - 1` or `mid + 1`: we already know `mid` is not the answer.

### Coding: Template A — exact search `[READY]`

```python
def binary_search_exact(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**Time complexity:** `O(log n)` worst case, where `n = len(nums)`. Each comparison removes about half of the remaining search interval.

**Space complexity:** `O(1)` extra space. Only the two bounds and midpoint are stored; the array is not copied.

If no match exists, the loop ends with `left > right`. Return `-1` for “not found.”
For **Search Insert Position**, change the final return to `left`: it is where the missing value belongs.

## 1.2 Template B: find the first value >= target

This is the **lower bound**: the first matching position or the insertion position.
Use `[left, right)` for the array elements still to check; `right` is excluded.

- `right = len(nums)`: the answer can be `n`, meaning “after the last element.”
- `while left < right`: stop when the boundary position is determined.
- `nums[mid] < target`: too small, so `left = mid + 1`.
- Otherwise, `mid` could be the first qualifying position, so `right = mid`.

### Coding: Template B — lower bound `[READY]`

```python
def lower_bound(nums, target):
    left, right = 0, len(nums)

    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

**Time complexity:** `O(log n)` for a nonempty array, where `n = len(nums)`. Each iteration halves the remaining interval; an empty array returns in `O(1)` time.

**Space complexity:** `O(1)` extra space. The iterative search stores only a fixed number of indices.

This works for empty arrays and targets beyond the last value without extra guards.
It never reads `nums[n]`: while the loop runs, `mid < right <= n`.

To find the **first value > target** instead, change just `nums[mid] < target` to
`nums[mid] <= target`. This skips equal values too; it is called the **upper bound**.

## 1.3 Why mid updates can get stuck

Ask: **after this comparison, could mid still be the answer?**

- No → exclude it with `mid + 1` or `mid - 1`.
- Yes → keep it as a possible answer; when searching leftward, use `right = mid`.

For a loop that shrinks a closed interval until `left == right`, use `while left < right`.
Continuing at equality with `right = mid` would repeat the same state forever.

The usual midpoint rounds down. With `left = 3, right = 4`, it gives `mid = 3`:
`left = mid` would not move, but `right = mid` would.

If you use the **last-valid-answer** variant with updates `left = mid` / `right = mid - 1`,
use the upper midpoint `(left + right + 1) // 2` and `while left < right`.
This variant assumes the initial closed interval contains an answer.
For example, LeetCode 69, **Sqrt(x)**, asks for the largest integer whose square is `<= x`.

These are rules for the templates described here, not universal rules for every binary-search implementation.

## 1.4 LeetCode examples

### 704. Binary Search — exact match

Use **Template A** directly.

`nums = [-1, 0, 3, 5, 9, 12], target = 9` → return index `4`.
If the target is `2`, return `-1`.

### 35. Search Insert Position — insertion position

Use **Template B** directly. For `nums = [1, 3, 5, 6], target = 2`:

| left, right before check | mid / value | Update |
|---|---|---|
| 0, 4 | 2 / 5 | `right = 2` |
| 0, 2 | 1 / 3 | `right = 1` |
| 0, 1 | 0 / 1 | `left = 1` |

Now `left == right == 1`: insert at index `1`.
For target `7`, the same template returns `4`, after the last element.

### 34. First and Last Position — two boundary searches

Use **Template B twice**: first `>= target`, then first `> target`.
The second boundary is just after the last match, so subtract one.

For `[5, 7, 7, 8, 8, 10]`, target `8`: the boundaries are `3` and `5`; return `[3, 4]`.

### Coding: LeetCode 34 — first and last occurrence `[READY]`

```python
def upper_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] <= target:  # Only change from lower_bound
            left = mid + 1
        else:
            right = mid
    return left


def search_range(nums, target):
    first = lower_bound(nums, target)  # Reuse Template B
    if first == len(nums) or nums[first] != target:
        return [-1, -1]
    return [first, upper_bound(nums, target) - 1]
```

**Time complexity:** `O(log n)` worst case, where `n = len(nums)`. At most two binary searches are performed, so `O(log n) + O(log n)` is still `O(log n)`.

**Space complexity:** `O(1)` extra space. Both searches are iterative, and the returned two-element list has constant size.

### 875. Koko Eating Bananas — search for the smallest valid answer

Search **speeds**, not array indices. A slow speed fails; a fast enough speed works.
Use the same boundary idea as B: too slow → `left = mid + 1`; works → `right = mid`.

The closed interval `[1, max(piles)]` contains an answer because the problem guarantees
`h >= len(piles)`. At the maximum speed, every pile takes one hour.
For `piles = [3, 6, 7, 11], h = 8`, speed `3` takes 10 hours; speed `4` takes 8. Return `4`.

### Coding: LeetCode 875 — smallest valid speed `[READY]`

```python
def min_eating_speed(piles, h):
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        # Round each pile's eating time up to a whole hour.
        hours = sum((pile + mid - 1) // mid for pile in piles)
        if hours <= h:
            right = mid
        else:
            left = mid + 1
    return left
```

**Time complexity:** `O(n log(M + 1))`, where `n = len(piles)` and `M = max(piles)`. Finding the initial maximum costs `O(n)`; each binary-search step scans all `n` piles to test a speed. The `+1` also covers `M = 1`, when only the initial scan is needed.

**Space complexity:** `O(1)` extra space. The generator passed to `sum` processes pile costs one at a time instead of building an `n`-element list.

**Practice links:** [704](https://leetcode.com/problems/binary-search/description/) ·
[35](https://leetcode.com/problems/search-insert-position/description/) ·
[34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/) ·
[875](https://leetcode.com/problems/koko-eating-bananas/description/) ·
[69](https://leetcode.com/problems/sqrtx/).

# 2. Doubly Linked Lists

The reusable operations are **remove a known node** and **append it at the back**.
Both take `O(1)` because each node stores `prev` and `next` pointers.
Finding a node by its value still takes `O(n)` unless another structure gives you the node directly.

## 2.1 Shared template: dummy head and tail

Use two dummy nodes so empty lists and single-node lists follow the same pointer rules.
They do not store real entries and must never be removed.

```text
Empty:      head <----------------> tail
Nonempty:   head <-> [A] <-> [B] <-> tail
                     ^       ^
                  first     last
```

`head.next` is the first real node; `tail.prev` is the last.
To remove `X`, connect its neighbors directly: `A <-> X <-> B` becomes `A <-> B`.
To append `X`, insert it between the current last node and `tail`.

### Coding: doubly linked list — two reusable operations `[READY]`

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key      # Needed by LRU; unused by the queue
        self.value = value
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def append(self, node):
        last = self.tail.prev
        node.prev = last
        node.next = self.tail
        last.next = node
        self.tail.prev = node

    def remove(self, node):
        left, right = node.prev, node.next
        left.next = right
        right.prev = left
        node.prev = node.next = None
```

**Time complexity:** `O(1)` for initialization, `append`, and `remove`. Given the node directly, each operation changes only a fixed number of pointers; finding a node by value would require a separate `O(n)` scan.

**Space complexity:** `O(1)` extra space per operation and `O(n)` total storage for `n` linked nodes. Initialization creates only two dummy nodes.

`append` expects a detached node; `remove` expects a real node already in this list.
To move an existing node to the back, **remove it first, then append it**.
Both solutions below reuse these classes: include this helper block before the solution class.

## 2.2 Design Circular Queue — FIFO with a capacity limit

Problem: [Design Circular Queue](https://neetcode.io/problems/design-circular-queue/question?list=neetcode250) (LeetCode 622).

Keep the **front next to head** and the **rear next to tail**.
Enqueue appends at the rear; dequeue removes the front. Track `size` to distinguish empty from full.
Enqueue on a full queue returns `False`; it does not overwrite the oldest entry.

```text
                  dequeue              enqueue
                     |                    |
                     v                    v
head <-> [front / oldest] <-> [rear / newest] <-> tail
```

**A doubly linked list is an option, not a requirement.** A ring-buffer array is the usual
alternative: wrap indices with `% capacity`. A singly linked list with front/rear pointers
also suffices. The version below implements the bounded FIFO behavior with dummy endpoints;
its pointers do not physically form a circle. Freed capacity becomes available after dequeue.

### Coding: Design Circular Queue — reuse the list template `[READY]`

```python
class MyCircularQueue:
    def __init__(self, k):
        self.capacity = k
        self.size = 0
        self.order = DoublyLinkedList()

    def enQueue(self, value):
        if self.isFull():
            return False
        self.order.append(Node(value=value))
        self.size += 1
        return True

    def deQueue(self):
        if self.isEmpty():
            return False
        self.order.remove(self.order.head.next)
        self.size -= 1
        return True

    def Front(self):
        return -1 if self.isEmpty() else self.order.head.next.value

    def Rear(self):
        return -1 if self.isEmpty() else self.order.tail.prev.value

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity
```

**Time complexity:** `O(1)` for initialization and every queue operation. Front/rear pointers and the size counter avoid scanning the list.

**Space complexity:** `O(k)` total space at capacity `k`, with at most `k` data nodes and two dummy nodes. Each operation needs `O(1)` additional working space; this linked-list version allocates nodes as values are enqueued.

**Example, capacity = 2:**

| Operation | Front → rear afterward | Result |
|---|---|---|
| `enQueue(7)` | `[7]` | `True` |
| `enQueue(9)` | `[7, 9]` | `True` |
| `enQueue(11)` | `[7, 9]` | `False`: full |
| `deQueue()` | `[9]` | `True` |
| `enQueue(11)` | `[9, 11]` | `True`: space reused |

Now `Front()` is `9` and `Rear()` is `11`. On an empty queue, both return `-1`.

## 2.3 LRU Cache — hash map + recency order

Problem: [LRU Cache](https://neetcode.io/problems/lru-cache/question?list=neetcode250) (LeetCode 146).

The **hash map finds a node by key**; the **doubly linked list tracks usage order**.
Keep the least recently used (LRU) node next to `head` and the most recently used (MRU) next to `tail`.

```text
cache[key] --------------------+
                               v
head <-> [LRU] <-> ... <-> [key, value] <-> ... <-> [MRU] <-> tail
          ^                                         ^
        evict                              move accessed node here
```

- `get(key)`: missing → `-1`; found → move its node to MRU and return the value.
- `put(key, value)`: update an existing node or append a new one; either way, it becomes MRU.
- If capacity is exceeded, remove `head.next` **from both the list and the hash map**.

Why store the key inside each node? Eviction starts from the list's LRU node, so its key
is needed to delete the matching map entry. A map of `key → value` alone cannot locate list nodes in `O(1)`.

### Coding: LRU Cache — reuse the list template `[READY]`

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}      # key -> Node, not key -> value
        self.order = DoublyLinkedList()

    def _touch(self, node):
        self.order.remove(node)
        self.order.append(node)

    def get(self, key):
        node = self.cache.get(key)
        if node is None:
            return -1
        self._touch(node)
        return node.value

    def put(self, key, value):
        node = self.cache.get(key)
        if node is not None:
            node.value = value
            self._touch(node)
            return

        node = Node(key, value)
        self.cache[key] = node
        self.order.append(node)

        if len(self.cache) > self.capacity:
            oldest = self.order.head.next
            self.order.remove(oldest)
            del self.cache[oldest.key]
```

**Time complexity:** `O(1)` average per `get` or `put`, using average-case hash-map operations and constant-time list updates. Dictionary insertion is amortized because occasional resizing costs more; pathological hash collisions can make an operation `O(k)` for capacity `k`. Initialization is `O(1)`.

**Space complexity:** `O(k)` total space for the hash map and linked list, where `k` is the capacity. The map points to the same list nodes; moving a node does not copy the list.

**Example, capacity = 2.** The list below shows keys, ordered LRU → MRU:

| Operation | Keys afterward | Result |
|---|---|---|
| `put(1, 10)` | `[1]` | Add key 1 |
| `put(2, 20)` | `[1, 2]` | Add key 2 |
| `get(1)` | `[2, 1]` | Return `10`; key 1 becomes MRU |
| `put(3, 30)` | `[1, 3]` | Evict key 2 |
| `get(2)` | `[1, 3]` | Return `-1`; order unchanged |
| `put(1, 99)` | `[3, 1]` | Update value and make key 1 MRU |

Finding/removing the LRU node never requires scanning the list.

## 2.4 What to remember

| | Circular Queue | LRU Cache |
|---|---|---|
| Ordering | Insertion order: FIFO | Most recent successful access/update |
| Insert | Append at rear | Append at MRU end |
| Remove | Front | LRU end, or a known node before moving it |
| Extra state | Size + capacity | Hash map + capacity |
| When full | Reject enqueue | Evict LRU when adding a new key |

Update **both directions** of every link. Never remove a dummy node.
For LRU, reads change order, updates do not create duplicate nodes, and evictions must update both structures.
Test empty/full queues, capacity one, repeated reads, existing-key updates, and eviction.

# 3. Recursion: Tree DFS, Backtracking, Memoization

Tree DFS, backtracking and DFS with memoization are one technique: a function that solves a problem by calling itself on smaller pieces. They differ in what one call returns and in whether two calls can ask the same question. Section 3.1 puts the three side by side, with one template and one way to count the cost for each; then [3.2](#32-tree-dfs) covers tree DFS, [3.3](#33-backtracking) backtracking, and [3.4](#34-dfs--memoization) memoization, each with the same template and cost analysis.



## 3.1 One recursion, three questions

Answer three questions before writing the function:

1. **What does one call return?** A fact about its piece, such as the height of a subtree or the number of ways to finish, is **returned** to the caller. A list of complete answers is **collected**: the call returns nothing and appends `path.copy()` to an outer list.
2. **Can two calls ask the same question?** Every tree node is visited once, so there is nothing to reuse. A tree of choices reaches the same remaining problem along many paths; when the answer depends only on the arguments, cache it.
3. **How many calls, and how much work in each?** Their product is the running time, plus the cost of copying answers out.

| | Tree DFS | Backtracking | DFS + memoization |
|---|---|---|---|
| The recursion walks | the nodes of a given tree | a tree of choices it builds as it goes | the distinct states of a subproblem |
| One call returns | a fact about its subtree | nothing: complete paths go to `result` | the answer for its state |
| Shared mutable data | none, except a path being collected | `path` and `used`: change, recurse, undo | none: the answer depends only on the arguments |
| Same arguments twice? | never | yes, but each path must still be listed | yes, served from the memo |
| Time | nodes × work per node | nodes of the choice tree × work per node + output copied | distinct states × work per state |
| Extra space | recursion depth `O(h)` | depth and current path, plus the output | memo entries + recursion depth |
| Examples here | 104, 100, 98, 701, 450, 236, 113, 110, 543 | 78, 77, 46, 39, 90, 40, 47 | 377, 279, 300, 139, 416, 152 |

**The three templates.** The skeleton is the same: stop at a base case, try each child or choice, combine. What changes is where the answer goes.

```text
# Tree DFS: return a fact about the subtree
def dfs(node):
    if node is None:
        return base
    left = dfs(node.left)
    right = dfs(node.right)
    return combine(node, left, right)

# Backtracking: collect complete paths in result
def dfs(state):
    if complete(state):
        result.append(path.copy())
        return
    for choice in choices(state):
        path.append(choice)                 # choose
        dfs(next_state(state, choice))      # explore
        path.pop()                          # undo

# DFS + memoization: return the answer of a state, computed once
def dfs(state):
    if is_base(state):
        return base_value(state)
    if state in memo:
        return memo[state]
    answer = identity                       # 0 for +, inf for min, False for or
    for choice in choices(state):
        answer = combine(answer, dfs(next_state(state, choice)))
    memo[state] = answer
    return answer
```

**The cost, counted the same way for all three.** Count the calls, multiply by the work one call does apart from its recursive calls, and add what is copied into the output. Space is the deepest chain of calls, plus the memo, plus the output.

| Problem | Calls | Work per call | Output | Time | Extra space |
|---|---|---|---|---|---|
| 104. Maximum depth | `n` nodes | `O(1)` | one number | `O(n)` | `O(h)` |
| 78. Subsets | `2^n` nodes of the choice tree | `O(1)`, plus copying up to `n` values | `n · 2^n` values | `O(n · 2^n)` | `O(n)` + output |
| 46. Permutations | fewer than `e · n!` nodes | `O(n)` scan of `used` | `n · n!` values | `O(n · n!)` | `O(n)` + output |
| 377. Combination Sum IV | `target + 1` states | `O(len(nums))` | one number | `O(target · len(nums))` | `O(target)` |
| 416. Partition Equal Subset Sum | `n · (target + 1)` states | `O(1)` | `True` or `False` | `O(n · target)` | `O(n · target)` |

The permutation tree has `n!/(n-k)!` nodes at depth `k`, and the sum over all depths is `n! · (1 + 1/1! + 1/2! + ...) < e · n!`. For backtracking the output alone is exponential, so no cache can make it polynomial. For memoization the same arguments come back, so the number of calls drops from the size of the choice tree to the number of distinct states.

**One input, three questions.** The same numbers give different recursions depending on what is asked. With `nums = [1, 2, 3]` and `target = 4`:

| Question | Answer | Form | State | Where |
|---|---|---|---|---|
| List the combinations, each number reusable (39) | `[1,1,1,1]`, `[1,1,2]`, `[1,3]`, `[2,2]` | backtracking | `(start, remaining)` and `path` | [3.3.6](#336-allow-reusing-an-element-i-instead-of-i--1) |
| Count those combinations (518) | 4 | memoization, `+` | `(start, remaining)` | [3.4.3](#343-377-combination-sum-iv-count-ordered-ways-combine-with-) |
| Count the ordered sequences (377) | 7 | memoization, `+` | `remaining` | [3.4.3](#343-377-combination-sum-iv-count-ordered-ways-combine-with-) |
| Does some subset, each number used once, sum to 4? (the core of 416) | yes, `1 + 3` | memoization, `or` | `(i, remaining)` | [3.4.7](#347-416-partition-equal-subset-sum-take-or-skip-combine-with-or) |

Listing needs the path, so every path is walked. Counting or deciding needs only what is left to do, so two paths that leave the same problem share one answer:

```text
nums = [1, 2, 3, 4], target 5             state = (next index i, remaining)

take 1, take 2, skip 3   →  (3, 2)  ┐
skip 1, skip 2, take 3   →  (3, 2)  ┘  the same question: can a subset of [4] sum to 2?
```

Backtracking explores below both paths; memoization answers `(3, 2)` once.

**Which one to write.**

1. **The input is a tree and the question is about it:** tree DFS. Decide first what one call returns ([3.2.1](#321-choose-the-return-contract)).
2. **The answer is every solution:** backtracking. The output itself is exponential; prune branches that cannot finish.
3. **The answer is a count, a best value, or yes or no, and what is left can be described by a few numbers:** DFS + memoization, with those numbers as the state ([3.4.9](#349-finding-the-state-and-common-mistakes)).
4. **The state would have to include the whole path:** the subproblems never repeat, so a memo cannot help. Stay with backtracking and prune.

<a id="3-tree-dfs"></a>

## 3.2 Tree DFS

Before writing recursion, finish this sentence: **“For the subtree rooted at `node`, this function returns ___.”**
Then choose the empty-tree result, recurse into children, and combine their answers. The return type follows the meaning—not the other way around.

<a id="31-choose-the-return-contract"></a>

### 3.2.1 Choose the return contract

| Return from DFS | Meaning | Representative problems |
|---|---|---|
| Number | A measurement of this subtree | 104: maximum depth |
| `bool` | Whether this subtree satisfies a condition | 100: same tree; 98: valid BST |
| Node: updated root | The subtree after modification | 701: BST insertion; 450: BST deletion |
| Node: answer | A matching node or an ancestor | 236: lowest common ancestor |
| `None` + collected output | Visit paths and save results elsewhere | 113: path sum II |
| Tuple | Several facts the parent needs together | 110: balanced binary tree |

**Shared node definition.** LeetCode supplies this class; include it when running the examples locally.
The functions below are standalone helpers rather than `Solution` methods.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

Here, `n` means the number of nodes and `h` the tree height. Recursive stack space is `O(h)`, which can be `O(n)` for a chain—not always `O(log n)`. Very deep trees can exceed Python's recursion limit; an explicit stack is the iterative alternative.

<a id="32-return-a-number-subtree-information"></a>

### 3.2.2 Return a number: subtree information

**Template:** empty value → ask both children → combine their numbers.

For [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/), `max_depth(node)` returns the number of nodes on its longest downward path. Empty tree → `0`; leaf → `1`.

#### Coding: DFS returns a number — maximum depth `[READY]`

```python
def max_depth(root):
    if root is None:
        return 0

    left = max_depth(root.left)
    right = max_depth(root.right)
    return 1 + max(left, right)
```

**Time complexity:** `O(n)`. Each node is visited once, and combining its two child heights takes constant time.

**Space complexity:** `O(h)` extra space for the recursion stack: `O(log n)` for a balanced tree and `O(n)` for a chain. The returned height occupies `O(1)` space.

For a root whose left subtree has height `2` and right subtree has height `1`, return `1 + max(2, 1) = 3`.
For a node-count problem, the same pattern becomes `1 + left + right` instead.

<a id="33-return-bool-check-a-condition"></a>

### 3.2.3 Return bool: check a condition

**Template:** handle empty cases → reject a local mismatch → combine recursive checks.

Use **`and`** when both branches must pass. Use **`or`** when finding one successful branch is enough. The empty-tree result depends on the question: an empty BST is valid, but an empty tree has no root-to-leaf path.

#### Coding: DFS returns bool — same tree `[READY]`

[100. Same Tree](https://leetcode.com/problems/same-tree/) checks both structure and values. This DFS compares a **pair of nodes**, not just one node.

```python
def same_tree(p, q):
    if p is None or q is None:
        return p is q  # True only when both are None.
    if p.val != q.val:
        return False

    return same_tree(p.left, q.left) and same_tree(p.right, q.right)
```

**Time complexity:** `O(min(n1, n2) + 1)` worst case, where `n1` and `n2` are the two tree sizes. Each matching node pair is compared once, and a mismatch can stop the search early. The `+1` covers an empty-tree comparison.

**Space complexity:** `O(min(h1, h2) + 1)` extra space for the recursion stack, where `h1` and `h2` are the two heights. Recursion only continues while both nodes exist; branches run sequentially, not with separate full stacks.

Both empty → `True`; only one empty → `False`; otherwise values and both matching child pairs must agree.
For mirror symmetry, compare opposite children instead: left with right, then right with left.

#### Coding: DFS returns bool — validate BST with bounds `[READY]`

[98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) needs **ancestor constraints**, not just parent-child comparisons. Pass the allowed interval downward; return validity upward. Values must lie strictly inside the interval, so duplicates are invalid.

```python
def is_valid_bst(root):
    def dfs(node, low, high):
        if node is None:
            return True
        if not low < node.val < high:
            return False

        return (dfs(node.left, low, node.val)
                and dfs(node.right, node.val, high))

    return dfs(root, float("-inf"), float("inf"))
```

```text
        5
       / \
      3   8
         / \
        4   9

4 < 8, but 4 is also right of 5.
Its allowed range is (5, 8): invalid.
```

**Time complexity:** `O(n)` worst case. Each node is checked once against its inherited bounds; an invalid value or failed branch can stop the search early.

**Space complexity:** `O(h)` extra space for recursion. Each call stores only a node reference and two bounds, not a copy of its subtree.

<a id="34-return-node-rebuild-a-subtree"></a>

### 3.2.4 Return node: rebuild a subtree

**Template:** return the updated subtree root, and let the caller reconnect it.

```text
node.left  = dfs(node.left, ...)
node.right = dfs(node.right, ...)
return node
```

These are pointer assignments, not copies of entire subtrees. If a recursive call returns a replacement node or `None`, the parent must store that result. Reassigning a local parameter alone does not change the parent's child pointer.

**At the top level, capture the result too:** `root = insert_bst(root, value)` or `root = delete_bst(root, key)`.

#### Coding: DFS returns an updated root — BST insertion `[READY]`

[701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/) follows one branch using the BST ordering. Reaching `None` means the insertion position has been found: return a new node.

```python
def insert_bst(root, val):
    if root is None:
        return TreeNode(val)

    if val < root.val:
        root.left = insert_bst(root.left, val)
    elif val > root.val:
        root.right = insert_bst(root.right, val)
    return root
```

**Time complexity:** `O(h)`. BST ordering selects one root-to-insertion path: `O(log n)` for a balanced tree, but `O(n)` for a chain.

**Space complexity:** `O(h)` extra space for recursion, plus `O(1)` for the new node. The existing tree is updated in place rather than copied.

LeetCode guarantees a new value; this reusable version leaves an existing value unchanged.

**Insert `4` into a tree rooted at `5` with left child `3`:** the call on `3.right` receives `None`, returns `TreeNode(4)`, and the caller stores it in `3.right`. The next returns are node `3`, then node `5`—the updated root at each level.

#### Coding: DFS returns an updated root — BST deletion `[READY]`

For [450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/), first find the target, then choose its replacement:

- **No children:** return `None`.
- **One child:** return that child, bypassing the deleted node.
- **Two children:** copy the smallest value from the right subtree (the successor), then delete that successor from the right subtree.

```python
def delete_bst(root, key):
    if root is None:
        return None

    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        if root.left is None:
            return root.right  # Also handles a leaf: returns None.
        if root.right is None:
            return root.left

        successor = root.right
        while successor.left is not None:
            successor = successor.left
        root.val = successor.val
        root.right = delete_bst(root.right, successor.val)

    return root
```

```text
Delete 3: return its child 4.

      5                 5
     / \               / \
    3   8     ->      4   8
     \
      4

Delete 5: successor is 7.

      5                 7
     / \               / \
    3   8     ->      3   8
       / \                 \
      7   9                 9
```

**Time complexity:** `O(h)`. Finding the target and, when needed, finding and deleting its successor follow paths bounded by the tree height. Walking the successor path twice changes only the constant factor: balanced `O(log n)`, skewed `O(n)`.

**Space complexity:** `O(h)` extra space for recursion. The successor search uses one additional pointer, and deletion reconnects existing nodes without allocating a replacement tree.

In the first example, `root.val == 5`, and `root.left = delete_bst(root.left, 3)` reconnects the returned node `4`.
In the second, the old root object remains, but its value changes to `7`; the original successor node is removed from its old position. Do not copy the value and forget the second deletion.

Missing deletion keys leave the tree unchanged. These operations do not rebalance a skewed tree.

<a id="35-return-node-find-an-answer"></a>

### 3.2.5 Return node: find an answer

A returned node can instead be a **search result**, with no tree modification. For [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/), return a target found below, or the ancestor where the two search results meet.

#### Coding: DFS returns a candidate node — lowest common ancestor `[READY]`

```python
def lowest_common_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left is not None and right is not None:
        return root
    return left if left is not None else right
```

**Time complexity:** `O(n)` worst case. Unlike a BST search, both subtrees may need to be explored; each node is visited at most once.

**Space complexity:** `O(h)` extra space for recursion. The answer is one reference to an existing node, so output space is `O(1)`.

- Results on **both sides** → this node is where the targets meet.
- Result on **one side** → pass that result upward.
- Neither side → return `None`.

Use `is` to compare node identity, not just values. If `p` is itself an ancestor of `q`, returning `p` is correct.
This version relies on the problem's guarantee that **both targets exist**. It does not independently verify their presence; if that guarantee disappears, track whether each was found.

Unlike insertion/deletion, do **not** assign the returned candidate to `root.left` or `root.right`: that would change the tree.

<a id="36-return-none-collect-paths"></a>

### 3.2.6 Return None: collect paths

**Template:** append → explore children → pop. DFS changes a shared path and result list instead of returning the answer from every call. This is backtracking on a tree whose shape is given: the choose, explore, undo of [3.3.2](#332-the-shared-pattern-choose-explore-undo), with the two children as the only choices.

For [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/), save a path only when it reaches a **leaf** and its sum matches the target.

#### Coding: DFS with backtracking — root-to-leaf paths `[READY]`

```python
def path_sum(root, target_sum):
    result, path = [], []

    def dfs(node, remaining):
        if node is None:
            return

        path.append(node.val)
        remaining -= node.val
        if node.left is None and node.right is None:
            if remaining == 0:
                result.append(path.copy())
        else:
            dfs(node.left, remaining)
            dfs(node.right, remaining)
        path.pop()

    dfs(root, target_sum)
    return result
```

**Time complexity:** `O(n + L)`, where `L` is the total number of values in all returned paths. DFS visits every node once, and each successful `path.copy()` costs the path length. Output copying can make the worst case `O(n²)`, not always `O(n)`.

**Space complexity:** `O(h)` working space for recursion and the current path, plus `O(L)` for the returned paths: `O(h + L)` total extra space. The output can occupy `O(n²)` in the worst case.

The inner `dfs` returns `None`; the outer function returns the collected list.
`path.copy()` saves a snapshot. Without the copy, results would refer to the same list that later gets popped.
The final `pop()` restores the path before visiting a sibling; do not return early from a leaf and skip it.

`remaining` is an integer passed into each call, so its local subtraction needs no undo. Negative values are allowed: do not prune merely because `remaining < 0`.

For a tree with paths `1 → 2 → 5`, `1 → 2 → 4`, and `1 → 3 → 4`, target `8` gives `[[1, 2, 5], [1, 3, 4]]`.

<a id="37-return-a-tuple-combine-states"></a>

### 3.2.7 Return a tuple: combine states

Sometimes a boolean answer is not enough for the parent. [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) needs both **“is this subtree balanced?”** and **“how tall is it?”**.

#### Coding: DFS returns two facts — balanced tree `[READY]`

```python
def is_balanced(root):
    def dfs(node):
        if node is None:
            return True, 0

        left_ok, left_h = dfs(node.left)
        right_ok, right_h = dfs(node.right)
        ok = left_ok and right_ok and abs(left_h - right_h) <= 1
        return ok, 1 + max(left_h, right_h)

    balanced, _ = dfs(root)
    return balanced
```

**Time complexity:** `O(n)`. Each subtree's height and balance are computed once. This implementation visits both children even if the left subtree is already unbalanced.

**Space complexity:** `O(h)` extra space for recursion. Each call keeps a constant-size tuple and a few local values, rather than storing results for all nodes simultaneously.

The recursive contract is always `(balanced, height)`; the outer function keeps only the boolean.
The condition must hold at **every node**, not just the root. Repeatedly recalculating subtree heights instead can take `O(n²)` on a chain.

<a id="38-return-value-versus-final-answer"></a>

### 3.2.8 Return value versus final answer

For [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/), DFS returns **one downward height** to its parent, but the final answer may use **both branches**. The diameter counts edges and may lie entirely below the root.

#### Coding: DFS returns height, but updates diameter `[READY]`

```python
def diameter(root):
    best = 0

    def height(node):
        nonlocal best
        if node is None:
            return 0

        left = height(node.left)
        right = height(node.right)
        best = max(best, left + right)
        return 1 + max(left, right)

    height(root)
    return best
```

**Time complexity:** `O(n)`. Each node contributes one height calculation and one constant-time diameter update; subtree heights are not recalculated separately.

**Space complexity:** `O(h)` extra space for recursion. The shared `best` variable and the returned number use only `O(1)` additional space.

If `left = 2` and `right = 2`, the candidate diameter through this node is `4` edges, but it returns height `3` to its parent.
The parent can extend only **one branch** through this node; returning `left + right` would incorrectly let it reuse a fork as a single path.
`nonlocal best` updates the enclosing function's variable, and `best` starts fresh for each call.

<a id="39-traversal-order-and-common-mistakes"></a>

### 3.2.9 Traversal order and common mistakes

- **Preorder: node → left → right.** Useful when passing current-path state downward, such as a path or ancestor bounds.
- **Inorder: left → node → right.** Visits a strict BST in increasing value order; useful for ordered enumeration and kth-smallest problems.
- **Postorder: left → right → node.** Combine child answers after recursion: height, balance, diameter.

These describe where the useful work happens; they are not separate return types.

Before finishing a solution, check:

1. **Empty tree versus leaf:** `node is None` is not the same as a node with no children.
2. **Return meaning stays consistent:** a height is not a diameter; an answer node is not necessarily a replacement root.
3. **Mutation needs reconnection:** keep returned child roots and the returned top-level root.
4. **Path state needs restoration:** copy saved paths, and pop before returning to the parent.
5. **Use the actual guarantees:** BST ordering, duplicate policy, and target existence change the algorithm.

<a id="4-backtracking"></a>

## 3.3 Backtracking

Backtracking is DFS over a **tree of choices**: choose something, explore what follows, then undo that choice before trying the next option. It is the path collection of [3.2.6](#326-return-none-collect-paths), except that the code builds the tree as it goes: each node's children are the choices still allowed. The three core patterns are **subsets, combinations, and permutations**.

<a id="41-first-decide-whether-order-and-length-matter"></a>

### 3.3.1 First decide whether order and length matter

For `nums = [1, 2, 3]`:

| Type | What counts as an answer? | Does order matter? | When do we save `path`? |
|---|---|---|---|
| Subsets | Any selection: `[]`, `[1]`, `[1, 3]`, ... | No: `[1, 2]` and `[2, 1]` represent the same selection | At every recursion node |
| Combinations | A selection satisfying a condition, such as exactly `k = 2` elements | No: keep just one ordering, such as `[1, 2]` | When the size or target condition is met |
| Permutations | An ordering of all elements: `[1, 2, 3]`, `[2, 1, 3]`, ... | Yes: these are different answers | When every position has been filled |

**Two main selection rules:**

- **Subsets / combinations:** use `start` so indices only move forward. This avoids generating different orderings of the same selection.
- **Permutations:** consider every index at each position, but use `used[i]` to prevent reusing an index already in the current path.

The first three templates assume distinct input values. Reusing an element and handling duplicate input values are separate variations, introduced below.

<a id="42-the-shared-pattern-choose-explore-undo"></a>

### 3.3.2 The shared pattern: choose, explore, undo

```text
path describes the choices made so far

for each allowed choice:
    choose       → append to path; mark used if necessary
    explore      → recurse
    undo         → pop from path; clear used if necessary
```

`dfs` usually returns `None`; it appends completed answers to an outer `result` list. The outer function returns that list. Use **`path.copy()`** when saving an answer: otherwise every result could refer to the same list that later changes.

For example, after exploring the branch starting with `1`:

```text
[] → [1] → [1, 2] → pop → [1] → [1, 3] → pop → [1]
                                                   │
                                                 pop
                                                   ▼
                                                  []
```

The final undo lets the caller try a different first choice. `pop()` does not remove an already saved copy from `result`.

In the analyses below, **working space excludes the returned answers**. Output copying matters: generating exponentially many lists cannot take only linear time or total space.

<a id="43-subsets-save-every-node"></a>

### 3.3.3 Subsets: save every node

For [78. Subsets](https://leetcode.com/problems/subsets/description/), every partial selection is already a valid answer, including `[]`. After choosing index `i`, recurse with **`i + 1`**: that index cannot be reused.

```text
[]                         Save every node in this tree.
├── [1]
│   ├── [1, 2]
│   │   └── [1, 2, 3]
│   └── [1, 3]
├── [2]
│   └── [2, 3]
└── [3]
```

#### Coding: LeetCode 78 — Subsets `[READY]`

```python
def subsets(nums):
    result, path = [], []

    def dfs(start):
        result.append(path.copy())

        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return result
```

**Time complexity:** `O(n * 2^n)`, where `n = len(nums)`. There are `2^n` subsets, and saving a subset copies up to `n` values.

**Space complexity:** `O(n)` working space for the current path and recursion stack, plus `O(n * 2^n)` output space. The empty input returns `[[]]` in constant time and space.

`start` is the **first allowed next index**, not the current depth. After choosing `2`, we never go backward and choose `1`, so `[2, 1]` is not generated again.

<a id="44-combinations-save-only-when-the-size-reaches-k"></a>

### 3.3.4 Combinations: save only when the size reaches k

For [77. Combinations](https://leetcode.com/problems/combinations/description/), choose `k` distinct numbers from `1..n`. This is the subsets pattern with a different saving condition.

For `n = 4, k = 2`, return `[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]`. Do not save `[1]`, and do not continue to `[1, 2, 3]`.

#### Coding: LeetCode 77 — Combinations `[READY]`

```python
def combine(n, k):
    if k < 0 or k > n:
        return []
    result, path = [], []

    def dfs(start):
        if len(path) == k:
            result.append(path.copy())
            return

        needed = k - len(path)
        # Leave enough numbers to fill the remaining positions.
        for value in range(start, n - needed + 2):
            path.append(value)
            dfs(value + 1)
            path.pop()

    dfs(1)
    return result
```

**Time complexity:** `O(k * C(n, k))` for `1 <= k <= n`, where `C(n, k)` is the number of size-`k` combinations. The loop skips prefixes that cannot be completed, and copying each answer costs `O(k)`. `k = 0` and invalid `k` return in `O(1)` time.

**Space complexity:** `O(k)` working space for recursion and the current selection, plus `O(k * C(n, k))` output space. For `k = 0`, the result is `[[]]` with constant space.

**Why that loop bound?** If `n = 4` and we still need two numbers, the next value can be at most `3`: choosing `4` leaves no second number. The last allowed value is `n - needed + 1`; Python's `range` excludes its upper endpoint, hence `+2`.

That is **pruning**: skip a branch once we know it cannot produce an answer. The important template remains `start`, choose, recurse forward, and undo.

<a id="45-permutations-fill-positions-with-unused-indices"></a>

### 3.3.5 Permutations: fill positions with unused indices

For [46. Permutations](https://leetcode.com/problems/permutations/description/), order matters. After choosing `2`, we must still be allowed to choose `1`, so a forward-only `start` would miss valid answers.

```text
[]
├── [1] → [1, 2] → [1, 2, 3]
│       → [1, 3] → [1, 3, 2]
├── [2] → [2, 1] → [2, 1, 3]
│       → [2, 3] → [2, 3, 1]
└── [3] → [3, 1] → [3, 1, 2]
        → [3, 2] → [3, 2, 1]
```

#### Coding: LeetCode 46 — Permutations `[READY]`

```python
def permute(nums):
    result, path = [], []
    used = [False] * len(nums)

    def dfs():
        if len(path) == len(nums):
            result.append(path.copy())
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return result
```

**Time complexity:** `O(n * n!)`, where `n = len(nums)`. There are `n!` complete permutations; copying each one costs `O(n)`. Scanning the indices at intermediate nodes fits within this bound.

**Space complexity:** `O(n)` working space for `used`, `path`, and recursion, plus `O(n * n!)` output space. Empty input returns `[[]]` in constant space.

`used` belongs to the **current path**, not the whole search. When `[1, 2, 3]` is finished, undoing the choices allows those indices to participate in `[1, 3, 2]` and every other branch.

<a id="46-allow-reusing-an-element-i-instead-of-i--1"></a>

### 3.3.6 Allow reusing an element: i instead of i + 1

For [39. Combination Sum](https://leetcode.com/problems/combination-sum/description/), candidates are distinct positive integers and a chosen value may be reused. Order still does not matter.

Change the next start index from **`i + 1` to `i`**. For `[2, 3, 6, 7]`, target `7`, the results are `[[2, 2, 3], [7]]`. We can choose `2` again without generating reordered duplicates such as `[3, 2, 2]`.

#### Coding: LeetCode 39 — Combination Sum `[READY]`

```python
def combination_sum(candidates, target):
    values = sorted(candidates)
    result, path = [], []

    def dfs(start, remaining):
        if remaining == 0:
            result.append(path.copy())
            return

        for i in range(start, len(values)):
            if values[i] > remaining:
                break
            path.append(values[i])
            dfs(i, remaining - values[i])  # Reuse this value.
            path.pop()

    dfs(0, target)
    return result
```

**Time complexity:** `O(n log n + S + L)`, where `n` is the candidate count, `S` is the number of recursive search states visited, and `L` is the total number of values copied into the output. Sorting costs `O(n log n)`; every successful loop iteration creates one child state, with at most one final `break` per state. The search can be exponential, and its depth depends on the target, not just `n`.

**Space complexity:** `O(n + D)` working space for the sorted copy, path, and stack, plus `O(L)` output space, where `D = target // min(candidates)` for nonempty positive candidates. Empty candidates or target zero need only the corresponding constant-depth search.

Sorting makes `break` valid: if this value is already too large, every later value is too large too. `sorted` leaves the caller's list unchanged. Positivity is essential: allowing unlimited zero or negative values would invalidate the shrinking-remainder argument and can make the search nonterminating.

<a id="47-duplicate-inputs-skip-duplicate-branches-not-valid-repeated-values"></a>

### 3.3.7 Duplicate inputs: skip duplicate branches, not valid repeated values

**Input duplicates are not the same as permission to reuse an index.** In `[1, 1, 2]`, the two `1`s occupy different indices. We may use both when allowed, but must not emit the same answer twice.

Sort a copy first so equal values are adjacent. The central rule is: **skip equal alternatives at the same depth**, not equal values anywhere in the path.

```text
Sorted input: [1a, 1b, 2]      a/b distinguish equal-valued indices.

[]
├── choose 1a → [1] → choose 1b → [1, 1]   valid deeper choice
├── choose 1b → [1]                         duplicate sibling: skip
└── choose 2  → [2]
```

#### Coding: LeetCode 90 — Subsets II `[READY]`

[90. Subsets II](https://leetcode.com/problems/subsets-ii/description/) adds one same-depth duplicate check to the subsets template.

```python
def subsets_with_dup(nums):
    values = sorted(nums)
    result, path = [], []

    def dfs(start):
        result.append(path.copy())

        for i in range(start, len(values)):
            if i > start and values[i] == values[i - 1]:
                continue
            path.append(values[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return result
```

**Time complexity:** `O(n log n + n * 2^n)` worst case, including sorting and copying results. Duplicates reduce the number of unique subsets; the no-duplicate input gives the exponential worst case.

**Space complexity:** `O(n)` working space for sorting, the current path, and recursion. Returned subsets take up to `O(n * 2^n)` space.

For `[1, 2, 2]`, return `[]`, `[1]`, `[1, 2]`, `[1, 2, 2]`, `[2]`, and `[2, 2]`, each once. **`i > start`** means another equal choice was already offered in this call. Using `i > 0` instead would incorrectly block taking the second `2` in a deeper call.

#### Coding: LeetCode 40 — Combination Sum II `[READY]`

[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/) allows duplicate positive candidates, but each input index may be used at most once. Combine the same-depth skip with **`dfs(i + 1, ...)`**.

```python
def combination_sum2(candidates, target):
    values = sorted(candidates)
    result, path = [], []

    def dfs(start, remaining):
        if remaining == 0:
            result.append(path.copy())
            return

        for i in range(start, len(values)):
            if i > start and values[i] == values[i - 1]:
                continue
            if values[i] > remaining:
                break
            path.append(values[i])
            dfs(i + 1, remaining - values[i])
            path.pop()

    dfs(0, target)
    return result
```

**Time complexity:** `O(n log n + n * 2^n)` as a conservative worst-case bound: sort, search a pruned subset tree, and copy successful combinations. Unlike Combination Sum, depth is at most `n` because each index is used once.

**Space complexity:** `O(n)` working space for the sorted copy, path, and stack, plus `O(L)` for returned values. Output is at most `O(n * 2^n)`; pruning and duplicate skipping can make it much smaller.

For `[1, 1, 2]`, target `2`, return `[[1, 1], [2]]`. `[1, 1]` is valid because it uses two different input indices; the duplicate rule must not remove it.

#### Coding: LeetCode 47 — Permutations II `[READY]`

[47. Permutations II](https://leetcode.com/problems/permutations-ii/description/) keeps the permutation `used` array and adds a rule that equal-valued indices are selected in a consistent order.

```python
def permute_unique(nums):
    values = sorted(nums)
    result, path = [], []
    used = [False] * len(values)

    def dfs():
        if len(path) == len(values):
            result.append(path.copy())
            return

        for i in range(len(values)):
            if used[i]:
                continue
            if i > 0 and values[i] == values[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(values[i])
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return result
```

**Time complexity:** `O(n * n!)` as a worst-case bound, including sorting, candidate scans, and output copying. Duplicates reduce outputs, but this implementation still scans all `n` indices in each nonterminal call; one unique output does not imply linear runtime.

**Space complexity:** `O(n)` working space for sorting, `used`, `path`, and recursion, plus `O(n * U)` output space for `U` unique permutations. The worst-case output bound is `O(n * n!)`.

For two equal values `1a` and `1b`:

- If `1a` is **not used** in the current path, choosing `1b` now would duplicate the sibling branch that chooses `1a`. Skip it.
- If `1a` **is already used**, choosing `1b` next is valid: this lets us build `[1, 1, 2]`.

Thus `[1, 1, 2]` produces exactly `[[1, 1, 2], [1, 2, 1], [2, 1, 1]]`. `used[i]` means an **index** is taken; it does not mean every occurrence of that value is forbidden.

<a id="48-template-cheat-sheet-and-common-mistakes"></a>

### 3.3.8 Template cheat sheet and common mistakes

| Problem family | Loop over | Recursive update | Save when |
|---|---|---|---|
| Subsets | Indices from `start` | `dfs(i + 1)` | Every call |
| Fixed-size combinations | Values from `start`, leaving enough choices | `dfs(value + 1)` | `len(path) == k` |
| Combination Sum: reusable candidates | Indices from `start` | `dfs(i, remaining - value)` | `remaining == 0` |
| Combination Sum II: each index once | Indices from `start`; skip equal siblings | `dfs(i + 1, remaining - value)` | `remaining == 0` |
| Permutations | Every unused index | `dfs()` with updated `used` | `len(path) == n` |

Before writing code, ask: **Does order matter? Can an index be reused? Are there duplicate input values? What makes a path a complete answer?**

Common mistakes:

1. **Saving the shared list:** append `path.copy()`, not `path`.
2. **Forgetting undo:** every `append` needs a matching `pop`; every `used[i] = True` needs a reset after recursion.
3. **Moving from the wrong index:** after choosing `i`, use `i + 1`, not `start + 1`, when reuse is forbidden.
4. **Using `start` for permutations:** it would lose answers such as `[2, 1, 3]`.
5. **Globally banning equal values:** skip duplicate sibling choices, while allowing valid equal values deeper in the path.
6. **Unjustified pruning:** sum-based `break` requires sorted positive candidates; those assumptions do not apply to every backtracking problem.
7. **Ignoring output cost:** include copying complete paths when explaining time and distinguish working space from returned results.

The reusable functions also cover empty inputs: subsets/permutations of `[]` return `[[]]`; choosing zero elements returns `[[]]`; a positive-candidate combination summing to zero is the empty combination.

<a id="6-dfs--memoization"></a>

## 3.4 DFS + memoization

Backtracking (3.3) lists every answer, so its `dfs` returns nothing and appends finished paths to an outer list. Many problems only ask for a **number**: how many ways, the best length or the fewest pieces, or whether it is possible at all. Then `dfs` can **return** the answer for the subproblem its arguments describe. The same arguments are reached along many different paths, so caching each result turns an exponential search into one computation per distinct argument. This is top-down dynamic programming.

<a id="61-when-memoization-works"></a>

### 3.4.1 When memoization works

Three conditions must hold:

1. **`dfs` returns its result** instead of appending to an outer list.
2. **The result depends only on the arguments:** no shared `path`, no `used` array, no outer counter that changes between calls.
3. **The same arguments come back** along different paths; otherwise the cache never hits.

The cost then follows directly:

```text
time  = number of distinct states × work in one call (not counting the recursive calls)
space = memo entries + recursion depth
```

| Problem | State: the arguments | `dfs(state)` returns | Combine the children with |
|---|---|---|---|
| 377. Combination Sum IV | `remaining` | number of ordered ways to reach exactly 0 | `+`: count |
| 279. Perfect Squares | `remaining` | fewest perfect squares that sum to it | `min`: fewest |
| 300. Longest Increasing Subsequence | `(i, prev)`, or just `i` | length of the longest increasing subsequence from here | `max`: best |
| 139. Word Break | `index` | whether `s[index:]` splits into dictionary words | `or`: any |
| 416. Partition Equal Subset Sum | `(i, remaining)` | whether a subset of `nums[i:]` sums to `remaining` | `or`: take or skip |
| 152. Maximum Product Subarray | `i` | the largest and the smallest product of a subarray ending at `i` | `max` and `min`, then `max` over all `i` |

<a id="62-the-shared-template"></a>

### 3.4.2 The shared template

```text
memo = {}
def dfs(state):
    if state is a base case: return its value      # 1 way / length 0 / True
    if state in memo: return memo[state]
    result = identity                              # 0 for +, 0 for max of lengths, False for or
    for each choice from this state:
        result = combine(result, dfs(next state))
    memo[state] = result                           # store every result, failures included
    return result
return dfs(start)
```

Decorating the inner `dfs` with `functools.cache` does the same bookkeeping. Because the inner function is created anew on every outer call, its cache never leaks into the next test case.

Why it pays off, for `nums = [1, 2, 3]` and `target = 4` in 377:

```text
dfs(4)
├── take 1 → dfs(3)
│   ├── take 1 → dfs(2)    computed: 2 ways (1+1, 2); dfs(1) = 1 is computed inside it
│   ├── take 2 → dfs(1)    memo hit: 1
│   └── take 3 → dfs(0)    base case: 1               → dfs(3) = 2 + 1 + 1 = 4
├── take 2 → dfs(2)        memo hit: 2
└── take 3 → dfs(1)        memo hit: 1                → dfs(4) = 4 + 2 + 1 = 7
```

Without the memo, every subtree is rebuilt each time it appears, and the number of calls grows exponentially with `target`.

<a id="63-377-combination-sum-iv-count-ordered-ways-combine-with-"></a>

### 3.4.3 377. Combination Sum IV: count ordered ways, combine with +

[377. Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/description/): given distinct positive `nums`, count the sequences that sum to `target`, where different orders count as different sequences. For `nums = [1, 2, 3]` and `target = 4` the answer is 7: `(1,1,1,1)`, `(1,1,2)`, `(1,2,1)`, `(2,1,1)`, `(2,2)`, `(1,3)` and `(3,1)`.

Because order matters, any number may come next at every step, so the number of ways to finish depends only on how much is left. The state is `remaining`.

#### Coding: LeetCode 377 — Combination Sum IV `[READY]`

```python
def combination_sum4(nums, target):
    memo = {}

    def dfs(remaining):
        if remaining == 0:
            return 1                        # one way: stop here
        if remaining < 0:
            return 0
        if remaining in memo:
            return memo[remaining]
        ways = 0
        for value in nums:                  # order matters: any number can come next
            ways += dfs(remaining - value)
        memo[remaining] = ways
        return ways

    return dfs(target)
```

**Time complexity:** `O(target * n)`, where `n = len(nums)`: at most `target + 1` states, each looping over the `n` numbers.

**Space complexity:** `O(target)` for the memo and for the recursion, which goes at most `target` levels deep when the smallest number is 1.

The memo key is `remaining` alone, and that is exactly right. A signature such as `dfs(i, remaining)` whose `i` is never read (the loop variable overwrites it) only adds an unused argument; drop it. If order did **not** matter, as in [518. Coin Change II](https://leetcode.com/problems/coin-change-ii/description/), `(1, 3)` and `(3, 1)` would be one answer. The state would then also need where the choices may start, `dfs(start, remaining)` looping `for i in range(start, n)`, which is the `start` rule of [Combination Sum](#336-allow-reusing-an-element-i-instead-of-i--1) in the backtracking part. The problem requires positive numbers: with a zero or a negative value, the same `remaining` could recur forever.

<a id="64-279-perfect-squares-fewest-pieces-combine-with-min"></a>

### 3.4.4 279. Perfect Squares: fewest pieces, combine with min

[279. Perfect Squares](https://leetcode.com/problems/perfect-squares/description/): return the least number of perfect squares (1, 4, 9, ...) that sum to `n`. For `12 = 4 + 4 + 4` the answer is 3, and for `13 = 4 + 9` it is 2.

The state is the same as in 377, what is left, but the question differs: 377 counts every way and adds the children, here we want the fewest pieces and take the minimum over the next square, `dfs(r) = 1 + min(dfs(r - k*k))` over `k*k <= r`, with `dfs(0) = 0`. Greedy is wrong: always taking the largest square gives `12 = 9 + 1 + 1 + 1`, four pieces, so every square has to be tried.

#### Coding: LeetCode 279 — Perfect Squares `[READY]`

```python
import math


def num_squares(n):
    squares = [k * k for k in range(math.isqrt(n), 0, -1)]   # largest first
    memo = {}

    def dfs(remaining):                     # fewest perfect squares that sum to remaining
        if remaining == 0:
            return 0
        if remaining in memo:
            return memo[remaining]
        best = remaining                    # upper bound: 1 + 1 + ... + 1
        for square in squares:
            if square <= remaining:
                best = min(best, 1 + dfs(remaining - square))
        memo[remaining] = best
        return best

    return dfs(n)
```

**Time complexity:** `O(n * sqrt(n))`: at most `n + 1` states, each trying up to `sqrt(n)` squares.

**Space complexity:** `O(n)` for the memo. With the largest square tried first the recursion stays shallow, as measured below.

The order of the loop matters in Python. Trying `1` first makes the very first path `1 + 1 + ... + 1`, `n` calls deep, and LeetCode's `n = 10,000` breaks CPython's 1,000-frame limit. Trying the largest square first reaches small remainders quickly and fills the memo on the way: over every `n` up to 2,000 and 300 random values up to 10,000, the deepest chain measured was 8 calls. That is an observation, not a bound. The same recurrence written bottom-up needs no recursion at all.

#### Coding: LeetCode 279 — the same recurrence as a table `[READY]`

```python
def num_squares_table(n):
    dp = [0] + [n] * n                      # dp[r] = fewest perfect squares that sum to r
    for r in range(1, n + 1):
        k = 1
        while k * k <= r:
            dp[r] = min(dp[r], dp[r - k * k] + 1)
            k += 1
    return dp[n]
```

**Time complexity:** `O(n * sqrt(n))`: the same states and choices as the memoized version.

**Space complexity:** `O(n)` for the table, with no recursion.

Two other views are worth a sentence in an interview. As a shortest path, a BFS from `n` along "subtract a square" edges stops at the first level that reaches 0. And by Lagrange's four-square theorem the answer is never more than 4; Legendre's three-square theorem says it is exactly 4 when `n` has the form `4^a (8b + 7)`, which leads to an `O(sqrt(n))` solution.

<a id="65-300-longest-increasing-subsequence-best-length-combine-with-max"></a>

### 3.4.5 300. Longest Increasing Subsequence: best length, combine with max

[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/description/): return the length of the longest strictly increasing subsequence. For `[10, 9, 2, 5, 3, 7, 101, 18]` the answer is 4, for example `2, 3, 7, 18`.

The take-or-skip view: at index `i`, either skip `nums[i]` or take it, and taking is allowed only if it is larger than the last number taken. What can still be achieved depends on `i` and on the last number taken, so the state is `(i, prev)`, where `prev` is the index of the last taken number and `-1` means none yet.

#### Coding: LeetCode 300 — take or skip, state `(i, prev)` `[READY]`

```python
def length_of_lis_take_skip(nums):
    n = len(nums)
    memo = {}

    def dfs(i, prev):                       # prev: index of the last taken number, -1 if none
        if i == n:
            return 0
        if (i, prev) in memo:
            return memo[(i, prev)]
        best = dfs(i + 1, prev)             # skip nums[i]
        if prev == -1 or nums[prev] < nums[i]:
            best = max(best, 1 + dfs(i + 1, i))   # take nums[i]
        memo[(i, prev)] = best
        return best

    return dfs(0, -1)
```

**Time complexity:** `O(n^2)`: `i` and `prev` each range over about `n` values, with `prev < i`, and each state does constant work.

**Space complexity:** `O(n^2)` for the memo, plus `O(n)` recursion depth.

The two-argument state is correct but stores `n^2` entries, about 3 million for LeetCode's `n = 2,500`. One argument is enough if `dfs(i)` means "the longest increasing subsequence that **starts** with `nums[i]`": its next element is any later, larger number, so `dfs(i) = 1 + max(dfs(j))` over `j > i` with `nums[j] > nums[i]`, and the answer is the largest `dfs(i)`.

#### Coding: LeetCode 300 — one-argument state `dfs(i)` `[READY]`

```python
def length_of_lis(nums):
    n = len(nums)
    memo = {}

    def dfs(i):                             # longest increasing subsequence starting at nums[i]
        if i in memo:
            return memo[i]
        best = 1                            # nums[i] alone
        for j in range(i + 1, n):
            if nums[j] > nums[i]:
                best = max(best, 1 + dfs(j))
        memo[i] = best
        return best

    return max((dfs(i) for i in range(n)), default=0)
```

**Time complexity:** `O(n^2)`: `n` states, each scanning up to `n` later indices.

**Space complexity:** `O(n)` for the memo and the recursion depth.

Both recursive versions can go `n` levels deep, and LeetCode allows `n = 2,500`, above CPython's default limit of 1,000 frames: call `sys.setrecursionlimit(10_000)` first, or fill the same table bottom-up from the last index to the first. The `O(n log n)` solution is a different idea: keep `tails[k]`, the smallest last value of any increasing subsequence of length `k + 1`, and place each number with the [lower bound](#12-template-b-find-the-first-value--target) of Chapter 1.

<a id="66-139-word-break-feasibility-combine-with-or"></a>

### 3.4.6 139. Word Break: feasibility, combine with or

[139. Word Break](https://leetcode.com/problems/word-break/description/): can `s` be split into a sequence of dictionary words, each word usable any number of times? `"leetcode"` with `["leet", "code"]` gives `True`; `"catsandog"` with `["cats", "dog", "sand", "and", "cat"]` gives `False`.

The state is the index where the unsplit suffix begins: `dfs(index)` answers whether `s[index:]` can be split. It tries every word that matches at `index` and succeeds as soon as one continuation succeeds.

#### Coding: LeetCode 139 — Word Break `[READY]`

```python
def word_break(s, word_dict):
    n = len(s)
    memo = {}

    def dfs(index):                         # can s[index:] be split into dictionary words?
        if index == n:
            return True
        if index in memo:
            return memo[index]
        for word in word_dict:
            if s.startswith(word, index) and dfs(index + len(word)):
                memo[index] = True
                return True
        memo[index] = False                 # store failures too, or they are recomputed
        return False

    return dfs(0)
```

**Time complexity:** `O(n * W * L)`, where `n = len(s)`, `W` is the number of words and `L` the length of the longest word: `n + 1` states, each comparing up to `W` words of up to `L` characters.

**Space complexity:** `O(n)` for the memo and the recursion depth.

`s.startswith(word, index)` is the same test as `s[index:index + len(word)] == word` without building a substring. Storing `False` matters most in this problem: failing suffixes are what repeat, and without that line an input such as 30 `a`s followed by one `b`, with words `"a"`, `"aa"` and `"aaa"`, explores an exponential number of splits before giving up. When the dictionary is large, loop over the end index instead and test `s[index:end] in words` with a set.

### 3.4.7 416. Partition Equal Subset Sum: take or skip, combine with or

[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/description/): can the positive numbers `nums` be split into two groups with equal sums? `[1, 5, 11, 5]` gives `True`, as `1 + 5 + 5 = 11`; `[1, 2, 3, 5]` gives `False`.

Two equal groups each sum to half the total, so the question becomes: does some subset sum to `target = total / 2`? An odd total answers `False` at once. This is the subset tree of [3.3.3](#333-subsets-save-every-node) with a yes-or-no question instead of a list: at index `i`, take `nums[i]` or skip it. What is still possible depends only on `i` and on the sum still missing, so the state is `(i, remaining)`, the example of [3.1](#31-one-recursion-three-questions).

#### Coding: LeetCode 416 — Partition Equal Subset Sum `[READY]`

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False
    n = len(nums)
    memo = {}

    def dfs(i, remaining):                  # can a subset of nums[i:] sum to remaining?
        if remaining == 0:
            return True
        if i == n or remaining < 0:
            return False
        if (i, remaining) in memo:
            return memo[(i, remaining)]
        found = dfs(i + 1, remaining - nums[i]) or dfs(i + 1, remaining)   # take, then skip
        memo[(i, remaining)] = found
        return found

    return dfs(0, total // 2)
```

**Time complexity:** `O(n * target)`, where `target = sum(nums) / 2`: at most `n * (target + 1)` states, each doing constant work.

**Space complexity:** `O(n * target)` for the memo, plus `O(n)` recursion depth.

Without the memo this is the subset tree, `2^n` leaves. With it, the cost no longer depends on how many subsets exist, only on how many different `(i, remaining)` pairs do. `or` stops early: once taking `nums[i]` succeeds, skipping it is never tried. LeetCode allows 200 numbers up to 100, so `target` can reach 10,000 and the memo 2 million entries, which is slow as a Python dictionary. The table below keeps one row of booleans instead.

#### Coding: LeetCode 416 — the same recurrence as one row `[READY]`

```python
def can_partition_table(nums):
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    reachable = [True] + [False] * target    # reachable[s]: some subset of the numbers seen so far sums to s
    for value in nums:
        for s in range(target, value - 1, -1):   # downward: each number used at most once
            reachable[s] = reachable[s] or reachable[s - value]
    return reachable[target]
```

**Time complexity:** `O(n * target)`: every number updates every sum from `target` down to its own value.

**Space complexity:** `O(target)`: one row, overwritten for each number.

The direction of the inner loop is what makes each number count once. Going down, `reachable[s - value]` still holds its value from before `value` was considered, so `value` is used at most once: the 0/1 knapsack. Going up, `reachable[s - value]` may already include `value`, so the same number can be taken again: the unbounded version. For counting rather than yes or no, an outer loop over the values with an upward inner loop counts combinations (518), and an outer loop over the sums counts ordered sequences (377).

### 3.4.8 152. Maximum Product Subarray: return two values, combine with max and min

[152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/): the largest product of a non-empty contiguous subarray. `[2, 3, -2, 4]` gives 6, from `[2, 3]`; `[-2, 0, -1]` gives 0; `[-2, 3, -4]` gives 24, the whole array.

A subarray is contiguous, so the natural state is where it ends: `dfs(i)` describes the subarrays that end at index `i`. One number is not enough. A negative `nums[i]` turns the smallest product ending at `i - 1` into the largest one ending at `i`: in `[-2, 3, -4]` the largest product ending at `3` is `3`, but the smallest is `-6`, and `-6 * -4 = 24`. So `dfs(i)` returns a pair, like the tuple contract of tree DFS in [3.2.7](#327-return-a-tuple-combine-states):

```text
dfs(i)   = (hi, lo): the largest and the smallest product of a subarray ending at i
dfs(0)   = (nums[0], nums[0])
dfs(i)   = (max(c), min(c))   where c = (nums[i], hi * nums[i], lo * nums[i]) and (hi, lo) = dfs(i - 1)
answer   = the largest hi over all i
```

`nums[i]` on its own starts a new subarray. That handles zeros: after a `0` both extremes are `0`, and the next number may start afresh.

#### Coding: LeetCode 152 — Maximum Product Subarray `[READY]`

```python
def max_product(nums):
    memo = {}

    def dfs(i):                             # (largest, smallest) product of a subarray ending at i
        if i == 0:
            return nums[0], nums[0]
        if i in memo:
            return memo[i]
        hi, lo = dfs(i - 1)
        candidates = (nums[i], hi * nums[i], lo * nums[i])
        memo[i] = max(candidates), min(candidates)
        return memo[i]

    return max(dfs(i)[0] for i in range(len(nums)))
```

**Time complexity:** `O(n)`: `n` states, each combining three candidates.

**Space complexity:** `O(n)` for the memo. Calling `dfs` for `i = 0, 1, 2, ...` keeps the recursion one level deep, because `dfs(i - 1)` is already stored; calling `dfs(n - 1)` first would go `n` levels deep, beyond CPython's limit for LeetCode's `n = 20,000`.

#### Coding: LeetCode 152 — two variables instead of a memo `[READY]`

```python
def max_product_loop(nums):
    hi = lo = best = nums[0]
    for value in nums[1:]:
        candidates = (value, hi * value, lo * value)
        hi, lo = max(candidates), min(candidates)
        best = max(best, hi)
    return best
```

**Time complexity:** `O(n)`: one pass.

**Space complexity:** `O(1)`: the memo shrinks to the last pair.

Each state reads only the one before it, so the whole memo can shrink to that pair. This is the usual last step for a one-dimensional recursion: write `dfs(i)` from `dfs(i - 1)`, check it with a memo, then keep only the values the next step reads. `hi, lo = max(...), min(...)` assigns both from the old values; updating `hi` first and then computing `lo` from the new `hi` is a classic bug.

<a id="67-finding-the-state-and-common-mistakes"></a>

### 3.4.9 Finding the state, and common mistakes

Three questions, in order:

1. **What does one call return?** A count, a best value, or yes or no. That fixes the combine operator and its starting value.
2. **What must the call know to answer?** Exactly those values are the arguments; nothing about the path that led here.
3. **How many distinct states are there, and how much work does one call do?** Their product is the running time.

Every memoized DFS is a table filled lazily, and filling it in dependency order gives the bottom-up version. For 377:

```text
dp[0] = 1
dp[r] = sum(dp[r - v] for v in nums if v <= r)      for r = 1 .. target
```

Common mistakes:

1. **Memoizing a function that reads outside state:** if the answer depends on a shared `path` or `used`, two calls with equal arguments may need different answers. Problems that list every answer, as in backtracking (3.3), cannot be memoized this way.
2. **A state that is too small:** take-or-skip LIS memoized on `i` alone returns answers that were computed for a different last number.
3. **A state that is too large:** an argument that is never read multiplies the entries and lowers the hit rate.
4. **Caching only successes:** in Word Break, the failures are what repeat.
5. **A memo that outlives the call:** a module-level dictionary or a mutable default argument keeps answers from the previous test case. Create the memo inside the function, or decorate the inner `dfs` with `functools.cache`.
6. **Counting ordered versus unordered:** 377 counts sequences with state `remaining`; 518 counts combinations with state `(start, remaining)`.
7. **Recursion depth:** CPython stops at 1,000 frames by default. Raise the limit, order the choices so that the first path is short (the largest square first in 279), or write the table bottom-up.
8. **A return value that is too small:** in 152 the largest product ending at `i` cannot be extended alone, because a negative number turns the smallest product into the largest. Return what the caller needs, here a pair, as in the tuple contract of tree DFS.
9. **The wrong loop direction in a one-row table:** in 416 the sums must run downward so that each number is used once; running upward reuses it.

# 4. Dijkstra Shortest Path

<a id="5-dijkstra-shortest-path"></a>

Dijkstra's algorithm finds the cheapest route from one source to every node of a graph whose edge weights are **non-negative**. It is BFS with a priority queue: BFS pops nodes in order of hop count, Dijkstra pops them in order of **cost so far**. The two problems below are the classic sum-of-weights form and the less obvious max-of-weights form.
<a id="51-the-invariant-the-closest-unsettled-node-is-final"></a>

## 4.1 The invariant: the closest unsettled node is final

Keep `dist[v]`, the best known cost to reach `v`, starting with `dist[source] = 0` and everything else infinite. Repeatedly take the unsettled node with the smallest `dist`. Its value is final: any other route to it would have to leave the settled set through a node that is already at least as far, and non-negative edges never make a route cheaper. Then **relax** its outgoing edges: if `dist[u] + w < dist[v]`, record the better value.

```text
dist: s=0, a=inf, b=inf, c=inf        edges: s→a 4, s→b 1, b→a 2, b→c 5, a→c 1

pop s (0)   relax s→a: 4      relax s→b: 1
pop b (1)   relax b→a: 1+2=3 < 4, improve      relax b→c: 1+5=6
pop a (3)   relax a→c: 3+1=4 < 6, improve
pop c (4)   nothing left: every dist is final
```

A binary heap returns the smallest unsettled node in `O(log n)`. The heap is never edited in place: when a node improves, push a new `(dist, node)` pair and let the old pair become **stale**. When a stale pair is popped, `d > dist[u]` identifies it and it is skipped.

### Coding: Dijkstra template — distances from one source `[READY]`

```python
import heapq


def dijkstra(n, graph, source):
    """graph[u] lists (v, w) edges with w >= 0; nodes are 0..n-1.
    Returns dist[v] for every node, float('inf') when v is unreachable."""
    INF = float("inf")
    dist = [INF] * n
    dist[source] = 0
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                      # stale entry: u was settled with a smaller distance
        for v, w in graph[u]:
            candidate = d + w
            if candidate < dist[v]:
                dist[v] = candidate
                heapq.heappush(heap, (candidate, v))

    return dist
```

**Time complexity:** `O((V + E) log V)` for `V` nodes and `E` edges. Every edge is relaxed at most once, from its settled tail; each relaxation may push one heap entry, so the heap never holds more than `E + 1` entries and each pop or push costs `O(log E) = O(log V)`.

**Space complexity:** `O(V + E)` for the distance array and the heap, on top of the adjacency lists.

Three details carry the correctness: a node is settled when it is **popped**, not when it is pushed; a node may be pushed several times but is processed only on its first pop; and a negative edge would break the invariant, because a later node could then make an already settled node cheaper. Use Bellman-Ford for negative weights and plain BFS when every weight is the same.

### Coding: Dijkstra template — the `visited` set variant `[READY]`

The same template with a set of settled nodes instead of the stale check. `dist[u]` is written only when `u` is popped for the first time, because that value is final; a push writes nothing, so a node can sit in the heap several times and only its first pop counts.

```python
def dijkstra_visited(n, graph, source):
    dist = [float("inf")] * n
    visited = set()                           # nodes whose distance is final
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue                          # a cheaper pop settled u already
        visited.add(u)
        dist[u] = d                           # final on the first pop; never written at push time
        for v, w in graph[u]:
            if v not in visited:
                heapq.heappush(heap, (d + w, v))

    return dist
```

**Time complexity:** `O((V + E) log V)`, as for the stale-check form: every node is settled once, every edge is relaxed once from its settled tail, and the heap holds at most `E + 1` entries.

**Space complexity:** `O(V + E)` for the distance array, the set, and the heap.

`u in visited` and `d > dist[u]` test the same fact, that `u` was settled by an earlier pop: the stale-check form pushes only strict improvements, so the last entry pushed for `u` is its smallest and equals `dist[u]`, the first pop of `u` returns exactly that entry, and every later entry for `u` is strictly larger. The two forms differ only in what they push: this one pushes every edge to an unsettled neighbour, the stale-check form pushes only improvements. Writing `dist` at push time in this form would need the `candidate < dist[v]` filter again, so that a worse push cannot overwrite a better value, and with that filter the array already records what the set records. The comparison table in 5.3 applies to all three problems.

<a id="52-743-network-delay-time-the-longest-of-the-shortest-paths"></a>

## 4.2 743. Network Delay Time: the longest of the shortest paths

[743. Network Delay Time](https://leetcode.com/problems/network-delay-time/description/): a signal sent from node `k` travels along directed edges `(u, v, w)` in `w` time units. Return the time until every node has received it, or `-1` if some node never does. The answer is the largest single-source distance, so it is the template plus two lines.

For `times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]`, `n = 4`, `k = 2`:

```text
2 → 1          (1)            dist: 2=0, 1=1, 3=1, 4=2
2 → 3 → 4      (1 + 1 = 2)    answer = max = 2
```

### Coding: LeetCode 743 — Network Delay Time `[READY]`

```python
def network_delay_time(times, n, k):
    graph = [[] for _ in range(n + 1)]       # nodes are numbered 1..n
    for u, v, w in times:
        graph[u].append((v, w))

    dist = dijkstra(n + 1, graph, k)
    slowest = max(dist[1:])
    return -1 if slowest == float("inf") else slowest
```

**Time complexity:** `O((n + E) log n)` for `E = len(times)`: building the adjacency lists is linear and the Dijkstra run dominates.

**Space complexity:** `O(n + E)` for the adjacency lists, the distance array, and the heap.

Watch the `-1` case: an unreachable node keeps `float('inf')`, and `max` propagates it. Node `0` is unused because the problem numbers nodes from `1`; slicing `dist[1:]` keeps its infinite entry out of the maximum.

### Coding: LeetCode 743 — with the `visited` template `[READY]`

```python
def network_delay_time_visited(times, n, k):
    graph = [[] for _ in range(n + 1)]
    for u, v, w in times:
        graph[u].append((v, w))

    dist = dijkstra_visited(n + 1, graph, k)  # nodes never popped keep float('inf')
    slowest = max(dist[1:])
    return -1 if slowest == float("inf") else slowest
```

**Time complexity:** `O((n + E) log n)`, the same as with the stale-check template.

**Space complexity:** `O(n + E)` for the adjacency lists, the distance array, the set, and the heap.

Nothing else changes: the answer still needs the distance of every node, so the `visited` form keeps the `dist` array and merely fills it at pop time, and an unreachable node is still the `float('inf')` that was never popped.

<a id="53-1631-path-with-minimum-effort-minimize-the-largest-step-not-the-sum"></a>

## 4.3 1631. Path with Minimum Effort: minimize the largest step, not the sum

[1631. Path with Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/description/): walk from the top-left to the bottom-right cell of a height grid, moving in four directions. The effort of a route is the **largest** absolute height difference between consecutive cells; return the smallest possible effort.

The graph is implicit: every cell is a node and every neighbouring pair is an edge of weight `|h[a] - h[b]|`. The cost of a route is no longer a sum, but Dijkstra still applies, because the invariant only needs that extending a route never makes it cheaper, which holds for `max` just as it holds for `+`. The relaxation becomes:

```text
candidate = max(effort[u], |h[u] - h[v]|)      instead of      candidate = dist[u] + w
```

For `heights = [[1, 2, 2], [3, 8, 2], [5, 3, 5]]` the answer is `2`: the route `1 → 2 → 2 → 2 → 5` along the top row and the right column never changes height by more than `2`, whereas going down the left side through `3` and `5` costs `3`.

### Coding: LeetCode 1631 — Path with Minimum Effort `[READY]`

```python
def minimum_effort_path(heights):
    rows, cols = len(heights), len(heights[0])
    INF = float("inf")
    effort = [[INF] * cols for _ in range(rows)]
    effort[0][0] = 0
    heap = [(0, 0, 0)]                        # (effort so far, row, col)

    while heap:
        e, r, c = heapq.heappop(heap)
        if (r, c) == (rows - 1, cols - 1):
            return e                          # first pop of the target: its effort is final
        if e > effort[r][c]:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                candidate = max(e, abs(heights[nr][nc] - heights[r][c]))
                if candidate < effort[nr][nc]:
                    effort[nr][nc] = candidate
                    heapq.heappush(heap, (candidate, nr, nc))

    return effort[rows - 1][cols - 1]
```

**Time complexity:** `O(R * C * log(R * C))` for an `R x C` grid: `R * C` nodes and at most `4 * R * C` edges, each relaxed once from its settled cell.

**Space complexity:** `O(R * C)` for the effort grid and the heap.

Returning on the **first pop** of the target is safe for the same reason the invariant holds: a popped node's value is final. Checking the target when it is pushed is not safe, because a cheaper route to it may still be waiting in the heap. The `1 x 1` grid returns `0` at the first pop.

### Coding: LeetCode 1631 — the `visited` set variant `[READY]`

The same algorithm can be written without the `effort` grid. A set records which cells are already settled; a cell is settled the first time it is popped, and a later pop of the same cell is skipped because it is in the set. Neighbours are pushed unconditionally, except settled ones.

```python
def minimum_effort_path_visited(heights):
    rows, cols = len(heights), len(heights[0])
    visited = set()                           # cells whose effort is final
    heap = [(0, 0, 0)]                        # (effort so far, row, col)

    while heap:
        e, r, c = heapq.heappop(heap)
        if (r, c) in visited:
            continue                          # a cheaper pop settled this cell already
        visited.add((r, c))
        if (r, c) == (rows - 1, cols - 1):
            return e
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                heapq.heappush(heap, (max(e, abs(heights[nr][nc] - heights[r][c])), nr, nc))

    return 0                                  # never reached: a grid is always connected
```

**Time complexity:** `O(R * C * log(R * C))`, the same bound as above. Every cell is settled once and each settlement pushes at most four entries, so the heap holds at most `4 * R * C` entries; each push or pop costs `O(log(R * C))`.

**Space complexity:** `O(R * C)` for the heap and the set of settled cells.

Both versions are correct for the same reason: the invariant lives in the **pop order**, so a cell's effort is final the first time it is popped whatever bookkeeping records it. The difference is what gets pushed:

| | `effort` grid + stale check | `visited` set |
|---|---|---|
| A cell is settled | on its first pop; later pops fail `e > effort[r][c]` | on its first pop; later pops find it in the set |
| A neighbour is pushed | only when the new effort improves `effort[nr][nc]` | always, unless the neighbour is already settled |
| Heap entries | one per improvement | up to four per settled cell, including dominated duplicates |
| Extra memory | `R x C` numbers | the set of settled cells |
| Efforts of other cells | available in the grid afterwards | not kept |

The `effort` form is the standard Dijkstra shape and is the one to reuse when the distances to every node are the answer, as in 743. The `visited` form is shorter to write in an interview and just as fast in practice; its only cost is pushing entries that a better entry already in the heap makes useless, which the `candidate < effort[nr][nc]` test avoids. Two details of a first draft are usually unnecessary: a grid of best efforts that is never updated once the set exists, and a `return` after the loop, which is never reached because the target is always popped.

<a id="54-when-dijkstra-applies-and-what-else-to-reach-for"></a>

## 4.4 When Dijkstra applies, and what else to reach for

| Situation | Tool | Why |
|---|---|---|
| Non-negative weights, sum of weights | Dijkstra | the popped node is final; `O(E log V)` |
| Non-negative weights, max of weights (bottleneck) | Dijkstra with `max` relaxation | extending a route never lowers its cost |
| All weights equal | BFS | the queue already pops in distance order |
| Weights `0` and `1` only | 0-1 BFS with a deque | push weight-0 neighbours to the front |
| Negative weights, no negative cycle | Bellman-Ford | relaxes every edge for `V - 1` rounds |
| Minimum bottleneck with a yes/no check | binary search on the threshold + BFS | an alternative for 1631: `O(R * C * log(max height))` |

Common mistakes:

1. **Marking nodes visited when pushed:** a node can be pushed again with a better distance; settle it when it is popped and skip stale pops with `d > dist[u]`.
2. **Editing the heap entry instead of pushing a new one:** Python's `heapq` has no decrease-key, so push the improved pair and let the old one go stale.
3. **Stopping too early:** the answer for a target is final when it is popped, not when it is first reached.
4. **Mixing up the two costs:** `+` for travel time, `max` for bottleneck effort; the rest of the code is identical.
5. **Forgetting unreachable nodes:** keep `float('inf')` in the distance array and test for it where the problem asks, as in the `-1` of 743.
6. **Negative edges:** Dijkstra silently returns wrong answers; check the constraints before choosing it.

# 5. Notes Inbox

<a id="2-notes-inbox"></a>
<a id="3-notes-inbox"></a>
<a id="4-notes-inbox"></a>
<a id="6-notes-inbox"></a>
<a id="7-notes-inbox"></a>

No additional topics yet. Add the next problem, observation, or debugging lesson here.
