# Part VIII — Company Interviews

> Interview questions by company, each with its source and an answer.

The questions come from interview reports on [1point3acres](https://www.1point3acres.com/). Each one lists the level of the candidate who reported it and links to the report by thread id, for example [1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html). The answers are model answers, not the candidates' own.

# 1. Roblox — Machine Learning Engineer

Questions from Roblox Machine Learning Engineer interview reports, in four groups: LeetCode (1.1), ML questions (1.2), system design (1.3) and behavioral questions (1.4).

## 1.1 LeetCode

### 1.1.1 No two adjacent equal items (LeetCode 767) and the cooldown version (LeetCode 621)

**Asked:** PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html)), level not stated ([1125433](https://www.1point3acres.com/bbs/thread-1125433-1-1.html), [1124873](https://www.1point3acres.com/bbs/thread-1124873-1-1.html), [1057160](https://www.1point3acres.com/bbs/thread-1057160-1-1.html), [1050390](https://www.1point3acres.com/bbs/thread-1050390-1-1.html), [1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html))

**Question.**
- Given a list of games, return a reordering in which the same game never appears twice in a row ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html)).
- Counts form ([1125433](https://www.1point3acres.com/bbs/thread-1125433-1-1.html)): the input maps each item to its number of copies. Part 1: decide whether all copies fit in one row with no two equal neighbours, then rewrite that check as short as possible. Part 2: build such a row; after a heap answer, the follow-up was to reach $O(N)$.
- LeetCode 767 (Reorganize String) itself ([1057160](https://www.1point3acres.com/bbs/thread-1057160-1-1.html), [1124873](https://www.1point3acres.com/bbs/thread-1124873-1-1.html), [1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html)), also split into "is it feasible?" and "output any valid string" ([1050390](https://www.1point3acres.com/bbs/thread-1050390-1-1.html)). In a complexity follow-up the interviewer did not accept $O(N)$ and wanted the $k$ distinct items counted, $O(N+k)$ ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html)).
- Cooldown version, LeetCode 621 (Task Scheduler), posed as an "Experience Scheduler" ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html)): two equal tasks need at least $n$ slots between them and idle slots are allowed; return the fewest slots.

**Answer.** Drop zero counts, then let $N$ be the number of items, $k\le N$ the number of distinct items, $c_i$ their counts and $m=\max_i c_i$. The $m$ copies of the top item need $m-1$ separators, so

$$
\text{feasible} \iff N-m \ge m-1 \iff m \le \left\lceil \tfrac{N}{2} \right\rceil = \left\lfloor \tfrac{N+1}{2} \right\rfloor .
$$

The floor form is the one-line check, with no odd/even split. Linear construction: write the top item on slots $0,2,4,\dots$, continue with the other items in any order on the remaining even slots, then wrap to $1,3,5,\dots$. Only a block that wraps from the even to the odd slots can touch itself, and that needs $c \ge \lceil N/2\rceil$ copies. The top item has $m \le \lceil N/2\rceil$, so it never wraps; any later item that wraps starts at the $a$-th even slot with $a \ge m$, so $c \le m \le a < \lceil N/2\rceil$. Cost: $O(N+k)=O(N)$ time because $k\le N$, so both answers to the complexity follow-up describe the same bound; $O(k)$ space plus the output. Only the top item has to go first, which is what removes the sort and the heap; the heap greedy (always place the most frequent item that differs from the previous one) costs $O(N\log k)$. When no arrangement exists, return an empty list (LeetCode 767 itself returns `""`). Edge cases: $N\le 1$; a single distinct item with $c>1$; zero counts in the map; odd $N$ with $m=(N+1)/2$, which forces the top item onto every even slot.

For LeetCode 621, with $t$ the number of tasks tied at count $m$:

$$
T_{\min} = \max\bigl(N,\ (m-1)(n+1)+t\bigr).
$$

The top tasks open $m-1$ frames of $n+1$ slots plus a last frame of $t$ slots; the other tasks fill the gaps, and if they overflow the frames no idle slot is needed, so the answer is $N$. $O(N)$ time and $O(k)$ space; a heap simulation of $(n+1)$-slot cycles also works in $O(T\log k)$ for $T$ output slots. With $n=1$, $T_{\min}=N$ exactly when the LeetCode 767 test passes.

#### Coding: no two adjacent equal items, and LeetCode 621 `[READY]`

```python
from collections import Counter


def can_arrange(counts):
    """counts maps item -> copies. True if every copy fits in one row with no equal neighbours."""
    total = sum(counts.values())
    return max(counts.values(), default=0) <= (total + 1) // 2


def arrange_no_adjacent(items):
    """Reorder items so that no two neighbours are equal; [] if that is impossible."""
    counts = Counter(items)
    if not can_arrange(counts):
        return []
    top = max(counts, key=counts.get, default=None)
    order = [top] + [item for item in counts if item != top]   # only the most frequent must go first
    result = [None] * len(items)
    slot = 0
    for item in order:
        for _ in range(counts[item]):
            result[slot] = item
            slot += 2                       # even slots first, then odd slots
            if slot >= len(items):
                slot = 1
    return result


def least_interval(tasks, n):
    """LeetCode 621: fewest time slots when equal tasks must be at least n slots apart."""
    counts = Counter(tasks)
    top = max(counts.values(), default=0)
    tied = sum(1 for c in counts.values() if c == top)
    return max(len(tasks), (top - 1) * (n + 1) + tied)
```

**Time complexity:** `can_arrange` is $O(k)$ for $k$ distinct items; `arrange_no_adjacent` is $O(N)$: one count, one pass over the slots; `least_interval` is $O(N)$.

**Space complexity:** $O(k)$ for the counts, plus the $O(N)$ output.

Only the most frequent item has to go first; the other items can fill the remaining slots in any order. That is what removes the sort and the heap.

### 1.1.2 Sliding-window rate limiter, then per key

**Asked:** new grad ([1156803](https://www.1point3acres.com/bbs/thread-1156803-1-1.html))

**Question.** Implement a sliding-window rate limiter that decides, request by request, whether a timestamped request is allowed under a cap of $K$ requests per window of length $W$; the thread points to LeetCode 362 (Design Hit Counter) as the same shape and solves it with a deque. Follow-up: apply the limit separately per key, which the thread solves with a dict from key to deque.

**Answer.** Assume non-decreasing timestamps and keep a deque of accepted timestamps. For a request at time $t$, pop from the left while the oldest entry is $\le t-W$, so the window is $(t-W,t]$; accept if fewer than $K$ entries remain, and append $t$ only on accept, so denied requests never count. Per key, keep a dict from key to deque. If there are two caps, for example one per user and one per experience (a Roblox game), evict expired entries in both deques, accept only if both are under their caps, then append to both; recording the request in one deque when the other denies it would spend the user's budget on a rejected request. Each accepted timestamp is pushed and popped at most once per deque, so a request costs amortised $O(1)$ per cap, and memory is $O(K)$ per key seen. A deque is trimmed only when its own key sends a request, so idle keys keep stale entries: sweep periodically and delete the deques that become empty. Clarify whether the window is $(t-W,t]$ or $[t-W,t]$, whether denied requests count, whether timestamps can arrive out of order (then use buckets or a sorted structure), and what $K=0$ means. Alternatives to name: a fixed-window counter is cheaper but admits up to $2K$ requests across a window boundary; a sliding-window counter interpolates two fixed windows; a token bucket allows bursts up to its capacity; LeetCode 362 keeps a ring of $W$ one-second buckets.

#### Coding: sliding-window rate limiter, per key and with two caps `[READY]`

```python
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """At most max_requests per key in any window (t - window, t]. Rejected requests are not stored.
    Timestamps arrive in non-decreasing order."""

    def __init__(self, max_requests, window):
        self.max_requests, self.window = max_requests, window
        self.accepted = defaultdict(deque)        # key -> timestamps of accepted requests

    def allow(self, key, timestamp):
        recent = self.accepted[key]
        while recent and recent[0] <= timestamp - self.window:
            recent.popleft()                      # fell out of the window
        if len(recent) < self.max_requests:
            recent.append(timestamp)
            return True
        return False


class MultiCapRateLimiter:
    """Follow-up: one cap per dimension, e.g. limits = {"user": 3, "experience": 5}.
    allow(t, user="u1", experience="g7") accepts only if every dimension is under its cap."""

    def __init__(self, limits, window):
        self.limits, self.window = limits, window
        self.accepted = {dim: defaultdict(deque) for dim in limits}

    def allow(self, timestamp, **keys):
        queues = [(self.accepted[dim][keys[dim]], cap) for dim, cap in self.limits.items()]
        for recent, _ in queues:
            while recent and recent[0] <= timestamp - self.window:
                recent.popleft()
        if any(len(recent) >= cap for recent, cap in queues):
            return False                          # record nothing when any cap denies
        for recent, _ in queues:
            recent.append(timestamp)
        return True
```

**Time complexity:** amortised $O(1)$ per request and dimension: every accepted timestamp is appended once and popped at most once.

**Space complexity:** $O(K)$ timestamps per key and dimension, for a cap of $K$, for every key seen so far. A deque is trimmed only when its own key sends a request, so an idle key keeps up to $K$ stale timestamps and is never removed. To forget idle keys, sweep periodically (or keep keys in last-seen order), evict expired timestamps and delete the deques that become empty.

The two-cap version checks every dimension before it records anything, so a request denied by the experience cap does not use up the user's budget.

### 1.1.3 Valid Parentheses, extended step by step

**Asked:** intern ([1063311](https://www.1point3acres.com/bbs/thread-1063311-1-1.html))

**Question.** A medium-level extension of Valid Parentheses (LeetCode 20), apparently written by the interviewer, asked one step at a time. The individual steps were not reported.

**Answer.** Work through the standard ladder; every rung except the last is one linear pass.
- Several bracket types (LeetCode 20): push openers; on a closer, pop and compare with its opener, failing if the stack is empty; the string is valid if and only if the stack is empty at the end. $O(n)$ time and space.
- Fewest insertions to make it valid (LeetCode 921): count unmatched `(` as `open`; a `)` with `open` at 0 costs one insertion, otherwise it decrements `open`; the answer is the insertions plus the final `open`. $O(1)$ space.
- Fewest removals, returning the string (LeetCode 1249): collect the indices of unmatched `(` on a stack and of unmatched `)` as they occur, then rebuild the string skipping both. $O(n)$.
- Longest valid substring (LeetCode 32): a stack of indices seeded with $-1$; on `(` push $i$; on `)` pop, then push $i$ if the stack is empty, else the candidate length is $i-\text{top}$. Two counter passes, left to right and right to left, do it in $O(1)$ space.
- `*` may be `(`, `)` or empty (LeetCode 678): track the range $[lo,hi]$ of possible open counts; `(` adds 1 to both ends, `)` subtracts 1 from both, `*` moves $lo$ down and $hi$ up by 1; fail if $hi<0$, clamp $lo$ at 0, and accept if $lo=0$ at the end. $O(1)$ space.
- Generate every valid string with $n$ pairs (LeetCode 22): backtrack under $\text{close}\le\text{open}\le n$; the output has $C_n=\frac{1}{n+1}\binom{2n}{n}$ strings.

Edge cases: the empty string is valid; ask whether characters other than brackets are ignored or rejected.

### 1.1.4 Remove prefixes from an array

**Asked:** research intern ([1047347](https://www.1point3acres.com/bbs/thread-1047347-1-1.html))

**Question.** Remove prefixes from an array: give a brute force, analyse its complexity, then optimise, with a trie as the hinted direction. The input and output format was not given; the reading that fits the trie hint is a list of strings from which every string that is a prefix of another string is removed.

**Answer.** Let $n$ be the number of words, $L$ their total length and $\ell$ the longest length. The brute force runs a prefix check on every ordered pair, $O(n^2\ell)$; bucketing by first letter does not change the worst case. Trie: insert every word and count the words ending at each node; a word is a proper prefix of another word exactly when its end node has a child (or when its end count is above 1, if a duplicate counts as "another string"); keep the remaining words in input order. $O(L)$ time and $O(L)$ nodes. If the task is the other way round (drop every word that has another word as a prefix), drop a word when its path passes through another word's end node, in the same $O(L)$. Sort: after a lexicographic sort, the strings that start with $s$ form one contiguous block beginning with $s$, so $s$ is a proper prefix of some word exactly when the next string that differs from $s$ starts with $s$; $O(\ell\, n\log n)$. Clarify duplicates, the empty string (a prefix of every word), case sensitivity, and whether the task is instead to strip the longest common prefix from every string (LeetCode 14, $O(L)$).

### 1.1.5 Most frequent call path from function-call logs

**Asked:** senior ([1186156](https://www.1point3acres.com/bbs/thread-1186156-1-1.html)), staff ([1136387](https://www.1point3acres.com/bbs/thread-1136387-1-1.html))

**Question.** The input is a list of trace lines: `-> f` means `f` is called and `<- f` means `f` returns. Parse the lines and return the call stack, written from the root as `main->eventClick->eventClickAction`, that occurs most often. On `["-> main", "-> eventClick", "-> eventClickAction", "<- eventClickAction", "-> eventClickAction", "<- eventClickAction", "<- eventClick", "<- main"]` the answer is `main->eventClick->eventClickAction` with frequency 2, formed once by lines 1-3 and again by lines 1, 2 and 5. Follow-ups: also return the count ([1136387](https://www.1point3acres.com/bbs/thread-1136387-1-1.html)); the logs come from several threads, interleaved ([1186156](https://www.1point3acres.com/bbs/thread-1186156-1-1.html)).

**Answer.** Replay the log with a stack. On a call, push the name and add 1 to the counter of the current root-to-top path; on a return, check that the name matches the top and pop, and raise an error for a return that does not match or arrives on an empty stack. Keying the counter by the joined stack costs $O(D)$ per line, so $O(E\cdot D)$ for $E$ lines and maximum depth $D$. To get $O(E)$, give each distinct path a node id in a call tree: map `(parent_id, name)` to the child's id with one hash lookup, keep node ids on the stack, so a call is one lookup plus one increment, and rebuild only the winning path through parent pointers in $O(D)$; space is $O(P)$ for $P$ distinct paths. Track the best path as you go, ranked by (count, depth) with a strict comparison: counts only grow, so the running best is the final best, ties go to the deeper path, and among equal ranks the path that reached the count first wins (state this tie rule, since none was given). For interleaved threads, keep a dict from thread id to that thread's stack and share one counter, so equal paths merge across threads (confirm that this is wanted, and ask how a line names its thread); the complexity is unchanged. Recursion, a name repeating inside a path, needs no special case. Counting at the call already counts calls still open at the end; ask whether a stack that is a prefix of a longer one counts, since counting only leaf stacks gives different answers in general.

#### Coding: most frequent call path, single and multi-threaded `[READY]`

```python
def most_frequent_call_path(traces):
    """traces: '-> f' is a call, '<- f' a return. Every call counts the stack root -> f once.
    Returns (path, count); ties go to the deeper path, then to the path that reached the count first."""
    stack, counts = [], {}
    best, best_key = None, (0, 0)
    for line in traces:
        arrow, name = line.split()
        if arrow == '->':
            stack.append(name)
            path = '->'.join(stack)
            counts[path] = counts.get(path, 0) + 1
            key = (counts[path], len(stack))
            if key > best_key:              # strict: an equal key keeps the earlier path
                best, best_key = path, key
        elif arrow == '<-' and stack and stack[-1] == name:
            stack.pop()
        else:
            raise ValueError(f'unmatched trace line: {line!r}')
    return best, best_key[0]


def most_frequent_call_path_threads(logs):
    """Follow-up: lines are '<thread> -> f' or '<thread> <- f', interleaved across threads.
    One stack per thread; paths are counted across all threads."""
    stacks, counts = {}, {}
    best, best_key = None, (0, 0)
    for line in logs:
        thread, arrow, name = line.split()
        stack = stacks.setdefault(thread, [])
        if arrow == '->':
            stack.append(name)
            path = '->'.join(stack)
            counts[path] = counts.get(path, 0) + 1
            key = (counts[path], len(stack))
            if key > best_key:
                best, best_key = path, key
        elif arrow == '<-' and stack and stack[-1] == name:
            stack.pop()
        else:
            raise ValueError(f'unmatched trace line: {line!r}')
    return best, best_key[0]
```

**Time complexity:** $O(E \cdot D)$ for $E$ trace lines and stack depth $D$, because each call joins the current stack into a key. Keying the counter by call-tree node ids, as in the sketch above, makes it $O(E)$.

**Space complexity:** $O(P \cdot D)$ for $P$ distinct paths stored as strings, plus one stack per thread.

The input format of the multi-threaded logs was not reported; this version assumes each line starts with a thread id, such as `t1 -> main`. Ask before you parse.

### 1.1.6 Match-3 grid (Candy Crush)

**Asked:** senior ([1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html)), level not stated ([1150441](https://www.1point3acres.com/bbs/thread-1150441-1-1.html), [886138](https://www.1point3acres.com/bbs/thread-886138-1-1.html))

**Question.** The input is a grid of integers, and a horizontal or vertical run of 3 or more equal values is a match. Part 1: output every match as (value, length), listing all matches of one direction before the other (the poster recalls vertical first but is unsure), scanning left to right and top to bottom; a reader saw a version that outputs (position, length) instead. Part 2: set every matched cell to 0 and let the cells above fall into the gaps ([1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html)). LeetCode 723 (Candy Crush) as written: crush, drop, and repeat until the board is stable ([1150441](https://www.1point3acres.com/bbs/thread-1150441-1-1.html)). The third thread names the match-3 problem without details ([886138](https://www.1point3acres.com/bbs/thread-886138-1-1.html)).

**Answer.** Part 1: scan each column, then each row, for maximal runs with two pointers; a run $[j,k)$ of equal non-zero values with $k-j\ge 3$ is one match. Report each as (value, length, start cell, direction), which covers both output formats, with vertical runs first. $O(RC)$ time. A run of 5 is one match of length 5, not three overlapping triples, and an L or T shape gives two matches that share a cell. Part 2: mark every matched cell first and clear afterwards, because clearing during the scan breaks crossing runs; then, per column, move a write pointer up from the bottom, copy non-zero cells down to it and fill the top with 0, in $O(RC)$. LeetCode 723: repeat detect, clear and drop while any match exists. Every productive pass turns at least 3 non-zero cells into 0 and gravity creates none, so there are at most $\lfloor RC/3\rfloor+1$ passes and $O\bigl((RC)^2\bigr)$ time in the worst case. 0 means empty and never matches; on a rectangular board keep the row and column bounds straight.

#### Coding: match-3 runs, clear and drop, Candy Crush `[READY]`

```python
def find_matches(board):
    """Runs of 3+ equal non-zero values. Vertical runs first (columns left to right, top to bottom),
    then horizontal runs (rows top to bottom, left to right). Each run: (value, length, (row, col), direction)."""
    rows, cols = len(board), len(board[0]) if board else 0
    runs = []
    for c in range(cols):
        r = 0
        while r < rows:
            end = r
            while end + 1 < rows and board[end + 1][c] == board[r][c]:
                end += 1
            if board[r][c] and end - r + 1 >= 3:
                runs.append((board[r][c], end - r + 1, (r, c), 'V'))
            r = end + 1
    for r in range(rows):
        c = 0
        while c < cols:
            end = c
            while end + 1 < cols and board[r][end + 1] == board[r][c]:
                end += 1
            if board[r][c] and end - c + 1 >= 3:
                runs.append((board[r][c], end - c + 1, (r, c), 'H'))
            c = end + 1
    return runs


def crush_once(board):
    """Phase 2: set every matched cell to 0, then let the non-zero cells fall. Returns whether anything matched."""
    runs = find_matches(board)
    for _, length, (r, c), direction in runs:
        for k in range(length):
            if direction == 'V':
                board[r + k][c] = 0
            else:
                board[r][c + k] = 0
    for c in range(len(board[0]) if board else 0):
        column = [board[r][c] for r in range(len(board)) if board[r][c]]
        column = [0] * (len(board) - len(column)) + column
        for r in range(len(board)):
            board[r][c] = column[r]
    return bool(runs)


def candy_crush(board):
    """LeetCode 723: crush and drop until the board is stable."""
    while crush_once(board):
        pass
    return board
```

**Time complexity:** $O(RC)$ for `find_matches` and for `crush_once`; `candy_crush` calls `crush_once` at most $\lfloor RC/3\rfloor + 1$ times, because every call that matches turns at least 3 non-zero cells into 0, gravity never creates a non-zero cell, and the last call finds nothing; so $O((RC)^2)$ in the worst case.

**Space complexity:** `find_matches` uses $O(1)$ besides its output list, which can hold at most $2RC/3$ runs. `crush_once` adds an $O(R)$ column buffer for the drop and builds that run list internally, so its extra space is $O(RC)$ in the worst case. The board is changed in place.

Vertical runs are listed before horizontal ones because that is the order the report remembers; the poster was not sure, so confirm it. Each run is reported as value, length, start cell and direction, which covers both reported output formats.

### 1.1.7 Sliding windows that contain a target

**Asked:** senior ([1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html)), level not stated ([1173576](https://www.1point3acres.com/bbs/thread-1173576-1-1.html))

**Question.** Given a list of numbers, a fixed window size $w$ and a target: part 1, return every window of size $w$ that contains the target; part 2, return the first window in which the target appears the most times ([1173576](https://www.1point3acres.com/bbs/thread-1173576-1-1.html)). The other thread gives part 1 only, as "given windows and a target number, return the windows that contain it" ([1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html)).

**Answer.** The windows start at $i=0,\dots,n-w$. Slide a count of the target:

$$
\mathrm{cnt}_{i+1} = \mathrm{cnt}_i - [x_i = t] + [x_{i+w} = t].
$$

Part 1 returns every window with $\mathrm{cnt}_i>0$. Part 2 takes the argmax with a strict comparison, so the earliest window wins ties, and returns nothing (`None`) when no window contains the target. Sliding is $O(n)$ time and $O(1)$ extra space; returning copies of the windows costs $O(w)$ each, so part 1 is $O(nw)$ in the worst case and part 2 is $O(n+w)$. Return start indices instead if positions are enough. Edge cases: require $w\ge 1$; $w>n$ or an empty list gives no window, $w=n$ gives exactly one, and the last start is $n-w$. If the windows arrive as a given list, part 1 is a plain scan of each window.

#### Coding: windows that contain a target `[READY]`

```python
def windows_with_target(nums, k, target):
    """Every length-k window (in order) that contains target. One pass keeps the target count of the window."""
    windows, count = [], 0
    for i, x in enumerate(nums):
        count += x == target
        if i >= k:
            count -= nums[i - k] == target       # the element that just left the window
        if i >= k - 1 and count:
            windows.append(nums[i - k + 1:i + 1])
    return windows


def first_window_with_most_target(nums, k, target):
    """The earliest length-k window with the most copies of target; None if no window contains it."""
    best_start, best, count = None, 0, 0
    for i, x in enumerate(nums):
        count += x == target
        if i >= k:
            count -= nums[i - k] == target
        if i >= k - 1 and count > best:          # strict: ties keep the earlier window
            best, best_start = count, i - k + 1
    return None if best_start is None else nums[best_start:best_start + k]
```

**Time complexity:** $O(n)$ to slide the count, plus $O(k)$ to copy each returned window ($k$ is the window size, $w$ in the sketch). Part 1 is therefore $O(nk)$ in the worst case, when almost every window contains the target; part 2 copies one window, so $O(n+k)$.

**Space complexity:** $O(1)$ besides the returned windows.

Return start indices instead of copies if the interviewer only needs positions.

### 1.1.8 Find median from a data stream (LeetCode 295)

**Asked:** IC6 (principal) ([995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html)), level not stated ([1014514](https://www.1point3acres.com/bbs/thread-1014514-1-1.html))

**Question.** LeetCode 295: support adding numbers from a stream and returning the median of all numbers so far. Follow-ups: multiple data streams, and reservoir sampling ([1014514](https://www.1point3acres.com/bbs/thread-1014514-1-1.html)).

**Answer.** Use two heaps: a max-heap `lo` with the smaller half and a min-heap `hi` with the larger half, kept so that $\max(lo)\le\min(hi)$ and $|lo|-|hi|\in\{0,1\}$. To add, push into `lo`, move the top of `lo` to `hi`, and move the top of `hi` back if `hi` became larger. The median is the top of `lo` for an odd count and the mean of both tops for an even count; querying an empty stream is an error. $O(\log n)$ per add, $O(1)$ per query, $O(n)$ space. Python's `heapq` is a min-heap, so store negated values in `lo`, and return a float in the even case.

Multiple streams: for one global median, feed every stream into the same structure; for a median per stream, keep a dict from stream id to its own heap pair; for integer values in a bounded range of size $V$, a Fenwick tree over value counts finds the $k$-th smallest in $O(\log V)$ and merges streams by adding counts in $O(V)$; a median over the last $w$ items is LeetCode 480, two heaps with lazy deletion.

Reservoir sampling (Algorithm R) keeps a uniform sample of $k$ items in $O(k)$ memory: keep the first $k$; for item $i>k$, draw $j$ uniformly from $\{1,\dots,i\}$ and replace slot $j$ if $j\le k$. Item $i>k$ is in the final sample with probability

$$
\frac{k}{i}\prod_{j=i+1}^{n}\Bigl(1-\frac{1}{j}\Bigr) = \frac{k}{i}\cdot\frac{i}{n} = \frac{k}{n},
$$

and each of the first $k$ items with $\prod_{j=k+1}^{n}(1-1/j)=k/n$. The sample median then estimates the stream median. For a guaranteed rank error of $\varepsilon n$, use a GK sketch (deterministic) or a KLL sketch (with high probability); t-digest is accurate in practice but has no worst-case bound.

#### Coding: median of a data stream with two heaps `[READY]`

```python
import heapq


class MedianFinder:
    """LeetCode 295. low holds the smaller half as a max-heap (negated), high the larger half."""

    def __init__(self):
        self.low, self.high = [], []

    def add_num(self, x):
        heapq.heappush(self.low, -x)
        heapq.heappush(self.high, -heapq.heappop(self.low))   # the largest of low moves to high
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def find_median(self):
        if not self.low:
            raise ValueError('no numbers yet')
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2
```

**Time complexity:** $O(\log n)$ per `add_num`, $O(1)$ per `find_median`.

**Space complexity:** $O(n)$ for the two heaps.

### 1.1.9 Word search on a board (LeetCode 79 / 212)

**Asked:** level not stated ([919123](https://www.1point3acres.com/bbs/thread-919123-1-1.html), [832731](https://www.1point3acres.com/bbs/thread-832731-1-1.html))

**Question.** Given `char[][] board` and `string[] words`: (1) return the words that can be formed on the board; (2) also return a path for each such word; (3) different words may not reuse a board cell, so return the largest set of words that fit together ([919123](https://www.1point3acres.com/bbs/thread-919123-1-1.html)). A second ladder ([832731](https://www.1point3acres.com/bbs/thread-832731-1-1.html)): (1) decide whether a given character appears in any of the four directions from a coordinate; (2) LeetCode 79, decide whether one word is in the grid; (3) decide which of many words are in the grid, approach only (a trie of the words).

**Answer.** LeetCode 79: DFS from every cell along 4-neighbour steps, marking a cell used on the way in and freeing it on the way out; $O(RC\cdot 4\cdot 3^{L-1})$ for a word of length $L$. Many words (LeetCode 212): build a trie of the words and run one DFS per start cell that follows trie edges, so shared prefixes are walked once; record each word the first time its end is reached, and prune trie leaves that become empty. Building the trie is $O(L_{\text{total}})$ and the search is $O(RC\cdot 4\cdot 3^{L-1})$ for the longest word length $L$. Paths come for free: the DFS stack is the path, copied when a word ends.

Disjoint cells need every candidate path of each word, not the one path per word kept above. On the one-row board `a b a b` with words `ab` and `ba`, a row-major search keeps `ab` on $(0,0),(0,1)$ and a path for `ba` that starts at $(0,1)$, so the two overlap, yet `ba` also fits on $(0,3),(0,2)$. So collect all candidate paths per word, then backtrack: for each word either skip it or take one of its paths that avoids the cells already used, pruning when the current size plus the words left cannot beat the best. This is set packing and exponential in general, so it suits small boards. For the single-character question, ask whether "four directions" means the four neighbours, $O(1)$, or the four straight rays, $O(R+C)$. Clarify 4 or 8 directions, whether a word may reuse a cell, and case sensitivity.

#### Coding: word search on a board with a trie, with paths `[READY]`

```python
def find_words(board, words):
    """LeetCode 212 with the paths follow-up: map each word that can be traced on the board
    (4-neighbour steps, no cell twice within a word) to one path of (row, col) cells."""
    trie = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node['$word'] = word                      # multi-character key: never equals a board letter
    rows, cols = len(board), len(board[0]) if board else 0
    found, path, used = {}, [], set()

    def dfs(r, c, node):
        child = node.get(board[r][c])
        if child is None:
            return
        path.append((r, c))
        used.add((r, c))
        word = child.get('$word')
        if word is not None and word not in found:
            found[word] = path.copy()
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in used:
                dfs(rr, cc, child)
        used.discard((r, c))
        path.pop()

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie)
    return found
```

**Time complexity:** building the trie is $O(L_{\text{total}})$; the search is $O(RC \cdot 4 \cdot 3^{L-1})$ in the worst case for words of length up to $L$, and shared prefixes are walked once.

**Space complexity:** $O(L_{\text{total}})$ for the trie, $O(L)$ for the path and the used set.

Recording the path costs nothing extra: it is the DFS stack. The third part, the largest set of words with disjoint cells, needs every candidate path of each word, not only the first one this drill keeps. On the board `[['a','b','a','b']]` with words `ab` and `ba`, the stored paths share cell (0, 1), yet `ab` on (0, 0)-(0, 1) and `ba` on (0, 3)-(0, 2) are disjoint. Collect a list of paths per word, then backtrack over those lists.

### 1.1.10 Build order by levels

**Asked:** level not stated ([1181604](https://www.1point3acres.com/bbs/thread-1181604-1-1.html))

**Question.** You are given dependency declarations. Each is a list of strings whose first element is a component and whose remaining elements are the components it depends on, which must be built before it. Return a valid build order. At each step, every component whose dependencies are all built becomes buildable at the same time; within one such level, output components in the order their declarations appear in the input. Return `["Error"]` if a dependency is never declared as a first element, or if the dependencies form a cycle. Examples:
- `[["head"], ["torso"], ["leg"]]` gives `["head", "torso", "leg"]`.
- `[["head", "leg"], ["leg", "head"]]` gives `["Error"]` (a cycle).
- `[["head"], ["leg", "torso"]]` gives `["Error"]` (`torso` is never declared).
- `[["hair", "head"], ["torso"], ["leg", "torso"], ["head"]]` gives `["torso", "head", "hair", "leg"]`.

**Answer.** Kahn's algorithm, one level at a time. Map each component to its declaration index; if any dependency is undeclared, return `["Error"]`. Set each component's indegree to its number of distinct dependencies. Level 0 is the components with indegree 0, sorted by declaration index; to form the next level, decrement the indegree of every dependent of the current level, collect those that reach 0 and sort them by declaration index. If fewer than $V$ components come out, the rest sit on or behind a cycle: return `["Error"]`. $O(V+E+V\log V)$ time and $O(V+E)$ space. A single priority queue keyed by declaration index is wrong: on the fourth example it outputs `torso, leg, head, hair`, because `leg` (level 1) overtakes `head` (level 0). A self-dependency is a cycle; deduplicate repeated dependencies; ask what a component declared twice means.

#### Coding: build order by levels `[READY]`

```python
from collections import defaultdict


def build_order(declarations):
    """Each declaration is [component, *dependencies]. Build level by level; inside a level keep the
    declaration order. ["Error"] for a dependency that is never declared or for a cycle."""
    position = {decl[0]: i for i, decl in enumerate(declarations)}
    requires = {decl[0]: set(decl[1:]) for decl in declarations}
    if any(dep not in position for deps in requires.values() for dep in deps):
        return ["Error"]
    waiting = {c: len(deps) for c, deps in requires.items()}
    users = defaultdict(list)
    for component, deps in requires.items():
        for dep in deps:
            users[dep].append(component)
    level = sorted((c for c in requires if waiting[c] == 0), key=position.get)
    order = []
    while level:
        order.extend(level)
        ready = []
        for built in level:
            for user in users[built]:
                waiting[user] -= 1
                if waiting[user] == 0:
                    ready.append(user)
        level = sorted(ready, key=position.get)
    return order if len(order) == len(requires) else ["Error"]
```

**Time complexity:** $O(V + E + V \log V)$: Kahn's algorithm plus a sort of each level by declaration position.

**Space complexity:** $O(V + E)$.

### 1.1.11 Other problems

#### Closest BST value, harder variant (LeetCode 270)

**Asked:** level not stated ([1158688](https://www.1point3acres.com/bbs/thread-1158688-1-1.html))

**Question.** A harder variant of LeetCode 270 (given a binary search tree and a target, return the value closest to the target) with extra constraints that the poster did not recall.

**Answer.** Base: walk down from the root, update the best value whenever $|v-t|$ improves (ties to the smaller value), and go left if $t<v$, else right; $O(h)$ time and $O(1)$ space. Have the usual harder forms ready: the $k$ closest values (LeetCode 272), with a predecessor stack and a successor stack merged like two sorted lists in $O(h+k)$, or an in-order pass with a size-$k$ deque in $O(n)$; the floor or ceiling of $t$ (largest value $\le t$, smallest value $\ge t$) with the same walk; and a tree with no BST guarantee, which needs a full traversal in $O(n)$. Edge cases: an empty tree, a target outside the value range, duplicates, and overflow of $|v-t|$ with fixed-width integers.

#### Single-Threaded CPU (LeetCode 1834)

**Asked:** senior ([1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html))

**Question.** LeetCode 1834: `tasks[i] = [enqueueTime, processingTime]`. Whenever a single CPU is idle, it starts the available task with the shortest processing time, ties going to the smaller index, and runs it to completion; if no task is available, it waits for the next one. Return the order in which the tasks are processed.

**Answer.** Sort the indices by enqueue time and keep a min-heap of (processing time, index). Loop with a clock $T$: push every task with enqueue time $\le T$; if the heap is empty, jump $T$ to the next enqueue time; otherwise pop, append the index and add its processing time to $T$. Each task is pushed and popped once: $O(n\log n)$ time and $O(n)$ space. Use 64-bit time, since the clock can reach $10^5\cdot 10^9$, and push all tasks that share an enqueue time before choosing.

#### Key-value store with transactions

**Asked:** level not stated ([927016](https://www.1point3acres.com/bbs/thread-927016-1-1.html))

**Question.** Implement an in-memory key-value store that supports transactions, as plain coding: no locks, write-ahead log or isolation levels.

**Answer.** Support the usual API, `get`, `set`, `delete`, `begin`, `commit` and `rollback`, with nested transactions. Keep the current data in one dict and a stack of undo logs, one per open transaction; the first time a transaction changes a key, it records the key's previous value or "absent". `get` and `set` stay $O(1)$. `rollback` replays the top log in $O(\text{changes})$ and pops it. Committing a nested transaction merges its log into the parent's, keeping the parent's older entry for keys both touched; a top-level commit just drops the log. The alternative is a stack of overlay dicts with tombstones for deletes, where `get` searches from the top layer down in $O(\text{depth})$. If the API also counts keys per value, keep a value-to-count map and update it on every write and every undo. Edge cases: `commit` or `rollback` with no open transaction, whether `commit` closes only the innermost transaction or all of them, deleting a missing key, and a key set to its current value.

#### Initialize a connect-3 board

**Asked:** level not stated ([927016](https://www.1point3acres.com/bbs/thread-927016-1-1.html))

**Question.** Initialize a connect-3 board. Nothing more of the statement was reported, except that filling it with two plain nested loops can produce invalid boards, which a DFS fixed. The common reading: fill an $R\times C$ board with random colours so that no row or column already holds three equal cells in a row.

**Answer.** Fill the cells in row-major order. At $(r,c)$, forbid the colour of the two cells to the left if they are equal and the colour of the two cells above if they are equal, then pick at random from the remaining colours. With 3 or more colours at most 2 are forbidden, so the fill never gets stuck: $O(RC)$. With 2 colours both can be forbidden, and that is when a DFS with backtracking is needed. If the board must also allow at least one move that makes a match, test that after filling and regenerate.

#### Connect 4 winner

**Asked:** level not stated ([874270](https://www.1point3acres.com/bbs/thread-874270-1-1.html))

**Question.** Two players take turns dropping pieces into a Connect 4 board; determine which player is first to connect four.

**Answer.** Keep the board and a height per column, so a drop lands in $O(1)$. After each drop at $(r,c)$, for each of the 4 lines through it (horizontal, vertical and the two diagonals), count same-colour cells forward and backward; $1+\text{forward}+\text{backward}\ge 4$ wins. That is $O(1)$ per move, $O(k)$ for connect-$k$, with $O(RC)$ space; given a full move list, return the first winner and the move that won. Handle a move into a full column, a draw when the board fills, and moves after the game is over.

#### Mahjong hand check, then with sequences

**Asked:** level not stated ([886138](https://www.1point3acres.com/bbs/thread-886138-1-1.html))

**Question.** Two parts: check a mahjong hand, then the same check when sequences are allowed. The thread gives no more; the usual form is a hand of numbered tiles that is complete when it splits into exactly one pair plus any number of triples of equal tiles, and part 2 also accepts three consecutive numbers as a group.

**Answer.** Part 1: count the tiles; the hand is complete exactly when one tile value has a count $\equiv 2 \pmod 3$ and every other count is $\equiv 0 \pmod 3$ (a count of 5 is a pair plus a triple); $O(n)$. Part 2: the hand size must be $\equiv 2 \pmod 3$. Try each value with count $\ge 2$ as the pair and remove it, then scan values in increasing order: the smallest remaining value $x$, with count $c$, can only sit in triples or start sequences, so form $c \bmod 3$ sequences $x,x+1,x+2$ (fail if $x+1$ or $x+2$ runs short) and $\lfloor c/3\rfloor$ triples. Three equal sequences use the same tiles as three triples, so this greedy is exact. $O(V)$ per pair choice for $V$ tile values, $O(n+V^2)$ in total. Ask about suits (sequences never cross suits) and the hand size.

#### Find peak

**Asked:** level not stated ([889847](https://www.1point3acres.com/bbs/thread-889847-1-1.html))

**Question.** Find a peak element; the thread does not say whether in an array (LeetCode 162) or in a matrix (LeetCode 1901).

**Answer.** Array: treat out-of-range neighbours as $-\infty$ and binary search; if $a_{mid}<a_{mid+1}$ a peak lies to the right, otherwise at $mid$ or to its left. $O(\log n)$, assuming adjacent values differ. Matrix: binary search on columns; take the maximum of the middle column in $O(R)$ and move toward a strictly larger horizontal neighbour, or stop because that cell is a peak; $O(R\log C)$. With equal neighbours the array search still returns an element at least as large as both neighbours, but a strict peak may not exist at all (a constant array has none), so agree on the definition first.

#### Design git (object-oriented)

**Asked:** level not stated ([768938](https://www.1point3acres.com/bbs/thread-768938-1-1.html))

**Question.** An object-oriented design problem best described as "design git"; no further specification was reported.

**Answer.** Store objects by content hash: a blob (file content), a tree (names to blob or tree hashes) and a commit (tree hash, parent hashes, author, message, timestamp). Refs map branch names to commit hashes, and `HEAD` names a branch or, when detached, a commit. Operations: `add` stages a file in an index; `commit` builds a tree from the index, writes a commit whose parent is `HEAD` and advances the branch; `branch` creates a ref; `checkout` moves `HEAD` and rewrites the working files; `log` walks parent pointers; `merge` finds a merge base, a common ancestor in the commit DAG that no other common ancestor descends from (mark the ancestors of one commit by BFS, then BFS from the other; criss-cross histories can have several), then merges each file three ways and reports conflicts. Objects are immutable and deduplicated by their hash. Keep the classes small (`Repository`, `ObjectStore`, `Blob`, `Tree`, `Commit`, `Ref`) and agree on scope (single user, local only, text files) before writing code.

#### Analyze user website visit pattern, variant (LeetCode 1152)

**Asked:** level not stated ([830336](https://www.1point3acres.com/bbs/thread-830336-1-1.html))

**Question.** A variant of LeetCode 1152 with a very long description and many corner cases, plus one more corner case added after the code was written. The base problem: given parallel arrays `username`, `timestamp` and `website`, find the pattern of 3 websites, visited in that order by time but not necessarily consecutively, that the most users share; ties go to the lexicographically smallest pattern.

**Answer.** Group the visits by user and sort each user's visits by timestamp. For each user, generate the 3-patterns over index triples $i<j<k$ into a set, so a user counts once per pattern, and add the set into a global counter. Return the pattern with the highest count, ties to the lexicographically smallest. Cost: $O(N\log N)$ to sort plus $O\bigl(\sum_u \binom{m_u}{3}\bigr)$ for $m_u$ visits by user $u$. Corner cases: users with fewer than 3 visits, a site repeated inside a pattern (allowed), equal timestamps (ask how to order them), and the tie rule.

#### Implement two-phase commit

**Asked:** level not stated ([830336](https://www.1point3acres.com/bbs/thread-830336-1-1.html))

**Question.** Implement the two-phase commit protocol from a long written specification.

**Answer.** One coordinator and $n$ participants. Phase 1 (prepare): the coordinator sends PREPARE; each participant durably logs its vote and answers YES, after which it is prepared and holds its locks, or NO. Phase 2 (decision): if every vote is YES, the coordinator logs COMMIT and sends COMMIT; on any NO, or a timeout while waiting for votes, it logs and sends ABORT. Participants apply the decision, release their locks and acknowledge, and the coordinator logs END after every acknowledgement. The coordinator moves through INIT, WAIT, then COMMIT or ABORT; a participant through INIT, READY, then COMMIT or ABORT. Recovery reads the log: a coordinator with no decision record aborts, and a participant in READY cannot decide alone and must learn the outcome from the coordinator or from a participant that knows it, which is why 2PC blocks if the coordinator fails after collecting the votes. Make messages idempotent and resend the decision until it is acknowledged. A commit costs $4n$ messages. Paxos Commit removes the blocking; three-phase commit removes it only on a synchronous network without partitions.

## 1.2 ML questions

### 1.2.1 Transformers and LLMs

#### How the Transformer works

**Asked:** PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html)), IC6 (principal) ([995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html)), level not stated ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html), [1125433](https://www.1point3acres.com/bbs/thread-1125433-1-1.html), [1124873](https://www.1point3acres.com/bbs/thread-1124873-1-1.html), [1050390](https://www.1point3acres.com/bbs/thread-1050390-1-1.html))

**Question.** Explain how the Transformer works: the overall architecture and the role of each layer, the attention computation, and what the query, key and value (Q, K, V) are.

**Answer.** Each token embedding $x \in \mathbb{R}^{d}$ is projected three ways: a query $q = xW_Q$ (what this token is looking for), a key $k = xW_K$ (what it offers to be matched on) and a value $v = xW_V$ (the content it passes on). Stacking the tokens into matrices gives

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V
$$

Row $i$ of the softmax is a distribution over positions $j$ that says how much token $i$ reads from token $j$, and the output is the matching weighted sum of values. $M$ is $0$ or $-\infty$ per entry (causal or padding mask). If the components of $q$ and $k$ are independent with zero mean and unit variance, $q^{\top}k$ has variance $d_k$, so dividing by $\sqrt{d_k}$ keeps the logits at unit variance and the softmax out of saturation. Multi-head attention runs $h$ heads of width $d_k = d/h$, each with its own projections, concatenates them and mixes them with $W_O$, so different heads can track different relations at about the same cost as one full-width head.

A block wraps attention and a position-wise feed-forward network (FFN) in residual connections and layer normalization. In the pre-LN form used by modern LLMs:

$$
h = x + \mathrm{MHA}\bigl(\mathrm{LN}(x)\bigr), \qquad y = h + \mathrm{FFN}\bigl(\mathrm{LN}(h)\bigr), \qquad \mathrm{FFN}(h) = \phi(hW_1)\,W_2
$$

The original paper used post-LN, $\mathrm{LN}(x + \mathrm{sublayer}(x))$, with $\phi = \mathrm{ReLU}$ and $d_{ff} = 4d$; current models use GELU or SwiGLU and often RMSNorm. Attention is permutation-invariant, so position must be injected (see positional encodings below). Encoder blocks use bidirectional self-attention. Decoder blocks use causally masked self-attention, and in an encoder-decoder model they add cross-attention whose queries come from the decoder and whose keys and values come from the encoder output. A final linear layer and softmax over the vocabulary give next-token probabilities. Per layer, the attention scores cost $O(n^2 d)$ and the projections and FFN cost $O(nd^2)$.

#### Transformers compared with RNNs and LSTMs

**Asked:** level not stated ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html))

**Question.** What are the advantages of the Transformer over RNNs and LSTMs?

**Answer.**

RNNs process tokens sequentially, maintaining a hidden state that summarizes previous inputs. However, they struggle with long-range dependencies due to vanishing gradients and cannot parallelize computation across time steps.

LSTMs improve RNNs by introducing a cell state and input, forget, and output gates. These mechanisms help preserve information and improve gradient flow over longer sequences, but computation remains sequential.

Transformers replace recurrence with self-attention, enabling parallel processing during training and direct interactions between tokens. This significantly improves scalability and long-range dependency modeling. However, standard self-attention has quadratic complexity in sequence length, and autoregressive inference requires a growing KV cache.

In more detail:

- Parallel training: all positions are computed at once with matrix multiplications, while an RNN needs $n$ sequential steps. This suits GPUs and is what lets Transformers scale to large models and data.
- Short paths: any two tokens interact in one attention step (path length $O(1)$ instead of $O(n)$), so long-range dependencies are easier to learn and gradients do not pass through $n$ multiplied Jacobians.
- Input-dependent weighting: attention weights are computed from the content, so the model chooses what to read at each position.
- Costs: $O(n^2)$ time and memory in sequence length; no built-in notion of order, so positions must be encoded; and autoregressive decoding needs a KV cache that grows with length, while an RNN carries a fixed-size state with $O(1)$ memory per step.

#### What prevents vanishing gradients in a Transformer

**Asked:** level not stated ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html))

**Question.** What in the Transformer prevents vanishing gradients?

**Answer.**

- Residual connections: with $x_{l+1} = x_l + F_l(x_l)$, the Jacobian is $\partial x_{l+1} / \partial x_l = I + \partial F_l / \partial x_l$, so the gradient always has an identity path through the whole depth.
- Layer normalization keeps activation scales stable. Pre-LN leaves the residual stream itself un-normalized, so the identity path stays clean and training is stable without a long warmup; post-LN needs warmup.
- The $1/\sqrt{d_k}$ scaling keeps the softmax away from saturation, where its gradient goes to zero.
- Non-saturating activations (ReLU, GELU) in the FFN.
- Along the sequence, attention connects positions directly, avoiding the long products of Jacobians of backpropagation through time.
- Training details: initialization that scales the residual-branch output weights down with depth (GPT-2 uses $1/\sqrt{N}$ for $N$ residual layers), learning-rate warmup and gradient clipping.

#### Attention complexity and where the parameters are

**Asked:** level not stated ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html))

**Question.** What is the time complexity of an attention block, and is it quadratic? Which part of the Transformer has the most parameters?

**Answer.** Take one layer with sequence length $n$, width $d$ and $d_{ff} = 4d$. The multiply-accumulates are about $4nd^2$ for the Q, K, V and output projections, $2n^2 d$ for $QK^{\top}$ and the weighted sum of $V$, and $8nd^2$ for the FFN:

$$
\text{cost per layer} \approx 12nd^2 + 2n^2 d
$$

So attention is quadratic in $n$, but the quadratic term dominates only when $n > 6d$; for shorter sequences the projections and FFN dominate. The score matrix also needs $O(n^2)$ memory per head unless a fused kernel such as FlashAttention avoids materializing it. With a KV cache, generating one token costs, per layer, $O(nd)$ for attention plus $O(d^2)$ for the dense layers.

Parameters per layer: attention has $W_Q, W_K, W_V, W_O$, which is $4d^2$; the FFN has $W_1 \in \mathbb{R}^{d \times 4d}$ and $W_2 \in \mathbb{R}^{4d \times d}$, which is $8d^2$. In the standard configuration the FFN therefore holds about two-thirds of each block. Grouped-query attention shrinks the attention share further, and SwiGLU with $d_{ff} \approx \tfrac{8}{3}d$ uses three matrices and keeps the FFN at about $8d^2$. Outside the blocks, the token embedding table ($\lvert \mathcal{V} \rvert \cdot d$ for vocabulary $\mathcal{V}$) can be the largest single matrix in small models: about 39M of GPT-2 small's 124M parameters. The split depends on the configuration; for the standard one the answer is the FFN.

#### Making self-attention faster

**Asked:** PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html)), level not stated ([1123499](https://www.1point3acres.com/bbs/thread-1123499-1-1.html), [1125433](https://www.1point3acres.com/bbs/thread-1125433-1-1.html))

**Question.** How do you make self-attention faster? What are linear and sparse self-attention? Which Transformer optimizations, such as FlashAttention, do you know?

**Answer.** Separate exact kernel-level speedups from changes to the attention function itself.

1. Exact, IO-aware kernels. FlashAttention computes exact attention without writing the $n \times n$ score matrix to GPU high-bandwidth memory (HBM). It tiles $Q$, $K$ and $V$ into on-chip SRAM and uses an online softmax: each query row keeps a running max $m$, normalizer $\ell$ and unnormalized output $o$, and each new block of scores $s_j$ with values $v_j$ updates them as

   $$
   m' = \max\bigl(m, \max_j s_j\bigr), \qquad \ell' = e^{m - m'}\,\ell + \sum_j e^{s_j - m'}, \qquad o' = e^{m - m'}\, o + \sum_j e^{s_j - m'}\, v_j
   $$

   with the final output $o / \ell$. The backward pass recomputes the scores from the saved row statistics instead of storing them. FLOPs stay $O(n^2 d)$, but extra memory falls from $O(n^2)$ to $O(n)$ and HBM traffic drops sharply, which is where the speedup comes from. It is a kernel optimization, not an architectural change: the outputs match standard attention up to floating-point error. FlashAttention-2 cuts the non-matmul work and also parallelizes over the sequence dimension; FlashAttention-3 targets Hopper GPUs with asynchronous, warp-specialized pipelines (TMA), overlaps the softmax with the matmuls and adds FP8.
2. Sparse attention. Each query attends to a subset of keys: a local window of width $w$, a few global tokens, or strided or random positions (Sparse Transformer, Longformer, BigBird). The cost drops from $O(n^2 d)$ to $O(nwd)$. Sliding-window attention is also used in LLMs such as Mistral.
3. Linear attention. Replace $\exp(q^{\top}k)$ with a feature map $\phi(q)^{\top}\phi(k)$ and reassociate the product:

   $$
   o_i = \frac{\phi(q_i)^{\top} S}{\phi(q_i)^{\top} z}, \qquad S = \sum_j \phi(k_j)\, v_j^{\top}, \qquad z = \sum_j \phi(k_j)
   $$

   All queries share $S \in \mathbb{R}^{r \times d_v}$ and $z \in \mathbb{R}^{r}$, so the cost is $O(n r d_v)$, linear in $n$. The causal version keeps running prefix sums of $S$ and $z$ and behaves like an RNN with a matrix-valued state. Examples are the linear Transformer ($\phi(x) = \mathrm{elu}(x) + 1$) and Performer (random features that approximate the softmax kernel). There is usually some quality loss compared with softmax attention.
4. Other approximations of the softmax matrix: low-rank projection of $K$ and $V$ along the sequence to a fixed length (Linformer), and hashing similar queries and keys into buckets so attention is computed within buckets (Reformer).
5. Inference-side: a KV cache so past keys and values are not recomputed, multi-query or grouped-query attention to shrink that cache and its memory bandwidth, and PagedAttention to manage cache memory in batched serving.

<a id="gpt-and-bert-pre-training"></a>
<a id="t5-compared-with-gpt"></a>

#### Original Transformer, BERT, T5, and GPT

**Asked:** GPT/BERT pre-training and architecture — PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html)); T5/GPT comparison — level not stated ([1050390](https://www.1point3acres.com/bbs/thread-1050390-1-1.html)). These two reported topics are combined below; the original Transformer is added as explanatory context.

**Question.** Compare the original Transformer, BERT, T5 and GPT: architecture, attention visibility, training targets, loss functions, and how they are used after training.

**Answer.** Separate **architecture** from **training objective**. An encoder-decoder is a structure, not a particular pre-training task. Here, "original Transformer" means the 2017 translation model, BERT and T5 mean their original versions, and GPT means a causal decoder-only language model with **GPT-2** as the concrete implementation example. Do not assume that every later model uses the same normalization or position scheme.

**Side-by-side structure and training.**

<a id="transformer-family-comparison"></a>

| Dimension | Original Transformer | BERT | T5 | GPT / GPT-2 |
| --- | --- | --- | --- | --- |
| Main structure | Encoder + decoder | Encoder only | Encoder + decoder | Decoder-only stack |
| Self-attention visibility | Bidirectional encoder; causal decoder | Bidirectional over the corrupted input | Bidirectional encoder; causal decoder | Causal over one token stream |
| Cross-attention | Decoder reads encoder output | None | Decoder reads encoder output | None in the standard GPT-2 block |
| Training data and task | Paired source/target translations; supervised | Unlabeled text; masked tokens + sentence-pair objective | Unlabeled text; recover removed spans | Unlabeled text; predict the next token |
| Token-loss positions | Target-language positions | Selected token positions | Serialized missing spans and sentinels | Next-token targets across the sequence |
| Typical task interface | Source text → target text | Add a classification, tagging or span head | Text → text, including labels written as text | Prompt → continuation; task-specific fine-tuning is also possible |
| Position information | Sinusoidal absolute positions in the main configuration | Learned absolute positions + segment embeddings | Learned bucketed relative attention bias | Learned absolute positions in GPT-2 |
| Norm / FFN activation | Post-LN / ReLU | Post-LN / GELU | Pre-norm, RMS-style / ReLU | Pre-LN + final LN / GELU |

These are canonical configurations, not performance rankings. Architecture and objective are described in the [Transformer paper](https://arxiv.org/html/1706.03762v7), [BERT paper](https://arxiv.org/html/1810.04805v2), [T5 paper](https://arxiv.org/html/1910.10683v4), and [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Implementation details can be checked in the original [BERT code](https://github.com/google-research/bert/blob/master/modeling.py) and [GPT-2 code](https://github.com/openai/gpt-2/blob/master/src/model.py).

**Attention masks and tensor flow.**

Let $B$ be batch size, $L$ input length, $T$ output length, $d$ hidden width, $h$ the number of heads, $d_h=d/h$, and $V$ vocabulary size. Ignoring padding, bidirectional self-attention permits every input position; causal self-attention uses

$$
M_{ij}^{\mathrm{causal}}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

This mask is added to attention logits before softmax. In an encoder-decoder, the flow is

$$
H=\operatorname{Encoder}(x)\in\mathbb{R}^{B\times L\times d},\qquad
D=\operatorname{Decoder}(y_{\mathrm{shifted}},H)\in\mathbb{R}^{B\times T\times d},\qquad
Z=DW_{\mathrm{vocab}}+b\in\mathbb{R}^{B\times T\times V}.
$$

Here $y_{\mathrm{shifted}}=[\mathrm{BOS},y_1,\ldots,y_{T-1}]$, and $\operatorname{softmax}(Z_t)$ predicts $y_t$. In one cross-attention head, queries come from the current decoder states $S$, while keys and values come from the encoder:

$$
Q=SW_Q\in\mathbb{R}^{B\times T\times d_h},\quad
K=HW_K,\ V_{\mathrm{enc}}=HW_V\in\mathbb{R}^{B\times L\times d_h},\quad
QK^{\top}\in\mathbb{R}^{B\times T\times L}.
$$

The transpose is over the final two dimensions. Each output position can read all valid input positions. BERT has only the encoder path; GPT has a causal stream and vocabulary head, **not** this encoder/cross-attention path. Autoregressive does not imply cross-attention.

**1. Original Transformer — supervised conditional generation.**

Given a source sentence $x$ and its translation $y$, learn

$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid y_{<t},x),\qquad
\mathcal{L}_{\mathrm{translation}}=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},x).
$$

The decoder receives the shifted **ground-truth** translation during training: teacher forcing. This is paired translation training, not generic next-token pre-training on unlabeled text. The formula shows the basic token NLL; the original paper additionally uses label smoothing.

**2. BERT — predict selected tokens using both sides.**

Let $\mathcal{M}$ contain the selected positions and $\tilde x$ be the corrupted input:

$$
\mathcal{L}_{\mathrm{MLM}}=-\sum_{i\in\mathcal{M}}\log p_\theta(x_i\mid\tilde x),\qquad
\mathcal{L}_{\mathrm{BERT}}=\mathcal{L}_{\mathrm{MLM}}+\mathcal{L}_{\mathrm{NSP}}.
$$

Original BERT selects 15% of tokens: 80% of those become `[MASK]`, 10% become a random token, and 10% remain unchanged. **All selected positions** contribute to MLM, not only literal `[MASK]` positions. NSP is a binary next-sentence classification loss from `[CLS]`. At fine-tuning, a task head can use $h_{\mathrm{CLS}}$ or individual token states. This is not native left-to-right generation. [BERT pre-training details](https://arxiv.org/html/1810.04805v2#S3.SS1).

**3. T5 — encode corrupted text, autoregressively generate missing spans.**

Replace each removed span with a distinct sentinel to form $\tilde x$. Serialize the removed spans and their sentinel markers into a target sequence $z$:

$$
\mathcal{L}_{\mathrm{T5}}=-\sum_{t=1}^{|z|}\log p_\theta(z_t\mid z_{<t},\tilde x).
$$

T5 corrupts about 15% of tokens with mean span length 3. Its decoder predicts missing spans plus markers, **not the entire original input**. The encoder reads the corrupted input bidirectionally; the decoder remains causal. Fine-tuning keeps a text-to-text interface, such as a translation prefix or generating a class name. [T5 span-corruption objective](https://arxiv.org/html/1910.10683v4#S3.SS3.SSS4).

**4. GPT — autoregressive next-token pre-training.**

$$
p_\theta(x)=\prod_{t=1}^{L}p_\theta(x_t\mid x_{<t}),\qquad
\mathcal{L}_{\mathrm{GPT}}=-\sum_{t=1}^{L}\log p_\theta(x_t\mid x_{<t}).
$$

Input and continuation occupy one causal stream. There is no separate bidirectional encoder. At generation time, append a generated token and predict the next one. The [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) uses this language-model objective; prompting is a task interface, not a different pre-training loss.

**One toy sentence, three learning targets.** Words below stand for tokens; actual tokenizers may split them differently. These toy examples illustrate the objectives, not their real corruption percentages.

```text
Original text: I enjoy playing games

BERT input:   I [MASK] playing games
MLM target:   enjoy at the selected position

T5 input:     I <extra_id_0> games
T5 target:    <extra_id_0> enjoy playing <extra_id_1> EOS

GPT input:    BOS I enjoy playing
GPT targets:  I   enjoy playing games
              Each position predicts the next token.
```

**Training parallelism versus generation.** Teacher-forced original Transformer, T5, and GPT can compute all supervised decoder positions in parallel because the shifted true tokens are supplied and a causal mask prevents future-token leakage. At inference, they must generate successive tokens from their own prefixes. BERT predicts its selected positions in parallel without defining this left-to-right generation process.

**Norm placement is a separate design choice.** For a sublayer $F$, post-LN is $x'=\operatorname{LN}(x+F(x))$; pre-LN is $x'=x+F(\operatorname{Norm}(x))$. Do not confuse pre/post-normalization with bidirectional/causal masking: the former controls the residual computation; the latter controls which tokens can communicate.

#### Positional encodings

**Asked:** PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html))

**Question.** What kinds of positional encoding are there, and what are the pros and cons of each?

**Answer.**

| Type | Idea | Pros | Cons |
|---|---|---|---|
| Sinusoidal, absolute (original Transformer) | Add $PE_{(p,2i)} = \sin(p / 10000^{2i/d})$ and $PE_{(p,2i+1)} = \cos(p / 10000^{2i/d})$ to the input | No parameters; defined for any length; a fixed offset is a linear map of the encoding | Extrapolates poorly beyond trained lengths in practice; position is mixed into the content vector |
| Learned, absolute (BERT, GPT-2) | One trainable vector per position, added to the input | Simple and flexible | Hard maximum length; nothing learned for unseen positions |
| Relative (Shaw et al.; T5 bias) | A learned embedding or scalar bias per clipped or bucketed offset $i - j$, applied inside attention | Translation-invariant; generalizes better across lengths | Extra work inside every attention layer; buckets are coarse at long range |
| RoPE (LLaMA and most current LLMs) | Rotate each 2-D pair of $q$ and $k$ at position $m$ by angle $m\theta_i$, $\theta_i = 10000^{-2i/d}$, so $\langle R_m q, R_n k \rangle$ depends on position only through $n - m$ | Relative, no parameters, applied only to $q$ and $k$, so it works with the KV cache and fused kernels; context can be extended with position interpolation, NTK-aware scaling or YaRN | Degrades beyond the trained length without such scaling |
| ALiBi | Add $-m_h (i - j)$ to the logit of query $i$ and key $j \le i$, with a fixed slope $m_h$ per head | No parameters; extrapolates to longer inputs | Strong recency bias can hurt use of distant tokens; less common in recent LLMs |

### 1.2.2 Distributed training

#### Distributed training, tensor parallelism and pipeline parallelism

**Asked:** PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html))

**Question.** What kinds of distributed training are there? How does tensor parallelism (TP) work in detail? What are the drawbacks of pipeline parallelism (PP), and how do you mitigate them?

**Answer.**

- Data parallelism (DP). Each GPU holds a full replica and processes its own micro-batch, and the gradients are averaged with an all-reduce every step, overlapped with the backward pass by bucketing. A ring all-reduce moves about $2(N-1)/N$ times the gradient size per GPU, nearly independent of $N$. Mixed-precision Adam needs about $16\Psi$ bytes for $\Psi$ parameters (fp16 weights and gradients plus fp32 master weights and two moments), so ZeRO and FSDP shard the optimizer states (stage 1), the gradients (stage 2) and the parameters (stage 3), gathering parameters just in time; stage 3 costs about 1.5 times the communication of plain DP.
- Tensor parallelism (TP, Megatron-LM). Split the weight matrices inside each layer across $t$ GPUs. For the FFN $Z = \mathrm{GeLU}(XA)\,B$, split $A = [A_1, \dots, A_t]$ by columns, so GPU $i$ computes $Y_i = \mathrm{GeLU}(XA_i)$ locally (GeLU is elementwise, so no communication is needed); split $B$ by rows, so GPU $i$ computes $Y_i B_i$; then one all-reduce gives $Z = \sum_i Y_i B_i$. Attention is split by heads: $W_Q, W_K, W_V$ are column-parallel, so each GPU computes whole heads, and $W_O$ is row-parallel, followed by one all-reduce. That is two all-reduces per layer in the forward pass and two in the backward pass, all on the critical path, so TP is kept inside a node on NVLink (typically $t \le 8$). The embedding and output layers are split along the vocabulary. Sequence parallelism replaces each all-reduce with a reduce-scatter plus an all-gather and shards the LayerNorm and dropout activations along the sequence, which cuts activation memory.
- Pipeline parallelism (PP). Put consecutive layers on $p$ stages and stream $m$ micro-batches through them. Drawbacks:
  - the bubble: stages sit idle while the pipeline fills and drains, a fraction $(p-1)/(m+p-1)$ of the time with a GPipe schedule;
  - activation memory for all micro-batches in flight;
  - load imbalance between stages (the embedding and output head make the first and last stages heavier);
  - point-to-point latency between stages, and a large global batch, since $m \gg p$ is needed, which can affect convergence.

  Mitigations:
  - many more micro-batches than stages;
  - 1F1B scheduling, which keeps the GPipe bubble but caps in-flight activations at about $p$ micro-batches;
  - interleaved virtual stages: $v$ model chunks per GPU shrink the bubble by about a factor $v$, at the cost of more communication;
  - zero-bubble schedules, which split the backward pass into input-gradient and weight-gradient parts and use the weight-gradient work to fill the bubble;
  - activation recomputation, and balancing layers across stages.
- Others. Context (sequence) parallelism, such as ring attention, splits long sequences across GPUs; expert parallelism places MoE experts on different GPUs and routes tokens with all-to-all. Large runs combine them as 3-D parallelism: TP within a node, PP across nodes and DP or FSDP on top.

### 1.2.3 Recommender systems and two-tower models

#### Two-tower models

**Asked:** senior ([1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html)), IC6 (principal) ([995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html))

**Question.** Explain two-tower models and answer follow-up questions on them.

**Answer.** A user tower $u = f_\theta(\text{user features, history, context})$ and an item tower $v_i = g_\phi(\text{item ID, genre, creator, statistics, text})$, each an MLP over embeddings, output L2-normalized vectors, and the score is $s(u,i) = \langle u, v_i \rangle / \tau$ with temperature $\tau$. The towers meet only at the final dot product, which is what makes retrieval over millions of items cheap. Train with a sampled softmax over in-batch negatives. In-batch negatives are sampled roughly in proportion to item popularity $Q(j)$, which would over-penalize popular items, so subtract $\log Q(j)$ from every logit (logQ correction):

$$
\mathcal{L} = -\log \frac{\exp\bigl(s(u,i^+) - \log Q(i^+)\bigr)}{\sum_{j \in \mathcal{N} \cup \lbrace i^+ \rbrace} \exp\bigl(s(u,j) - \log Q(j)\bigr)}
$$

Mix in uniformly sampled negatives from the whole corpus, and add hard negatives (shown but not played) with care, since some are false negatives. At serving time, precompute item vectors into an approximate nearest-neighbor index (HNSW, IVF-PQ, ScaNN), compute the user vector per request and take the top $K$ by inner product in milliseconds. Refresh the index as items and the model change, and keep both towers on the same model version. Limitations and typical follow-ups: there are no early user-item feature crosses, so it is a retrieval model and a heavier ranker re-scores its output; new items get vectors from content features in the item tower rather than an untrained ID embedding; popular items dominate without logQ correction; evaluate it with Recall@K. Matrix factorization is the special case where both towers are ID embeddings only.

#### Optimizing a game recommender for play time

**Asked:** IC6 (principal) ([995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html))

**Question.** In an ML design question on in-session game recommendation for the home page, the objective discussed was play time. How do you model and optimize for play time?

**Answer.** Two standard formulations:

1. Decompose. Expected play time per impression is $\mathbb{E}[T] = P(\text{play}) \cdot \mathbb{E}[T \mid \text{play}]$. Train a BCE head for $P(\text{play})$ on all impressions and a regression head for $\mathbb{E}[T \mid \text{play}]$ on played impressions only (Huber loss, or MSE on $\log(1+T)$, because the tail is heavy), and rank by the product or by a weighted blend with other heads such as next-day return.
2. Weighted logistic regression. Train a classifier in which each positive is weighted by its play time $T_i$ and each negative by 1. With $N$ impressions of which $k$ were played, the learned odds are

   $$
   e^{z} \approx \frac{\sum_i T_i}{N - k} = \frac{\mathbb{E}[T]}{1 - P(\text{play})} \approx \mathbb{E}[T]\bigl(1 + P(\text{play})\bigr)
   $$

   so serving $e^{z}$ ranks by expected play time when the play rate is small.

Caveats: raw play time favors long-session genres and idle (AFK) sessions, so cap it, drop idle time or normalize per genre; it is known only when the session ends, so labels are delayed; and maximizing it alone can conflict with satisfaction and retention, so guard it with return rate and tune the blend weights online.

#### Home-page recommendation compared with an ads system

**Asked:** principal ([1132251](https://www.1point3acres.com/bbs/thread-1132251-1-1.html))

**Question.** While designing YouTube's home-page recommendations, the candidate, whose background was ads, was challenged on how the home-page problem differs from an ads system as a business problem. How do they differ?

**Answer.**

- Objective: ads maximize revenue under advertiser constraints and rank by expected value, roughly $\text{eCPM} \propto \text{bid} \times \text{pCTR}$ (times pCVR for conversion bids), inside an auction with budgets, pacing and reserve prices, while user experience acts as a constraint. A home feed has no bid; it optimizes user engagement and long-term satisfaction (watch time, return visits), plus ecosystem goals such as creator exposure.
- Calibration: essential in ads, because the predicted probability sets the price. In a feed the ordering matters most, although blending several objectives still needs scores on comparable scales.
- Unit of evaluation: ads are judged slot by slot with few slots; a home page is judged as a whole page (diversity, freshness, dedup across rows, coverage) with many slots.
- Labels: ads use clicks and conversions with delays and attribution problems; a feed uses watch time, likes and surveys, with long-term effects that need holdouts to measure.
- Inventory: ad candidates are limited by targeting and budgets; a feed can show any content and needs exploration for new items and creators.
- Shared: a retrieval-then-ranking funnel, position bias and feedback loops.

### 1.2.4 Features and training data

#### Feature selection and processing for a game recommender

**Asked:** PhD new grad ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html))

**Question.** For recommending games to users, which features would you select and how would you process them?

**Answer.** Features, grouped by source:

- user: coarse demographics, locale, language, device, account age;
- user history: the last $N$ games played, time-decayed play counts and play time per genre;
- game: ID, genre, creator, age, average session length, concurrent players, rating, historical CTR and play rate;
- context: time of day, day of week, platform;
- social: friends currently playing and friends' recent games;
- crosses: user-genre and user-creator affinity, and whether the user played this game before.

Processing:

- Numeric: clip outliers, log-transform heavy-tailed counts, then standardize or bucketize.
- Rates: smooth them toward a prior, for example $\widehat{\text{CTR}} = (\text{clicks} + \alpha) / (\text{impressions} + \alpha + \beta)$, so games with few impressions do not get extreme values.
- Categorical: embeddings or hashing (see the next question).
- Sequences: mean-pool or attend over the embeddings of recent games.
- Missing values: an explicit indicator or default value.
- Point in time: compute every feature as of the impression time.

Select features with ablations and importance scores, and keep only those available online within the latency budget.

#### Categorical features

**Asked:** senior ([1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html))

**Question.** How do you usually handle categorical features?

**Answer.**

- Low cardinality: one-hot encoding.
- High cardinality: learned embeddings in a DNN, or the hashing trick into a fixed number of buckets, with several hash functions to reduce collisions.
- Target (mean) encoding works well for tree models, but compute it out of fold and smooth it toward the global mean, or it leaks the label.
- Map rare and unseen values to an out-of-vocabulary bucket.
- Combine fields with crosses (hashed pairs, or a learned cross layer as in DCN).
- LightGBM and CatBoost handle categorical features natively; CatBoost uses ordered target statistics to avoid leakage.

#### Long-tail ID features

**Asked:** senior ([1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html), [1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html))

**Question.** An ID feature has a long-tail distribution. How do you handle it, and how do you handle the long tail in a game recommender's features and model in general?

**Answer.** There are two problems.

1. Sparse embeddings for rare IDs.
   - Send IDs below a frequency threshold to a shared or hashed bucket, and use multiple hashes or quotient-remainder embeddings to limit collisions.
   - Add side features (genre, creator, tags) so tail items borrow strength from their attributes.
   - Give rare IDs smaller embeddings, or regularize them more strongly in proportion to their rarity.
2. Popularity bias in training and exposure.
   - In retrieval, use logQ correction and popularity-aware negative sampling.
   - Add a content-based retrieval source for tail items.
   - Add exposure or diversity constraints in re-ranking, and reserve exploration slots.

Evaluate head and tail slices separately, because an overall metric can rise while tail quality falls.

#### Processing user interaction data

**Asked:** senior ([1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html))

**Question.** How do you process user interaction data?

**Answer.**

- Labels: join impression logs to clicks and plays. A game that was shown and played is a positive, one shown and skipped is a negative, and games never shown can only be sampled negatives (absence is not dislike).
- Cleaning: require a minimum play time so accidental clicks do not count as positives, and remove bots, duplicates and test traffic.
- Grouping: sessionize events and aggregate repeated actions.
- Position bias: train with position as a feature and set it to a constant at serving time, or weight examples by inverse propensity.
- Behavior features: sequences such as the last $N$ games, and time-decayed counts.
- Leakage: split train and test by time, and join each feature as of its event's timestamp.
- Heavy users: cap events per user so they do not dominate the loss.

#### Building training samples

**Asked:** PhD new grad ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html))

**Question.** For recommending games to users, how do you build the training samples?

**Answer.**

- Unit: one example per impression (user, game, context, timestamp), with features as of that timestamp.
- Labels: outcomes inside a fixed window after the impression, waiting until the window closes before training.
- Ranking: learns from shown-and-not-played negatives.
- Retrieval: needs sampled negatives (in-batch, random from the corpus, and some hard negatives).
- Downsampling: with very few positives, keep only a fraction $w$ of the negatives. This inflates the predicted probability, so recalibrate the model output $p'$ with

  $$
  p = \frac{p'}{p' + (1 - p')/w}
  $$

  This matters whenever scores are used as probabilities or blended.
- Per-user caps: limit examples per user, so heavy users do not dominate.
- Split by time: train on earlier days and validate on later ones.

#### Online and offline data consistency

**Asked:** level not stated ([1014514](https://www.1point3acres.com/bbs/thread-1014514-1-1.html))

**Question.** How do you keep online and offline data consistent?

**Answer.** This is training-serving skew. Common causes:

- feature logic implemented twice (batch SQL offline, service code online);
- online features that are stale or missing, or that come from a different data source;
- leakage, where offline features were computed after the label time;
- preprocessing or model versions that differ between training and serving.

Fixes:

- one feature definition used by both paths, through a feature store with point-in-time joins;
- logging the features actually served and training on those logs;
- monitoring online and offline feature distributions (population stability index, null rates);
- shadow scoring, comparing online and offline predictions on the same requests before launch.

### 1.2.5 Losses, optimization and evaluation

#### Classification loss

**Asked:** senior ([1147380](https://www.1point3acres.com/bbs/thread-1147380-1-1.html))

**Question.** Which loss is usually used for classification? Write the formula.

**Answer.** Binary cross-entropy for binary labels and softmax cross-entropy for $C$ classes:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\bigl[y_i \log p_i + (1-y_i)\log(1-p_i)\bigr], \qquad p_i = \sigma(z_i)
$$

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{z_{i,y_i}}}{\sum_{c=1}^{C} e^{z_{i,c}}}
$$

Each is the negative log-likelihood of a Bernoulli or categorical model. The gradient with respect to the logits is $p - y$, so it stays large when the model is confidently wrong, unlike MSE on a sigmoid output. Compute it from logits with log-sum-exp for numerical stability. For class imbalance use class weights or focal loss, $-\alpha_t (1-p_t)^{\gamma}\log p_t$; for multi-label targets use one sigmoid with BCE per label.

#### Loss and optimizer for a game recommender

**Asked:** PhD new grad ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html)), senior ([1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html))

**Question.** Define the loss for a game recommender. Which optimization algorithm would you use?

**Answer.** For ranking, the default is pointwise BCE on play labels, which gives calibrated probabilities; a multi-task ranker sums weighted per-head losses (click, play, play time, return). Pairwise BPR optimizes the order directly:

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \log \sigma\bigl(\hat{s}_{ui} - \hat{s}_{uj}\bigr) + \lambda \lVert \Theta \rVert^2
$$

where $i$ is a game user $u$ played and $j$ a sampled negative. Listwise sampled softmax (see two-tower models above) suits retrieval. Train the dense layers with Adam or AdamW:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2, \qquad \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_{t-1}\right)
$$

The $\lambda\theta$ term is AdamW's decoupled weight decay. Large embedding tables usually use sparse or row-wise Adagrad instead, because each step touches only a few rows and keeping two Adam moments per row is expensive; FTRL-Proximal is the classic choice for sparse logistic regression. Use learning-rate warmup and decay, and early stopping on a time-based validation set.

#### Evaluation metrics and A/B testing

**Asked:** PhD new grad ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html)), level not stated ([1014514](https://www.1point3acres.com/bbs/thread-1014514-1-1.html))

**Question.** Which evaluation metrics would you use, and how would you run the A/B test?

**Answer.** Offline, match the metric to the stage. For retrieval, use Recall@K: did the game the user eventually played make the candidate set? For ranking, use AUC or per-user GAUC (the impression-weighted mean of per-user AUC, skipping users with only one class), log loss and calibration for the probabilities, and NDCG@K for the order. AUC is the probability that a random positive scores above a random negative; with ranks assigned in ascending score order (ties averaged) it takes $O(n \log n)$:

$$
\mathrm{AUC} = \frac{\sum_{i \in \mathcal{P}} \mathrm{rank}_i - \frac{n_{+}(n_{+}+1)}{2}}{n_{+}\, n_{-}}
$$

$$
\mathrm{DCG@}K = \sum_{k=1}^{K} \frac{2^{\mathrm{rel}_k} - 1}{\log_2(k+1)}, \qquad \mathrm{NDCG@}K = \frac{\mathrm{DCG@}K}{\mathrm{IDCG@}K}
$$

When positives are rare, use PR-AUC.

Online:

- Randomize by user and pick one primary metric (play time per user, or D1/D7 return), with guardrails (latency, crashes, abuse reports, creator exposure).
- Size the test for a minimum detectable effect $\delta$:

  $$
  n_{\text{per arm}} \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\, \sigma^2}{\delta^2}
  $$

- Check for sample-ratio mismatch, and run long enough for novelty effects to fade.
- Use variance reduction (CUPED) to shorten the test.
- Watch for interference: on a social platform, treated users change what their friends play, so randomize by friend cluster or measure the spillover.
- Offline gains rarely carry over one to one; for rankers, interleaving is a cheap, sensitive online check before a full A/B test.

### 1.2.6 Classical ML and sampling

#### Classification and regression algorithms

**Asked:** PhD new grad ([1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html))

**Question.** Which classification and regression algorithms do you know? Give the pros and cons of each.

**Answer.**

| Algorithm | Pros | Cons |
|---|---|---|
| Linear / logistic regression | Fast, convex, interpretable; logistic regression gives fairly calibrated probabilities; L1 for sparsity, L2 for shrinkage | Linear decision boundary; needs hand-built feature crosses |
| Decision tree | Nonlinear, handles mixed feature types, needs no scaling, readable | High variance, overfits |
| Random forest | Bagging lowers variance; robust, little tuning | Large models, slower inference; usually behind GBDT; cannot extrapolate |
| GBDT (XGBoost, LightGBM) | Usually the best on tabular data; handles missing values and monotone constraints | Sequential training; many hyperparameters; weak on raw high-cardinality IDs; cannot extrapolate |
| SVM | Maximum margin; kernels for nonlinearity; effective in high dimensions | Kernel training scales between $O(n^2)$ and $O(n^3)$; no native probabilities |
| k-NN | No training, nonparametric | Slow queries, curse of dimensionality, needs feature scaling |
| Naive Bayes | Very fast; a good text baseline | Independence assumption; poorly calibrated |
| Neural networks | Learn representations for text, images, sequences and sparse IDs | Data-hungry, costly to tune, less interpretable |

For regression the same families apply, trained with squared, absolute or Huber loss (Huber is robust to outliers). Frame the comparison in terms of the bias-variance trade-off and regularization.

#### Reservoir sampling

**Asked:** level not stated ([1014514](https://www.1point3acres.com/bbs/thread-1014514-1-1.html))

**Question.** As follow-ups to Find Median from Data Stream (LeetCode 295), the interviewer asked about multiple data streams and about reservoir sampling: keeping a uniform random sample of $k$ items from a stream whose length is not known in advance.

**Answer.** Algorithm R: keep the first $k$ items; for item $t > k$, draw $j$ uniformly from $1, \dots, t$ and, if $j \le k$, overwrite slot $j$ with item $t$. This is one pass with $O(1)$ time per item and $O(k)$ memory. Item $t$ enters with probability $k/t$, and a later item $s$ evicts a given slot with probability $\frac{k}{s} \cdot \frac{1}{k} = \frac{1}{s}$, so

$$
P(\text{item } t \text{ in final sample}) = \frac{k}{t}\prod_{s=t+1}^{n}\Bigl(1-\frac{1}{s}\Bigr) = \frac{k}{t}\cdot\frac{t}{n} = \frac{k}{n}
$$

The same telescoping product starting at $s = k+1$ gives $k/n$ for the first $k$ items. For long streams, Algorithm L skips ahead by geometrically distributed gaps, so only $O(k \log(n/k))$ random draws are needed. For weighted sampling, give each item the key $u^{1/w}$ with $u \sim \mathrm{Uniform}(0,1)$ and weight $w$, and keep the $k$ largest keys in a min-heap (A-Res). For multiple or distributed streams, give every item an independent uniform key, keep the $k$ largest keys per stream, and merge by keeping the $k$ largest keys overall, which is a uniform sample of the union.

## 1.3 System design

### 1.3.1 ML system design

For the ML Modeling Knowledge round, use the [three CV / multimodal design mocks](ml_modeling_practice/roblox_cv/index.html), each with separate questions and a worked answer. They connect the reported topics below to the [CV foundations](02_ml_domain.md#26-interview-questions--computer-vision); the mock constraints, architectures and follow-ups are original practice adaptations, not newly verified interview reports.

#### Recommend games to a Roblox user

**Asked:** PhD new grad ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html), [1157817](https://www.1point3acres.com/bbs/thread-1157817-1-1.html), [1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html)), senior ([1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html)), staff ([1134892](https://www.1point3acres.com/bbs/thread-1134892-1-1.html)), IC6 (principal) ([995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html)), level not stated ([1074422](https://www.1point3acres.com/bbs/thread-1074422-1-1.html), [1057160](https://www.1point3acres.com/bbs/thread-1057160-1-1.html), [968319](https://www.1point3acres.com/bbs/thread-968319-1-1.html), [889847](https://www.1point3acres.com/bbs/thread-889847-1-1.html), [874270](https://www.1point3acres.com/bbs/thread-874270-1-1.html))

**Question.** Design a system that recommends games (Roblox calls them experiences) to a user: for example, personalized game recommendations on the home page for millions of users, served in real time, with training covered as well as serving [968319](https://www.1point3acres.com/bbs/thread-968319-1-1.html). Scope and probes reported:

- the objective (play time), candidate retrieval, ranking, features and model architecture [995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html);
- feature selection and processing, the model, training samples, the loss, the optimization algorithm, evaluation and A/B testing [1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html);
- how to handle long-tail items in the features and the model, and how to define the loss [1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html);
- the main components, then personalization [1057160](https://www.1point3acres.com/bbs/thread-1057160-1-1.html);
- the features you propose are declared unavailable one by one, so the design has to work without them [874270](https://www.1point3acres.com/bbs/thread-874270-1-1.html);
- real time [1074422](https://www.1point3acres.com/bbs/thread-1074422-1-1.html): a conventional design is not enough, because a newly launched game must be recommended and get exposure right away; and if a game suddenly becomes very popular, how do you update its popularity signal quickly and insert it into a feed that is already ranked?

Other reports give the prompt only as an open-ended recommender-system design [1157817](https://www.1point3acres.com/bbs/thread-1157817-1-1.html), [1124594](https://www.1point3acres.com/bbs/thread-1124594-1-1.html), [1134892](https://www.1point3acres.com/bbs/thread-1134892-1-1.html), [889847](https://www.1point3acres.com/bbs/thread-889847-1-1.html).

**Answer.**

1. **Framing.** Clarify the surface (home-page rows, "continue playing", a sort), the users (age band, region, device), the scale (tens of millions of DAU, millions of experiences, a p99 latency budget around 100-200 ms) and the objective: long-term engagement rather than clicks. A useful label is a *qualified play*, a join followed by at least $N$ minutes of play, combined with return visits. Constraints: freshness for new and trending games, age-appropriate content, and the creator long tail.
2. **Data and labels.** Impression logs with position, joins, session length, next-day return, favorites/likes, purchases, friend graph, search queries. Labels: $y^{\text{join}}$ per impression and play time $t$ per join. For ranking, negatives are impressed-but-not-joined items, not random items. Handle position bias by feeding position as a training feature and fixing it to a constant at serving time, or by inverse-propensity weighting. Play time arrives late, so close the label window (for example 24 h) before training.
3. **Features.** User features: history sequence, genre affinities, recency, device, locale, age band. Item features: genre, creator, age of the experience, concurrent players (CCU), like ratio, average session length, maturity rating. Context: time of day, platform. Cross features: friends currently in this experience (a strong Roblox-specific signal), user-genre affinity. Streaming counters: joins and CCU over the last 5 minutes. For long-tail ID features: hashed embeddings, a minimum-frequency cutoff that sends rare IDs to a shared bucket, and content features so cold items share statistical strength. If most side features are ruled out, the impression and join logs alone still support a working model: user and game ID embeddings (matrix factorization or a two-tower on IDs) plus item-to-item co-play.
4. **Candidate generation (union of sources, a few thousand items).** A two-tower model with ANN search (HNSW/ScaNN), "continue playing", friends-playing, item-to-item co-play, trending by region and age band, and an exploration pool of new games.
5. **Ranking.** A multi-task DNN (shared bottom or MMoE, with DCN/DLRM-style crosses) predicting $\hat p^{\text{join}}$, play time and the return probability. Score by expected play time per impression, $\hat p^{\text{join}}\cdot \mathbb{E}[t\mid\text{join}]$, plus a weighted return term. The play-time head below regresses $\log(1+t)$, and $e^{\hat t}-1$ underestimates the mean of $t$ (Jensen's inequality), so calibrate that conversion on held-out joins. Then re-rank for genre diversity (MMR), safety and maturity filters, deduplication and exploration slots.
6. **Losses.** Retrieval uses in-batch sampled softmax with logQ correction, which stops popular items from being over-penalized as negatives:

$$
\mathcal{L}_{\text{ret}} = -\frac{1}{|B|}\sum_{(u,i)\in B}\log\frac{\exp\big(s^{c}(u,i)\big)}{\sum_{j\in B}\exp\big(s^{c}(u,j)\big)},\qquad s^{c}(u,j)=\frac{\langle f(u),g(j)\rangle}{\tau}-\log q_j
$$

Here $q_j$ is item $j$'s sampling probability, roughly its frequency in the training stream. Ranking uses binary cross-entropy for the join head plus a play-time head $\hat t_n$ (predicting $\log(1+t_n)$) trained only on joins:

$$
\mathcal{L}_{\text{rank}} = -\sum_{n}\Big[y_n\log\hat p_n + (1-y_n)\log(1-\hat p_n)\Big] + \lambda\sum_{n:\,y_n=1}\big(\log(1+t_n)-\hat t_n\big)^2
$$

An alternative is weighted logistic regression, where positives are weighted by play time and negatives by 1. The learned odds $e^{z}$ then equal expected play time per impression times $1/(1-P)$, where $P$ is the join rate, so when $P$ is small they approximate expected play time per impression.

7. **Long tail and cold start.** The logQ correction, content-based item-tower inputs (genre, description and thumbnail embeddings) so a new game gets an embedding before any plays, and bandit exploration slots with capped traffic.
8. **Offline evaluation.** Use a time-based split to avoid leakage. Retrieval: Recall@K. Ranking: AUC and log loss per head, NDCG@K, calibration. Report slices for new users, new games, long-tail creators and age bands.
9. **Online A/B test.** Randomize by user. Primary metric: play time per DAU or D7 retention. Secondary: join rate, sessions, diversity, creator coverage. Guardrails: latency, safety reports, revenue. Run whole weeks to cover weekly cycles, and check whether the lift decays over time (a novelty effect). Size each arm with $n \approx 2(z_{1-\alpha/2}+z_{1-\beta})^2\sigma^2/\delta^2$. Friends influence each other, so consider cluster (friend-graph) randomization.
10. **Serving and real time.** Events (impressions, joins, leaves) go to a stream (Kafka), then windowed and decayed counters (Flink), then the online feature store, so popularity features lag by seconds, not a day. A decayed count updates in $O(1)$ per event: $C(t) = C(t')\,e^{-\lambda(t-t')} + 1$, with half-life $\ln 2/\lambda$.
    - *A game suddenly becomes hot.* Flag it when its join rate departs from its own baseline, $z = (r_{\text{now}} - \mu)/\sigma$ with $\mu,\sigma$ taken from the same game at the same hour of week, recompute the trending source every minute or so, and push the fresh popularity features. For feeds that are already ranked and cached, score only the delta candidates with the ranker on fresh features and insert each into the cached list at its score position (found by binary search in $O(\log n)$ comparisons), or invalidate the cached feed on the event; a short TTL bounds staleness either way.
    - *A new game must be recommended at once.* The item tower computes its embedding from content features at publish time and the game is inserted into the ANN index incrementally (HNSW supports online insertion), so it is retrievable before any plays. A new-game source fills reserved exploration slots by Thompson sampling: draw $\theta_i \sim \text{Beta}(1+s_i,\,1+f_i)$, with $s_i$ joins and $f_i$ impressions without a join, and show the games with the largest draws. Exposure does not wait for retraining.
    - Log the features used at serving time and train on those logs, so online and offline features match. Retrain daily, or incrementally from a warm start.
11. **Monitoring.** Feature drift (PSI), prediction and calibration drift, pipeline freshness, p99 latency, and online metric dashboards with rollback.

#### Game genre classification from scratch

**Asked:** level not stated ([1173576](https://www.1point3acres.com/bbs/thread-1173576-1-1.html))

**Question.** Design a system that classifies the genre of a game, built from scratch (nothing exists yet).

**Answer.**

1. **Framing.** It is multi-label, since a game can be both obby and roleplay. Agree the taxonomy with product. Downstream uses are discovery, search filters and recsys features. "From scratch" means there are no labels yet and maybe no taxonomy. Inference can be batch, run when a game is published or updated.
2. **Labels.** Creator-supplied tags (noisy), keyword labeling functions on title and description (weak supervision), LLM zero-shot labels checked by people, and a stratified human-labeled gold set with inter-annotator agreement (Cohen's $\kappa$ for two annotators, Fleiss' $\kappa$ for more). Use active learning (uncertainty sampling) to choose what to label next.
3. **Inputs.** Multilingual text (title, description, tags), images (icon, thumbnails, gameplay frames), gameplay metadata (max players, average session length, asset and script types, monetization patterns), and behavior. The co-play graph is useful: games played by the same users tend to share a genre, so graph embeddings help.
4. **Model ladder.** v0 is TF-IDF + logistic regression, or zero-shot CLIP/LLM. v1 is a fine-tuned multilingual text encoder plus an image encoder (CLIP), fused late with tabular and graph features in an MLP, with one sigmoid output per genre.
5. **Loss.** Weighted multi-label BCE (or focal loss for rare genres):

$$
\mathcal{L} = -\sum_{c=1}^{C} w_c\Big[y_c\log\sigma(z_c) + (1-y_c)\log\big(1-\sigma(z_c)\big)\Big]
$$

6. **Evaluation.** Per-class precision and recall, macro-F1 and micro-F1, mAP. Tune a threshold per class on validation. Do human audits and confusion analysis between neighbouring genres. Online, A/B test the downstream effect: genre-filter CTR, search success, recsys engagement.
7. **Serving and monitoring.** Event-driven batch inference written to the metadata/feature store. Watch the distribution of predicted genres for drift as new genres trend. Re-label periodically, version the taxonomy, and give creators a way to appeal.

#### Harmful chat and video content detection

**Asked:** target L6 ([1125288](https://www.1point3acres.com/bbs/thread-1125288-1-1.html))

**Question.** Build a model that catches bad content such as chat messages and videos. A discussion point reported: one multimodal model versus separate models per content type.

**Answer.** A tiered system: cheap, high-throughput filters first, heavier per-modality models behind them, and a multimodal model only on escalations.

1. **Framing.** Separate text chat (real time, on the send path, tens of ms) from images and video (asynchronous). Classes come from policy (harassment, sexual content, grooming, self-harm, scams, sharing personal information); it is multi-label. Use a high-precision threshold for automatic action (block, blur) and a high-recall threshold for the human review queue. Many users are minors, so the cost of a miss is high.
2. **Labels.** Moderator decisions, user reports (noisy and biased), appeals (label corrections), and red-team data. Adversaries adapt (misspellings, spacing, text inside images), so plan for continuous retraining.
3. **Model tiers.** Hash matching (perceptual hashes) and blocklists for known content, at near-zero cost. A small, distilled multilingual text classifier at send time, with the last few messages and sender features as context. Heavier per-modality models offline: an image classifier on sampled video frames, OCR for text in images, ASR plus the text model for audio. A multimodal model only for the ambiguous slice that the cheap tiers escalate.
4. **One multimodal model versus separate models.** Separate models keep independent latency budgets per modality, retrain independently as each modality's adversaries shift, reuse existing classifiers, and make errors easy to attribute and thresholds easy to set per policy. One multimodal model catches cross-modal violations (a benign image plus benign text that are harmful together) and shares representations, but it is expensive, too slow for the chat path, and a single point of failure. The practical design is separate cheap models first, with their scores fused, and the multimodal model on escalations.
5. **Loss and evaluation.** Class-weighted BCE per policy, or focal loss $\mathcal{L} = -\alpha_t(1-p_t)^{\gamma}\log p_t$, where $p_t$ is the predicted probability of the true label, to down-weight easy negatives. The classes are heavily imbalanced, so report PR-AUC and precision at a fixed recall rather than accuracy. Estimate prevalence by labeling a random sample of traffic. Online metrics: user report rate, review-queue precision, and the appeal overturn rate.
6. **Monitoring.** Adversarial drift, per-language slices, false-positive audits (over-blocking hurts normal chat), and threshold re-tuning.

#### Bad-account detection with existing sub-models

**Asked:** level not stated ([927016](https://www.1point3acres.com/bbs/thread-927016-1-1.html))

**Question.** Detect bad accounts, given that many sub-models already exist, for example models that detect spam.

**Answer.**

1. **Framing.** Define the bad classes (spam bots, scammers, compromised accounts, ban evaders) and the decision points: at sign-up (few signals), after the first actions, and continuously. Map scores to an action ladder: no action, friction (CAPTCHA, rate limit, verification), restriction, suspension.
2. **Sub-models as features.** Aggregate each sub-model's per-event scores per account over time windows (last hour, day, lifetime): max, mean, and count or fraction of events above threshold. Add behavioral features (account age, message and friend-request rates, friend-request acceptance ratio, distinct recipients, near-duplicate messages via MinHash), device and IP fingerprints, payment signals, and graph features (shared devices or IPs with banned accounts, label propagation from known bad accounts).
3. **Model.** GBDT on the aggregated features as the baseline: it handles heterogeneous and missing features and is fast. Then a sequence model over the account's event stream, and a GNN for coordinated rings. Train the account model on a later time period than the sub-models' training data, so the sub-model scores are not optimistically leaky; retrain or recalibrate the account model whenever a sub-model changes, since its score distribution shifts.
4. **Labels.** Past enforcement actions (biased toward what the old system caught), appeal overturns as negatives, and manual review of a random sample for unbiased evaluation.
5. **Evaluation.** PR-AUC, precision at the action threshold, recall at a fixed false-positive rate, with costs set by action: a wrongful suspension of a real, often young, user costs far more than adding friction. Online: spam reports per DAU, appeal overturn rate, time to detection.
6. **Serving.** Streaming feature updates and re-scoring on triggers (a burst of messages or friend requests), plus a daily batch re-score of all accounts and daily graph jobs.

#### Search-box recommendation

**Asked:** level not stated ([1150441](https://www.1point3acres.com/bbs/thread-1150441-1-1.html))

**Question.** Design the recommendations shown in the search box.

**Answer.**

1. **Framing.** Clarify whether it means suggestions before typing (zero prefix: recent searches, trending queries, personalized games), completions while typing, or both. Objective: search success (a click or join after the search) and keystrokes saved. A request fires per keystroke, so the server budget is tens of ms; the client debounces.
2. **Candidates.** Prefix lookup in a trie or FST over normalized popular queries and game titles, with a precomputed top-k per prefix; the user's own history; trending queries from streaming counts; typo tolerance through an edit-distance index (edit distance 1-2); locale-aware normalization.
3. **Ranking.** Learning to rank (LambdaMART) over a small candidate list, with features such as match type, decayed query popularity, suggestion CTR by position, user-genre affinity, freshness and locale. Labels are clicked suggestions, with position-bias correction.
4. **Safety.** Filter offensive or age-inappropriate completions with a blocklist plus a classifier, at build time and at serve time.
5. **Evaluation.** Offline, MRR $= \frac{1}{|Q|}\sum_{q\in Q} 1/\text{rank}_q$ of the eventually chosen query or game, and keystrokes saved. Online A/B: search success rate, time to first join, abandonment.
6. **Serving.** Cache the top-k for head prefixes (short prefixes carry most traffic), re-rank a small list per user, rebuild the trie from logs on a schedule, and overlay trending queries in near real time.

#### Language detection of text

**Asked:** level not stated ([889847](https://www.1point3acres.com/bbs/thread-889847-1-1.html))

**Question.** Design an NLP model that detects the language of a piece of text.

**Answer.**

1. **Framing.** Output a distribution over languages plus "unknown". Inputs are often short chat messages, noisy (slang, emoji, typos), mixed (code-switching) or transliterated (romanized Hindi or Arabic). Uses: routing to per-language moderation, translation, search.
2. **Baseline.** A Unicode-script rule first (Hangul is Korean, kana is Japanese). Then hashed character n-grams (1-5) with a linear softmax, as in fastText: $p(\ell\mid x)=\operatorname{softmax}\big(W\,\bar e(x)\big)_\ell$, where $\bar e(x)$ is the mean n-gram embedding. Cost is $O(\text{length})$ and it runs in microseconds on CPU.
3. **Stronger model.** A small multilingual transformer or byte-level CNN, fine-tuned, only if the baseline fails on short text.
4. **Data.** Wikipedia, parallel-sentence corpora and web text with known language; platform text weakly labeled by account locale (noisy); a human-labeled test set of real short messages per language. Rebalance with temperature sampling $p_\ell \propto n_\ell^{1/T}$, $T>1$, to lift low-resource languages.
5. **Loss.** Cross-entropy. For code-switching, multi-label sigmoids or token-level tagging.
6. **Edge cases.** Very short input: combine with a per-user prior, assuming $x$ and $u$ independent given $\ell$: $p(\ell\mid x,u)\propto p(\ell\mid x)\,p(\ell\mid u)/p(\ell)$. Close pairs (Spanish and Portuguese, Indonesian and Malay, Serbian and Croatian, Danish and Norwegian). Numbers, emoji or URLs only: return "unknown" below a confidence threshold.
7. **Evaluation.** Per-language precision and recall, macro-F1, accuracy bucketed by text length, the confusion matrix, and CPU latency.

#### YouTube home-page recommendation

**Asked:** principal ([1132251](https://www.1point3acres.com/bbs/thread-1132251-1-1.html))

**Question.** Design recommendations for the YouTube home page. A point the interviewer challenged: how the home-page business problem differs from an ads system.

**Answer.**

1. **Clarify.** Logged-in or logged-out users, devices (TV, mobile), the objective (long-term satisfaction: watch time, surveys, likes, return visits, not clicks), scale (billions of videos and users, a latency budget of a few hundred ms) and constraints (freshness, diversity, borderline content).
2. **Candidate generation.** Several sources: a two-tower or deep retrieval model with sampled softmax over the watch history, new uploads from subscribed channels, co-watch item-to-item, trending, and topic exploration; a few thousand candidates in total.
3. **Ranking.** A multi-task model (MMoE) predicting click, watch time, like, share, dismissal and survey satisfaction, combined as $\text{score}=\sum_k w_k\,\hat y_k$. Watch time can use weighted logistic regression, whose odds approximate expected watch time per impression. A shallow tower on position and device absorbs selection bias and is dropped at serving.
4. **Re-ranking.** Diversity (MMR or DPP), freshness boosts, per-channel caps, policy demotion.
5. **Evaluation.** Offline: Recall@K for retrieval, AUC and NDCG for ranking on a time split. Online A/B: watch time per user, DAU, survey score, with guardrails.
6. **How it differs from ads.**
   - *Objective.* The home page maximizes the user's long-term satisfaction. Ads maximize advertiser value and revenue under a user-experience constraint, ranking by expected value per impression, for example $\text{eCPM}=1000\cdot\text{bid}_{\text{CPA}}\cdot\hat p_{\text{click}}\cdot\hat p_{\text{conv}\mid\text{click}}$, inside an auction (generalized second price or VCG) that sets the price.
   - *Calibration.* In ads the predicted probabilities set bids and prices, so they must be calibrated in absolute terms (log loss, predicted-to-observed ratio). On the home page the ranking order is what matters; the heads only need stable relative scales so that the weighted sum stays meaningful.
   - *Labels.* Conversions are rare and delayed by attribution windows, so ads need delayed-feedback modeling. The home page has dense, immediate engagement labels.
   - *Candidates.* Ads are limited by targeting, budgets and pacing (an ad stops when its budget is spent). Any eligible video can be a home-page candidate.
   - *Page-level concerns.* The home page cares about diversity, exploration and the creator ecosystem; ads care about ad load and advertiser ROI.

#### A relevance-based product backed by an ML model

**Asked:** level not stated ([768938](https://www.1point3acres.com/bbs/thread-768938-1-1.html))

**Question.** Design a relevance-based product backed by an ML model.

**Answer.**

1. **Framing.** Pin the product down first, for example game search or "related games". Define the query or context, the unit ranked, graded relevance (perfect, good, fair, bad), the business metric and the latency budget.
2. **Labels.** Human-rated graded relevance on sampled query-item pairs, with written guidelines and agreement checks, plus clicks corrected for position bias by inverse propensity weighting, $w = 1/P(\text{examined}\mid\text{position})$, with propensities estimated from result randomization.
3. **Retrieval.** Lexical (BM25 on an inverted index) unioned with semantic (dual-encoder embeddings with ANN) for recall.
4. **Ranking.** A multi-stage funnel: GBDT LambdaMART on hundreds of candidates (BM25 score, embedding similarity, popularity, freshness, quality, personalization), then a cross-encoder on the top few dozen. LambdaMART weights each pairwise gradient by the $|\Delta\text{NDCG}|$ of swapping the pair.
5. **Metrics.** Offline NDCG@K and MRR:

$$
\text{DCG@}K=\sum_{i=1}^{K}\frac{2^{\text{rel}_i}-1}{\log_2(i+1)},\qquad \text{NDCG@}K=\frac{\text{DCG@}K}{\text{IDCG@}K}
$$

Online A/B: success rate, abandonment, reformulation rate.
6. **Serving.** Cache head queries, update the index incrementally, monitor drift in query mix and score distributions, and fall back to lexical ranking if the model service fails.

### 1.3.2 System design

#### Game matchmaking

**Asked:** staff ([1136387](https://www.1point3acres.com/bbs/thread-1136387-1-1.html))

**Question.** Design a matchmaking system for many games with 1M DAU. After choosing a game, a user enters a waiting room; the system groups users of similar skill and starts the game as soon as a group forms. Skill is a value from 0 to 100 that an existing API returns. Set your own constraints, for example that a user must not wait too long, and that the minimum group size can shrink when few players are online. Another version of the prompt, quoted in the thread, adds: scale to matching hundreds or thousands of games, a message queue to absorb bursts, regional server clusters, and how servers connect to clients.

**Answer.**

1. **Requirements and scale.** Join a queue for a game (and region), cancel, get notified of a match with a server address; groups of size between $m_{\min}$ and $m_{\max}$ (per-game config) with close skill; wait bounded by $T_{\max}$; no player in two matches. With 1M DAU and about 10 matches per user per day, there are 10M joins per day, about 120 per second on average; allowing for evening peaks and bursts, size for 1-2k per second. By Little's law $L=\lambda W$, a 10 s average wait at 1.5k joins per second keeps about 15k tickets waiting: a few hundred bytes each, so the queues fit in memory.
2. **API.** `POST /tickets {userId, gameId, region, partyId?}` returns `ticketId`; `DELETE /tickets/{id}`; a WebSocket push of `MatchFound {matchId, serverAddr, joinToken}`. Internal: the skill API (cached, since skill changes slowly) and a game-server allocator `allocate(gameId, region, n)`.
3. **Data model.** `Ticket {ticketId, userId, gameId, region, skill, enqueuedAt, status: WAITING | MATCHED | CANCELLED | EXPIRED}`; `Match {matchId, gameId, region, ticketIds, serverId, state}`; per-game config `{mMin, mMax, Tmax, widening schedule}`.
4. **Architecture.** Client, then a connection gateway (WebSocket), then a ticket service (validates, fetches skill, writes the ticket to Redis), then a Kafka topic partitioned by (game, region), then matchmaker workers. Each (game, region) queue has exactly one owning worker, which removes double matching inside a queue. When a group forms, the worker allocates a server from the regional fleet, writes the match, and publishes `MatchFound`; the gateway pushes it, and clients connect to the game server with a signed join token.
5. **Deep dive: skill buckets with widening windows.** Each queue keeps 101 FIFO buckets, one per integer skill value (round a real-valued skill). A ticket that has waited $w$ accepts skills within

$$
\Delta(w)=\min\big(\Delta_0 + r\,w,\ \Delta_{\max}\big),\qquad m(w)=\max\big(m_{\min},\ m_{\max}-\lfloor w/\tau\rfloor\big)
$$

where $m(w)$ is the group size it accepts, shrinking with wait (and with low online counts). A group $G$ is valid when $\max_{i\in G}s_i-\min_{i\in G}s_i\le\min_{i\in G}\Delta(w_i)$, so every member accepts the spread. Every tick (a few hundred ms), take the oldest ticket as the anchor and read the buckets inside its window in skill order, which yields the candidates already sorted by skill in $O(B + c)$ for $B\le 101$ buckets and $c$ candidates. Slide a window of $m(w)$ consecutive candidates that contains the anchor, keeping the window's minimum $\Delta$ in a monotonic deque, until a window is valid: $O(c)$ more. At $T_{\max}$, start with $m_{\min}$, fill with bots if the game allows, or return a timeout. Treat a party as one ticket with a size and its maximum (or mean) skill. Track the skill spread per match and wait-time percentiles.
6. **Bursts and scale.** Kafka buffers joins and workers consume at their own pace; autoscale workers by partition lag. A single very hot game is split by skill range across workers, with tickets near a boundary visible to both neighbours. Players are routed to the lowest-latency region, and the search widens to neighbouring regions as wait grows, just as the skill window widens.
7. **Failure handling.** A worker crash triggers partition reassignment; the new owner rebuilds the queue from Redis or replays from the committed offset. Claim tickets with a compare-and-set from `WAITING` to `MATCHED` (a Redis Lua script or a transaction) so no ticket lands in two matches. Client heartbeats expire stale tickets. If server allocation fails, or a player does not accept in time, return the other tickets to the queue with their original `enqueuedAt` so they keep priority.

#### Delayed (scheduled) payment system

**Asked:** staff ([1136387](https://www.1point3acres.com/bbs/thread-1136387-1-1.html))

**Question.** Design a delayed payment system, where a payment is created now and executed at a scheduled later time. The requirements quoted in the thread: how to implement the delay is the core; support cancelling a scheduled payment; QPS in the thousands; find where the design bottlenecks at peak traffic and how to fix it; avoid duplicate operations.

**Answer.**

1. **Requirements and scale.** Create (payer, payee, amount, currency, `executeAt`), cancel before execution, query status; execute within seconds of `executeAt`; exactly-once effect; durable and auditable. At 2k creates per second there are about 170M payments per day, around 170 GB per day at 1 KB each, so the store is sharded with old rows tiered to cold storage. Executions spike far above the average because many payments are due at the same instant (midnight, the top of the hour, the first of the month).
2. **API.** `POST /scheduled-payments` with an `Idempotency-Key` header returns `{id, status}`; `DELETE /scheduled-payments/{id}` cancels; `GET /scheduled-payments/{id}`.
3. **Data model.** `scheduled_payment(id, payer_id, payee_id, amount_minor BIGINT, currency, execute_at, status, version, attempt_count, lease_until, idempotency_key UNIQUE per payer)`, where status is one of `SCHEDULED`, `CANCELLED`, `EXECUTING`, `SUCCEEDED`, `FAILED`; an index on `(shard_bucket, execute_at)` for due rows; an append-only double-entry ledger; an outbox table. Money is stored as integer minor units, never floats.
4. **Architecture.** API service, then a database sharded by payment id (the source of truth). Scheduler workers each own shards and every second claim due rows with `SELECT ... WHERE status = 'SCHEDULED' AND execute_at <= now() LIMIT n FOR UPDATE SKIP LOCKED`, or pre-load the next few minutes of due rows into an in-memory timing wheel or a Redis sorted set keyed by `execute_at`. Long delays (days or months) live only in the database; just the near-term slice sits in memory or a queue. Claimed payments go to an execution queue (Kafka), and executor workers call the payment processor and the ledger. Message brokers are a poor fit for the delay itself: Kafka has no per-message delay and SQS caps delays at 15 minutes.
5. **Deep dive: idempotency and exactly-once execution.** Queues and retries are at-least-once, so the exactly-once effect comes from four guards. (a) A state machine with conditional updates: `UPDATE ... SET status = 'EXECUTING', version = version + 1 WHERE id = ? AND status = 'SCHEDULED'`; only one worker gets a row count of 1. (b) The payment's idempotency key is passed to the processor, so a retry after a timeout cannot charge twice. (c) A unique constraint on `(payment_id, entry_type)` in the ledger. (d) A transactional outbox that publishes events atomically with the state change, with consumers deduplicating by event id. Cancellation uses the same compare-and-set, `WHERE status = 'SCHEDULED'`; zero rows updated means the payment was already cancelled (answer success, idempotently) or execution has already started, in which case the API answers "too late" (and a refund flow takes over). Execute and cancel therefore cannot both win.
6. **Peak bottlenecks and fixes.** Hot time buckets: shard the due index by `(minute, hash(id) mod k)` so many workers drain one instant in parallel, and pre-load the next minute. Processor rate limits: rate-limit the execution queue, apply backpressure, autoscale executors, and allow jitter inside an agreed tolerance. Database contention: batch claims and `SKIP LOCKED`. Status reads: replicas or a cache.
7. **Failure handling.** A worker that dies after claiming leaves the row `EXECUTING` with an expired `lease_until`; a sweeper re-queues it, which is safe because the idempotency key makes re-execution a no-op. If the processor times out with an unknown outcome, query it by the idempotency key before retrying. Retry transient failures with exponential backoff and jitter, cap the attempts, then mark `FAILED` and notify. Use database time, not worker clocks. A daily reconciliation compares the ledger with the processor's settlement reports.

#### To-do list with concurrent editing

**Asked:** level not stated ([927016](https://www.1point3acres.com/bbs/thread-927016-1-1.html))

**Question.** Design a to-do list that handles concurrent editing of the same list.

**Answer.**

1. **Requirements and scale.** Shared lists; items with text, a done flag, an order and a due date; create, edit, delete, reorder, complete; concurrent edits converge with no lost updates; changes sync in real time to other viewers; offline edits merge later. At 10M DAU and 50 operations per user per day there are 500M operations per day, about 6k per second on average and 20k at peak; the data itself is only tens of GB.
2. **API.** `GET /lists/{id}?sinceSeq=s` for a snapshot or the operations since `s`; `POST /lists/{id}/ops` with a batch of `{opId, clientId, baseSeq, type, itemId, field, value}`; a WebSocket subscription that pushes new operations.
3. **Data model.** `list(id, owner_id, seq)`; `item(list_id, item_id, text, done, position_key, deleted, per-field timestamps)`; `op_log(list_id, seq, op_id UNIQUE, client_id, payload)`; an ACL table.
4. **Architecture.** Clients, a WebSocket gateway, and a sync service that routes by `list_id` so one leader per list orders its operations, writes them durably to the op log and the database (sharded by `list_id`), then fans them out through pub/sub to the list's other subscribers.
5. **Deep dive: conflicts.** Operate per field, not per document, so edits to different items or fields never conflict. Then pick one of two designs.
   - *Server-ordered with versioning.* The leader assigns each operation a monotonically increasing `seq`. Same-field conflicts resolve last-writer-wins by `seq`, or by a Lamport timestamp `(counter, clientId)` when clients must agree without the server. The `opId` makes retries idempotent.
   - *CRDT, for offline-first clients.* Each field is an LWW register keyed by a Lamport timestamp; the set of items is an OR-Set with an explicit rule when a delete races an edit (delete wins with a tombstone, or the edit re-adds the item); order uses fractional indexing, so a move rewrites one position key placed between its neighbours' keys, with `clientId` as the tie-breaker. Merges are commutative, associative and idempotent, so every replica converges.
   - *Collaborative text inside an item*, if required: OT with a central server, where for concurrent operations $o_1, o_2$ on state $S$ the transform $T$ must satisfy $S\circ o_1\circ T(o_2,o_1)=S\circ o_2\circ T(o_1,o_2)$, or a sequence CRDT (RGA, or YATA as used by Yjs). For short to-do text, LWW is usually acceptable; confirm which is needed.
6. **Failure handling.** Offline clients queue operations with their `opId`s and replay them on reconnect from the last seen `seq`; the unique `op_id` removes duplicates. The server acknowledges only after the op log write is durable. A dropped socket resubscribes from the last seen `seq`. Periodic snapshots compact the op log, and tombstones are garbage-collected once every client has acknowledged past them.

#### Commuter train system for 5,000 employees

**Asked:** level not stated ([927016](https://www.1point3acres.com/bbs/thread-927016-1-1.html))

**Question.** Design a train system that helps a company transport 5,000 employees from San Francisco to San Mateo every morning.

**Answer.**

1. **Clarify.** Whether the ask is the physical service, the software that runs it, or both; the arrival window (say 8-10 am); where employees live and how they reach a boarding point; whether seats on an existing commuter line can be bought or trains chartered, with buses or shuttles for the first and last mile; budget; and the success metrics: on-time arrival rate, door-to-door time, load factor, cost per rider, satisfaction.
2. **Demand.** Home locations (or a survey) give an origin and desired-arrival-time matrix. Cluster homes into a few boarding stations (k-median), and size for the worst day, when all 5,000 ride.
3. **Capacity and schedule.** With $R$ riders, $C$ seats per train and an arrival window $W$, the service needs $n=\lceil R/C\rceil$ runs at headway $h=W/n$. For $R=5000$, $C=600$ and $W=2$ h: $n=9$, a train about every 13 minutes. By Little's law, trainsets in service $\approx$ runs per hour $\times$ cycle time; at 4.5 runs per hour and a 1.5 h round trip that is about 7 trainsets, plus a spare. Mix express runs from the largest station with stopping runs from the others, and keep bus bridges as a fallback.
4. **Software.** A booking app: reserve a seat on a departure, board with a badge or QR code, waitlist when a run is full. Data model `Trip(id, date, departure, route, capacity, seats_left)` and `Booking(user_id, trip_id, status)` with a unique `(user_id, date)` constraint; book with a conditional decrement, `UPDATE trip SET seats_left = seats_left - 1 WHERE id = ? AND seats_left > 0`, so a run is never overbooked. A scheduling job reshapes next week's timetable from bookings and load factors. Live tracking: vehicle GPS goes through pub/sub to ETAs and delay notifications in the app. Scale is tiny (5,000 bookings a day), so one relational database with a replica is enough; the hard parts are operations and reliability, not throughput.
5. **Failure handling.** A breakdown or delay triggers notifications, rebooking onto the next run and a spare trainset or bus bridge. No-shows release seats at a cutoff time. Monitor on-time performance and load factor per run and adjust headways.

#### Autocomplete

**Asked:** level not stated ([830336](https://www.1point3acres.com/bbs/thread-830336-1-1.html))

**Question.** Design an autocomplete (typeahead) system that suggests completions as the user types.

**Answer.**

1. **Requirements and scale.** Return the top 5-10 completions for a prefix, ranked by popularity (optionally personalized and fresh), with p99 under 100 ms end to end, so a few tens of ms on the server; highly available; suggestions refreshed hourly or daily, trending queries within minutes; offensive suggestions filtered. At 10M DAU, 10 searches each and about 6 debounced requests per search, there are 600M requests per day, about 7k QPS on average and 20k at peak.
2. **API.** `GET /autocomplete?prefix=ro&locale=en&k=10` returns `[{text, score}]`. Submitted searches are logged to feed the counts.
3. **Data model.** A query-count table `(query, locale, decayed_count, last_seen)` and versioned trie snapshots in blob storage.
4. **Architecture.** Query logs go to Kafka, then hourly batch aggregation with time decay plus a streaming counter for trending queries, then a trie builder that writes a snapshot. Read-only trie servers load the snapshot, replicated and sharded by prefix range, with ranges split by traffic rather than by alphabet. API servers sit in front, an edge cache serves the shortest prefixes, and the client caches responses and debounces keystrokes.
5. **Deep dive: trie with cached top-k.** Each node stores the ids of the top-$k$ completions in its subtree, so a lookup walks the prefix in $O(p)$ and returns the cached list in $O(k)$, independent of subtree size. Build bottom-up, merging the children's lists: $O(M\,k\log k)$ for $M$ nodes. Rebuild offline and swap snapshots atomically; for a hot incremental update, walk the query's path and fix each node's list in $O(L\,k)$. Score with a decayed count, $s(q)=\sum_t c_t(q)\,e^{-\lambda(T-t)}$, and personalize by re-ranking a longer cached list (say 50) with the user's history at the API layer. A simpler alternative is a key-value store from prefix to top-$k$ list for head prefixes, with a sorted query array and binary search for the rest; an FST compresses memory further. Add edit-distance-1 variants for typos.
6. **Failure handling.** Several replicas per shard behind health-checked load balancers. Sanity-check each new snapshot (size, top prefixes) and roll back to the previous version if it fails. If the trending pipeline is down, serve the last snapshot. On errors return an empty list instead of blocking search. Apply the blocklist at both build and serve time.

#### Elevator system

**Asked:** level not stated ([889847](https://www.1point3acres.com/bbs/thread-889847-1-1.html))

**Question.** Design an elevator system.

**Answer.**

1. **Requirements and scale.** A building with $F$ floors and $N$ cars; hall calls (floor and direction) and car calls (destination buttons); assign hall calls to cars, move cars, open and close doors; respect capacity; minimize average wait and travel time with bounded worst-case wait; safety (doors never open while moving, emergency stop, fire mode, maintenance mode). Event rates are at most dozens per second, so the hard part is real-time correctness and safety, not throughput.
2. **Interfaces.** Inputs `hallCall(floor, dir)` and `carCall(carId, floor)`; car events `arrived(carId, floor)`, `doorClosed`, `doorObstructed`, `overload`, `fault`; outputs `moveTo`, `openDoor`, display updates.
3. **Data model.** `Car {id, floor, direction, state, upStops: min-heap, downStops: max-heap, load}`; `HallCall {floor, dir, assignedCar, createdAt}`; a `Controller` holding the cars, pending hall calls and a pluggable `SchedulingStrategy` (strategy pattern).
4. **Car state machine.** `IDLE` goes to `MOVING_UP` or `MOVING_DOWN` when a stop exists; on arrival at a stop, `DOORS_OPENING`, `DOORS_OPEN` (dwell timer), `DOORS_CLOSING`, then back to moving or `IDLE`. An obstruction during `DOORS_CLOSING` returns to `DOORS_OPENING`. Any state goes to `EMERGENCY_STOP` on a fault, and `MAINTENANCE` takes the car out of service. Invariant: the motor runs only when the door is closed.
5. **Scheduling.** Within a car, use LOOK: keep moving in the current direction while stops lie ahead, serve them in order, reverse when none remain, and idle otherwise; adding a stop costs $O(\log S)$. Across cars, assign each hall call at floor $f$ in direction $d$ to the car with the lowest cost

$$
\text{cost}(c)=t_{\text{floor}}\cdot D_c(f,d)+t_{\text{stop}}\cdot k_c(f,d)
$$

where $D_c(f,d)$ is the number of floors car $c$ travels under LOOK, given its current stops, before it reaches $f$ heading in direction $d$, and $k_c(f,d)$ is the number of stops it makes on the way. Simulating each car's sweep is $O(S)$, so one assignment is $O(N\cdot S)$. Re-evaluate unserved assignments periodically and give old calls priority so the worst-case wait stays bounded. Peak modes: park idle cars at the lobby in the morning up-peak and split the building into zones; destination dispatch (riders enter the destination at the lobby and are grouped by it) is a stronger variant.
6. **Failure handling.** A faulty car is marked out of service and its hall calls are reassigned. If the central controller crashes, each car finishes its own car calls and stops at the nearest floor, and hall-call state is rebuilt from the button states. Disagreeing sensors trigger a safe stop. Overload keeps the doors open. Fire mode sends cars to the ground floor and disables hall calls; on power loss, the brake engages and battery power brings the car to the nearest floor.

#### Real-time game recommendation serving system

**Asked:** IC6 (principal) ([995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html)), level not stated ([1137143](https://www.1point3acres.com/bbs/thread-1137143-1-1.html))

**Question.** Design a real-time game recommendation system for millions of games and tens of millions of users, focusing on the system rather than the model; in essence, serve each user's top-k games in real time at that scale [1137143](https://www.1point3acres.com/bbs/thread-1137143-1-1.html). A variant asks for the systems side of game recommendation, candidate retrieval and ranking, where adding caching covered most of what was expected [995525](https://www.1point3acres.com/bbs/thread-995525-1-1.html).

**Answer.**

1. **Requirements and scale.** Assume 5M games, 50M DAU and 200M users in total. At 5 home-page loads per user per day there are 250M requests per day, about 3k QPS on average and 10k at peak, with p99 under 200 ms. User actions should show within seconds to minutes and new games within minutes. Events: 50M users times 100 events per day is 5B per day, about 60k per second. Game embeddings: $5\text{M}\times 64\times 4\,\text{B}\approx 1.3$ GB, which fits in RAM on every ANN replica. User embeddings: $200\text{M}\times 64\times 4\,\text{B}\approx 51$ GB, in a sharded key-value store.
2. **API.** `GET /recommendations?userId=&surface=home&k=50&cursor=` returns a ranked list with a request id for logging; `POST /events` takes batched impressions, joins and leaves.
3. **Data model.** An online feature store keyed by `user:{id}` (embedding, recent games, genre affinities) and `game:{id}` (CCU, joins in the last 5 minutes, like ratio, maturity); a feed cache `feed:{userId}` holding a ranked list and its timestamp; offline, event logs in a data lake, a model registry and ANN index snapshots.
4. **Architecture.**
   - *Online path.* Gateway, then an orchestrator that fans out in parallel, each call with a timeout, to the candidate sources: an ANN service (HNSW over game embeddings), continue-playing, friends-playing (from the presence service), trending (a precomputed top-k per region and age band, refreshed every minute) and a new-game pool. It then merges and deduplicates 1-2k candidates, hydrates features with batched multi-gets, scores them in a ranking service with batched inference, applies re-ranking and filters, responds, writes the cache, and logs the served features.
   - *Streaming path.* Client events go through Kafka to Flink (windowed counts, CCU, trending) and on to the feature store.
   - *Offline path.* Daily training from the data lake, the model registry, canary deployment, a nightly ANN rebuild, and incremental index insertion of new games.
5. **Deep dive: top-k at scale.**
   - *ANN, not brute force.* Exact scoring is $O(N d)=5\text{M}\times 64\approx 3.2\times 10^{8}$ multiply-adds per request, about $3\times 10^{12}$ per second at 10k QPS. HNSW is roughly $O(\log N)$ per query at 95%+ recall. Shard the index across $S$ shards, query them in parallel, and merge the per-shard top-$k$ lists with a heap in $O(Sk\log k)$; add replicas for QPS.
   - *Trending top-k.* Count per game and region on shards partitioned by game id, keep a local top-$k$ heap per shard, and merge those every few seconds. Because each game's full count lives on one shard, the top-$k$ of the union of local top-$k$ lists is exactly the global top-$k$. A count-min sketch plus a heap bounds memory when there are too many keys.
   - *Caching.* Precompute feeds for active users at session start and cache them with a short TTL, invalidating on significant events (such as finishing a game). For freshness, re-rank the cached candidates with fresh features instead of recomputing from scratch, and insert newly hot or new games into a cached ranked list at the position found by binary search on score. Non-personalized lists (trending by region) can be cached at the edge.
6. **Failure handling.** If a candidate source times out, return what the other sources produced. If the ranker is down, fall back to retrieval-score order or the cached feed. A feature-store miss uses defaults. A severe outage serves popular-by-region lists from the edge cache. Add circuit breakers and load shedding at the gateway, model rollback through the registry, and replicas across availability zones.

#### Payment backend architecture

**Asked:** level not stated ([969361](https://www.1point3acres.com/bbs/thread-969361-1-1.html))

**Question.** Describe the architecture of a payment backend.

**Answer.**

1. **Requirements and scale.** Accept payments (cards, wallets, app stores), credit the purchased item or virtual currency, and handle refunds, chargebacks and payouts; correctness over latency; exactly-once crediting; a full audit trail; PCI DSS scope kept small; multiple currencies. Volume is modest (about 1k payments per second at a holiday peak), but the ledger is append-only and grows forever.
2. **API.** `POST /payments {userId, amount, currency, method, productId}` with an idempotency key creates a payment intent; `POST /payments/{id}/confirm`; a webhook endpoint for processor callbacks; `POST /refunds`; `GET /payments/{id}`.
3. **Data model.** `payment_intent(id, user_id, amount_minor, currency, status, processor_ref, idempotency_key UNIQUE)`; a double-entry `ledger_entry(id, txn_id, account_id, amount_minor signed, currency, created_at)` whose entries satisfy $\sum_{e\in\text{txn}} a_e = 0$ for every transaction and every currency in it (a currency conversion books through FX accounts); accounts for user wallets, revenue, processor clearing and refunds; wallet balances as a projection of the ledger; an outbox.
4. **Architecture.** Gateway (auth, rate limits), then a payment service running the state machine `CREATED`, `AUTHORIZED`, `CAPTURED`, `SETTLED`, with `FAILED`, `REFUNDED` and `CHARGEBACK` branches. A risk and fraud check sits in the authorization path. A processor adapter works with tokenized cards: raw card numbers live only in the processor's vault, which keeps the backend out of most PCI scope. Behind it, a ledger service on a strongly consistent relational database, a wallet service, and fulfillment (granting the item or currency) driven by outbox events, plus reconciliation jobs against the processor's settlement files.
5. **Deep dive: correctness.** Idempotency keys end to end; state transitions as conditional updates; processor webhooks are at-least-once and can arrive out of order, so deduplicate by event id and apply only valid transitions; the ledger entries and the status change commit in one database transaction. A virtual-currency debit must never go negative: `UPDATE wallet SET balance = balance - x WHERE id = ? AND balance >= x`.
6. **Failure handling.** On a processor timeout the status is unknown, so query by idempotency key before retrying. Several processors give routing failover. A synchronous replica protects the primary database. A failed fulfillment retries from the outbox; if it fails for good, a compensating refund runs (a saga). Daily reconciliation raises alerts on any mismatch.

#### Search-index backend architecture

**Asked:** level not stated ([969361](https://www.1point3acres.com/bbs/thread-969361-1-1.html))

**Question.** Describe the architecture of a backend index system similar to a search index.

**Answer.**

1. **Requirements and scale.** Index documents (for example games, users or catalog items) with text and attributes; keyword queries with filters and ranking at p99 under 100 ms; updates searchable within seconds to minutes. Assume 50M documents and 10k QPS; a compressed inverted index takes a fraction of the raw text size.
2. **API.** `PUT /docs/{id}` (upsert with a version), `DELETE /docs/{id}`, `GET /search?q=&filters=&sort=&k=&cursor=`.
3. **Data model.** The source of truth is the primary database. The inverted index maps each term to a postings list of `(docId, termFreq, positions)` sorted by `docId` and compressed (delta plus variable-byte encoding); a doc store holds the stored fields; each document carries a version.
4. **Architecture.** Writes go to the primary database; change data capture streams them to Kafka; indexer workers tokenize, normalize and compute features, then write to index shards. Shard by document (each shard indexes a subset of documents, and a query scatters to all shards and gathers their top-$k$), not by term (hot terms and multi-term intersections make term sharding costly). Replicas serve QPS and availability. Each shard keeps Lucene-style immutable segments: new documents fill an in-memory buffer that is flushed to a new segment about every second (near-real-time refresh), deletes are a tombstone bitmap, and background merges compact segments. A query intersects postings with skip lists, scores with BM25,

$$
\text{BM25}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f_{t,d}\,(k_1+1)}{f_{t,d}+k_1\big(1-b+b\,\frac{|d|}{\text{avgdl}}\big)}
$$

keeps a top-$k$ heap per shard, and the coordinator merges the shard results; an optional ML model re-ranks the top hundred.
5. **Failure handling.** Replica failover. The index is derived data, so it can be rebuilt from the database and the change log. A write whose version is not newer than the stored one is ignored, so replayed change events are idempotent and out-of-order events cannot overwrite newer data. A slow shard hits a timeout and the response is returned as partial with a flag; hedged requests to another replica cut tail latency. A full reindex builds a new index in the background and switches over by an alias swap.

## 1.4 Behavioral (BQ)

The questions are grouped by type: motivation (1.4.1 to 1.4.3), background and scope (1.4.4 to 1.4.10), your projects (1.4.11 to 1.4.16), and judgment, people and self-awareness (1.4.17 to 1.4.20). Each answer says what the interviewer is probing, gives a STAR outline (Situation, Task, Action, Result) and a short example for a PhD researcher in video and multimodal ML moving to industry. Replace the bracketed parts with your own facts.

### 1.4.1 Why Roblox?

**Asked:** senior ([1186156](https://www.1point3acres.com/bbs/thread-1186156-1-1.html)), director-level IC ([1126252](https://www.1point3acres.com/bbs/thread-1126252-1-1.html))

**Answer.** The interviewer is checking that your interest is specific and informed (you know the product, its users and its ML problems) and that it will last. "A big platform with cool tech" sounds like a fallback application.

- **Situation:** where you are now and what you work on.
- **Task:** what you want next: problems at scale, with real users in the loop, that need your skills.
- **Action:** what you did to check the fit: played experiences, looked at how recommendations, safety or creator tools work, read the job description closely.
- **Result:** two or three concrete links between your skills and their problems, and what you want to learn there.

*Example.* "My PhD is on [video-language understanding]: learning from video, audio and text together. At Roblox that kind of data is the core product, not a side feature: a huge catalog of user-made 3D experiences, plus avatars, voice and chat. Before applying I played [two or three experiences] and noticed how much depends on helping players find experiences and on keeping content safe. Both are multimodal ranking and classification problems at a scale no academic dataset reaches. I want my models judged by player engagement and safety metrics rather than by a benchmark, and this role owns exactly that."

### 1.4.2 Why do you want to leave your current job?

**Asked:** senior ([1186156](https://www.1point3acres.com/bbs/thread-1186156-1-1.html), [1110219](https://www.1point3acres.com/bbs/thread-1110219-1-1.html))

**Answer.** The interviewer wants a pull, not a push: you are moving toward something, you do not badmouth your current place, and the timing makes sense, so answer "why now" even if it is not asked. For a PhD the real question is "why industry, and why now rather than a postdoc?"

- **Situation:** your current role or stage (finishing the PhD, a research position).
- **Task:** what is missing that you now need: production impact, scale, feedback from real users.
- **Action:** what you have already done toward it: an internship that shipped something, engineering skills you built, production ML you studied.
- **Result:** why this role closes that gap, and why the timing is right.

*Example.* "I'm finishing a PhD on [multimodal video models]. In [an internship / an industry collaboration] I saw a model I trained [run on real traffic], and that feedback loop was faster and more honest than paper reviews. I have done the full research cycle, [first-author papers at venues], so the next thing I want is to own a model end to end in production. The timing fits because my thesis is [wrapping up], and I would rather learn production ML now than after a postdoc."

### 1.4.3 Why are you interested in my team's area?

**Asked:** level not stated ([969361](https://www.1point3acres.com/bbs/thread-969361-1-1.html))

**Answer.** The interviewer is testing whether you looked into this team, not just the company, and whether you will stay engaged with its problems. If the area is outside your expertise (payments, say), answer through the problems that transfer rather than pretending to be an expert.

- **Situation:** what you know about the team's area.
- **Task:** the problem in it you find compelling.
- **Action:** evidence that you engaged: related work you did, what you read, what you asked people who know the team.
- **Result:** what you would bring and what you would learn.

*Example.* "From the job description, your team owns [game recommendation ranking]. What draws me is that it learns from sparse, delayed signals: a player's choice says little until you see whether they come back. In my research I modeled [long videos where the label arrives only at the end], so I know how to work with weak, delayed supervision. I would bring content representations that help with cold start, and I would learn the online experimentation and product side that research never gave me."

### 1.4.4 Walk me through your background and experience.

**Asked:** senior ([1186156](https://www.1point3acres.com/bbs/thread-1186156-1-1.html), [1110219](https://www.1point3acres.com/bbs/thread-1110219-1-1.html), [761789](https://www.1point3acres.com/bbs/thread-761789-1-1.html), [1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html)), level not stated ([1130525](https://www.1point3acres.com/bbs/thread-1130525-1-1.html), [886138](https://www.1point3acres.com/bbs/thread-886138-1-1.html), [919123](https://www.1point3acres.com/bbs/thread-919123-1-1.html))

**Answer.** The interviewer is judging communication (a short story, not a reading of your CV), relevance to the job description, and the scope you have owned, and is picking what to ask next. The listener may not be an ML specialist, so clarity counts as much as depth.

- **Situation:** one line on who you are now.
- **Task:** the through-line: the problem you keep coming back to.
- **Action:** two or three milestones, each with what you did and one number.
- **Result:** why that path leads to this role. End on a hook the interviewer can pick up.

*Example.* "I'm a PhD candidate in [lab, university], working on multimodal learning for video. The through-line is getting models to understand [what happens in long videos] from cheap supervision. First, I built [a self-supervised video-text pretraining method] that [improved retrieval by X points]. Second, in an internship at [company] I [adapted a model for content tagging and put it into a batch pipeline]. Now I'm [working on making video models cheaper to run]. I'm looking for an MLE role where content understanding plus efficient inference feed recommendations and safety."

### 1.4.5 What do you work on now, what are your strengths, and which projects interest you?

**Asked:** senior ([1110219](https://www.1point3acres.com/bbs/thread-1110219-1-1.html))

**Answer.** The interviewer wants your current scope in concrete terms, strengths you can back with evidence, and interests that match work the company actually has. Name two strengths, not five, and tie each to something you did.

- **Situation:** your current work in one sentence.
- **Task:** your role in it.
- **Action:** two strengths, each shown through something you did.
- **Result:** the proof (a number, something others adopted). Then name two kinds of project you want, linked to the company's problems.

*Example.* "Right now I'm [training a video-language model that finds the moment in a video matching a text query]. My first strength is turning a fuzzy research question into a measurable setup: [I built the evaluation set my group now uses]. My second is making large training runs work: [multi-GPU training and fast data loading for video]. The projects that interest me most are content understanding for recommendations and multimodal safety classification, because both use the video, audio and text signals I already work with."

### 1.4.6 What computer vision and recommender-system experience do you have?

**Asked:** senior ([1110219](https://www.1point3acres.com/bbs/thread-1110219-1-1.html))

**Answer.** The interviewer is gauging hands-on depth in each area, whether you built things that were used or only benchmarked them, and how honestly you rate your gaps. Keep research and production experience clearly separate. If recommender systems are thin, connect them to what you know (retrieval, embeddings, ranking metrics) and show you learned the basics on purpose.

- **Situation:** where each kind of experience comes from.
- **Task:** what you were responsible for.
- **Action:** the model, data, scale and tools.
- **Result:** the metric and where the model ran.

*Example.* "Computer vision is my core: [video action recognition and video-text retrieval], from data pipelines to [distributed training on N GPUs], with [result X on benchmark Y]. Recommender systems are newer for me. My [video-text retrieval] work uses the same two-tower contrastive setup as candidate generation, and I [built a content-based recommender on a public dataset] to learn the industry side: negative sampling, recall@k and NDCG offline, and why offline gains do not always show up online."

### 1.4.7 Have you built a state-of-the-art model?

**Asked:** senior ([1110219](https://www.1point3acres.com/bbs/thread-1110219-1-1.html))

**Answer.** The interviewer is checking that your research claims are real and verifiable (have your Google Scholar link ready; it was requested), and that you know what "state of the art" means: one benchmark, a comparable setup, and limits that matter in industry. Make a precise claim, say what you did yourself, then turn to practical value. If you have no state-of-the-art result, say so and give your strongest result with its comparison.

- **Situation:** the task and benchmark.
- **Task:** the gap in existing methods.
- **Action:** your key idea and your part in the work.
- **Result:** the margin, the venue, whether it held up elsewhere, and the caveats.

*Example.* "Yes, on [benchmark] for [text-to-video retrieval], published at [venue]. Existing methods [ignored the order of events]; I proposed [a temporal alignment loss] and [ran all the ablations]. It beat the previous best by [X points of recall@1] with the same backbone. What matters for industry is that the gain held when I [changed the backbone and tested on a second dataset], and I also [measured the added inference cost, which was small]."

### 1.4.8 What scale of dataset have you trained models on?

**Asked:** director-level IC ([1126252](https://www.1point3acres.com/bbs/thread-1126252-1-1.html))

**Answer.** The interviewer wants to know whether you have hit the problems that only appear at scale: storage formats, data-loading throughput, distributed training, deduplication, label noise and cost. A number alone is not an answer; say what broke and what you did about it.

- **Situation:** the dataset: size (samples, hours, terabytes) and modalities.
- **Task:** what you were training (pretraining, fine-tuning) and for what.
- **Action:** how you made it work: sharded storage, streaming loaders, decoding cost, mixed precision, the distributed setup, filtering.
- **Result:** throughput, training time, model quality and cost.

*Example.* "The largest was [N million web video clips with captions, about X TB] for pretraining. The bottleneck was [video decoding], not the GPUs, [which sat partly idle]. I [pre-extracted frames at a fixed rate into sharded files and took decoding out of the training loop], which [raised throughput by X times]. I also [removed near-duplicates and low-quality captions with an embedding-similarity filter], which [improved downstream retrieval at the same compute]."

### 1.4.9 Describe your experience leading projects and the scope of your past work.

**Asked:** senior ([1186156](https://www.1point3acres.com/bbs/thread-1186156-1-1.html)), level not stated ([968319](https://www.1point3acres.com/bbs/thread-968319-1-1.html))

**Answer.** The interviewer is calibrating your level: how big a problem you can own, how ambiguous it was at the start, how many people and teams depended on it, and what you did versus what the team did. For a PhD, leading means setting the research direction, mentoring students, coordinating collaborators, and owning a shared codebase or dataset.

- **Situation:** the project and why it mattered.
- **Task:** what you were accountable for.
- **Action:** how you led: set direction, split the work, unblocked people, made the trade-off calls.
- **Result:** the outcome and what grew from it (adoption by others, follow-up work).

*Example.* "The biggest project I led was [a multimodal video benchmark] in my lab, with [three junior students and two outside collaborators]. I defined the task and metrics, wrote the data specification and set up the shared training code. I ran weekly reviews, and when annotation quality slipped I [rewrote the labeling guide and added an agreement check]. It became [a paper at venue], and [other groups now use the dataset]. My scope ran from problem definition to release. In industry I would expect to start with a narrower area and grow into owning a model surface."

### 1.4.10 Describe how an ML system evolves over several quarters, from research into production.

**Asked:** director-level IC ([1126252](https://www.1point3acres.com/bbs/thread-1126252-1-1.html))

**Answer.** The interviewer is testing whether you understand the whole lifecycle and can tell a long story: prototype, offline gains, production constraints (latency, cost), launch, iteration, and the decision you made at each stage. Tell it through a system you worked on. A PhD without a full production story should use an internship or a research project that someone adopted, say plainly which stages they saw, and describe how they would plan the stages they did not see.

- **Situation:** the starting problem and the baseline.
- **Task:** the target metric and the constraints.
- **Action:** the phases: prototype and offline evaluation; production work (distillation, feature parity, serving); launch behind an A/B test; iteration driven by monitoring.
- **Result:** the impact and the lessons.

*Example.* "In my internship the team wanted better [video tags] for [search]. First I [fine-tuned a pretrained video-text model] and beat the existing [tag classifier] offline by [X]. It was too slow to serve, so I [distilled it into a smaller model and computed video embeddings once, at upload]. Then [an online test showed a gain in search click-through]. I left before the last stage, but my handoff plan covered [drift monitoring and retraining on new content]. What I would change: agree on the online metric before building the offline one."

### 1.4.11 Which project are you proudest of, and why?

**Asked:** research intern ([1047347](https://www.1point3acres.com/bbs/thread-1047347-1-1.html))

**Answer.** The interviewer is learning what you value (impact, craft, learning, people) and what your role was. Most of the signal is in the "why" follow-up: "it got into a top venue" is the weakest answer. Pick a project with a hard obstacle you overcame, a clear personal contribution and a real outcome.

- **Situation and task:** the problem and your goal.
- **Action:** the hard part and the decision only you made.
- **Result:** what it achieved.
- **Why proud:** the obstacle and what it taught you, tied to something the team cares about.

*Example.* "I'm proudest of [my work on long-video question answering]. Standard models [only looked at a few seconds of frames], so they failed on questions about events far apart. I proposed [a memory module that picks the relevant clips]. The hard part was that [training was unstable]; I traced it to [the clip-selection step] and [replaced it with a smooth relaxation]. It [improved accuracy by X points]. I'm proud of it not for the number but because I debugged a failure no one in the lab had seen, and [another group built on the code]."

### 1.4.12 Walk me through one of your projects and its hardest part.

**Asked:** senior ([1128678](https://www.1point3acres.com/bbs/thread-1128678-1-1.html)), level not stated ([886138](https://www.1point3acres.com/bbs/thread-886138-1-1.html), [919123](https://www.1point3acres.com/bbs/thread-919123-1-1.html), [1014514](https://www.1point3acres.com/bbs/thread-1014514-1-1.html))

**Answer.** The interviewer is checking that you can explain a problem, your role and a real difficulty clearly, possibly to a non-specialist, and how you reasoned through it. Lead with the problem and the impact, and go deeper when asked. The hardest part should be a real technical or organizational obstacle, not "the deadline was tight".

- **Situation:** the problem and why it matters, in one or two sentences.
- **Task:** your goal and your role.
- **Action:** the approach, then the hardest part: what went wrong, the options you weighed, what you chose.
- **Result:** the outcome and what you learned.

*Example.* "I built [a model that finds the moment in a video matching a text query]. I owned the model and the evaluation. The hardest part was that [the training timestamps were noisy, often off by several seconds]. I compared [filtering the bad labels, label smoothing, and training with soft boundaries]; [soft boundaries] won because it kept all the data. The result was [X points better than the baseline], and I learned to look at the data before tuning the model."

### 1.4.13 Tell me about a complex system you built.

**Asked:** staff ([1136387](https://www.1point3acres.com/bbs/thread-1136387-1-1.html))

**Answer.** The interviewer is looking for system-level ownership, not just a model: components, interfaces, data flow, scale, failure modes, trade-offs, and how the system changed over time. Expect follow-ups on bottlenecks and on what you would redesign.

- **Situation:** what the system had to do, and for whom.
- **Task:** the requirements (throughput, latency, cost, reliability) and your role.
- **Action:** the architecture, the two or three key design decisions with the alternatives you rejected, and how you tested and monitored it.
- **Result:** the measured outcome, what broke later and how you fixed it.

*Example.* "I built the [training and evaluation pipeline for video-language models] that [my lab] runs on. It takes [raw videos, transcripts and captions] and produces [trained checkpoints and a benchmark report]. Key decisions: [store pre-decoded frames in sharded files] instead of decoding on the fly, trading storage for [X times the throughput]; [a config-driven experiment runner] so every run is reproducible; and [an evaluation job that scores every checkpoint on a fixed suite]. It broke when [the dataset index no longer fit in memory], and I [moved to one index per shard]. It now carries [N people's experiments]."

### 1.4.14 What was your role in this project?

**Asked:** senior ([1177250](https://www.1point3acres.com/bbs/thread-1177250-1-1.html))

**Answer.** Asked after you present a project, this separates your contribution from the team's or your co-authors'. Be precise: what you owned end to end, what you contributed to, what others did. Say "I" for your parts and name the others' parts plainly.

- **Situation:** the project and the team.
- **Task:** your responsibility.
- **Action:** the specific decisions and work that were yours.
- **Result:** what would not exist without you.

*Example.* "It was a [four-author] paper. I proposed the core idea, [the cross-modal alignment loss], wrote the training code and ran every ablation. [A co-author] built the data pipeline, and [my advisor] shaped the framing. [The second set of experiments] came from a collaborator; I integrated it. In short, the method is mine; the paper also needed their data pipeline and [second set of experiments]."

### 1.4.15 Tell me about your projects. How do you see them now?

**Asked:** intern ([1063311](https://www.1point3acres.com/bbs/thread-1063311-1-1.html))

**Answer.** This is a reflective question, not a technical grilling. The interviewer wants to see whether you can judge your own work: its value, its limits, the decisions you would change and what you would do next.

- **Situation:** the project in one sentence.
- **Task:** what you set out to do.
- **Action:** what you did, and in hindsight which decisions you would keep or change.
- **Result:** what it achieved, where it falls short, and what you would do next with industry data and compute.

*Example.* "My [video captioning] project did well on benchmarks, [X better than the baseline], but looking back I optimized [a metric that rewards generic captions]. If I did it again I would [add a human preference check early]. What I would keep is [the data cleaning step], which mattered more than the architecture. With production data, I would test whether the captions actually help [search or recommendations], which is the real measure of their value."

### 1.4.16 How would your research apply to Roblox?

**Asked:** PhD new grad ([1161498](https://www.1point3acres.com/bbs/thread-1161498-1-1.html))

**Answer.** Asked about a paper you present, this tests whether you can carry research over to the company's products and constraints. Prepare one concrete mapping: the product problem, the data, what the model would do, which metric should move, and what would have to change (latency, scale, labels).

- **Situation:** the Roblox problem.
- **Task:** what is hard about it today (frame it as your hypothesis).
- **Action:** how your method fits and what you would adapt.
- **Result:** the metric you would target, how you would test it, and the risks.

*Example.* "My work [learns joint embeddings of video and text]. At Roblox, one use is [representing experiences by their thumbnails, screenshots or gameplay clips plus their descriptions]. That helps cold start in recommendations: a new experience with no play history still gets a meaningful embedding. I would adapt it by [using short clips or images instead of long videos] to keep the cost down, and test it by [adding it as a candidate source and measuring engagement with new experiences in an A/B test]. The risk is that [visual similarity does not match what players enjoy], so I would combine it with behavioral signals."

### 1.4.17 How do you understand risk?

**Asked:** research intern ([1047347](https://www.1point3acres.com/bbs/thread-1047347-1-1.html))

**Answer.** The interviewer is probing judgment: whether you treat risk as something to manage rather than avoid, and how you weigh upside against downside, reversibility and blast radius. Give a short working definition, then back it with a quick story.

- **Definition:** the chance that a decision costs more than it returns, weighted by how hard it is to undo. Reversible bets can be taken fast; irreversible ones need more evidence. You manage risk with small bets, early signals and fallbacks.
- **Situation and task:** a decision with an uncertain payoff.
- **Action:** how you limited the downside (a time box, a kill criterion, a fallback).
- **Result:** what happened, and how you would apply the same thinking in production.

*Example.* "To me, risk is the chance a decision costs more than it returns, weighted by how hard it is to undo. A new research direction is a reversible bet if I time-box it. When I started [an audio-visual learning project], I [set a two-week milestone: if a simple version did not beat the baseline, I would drop it]. It did, so I continued. In production the costly risks are the irreversible ones, such as [shipping a model that harms users], which is why staged rollouts and monitoring matter."

### 1.4.18 Tell me about the biggest risk you have taken.

**Asked:** level not stated ([927016](https://www.1point3acres.com/bbs/thread-927016-1-1.html))

**Answer.** The interviewer wants courage plus judgment. The risk should be real (a meaningful downside), deliberate (you saw it and chose it) and managed (you reduced the downside and had a fallback). The reasoning matters more than the outcome; if it failed, say what you learned.

- **Situation:** the context.
- **Task:** the decision and what was at stake.
- **Action:** how you assessed the risk and reduced it.
- **Result:** what happened, and the lesson.

*Example.* "[Midway through my PhD] I dropped [a line of incremental image work that was producing results] to move to [video], where I had nothing yet. The downside was [a long stretch without a paper]. I reduced it by [reproducing a strong baseline first, so I would at least have a solid codebase] and by [agreeing on checkpoints with my advisor]. [The first idea failed; the second, built on what that failure showed,] became [the main paper of my thesis]. The lesson: take risks on direction, not on execution."

### 1.4.19 How do you work with a low performer?

**Asked:** principal ([1132251](https://www.1point3acres.com/bbs/thread-1132251-1-1.html))

**Answer.** Do not answer it as a mentoring question: a candidate who answered about coaching junior colleagues was told the question is about working with a low performer. The interviewer wants two things at once, so cover both. Delivery: you protect the outcome. People: you help the person improve, directly and with empathy, and you escalate when that does not work.

- **Situation:** the collaboration and how the gap showed.
- **Task:** what you were responsible for delivering, and whether you were a peer or a lead.
- **Action:** diagnose first (unclear expectations, missing skills, motivation, personal issues); talk in private with specifics; adjust (clearer scope, pairing, check-ins); protect the critical path; escalate with facts if nothing changes.
- **Result:** the outcome for the project and for the person.

*Example.* "On [a joint paper], a collaborator kept missing [experiment deadlines] close to [the submission date]. I asked them one on one what was blocking them; it turned out [they did not know the training codebase and were embarrassed to ask]. I [paired with them for two sessions and split their task into smaller pieces with check-ins], and I took [the critical experiment] myself to protect the deadline. Their [ablation section] came in on time, and they [later ran a follow-up project on their own]. If nothing had changed, I would have raised it early with [my advisor], with specifics."

### 1.4.20 What are your weaknesses?

**Asked:** staff ([1136387](https://www.1point3acres.com/bbs/thread-1136387-1-1.html))

**Question.** Pointed behavioral questions designed to expose your weaknesses; the exact wording was not reported.

**Answer.** The interviewer is testing self-awareness, honesty and coachability, and whether your weakness is a risk for this role. Likely forms (not reported wording): your biggest weakness, a time you failed, the hardest feedback you received, what peers would criticize, what you would do differently. Pick a real weakness that is relevant but not disqualifying, show that it cost something, and show what you changed. Avoid disguised strengths ("I work too hard"), stay calm under follow-ups, and keep a second example ready.

- **Situation:** when the weakness showed.
- **Task:** what was at stake.
- **Action:** what you changed.
- **Result:** evidence of improvement, and what still needs work.

*Example.* "My weakness has been [polishing a model past the point of usefulness]. In research I would [chase another half point on a benchmark] when a simpler model was enough, and on [a shared project] that [delayed a result my collaborators needed by weeks]. Since then I [set a 'good enough' target and a deadline before I start, and share rough results early]. On my latest project [a baseline I shared in the first week changed our direction]. I still have to watch for it, especially when the problem is interesting."
