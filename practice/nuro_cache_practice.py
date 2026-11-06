"""Round 1 skeleton: single-flight cache for expensive misses.

Fill in every `...` body. Run: python3 practice/check.py cache

Contract (white-box names the tests rely on):
- self.values: dict key -> cached value
- self.key_locks: dict key -> threading.Lock  (one lock per key)
- self.lock: threading.Lock guarding both dicts (held only briefly)
- get(key, load): return the cached value, or call load(key) exactly once
  per key even when many threads miss at the same time. A failed load
  caches nothing; the next caller retries. Loads of different keys must
  not block each other.
Keep the BEGIN/END markers: one test executes both regions standalone.
"""

# BEGIN nuro-cache-interview
import threading


class Cache:
    def __init__(self):
        ...

    def get(self, key, load):
        # 1. Under self.lock: get-or-create this key's lock.
        # 2. Under key_lock: re-check values (someone may have filled it while
        #    you waited); on miss call load(key) with NO lock held except
        #    key_lock; then store under self.lock.
        # Lock order is always key_lock -> self.lock, never the reverse.
        ...
# END nuro-cache-interview


# BEGIN nuro-cache-demo
if __name__ == "__main__":
    import time

    start = time.monotonic()
    cache = Cache()  # All three callers share this one cache.

    def load(key):
        time.sleep(0.5)  # Stand in for a slow I/O call.
        return f"value for {key}"

    def request(key):
        value = cache.get(key, load)
        print(key, value)

    workers = [
        threading.Thread(target=request, args=("video_a",)),
        threading.Thread(target=request, args=("video_a",)),
        threading.Thread(target=request, args=("video_b",)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    request("video_a")  # A cache hit: no additional load.
# END nuro-cache-demo
