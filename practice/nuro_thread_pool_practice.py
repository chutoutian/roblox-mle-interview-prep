"""Round 1 skeleton: fixed-size thread pool.

Fill in every `...` body. Run: python3 practice/check.py pool

Contract (white-box names the tests rely on):
- self.ready: queue.Queue (unbounded FIFO) shared by all workers
- self.lock: threading.Lock guarding the lifecycle flag
- self.closed: bool
- self.workers: list of num_workers threads, started in __init__,
  daemon=False, each with a unique name
- submit(fn): non-callable -> TypeError; after shutdown -> RuntimeError;
  returns None and never blocks on the task.
- shutdown(): rejects new work, lets every accepted task finish, joins all
  workers; safe to call twice; raises RuntimeError if called from a worker.
Keep the module-level `import traceback`: a test patches traceback.print_exc.
"""

# BEGIN nuro-pool-interview
import queue
import threading
import traceback


class ThreadPool:
    def __init__(self, num_workers=4):
        # num_workers must be a positive int (bool is not an int here) -> ValueError.
        ...

    def submit(self, fn):
        # Hint: checking `closed` and putting on the queue must happen under
        # the same lock, or shutdown can slip its exit markers in between.
        ...

    def _work(self):
        # Loop: get() blocks until work arrives (no polling); a None item
        # means "this worker exits". Run fn() outside the lock; catch
        # Exception and traceback.print_exc() so one bad task does not kill
        # the worker.
        ...

    def shutdown(self):
        # Under the lock: if not already closed, mark closed and enqueue one
        # None per worker (they land after every accepted task). Then join
        # every worker OUTSIDE the lock, on every call.
        ...
# END nuro-pool-interview


# BEGIN nuro-pool-demo
if __name__ == "__main__":
    import time
    from functools import partial

    started = time.monotonic()

    def log(message):
        elapsed = time.monotonic() - started
        worker = threading.current_thread().name
        print(f"+{elapsed:.3f}s | {worker} | {message}", flush=True)

    def work(job_id):
        log(f"job {job_id} starts")
        time.sleep(0.1)  # Simulate an I/O wait.
        log(f"job {job_id} finishes")

    pool = ThreadPool(num_workers=2)
    try:
        for job_id in range(4):
            pool.submit(partial(work, job_id))
        log("caller continues while workers run")
        log("shutdown waits for queued and running jobs")
    finally:
        pool.shutdown()
    log("shutdown returned; all jobs finished")
# END nuro-pool-demo
