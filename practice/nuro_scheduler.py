"""Round 1 skeleton: background scheduler with cancel-by-id.

Fill in every `...` body. Run: python3 practice/check.py scheduler

Contract (the tests are white-box and rely on these names):
- self._jobs: dict job_id -> (stop_event, worker_thread); self._lock guards it.
- schedule / schedule_every / submit return an int job id.
- Use threading.Event(), threading.Thread(...), threading.current_thread()
  and time.monotonic() through the module names (the tests patch them).
"""

# BEGIN nuro-scheduler
import itertools
import threading
import time


class TaskScheduler:
    def __init__(self):
        # A lock, an id counter, and the registry job_id -> (stop_event, worker).
        ...
        self._lock = threading.Lock()
        self._next_id = itertools.count(1)
        self._jobs = {}

    def schedule(self, fn, frequency):
        # frequency in Hz; <= 0 -> ValueError. Delegates to schedule_every.
        ...
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        return self.schedule_every(fn, 1.0/frequency)

    def schedule_every(self, fn, period_seconds):
        # period <= 0 -> ValueError. Delegates to _start.
        ...
        if period_seconds <= 0:
            raise ValueError("frequency must be positive")
        return self._start(fn, period_seconds)

    def submit(self, fn):
        # Run once, as soon as possible: _start with period=None.
        ...
        return self._start(fn, None)

    def _start(self, fn, period):
        # Non-callable fn -> TypeError.
        # One thread per job. Fixed-rate loop: first call after one period,
        # then deadline += period; an overdue deadline runs immediately.
        # Hints: register the job BEFORE worker.start(); the worker removes
        # itself from the registry in a finally (cancel / one-shot / exception).
        ...
        if not callable(fn):
            raise TypeError("xxx")
        
        stop = threading.Event()
        with self._lock:
            job_id = next(self._next_id)
        

        def run():
            try:
                deadline = time.monotonic() + (0 if period is None else period)
                while not stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        stop.wait(remaining)
                        continue
                    fn()
                    if period is None:
                        return
                    deadline += period
            finally:
                with self._lock:
                    self._jobs.pop(job_id)

        worker = threading.Thread(target=run)
        with self._lock:
            self._jobs[job_id] = (stop,worker)
        
        worker.start()
        return job_id
        

    
    def deschedule(self, job_id, wait=True):
        # Cancel one job. Unknown or finished id -> return False (never raise).
        # Hints: join OUTSIDE the lock; never join the current thread.
        ...
        with self._lock:
            entry = self._jobs.get(job_id)
        
        stop, worker = entry
        stop.set()
        if wait and worker is not threading.current_thread():
            worker.join()
        return True

    def deschedule_all(self, wait=True):
        # Cancel every registered job.
        ...
        with self._lock:
            job_ids = list(self._jobs)
        for job_id in job_ids:
            self.deschedule(job_id, wait)

    def join(self, job_id, timeout=None):
        # Wait for a job to end on its own. True once it is no longer running;
        # False on timeout or when called from that job's own thread.
        ...
        with self._lock:
            entry = self._jobs.get(job_id)
        if entry is None:
            return True
        worker = entry[1]
        
        if worker is threading.current_thread():
            return False
        worker.join(timeout)
        return not worker.is_alive()

    def active_jobs(self):
        # Sorted list of ids still running.
        ...
        with self._lock:
            return sorted(self._jobs)
# END nuro-scheduler
