"""Run with: python3 demo_simple_scheduler.py

Uses only the Python standard library and the neighboring scheduler module.
The worker waits 0.5 seconds before each heartbeat (fixed delay, 2 Hz).
"""

import time

from nuro_scheduler_steps import SimpleScheduler


def main():
    scheduler = SimpleScheduler()
    try:
        job_id = scheduler.schedule(lambda: print("heartbeat"), frequency=2)
        print("schedule returned; the caller can continue")
        time.sleep(1.1)  # Only the main thread sleeps; the worker keeps running.
        scheduler.deschedule(job_id)
        print("Job stopped; its worker thread has finished")
    finally:
        scheduler.close()
        print("Scheduler closed")


if __name__ == "__main__":
    main()
