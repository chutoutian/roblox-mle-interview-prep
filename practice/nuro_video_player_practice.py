"""Round 1 skeleton: buffered video player (reader + renderer threads).

Fill in every `...` body. Run: python3 practice/check.py video

Contract (white-box names the tests rely on):
- __init__(read_frame, render_frame, fps=25, buffer_size=3):
  fps must be finite and > 0; buffer_size a positive int -> else ValueError.
  self.period = 1.0 / fps; self.frames = queue.Queue(maxsize=buffer_size);
  self.api_lock = threading.Lock(); self.stopped = threading.Event();
  self.workers = []  (empty until play).
- on_play_click(): plays once. Creates exactly [reader, renderer] in that
  order (names "reader", "renderer"), starts them; a second click, or a
  click after stop, does nothing.
- read_frame() returns a frame or None at EOF; render_frame(frame) renders.
  Both callbacks run under api_lock (they are assumed not thread-safe).
- _run(work): runs work(); on Exception sets stopped and traceback.print_exc().
- _read(): loop until stopped: read one frame under api_lock (re-check
  stopped after acquiring), then put it with a short timeout, retrying while
  not stopped; None (EOF) goes through the queue AFTER the last frame.
- _render(): pace frames to fps using time.monotonic() deadlines and
  stopped.wait(remaining) so stop wakes it early; get() with a short
  timeout; None means EOF -> return; render under api_lock.
- wait(): join both workers. stop(): set stopped, then wait().
  Both raise RuntimeError when called from a worker thread.
Keep the module-level imports: tests patch time.monotonic and traceback.print_exc.
"""

# BEGIN nuro-video-interview
import math
import queue
import threading
import time
import traceback


class VideoPlayer:
    def __init__(self, read_frame, render_frame, fps=25, buffer_size=3):
        ...

    def on_play_click(self):
        ...

    def _run(self, work):
        ...

    def _read(self):
        ...

    def _render(self):
        ...

    def wait(self):
        ...

    def stop(self):
        ...
# END nuro-video-interview


# BEGIN nuro-video-demo
if __name__ == "__main__":
    started = time.monotonic()
    source = iter(range(8))  # Frame numbers stand in for decoded images.

    def read_frame():
        time.sleep(0.005)  # Stand in for file I/O or decoding.
        return next(source, None)

    def render_frame(frame):
        elapsed = (time.monotonic() - started) * 1000
        name = threading.current_thread().name
        print(f"+{elapsed:.3f}ms [{name}] frame {frame}", flush=True)
        time.sleep(0.005)  # Stand in for rendering work.

    player = VideoPlayer(read_frame, render_frame, fps=25, buffer_size=3)
    try:
        player.on_play_click()
        print("Play returned; the caller can continue.", flush=True)
        player.wait()  # Terminal demo only: wait for normal EOF, not cancellation.
        print("EOF: all eight frames were rendered.", flush=True)
    finally:
        player.stop()
# END nuro-video-demo
