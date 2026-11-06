"""Run the repo's tests against the skeletons in this directory.

    python3 practice/check.py scheduler   # test_nuro_scheduler
    python3 practice/check.py cache       # test_nuro_cache_practice
    python3 practice/check.py pool        # test_nuro_thread_pool_practice
    python3 practice/check.py video       # test_nuro_video_player_practice
    python3 practice/check.py all

This directory is first on sys.path, so `import nuro_scheduler` resolves to
practice/nuro_scheduler.py, not the reference solution in the repo root.
"""
import importlib
import os
import sys
import unittest

PRACTICE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PRACTICE)
sys.path[:0] = [PRACTICE]           # skeletons win over the root solutions
if ROOT not in sys.path:
    sys.path.append(ROOT)           # the test modules live in the root

TARGETS = {
    "scheduler": ("nuro_scheduler", "test_nuro_scheduler"),
    "cache": ("nuro_cache_practice", "test_nuro_cache_practice"),
    "pool": ("nuro_thread_pool_practice", "test_nuro_thread_pool_practice"),
    "video": ("nuro_video_player_practice", "test_nuro_video_player_practice"),
}


def main(argv):
    names = argv[1:] or ["all"]
    if names == ["all"]:
        names = list(TARGETS)
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        sys.exit(f"unknown target(s) {unknown}; choose from {list(TARGETS)} or all")
    ok = True
    for name in names:
        module, test = TARGETS[name]
        implementation = importlib.import_module(module)
        print(f"\n=== {name}: {test} against {implementation.__file__}", flush=True)
        suite = unittest.defaultTestLoader.loadTestsFromName(test)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        ok &= result.wasSuccessful()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv)
