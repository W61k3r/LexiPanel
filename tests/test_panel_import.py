#!/usr/bin/env python3
"""panel.py imports cleanly and binds every module to itself. Compiling each file cannot catch
a module used before it is imported (a NameError at the panel's start, found on a scratch
panel on 2026-09-25); importing the whole panel does. Runs in a subprocess with a scratch
INF01_PANEL_DIR so no test module sees the real panel's globals."""
import os, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = """
import panel
for name in ("benchlab", "restarts", "remotes", "vllm_engine", "batches"):
    mod = getattr(panel, name)
    assert mod.P is panel, name + " is not bound to the panel"
assert panel.restarts.G is panel.gateway and panel.restarts.A is panel.auth
print("bound")
"""


class PanelImport(unittest.TestCase):
    def test_imports_and_binds(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, INF01_PANEL_DIR=d, PANEL_PORT="0")
            r = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env, capture_output=True,
                               text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("bound", r.stdout)


if __name__ == "__main__":
    unittest.main()
