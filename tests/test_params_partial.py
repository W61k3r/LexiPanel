#!/usr/bin/env python3
"""A partial parameter update (only CTX, as an API client or an MCP tool sends it) must keep the
instance's backend and everything else. It used to fall back to params-vulkan.env: a CUDA
instance lost its model (2026-09-26). Runs the real panel.save_params in a subprocess against a
scratch INF01_PANEL_DIR."""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = r"""
import json, panel
from pathlib import Path
d = panel.INSTANCES_DIR / "gpu2"
d.mkdir(parents=True)
(d / "instance.json").write_text(json.dumps(dict(name="gpu2", device="cpu", devices=["cpu"])))
vals = dict(panel.DEFAULTS, BACKEND="cuda", MODEL="/m/small.gguf", CTX="8192", PARALLEL="16")
panel._write_env_file(d / "params-cuda.env", vals)
panel._write_env_file(d / "params.env", vals)
inst = panel.get_instance("gpu2")
with panel.using_instance(inst):
    panel.save_params({"CTX": "12288"})
after = panel._read_env_file(d / "params.env")
print(json.dumps({k: after.get(k) for k in ("BACKEND", "MODEL", "CTX", "PARALLEL")}))
"""


class PartialUpdate(unittest.TestCase):
    def test_keeps_backend_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, INF01_PANEL_DIR=tmp, PANEL_PORT="0")
            r = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            after = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(after, dict(BACKEND="cuda", MODEL="/m/small.gguf", CTX="12288", PARALLEL="16"))


if __name__ == "__main__":
    unittest.main()
