"""vLLM engine: settings and their validation, the API key's handling (never on the command line,
in params.env or in the plan's logged environment), the launch plan, the card guard, and a
server row read from a stand-in vLLM (OpenAI API + Prometheus metrics).
  python3 -m unittest discover tests"""
import contextlib, http.server, json, os, stat, sys, tempfile, threading, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vllm_engine as V  # noqa: E402

KEY = "vl-secret-key-123"


def read_env(f):
    return dict(l.split("=", 1) for l in Path(f).read_text().splitlines() if "=" in l and not l.startswith("#")) \
        if Path(f).exists() else {}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.d = d
        (d / "inst").mkdir()
        (d / "model").mkdir()
        (d / "model" / "config.json").write_text("{}")
        (d / "model" / "model.safetensors").write_bytes(b"x" * 1024)
        venv = d / "vllm" / "venv-cu" / "bin"
        venv.mkdir(parents=True)
        (venv / "vllm").write_text("#!/bin/sh\n")
        self.inst = dict(id="vl", dir=d / "inst", rundir=d / "run", engine="vllm", device="0000:07:00.0",
                         devices=["0000:07:00.0"])
        self.others = {}
        P = types.SimpleNamespace(_read_env_file=read_env, _env_quote=lambda v: str(v), HOME=d,
                                  _meminfo_mb=lambda k: 64000, using_instance=lambda i: contextlib.nullcontext(),
                                  server_pid=lambda: None, instance_ids=lambda: ["vl", *self.others],
                                  get_instance=lambda iid: self.others.get(iid, self.inst),
                                  _argv_get=lambda argv, names: next((argv[i + 1] for i, a in enumerate(argv)
                                                                      if a in names and i + 1 < len(argv)), None))
        V.bind(P)
        self.P = P
        V._rt_cache.clear()
        self._rt, self._gi = V.runtime, V.gpu_index
        V.runtime = lambda venv: dict(vllm="0.17.1", torch="2.9.0", cuda="12.9", hip=None, devices=1)
        V.gpu_index = lambda pci, backend: 0 if pci == "0000:07:00.0" and backend == "cuda" else None
        V.write_params(self.inst, dict(V.DEFAULTS, VL_MODEL=str(d / "model")))

    def tearDown(self):
        V.runtime, V.gpu_index = self._rt, self._gi
        self.tmp.cleanup()


class Settings(Base):
    def test_every_setting_has_a_tooltip(self):
        meta = V.meta()
        self.assertEqual(set(meta), set(V.DEFAULTS))
        for k, m in meta.items():
            self.assertGreater(len(m["tip"]), 30, k)

    def test_bad_values_are_refused(self):
        for bad in (dict(PORT=8090), dict(PORT=80), dict(HOST="198.51.100.5"), dict(VL_GPU_MEM_UTIL="1.5"),
                    dict(VL_GPU_MEM_UTIL="lots"), dict(VL_DTYPE="int3"), dict(VL_KV_CACHE_DTYPE="fp4"),
                    dict(VL_QUANTIZATION="magic"), dict(VL_MAX_NUM_SEQS="0"), dict(VL_TP="-1"),
                    dict(VL_MODEL="relative/path/../x"), dict(VL_MODEL="not a repo"), dict(BACKEND="vulkan"),
                    dict(VL_EXTRA="--port 9000"), dict(VL_EXTRA="--api-key x"), dict(VL_EXTRA="--trust-remote-code"),
                    dict(VL_EXTRA="--ssl-keyfile /etc/shadow"), dict(VL_EXTRA="-X"), dict(VL_EXTRA="--x\n--y"),
                    dict(VL_VENV="relative"), dict(NOT_A_KEY=1)):
            with self.assertRaises(ValueError, msg=bad):
                V.save_params(self.inst, bad)

    def test_good_values(self):
        cur = V.save_params(self.inst, dict(VL_MODEL="Qwen/Qwen3-1.7B", VL_EXTRA="--swap-space 4 --seed=1",
                                            VL_MAX_NUM_SEQS="64", VL_GPU_MEM_UTIL="0.9"))
        self.assertEqual((cur["VL_MODEL"], cur["VL_MAX_NUM_SEQS"]), ("Qwen/Qwen3-1.7B", 64))

    def test_served_name_is_what_clients_must_send(self):
        self.assertEqual(V.served_name(dict(VL_MODEL="/m/Qwen3-1.7B")), "Qwen3-1.7B")
        self.assertEqual(V.served_name(dict(VL_MODEL="Qwen/Qwen3-1.7B")), "Qwen/Qwen3-1.7B")
        self.assertEqual(V.served_name(dict(VL_MODEL="/m/x", VL_SERVED_NAME="qwen")), "qwen")
        p = V.launch_plan(self.inst)
        self.assertEqual(p["argv"][p["argv"].index("--served-model-name") + 1], V.served_name(V.load_params(self.inst)))

    def test_api_key_is_kept_out_of_params_and_masked(self):
        V.save_params(self.inst, dict(VL_API_KEY=KEY))
        kf = self.inst["dir"] / "vllm-api-key"
        self.assertEqual(stat.S_IMODE(kf.stat().st_mode), 0o600)
        self.assertNotIn(KEY, (self.inst["dir"] / "params.env").read_text())
        self.assertEqual(V.load_params(self.inst)["VL_API_KEY"], V.MASK)
        V.save_params(self.inst, dict(VL_API_KEY=V.MASK, PORT=8085))            # the masked value keeps it
        self.assertEqual(V.api_key(self.inst), KEY)
        V.save_params(self.inst, dict(VL_API_KEY=""))                            # empty removes it
        self.assertIsNone(V.api_key(self.inst))
        self.assertFalse(kf.exists())


class Plan(Base):
    def test_plan(self):
        V.save_params(self.inst, dict(VL_API_KEY=KEY, VL_MAX_MODEL_LEN="4096", VL_ENFORCE_EAGER="1",
                                      VL_EXTRA="--swap-space 2"))
        p = V.launch_plan(self.inst)
        self.assertEqual(p["errors"], [])
        a = p["argv"]
        self.assertEqual(a[1:3], ["serve", str(self.d / "model")])
        for flag in ("--max-model-len", "--enforce-eager", "--enable-prefix-caching", "--swap-space",
                     "--gpu-memory-utilization", "--max-num-seqs", "--download-dir"):
            self.assertIn(flag, a)
        self.assertNotIn(KEY, " ".join(a))                                       # never on the command line
        self.assertNotIn(KEY, json.dumps(p))                                     # nor anywhere in the plan (API)
        self.assertEqual(p["secret_env_names"], ["VLLM_API_KEY"])
        self.assertEqual(V.secrets(self.inst), dict(VLLM_API_KEY=KEY))           # the launcher's only source
        self.assertEqual((p["env"]["VLLM_NO_USAGE_STATS"], p["env"]["DO_NOT_TRACK"]), ("1", "1"))
        self.assertEqual((p["env"]["CUDA_DEVICE_ORDER"], p["env"]["CUDA_VISIBLE_DEVICES"]), ("PCI_BUS_ID", "0"))
        self.assertEqual(p["env"]["HF_HUB_OFFLINE"], "1")                        # a local folder: no downloads

    def test_refusals(self):
        (self.d / "vllm" / "venv-cu" / "bin" / "vllm").unlink()
        self.assertTrue(any("install-vllm.sh" in e for e in V.launch_plan(self.inst)["errors"]))
        (self.d / "vllm" / "venv-cu" / "bin" / "vllm").write_text("#!/bin/sh\n")
        V.write_params(self.inst, dict(V.load_params(self.inst), VL_MODEL=str(self.d / "x.gguf")))
        self.assertTrue(any("GGUF files are for llama.cpp" in e for e in V.launch_plan(self.inst)["errors"]))
        V.write_params(self.inst, dict(V.load_params(self.inst), VL_MODEL=str(self.d / "model"), HOST="0.0.0.0"))
        self.assertTrue(any("needs an API key" in e for e in V.launch_plan(self.inst)["errors"]))

    def test_the_card_must_be_free(self):
        other = dict(id="llama-2060", devices=["0000:07:00.0"], device="0000:07:00.0")
        self.others["llama-2060"] = other
        self.P.server_pid = lambda: 4242                                         # the other one is running
        errs = V.launch_plan(self.inst)["errors"]
        self.assertTrue(any("in use by llama-2060" in e for e in errs), errs)

    def test_not_an_nvidia_card(self):
        self.inst.update(device="0000:03:00.0", devices=["0000:03:00.0"])
        self.assertTrue(any("not an NVIDIA card" in e for e in V.launch_plan(self.inst)["errors"]))

    def test_trust_remote_code_is_explicit(self):
        V.save_params(self.inst, dict(VL_TRUST_REMOTE_CODE="1"))
        p = V.launch_plan(self.inst)
        self.assertIn("--trust-remote-code", p["argv"])
        self.assertTrue(any("trust-remote-code" in w for w in p["warnings"]))

    def test_launcher_passes_secret_env_but_never_logs_it(self):
        src = (ROOT / "instance_launch.py").read_text()
        log_line = src.index('say(out, "env  "')
        secret = src.index("mod.secrets(inst)")
        self.assertGreater(secret, log_line)                                    # added after the log line
        self.assertLess(secret, src.index("subprocess.Popen(plan[\"argv\"], env=env", secret - 400))


class Stand(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            b, ct = b"", "text/plain"
        elif self.path == "/v1/models":
            b, ct = json.dumps(dict(data=[dict(id="qwen3-1.7b", max_model_len=4096)])).encode(), "application/json"
        elif self.path == "/metrics":
            b, ct = (b"vllm:num_requests_running 3.0\nvllm:num_requests_waiting 1.0\n"
                     b"vllm:kv_cache_usage_perc 0.25\n"), "text/plain"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class Running(Base):
    def test_server_row_from_a_retitled_process(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stand)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            V.write_params(self.inst, dict(V.load_params(self.inst), PORT=srv.server_address[1]))
            orig = V._instance_of_pid
            V._instance_of_pid = lambda pid: self.inst                          # found by its unit (cgroup)
            try:
                row = V.describe(4242, ["VLLM::APIServer"])                     # the command line is gone
            finally:
                V._instance_of_pid = orig
            self.assertEqual((row["health"], row["model_name"], row["ctx"], row["port"]),
                             ("ok", "qwen3-1.7b", 4096, srv.server_address[1]))
            self.assertEqual((row["running"], row["waiting"], row["kv_usage"], row["slots_busy"]), (3.0, 1.0, 25.0, 3))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
