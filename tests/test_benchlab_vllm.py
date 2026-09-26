#!/usr/bin/env python3
"""Bench against a stand-in vLLM: OpenAI streaming completions (no server timings, so Bench
times them), /tokenize, /v1/models and vllm:* metrics. The stand-in serves concurrent
requests in parallel like vLLM's scheduler, so goodput's total throughput must rise with
the number of streams."""
import http.server, json, socketserver, sys, threading, time, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlab as B  # noqa: E402
from test_benchlab import make_env, FakeLlama  # noqa: E402

PER_TOKEN_S = 0.004


class FakeVllm:
    def __init__(self):
        self.waiting = 0

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json(dict(data=[dict(id="qwen3-1.7b", max_model_len=8192)]))
                if self.path == "/metrics":
                    b = f"vllm:num_requests_running 0.0\nvllm:num_requests_waiting {fake.waiting}.0\n".encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    return self.wfile.write(b)
                return self._json({})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/tokenize":
                    return self._json(dict(tokens=list(range(len(body.get("prompt", "")) // 4))))
                assert self.path == "/v1/completions" and body["stream"] and body["ignore_eos"]
                n = int(body["max_tokens"])
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(0.01 + len(body["prompt"]) / 200000)             # prefill
                for i in range(n):
                    time.sleep(PER_TOKEN_S)                                  # every stream advances each step
                    self.wfile.write(f"data: {json.dumps(dict(choices=[dict(text='t')]))}\n\n".encode())
                    self.wfile.flush()
                u = dict(choices=[], usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=n))
                self.wfile.write(f"data: {json.dumps(u)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()

        class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 128                # the default 5 makes the 16th connection wait a TCP retry
        fake = self
        self.srv = Srv(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class VllmBench(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.llama = FakeLlama()                                             # make_env wants one; unused
        self.P, self.state, _pf, _saved = make_env(self.tmp, self.llama)
        self.vl = FakeVllm()
        inst = dict(id="vl", engine="vllm", devices=["0000:07:00.0"], device="0000:07:00.0")
        self.P.get_instance = lambda iid=None: inst
        self.P.INST = lambda: dict(inst, legacy=False)
        self.P.vllm_engine = types.SimpleNamespace(
            load_params=lambda i: dict(HOST="127.0.0.1", PORT=self.vl.port, VL_SERVED_NAME="qwen3-1.7b",
                                       VL_MODEL="/m/qwen3-1.7b", VL_MAX_NUM_SEQS=16),
            api_key=lambda i: None)
        B.DC.FILLER_GLOB = "/usr/lib/python3*/**/*.py"                      # the real filler source
        self._sleep = B._sleep
        B._sleep = lambda sec: (_ for _ in ()).throw(B.Stopped()) if B._stop.wait(min(sec, 0.02)) else None
        B._run = None
        B._stop.clear()

    def tearDown(self):
        B._sleep = self._sleep
        if B._run and not B._run.get("_done"):
            B.stop()
            B._thread.join(10)
        self.vl.close()
        self.llama.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_kind(self, **body):
        body.setdefault("quiet_s", 0)
        B.start(dict(body, instance="vl"))
        B._thread.join(120)
        self.assertFalse(B._thread.is_alive())
        return B.read_run("vl", B._run["id"])

    def test_goodput_scales_with_streams(self):
        r = self.run_kind(kind="goodput", concurrency=[1, 4, 16], goodput_depth=1024, n_predict=32, slo_ttft_s=5)
        self.assertEqual(r["state"], "done", r.get("error"))
        lv = {x["streams"]: x for x in r["result"]["levels"]}
        self.assertEqual(lv[16]["tokens"], 16 * 32)
        self.assertGreater(lv[4]["throughput_tps"], 2.5 * lv[1]["throughput_tps"])    # batched, not queued
        self.assertGreater(lv[16]["throughput_tps"], 2 * lv[4]["throughput_tps"])
        self.assertEqual(r["result"]["slots"], 16)                            # vLLM's max-num-seqs
        self.assertEqual(r["result"]["depth"], 1024)                          # not divided between streams
        per = lv[1]["decode_per_stream"]["mean"]
        self.assertAlmostEqual(per, 1 / PER_TOKEN_S, delta=0.5 / PER_TOKEN_S)  # client-side decode timing

    def test_profile_on_vllm(self):
        r = self.run_kind(kind="profile", preset="quick", reps=2, n_predict=16)
        self.assertEqual(r["state"], "done", r.get("error"))
        self.assertEqual(r["shape"]["n_ctx"], 8192)                          # from /v1/models max_model_len
        for d, v in r["result"]["per_depth"].items():
            self.assertIsNotNone(v["decode"]["mean"], d)


if __name__ == "__main__":
    unittest.main()
