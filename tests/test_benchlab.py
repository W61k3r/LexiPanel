#!/usr/bin/env python3
"""benchlab end to end against a fake llama-server that behaves like the real one where it
matters: one slot with prompt caching, speed that falls with depth, MTP acceptance that moves
decode speed, sampled vs greedy output, settings that only change on restart, and
llama-server's requests_deferred when a real request queues behind ours."""
import contextlib, hashlib, http.server, json, math, os, random, shutil, socketserver, sys
import tempfile, threading, time, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchlab as B  # noqa: E402
import benchstats as S  # noqa: E402

SPEED = {("UBATCH", "1024"): 1.08, ("THREADS", "8"): 1.0}     # the candidates' true effects


class FakeLlama:
    def __init__(self, noise=0.003, seed=1):
        self.cfg = {}
        self.pid = 1000
        self.cached = []
        self.deferred = 0             # real requests waiting (requests_deferred)
        self.defer_while_busy = 0     # show one this many times while a request runs
        self.slow_first = 0.0
        self.rng = random.Random(seed)
        self.noise = noise
        self.slot = threading.Lock()
        self.calls = 0
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/metrics":
                    b = f"llamacpp:requests_processing 0\nllamacpp:requests_deferred {fake.deferred}\n".encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    return self.wfile.write(b)
                if self.path.startswith("/slots"):
                    return self._json([dict(id=0, is_processing=False)])
                return self._json(dict(status="ok"))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/tokenize":
                    return self._json(dict(tokens=list(range(len(body.get("content", "")) // 4))))
                with fake.slot:                                  # one slot: requests queue
                    return fake.complete(self, body)

        class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True
        self.srv = Srv(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def factor(self):
        f = 1.0
        for k, v in self.cfg.items():
            f *= SPEED.get((k, str(v)), 1.0)
        return f

    def complete(self, h, body):
        self.calls += 1
        prompt = body.get("prompt") or []
        n = int(body.get("n_predict") or 16)
        depth = len(prompt)
        common = 0
        for x, y in zip(self.cached, prompt):
            if x != y:
                break
            common += 1
        new = max(depth - common, 1)
        greedy = body.get("temperature") == 0.0
        seed = body.get("seed")
        if greedy:
            acc = 0.85 + 0.03 * math.sin(depth / 1000.0)          # same tokens -> same acceptance
        else:
            acc = random.Random(seed).uniform(0.6, 0.95)
        dec = 60 * (1 - depth / 400000) * self.factor() * math.exp(0.82 * (acc - 0.85)) \
            * math.exp(self.rng.gauss(0, self.noise))
        text = hashlib.sha1(f"{depth}:{seed}:{greedy}:{self.cfg.get('MODEL')}".encode()).hexdigest()
        if not greedy:
            text += str(self.rng.random())
        self.cached = list(prompt) + [-1] * n
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        if self.slow_first and self.calls == 1:
            time.sleep(self.slow_first)
        if self.defer_while_busy:
            self.deferred = 1
            self.defer_while_busy -= 1
            time.sleep(2.5)                                         # the watcher polls every second (margin for a loaded box)
        try:
            for piece in (text[:10], text[10:20], text[20:]):
                h.wfile.write(f"data: {json.dumps(dict(content=piece))}\n\n".encode())
                h.wfile.flush()
            t = dict(prompt_n=new, prompt_ms=new / 800 * 1000, prompt_per_second=800.0, predicted_n=n,
                     predicted_per_second=dec, draft_n=100, draft_n_accepted=int(acc * 100))
            h.wfile.write(f"data: {json.dumps(dict(content='', stop=True, timings=t))}\n\n".encode())
            h.wfile.flush()
        except OSError:
            pass
        finally:
            self.deferred = 0 if not self.defer_while_busy else self.deferred

    def restart(self, params):
        self.cfg = dict(params)
        self.pid += 1
        self.cached = []

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def make_env(tmp, fake):
    """Fake panel, optimizer, workload, depthcurve and gputune modules around a FakeLlama."""
    panel_dir = Path(tmp) / "panel"
    panel_dir.mkdir()
    params_file = panel_dir / "params.env"
    saved = dict(UBATCH="512", THREADS="6", MODEL="/m/model.gguf", CTX="65536", BACKEND="vulkan")
    params_file.write_text(json.dumps(saved))
    fake.restart(saved)
    state = dict(pending=None, restarts=0)

    P = types.SimpleNamespace()
    P.PANEL = panel_dir
    P.DEFAULTS = {k: "" for k in ("UBATCH", "THREADS", "MODEL", "CTX", "BACKEND", "PORT", "HOST",
                                  "SPEC_DRAFT_MODEL", "KV_TYPE", "TEMP")}
    P.CURVE_BUCKETS = [0, 10000, 25000, 50000, 80000, 120000, 160000, 200000, 262144]
    P.FAIL_FILE = panel_dir / ".launch-fails"
    P.get_instance = lambda iid=None: dict(id=iid or "main", engine=None, devices=["0000:03:00.0"])
    P.using_instance = lambda inst: contextlib.nullcontext()
    P.load_params = lambda backend=None: json.loads(params_file.read_text())
    P.save_params = lambda vals, backend=None: params_file.write_text(json.dumps(vals))
    P.server_pid = lambda: fake.pid
    P.INST = lambda: dict(legacy=True, id="main")
    P.unit_installed = lambda inst=None: True

    def systemctl(verb, inst=None):
        state["restarts"] += 1
        fake.restart(json.loads(params_file.read_text()))
        return True, "ok"
    P._systemctl = systemctl
    P.unit_state = lambda inst=None: dict(active="active")
    P.estimate = lambda vals: dict(vram=dict(headroom_mib=900))
    P.ram_budget = lambda vals, include_others=False: dict(verdict="ok")

    O = types.SimpleNamespace(THERMAL_DEFAULTS=dict(pause_c=109, resume_c=100, abort_c=112, mem_abort_c=106),
                              _gpu_temps=lambda devices: dict(junction=70, mem=70),
                              _config_files=lambda: [str(params_file), str(P.FAIL_FILE)])
    O._live_matches = lambda vals: [f"{k} wanted {v}, running {fake.cfg.get(k)}"
                                    for k, v in vals.items() if str(fake.cfg.get(k)) != str(v)]
    O._run = None

    W = types.SimpleNamespace(rows=[])
    W.envelope = lambda inst, **kw: dict(mix=dict(depths=[4096, 16384], weights=[0.4, 0.6], prompt_tokens=64,
                                                  output_tokens=64), slot_ctx=65536, depth=dict(n=100))
    W.busy_now = lambda iid, now=None: 0
    W.live_fp = lambda: ("fp-test", dict(model="model.gguf"))
    W.requests = lambda iid, since=0, include_bench=False: list(W.rows)
    W._dir = lambda iid: panel_dir / "workload"

    argv = ["llama-server", "-m", "/m/model.gguf", "-c", "65536", "--port", str(fake.port)]
    DC = types.SimpleNamespace(_target=lambda inst: ("127.0.0.1", fake.port, ["0000:03:00.0"], argv),
                               _filler_tokens=lambda host, port, need: list(range(need)))
    GT = types.SimpleNamespace(sample=lambda devices: dict(power_w=250.0, junction=75, mem=80))
    B.bind(P, O, W, DC, GT)
    return P, state, params_file, saved


class BenchLab(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fake = FakeLlama()
        self.P, self.state, self.params_file, self.saved = make_env(self.tmp, self.fake)
        self._sleep = B._sleep
        B._sleep = lambda sec: (_ for _ in ()).throw(B.Stopped()) if B._stop.wait(min(sec, 0.02)) else None
        B._run = None
        B._stop.clear()

    def tearDown(self):
        B._sleep = self._sleep
        if B._run and not B._run.get("_done"):
            B.stop()
            B._thread.join(10)
        self.fake.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_kind(self, **body):
        body.setdefault("quiet_s", 0)
        B.start(body)
        B._thread.join(120)
        self.assertFalse(B._thread.is_alive(), "run did not finish")
        return B.read_run("main", B._run["id"])

    def test_profile(self):
        r = self.run_kind(kind="profile", reps=3, n_predict=32)
        self.assertEqual(r["state"], "done", r.get("error"))
        res = r["result"]
        self.assertEqual(sorted(res["per_depth"]), ["16384", "4096"])
        self.assertTrue(all(v["deterministic"] for v in res["per_depth"].values()))
        self.assertGreater(res["request_s"], 0)
        blk = r["blocks"][0]["per_depth"]["16384"]
        self.assertEqual(len(blk["decode"]), 3)
        self.assertIsNotNone(blk["turn_ms"])
        self.assertEqual(self.state["restarts"], 0)          # a profile never restarts

    def test_calibrate_selftest(self):
        r = self.run_kind(kind="calibrate", blocks=6, reps=2, n_predict=32)
        self.assertEqual(r["state"], "done", r.get("error"))
        res = r["result"]
        self.assertEqual(res["scope"], "within one server run (no restart)")
        self.assertTrue(res["deterministic_across_blocks"])
        self.assertLessEqual(res["selftest"]["false_win"], 0.06)
        self.assertIn("2%", res["pairs_for"])
        cal = B.calibration("main")["latest"]
        self.assertEqual(cal["id"], r["id"])

    def test_restarting_runs_need_consent(self):
        with self.assertRaisesRegex(ValueError, "allow_restart"):
            B.start(dict(kind="compare", candidate=dict(UBATCH="1024"), quiet_s=0))
        with self.assertRaisesRegex(ValueError, "allow_restart"):
            B.start(dict(kind="calibrate", restart=True, quiet_s=0))

    def test_candidate_validation(self):
        for cand, msg in ((dict(PORT="9000"), "not varied"), (dict(SPEC_DRAFT_MODEL="/m/x.gguf"), "not varied"),
                          (dict(NOPE="1"), "not a setting"), (dict(UBATCH="512"), "equals the saved"),
                          (dict(MODEL="/nonexistent.gguf"), "existing .gguf")):
            with self.assertRaisesRegex(ValueError, msg):
                B.start(dict(kind="compare", candidate=cand, allow_restart=True, quiet_s=0))

    def test_compare_finds_real_gain_and_restores(self):
        r = self.run_kind(kind="compare", candidate=dict(UBATCH="1024"), allow_restart=True, reps=2, n_predict=32)
        self.assertEqual(r["state"], "done", r.get("error"))
        res = r["result"]
        self.assertEqual(res["verdict"], "better")
        self.assertLessEqual(res["pairs"], 5)
        self.assertLess(res["request_time_pct"], -3)        # about 8 % faster -> ~7 % less time
        self.assertEqual([b["side"] for b in r["blocks"]][:4], ["A", "B", "B", "A"])
        self.assertEqual(json.loads(self.params_file.read_text()), self.saved)   # files put back
        self.assertEqual(self.fake.cfg.get("UBATCH"), "512")                    # server back on A
        self.assertTrue(all(v["same_output"] for v in res["per_depth"].values()))

    def test_never_restarts_under_a_real_request(self):
        # a real request is running whenever the platform is about to switch configurations
        busy = dict(n=0, seen=0)
        def busy_now(iid, now=None):
            if busy["n"] > 0:
                busy["n"] -= 1
                busy["seen"] += 1
                return 1
            return 0
        B.W.busy_now = busy_now
        orig = self.P._systemctl
        def systemctl(verb, inst=None):
            self.assertEqual(busy["n"], 0, "restarted while a real request was running")
            busy["n"] = 3                      # the next switch will find the server busy again
            return orig(verb, inst)
        self.P._systemctl = systemctl
        busy["n"] = 3
        r = self.run_kind(kind="compare", candidate=dict(UBATCH="1024"), allow_restart=True, reps=2, n_predict=32)
        self.assertEqual(r["state"], "done", r.get("error"))
        self.assertGreater(busy["seen"], 3)
        self.assertTrue(any("waited" in l or "waiting" in l for l in r["log"]) or busy["seen"] > 0)

    def test_compare_no_effect_is_not_a_win(self):
        r = self.run_kind(kind="compare", candidate=dict(THREADS="8"), allow_restart=True, reps=2, n_predict=32)
        self.assertEqual(r["state"], "done", r.get("error"))
        self.assertIn(r["result"]["verdict"], ("same", "undecided", "small"))

    def test_real_request_preempts_measurement(self):
        self.fake.defer_while_busy = 1           # a real request queues behind our first request
        r = self.run_kind(kind="profile", reps=2, n_predict=32)
        self.assertEqual(r["state"], "done", r.get("error"))
        self.assertEqual(r["blocks"][0]["attempt"], 2)
        self.assertTrue(any("real request" in line for line in r["log"]))

    def test_goodput_one_slot_queues(self):
        r = self.run_kind(kind="goodput", concurrency=[1, 3], goodput_depth=2048, n_predict=32, slo_ttft_s=0.05)
        self.assertEqual(r["state"], "done", r.get("error"))
        lv = {x["streams"]: x for x in r["result"]["levels"]}
        self.assertEqual(lv[3]["tokens"], 96)
        self.assertGreaterEqual(lv[3]["ttft_s"]["max"], lv[1]["ttft_s"]["max"])  # one slot: turns queue

    def test_no_overlap(self):
        self.fake.slow_first = 1.0
        B.start(dict(kind="profile", reps=2, n_predict=32, quiet_s=0))
        with self.assertRaisesRegex(ValueError, "already active"):
            B.start(dict(kind="profile", quiet_s=0))
        self.assertTrue(B.active())
        B._thread.join(60)

    def test_stop(self):
        self.fake.slow_first = 2.0
        B.start(dict(kind="profile", reps=3, n_predict=32, quiet_s=0))
        time.sleep(0.3)
        B.stop()
        B._thread.join(30)
        self.assertEqual(B._run["state"], "stopped")

    def test_recover_on_startup_restores_files(self):
        d = self.P.PANEL / "bench" / "main"
        d.mkdir(parents=True)
        snap = d / "compare_20260925_120000.snapshot"
        snap.mkdir()
        (snap / "0").write_text(json.dumps(self.saved))
        (snap / "manifest.json").write_text(json.dumps({str(self.params_file): "0", str(self.P.FAIL_FILE): None}))
        self.params_file.write_text(json.dumps(dict(self.saved, UBATCH="1024")))     # the trial, left behind
        (d / "compare_20260925_120000.json").write_text(json.dumps(dict(
            id="compare_20260925_120000", kind="compare", instance="main", state="running",
            snapshot=str(snap), applied=dict(UBATCH="1024"))))
        B.recover_on_startup()
        r = B.read_run("main", "compare_20260925_120000")
        self.assertEqual(r["state"], "interrupted")
        self.assertTrue(r["needs_restart"])
        self.assertEqual(json.loads(self.params_file.read_text()), self.saved)

    def test_traffic_verdict_per_change(self):
        rng = random.Random(4)
        rows = []
        for i in range(160):
            fp = "fpA" if i < 80 else "fpB"
            acc = rng.uniform(0.65, 0.95)
            depth = rng.choice([30000, 90000, 150000])
            tps = 55 * (1 - depth / 400000) * (1.06 if fp == "fpB" else 1.0) * math.exp(0.8 * (acc - 0.8)
                                                                                       + rng.gauss(0, 0.02))
            rows.append(dict(t=1000 + i, fp=fp, depth=depth, accept=acc, decode_tps=tps, eval_tokens=200))
        B.W.rows = rows
        t = B.traffic("main")
        self.assertEqual(len(t["changes"]), 1)
        ch = t["changes"][0]
        self.assertEqual(ch["verdict"], "better")
        self.assertGreater(ch["adjusted"]["pct_lo"], 2)
        self.assertLess(t["noise"]["adjusted_sd"], t["noise"]["raw_sd"])


if __name__ == "__main__":
    unittest.main()
