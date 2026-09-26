#!/usr/bin/env python3
"""restarts.py: every planned restart journaled, drained, verified, recovered and reported.
A fake llama-server that can save/restore slots, fall back to a safe tier on a launch it
cannot fit, and go silently wrong (same settings, different output) on demand."""
import contextlib, hashlib, http.server, json, os, shutil, socketserver, stat, sys, tempfile
import threading, time, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import restarts as R  # noqa: E402
import gateway as GW  # noqa: E402
import auth as AU  # noqa: E402

FALLBACK_CTX = "65536"          # what the launcher falls back to when CTX does not fit
TOO_BIG_CTX = "999999"


class FakeServer:
    def __init__(self, slot_dir):
        self.cfg, self.pid, self.cached = {}, 100, []
        self.broken = False           # silently wrong output from now on
        self.failed = False           # the unit failed: nothing listens
        self.slot_dir = slot_dir
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
                    b = b"llamacpp:requests_deferred 0\n"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    return self.wfile.write(b)
                if self.path == "/slots":
                    return self._json([dict(id=0, n_ctx=int(fake.cfg.get("CTX", 0)), is_processing=False)])
                return self._json(dict(status="ok"))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path.startswith("/slots/0?action="):
                    name = body["filename"]
                    assert "/" not in name
                    p = os.path.join(fake.slot_dir, name)
                    if self.path.endswith("save"):
                        with open(p, "w") as f:
                            json.dump(fake.cached, f)
                        return self._json(dict(id_slot=0, filename=name, n_saved=len(fake.cached)))
                    with open(p) as f:
                        fake.cached = json.load(f)
                    return self._json(dict(id_slot=0, filename=name, n_restored=len(fake.cached)))
                if self.path == "/completion":
                    seed = f"{fake.cfg.get('MODEL')}|{fake.cfg.get('CTX')}|{body.get('prompt')}"
                    text = hashlib.sha1((seed + ("|broken" if fake.broken else "")).encode()).hexdigest() * 2
                    return self._json(dict(content=text, timings=dict(predicted_n=body.get("n_predict"))))
                return self._json(dict(error="?"), 404)

        class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True
        self.srv = Srv(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def launch(self, params):
        cfg = dict(params)
        self.failed = str(cfg.get("MODEL", "")).endswith("does-not-fit.gguf")
        if self.failed:                       # the model does not load: the server never comes up
            self.pid = None
            return
        if cfg.get("CTX") == TOO_BIG_CTX:
            cfg["CTX"] = FALLBACK_CTX         # the launcher's fallback tier: comes up, but not as asked
        self.cfg = cfg
        self.launches = getattr(self, "launches", 0) + 1
        self.pid = 100 + self.launches
        self.cached = []

    def argv(self):
        return ["/opt/llama/b1/llama-server", "-m", self.cfg.get("MODEL", ""), "-c", self.cfg.get("CTX", ""),
                "-ctk", self.cfg.get("KV_TYPE", "q8_0"), "--port", str(self.port), "--slot-save-path", self.slot_dir]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "panel").mkdir()
        (self.tmp / "slots").mkdir()
        self.srv = FakeServer(str(self.tmp / "slots"))
        self.params_file = self.tmp / "panel" / "params.env"
        self.good = dict(MODEL=str(self.tmp / "model.gguf"), CTX="131072", KV_TYPE="q8_0")
        Path(self.good["MODEL"]).write_text("weights")
        self.params_file.write_text(json.dumps(self.good))
        self.srv.launch(self.good)
        self.restarts_done = 0
        self.busy = dict(n=0)
        self.audit = []

        P = types.SimpleNamespace(PANEL=self.tmp / "panel", FAIL_FILE=self.tmp / "panel" / ".launch-fails")
        P.INST = lambda: dict(id="main", legacy=True)
        P.get_instance = lambda iid=None: dict(id=iid or "main", legacy=True)
        P.using_instance = lambda inst: contextlib.nullcontext()
        P.server_pid = lambda: self.srv.pid
        P.live_cmdline_args = lambda: self.srv.argv()
        P._argv_get = lambda argv, names: next((argv[i + 1] for i, a in enumerate(argv) if a in names), None)
        P.load_params = lambda backend=None: json.loads(self.params_file.read_text())
        P.unit_installed = lambda inst=None: True
        P.unit_state = lambda inst=None: dict(active="failed" if self.srv.failed else "active")

        def systemctl(verb, inst=None):
            self.restarts_done += 1
            self.srv.launch(json.loads(self.params_file.read_text()))
            return True, "ok"
        P._systemctl = systemctl
        O = types.SimpleNamespace(_config_files=lambda: [str(self.params_file), str(P.FAIL_FILE)])
        O._live_matches = lambda want: [f"{k} wanted {v}, running {self.srv.cfg.get(k)}"
                                        for k, v in want.items() if k in ("CTX", "MODEL", "KV_TYPE")
                                        and str(self.srv.cfg.get(k)) != str(v)]

        def busy_now(iid, now=None):
            if self.busy["n"] > 0:
                self.busy["n"] -= 1
                return 1
            return 0
        W = types.SimpleNamespace(busy_now=busy_now)
        W.live_fp = lambda: (hashlib.sha1(json.dumps(self.srv.cfg, sort_keys=True).encode()).hexdigest()[:12],
                             dict(self.srv.cfg))
        self.holds = []
        G = types.SimpleNamespace(hold=lambda iid, s: self.holds.append(("hold", iid)),
                                  release=lambda iid: (self.holds.append(("release", iid)), dict(held=2, expired=0))[1])
        A = types.SimpleNamespace(audit=lambda *a: self.audit.append(a))
        self._sleep = time.sleep
        R.time.sleep = lambda s: self._sleep(min(s, 0.01))
        R.bind(P, O, W, G, A)
        R._active.clear()
        self.P = P

    def tearDown(self):
        R.time.sleep = self._sleep
        self.srv.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def go(self, reason="test", **kw):
        return R.restart(self.P.get_instance("main"), reason, by="tester", **kw)


class Procedure(Env):
    def test_ok_path_journals_holds_verifies_and_audits(self):
        r = self.go()
        self.assertEqual(r["outcome"], "ok", r)
        self.assertEqual([s["step"] for s in r["steps"]], ["planned", "draining", "restarting", "verifying", "ok"])
        self.assertIn("recorded", r["verify"]["canary"])
        self.assertEqual(self.holds, [("hold", "main"), ("release", "main")])
        self.assertEqual(r["gateway"], dict(held=2, expired=0))
        self.assertTrue((self.tmp / "panel/restarts/main/last-good/manifest.json").exists())
        self.assertEqual(self.audit[-1][:3], ("tester", "system", "RESTART"))
        self.assertEqual(self.audit[-1][5], 200)
        self.assertEqual(R.report("main", r["id"])["outcome"], "ok")          # journal on disk

    def test_a_fleet_member_tells_its_primary(self):
        told = []
        self.P.fleet = types.SimpleNamespace(announce=lambda ev, iid, secs=0: (told.append((ev, iid, secs)), True)[1])
        try:
            r = self.go()
        finally:
            del self.P.fleet
        self.assertEqual(r["outcome"], "ok", r)
        s = R.DEFAULTS
        self.assertEqual(told, [("hold", "main", s["hold_s"] + s["drain_s"]), ("release", "main", 0)])

    def test_settings_that_never_come_up_are_rolled_back(self):
        self.go()                                                          # last-good = the good config
        self.params_file.write_text(json.dumps(dict(self.good, MODEL="/m/does-not-fit.gguf")))
        r = self.go()
        self.assertEqual(r["outcome"], "recovered", r)
        self.assertIn("did not come back", r["verify"]["reason"])
        self.assertEqual(json.loads(self.params_file.read_text())["MODEL"], self.good["MODEL"])
        self.assertEqual(self.srv.cfg["MODEL"], self.good["MODEL"])         # up again, on the known-good

    def test_a_known_good_that_never_comes_up_ends_failed_not_looping(self):
        bad = dict(self.good, MODEL="/m/does-not-fit.gguf")
        self.params_file.write_text(json.dumps(bad))
        snap = self.tmp / "snap"
        R._snapshot(self.P.get_instance("main"), snap)                   # a "known-good" that is bad too
        n0 = self.restarts_done
        r = self.go(known_good=str(snap))
        self.assertEqual(r["outcome"], "failed", r)
        self.assertIn("known-good configuration failed too", r["detail"])
        self.assertEqual(self.restarts_done - n0, 2)                       # the restart and one recovery

    def test_second_start_on_same_config_matches_canary(self):
        self.go()
        r = self.go()
        self.assertEqual(r["verify"]["canary"], "matches (2 starts)")

    def test_fallback_tier_is_caught_and_known_good_restored(self):
        self.go()                                                          # last-good = the good config
        self.params_file.write_text(json.dumps(dict(self.good, CTX=TOO_BIG_CTX)))
        r = self.go("apply")
        self.assertEqual(r["outcome"], "recovered", r)
        self.assertIn("fallback tier", r["detail"])
        self.assertEqual(json.loads(self.params_file.read_text()), self.good)   # files put back
        self.assertEqual(self.srv.cfg["CTX"], "131072")                         # server back on them
        self.assertEqual(self.audit[-1][5], 409)

    def test_silently_wrong_output_on_a_proven_config_fails_without_looping(self):
        self.go()
        self.go()                                                          # reproduced once: enforced
        self.srv.broken = True
        n0 = self.restarts_done
        r = self.go()
        self.assertEqual(r["outcome"], "failed")
        self.assertIn("silently wrong", r["detail"])
        self.assertEqual(self.restarts_done - n0, 2)                       # one try, one recovery, no loop

    def test_canary_not_enforced_before_it_reproduces(self):
        self.go()
        self.srv.broken = True
        r = self.go()
        self.assertEqual(r["outcome"], "ok")
        self.assertIn("not enforced", r["verify"]["canary"])

    def test_no_known_good_means_failed_not_guessing(self):
        self.params_file.write_text(json.dumps(dict(self.good, CTX=TOO_BIG_CTX)))
        r = self.go()
        self.assertEqual(r["outcome"], "failed")
        self.assertIn("no known-good", r["detail"])

    def test_explicit_known_good_snapshot(self):
        snap = R.snapshot_for(self.P.get_instance("main"), "autofit-x1-apply")
        self.params_file.write_text(json.dumps(dict(self.good, CTX=TOO_BIG_CTX)))
        r = self.go(known_good=snap)
        self.assertEqual(r["outcome"], "recovered")
        with self.assertRaises(ValueError):
            R.snapshot_for(self.P.get_instance("main"), "../escape")

    def test_drain_waits_for_real_requests(self):
        self.busy["n"] = 4
        r = self.go()
        self.assertEqual(r["outcome"], "ok")
        self.assertEqual(self.busy["n"], 0)                                # every busy poll consumed first

    def test_one_restart_per_instance(self):
        R._active["main"] = dict(id="x")
        with self.assertRaisesRegex(ValueError, "already in progress"):
            self.go()


class Handoff(Env):
    def setUp(self):
        super().setUp()
        R.set_settings(dict(instance="main", kv_handoff=True))
        self.srv.cached = list(range(1000))                                # a pinned agent session

    def test_session_survives_the_restart(self):
        seen = {}
        orig = self.P._systemctl

        def systemctl(verb, inst=None):                                    # look at the file mid-restart
            f = next(iter(Path(self.srv.slot_dir).glob("lp-handoff-*.bin")))
            seen["mode"] = stat.S_IMODE(f.stat().st_mode)
            return orig(verb, inst)
        self.P._systemctl = systemctl
        r = self.go()
        self.assertEqual(r["outcome"], "ok", r)
        self.assertEqual(seen["mode"], 0o600)                              # conversation tokens: owner only
        self.assertEqual(r["kv_restored"], [dict(slot=0, tokens=1000)])
        self.assertEqual(self.srv.cached, list(range(1000)))
        self.assertEqual(list(Path(self.srv.slot_dir).glob("lp-handoff-*")), [])   # deleted after use

    def test_changed_layout_discards_the_cache(self):
        new = dict(self.good, KV_TYPE="q4_1")
        self.params_file.write_text(json.dumps(new))
        r = self.go(intended=new)
        self.assertEqual(r["outcome"], "ok", r)
        self.assertEqual(r["kv_restored"], [])
        self.assertTrue(any("layout changed" in n for n in r["notes"]))
        self.assertEqual(list(Path(self.srv.slot_dir).glob("lp-handoff-*")), [])


class CrashRecovery(Env):
    def _journal(self, state, intended):
        d = self.tmp / "panel/restarts/main"
        d.mkdir(parents=True, exist_ok=True)
        rec = dict(id="20260925_120000_abcdef", instance="main", reason="test", by="tester", started="2026-09-25T12:00:00Z",
                   steps=[dict(step=state, t=time.time())], notes=[], intended=intended, known_good=None,
                   settings=R.settings("main"), kv=None, outcome=None, state=state)
        (d / f"{rec['id']}.json").write_text(json.dumps(rec))
        return rec["id"]

    def test_before_the_server_was_touched_is_aborted(self):
        rid = self._journal("draining", self.good)
        R.recover_on_startup()
        self.assertEqual(R.report("main", rid)["outcome"], "aborted")

    def test_after_the_restart_is_verified_and_finished(self):
        rid = self._journal("verifying", self.good)
        R.recover_on_startup()
        for _ in range(200):
            if R.report("main", rid).get("outcome"):
                break
            self._sleep(0.02)
        self.assertEqual(R.report("main", rid)["outcome"], "ok")


class Reporting(Env):
    def test_summary_and_csv(self):
        self.go()
        self.params_file.write_text(json.dumps(dict(self.good, CTX=TOO_BIG_CTX)))
        self.go()
        s = R.summary("main")
        self.assertEqual(s["restarts"], 2)
        self.assertEqual(s["outcomes"], dict(ok=1, recovered=1))
        self.assertEqual(s["caught"], 1)
        rows = R.export_csv("main").strip().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].startswith("id,started"))

    def test_manual_restart_refuses_during_a_measurement(self):
        self.P.benchlab = types.SimpleNamespace(active=lambda: True)
        with self.assertRaisesRegex(ValueError, "Bench run is active"):
            R.start(dict(instance="main"), user="alice")
        self.P.benchlab = types.SimpleNamespace(active=lambda: False)

    def test_start_uses_the_authenticated_user(self):
        R.start(dict(instance="main", by="someone-else"), user="alice")
        for _ in range(200):
            if not R._active:
                break
            self._sleep(0.02)
        self.assertEqual(self.audit[-1][0], "alice")


class GatewayHold(unittest.TestCase):
    def setUp(self):
        self._targets = GW.targets
        GW.targets = lambda: {"qwen": [("127.0.0.1", 1, None, "main")], "main": [("127.0.0.1", 1, None, "main")]}
        GW._holds.clear()

    def tearDown(self):
        GW.targets = self._targets
        GW._holds.clear()

    def test_no_hold_passes_straight_through(self):
        self.assertIsNone(GW.wait_if_held("qwen"))

    def test_held_until_released(self):
        GW.hold("main", 10)
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", GW.wait_if_held("qwen", poll=0.01)))
        th.start()
        time.sleep(0.1)
        self.assertTrue(th.is_alive())                                     # waiting, not failing
        stats = GW.release("main")
        th.join(2)
        self.assertEqual(out["r"], "released")
        self.assertEqual(stats["held"], 1)

    def test_hold_expires(self):
        GW.hold("main", 0.05)
        self.assertEqual(GW.wait_if_held("main", poll=0.01), "expired")


class AuditChain(unittest.TestCase):
    def test_restart_records_keep_the_chain_valid(self):
        with tempfile.TemporaryDirectory() as d:
            AU.bind(Path(d))
            AU.audit("auto-fit", "system", "RESTART", "/restart/auto-fit apply/ok", "main", 200)
            AU.audit("alice", "system", "RESTART", "/restart/operator/recovered", "main", 409)
            self.assertEqual(AU.verify(), dict(ok=True, lines=2))


if __name__ == "__main__":
    unittest.main()
