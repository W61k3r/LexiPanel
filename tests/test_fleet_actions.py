#!/usr/bin/env python3
"""Fleet v2: remote actions, drain and restart holds. A primary and a member are two separate
copies of fleet.py, each with its own panel folder; every message between them is a JSON round
trip through the same functions the routes call. The member's own code paths (start, stop,
safe restart, save settings) are stand-ins that record what they were asked to do."""
import contextlib, importlib.util, json, os, stat, sys, tempfile, threading, time, types, unittest
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gateway as GW  # noqa: E402
import auth as AU  # noqa: E402


def load_fleet(tag):
    spec = importlib.util.spec_from_file_location(f"fleet_{tag}", ROOT / "fleet.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Member:
    """The member box's panel, as far as fleet.py uses it."""
    def __init__(self, d):
        self.PANEL, self.MODELS = d / "panel", d / "models"
        self.PANEL.mkdir()
        self.MODELS.mkdir()
        (self.MODELS / "small.gguf").write_text("weights")
        self.running = {"main": 101, "gpu2": None}
        self.params = {"main": dict(MODEL=str(self.MODELS / "small.gguf"), CTX="8192", PORT="8081"),
                       "gpu2": dict(MODEL=str(self.MODELS / "small.gguf"), CTX="4096", PORT="8085")}
        self.calls, self.audits, self.cur = [], [], None
        self.measuring = None
        self.outcome = "ok"
        me = self
        self.restarts = types.SimpleNamespace(
            _active={}, _measuring=lambda: me.measuring,
            restart=lambda inst, reason, by="x", intended=None, known_good=None: me._restart(inst, reason, by, known_good),
            snapshot_for=lambda inst, tag: (me.calls.append(("snapshot", inst["id"], tag)), f"/snap/{tag}")[1])
        self.auth = types.SimpleNamespace(audit=lambda *a: me.audits.append(a))

    def _restart(self, inst, reason, by, known_good):
        self.calls.append(("restart", inst["id"], reason, by, known_good))
        return dict(id="r1", outcome=self.outcome, downtime_s=4.2, gateway=dict(held=3),
                    detail="canary differs" if self.outcome != "ok" else None)

    def instance_ids(self):
        return list(self.running)

    def get_instance(self, iid=None):
        if iid not in self.running:
            raise ValueError(f"no instance {iid!r}")
        return dict(id=iid, name=iid, legacy=iid == "main", engine="llama.cpp")

    def using_instance(self, inst):
        me = self

        @contextlib.contextmanager
        def cm():
            prev, me.cur = me.cur, inst["id"]
            try:
                yield inst
            finally:
                me.cur = prev
        return cm()

    def server_pid(self):
        return self.running[self.cur]

    def start_server(self):
        self.calls.append(("start", self.cur))
        self.running[self.cur] = 202
        return True, "starting via systemd --user"

    def stop_server(self):
        self.calls.append(("stop", self.cur))
        self.running[self.cur] = None
        return True, "stopped"

    def load_params(self, backend=None):
        return dict(self.params[self.cur])

    def save_params(self, new, backend=None):
        self.calls.append(("save", self.cur, dict(new)))
        self.params[self.cur].update(new)
        return self.params[self.cur]

    def live_cmdline_args(self):
        p = self.params[self.cur]
        return ["llama-server", "--host", "0.0.0.0", "--port", p["PORT"], "-m", p["MODEL"], "-a", "qwen"]

    @staticmethod
    def _argv_get(argv, names):
        return next((argv[i + 1] for i, a in enumerate(argv) if a in names and i + 1 < len(argv)), None)

    def gpu_devices(self, probe=False):
        return []

    def _meminfo_mb(self, k):
        return 32000


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "p").mkdir()
        (self.tmp / "m").mkdir()
        self.pr, self.mb = load_fleet("primary"), load_fleet("member")
        self.pr.MIN_GAP_S = 0                                  # reports back to back in a test
        self.primaryP = types.SimpleNamespace(PANEL=self.tmp / "p", instance_ids=lambda: [], fleet=self.pr,
                                              auth=types.SimpleNamespace(audit=lambda *a: self.p_audits.append(a)))
        self.p_audits = []
        GW.bind(self.primaryP)
        self.primaryP.gateway = GW
        GW._holds.clear()
        self.pr.bind(self.primaryP)
        self.M = Member(self.tmp / "m")
        self.mb.bind(self.M)
        self.pr.set_config(dict(role="primary", name="prime"))
        self.sent = []                                         # every reply the member got
        self.mb._post = self.post
        code = self.pr.new_join_code()["code"]
        self.mb.set_config(dict(role="member", name="rig-2", primary_url="https://127.0.0.1:8443", code=code, share=True))
        self.box = self.mb.config()["box_id"]

    def tearDown(self):
        GW._holds.clear()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def post(self, url, obj, token=None, timeout=15):
        """The network between the two boxes: JSON out, JSON back, errors as HTTP codes."""
        path, raw = urllib.parse.urlparse(url).path, json.dumps(obj).encode()
        hdr = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            r = {"/api/fleet/join": lambda: self.pr.join(json.loads(raw)),
                 "/api/fleet/report": lambda: self.pr.intake(hdr, raw),
                 "/api/fleet/event": lambda: self.pr.event(hdr, raw)}[path]()
        except PermissionError as e:
            return 403, dict(error=str(e))
        except ValueError as e:
            return 400, dict(error=str(e))
        r = json.loads(json.dumps(r))
        self.sent.append((path, r))
        return 200, r

    def allow(self, *actions, instances=("*",), **kw):
        self.mb.set_policy(dict(allow=list(actions), instances=list(instances), **kw))
        self.mb.send_now()                                    # the primary learns the policy; the key comes back

    def queue(self, action, **args):
        return self.pr.queue_action(dict(box_id=self.box, action=action, args=args), "alice")

    def deliver(self):
        """One report: the member takes what the primary sends and runs it."""
        self.mb.send_now()
        ran = []
        while not self.mb._jobs.empty():
            ran.append(self.mb.run_one(self.mb._jobs.get_nowait()))
        return ran

    def box_rec(self):
        return self.pr.boxes()[0]

    def actions(self):
        return {a["id"]: a for a in self.box_rec()["actions"]}


class KeyAndPolicy(Env):
    def test_nothing_is_allowed_by_default(self):
        self.mb.send_now()
        self.assertEqual(self.mb.policy()["allow"], [])
        b = self.box_rec()
        self.assertEqual((b["remote"]["allow"], b["remote"]["key"]), ([], "none"))
        with self.assertRaisesRegex(ValueError, "does not accept instance.restart"):
            self.queue("instance.restart", instance="main")
        self.assertIsNone(self.mb._key())                     # no key without an allowed action

    def test_the_key_is_sent_once_and_stays_private(self):
        tok = self.M.PANEL / "fleet" / "token"
        self.assertEqual(stat.S_IMODE(tok.stat().st_mode), 0o600)             # the join token, too
        self.allow("instance.restart")
        key = self.mb._key()
        self.assertRegex(key, r"^[0-9a-f]{64}$")
        f = self.M.PANEL / "fleet" / "action_key"
        self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        self.mb.send_now()
        self.assertEqual(self.box_rec()["remote"]["key"], "matches")
        self.assertNotIn(key, json.dumps(self.pr.status()))   # never shown by the primary
        f.unlink()                                            # lost on the member
        self.mb.send_now()
        self.assertNotIn("action_key", self.sent[-1][1])      # not sent again on request
        self.assertEqual(self.box_rec()["remote"]["key"], "lost on the box: re-key")
        self.pr.rekey(dict(box_id=self.box))
        self.mb.send_now()
        self.assertRegex(self.mb._key(), r"^[0-9a-f]{64}$")
        self.assertNotEqual(self.mb._key(), key)
        self.mb.send_now()
        self.assertEqual(self.box_rec()["remote"]["key"], "matches")

    def test_policy_validation(self):
        for bad in (dict(allow=["shell"]), dict(allow="instance.stop"), dict(allow=[], instances=["../x"])):
            with self.assertRaises(ValueError):
                self.mb.set_policy(bad)
        self.mb.set_policy(dict(allow=["instance.stop"], instances="main, gpu2"))
        self.assertEqual(self.mb.policy()["instances"], ["main", "gpu2"])


class Commands(Env):
    def test_restart_round_trip_audited_on_both_boxes(self):
        self.allow("instance.restart")
        a = self.queue("instance.restart", instance="main", reason="deploy check")
        self.assertEqual(a["state"], "queued")
        self.assertEqual(self.deliver(), ["ok"])
        self.assertEqual(self.M.calls, [("restart", "main", "deploy check", "fleet:alice", None)])
        self.assertEqual(self.actions()[a["id"]]["state"], "sent")          # the result travels with the next report
        self.mb.send_now()
        got = self.actions()[a["id"]]
        self.assertEqual((got["state"], got["restart"]["outcome"], got["restart"]["held"]), ("ok", "ok", 3))
        self.assertEqual(self.M.audits[-1][3], "/fleet/instance.restart/ok")
        self.assertEqual(self.p_audits[-1][3], f"/fleet/{self.box}/instance.restart/ok")
        log = self.mb.action_log()
        self.assertEqual([x["state"] for x in log[:2]], ["ok", "running"])
        self.assertEqual(json.loads((self.M.PANEL / "fleet" / "outbox.json").read_text()), [])

    def test_start_and_stop_go_through_the_members_own_functions(self):
        self.allow("instance.start", "instance.stop")
        self.queue("instance.start", instance="gpu2")
        self.queue("instance.stop", instance="main")
        self.assertEqual(self.deliver(), ["ok", "ok"])
        self.assertEqual(self.M.calls, [("start", "gpu2"), ("stop", "main")])

    def test_a_command_runs_once_even_when_sent_again(self):
        self.allow("instance.restart")
        a = self.queue("instance.restart", instance="main")
        self.mb.send_now()
        cmd = self.sent[-1][1]["actions"][0]
        (self.M.PANEL / "fleet" / "outbox.json").write_text("[]")          # its acknowledgement got lost
        self.mb.send_now()
        self.assertEqual(self.sent[-1][1]["actions"][0]["id"], a["id"])   # so the primary sends it again
        self.mb._receive(cmd, self.mb.config())                           # and a third copy
        self.assertEqual(self.mb._jobs.qsize(), 1)

    def test_altered_forged_misaddressed_late_and_replayed_commands_are_refused(self):
        self.allow("instance.restart", "instance.stop")
        self.queue("instance.restart", instance="main")
        self.mb.send_now()
        good = self.sent[-1][1]["actions"][0]
        self.mb._jobs.get_nowait()
        key, c = self.mb._key(), self.mb.config()

        def signed(**kw):
            cmd = dict(good, id="act_" + os.urandom(12).hex(), **kw)
            cmd["sig"] = self.mb._sign(key, cmd)
            return cmd
        altered = dict(good, id="act_" + "1" * 24, action="instance.stop")          # sig no longer matches
        forged = dict(signed(action="instance.stop"))
        forged["sig"] = self.mb._sign("ab" * 32, forged)                             # someone else's key
        now = int(time.time())
        cases = {"bad signature": [altered, forged],
                 "meant for another box": [signed(box_id="00000000-0000-0000-0000-000000000000")],
                 "expired": [signed(issued=now - 7200, expires=now - 3600)]}
        for why, cmds in cases.items():
            for cmd in cmds:
                self.assertIn(why, self.mb.verify(cmd, c), cmd)
        self.assertEqual(self.mb.verify(good, c), "seen")                           # replay of one it took
        self.assertEqual(self.mb.verify(dict(good, id="x"), c), "malformed")
        self.assertTrue(self.mb._jobs.empty())

    def test_the_member_rechecks_its_own_rules(self):
        self.allow("instance.restart", instances=["gpu2"])
        with self.assertRaisesRegex(ValueError, "instance main"):
            self.queue("instance.restart", instance="main")                        # refused early by the primary
        self.allow("instance.restart")
        a = self.queue("instance.restart", instance="main")
        self.mb.set_policy(dict(allow=[]))                                          # changed its mind meanwhile
        self.assertEqual(self.deliver(), [])
        self.mb.send_now()
        got = self.actions()[a["id"]]
        self.assertEqual(got["state"], "refused")
        self.assertIn("not allowed on this box", got["detail"])
        self.assertEqual(self.M.calls, [])

    def test_plain_http_needs_the_members_consent(self):
        cfg = self.M.PANEL / "fleet" / "config.json"
        c = json.loads(cfg.read_text())
        cfg.write_text(json.dumps(dict(c, primary_url="http://127.0.0.1:8443")))
        self.allow("instance.stop")
        self.queue("instance.stop", instance="main")
        self.assertEqual(self.deliver(), [])
        self.assertIn("plain http", self.mb.action_log()[0]["detail"])
        self.allow("instance.stop", allow_http=True)
        self.queue("instance.stop", instance="main")
        self.assertEqual(self.deliver(), ["ok"])

    def test_busy_member_refuses_and_says_why(self):
        self.allow("instance.restart")
        self.M.measuring = "a Bench run is active"
        self.queue("instance.restart", instance="main")
        self.assertEqual(self.deliver(), ["refused"])
        self.assertIn("a Bench run is active", self.mb.action_log()[0]["detail"])
        self.assertEqual(self.M.calls, [])

    def test_expired_and_cancelled_commands_are_never_delivered(self):
        self.allow("instance.stop")
        a = self.queue("instance.stop", instance="main")
        b = self.queue("instance.stop", instance="gpu2")
        self.pr.cancel_action(dict(box_id=self.box, id=b["id"]))
        f = self.tmp / "p" / "fleet" / "boxes" / f"{self.box}.json"
        rec = json.loads(f.read_text())
        rec["actions"][-2]["expires"] = int(time.time()) - 1
        f.write_text(json.dumps(rec))
        self.assertEqual(self.deliver(), [])
        self.assertEqual((self.actions()[a["id"]]["state"], self.actions()[b["id"]]["state"]), ("expired", "cancelled"))

    def test_results_wait_for_a_report_that_gets_through(self):
        self.allow("instance.stop")
        a = self.queue("instance.stop", instance="main")
        self.deliver()
        self.pr.set_config(dict(role="standalone"))                                # the primary is away
        self.mb.send_now()
        self.assertTrue(json.loads((self.M.PANEL / "fleet" / "outbox.json").read_text()))
        self.pr.set_config(dict(role="primary"))
        self.mb.send_now()
        self.assertEqual(self.actions()[a["id"]]["state"], "ok")

    def test_interrupted_action_is_reported_not_rerun(self):
        (self.M.PANEL / "fleet" / "running.json").write_text(json.dumps(dict(id="act_" + "a" * 24,
                                                                              action="instance.restart",
                                                                              args=dict(instance="main"))))
        self.mb._recover_interrupted()
        out = json.loads((self.M.PANEL / "fleet" / "outbox.json").read_text())
        self.assertEqual((out[0]["state"], out[0]["id"]), ("failed", "act_" + "a" * 24))
        self.assertIn("restarted during the action", out[0]["detail"])
        self.assertFalse((self.M.PANEL / "fleet" / "running.json").exists())
        self.assertTrue(self.mb._jobs.empty())


class SetSettings(Env):
    def test_only_model_choice_and_sizing(self):
        self.allow("instance.set")
        for params in (dict(PORT="9999"), dict(HOST="0.0.0.0"), dict(API_KEY="x"), dict(EXTRA_ARGS="--log-file /x"),
                       dict(SPEC_DRAFT_MODEL="/m/big.gguf"), dict(BACKEND="cpu"), dict(CTX="1\nPORT=1"),
                       dict(CTX=True), dict(CTX=[1]), {}):
            with self.assertRaises(ValueError, msg=params):
                self.queue("instance.set", instance="main", params=params)

    def test_deploy_a_model_then_restart_safely_from_a_snapshot(self):
        self.allow("instance.set")
        new = str(self.M.MODELS / "small.gguf")
        a = self.queue("instance.set", instance="main", params=dict(MODEL=new, CTX="16384"))
        self.assertEqual(self.deliver(), ["ok"])
        snap = next(c for c in self.M.calls if c[0] == "snapshot")
        self.assertEqual(self.M.calls.index(snap), 0)                              # the known-good comes first
        self.assertIn(("save", "main", dict(CTX="16384", MODEL=new)), self.M.calls)
        self.assertEqual(self.M.calls[-1], ("restart", "main", "fleet", "fleet:alice", f"/snap/fleet-{a['id']}"))

    def test_a_model_must_already_be_on_the_member(self):
        self.allow("instance.set")
        outside = self.tmp / "elsewhere.gguf"
        outside.write_text("x")
        for m in (str(outside), str(self.M.MODELS / "missing.gguf"), str(self.M.MODELS / ".." / "elsewhere.gguf")):
            self.queue("instance.set", instance="main", params=dict(MODEL=m))
            self.assertEqual(self.deliver(), ["refused"], m)
            self.assertIn("models folder", self.mb.action_log()[0]["detail"])
        self.assertFalse(any(c[0] == "save" for c in self.M.calls))

    def test_settings_that_do_not_verify_are_rolled_back_and_reported(self):
        self.allow("instance.set")
        self.M.outcome = "recovered"
        a = self.queue("instance.set", instance="main", params=dict(CTX="999999"))
        self.assertEqual(self.deliver(), ["failed"])
        self.mb.send_now()
        got = self.actions()[a["id"]]
        self.assertEqual((got["state"], got["restart"]["outcome"]), ("failed", "recovered"))
        self.assertIn("known-good", got["detail"])

    def test_saved_only_when_stopped_or_asked(self):
        self.allow("instance.set")
        self.queue("instance.set", instance="gpu2", params=dict(CTX="2048"))       # gpu2 is stopped
        self.queue("instance.set", instance="main", params=dict(CTX="2048"), restart=False)
        self.assertEqual(self.deliver(), ["ok", "ok"])
        self.assertFalse(any(c[0] == "restart" for c in self.M.calls))


class DrainAndHold(Env):
    def shared_report(self):
        self.mb.send_now()                                                         # share on: main is shared
        return self.box_rec()["report"]["shared"]

    def test_the_member_shares_its_lan_instances(self):
        s = self.shared_report()
        self.assertEqual([(x["instance"], x["port"], x["names"]) for x in s], [("main", 8081, ["qwen", "small.gguf"])])

    def test_drain_takes_a_box_out_of_the_gateway(self):
        self.shared_report()
        self.assertIn("qwen", GW.targets())
        self.pr.drain(dict(box_id=self.box, on=True), "alice")
        self.assertNotIn("qwen", GW.targets())
        self.assertEqual(self.box_rec()["drain"]["by"], "alice")
        self.pr.drain(dict(box_id=self.box, on=False))
        self.assertIn("qwen", GW.targets())

    def test_a_member_restart_holds_the_primarys_gateway(self):
        self.shared_report()
        self.assertTrue(self.mb.announce("hold", "main", 30))
        for _ in range(100):
            if GW._holds:
                break
            time.sleep(0.02)
        self.assertNotIn("qwen", GW.targets())                                     # not sent to a restarting server
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", GW.wait_if_held("qwen", poll=0.01)))
        th.start()
        time.sleep(0.15)
        self.assertTrue(th.is_alive())                                              # the only replica: it waits
        self.mb.announce("release", "main")
        th.join(3)
        self.assertEqual(out.get("r"), "released")
        self.assertIn("qwen", GW.targets())

    def test_what_the_primary_held_is_on_the_record(self):
        self.allow("instance.restart")
        self.shared_report()
        a = self.queue("instance.restart", instance="main")
        self.mb.send_now()                                                         # delivered: running
        tok = {"Authorization": "Bearer " + (self.M.PANEL / "fleet" / "token").read_text()}
        ev = lambda e: self.pr.event(tok, json.dumps(dict(box_id=self.box, event=e, instance="main", seconds=30)).encode())
        ev("hold")
        th = threading.Thread(target=GW.wait_if_held, args=("qwen", 0.01))
        th.start()
        time.sleep(0.1)
        self.assertEqual(ev("release")["held"], 1)
        th.join(2)
        b = self.box_rec()
        self.assertEqual((b["holds"][0]["instance"], b["holds"][0]["held"]), ("main", 1))
        self.assertEqual(self.actions()[a["id"]]["held_at_primary"], 1)

    def test_with_another_replica_nothing_waits(self):
        self.shared_report()
        self.pr.event({"Authorization": "Bearer " + (self.M.PANEL / "fleet" / "token").read_text()},
                      json.dumps(dict(box_id=self.box, event="hold", instance="main", seconds=30)).encode())
        GW.targets, orig = (lambda: {"qwen": [("192.0.2.9", 8081, None, "rig-3/main")]}), GW.targets
        try:
            t0 = time.time()
            self.assertIsNone(GW.wait_if_held("qwen"))
            self.assertLess(time.time() - t0, 0.5)
        finally:
            GW.targets = orig

    def test_events_are_token_checked_bounded_and_rate_limited(self):
        self.shared_report()
        tok = "Bearer " + (self.M.PANEL / "fleet" / "token").read_text()
        ev = lambda **kw: self.pr.event({"Authorization": tok}, json.dumps(dict(dict(box_id=self.box, event="hold",
                                                                                    instance="main", seconds=10), **kw)).encode())
        with self.assertRaises(PermissionError):
            self.pr.event({"Authorization": "Bearer nope"}, json.dumps(dict(box_id=self.box, event="hold")).encode())
        with self.assertRaisesRegex(ValueError, "not shared"):
            ev(instance="gpu2")
        with self.assertRaises(ValueError):
            ev(event="reboot")
        self.assertEqual(ev(seconds=10 ** 9)["seconds"], self.pr.HOLD_MAX_S)
        with self.assertRaisesRegex(ValueError, "too many events"):
            for _ in range(self.pr.EVENTS_PER_10MIN + 1):
                ev(event="release")

    def test_mid_restart_the_member_keeps_its_place(self):
        self.shared_report()
        self.M.running["main"] = None
        self.M.restarts._active["main"] = {}
        self.assertEqual([x["instance"] for x in self.shared_report()], ["main"])
        self.M.restarts._active.clear()
        self.assertEqual(self.shared_report(), [])


class Surface(unittest.TestCase):
    def test_routes_and_roles(self):
        self.assertIsNone(AU.needed("POST", "/api/fleet/event"))                 # token-checked by fleet.py
        for p in ("/api/fleet/action", "/api/fleet/action/cancel", "/api/fleet/drain", "/api/fleet/rekey",
                  "/api/fleet/policy"):
            self.assertEqual(AU.needed("POST", p), "admin", p)
        src = (ROOT / "panel.py").read_text()
        for p in ("/api/fleet/event", "/api/fleet/action", "/api/fleet/action/cancel", "/api/fleet/drain",
                  "/api/fleet/rekey", "/api/fleet/policy"):
            self.assertIn(f'p == "{p}"', src)
        caddy = (ROOT / "systemd" / "Caddyfile.new").read_text()
        self.assertIn("/api/fleet/event", caddy)


if __name__ == "__main__":
    unittest.main()
