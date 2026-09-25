"""Auto-fit: when it may start, what is due, keep / propose / apply, stopping on real traffic,
restart detection, verification on real traffic and rollback. A fake optimizer stands in.
Run from the panel folder:  python3 -m unittest discover tests"""
import json, sys, tempfile, time, types, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fakepanel  # noqa: E402
import workload as W  # noqa: E402
import autofit as A  # noqa: E402

DAY = 86400


class FakeOptimizer:
    GOALS = {"agentic": dict(speed=0.3), "speed": dict(speed=0.6), "quality": dict(speed=0.15)}
    MIN_GAIN = 0.005

    def __init__(self):
        self._run, self._conn = None, None
        self.started, self.stopped, self.applied, self.rolled = [], 0, [], []
        self.runs = {}

    def start(self, iid, body):
        rid = f"run{len(self.started)}"
        self.started.append(body)
        self._run = dict(id=rid, instance=iid, step="baseline")
        return dict(ok=True, id=rid)

    def stop(self):
        self.stopped += 1

    def read_run(self, rid):
        return self.runs[rid]

    def apply(self, rid, cid, target):
        self.applied.append((rid, cid))
        return dict(ok=True, applied={"UBATCH": dict(before="512", after="1024")})

    def rollback(self, rid):
        self.rolled.append(rid)
        return dict(ok=True, restored={"UBATCH": "512"}, kept=[], note="restored 1 setting(s)")


def run_record(rec_launch=None, rec_label="UBATCH=1024", wl=(40.0, 44.0), q=(0.9, 0.9), state="finished"):
    base = dict(id="c00", phase="baseline", label="baseline (current saved settings)", status="ok", score=0.85,
                metrics=dict(quality=q[0], workload_tps=wl[0]))
    cands = [base]
    rec = "c00"
    if rec_launch:
        cands.append(dict(id="c05", phase="validate", label=rec_label, status="ok", score=0.87, speed_ratio=1.1,
                          launch=rec_launch, metrics=dict(quality=q[1], workload_tps=wl[1])))
        rec = "c05"
    return dict(state=state, candidates=cands, recommended=rec,
                baseline_params=dict(UBATCH="512", CTX="131072", PARALLEL="1"))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.P = fakepanel.make(Path(self.tmp.name))
        self.O = FakeOptimizer()
        self.P.optimizer = self.O
        self.P.unit_installed = lambda inst=None: False
        self.P.stop_server = lambda: None
        self.restarts = []
        self.P.start_server = lambda: (self.restarts.append(1), (True, "started"))[1]
        W.bind(self.P)
        A.bind(self.P, self.O, W)
        W._state.clear(); W._act.clear(); W._env_cache.clear(); W._find_cache.clear(); A._ext.clear()
        self.inst = self.P.get_instance("main")
        self.now = time.time()
        self.fp = self.live_fp()

    def tearDown(self):
        self.tmp.cleanup()

    def live_fp(self):
        with self.P.using_instance(self.inst):
            return W.live_fp()[0]

    def seed(self, n=120, fp=None, t0=None, decode=40.0, depth=lambda i: 8000 + (i % 10) * 3000):
        t0 = t0 or self.now - 6 * DAY
        with open(W._dir("main") / "requests.jsonl", "a") as f:
            for i in range(n):
                f.write(json.dumps(dict(t=int(t0 + i * 6 * DAY / n), depth=depth(i), prompt_tokens=400,
                                        eval_tokens=200, decode_tps=decode, fp=fp or self.fp)) + "\n")
        W._env_cache.clear(); W._find_cache.clear()

    def quiet(self, minutes=60):
        W._act["main"] = dict(h=int(self.now // 3600), samples=1.0, busy=0.0, full=0.0, max_busy=0,
                              first=self.now - minutes * 60, last=self.now, busy_now=0, busy_at=self.now)

    def st(self, **settings):
        st = A.load("main")
        st["settings"].update(settings)
        return st

    def env(self):
        return W.envelope(self.inst, fresh=True)


class Gate(Base):
    def test_the_gates_in_order(self):
        self.seed(); self.quiet(60)
        st = self.st(window="00-23")
        env = self.env()
        self.P.server_pid = lambda: None
        self.assertIn("not running", A.can_start(self.inst, st, env, self.now)[1])
        self.P.server_pid = lambda: 4242
        self.O._run = dict(id="x", instance="main")
        self.assertIn("optimizer", A.can_start(self.inst, st, env, self.now)[1])
        self.O._run = None
        W._act["main"]["busy_now"] = 1
        self.assertIn("being served", A.can_start(self.inst, st, env, self.now)[1])
        W._act["main"]["busy_now"] = 0
        self.quiet(5)
        self.assertIn("quiet minutes", A.can_start(self.inst, st, env, self.now)[1])
        self.assertTrue(A.can_start(self.inst, st, env, self.now, manual=True)[0])   # by hand: no window, no quiet
        self.quiet(60)
        ok, why = A.can_start(self.inst, st, env, self.now)
        lt = time.localtime(self.now)
        self.assertEqual(ok, lt.tm_hour < 22, why)                  # the set window needs an hour left

    def test_learned_window_needs_two_weeks(self):
        self.seed(); self.quiet(60)
        ok, why = A.can_start(self.inst, self.st(), self.env(), self.now)
        self.assertFalse(ok)
        self.assertIn("two weeks", why)

    def test_weekly_limit(self):
        self.seed(); self.quiet(60)
        st = self.st(window="00-23", max_per_week=1)
        st["experiments"].append(dict(id="x1", kind="tune", started_t=int(self.now - DAY), state="done"))
        if time.localtime(self.now).tm_hour < 22:
            self.assertIn("weekly limit", A.can_start(self.inst, st, self.env(), self.now)[1])

    def test_one_change_at_a_time(self):
        self.seed(); self.quiet(60)
        st = self.st(window="00-23")
        st["experiments"].append(dict(id="x1", kind="tune", state="done", verify=dict(state="pending")))
        self.assertIn("one change at a time", A.can_start(self.inst, st, self.env(), self.now, manual=True)[1])


class Due(Base):
    def test_learning_then_tune_then_reshape(self):
        self.seed(20)
        self.assertIsNone(A.due(self.inst, self.st(), self.env(), self.fp, self.now)[0])
        self.seed(120)
        st = self.st()
        kind, why, _c = A.due(self.inst, st, self.env(), self.fp, self.now)
        self.assertEqual(kind, "tune")
        st["experiments"].append(dict(id="x1", kind="tune", state="done", fp_before=self.fp,
                                      finished_t=int(self.now), p90=self.env()["depth"]["p90"]))
        kind, why, cands = A.due(self.inst, st, self.env(), self.fp, self.now)
        self.assertEqual(kind, "reshape")                          # the deepest request leaves context unused
        self.assertEqual(cands[0]["launch"], dict(CTX="49152"))
        st["dismissed"]["CTX=49152"] = self.fp
        self.assertIsNone(A.due(self.inst, st, self.env(), self.fp, self.now)[0])

    def test_a_pending_proposal_holds_everything(self):
        self.seed()
        st = self.st()
        st["experiments"].append(dict(id="x1", kind="tune", decision="proposal", proposal="pending"))
        self.assertIsNone(A.due(self.inst, st, self.env(), self.fp, self.now)[0])

    def test_failed_attempts_back_off(self):
        self.seed()
        st = self.st()
        st["experiments"].append(dict(id="x1", kind="tune", state="stopped", fp_before=self.fp, started_t=int(self.now - 3600)))
        self.assertNotEqual(A.due(self.inst, st, self.env(), self.fp, self.now)[0], "tune")

    def test_body_is_workload_weighted(self):
        self.seed()
        body = A._body(self.inst, self.st(), self.env(), "tune", None)
        self.assertEqual(body["phases"], ["launch"])
        self.assertEqual(len(body["workload"]["depths"]), 3)
        self.assertEqual(body["source"], "autofit")
        self.assertEqual(body["min_gain"], 0.005)                  # 3 % margin: the noise floor wins
        st = self.st()
        st["settings"].update(goal="speed", min_gain=0.1)
        self.assertEqual(A._body(self.inst, st, self.env(), "tune", None)["min_gain"], 0.015)
        body = A._body(self.inst, self.st(), self.env(), "reshape", [dict(label="CTX=49152", launch=dict(CTX="49152"))])
        self.assertEqual(body["candidates"][0]["launch"], dict(CTX="49152"))


class Decide(Base):
    def exp(self, kind="tune"):
        return dict(id="x1", kind=kind, run_id="run0", state="running", fp_before=self.fp, started_t=int(self.now))

    def test_baseline_wins_means_keep(self):
        st, e = self.st(mode="auto"), self.exp()
        A.decide(self.inst, st, e, run_record(), self.now)
        self.assertEqual(e["decision"], "keep")
        self.assertEqual(self.O.applied, [])

    def test_auto_applies_a_clear_speed_only_win(self):
        st, e = self.st(mode="auto"), self.exp()
        A.decide(self.inst, st, e, run_record(dict(UBATCH="1024"), rec_label="combined: UBATCH=1024"), self.now)
        self.assertEqual(e["decision"], "applied")
        self.assertEqual(e["label"], "UBATCH=1024")                 # not the optimizer's validation label
        self.assertEqual(self.O.applied, [("run0", "c05")])
        self.assertAlmostEqual(e["gain"], 0.1, places=3)           # 44 / 40 at the workload's depths
        self.assertEqual(e["pending_restart"]["reason"], "apply")

    def test_everything_else_is_a_proposal(self):
        cases = [(dict(mode="auto"), dict(CTX="49152"), (40, 44), (0.9, 0.9), "more than speed"),
                 (dict(mode="propose"), dict(UBATCH="1024"), (40, 44), (0.9, 0.9), "propose mode"),
                 (dict(mode="auto"), dict(UBATCH="1024"), (40, 44), (0.9, 0.8), "quality"),
                 (dict(mode="auto"), dict(UBATCH="1024"), (40, 40.4), (0.9, 0.9), "margin")]
        for settings, launch, wl, q, why in cases:
            st, e = self.st(**settings), self.exp()
            A.decide(self.inst, st, e, run_record(launch, wl=wl, q=q), self.now)
            self.assertEqual(e["decision"], "proposal", why)
            self.assertIn(why.split()[0], e["outcome"])
        self.assertEqual(self.O.applied, [])

    def reshape_run(self, *cands):
        base = dict(id="c00", phase="baseline", label="baseline (current saved settings)", status="ok", score=0.85,
                    metrics=dict(quality=0.9, workload_tps=40.0))
        out = [base] + [dict(id=f"c{i + 1:02d}", phase="candidates", label=lab, status=status, score=0.85,
                             speed_ratio=sr, launch=launch, metrics=dict(quality=q, workload_tps=40.0 * (sr or 1)))
                        for i, (lab, launch, sr, q, status) in enumerate(cands)]
        return dict(state="finished", candidates=out, recommended="c00",
                    baseline_params=dict(UBATCH="512", CTX="131072", PARALLEL="2", SPEC_TYPE="draft-mtp"))

    def test_reshape_proposes_a_capacity_change_that_is_no_slower(self):
        st, e = self.st(mode="auto"), dict(self.exp("reshape"), findings={"CTX=49152": "ctx_unused",
                                                                            "SPEC_TYPE=none": "spec_low"})
        # the optimizer recommends the baseline (nothing is FASTER), yet freeing context is the point
        A.decide(self.inst, st, e, self.reshape_run(("CTX=49152", dict(CTX="49152"), 0.99, 0.9, "ok"),
                                                     ("SPEC_TYPE=none", dict(SPEC_TYPE="none"), 1.01, 0.9, "ok")),
                 self.now)
        self.assertEqual(e["decision"], "proposal")
        self.assertEqual((e["label"], e["diffs"]), ("CTX=49152", dict(CTX="49152")))
        self.assertIn("frees VRAM", e["outcome"])
        self.assertEqual(self.O.applied, [])                         # auto mode still does not apply it

    def test_reshape_keeps_when_nothing_is_worth_it(self):
        st = self.st(mode="auto")
        cases = [(("CTX=49152", dict(CTX="49152"), 0.92, 0.9, "ok"), "ctx_unused"),      # too much slower
                 (("CTX=49152", dict(CTX="49152"), 1.0, 0.85, "ok"), "ctx_unused"),      # quality lower
                 (("SPEC_TYPE=none", dict(SPEC_TYPE="none"), 1.01, 0.9, "ok"), "spec_low"),  # below the margin
                 (("CTX=147456", dict(CTX="147456"), None, 0.9, "skipped"), "ctx_limit")]
        for cand, fid in cases:
            e = dict(self.exp("reshape"), findings={cand[0]: fid})
            A.decide(self.inst, st, e, self.reshape_run(cand), self.now)
            self.assertEqual(e["decision"], "keep", cand)
        self.assertIn("did not fit: CTX=147456", e["outcome"])

    def test_reshape_speed_change_needs_the_margin(self):
        st, e = self.st(mode="propose"), dict(self.exp("reshape"), findings={"SPEC_TYPE=none": "spec_low"})
        A.decide(self.inst, st, e, self.reshape_run(("SPEC_TYPE=none", dict(SPEC_TYPE="none"), 1.06, 0.9, "ok")),
                 self.now)
        self.assertEqual(e["decision"], "proposal")
        self.assertAlmostEqual(e["gain"], 0.06)

    def test_a_change_once_rolled_back_is_never_auto_applied_again(self):
        st, e = self.st(mode="auto"), self.exp()
        st["rejected"] = {"UBATCH=1024": self.fp}
        A.decide(self.inst, st, e, run_record(dict(UBATCH="1024")), self.now)
        self.assertEqual(e["decision"], "proposal")

    def test_stopped_run(self):
        st, e = self.st(mode="auto"), self.exp()
        e["stop_reason"] = "a real request arrived"
        A.decide(self.inst, st, e, run_record(state="stopped"), self.now)
        self.assertEqual((e["state"], e["outcome"]), ("stopped", "a real request arrived"))


class Loop(Base):
    def running(self, mode="auto"):
        st = self.st(mode=mode, window="00-23")
        st["experiments"].append(dict(id="x1", kind="tune", run_id="run0", state="running",
                                      fp_before=self.fp, started_t=int(self.now)))
        A.save("main", st)
        self.O._run = dict(id="run0", instance="main", step="c01: task 3/9")

    def test_a_real_request_stops_the_experiment(self):
        self.seed(); self.quiet(60); self.running()
        W._act["main"].update(busy_now=1, busy_at=self.now)
        self.assertTrue(A.tick_one(self.inst, self.now))
        self.assertEqual(self.O.stopped, 0)                        # one poll is not enough
        A.tick_one(self.inst, self.now + 5)
        self.assertEqual(self.O.stopped, 1)
        self.assertEqual(A.load("main")["experiments"][-1]["stop_reason"], "a real request arrived")

    def test_over_budget_stops_it(self):
        self.seed(); self.quiet(60); self.running()
        A.tick_one(self.inst, self.now + 121 * 60)
        self.assertEqual(self.O.stopped, 1)

    def test_finished_run_is_decided_then_restarted_then_verified(self):
        self.seed(120, decode=40.0, depth=lambda i: 20000)
        self.quiet(60); self.running()
        self.O._run = None
        self.O.runs["run0"] = run_record(dict(UBATCH="1024"))
        A.tick_one(self.inst, self.now)
        e = A.load("main")["experiments"][-1]
        self.assertEqual(e["decision"], "applied")
        self.assertEqual(len(self.restarts), 1)                    # quiet, so restarted at once
        self.P.live_argv["main"] = self.P.live_argv["main"] + ["-ub", "1024"]
        new_fp = self.live_fp()
        A.tick_one(self.inst, self.now + 60)
        e = A.load("main")["experiments"][-1]
        self.assertEqual((e["fp_after"], e["verify"]["state"]), (new_fp, "pending"))
        # real traffic after the change runs slower at the same depths: auto mode rolls it back
        with open(W._dir("main") / "requests.jsonl", "a") as f:
            for i in range(30):
                f.write(json.dumps(dict(t=int(self.now + 120 + i), depth=20000, eval_tokens=200,
                                        decode_tps=33.0, fp=new_fp)) + "\n")
        A.tick_one(self.inst, self.now + 600)
        e = A._find(A.load("main"), "x1")
        self.assertEqual(e["verify"]["state"], "regressed")
        self.assertEqual(self.O.rolled, ["run0"])
        self.assertEqual(e["pending_restart"]["reason"], "rollback")
        self.assertEqual(A.load("main")["rejected"], {"UBATCH=1024": self.fp})
        self.assertEqual(len(A.load("main")["experiments"]), 1)   # nothing new starts before the restart

    def test_a_faster_change_is_confirmed(self):
        self.seed(120, decode=40.0, depth=lambda i: 20000)
        st = self.st(mode="auto")
        new_fp = "f" * 12
        st["experiments"].append(dict(id="x1", kind="tune", state="done", decision="applied", label="UBATCH=1024",
                                      run_id="run0", fp_before=self.fp, fp_after=new_fp, applied_at=int(self.now),
                                      restarted_at=int(self.now), verify=dict(state="pending")))
        A.save("main", st)
        with open(W._dir("main") / "requests.jsonl", "a") as f:
            for i in range(30):
                f.write(json.dumps(dict(t=int(self.now + 10 + i), depth=20000, eval_tokens=200,
                                        decode_tps=43.0, fp=new_fp)) + "\n")
        A.tick_one(self.inst, self.now + 600)
        self.assertEqual(A.load("main")["experiments"][-1]["verify"]["state"], "confirmed")
        self.assertEqual(self.O.rolled, [])

    def test_proposal_apply_and_dismiss(self):
        st = self.st(mode="propose")
        st["experiments"] += [dict(id="x1", kind="reshape", state="done", decision="proposal", proposal="pending",
                                   run_id="run0", cand_id="c02", label="CTX=49152", candidates=["CTX=49152"],
                                   fp_before=self.fp),
                              dict(id="x2", kind="reshape", state="done", decision="proposal", proposal="pending",
                                   run_id="run1", cand_id="c02", label="PARALLEL=1", candidates=["PARALLEL=1"],
                                   fp_before=self.fp)]
        A.save("main", st)
        e = A.apply_proposal(self.inst, "x1", restart=True)
        self.assertEqual((e["decision"], e["applied_by"]), ("applied", "you"))
        self.assertEqual(len(self.restarts), 1)
        A.dismiss(self.inst, "x2")
        self.assertEqual(A.load("main")["dismissed"], {"PARALLEL=1": self.fp})
        with self.assertRaises(ValueError):
            A.apply_proposal(self.inst, "x2")

    def test_settings_are_validated(self):
        for bad in (dict(mode="yolo"), dict(window="25-03"), dict(min_gain=0), dict(max_per_week=99), dict(goal="x")):
            with self.assertRaises(ValueError, msg=bad):
                A.set_settings("main", bad)
        self.assertEqual(A.set_settings("main", dict(mode="propose", window="02-06"))["window"], "02-06")


if __name__ == "__main__":
    unittest.main()
