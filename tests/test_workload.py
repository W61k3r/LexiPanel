"""Workload profile: truncation-safe ingest, bench exclusion, activity, envelope, findings.
Run from the panel folder:  python3 -m unittest discover tests"""
import json, os, sys, tempfile, time, types, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fakepanel  # noqa: E402
import workload as W  # noqa: E402

DAY = 86400


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.P = fakepanel.make(self.root)
        W.bind(self.P)
        W._state.clear(); W._act.clear(); W._env_cache.clear(); W._seen_measuring.clear(); W._find_cache.clear()
        self.inst = self.P.get_instance("main")

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, lines, mode="a"):
        return fakepanel.write_log(self.P, "main", lines, mode)

    def stored(self):
        return W._read_jsonl(W._dir("main") / "requests.jsonl")

    def seed(self, reqs):
        """Write request records straight into the store (for envelope / findings tests)."""
        with open(W._dir("main") / "requests.jsonl", "a") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        W._env_cache.clear()


class Ingest(Base):
    def test_reads_each_request_once_and_never_a_half_written_one(self):
        self.log(fakepanel.req_lines(12000) + fakepanel.req_lines(30000, dec=35.0))
        W._state["main"] = dict(backfilled=True)                  # skip the first-run backfill
        self.assertEqual(W.ingest(self.inst), 2)
        self.assertEqual(W.ingest(self.inst), 0)                   # nothing new
        lines = fakepanel.req_lines(50000, dec=30.0)
        path = self.log(lines[:2])                                 # timings written, release not yet
        with open(path, "a") as f:
            f.write(lines[2][:20])                                 # half a line, no newline
        self.assertEqual(W.ingest(self.inst), 0)
        with open(path, "a") as f:
            f.write(lines[2][20:] + "\n")
        self.assertEqual(W.ingest(self.inst), 1)
        s = self.stored()
        self.assertEqual([r["depth"] for r in s], [12000, 30000, 50000])
        self.assertEqual(s[2]["decode_tps"], 30.0)
        self.assertTrue(all(r["fp"] for r in s))

    def test_a_restart_that_truncates_the_log_is_followed(self):
        W._state["main"] = dict(backfilled=True)
        self.log(fakepanel.req_lines(10000) * 3)
        W.ingest(self.inst)
        self.log(fakepanel.req_lines(20000), mode="w")             # shorter new log
        self.assertEqual(W.ingest(self.inst), 1)
        # restarted and already longer than the old offset: the tail signature catches it
        self.log(fakepanel.req_lines(40000, task=9) * 6, mode="w")
        self.assertEqual(W.ingest(self.inst), 6)
        self.assertEqual([r["depth"] for r in self.stored()][-7:], [20000] + [40000] * 6)

    def test_first_run_backfills_archives_and_marks_what_predates_watching(self):
        arch = self.root / "logs" / "engine_debug_old.log"
        arch.write_text("".join(l + "\n" for l in fakepanel.req_lines(8000) * 4))
        os.utime(arch, (time.time() - 5 * DAY,) * 2)
        self.P.archives = [arch]
        self.log(fakepanel.req_lines(9000) * 2)                    # already in the live log
        self.assertEqual(W.ingest(self.inst), 6)
        s = self.stored()
        self.assertTrue(all(r.get("bf") for r in s))
        self.assertLess(s[0]["t"], time.time() - 4 * DAY)          # the archive's own time
        self.log(fakepanel.req_lines(9500))
        W.ingest(self.inst)
        self.assertFalse(self.stored()[-1].get("bf"))              # new traffic is live

    def test_lexipanels_own_measurements_are_tagged_and_left_out(self):
        W._state["main"] = dict(backfilled=True)
        self.P.optimizer = types.SimpleNamespace(_run=dict(instance="main"))
        self.log(fakepanel.req_lines(9000) * 5)
        W.ingest(self.inst)
        self.assertTrue(all(r.get("src") == "bench" for r in self.stored()))
        self.P.optimizer._run = None
        self.log(fakepanel.req_lines(9000))
        W.ingest(self.inst, now=time.time() + 120)                 # well after the run
        env = W.envelope(self.inst, fresh=True)
        self.assertEqual((env["requests"], env["bench_excluded"]), (1, 5))

    def test_config_fingerprint_ignores_port_and_log_file_but_not_ctx(self):
        with self.P.using_instance(self.inst):
            a = W.live_fp()[0]
            self.P.live_argv["main"] = self.P.live_argv["main"][:-4] + ["--port", "9999", "--spec-type", "draft-mtp"]
            self.assertEqual(W.live_fp()[0], a)
            self.P.live_argv["main"][4] = "65536"
            self.assertNotEqual(W.live_fp()[0], a)


class Activity(Base):
    def test_hourly_buckets_busy_full_and_quiet(self):
        t0 = 1_800_000_000 - (1_800_000_000 % 3600)                # on an hour boundary
        for i in range(0, 1800, 2):                                 # half an hour, one slot
            W.note_slots("main", 1 if 600 <= i < 900 else 0, 1, 131072, now=t0 + i)
        self.assertAlmostEqual(W.quiet_s("main", now=t0 + 1798), 1798 - 898, delta=3)
        W.note_slots("main", 0, 1, now=t0 + 3600 + 1)              # next hour flushes the last
        rec = W._read_jsonl(W._dir("main") / "activity.jsonl")[0]
        self.assertAlmostEqual(rec["busy"], 300, delta=4)
        self.assertAlmostEqual(rec["full"], 300, delta=4)
        self.assertEqual(rec["max_busy"], 1)

    def test_idle_windows_need_two_weeks_of_quiet(self):
        now = time.time()
        act = []
        for d in range(14):
            for hh in range(24):
                t = now - (14 - d) * DAY
                h = int(t // 3600) - (time.localtime(t).tm_hour) + hh
                busy = 1800 if 9 <= time.localtime(h * 3600).tm_hour < 18 else 0
                act.append(dict(h=h, samples=3600, busy=busy))
        m = W.idle_map(act)
        by_hour = {}
        for x in m:
            by_hour.setdefault(x["how"] % 24, set()).add(x["idle"])
        self.assertEqual(by_hour[3], {True})
        self.assertEqual(by_hour[12], {False})
        one_week = W.idle_map([a for a in act if a["h"] * 3600 > now - 7 * DAY])
        self.assertFalse(any(x["idle"] for x in one_week))          # one sighting is not a pattern


class Envelope(Base):
    def reqs(self, n, depth=lambda i: 8000 + (i % 10) * 3000, t0=None, **kw):
        t0 = t0 or time.time() - 6 * DAY
        return [dict(dict(t=int(t0 + i * 6 * DAY / n), depth=depth(i), prompt_tokens=400, eval_tokens=200,
                          decode_tps=40.0, prompt_tps=1400.0, fp="aaa"), **kw) for i in range(n)]

    def test_percentiles_mix_and_enough(self):
        self.seed(self.reqs(120))
        env = W.envelope(self.inst, fresh=True)
        self.assertTrue(env["enough"])
        self.assertEqual(env["depth"]["max"], 35000)
        self.assertEqual(env["slot_ctx"], 131072)
        mix = env["mix"]
        self.assertEqual(len(mix["depths"]), 3)
        self.assertAlmostEqual(sum(mix["weights"]), 1.0, places=3)
        self.assertEqual(mix["depths"], sorted(mix["depths"]))

    def test_mix_weights_follow_generated_tokens(self):
        rs = self.reqs(90, depth=lambda i: 4000 if i < 45 else 60000)
        for r in rs[45:]:
            r["eval_tokens"] = 1800                                # deep requests generate 9x more
        mix = W.workload_mix(rs)
        deep = mix["weights"][mix["depths"].index(max(mix["depths"]))]
        self.assertGreater(deep, 0.6)

    def test_slot_context_split_by_parallel_unless_unified(self):
        self.assertEqual(W.slot_ctx_of(dict(CTX="131072", PARALLEL="2")), 65536)
        self.assertEqual(W.slot_ctx_of(dict(CTX="131072", PARALLEL="2", KV_UNIFIED="on")), 131072)
        self.assertEqual(W.slot_ctx_of(dict(CTX="131072", PARALLEL="2"), live=70000), 70000)

    def test_compare_matches_depths(self):
        before = [dict(depth=9000, decode_tps=40.0)] * 10 + [dict(depth=100000, decode_tps=20.0)] * 10
        after = [dict(depth=9000, decode_tps=36.0)] * 10                    # only shallow traffic after
        c = W.compare(before, after)
        self.assertAlmostEqual(c["ratio"], 0.9, places=3)                   # not 36/30 across depths
        self.assertEqual(len(c["buckets"]), 1)


class Findings(Envelope):
    def ids(self, **params):
        self.P.params["main"].update(params)
        W._env_cache.clear()
        return {f["id"]: f for f in W.findings(self.inst, W.envelope(self.inst, fresh=True))}

    def test_little_data_says_learning_and_proposes_nothing(self):
        self.seed(self.reqs(20))
        f = self.ids()
        self.assertIn("learning", f)
        self.assertFalse([x for x in f.values() if x.get("candidate")])

    def test_context_never_reached(self):
        self.seed(self.reqs(120))                                  # deepest 35000 of 131072
        f = self.ids()["ctx_unused"]
        self.assertEqual(f["candidate"]["launch"], dict(CTX="49152"))
        self.assertGreater(f["evidence"]["freed_mib"], 3000)

    def test_context_limit_reached(self):
        self.seed(self.reqs(120, depth=lambda i: 128000 if i % 20 == 0 else 20000))
        f = self.ids()["ctx_limit"]
        self.assertGreaterEqual(f["evidence"]["near_limit"], 3)
        self.assertEqual(f["candidate"]["launch"], dict(CTX="147456"))

    def test_speculation_that_does_not_pay(self):
        self.seed(self.reqs(120, accept=0.3))
        self.assertEqual(self.ids()["spec_low"]["candidate"]["launch"], dict(SPEC_TYPE="none"))
        self.assertNotIn("spec_low", self.ids(SPEC_TYPE="none"))

    def test_parallel_slots_never_used(self):
        self.seed(self.reqs(120))
        now = time.time()
        with open(W._dir("main") / "activity.jsonl", "a") as f:
            for i in range(24 * 8):
                f.write(json.dumps(dict(h=int(now // 3600) - i, samples=3600, busy=600, full=0, max_busy=1, n_slots=2)) + "\n")
        self.assertEqual(self.ids(PARALLEL="2")["parallel_unused"]["candidate"]["launch"], dict(PARALLEL="1"))

    def test_all_slots_busy_is_reported_only_with_several_slots(self):
        self.seed(self.reqs(120))
        now = time.time()
        f = W._dir("main") / "activity.jsonl"
        f.write_text("".join(json.dumps(dict(h=int(now // 3600) - i, samples=3600, busy=300, full=300, max_busy=1,
                                             n_slots=1)) + "\n" for i in range(24 * 4)))
        self.assertNotIn("slots_full", self.ids())             # one slot busy is just busy
        f.write_text("".join(json.dumps(dict(h=int(now // 3600) - i, samples=3600, busy=600, full=300, max_busy=2,
                                             n_slots=2)) + "\n" for i in range(24 * 4)))
        self.assertIn("slots_full", self.ids(PARALLEL="2"))

    def test_real_decode_below_the_measured_curve(self):
        self.P.baseline = dict(source="curve_x", points=[dict(depth=8192, decode_tps=50.0), dict(depth=65536, decode_tps=40.0)])
        self.seed(self.reqs(60, t0=time.time() - 10 * DAY) + self.reqs(40, t0=time.time() - 2 * DAY, decode_tps=30.0))
        f = self.ids()["below_curve"]
        self.assertLess(f["evidence"]["ratio"], 0.85)

    def test_a_change_that_made_real_requests_slower(self):
        rs = self.reqs(60, depth=lambda i: 20000, fp="old", t0=time.time() - 8 * DAY) + \
            self.reqs(40, depth=lambda i: 20000, fp="new", decode_tps=32.0, t0=time.time() - 2 * DAY)
        self.seed(rs)
        f = self.ids()["change_slower"]
        self.assertAlmostEqual(f["evidence"]["change"]["ratio"], 0.8, places=2)


if __name__ == "__main__":
    unittest.main()
