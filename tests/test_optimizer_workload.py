"""Optimizer: the explicit-candidates phase and workload-weighted scoring used by auto-fit.
Run from the panel folder:  python3 -m unittest discover tests"""
import contextlib, sys, tempfile, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optimizer as O  # noqa: E402


def run_with(workload=None, phases=("candidates",), candidates=()):
    return dict(id="r", instance="main", weights=dict(quality=0.5, speed=0.3, consistency=0.2),
                budget=dict(O.BUDGETS["quick"]), candidates=[], log=[], baseline_params=dict(UBATCH="512", CTX="131072"),
                baseline_headroom=2000,
                opts=dict(phases=list(phases), workload=workload, candidates=list(candidates), min_gain=0.005,
                          allow_quality_drop=0.0, budget="quick", thermal=dict(O.THERMAL_DEFAULTS)))


class WorkloadRate(unittest.TestCase):
    def test_harmonic_mean_weighted_by_decode_share(self):
        run = run_with(dict(depths=[8192, 65536], weights=[0.25, 0.75]))
        m = dict(curve=[dict(depth=8192, decode_tps=50.0), dict(depth=65536, decode_tps=25.0)])
        # time per token: .25/50 + .75/25 = 0.035 s  ->  28.57 t/s, not the 31.25 arithmetic mean
        self.assertEqual(O._workload_tps(run, m), 28.57)
        self.assertEqual(m["workload_depths"], [8192, 65536])

    def test_missing_depth_gives_no_rate(self):
        run = run_with(dict(depths=[8192, 65536], weights=[0.5, 0.5]))
        self.assertIsNone(O._workload_tps(run, dict(curve=[dict(depth=8192, decode_tps=50.0)])))
        self.assertIsNone(O._workload_tps(run_with(None), dict(curve=[dict(depth=8192, decode_tps=50.0)])))

    def test_score_uses_the_workload_rate_when_both_sides_have_it(self):
        run = run_with(dict(depths=[8192], weights=[1.0], prompt_tokens=400, output_tokens=300))
        base = dict(decode_tps=50.0, workload_tps=30.0, workload_depths=[8192], quality=0.9)
        fast_shallow = dict(decode_tps=60.0, workload_tps=27.0, workload_depths=[8192], quality=0.9)
        s = O._score(run, fast_shallow, base)
        self.assertEqual(s["speed_basis"], "workload")
        self.assertLess(s["speed_ratio"], 1.0)                     # faster at depth 0 is not faster here
        self.assertEqual(O._score(run, dict(fast_shallow, workload_tps=None), base)["speed_basis"], "suite")


    def test_request_time_counts_prompt_and_output(self):
        wl = dict(prompt_tokens=1000, output_tokens=300)
        # 1000 / 1000 t/s + 300 / 30 t/s = 11 s
        self.assertAlmostEqual(O._request_time(wl, dict(prefill_tps=1000, workload_tps=30)), 11.0)
        run = run_with(dict(depths=[8192], weights=[1.0], **wl))
        base = dict(prefill_tps=1000, workload_tps=30, workload_depths=[8192], quality=0.9)
        cand = dict(prefill_tps=1000, workload_tps=33, workload_depths=[8192], quality=0.9)
        self.assertAlmostEqual(O._score(run, cand, base)["speed_ratio"], 11 / (1 + 300 / 33), places=3)


class SpeedTrials(unittest.TestCase):
    """A speed-only trial in a workload run is judged at the workload's depths too."""
    def setUp(self):
        names = ("_apply_launch", "_speed_probe", "_run_tasks", "_model_extras", "_save", "_log")
        self.saved = {k: getattr(O, k) for k in names}
        O._apply_launch = lambda run, launch, label: None
        O._save = lambda run: None
        O._log = lambda run, msg: None
        self.extras = []

        def probe(run, cand, eff):
            return dict(decode_tps=40.0 if eff.get("UBATCH") == "1024" else 38.0, decode_cv=0.01,
                        prefill_tps=1500.0, turn_ms=500.0, draft_accept=None, context_tokens=None)
        O._speed_probe = probe
        O._run_tasks = lambda run, cand, eff, which, reps: dict(quality=0.9, agreement=None, suite_s=10.0, tasks={})

        def extras(run, cand):
            self.extras.append(cand["label"])
            fast = (cand.get("launch") or {}).get("UBATCH") == "1024"
            cand["metrics"]["curve"] = [dict(depth=d, decode_tps=33.0 if fast else 30.0) for d in (8192, 32768)]
        O._model_extras = extras

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(O, k, v)

    def test_speed_mode_measures_the_depth_mix(self):
        run = run_with(dict(depths=[8192, 32768], weights=[0.5, 0.5], prompt_tokens=500, output_tokens=400),
                       phases=("launch",))
        base = O._new_cand(run, "baseline (current saved settings)", "baseline")
        O._evaluate(run, base, "full")
        run["baseline_metrics"] = base["metrics"]
        c = O._new_cand(run, "UBATCH=1024", "launch", launch=dict(UBATCH="1024"))
        O._evaluate(run, c, "speed")
        self.assertEqual(self.extras, ["baseline (current saved settings)", "UBATCH=1024"])
        self.assertEqual(c["metrics"]["workload_tps"], 33.0)
        self.assertEqual(c["speed_basis"], "workload")
        self.assertGreater(c["speed_ratio"], 1.05)                   # 10 % at depth, not diluted

    def test_speed_candidate_scored_on_the_baselines_basis(self):
        # Standard/Thorough budgets measure the baseline's agreement over repeats; a speed-only
        # candidate runs the sanity tasks once. Same speed, same quality -> the same score.
        O._run_tasks = lambda run, cand, eff, which, reps: dict(
            quality=0.9, agreement=None if which == "sanity" else 0.8, suite_s=10.0, tasks={})
        run = run_with(None, phases=("launch",))
        base = O._new_cand(run, "baseline (current saved settings)", "baseline")
        O._evaluate(run, base, "full")
        run["baseline_metrics"] = base["metrics"]
        run["baseline_sanity"] = 0.9
        base.update(O._score(run, base["metrics"], base["metrics"]))
        c = O._new_cand(run, "THREADS=8", "launch", launch=dict(THREADS="8"))    # no speed effect
        O._evaluate(run, c, "speed")
        self.assertEqual(c["metrics"]["agreement"], 0.8)
        self.assertAlmostEqual(c["score"], base["score"], places=9)

    def test_no_workload_no_extra_measurements(self):
        run = run_with(None, phases=("launch",))
        c = O._new_cand(run, "UBATCH=1024", "launch", launch=dict(UBATCH="1024"))
        O._evaluate(run, c, "speed")
        self.assertEqual(self.extras, [])


class DepthCurve(unittest.TestCase):
    """_model_extras itself, against a stand-in server: decode at each workload depth."""
    def setUp(self):
        import depthcurve as D
        self.D = D
        self.saved = {k: getattr(O, k) for k in ("P", "_http", "_trial", "_gate", "_card_memory", "_set_step")}
        self.saved_filler = D._filler_tokens
        O.P = types.SimpleNamespace(api_base=lambda: "http://127.0.0.1:1")
        D._filler_tokens = lambda host, port, n: list(range(n))
        self.trials = []
        O._trial = lambda run, rec: self.trials.append(rec)
        O._gate = lambda run, cand: None
        O._card_memory = lambda devices: []
        O._set_step = lambda run, text: None
        O._http = lambda method, path, body, timeout=0: (200, dict(timings=dict(
            predicted_per_second=40.0 - len(body["prompt"]) / 4096, prompt_per_second=1400.0,
            prompt_n=len(body["prompt"]), draft_n=10, draft_n_accepted=7)))

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(O, k, v)
        self.D._filler_tokens = self.saved_filler

    def test_curve_at_each_depth_that_fits(self):
        run = run_with(dict(depths=[8192, 32768], weights=[0.5, 0.5]))
        run.update(devices=[])
        run["opts"]["curve_depths"] = [8192, 32768, 262144]          # the last is past CTX
        cand = dict(id="c01", label="x", launch={}, metrics={})
        O._model_extras(run, cand)
        m = cand["metrics"]
        self.assertNotIn("curve_error", m)
        self.assertEqual([c["depth"] for c in m["curve"]], [8192, 32768])
        self.assertEqual(m["curve"][1]["decode_tps"], 32.0)
        self.assertEqual([t["depth"] for t in self.trials], [8192, 32768])
        self.assertEqual(O._workload_tps(run, m), round(2 / (1 / 38.0 + 1 / 32.0), 2))


class CandidatesPhase(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(O, k) for k in ("_evaluate", "_fits", "_save", "_log")}
        self.saved_repo = O.S.build_repo
        O.S.build_repo = lambda n: ("repo", {"facts": 1})
        O._save = lambda run: None
        O._log = lambda run, msg: run["log"].append(msg)
        speed = {"baseline (current saved settings)": 30.0, "CTX=65536": 33.0, "SPEC_TYPE=none": 26.0,
                 "baseline (validation)": 30.0}

        def fake_eval(run, cand, mode):
            tps = speed.get(cand["label"], 33.0 if "CTX=65536" in cand["label"] else 30.0)
            cand["metrics"] = dict(quality=0.9, decode_tps=tps, workload_tps=tps, workload_depths=[8192],
                                   prefill_tps=1000, turn_ms=500, suite_s=100, task_set="core", agreement=1.0,
                                   decode_cv=0.01, tasks={})
            cand["status"] = "ok"
            cand.update(O._score(run, cand["metrics"], run.get("baseline_metrics") or cand["metrics"]))
            return cand
        O._evaluate = fake_eval

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(O, k, v)
        O.S.build_repo = self.saved_repo

    def test_the_best_candidate_that_fits_is_recommended(self):
        O._fits = lambda run, launch: (launch.get("PARALLEL") != "4", "estimated VRAM headroom 10 MiB")
        run = run_with(dict(depths=[8192], weights=[1.0]), candidates=[
            dict(label="CTX=65536", launch=dict(CTX="65536")), dict(label="SPEC_TYPE=none", launch=dict(SPEC_TYPE="none")),
            dict(label="PARALLEL=4", launch=dict(PARALLEL="4"))])
        O._search(run)
        rec = next(c for c in run["candidates"] if c["id"] == run["recommended"])
        self.assertIn("CTX=65536", rec["label"])
        skipped = [c for c in run["candidates"] if c["status"] == "skipped"]
        self.assertEqual([c["label"] for c in skipped], ["PARALLEL=4"])
        self.assertTrue(all(c["phase"] in ("baseline", "candidates", "validate") for c in run["candidates"]))

    def test_nothing_better_keeps_the_baseline(self):
        O._fits = lambda run, launch: (True, "fits")
        run = run_with(dict(depths=[8192], weights=[1.0]),
                       candidates=[dict(label="SPEC_TYPE=none", launch=dict(SPEC_TYPE="none"))])
        O._search(run)
        rec = next(c for c in run["candidates"] if c["id"] == run["recommended"])
        self.assertEqual(rec["phase"], "baseline")


class StartValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = {k: getattr(O, k) for k in ("P", "RUNS_DIR", "_worker", "_healthy", "_external_busy", "_run")}
        O.P = types.SimpleNamespace(
            get_instance=lambda iid: dict(id=iid, name=iid, device="0000:03:00.0", devices=["0000:03:00.0"]),
            using_instance=lambda inst: contextlib.nullcontext(), server_pid=lambda: 1,
            load_params=lambda: dict(CTX="131072", MODEL="/m/a.gguf"), estimate=lambda p: dict(vram=dict(headroom_mib=2000)),
            DEFAULTS=dict(CTX="", UBATCH="", PARALLEL="", SPEC_TYPE="", MODEL="", BACKEND="", PORT=""), gputune=None)
        O.RUNS_DIR = Path(self.tmp.name)
        O._worker = lambda run: None
        O._healthy = lambda: True
        O._external_busy = lambda: False
        O._run = None

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(O, k, v)
        self.tmp.cleanup()

    def test_refuses_model_swaps_unknown_keys_and_bad_workloads(self):
        bad = [dict(phases=["candidates"], candidates=[dict(launch=dict(MODEL="/m/b.gguf"))]),
               dict(phases=["candidates"], candidates=[dict(launch=dict(NOT_A_KEY="1"))]),
               dict(phases=["candidates"], candidates=[]),
               dict(phases=["candidates"], candidates=[dict(launch=dict(CTX=str(i)))for i in range(7)]),
               dict(phases=["launch"], workload=dict(depths=[8192, 16384], weights=[1.0]))]
        for body in bad:
            with self.assertRaises(ValueError, msg=body):
                O.start("main", body)
            O._run = None

    def test_accepts_a_workload_run(self):
        r = O.start("main", dict(phases=["candidates"], source="autofit",
                                 candidates=[dict(label="CTX=65536", launch=dict(CTX=65536))],
                                 workload=dict(depths=[500, 40000], weights=[0.3, 0.7])))
        self.assertTrue(r["ok"])
        self.assertEqual(O._run["opts"]["workload"], dict(depths=[1024, 40000], weights=[0.3, 0.7]))
        self.assertEqual(O._run["opts"]["curve_depths"], [1024, 40000])
        self.assertEqual(O._run["opts"]["candidates"][0]["launch"], dict(CTX="65536"))
        self.assertEqual(O._run["source"], "autofit")


if __name__ == "__main__":
    unittest.main()
