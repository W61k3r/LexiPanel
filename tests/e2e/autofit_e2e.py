#!/usr/bin/env python3
"""Auto-fit end to end, against the REAL panel, optimizer, workload store and auto-fit loop.

Faked, and only these:
  * the llama-server process: demo/fake_llama.py takes llama-server's flags and answers
    /health /slots /props /tokenize /completion /v1/chat/completions, at a speed that
    follows its -ub / --spec-draft-n-max flags (speed_model.py), and writes llama-server's
    timing lines to its --log-file, which it truncates on start like the real one
  * start_server / stop_server / server_pid, to launch and find that process
  * the optimizer's chat-based probes (_speed_probe decode, _run_tasks task suite): grading the
    suite needs a real model. The depth curve at the workload's depths is REAL HTTP to the fake
  * the clock between loop ticks: auto-fit and ingest are ticked every 2 s, not 60 s / 30 s,
    and the activity tracker starts out having watched an idle server for an hour
Real traffic is written into the engine log the way llama-server logs a served request.

  bash tests/e2e/run.sh confirm|regress|interrupt [port]     (runs this on a temporary copy)
  autofit_e2e.py <panel_dir> <panel_port> confirm|regress|interrupt
    confirm    tune finds UBATCH=1024 (+7 % at depth), auto-applies it, restarts, real
               traffic confirms it, then the reshape experiment measures CTX and proposes it
    regress    same win on the benchmark, but real requests run 15 % slower afterwards:
               verification catches it and auto-fit rolls back and restarts
    interrupt  a real request arrives mid-experiment: stopped, saved settings restored
"""
import json, math, os, random, re, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
panel_dir, port, scenario = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, panel_dir)
sys.path.insert(0, str(HERE))
os.environ["INF01_PANEL_DIR"] = panel_dir
os.environ["PANEL_PORT"] = str(port)
import speed_model as SM  # noqa: E402
import panel as P  # noqa: E402

O, W, A = P.optimizer, P.workload, P.autofit
LLAMA_PORT = port + 100
LOG = Path(panel_dir) / "demo-engine.log"
BUSY = Path(panel_dir) / "demo-busy"
FAKE = dict(proc=None)
T0 = time.time()
rnd = random.Random(11)


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- the process
def fake_pid():
    p = FAKE["proc"]
    if P.INST()["id"] != "main" or p is None or p.poll() is not None:
        return None
    return p.pid


def fake_start():
    if fake_pid():
        return False, "already running"
    p = P.load_params()
    argv = [sys.executable, str(HERE / "fake_llama.py"), "-m", str(p.get("MODEL")), "-c", str(p.get("CTX"))]
    for flag, key in (("-ub", "UBATCH"), ("-b", "BATCH"), ("--cache-type-k", "KV_TYPE"), ("-np", "PARALLEL"),
                      ("--spec-type", "SPEC_TYPE"), ("--spec-draft-n-max", "SPEC_N_MAX"),
                      ("--spec-draft-p-min", "SPEC_P_MIN"), ("--cache-reuse", "CACHE_REUSE")):
        if str(p.get(key) or "") != "":
            argv += [flag, str(p[key])]
    argv += ["--port", str(p.get("PORT")), "--log-file", str(LOG)]
    LOG.write_text("")                                  # llama-server truncates its log on start
    FAKE["proc"] = subprocess.Popen(argv, env=dict(os.environ, FAKE_BUSY_FILE=str(BUSY)))
    for _ in range(50):
        try:
            if P.api_get("/health", timeout=1):
                break
        except Exception:
            pass
        time.sleep(0.1)
    FAKE.setdefault("starts", []).append(time.time())
    say(f"  server started: ubatch {p.get('UBATCH')}, batch {p.get('BATCH')}, spec-n {p.get('SPEC_N_MAX')}, "
        f"ctx {p.get('CTX')} (pid {FAKE['proc'].pid})")
    return True, "started (demo)"


def fake_stop():
    p = FAKE["proc"]
    if p and p.poll() is None:
        p.terminate()
        p.wait(5)
    FAKE["proc"] = None
    return True, "stopped (demo)"


P.server_pid = fake_pid
P.start_server = fake_start
P.stop_server = fake_stop
P.unit_installed = lambda inst=None: False


# ---------------------------------------------------------------- optimizer probes
def speed_probe(run, cand, eff):
    dec = []
    for i in range(run["budget"]["speed_repeats"]):
        O._set_step(run, f"{cand['label']}: speed probe {i + 1} (demo)")
        st, r = O._http("POST", "/completion", dict(prompt=list(range(300)), n_predict=64))
        dec.append(r["timings"]["predicted_per_second"])
    live = SM.params_of(P.live_cmdline_args())
    return dict(decode_tps=O._med(dec), decode_cv=O._cv(dec), prefill_tps=SM.prefill(run["budget"]["depth"], live),
                turn_ms=500.0, draft_accept=0.71, context_tokens=None)


def run_tasks(run, cand, eff, which, repeats):
    O._set_step(run, f"{cand['label']}: task suite '{which}' x{repeats} (demo: graded as unchanged)")
    O._sleep(0.5)
    return dict(quality=0.93, agreement=1.0 if repeats > 1 else None, suite_s=60.0, tasks={})


O._speed_probe = speed_probe
O._run_tasks = run_tasks
O._fits = lambda run, launch: (True, "fits (demo)")
P.depthcurve.FILLER_GLOB = str(Path(panel_dir) / "*.py")


# ---------------------------------------------------------------- real traffic
def real_requests(n, factor=1.0, params=None):
    """n served requests as llama-server logs them, at the speed the RUNNING settings give
    (times factor: what the benchmark did not see)."""
    live = params or SM.params_of(P.live_cmdline_args())
    lines = []
    for i in range(n):
        depth = min(60000, int(rnd.lognormvariate(math.log(17000), 0.75)))
        out = max(20, int(rnd.lognormvariate(math.log(320), 0.8)))
        new = max(30, int(rnd.lognormvariate(math.log(900), 1.0)))
        lines += SM.log_lines(depth, new, out, round(SM.decode(depth, live, noise=0.03) * factor, 2),
                              SM.prefill(depth, live), 9000 + i)
    with open(LOG, "a") as f:
        f.write("".join(l + "\n" for l in lines))


def seed(fp):
    subprocess.run([sys.executable, str(HERE / "seed_workload.py"), panel_dir], check=True)
    d = W._dir("main")
    (d / "autofit.json").unlink()
    txt = (d / "requests.jsonl").read_text().replace('"fp-demo-1"', f'"{fp}"')
    (d / "requests.jsonl").write_text(txt)
    cfg = json.loads((d / "configs.json").read_text())
    cfg[fp] = cfg.pop("fp-demo-1")
    (d / "configs.json").write_text(json.dumps(cfg))


# ---------------------------------------------------------------- run
inst = P.get_instance("main")
with P.using_instance(inst):
    cur = P.load_params()
    # the stand-in needs no projector or template; keep the backend the copy has if it validates
    demo = dict(cur, PORT=str(LLAMA_PORT), BATCH="2048", UBATCH="512", CTX="131072", SPEC_N_MAX="2",
                USE_MMPROJ="0", TEMPLATE_SRC="")
    try:
        P.save_params(demo, cur.get("BACKEND"))
    except ValueError:
        P.save_params(dict(demo, BACKEND="cpu"), "cpu")
    fake_start()
    fp0, summ0 = W.live_fp()
    say(f"engine log {P.ENGINE_LOG}; live fingerprint {fp0}")
seed(fp0)
h = time.localtime().tm_hour
A.set_settings("main", dict(mode="auto", window=f"{(h - 1) % 24:02d}-{(h + 3) % 24:02d}", quiet_min=5,
                            reshape=scenario == "confirm", max_per_week=4))
now = time.time()
W._act["main"] = dict(h=int(now // 3600), samples=0.0, busy=0.0, full=0.0, max_busy=0, first=now - 3600, last=None)
W.tick()                                       # first ingest: backfill + what the log already held
real_requests(40)
W.tick()
say(f"seeded 14 days of traffic; {len(W.requests('main', now - 60))} fresh real requests ingested")

threading.Thread(target=P.sampler, daemon=True).start()
srv = P.Threaded((P.BIND, port), P.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
say(f"panel UI on http://127.0.0.1:{port}/  (fake llama-server on :{LLAMA_PORT})")

seen, wrote_after, busy_since, result = {}, False, None, {}
deadline = time.time() + 900
while time.time() < deadline:
    W.tick()
    A.tick()
    st = A.load("main")
    run = O._run
    for e in st["experiments"]:
        k = json.dumps(dict({x: e.get(x) for x in ("state", "decision", "proposal", "rollback", "outcome")},
                            v=(e.get("verify") or {}).get("state"), pr=bool(e.get("pending_restart"))),
                       default=str, sort_keys=True)
        if seen.get(e["id"]) != k:
            seen[e["id"]] = k
            v = e.get("verify") or {}
            say(f"experiment {e['id']} {e['kind']}: state={e['state']} decision={e.get('decision')} "
                f"proposal={e.get('proposal')} verify={v.get('state')}"
                + (f" ({v.get('ratio')} over {v.get('matched')} requests)" if v.get("ratio") else "")
                + (f" rollback={e['rollback'].get('note')}" if e.get("rollback") else "")
                + (f" pending_restart={e['pending_restart']['reason']}" if e.get("pending_restart") else "")
                + (f"\n            outcome: {e.get('outcome')}" if e.get("outcome") else ""))
    if run and run.get("step") and seen.get("step") != run["step"]:
        seen["step"] = run["step"]
        say(f"  optimizer: {run['step']}")
    exp = next((e for e in st["experiments"] if e["kind"] == "tune"), None)
    if scenario == "interrupt" and exp and exp["state"] == "running" and run and busy_since is None \
            and "UBATCH" in str(run.get("step")):
        BUSY.write_text("1")
        busy_since = time.time()
        say("  >>> a real request arrives (slot 0 busy)")
    if busy_since and exp and exp.get("stop_reason") and BUSY.exists() and time.time() - busy_since > 14:
        BUSY.unlink()
        FAKE["busy_end"] = time.time()
        say("  >>> the real request finished")
    # real traffic after the change, once the optimizer's own last requests are out of the way
    v = (exp or {}).get("verify") or {}
    if v.get("state") == "pending" and not wrote_after and time.time() - W._seen_measuring.get("main", 0) > 65:
        if scenario == "regress":
            real_requests(45, factor=0.85, params=dict(UBATCH="512", SPEC_N_MAX="2"))
            say("  >>> 45 real requests served on the new settings, 15 % slower than before")
        else:
            real_requests(45)
            say("  >>> 45 real requests served on the new settings")
        wrote_after = True
    live = SM.params_of(P.live_cmdline_args()) if fake_pid() else {}
    if scenario == "interrupt" and exp and exp["state"] in ("stopped", "failed") and O._run is None:
        with P.using_instance(inst):
            saved = str(P.load_params().get("UBATCH"))
        restore_t = FAKE["starts"][-1]
        result = dict(state=exp["state"], outcome=exp.get("outcome"), live_ubatch=live.get("UBATCH"),
                      saved_ubatch=saved, restored_after_request_s=round(restore_t - FAKE.get("busy_end", 1e18), 1))
        ok = (exp["state"] == "stopped" and saved == "512" and live.get("UBATCH") == "512"
              and restore_t > FAKE.get("busy_end", 1e18))
        break
    if scenario == "regress" and exp and (exp.get("rollback") or {}).get("restarted_at"):
        result = dict(verify=exp["verify"].get("state"), ratio=exp["verify"].get("ratio"),
                      rollback=exp["rollback"].get("note"), live_ubatch=live.get("UBATCH"),
                      rejected=st.get("rejected"))
        ok = result["verify"] == "rolled-back" or result["verify"] == "regressed"
        ok = ok and live.get("UBATCH") == "512" and "UBATCH=1024" in (st.get("rejected") or {})
        break
    if scenario == "confirm" and exp and v.get("state") == "confirmed":
        rs = next((e for e in st["experiments"] if e["kind"] == "reshape" and e["state"] != "running"), None)
        if rs:
            result = dict(verify=v.get("state"), ratio=v.get("ratio"), matched=v.get("matched"),
                          live_ubatch=live.get("UBATCH"), reshape=rs.get("decision"), proposal=rs.get("label"),
                          gain=rs.get("gain"))
            ok = (live.get("UBATCH") == "1024" and rs.get("decision") == "proposal"
                  and rs.get("label") == "CTX=81920" and "frees VRAM" in str(rs.get("outcome")))
            result["outcome"] = rs.get("outcome")
            break
    time.sleep(2)
else:
    ok, result = False, dict(error="timed out")

reqs = W.requests("main", T0 - 1, include_bench=True)
result["bench_tagged"] = sum(1 for r in reqs if r.get("src") == "bench")
result["real_since_start"] = sum(1 for r in reqs if r.get("src") != "bench")
say("RESULT " + json.dumps(result, default=str))
say(("PASS " if ok else "FAIL ") + scenario)
hold = int(os.environ.get("HOLD", "0"))
if hold:
    say(f"holding the UI for {hold} s")
    time.sleep(hold)
fake_stop()
sys.exit(0 if ok else 1)
