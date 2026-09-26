"""Optimizer: find the best settings for one instance for agentic coding work.

What it measures, per candidate configuration
  speed        decode t/s (code generation, greedy), cold prefill t/s at a set
               context depth, time to first token for a follow-up agent turn
               on a cached prefix, and wall time of the task suite
  quality      weighted pass rate on optimize_suite: coding tasks graded by
               hidden unit tests, tool-use tasks graded by schema and by
               executing the edit, strict JSON, two-turn long-context lookup
  consistency  agreement of pass/fail across repeats of the same task, and
               the spread of decode speed across repeats
  stability    launch failures, server crashes and thermal aborts disqualify

How it searches
  launch   coordinate descent over speed knobs that need a restart (UBATCH,
           BATCH, SPEC_N_MAX, SPEC_P_MIN, CACHE_REUSE, THREADS off-GPU),
           each checked with the speed probe plus the sanity tasks
  kv       KV cache precision, full suite, context kept >= min_ctx and every
           candidate must fit the memory estimator first
  sampling reasoning effort, then temperature, then min_p - request-time
           settings, so no restarts
  candidates  explicit alternatives handed in by the caller (auto-fit, from what
           the workload profile found: context never used, slots never used,
           speculation that does not pay), each measured on the full suite
  validate the best combination is run again with repeats and compared

Workload weighting (auto-fit runs): given the depths the instance's real requests
reach and how often, every full evaluation also measures decode at those depths,
and the speed score is how much faster a typical request of that workload finishes:
its new prompt tokens at the cold prefill rate plus its generated tokens at the
effective decode rate over the depth mix (the harmonic mean, i.e. tokens per second
of time actually spent). Manual runs keep the suite's geometric mean.

Guarantees
  * The instance's saved parameter files and failure counter are snapshotted
    before the first change and restored byte-for-byte when the run ends,
    stops or errors, and at the next panel start if the panel died mid-run.
    Nothing is applied until the operator presses Apply on a result.
  * Every restart goes through save_params, so the full-size-draft, backend
    and port guards all apply; candidates that the estimator says will not fit
    are skipped, never launched.
  * A background thermal watch pauses before a request when the GPU is hot
    and aborts the candidate if it crosses the abort limit.
  * External traffic is waited out: requests are only sent when no slot is
    busy with someone else's work.
  * Everything is logged with fsync, so after a host freeze the run shows
    which candidate was in flight.

Engine-agnostic by construction: the only server surface used is
/v1/chat/completions (+ /health, and /slots and /tokenize when present).
"""
import http.client, json, math, os, shutil, statistics, threading, time, urllib.parse, uuid

import optimize_suite as S

P = None                       # the panel module, bound at import by panel.py
RUNS_DIR = None
_lock = threading.RLock()
_run = None                    # live state of the active run
_thread = None
_stop = threading.Event()
_conn_lock = threading.Lock()
_conn = None                   # in-flight HTTP connection, closed by stop()


def bind(panel_module):
    global P, RUNS_DIR
    P = panel_module
    RUNS_DIR = P.PANEL / "optimize"


# ============================================================================
# configuration
# ============================================================================
GOALS = {
    "agentic": dict(label="Agentic coding (balanced)", quality=0.5, speed=0.3, consistency=0.2),
    "speed":   dict(label="Speed first", quality=0.25, speed=0.6, consistency=0.15),
    "quality": dict(label="Quality first", quality=0.65, speed=0.15, consistency=0.2),
}
BUDGETS = {
    "quick":    dict(label="Quick", tasks="core", repeats=1, max_tokens=6000, depth=8192,
                     speed_repeats=2, knob_span="neighbors"),
    "standard": dict(label="Standard", tasks="all", repeats=2, max_tokens=10000, depth=16384,
                     speed_repeats=2, knob_span="all"),
    "thorough": dict(label="Thorough", tasks="all", repeats=3, max_tokens=16000, depth=32768,
                     speed_repeats=3, knob_span="all"),
}
LAUNCH_KNOBS = [
    ("UBATCH", ["256", "512", "1024", "2048"]),
    ("BATCH", ["512", "1024", "2048", "4096"]),
    ("SPEC_N_MAX", ["1", "2", "3", "4"]),
    ("SPEC_P_MIN", ["0", "0.5", "0.75"]),
    ("CACHE_REUSE", ["0", "256", "1024"]),
    ("THREADS", ["4", "6", "8"]),
]
KV_CHOICES = ["q8_0", "q5_1", "q4_1"]
EFFORTS = ["low", "medium", "high", "xhigh"]
TEMPS = ["0.2", "0.5", "0.8", "1.0"]
# params -> argv flags, used to prove a trial really launched with its values
# (a fallback tier would launch with different ones)
FLAG_OF = {"CTX": ("-c", "--ctx-size"), "KV_TYPE": ("--cache-type-k", "-ctk"),
           "UBATCH": ("--ubatch-size", "-ub"), "BATCH": ("--batch-size", "-b"),
           "SPEC_N_MAX": ("--spec-draft-n-max",), "SPEC_P_MIN": ("--spec-draft-p-min",),
           "CACHE_REUSE": ("--cache-reuse",), "THREADS": ("--threads", "-t"),
           "MODEL": ("-m", "--model")}
RUNTIME_KEYS = ("TEMP", "TOP_P", "TOP_K", "MIN_P", "REASONING_EFFORT")
# A setting only replaces the current one when its score beats it by this much.
# Decode rate alone wobbles +-5 % between identical runs; without a margin the
# search "keeps" knobs that won on noise and the recommendation drifts.
MIN_GAIN = 0.005
# At the end of a run, how long to let someone else's in-flight request finish on the trial
# configuration before the restart back to the saved one.
RESTORE_WAIT_S = 600
# amdgpu reports junction crit 110 / emergency 115 and memory crit 108 on the 7900 XTX.
# This card already runs 107-109 C junction under normal load, so pausing lower than
# that would measure a cold card that never exists in service.
THERMAL_DEFAULTS = dict(pause_c=109, resume_c=100, abort_c=112, mem_abort_c=106)


class Stopped(Exception):
    pass


class CandidateFailed(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


# ============================================================================
# persistence
# ============================================================================
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _public(run):
    """The run without its private working keys (the synthetic repo is ~100 KB)."""
    return {k: v for k, v in run.items() if not k.startswith("_")}


def _save(run):
    with _lock:
        _atomic_json(RUNS_DIR / run["id"] / "run.json", _public(run))


def _trial(run, rec):
    with open(RUNS_DIR / run["id"] / "trials.jsonl", "a") as f:
        f.write(json.dumps(dict(rec, ts=_now()), default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _log(run, msg):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
    with _lock:
        run["log"].append(line)
        del run["log"][:-400]
    with open(RUNS_DIR / run["id"] / "log.txt", "a") as f:
        f.write(line + "\n")


def list_runs():
    out = []
    if RUNS_DIR is None or not RUNS_DIR.is_dir():
        return out
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        try:
            r = json.loads((d / "run.json").read_text())
        except (OSError, ValueError):
            continue
        best = next((c for c in r.get("candidates", []) if c["id"] == r.get("recommended")), None)
        out.append(dict(id=r["id"], instance=r["instance"], state=r["state"], started=r["started"],
                        finished=r.get("finished"), goal=r["opts"]["goal"],
                        budget=r["opts"]["budget"], n_candidates=len(r.get("candidates", [])),
                        recommended=best and best["label"], needs_restart=r.get("needs_restart")))
    return out


def read_run(run_id):
    with _lock:
        if _run and _run["id"] == run_id:
            return json.loads(json.dumps(_public(_run), default=str))
    d = _run_dir(run_id)
    try:
        return json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        raise ValueError(f"no optimizer run {run_id!r}")


def _run_dir(run_id):
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise ValueError("bad run id")
    return RUNS_DIR / run_id


def status():
    with _lock:
        live = json.loads(json.dumps(_public(_run), default=str)) if _run else None
    return dict(active=live, runs=list_runs()[:20], goals=GOALS, budgets=BUDGETS,
                thermal_defaults=THERMAL_DEFAULTS, bwrap=bool(S._BWRAP))


def describe_suite():
    return dict(tasks=S.describe(), launch_knobs=LAUNCH_KNOBS, kv_choices=KV_CHOICES,
                efforts=EFFORTS, temps=TEMPS, goals=GOALS, budgets=BUDGETS)


# ============================================================================
# snapshot / restore of the instance's saved configuration
# ============================================================================
def _config_files():
    files = [P.PARAMS_ENV, P.FAIL_FILE] + [P._backend_file(b) for b in P.BACKENDS]
    return [os.fspath(f) for f in files]


def _snapshot(run):
    snap = RUNS_DIR / run["id"] / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, f in enumerate(_config_files()):
        if os.path.exists(f):
            shutil.copy2(f, snap / f"{i}")
            manifest[f] = str(i)
        else:
            manifest[f] = None
    _atomic_json(snap / "manifest.json", manifest)
    run["snapshot"] = manifest


def _restore_files(run_id):
    snap = RUNS_DIR / run_id / "snapshot"
    manifest = json.loads((snap / "manifest.json").read_text())
    for f, key in manifest.items():
        if key is None:
            if os.path.exists(f):
                os.unlink(f)
        else:
            shutil.copy2(snap / key, f)


def recover_on_startup():
    """A run the panel did not finish: put the saved files back, say so."""
    if RUNS_DIR is None or not RUNS_DIR.is_dir():
        return
    for d in RUNS_DIR.iterdir():
        try:
            r = json.loads((d / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if r.get("state") not in ("starting", "running", "stopping", "restoring"):
            continue
        try:
            if (d / "snapshot" / "manifest.json").exists():
                _restore_files(r["id"])
                r["restored"] = True
        except Exception as e:
            r["restore_error"] = str(e)
        flight = r.get("in_flight")
        for c in r.get("candidates", []):
            if c["id"] == flight and c.get("status") in ("running", "launching", "queued"):
                c["status"] = "interrupted"
                c["detail"] = ("the panel stopped while this candidate was running - if the host "
                               "froze, suspect this configuration")
        r["state"] = "interrupted"
        r["finished"] = _now()
        r["needs_restart"] = bool(r.get("applied_launch"))
        r["log"] = (r.get("log") or []) + [
            f"{time.strftime('%H:%M:%S', time.gmtime())} panel restarted mid-run: saved "
            "parameter files restored" + ("; the server may still be running trial launch "
                                          "settings - restart it" if r["needs_restart"] else "")]
        _atomic_json(d / "run.json", r)


# ============================================================================
# server access (engine-agnostic: OpenAI-compatible endpoints only)
# ============================================================================
def _headers():
    h = {"Content-Type": "application/json"}
    key = str(P.load_params().get("API_KEY") or "").strip()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _http(method, path, body=None, timeout=600):
    global _conn
    u = urllib.parse.urlparse(P.api_base())
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    with _conn_lock:
        _conn = conn
    try:
        if _stop.is_set():
            raise Stopped()
        conn.request(method, path, body=json.dumps(body) if body is not None else None,
                     headers=_headers())
        r = conn.getresponse()
        data = r.read()
        try:
            obj = json.loads(data or b"null")
        except ValueError:
            obj = {"raw": data[:500].decode(errors="replace")}
        return r.status, obj
    finally:
        with _conn_lock:
            _conn = None
        conn.close()


def _healthy():
    try:
        st, obj = _http("GET", "/health", timeout=5)
        return st == 200 and (not isinstance(obj, dict) or obj.get("status") in (None, "ok"))
    except Stopped:
        raise
    except Exception:
        return False


def _external_busy():
    try:
        st, obj = _http("GET", "/slots", timeout=5)
    except Stopped:
        raise
    except Exception:
        return False
    if st != 200 or not isinstance(obj, list):
        return False
    return any(s.get("is_processing") for s in obj if isinstance(s, dict))


def _request_fields(eff, max_tokens):
    body = dict(max_tokens=int(max_tokens), stream=False, cache_prompt=True)
    for k, field, cast in (("TEMP", "temperature", float), ("TOP_P", "top_p", float),
                           ("TOP_K", "top_k", int), ("MIN_P", "min_p", float)):
        v = str(eff.get(k, "")).strip()
        if v != "":
            try:
                body[field] = cast(float(v)) if cast is int else cast(v)
            except ValueError:
                pass
    effort = str(eff.get("REASONING_EFFORT") or "").strip()
    if effort and effort != "default":
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    return body


def _chat(run, cand, eff, messages, max_tokens, tools=None, extra=None, what=""):
    _gate(run, cand)
    body = _request_fields(eff, max_tokens)
    body["messages"] = messages
    if tools:
        body["tools"] = tools
    body.update(extra or {})
    pid_before = P.server_pid()
    t0 = time.time()
    try:
        st, resp = _http("POST", "/v1/chat/completions", body,
                         timeout=int(max_tokens / 8) + 300)
    except Stopped:
        raise
    except Exception as e:
        if _stop.is_set():
            raise Stopped()
        if run.get("thermal_abort"):
            raise CandidateFailed("thermal_abort", run["thermal_abort"])
        pid_now = P.server_pid()
        if not pid_now or pid_now != pid_before:
            raise CandidateFailed("crashed", f"server died during {what}: {e}")
        raise CandidateFailed("error", f"request failed during {what}: {e}")
    wall = time.time() - t0
    if st != 200:
        raise CandidateFailed("error", f"HTTP {st} during {what}: {str(resp)[:300]}")
    t = resp.get("timings") or {}
    usage = resp.get("usage") or {}
    return resp, dict(
        wall_s=round(wall, 2),
        prompt_n=t.get("prompt_n", usage.get("prompt_tokens")),
        prompt_ms=t.get("prompt_ms"),
        prompt_tps=t.get("prompt_per_second"),
        gen_n=t.get("predicted_n", usage.get("completion_tokens")),
        gen_tps=t.get("predicted_per_second") or (
            usage.get("completion_tokens", 0) / wall if wall and not t else None),
        draft_n=t.get("draft_n"), draft_accepted=t.get("draft_n_accepted"),
        finish=S._finish(resp))


# ============================================================================
# thermal watch and traffic gate
# ============================================================================
def _gpu_temps(devices):
    out = dict(junction=None, mem=None, edge=None, power_w=None)
    for pci in devices or []:
        base = f"/sys/bus/pci/devices/{pci}/hwmon"
        try:
            hw = [os.path.join(base, h) for h in os.listdir(base)]
        except OSError:
            continue
        for h in hw:
            for f in os.listdir(h):
                if f.startswith("temp") and f.endswith("_label"):
                    try:
                        label = open(os.path.join(h, f)).read().strip()
                        val = int(open(os.path.join(h, f.replace("_label", "_input"))).read()) / 1000
                    except (OSError, ValueError):
                        continue
                    if label in out and (out[label] is None or val > out[label]):
                        out[label] = val
            try:
                w = int(open(os.path.join(h, "power1_average")).read()) / 1e6
                out["power_w"] = round((out["power_w"] or 0) + w, 1)
            except (OSError, ValueError):
                pass
    return out


def _thermal_watch(run):
    th = run["opts"]["thermal"]
    hot = 0
    while not run.get("_done"):
        t = _gpu_temps(run["devices"])
        with _lock:
            run["thermal"] = dict(t, at=_now())
            c = run.get("_cand")
            if c is not None:
                pk = c.setdefault("thermal", dict(peak_junction=None, peak_mem=None))
                for k, src in (("peak_junction", "junction"), ("peak_mem", "mem")):
                    if t[src] is not None and (pk[k] is None or t[src] > pk[k]):
                        pk[k] = t[src]
                if t["power_w"] is not None:
                    pw = pk.setdefault("power", [])
                    pw.append(t["power_w"])
                    del pw[:-2000]
        over = ((t["junction"] or 0) >= th["abort_c"]) or ((t["mem"] or 0) >= th["mem_abort_c"])
        hot = hot + 1 if over else 0
        if hot >= 3 and not run.get("thermal_abort"):
            run["thermal_abort"] = (f"GPU junction {t['junction']} C / memory {t['mem']} C held "
                                    f"above the abort limit ({th['abort_c']}/{th['mem_abort_c']} C)")
            _log(run, "THERMAL ABORT: " + run["thermal_abort"])
            _drop_connection()
        time.sleep(2)


def _drop_connection():
    with _conn_lock:
        c = _conn
    if c is not None and c.sock is not None:
        try:
            import socket
            c.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _sleep(sec):
    if _stop.wait(sec):
        raise Stopped()


def _gate(run, cand):
    """Before each request: stop flag, thermal pause, external traffic."""
    if _stop.is_set():
        raise Stopped()
    if run.get("thermal_abort"):
        raise CandidateFailed("thermal_abort", run["thermal_abort"])
    th = run["opts"]["thermal"]
    t = _gpu_temps(run["devices"])
    if (t["junction"] or 0) >= th["pause_c"]:
        _set_step(run, f"cooling: junction {t['junction']} C >= {th['pause_c']} C")
        _log(run, f"thermal pause at junction {t['junction']} C, waiting for {th['resume_c']} C")
        t0 = time.time()
        while (t["junction"] or 0) > th["resume_c"] and time.time() - t0 < 900:
            _sleep(5)
            t = _gpu_temps(run["devices"])
        cand.setdefault("notes", []).append(f"thermal pause {int(time.time() - t0)} s")
    waited = 0
    while _external_busy():
        if waited == 0:
            _set_step(run, "waiting: the server is busy with someone else's request")
            _log(run, "external request in progress - waiting for the slot to free")
        _sleep(5)
        waited += 5
    if waited:
        cand["external_wait_s"] = cand.get("external_wait_s", 0) + waited


def _set_step(run, text):
    with _lock:
        run["step"] = text
        run["step_at"] = _now()


# ============================================================================
# launching trial configurations
# ============================================================================
def _eff(run, cand):
    e = dict(run["baseline_params"])
    e.update(cand.get("launch") or {})
    e.update(cand.get("runtime") or {})
    return e


def _fits(run, launch):
    vals = dict(run["baseline_params"], **launch)
    try:
        est = P.estimate(vals)
        head = est["vram"]["headroom_mib"]
    except Exception as e:
        return False, f"estimator failed: {e}"
    need = min(400, run["baseline_headroom"]) if run["baseline_headroom"] is not None else 400
    if head < need:
        return False, f"estimated VRAM headroom {head} MiB < {need} MiB"
    try:
        rb = P.ram_budget(vals, include_others=True)
        if rb.get("verdict") == "impossible":
            return False, "host RAM budget: " + rb.get("detail", "impossible")
    except Exception:
        pass
    return True, f"fits, est. headroom {head} MiB"


def _live_matches(launch_full):
    argv = P.live_cmdline_args()
    bad = []
    for k, flags in FLAG_OF.items():
        want = str(launch_full.get(k, "")).strip()
        if want == "":
            continue
        got = next((argv[argv.index(f) + 1] for f in flags if f in argv and argv.index(f) + 1 < len(argv)), None)
        if got is None:
            continue
        try:
            same = math.isclose(float(got), float(want))
        except ValueError:
            same = got == want
        if not same:
            bad.append(f"{k} wanted {want}, running {got}")
    return bad


def _restart_and_wait(run, what, timeout=900):
    pid0 = P.server_pid()
    inst = P.INST()
    if inst["legacy"] and P.unit_installed():
        ok, msg = P._systemctl("restart")
    else:
        if pid0:
            P.stop_server()
            _sleep(2)
        ok, msg = P.start_server()
    if not ok:
        raise CandidateFailed("launch_failed", f"restart refused: {msg}")
    t0 = time.time()
    _set_step(run, f"launching {what}")
    while time.time() - t0 < timeout:
        _sleep(3)
        pid = P.server_pid()
        if pid and pid != pid0 and _healthy():
            return pid
        us = P.unit_state() or {}
        if us.get("active") == "failed":
            break
    tail = P.sh(f"journalctl -u {P.UNIT} -n 25 --no-pager 2>/dev/null" if inst["legacy"]
                else f"tail -n 25 {P.LAUNCH_OUT}", timeout=10)
    raise CandidateFailed("launch_failed", f"not healthy after {int(time.time() - t0)} s. "
                                           f"Last log lines:\n{tail[-1500:]}")


def _apply_launch(run, launch, label):
    """Make the server run baseline + launch diffs. No-op when already there."""
    if (run.get("applied_launch") or {}) == (launch or {}):
        return
    vals = dict(run["baseline_params"], **(launch or {}))
    P.save_params(vals, vals.get("BACKEND"))
    with _lock:
        run["applied_launch"] = dict(launch or {})
    _save(run)
    _restart_and_wait(run, label)
    bad = _live_matches(vals)
    try:
        P.FAIL_FILE.write_text("0\n")            # healthy: don't let the next trial fall a tier
    except OSError:
        pass
    if bad:
        raise CandidateFailed("launch_failed", "server came up with other settings (a fallback "
                                               "tier?): " + "; ".join(bad))
    _set_step(run, f"warming up {label}")
    _http("POST", "/v1/chat/completions",
          dict(messages=[{"role": "user", "content": "Say OK."}], max_tokens=8, stream=False),
          timeout=300)


def _back_to_baseline(run):
    try:
        with _lock:
            run["applied_launch"] = {"__unknown__": True}
        _apply_launch(run, {}, "baseline (recovery)")
        return True
    except (CandidateFailed, Exception) as e:
        _log(run, f"could not return to baseline: {e}")
        return False


# ============================================================================
# measurement
# ============================================================================
def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def _cv(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2 or not statistics.mean(xs):
        return None
    return round(statistics.pstdev(xs) / statistics.mean(xs), 4)


def _speed_probe(run, cand, eff):
    b = run["budget"]
    repo, facts = run["_repo"]
    dec, pre, turn, acc = [], [], [], []
    for i in range(b["speed_repeats"]):
        _set_step(run, f"{cand['label']}: speed probe {i + 1}/{b['speed_repeats']} - decode")
        _, m = _chat(run, cand, dict(eff, TEMP="0"), S.SPEED_PROMPT, 700, what="decode probe")
        dec.append(m["gen_tps"])
        if m["draft_n"]:
            acc.append(m["draft_accepted"] / m["draft_n"])
        _trial(run, dict(cand=cand["id"], kind="speed.decode", **m))
        nonce = uuid.uuid4().hex[:12]
        _set_step(run, f"{cand['label']}: speed probe {i + 1} - cold prefill at ~{b['depth']} tokens")
        _, m1 = _chat(run, cand, eff, S.needle_messages(repo, facts, nonce), 1, what="prefill probe")
        pre.append(m1["prompt_tps"] or (m1["prompt_n"] / m1["wall_s"] if m1["prompt_n"] else None))
        _trial(run, dict(cand=cand["id"], kind="speed.prefill", **m1))
        _set_step(run, f"{cand['label']}: speed probe {i + 1} - follow-up agent turn")
        _, m2 = _chat(run, cand, eff, S.needle_messages(repo, facts, nonce, turn=2), 1,
                      what="agent-turn probe")
        turn.append(m2["prompt_ms"] if m2["prompt_ms"] is not None else m2["wall_s"] * 1000)
        _trial(run, dict(cand=cand["id"], kind="speed.turn", **m2))
    return dict(decode_tps=_med(dec), decode_cv=_cv(dec), prefill_tps=_med(pre),
                turn_ms=_med(turn), draft_accept=_med(acc), context_tokens=None)


def _run_tasks(run, cand, eff, which, repeats):
    b = run["budget"]
    tasks = S.tasks_for(which)
    results = {}
    wall = 0.0
    repo, facts = run["_repo"]
    use_needle = which != "sanity"
    total = repeats * (len(tasks) + (2 if use_needle else 0))
    k = 0
    for rep in range(repeats):
        for t in tasks:
            k += 1
            _set_step(run, f"{cand['label']}: task {k}/{total} {t['id']} (repeat {rep + 1})")
            resp, m = _chat(run, cand, eff, t["messages"], b["max_tokens"], tools=t.get("tools"),
                            what=t["id"])
            ok, detail = t["grade"](resp)
            wall += m["wall_s"]
            results.setdefault(t["id"], []).append(ok)
            msg = S._msg(resp)
            _trial(run, dict(cand=cand["id"], kind="task", task=t["id"], rep=rep, ok=ok,
                             detail=detail, **m,
                             content=None if ok else (msg.get("content") or "")[-4000:],
                             tool_calls=None if ok else msg.get("tool_calls")))
        if use_needle:
            nonce = uuid.uuid4().hex[:12]
            for turn in (1, 2):
                k += 1
                _set_step(run, f"{cand['label']}: task {k}/{total} context.needle turn {turn}")
                resp, m = _chat(run, cand, eff, S.needle_messages(repo, facts, nonce, turn=turn),
                                b["max_tokens"], what=f"needle turn {turn}")
                ok, detail = S.grade_needle(resp, facts, turn)
                wall += m["wall_s"]
                results.setdefault("context.needle", []).append(ok)
                _trial(run, dict(cand=cand["id"], kind="task", task=f"context.needle.t{turn}",
                                 rep=rep, ok=ok, detail=detail, **m))
    weights = {t["id"]: t["weight"] for t in tasks}
    weights["context.needle"] = 1.0
    wsum = sum(weights[tid] * len(v) for tid, v in results.items())
    quality = sum(weights[tid] * sum(v) for tid, v in results.items()) / wsum if wsum else None
    agree = None
    if repeats > 1:
        per = [max(sum(v), len(v) - sum(v)) / len(v) for v in results.values()]
        agree = round(statistics.mean(per), 4) if per else None
    return dict(quality=round(quality, 4) if quality is not None else None, agreement=agree,
                suite_s=round(wall / repeats, 1),
                tasks={tid: dict(passed=sum(v), runs=len(v)) for tid, v in results.items()})


def _workload_tps(run, m):
    """Effective decode rate over the workload's depth mix: sum(w) / sum(w_i / tps_i).
    Time is what a user waits, so rates are averaged harmonically, weighted by how often
    requests land at each depth. None unless every workload depth was measured."""
    wl = run["opts"].get("workload")
    curve = {c["depth"]: c.get("decode_tps") for c in m.get("curve") or [] if c.get("decode_tps")}
    if not wl or not curve:
        return None
    pairs = [(w, curve.get(d)) for d, w in zip(wl["depths"], wl["weights"]) if w > 0]
    if not pairs or any(t is None for _w, t in pairs):
        return None
    m["workload_depths"] = [d for d, w in zip(wl["depths"], wl["weights"]) if w > 0]
    return round(sum(w for w, _t in pairs) / sum(w / t for w, t in pairs), 2)


def _request_time(wl, x):
    """Seconds a typical request of this workload takes: its new prompt tokens at the cold
    prefill rate plus its generated tokens at the workload's effective decode rate."""
    pn, on = float(wl.get("prompt_tokens") or 0), float(wl.get("output_tokens") or 1)
    pre = pn / x["prefill_tps"] if pn and x.get("prefill_tps") else 0.0
    return pre + on / x["workload_tps"]


def _score(run, m, base):
    w = run["weights"]
    wl = run["opts"].get("workload")
    if (wl and m.get("workload_tps") and (base or {}).get("workload_tps")
            and m.get("workload_depths") == base.get("workload_depths")):
        # Workload runs: speed is how much faster a typical request of THIS workload finishes.
        # The suite's geometric mean would dilute a real decode-at-depth gain ~4x.
        speed_ratio, basis = _request_time(wl, base) / _request_time(wl, m), "workload"
    else:
        ratios, basis = [], "suite"
        for k, higher in (("decode_tps", True), ("prefill_tps", True), ("turn_ms", False),
                          ("suite_s", False)):
            a, b = m.get(k), (base or {}).get(k)
            if a and b and (k != "suite_s" or m.get("task_set") == base.get("task_set")):
                ratios.append(a / b if higher else b / a)
        speed_ratio = math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else 1.0
    cv = m.get("decode_cv")
    stab = max(0.0, 1 - 4 * cv) if cv is not None else 1.0
    cons = 0.6 * m["agreement"] + 0.4 * stab if m.get("agreement") is not None else stab
    q = m.get("quality") if m.get("quality") is not None else 0.0
    score = w["quality"] * q + w["consistency"] * cons + w["speed"] * min(1.0, speed_ratio / 2)
    return dict(speed_ratio=round(speed_ratio, 3), consistency=round(cons, 4),
                score=round(score, 4), speed_basis=basis)


def _evaluate(run, cand, mode):
    """mode: 'full' (suite + speed) or 'speed' (speed + sanity tasks)."""
    with _lock:
        run["in_flight"] = cand["id"]
        run["_cand"] = cand
        run["thermal_abort"] = None
        cand["status"] = "launching"
        cand["started"] = _now()
    _save(run)
    b = run["budget"]
    try:
        _apply_launch(run, cand.get("launch") or {}, cand["label"])
        cand["status"] = "running"
        eff = _eff(run, cand)
        m = _speed_probe(run, cand, eff)
        if mode == "full":
            q = _run_tasks(run, cand, eff, b["tasks"], cand.get("repeats") or b["repeats"])
            m["task_set"] = b["tasks"]
        else:
            q = _run_tasks(run, cand, eff, "sanity", 1)
            m["task_set"] = "sanity"
            base = run.get("baseline_metrics") or {}
            base_sanity = run.get("baseline_sanity")
            if base_sanity is not None:
                # speed knobs do not change the maths: assume baseline quality
                # unless the sanity tasks got worse
                q["sanity_quality"] = q["quality"]
                q["quality"] = (base.get("quality") if q["quality"] >= base_sanity
                                else round((base.get("quality") or 0) * q["quality"] /
                                           max(base_sanity, 1e-9), 4))
                q["quality_basis"] = "assumed from baseline; sanity tasks checked"
                # Consistency on the baseline's basis too: one sanity pass cannot measure how
                # often answers repeat, so without this a speed candidate was scored on decode
                # stability alone against a baseline scored on 0.6 agreement + 0.4 stability,
                # a bias of up to ~0.02 score that had nothing to do with the knob.
                if base.get("agreement") is not None:
                    q["agreement"] = base["agreement"]
        m.update(q)
        th = cand.get("thermal") or {}
        if th.get("power"):
            m["avg_power_w"] = round(statistics.mean(th["power"]), 1)
            m["tps_per_watt"] = (round(m["decode_tps"] / m["avg_power_w"], 3)
                                 if m.get("decode_tps") and m["avg_power_w"] else None)
            th["power_samples"] = len(th.pop("power"))
        m["repeats"] = cand.get("repeats") or b["repeats"] if mode == "full" else 1
        m["peak_junction"] = th.get("peak_junction")
        m["peak_mem"] = th.get("peak_mem")
        cand["metrics"] = m
        if run["opts"].get("workload") and "models" not in run["opts"]["phases"]:
            # Decode at the depths real requests reach, for speed-only trials too: a knob is
            # kept or dropped on what it does to this workload, not on a depth-0 probe.
            _model_extras(run, cand)
            m["workload_tps"] = _workload_tps(run, m)
        cand.update(_score(run, m, run.get("baseline_metrics")))
        cand["status"] = "ok"
    except Stopped:
        cand["status"] = "stopped"
        raise
    except CandidateFailed as e:
        cand["status"], cand["detail"] = e.status, e.detail
        _log(run, f"{cand['label']}: {e.status} - {e.detail[:300]}")
        if e.status == "thermal_abort":
            run["thermal_aborts"] = run.get("thermal_aborts", 0) + 1
            run["thermal_abort"] = None
            _cooldown(run)
        if e.status in ("crashed", "launch_failed"):
            run["failures"] = run.get("failures", 0) + 1
            if not _back_to_baseline(run):
                raise RuntimeError("the server did not come back on the baseline configuration")
    finally:
        with _lock:
            cand["finished"] = _now()
            run["_cand"] = None
            run["in_flight"] = None
        _save(run)
    if run.get("thermal_aborts", 0) >= 2:
        raise RuntimeError("two thermal aborts - stopping so the card can cool; see the log")
    if run.get("failures", 0) >= 4:
        raise RuntimeError("four launch failures or crashes - stopping")
    return cand


def _cooldown(run):
    th = run["opts"]["thermal"]
    t0 = time.time()
    while time.time() - t0 < 900:
        t = _gpu_temps(run["devices"])
        if (t["junction"] or 0) <= th["resume_c"]:
            return
        _set_step(run, f"cooling after thermal abort: junction {t['junction']} C")
        _sleep(5)


def _new_cand(run, label, phase, launch=None, runtime=None, repeats=None):
    c = dict(id=f"c{len(run['candidates']):02d}", label=label, phase=phase,
             launch=dict(launch or {}), runtime=dict(runtime or {}), status="queued",
             repeats=repeats)
    with _lock:
        run["candidates"].append(c)
    return c


def _eligible(run, c):
    if c.get("status") != "ok":
        return False
    bq = (run.get("baseline_metrics") or {}).get("quality")
    q = (c.get("metrics") or {}).get("quality")
    if bq is None or q is None:
        return True
    return q >= bq - run["opts"]["allow_quality_drop"] - 1e-9


def _best(run, cands):
    ok = [c for c in cands if _eligible(run, c)]
    return max(ok, key=lambda c: c["score"]) if ok else None


def _values_for(run, key, values, current):
    if run["budget"]["knob_span"] == "all":
        return [v for v in values if v != current]
    if current in values:
        i = values.index(current)
        return [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]
    return values[:2]


# ============================================================================
# the run
# ============================================================================
def _worker(run):
    thermal = threading.Thread(target=_thermal_watch, args=(run,), daemon=True)
    with P.using_instance(run["instance"]):
        try:
            _snapshot(run)
            run["state"] = "running"
            _save(run)
            thermal.start()
            _search(run)
            run["state"] = "finished"
        except Stopped:
            run["state"] = "stopped"
            _log(run, "stopped by operator")
        except Exception as e:
            run["state"] = "error"
            run["error"] = str(e)
            _log(run, f"ERROR: {e}")
        finally:
            _finish(run)


def _finish(run):
    stopped = _stop.is_set()
    _stop.clear()                 # restore must be able to talk to the server
    run["state_before_restore"] = run["state"]
    prev = run["state"]
    run["state"] = "restoring"
    _set_step(run, "restoring the saved configuration")
    _save(run)
    try:
        _restore_files(run["id"])
        run["restored"] = True
        _log(run, "saved parameter files restored")
        if run.get("applied_launch"):
            if run.get("was_running"):
                # A real request that arrived mid-run (auto-fit stops a run for exactly that) is
                # being served by the trial configuration: let it finish instead of cutting it off.
                t0 = time.time()
                while time.time() - t0 < RESTORE_WAIT_S and _external_busy():
                    _set_step(run, "waiting for a real request to finish before restoring")
                    time.sleep(2)
                _log(run, "restarting the server on the saved configuration")
                with _lock:
                    run["applied_launch"] = run["applied_launch"]
                try:
                    _restart_and_wait(run, "saved configuration")
                    run["applied_launch"] = {}
                    try:
                        P.FAIL_FILE.write_text("0\n")
                    except OSError:
                        pass
                except (CandidateFailed, Stopped) as e:
                    run["needs_restart"] = True
                    _log(run, f"restart to the saved configuration did not come up healthy: {e}")
            else:
                P.stop_server()
                run["applied_launch"] = {}
    except Exception as e:
        run["restore_error"] = str(e)
        run["needs_restart"] = True
        _log(run, f"RESTORE ERROR: {e}")
    run["state"] = "stopped" if stopped and prev == "running" else prev
    run["finished"] = _now()
    run["step"] = None
    run["_done"] = True
    _save(run)
    global _run
    with _lock:
        _run = None


def _search(run):
    if "models" in run["opts"]["phases"]:
        return _compare_models(run)
    b = run["budget"]
    phases = run["opts"]["phases"]
    base = run["baseline_params"]
    run["_repo"] = S.build_repo(int(b["depth"] * 3.4))
    _log(run, f"synthetic repository built for ~{b['depth']} tokens of context")

    # 1. baseline -------------------------------------------------------------
    c0 = _new_cand(run, "baseline (current saved settings)", "baseline")
    _evaluate(run, c0, "full")
    if c0["status"] != "ok":
        raise RuntimeError(f"baseline did not complete: {c0.get('detail')}")
    run["baseline_metrics"] = c0["metrics"]
    c0.update(_score(run, c0["metrics"], c0["metrics"]))
    sanity = [t["id"] for t in S.tasks_for("sanity")]
    passed = sum(c0["metrics"]["tasks"].get(t, {}).get("passed", 0) for t in sanity)
    runs = sum(c0["metrics"]["tasks"].get(t, {}).get("runs", 0) for t in sanity)
    run["baseline_sanity"] = passed / runs if runs else None
    _log(run, f"baseline: quality {c0['metrics']['quality']}, decode {c0['metrics']['decode_tps']} t/s, "
              f"prefill {c0['metrics']['prefill_tps']} t/s, turn {c0['metrics']['turn_ms']} ms")
    _save(run)
    best_launch, best_runtime = {}, {}
    best_cand = c0

    # 1b. explicit candidates (auto-fit) --------------------------------------
    # Alternatives, not steps: each is the baseline plus its own overrides.
    if "candidates" in phases:
        tried = []
        for spec in run["opts"].get("candidates") or []:
            c = _new_cand(run, spec["label"], "candidates", launch=dict(spec["launch"]),
                          runtime=best_runtime)
            fits, why = _fits(run, c["launch"])
            if not fits:
                c.update(status="skipped", detail=why)
                _log(run, f"{c['label']}: skipped - {why}")
                continue
            _evaluate(run, c, "full")
            tried.append(c)
        win = _best(run, tried + [best_cand])
        if win is not None and win is not best_cand and win["score"] > best_cand["score"] + run["opts"]["min_gain"]:
            best_cand, best_launch = win, dict(win["launch"])
            _log(run, f"candidates: keeping {win['label']} (score {win['score']})")

    # 2. launch speed knobs ---------------------------------------------------
    if "launch" in phases:
        spec_on = any(t.strip().startswith("draft-") for t in str(base.get("SPEC_TYPE") or "").split(","))
        gpu_only = str(base.get("BACKEND")) != "cpu" and int(base.get("NGL") or 0) >= 99
        for key, values in LAUNCH_KNOBS:
            if key.startswith("SPEC_") and not spec_on:
                continue
            if key == "THREADS" and gpu_only:
                continue
            cur = str(dict(base, **best_launch).get(key, ""))
            tried = []
            for v in _values_for(run, key, values, cur):
                launch = dict(best_launch, **{key: v})
                eff = dict(base, **launch)
                if key in ("UBATCH", "BATCH") and int(eff.get("BATCH") or 0) < int(eff.get("UBATCH") or 0):
                    continue
                c = _new_cand(run, f"{key}={v}", "launch", launch=launch, runtime=best_runtime)
                fits, why = _fits(run, launch)
                if not fits:
                    c.update(status="skipped", detail=why)
                    _log(run, f"{c['label']}: skipped - {why}")
                    continue
                _evaluate(run, c, "speed")
                tried.append(c)
            win = _best(run, tried + [best_cand])
            if win is not None and win is not best_cand and win["score"] > best_cand["score"] + run["opts"]["min_gain"]:
                best_cand, best_launch = win, dict(win["launch"])
                _log(run, f"launch: keeping {key}={best_launch[key]} (score {win['score']})")

    # 3. KV cache precision ---------------------------------------------------
    if "kv" in phases:
        cur = str(base.get("KV_TYPE"))
        tried = []
        for kv in _values_for(run, "KV_TYPE", KV_CHOICES, cur):
            launch = dict(best_launch, KV_TYPE=kv)
            fits, why = _fits(run, launch)
            if not fits:
                for ctx in sorted((int(x) for x in P.OPT_CTX), reverse=True):
                    if ctx < run["opts"]["min_ctx"] or ctx >= int(base.get("CTX") or 0):
                        continue
                    fits, why = _fits(run, dict(launch, CTX=str(ctx)))
                    if fits:
                        launch["CTX"] = str(ctx)
                        break
            label = f"KV_TYPE={kv}" + (f", CTX={launch['CTX']}" if "CTX" in launch else "")
            c = _new_cand(run, label, "kv", launch=launch, runtime=best_runtime)
            if not fits:
                c.update(status="skipped", detail=f"no context >= {run['opts']['min_ctx']} fits: {why}")
                continue
            _evaluate(run, c, "full")
            tried.append(c)
        win = _best(run, tried + [best_cand])
        if win is not None and win is not best_cand and win["score"] > best_cand["score"] + run["opts"]["min_gain"]:
            best_cand, best_launch = win, dict(win["launch"])
            _log(run, f"kv: keeping {win['label']} (score {win['score']})")

    # 4. sampling and reasoning (no restarts) ---------------------------------
    if "sampling" in phases:
        for key, values in (("REASONING_EFFORT", EFFORTS), ("TEMP", TEMPS)):
            cur = str(dict(base, **best_runtime).get(key, ""))
            tried = []
            for v in _values_for(run, key, values, cur):
                runtime = dict(best_runtime, **{key: v})
                c = _new_cand(run, f"{key}={v}", "sampling", launch=best_launch, runtime=runtime)
                _evaluate(run, c, "full")
                tried.append(c)
            win = _best(run, tried + [best_cand])
            if win is not None and win is not best_cand and win["score"] > best_cand["score"] + run["opts"]["min_gain"]:
                best_cand, best_runtime = win, dict(win["runtime"])
                _log(run, f"sampling: keeping {key}={best_runtime[key]} (score {win['score']})")
        if run["opts"]["budget"] == "thorough":
            runtime = dict(best_runtime, MIN_P="0.05")
            if str(dict(base, **best_runtime).get("MIN_P")) != "0.05":
                c = _new_cand(run, "MIN_P=0.05", "sampling", launch=best_launch, runtime=runtime)
                _evaluate(run, c, "full")
                win = _best(run, [c, best_cand])
                if win is c and c["score"] > best_cand["score"] + run["opts"]["min_gain"]:
                    best_cand, best_runtime = c, runtime

    # 5. validate the combination ---------------------------------------------
    # Baseline and winner are re-measured back to back at the same repeat count,
    # so agreement, thermal state and drift are comparable. Scores in this pool
    # are relative to the re-measured baseline; ties go to the baseline.
    combo = dict(launch=best_launch, runtime=best_runtime)
    rec = c0
    if best_launch or best_runtime:
        reps = max(2, b["repeats"])
        base_v = c0
        if b["repeats"] < reps:
            base_v = _new_cand(run, "baseline (validation)", "validate", repeats=reps)
            _evaluate(run, base_v, "full")
        label = "combined: " + ", ".join(f"{k}={v}" for k, v in {**best_launch, **best_runtime}.items())
        combo_c = _new_cand(run, label, "validate", launch=best_launch, runtime=best_runtime,
                            repeats=reps)
        _evaluate(run, combo_c, "full")
        if base_v.get("status") == "ok":
            run["baseline_metrics"] = base_v["metrics"]
            pool = [x for x in (base_v, combo_c) if x.get("status") == "ok"]
            for x in pool:
                x.update(_score(run, x["metrics"], base_v["metrics"]))
            win = _best(run, pool) or base_v
            rec = win if win is base_v or win["score"] > base_v["score"] + run["opts"]["min_gain"] else base_v
    run["recommended"] = rec["id"]
    run["combo"] = combo
    _log(run, f"recommended: {rec['label']} (score {rec.get('score')}, baseline {c0.get('score')})")


# ============================================================================
# models mode (LexiPanel Fit, phase D): the same measurements, one model each
# ============================================================================
def _card_memory(devices):
    out = []
    for pci in devices or []:
        base = f"/sys/bus/pci/devices/{pci}"
        try:
            out.append(dict(pci=pci, vram_used_mib=int(open(f"{base}/mem_info_vram_used").read()) // 1048576,
                            vram_total_mib=int(open(f"{base}/mem_info_vram_total").read()) // 1048576,
                            gtt_used_mib=int(open(f"{base}/mem_info_gtt_used").read()) // 1048576))
        except (OSError, ValueError):
            try:
                r = P._nvidia_smi(pci) or {}
                out.append(dict(pci=pci, vram_used_mib=int(float(r["mem_used_mib"])),
                                vram_total_mib=int(float(r["mem_total_mib"])), gtt_used_mib=None))
            except Exception:
                pass
    return out


def _model_extras(run, cand):
    """While the candidate is loaded: what it really occupies, and decode by depth."""
    m = cand.setdefault("metrics", {})
    m["memory"] = _card_memory(run["devices"])
    depths = run["opts"].get("curve_depths") or []
    if not depths:
        return
    try:
        import depthcurve as D
        u = urllib.parse.urlparse(P.api_base())
        ctx = int(dict(run["baseline_params"], **(cand.get("launch") or {})).get("CTX") or 0)
        use = [d for d in depths if d <= ctx - 1024]
        if not use:
            return
        _set_step(run, f"{cand['label']}: tokenizing depth-curve text")
        toks = D._filler_tokens(u.hostname, u.port, max(use) + 16)
        curve = []
        for d in use:
            _gate(run, cand)
            _set_step(run, f"{cand['label']}: depth curve at {d:,} tokens")
            st, resp = _http("POST", "/completion", dict(prompt=toks[:d], n_predict=128,
                                                          ignore_eos=True, cache_prompt=True,
                                                          temperature=0), timeout=3600)
            t = (resp or {}).get("timings") or {}
            if st != 200 or not t:
                curve.append(dict(depth=d, error=f"HTTP {st}"))
                continue
            dn, da = t.get("draft_n") or 0, t.get("draft_n_accepted") or 0
            curve.append(dict(depth=d, decode_tps=round(t.get("predicted_per_second") or 0, 2),
                              prefill_tps=round(t.get("prompt_per_second") or 0, 1),
                              prefill_n=t.get("prompt_n"),
                              draft_acc=round(da / dn, 3) if dn else None))
            _trial(run, dict(curve[-1], cand=cand["id"], kind="curve"))
        m["curve"] = curve
        m["memory_after_curve"] = _card_memory(run["devices"])
    except Stopped:
        raise
    except Exception as e:
        m["curve_error"] = str(e)[:300]
        _log(run, f"{cand['label']}: depth curve failed - {e}")


def _compare_models(run):
    b = run["budget"]
    run["_repo"] = S.build_repo(int(b["depth"] * 3.4))
    base_model = run["baseline_params"].get("MODEL") or ""
    c0 = _new_cand(run, f"today: {os.path.basename(base_model)}", "baseline")
    _evaluate(run, c0, "full")
    if c0["status"] != "ok":
        raise RuntimeError(f"baseline did not complete: {c0.get('detail')}")
    _model_extras(run, c0)
    run["baseline_metrics"] = c0["metrics"]
    c0.update(_score(run, c0["metrics"], c0["metrics"]))
    _save(run)
    for mpath in run["opts"]["models"]:
        c = _new_cand(run, os.path.basename(mpath), "model", launch={"MODEL": mpath})
        fits, why = _fits(run, c["launch"])
        if not fits:
            c.update(status="skipped", detail=why)
            _log(run, f"{c['label']}: skipped - {why}")
            continue
        _evaluate(run, c, "full")
        if c.get("status") == "ok":
            _model_extras(run, c)
        _save(run)
    pool = [c for c in run["candidates"] if c.get("status") == "ok"]
    win = _best(run, pool) or c0
    rec = win if win is c0 or win["score"] > c0["score"] + run["opts"]["min_gain"] else c0
    run["recommended"] = rec["id"]
    _log(run, f"recommended: {rec['label']} (score {rec.get('score')}, today {c0.get('score')})")


# ============================================================================
# public control
# ============================================================================
def start(iid, body):
    global _run, _thread
    body = body or {}
    with _lock:
        if _run is not None:
            raise ValueError(f"an optimizer run is already active on '{_run['instance']}'")
        if getattr(P, "gputune", None) and P.gputune.busy():
            raise ValueError("a GPU Tuning benchmark is running; wait for it or stop it")
        if getattr(getattr(P, "benchlab", None), "active", lambda: False)():
            raise ValueError("a Bench run is active; wait for it or stop it")
        inst = P.get_instance(iid)
        with P.using_instance(inst):
            if not P.server_pid():
                raise ValueError("start the instance first - the optimizer measures a running server")
            if not _healthy():
                raise ValueError("the server is not answering /health yet")
            if _external_busy() and not body.get("force"):
                raise ValueError("the server is busy with other requests right now; wait for it to "
                                 "be idle (or pass force to wait in-run)")
            params = P.load_params()
            try:
                head = P.estimate(params)["vram"]["headroom_mib"]
            except Exception:
                head = None
        goal = str(body.get("goal") or "agentic")
        if goal not in GOALS and goal != "custom":
            raise ValueError(f"unknown goal {goal!r}")
        w = dict(GOALS.get(goal) or {})
        if goal == "custom":
            w = {k: float(body.get("weights", {}).get(k, 0)) for k in ("quality", "speed", "consistency")}
        tot = sum(w[k] for k in ("quality", "speed", "consistency"))
        if tot <= 0:
            raise ValueError("weights must add up to more than zero")
        weights = {k: w[k] / tot for k in ("quality", "speed", "consistency")}
        budget = str(body.get("budget") or "quick")
        if budget not in BUDGETS:
            raise ValueError(f"unknown budget {budget!r}")
        bud = dict(BUDGETS[budget])
        if body.get("depth"):
            bud["depth"] = max(1024, min(int(body["depth"]), int(params.get("CTX") or 8192) - 2048))
        phases = [p for p in (body.get("phases") or ["launch", "kv", "sampling"])
                  if p in ("launch", "kv", "sampling", "models", "candidates")]
        cands = []
        if "candidates" in phases:
            for c in body.get("candidates") or []:
                launch = {str(k): str(v) for k, v in ((c or {}).get("launch") or {}).items()}
                bad = [k for k in launch if k not in P.DEFAULTS or k in ("MODEL", "BACKEND", "PORT", "HOST")]
                if not launch or bad or any(len(v) > 80 or "\n" in v for v in launch.values()):
                    raise ValueError(f"candidate {str((c or {}).get('label'))[:40]!r}: launch settings "
                                     f"must be known settings other than MODEL/BACKEND/PORT/HOST"
                                     + (f" (not: {', '.join(bad)})" if bad else ""))
                label = str(c.get("label") or ", ".join(f"{k}={v}" for k, v in launch.items()))[:120]
                cands.append(dict(label=label, launch=launch))
            if not cands:
                raise ValueError("the candidates phase needs at least one candidate")
            if len(cands) > 6:
                raise ValueError("at most 6 candidates per run")
        workload = None
        if body.get("workload"):
            wl = body["workload"]
            try:
                depths = [max(1024, int(d)) for d in (wl.get("depths") or [])][:4]
                wts = [float(x) for x in (wl.get("weights") or [])][:4]
            except (TypeError, ValueError, AttributeError):
                raise ValueError("workload: {depths: [...], weights: [...]}")
            if not depths or len(depths) != len(wts) or min(wts) < 0 or sum(wts) <= 0:
                raise ValueError("workload: depths and weights of the same length, weights >= 0")
            workload = dict(depths=depths, weights=wts)
            for k in ("prompt_tokens", "output_tokens"):
                if wl.get(k) is not None:
                    try:
                        workload[k] = max(0.0, min(1e6, float(wl[k])))
                    except (TypeError, ValueError):
                        raise ValueError(f"workload.{k}: a number")
        models = []
        if "models" in phases:
            phases = ["models"]
            for m in body.get("models") or []:
                m = str(m)
                if not os.path.isfile(m) or not m.endswith(".gguf"):
                    raise ValueError(f"not a model file: {m}")
                if m not in models and m != params.get("MODEL"):
                    models.append(m)
            if not models:
                raise ValueError("models mode needs at least one model other than the current one")
        th = dict(THERMAL_DEFAULTS)
        th.update({k: float(v) for k, v in (body.get("thermal") or {}).items() if k in th})
        opts = dict(goal=goal, budget=budget, phases=phases, thermal=th,
                    min_ctx=int(body.get("min_ctx") or params.get("CTX") or 0),
                    allow_quality_drop=max(0.0, min(0.5, float(body.get("allow_quality_drop") or 0))),
                    min_gain=max(0.0, min(0.1, float(body.get("min_gain", MIN_GAIN)))),
                    models=models, candidates=cands, workload=workload,
                    curve_depths=sorted({int(d) for d in (body.get("curve_depths")
                                                          or (workload or {}).get("depths") or [])
                                         if 1024 <= int(d) <= 1_000_000})[:6])
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{iid}"
        (RUNS_DIR / run_id).mkdir(parents=True, exist_ok=False)
        _run = dict(id=run_id, instance=iid, instance_name=inst["name"], state="starting",
                    source=str(body.get("source") or "manual")[:20],
                    started=_now(), finished=None, opts=opts, weights=weights, budget=bud,
                    devices=[d for d in (inst.get("devices") or [inst["device"]]) if d != "cpu"],
                    baseline_params=dict(params),
                    baseline_headroom=head, was_running=True, candidates=[], log=[],
                    applied_launch={}, recommended=None, step="starting", model=params.get("MODEL"))
        _log(_run, f"run {run_id} on '{iid}': goal {goal}, budget {budget}, phases {', '.join(phases)}")
        _stop.clear()
        _thread = threading.Thread(target=_worker, args=(_run,), daemon=True, name="optimizer")
        _thread.start()
        return dict(ok=True, id=run_id)


def stop():
    with _lock:
        if _run is None:
            raise ValueError("no optimizer run is active")
        _run["state"] = "stopping"
        _run["step"] = "stopping - finishing the current request, then restoring"
    _stop.set()
    _drop_connection()
    return dict(ok=True)


def rollback(run_id):
    """Put back the settings the last Apply of this run replaced. A setting changed by hand
    since then is left as it is (and named), so a rollback never undoes later work."""
    f = _run_dir(run_id) / "rollback.json"
    try:
        rb = json.loads(f.read_text())
    except (OSError, ValueError):
        raise ValueError("nothing to roll back: no result of this run was applied to the settings")
    with _lock:
        if _run is not None and _run["instance"] == rb["instance"]:
            raise ValueError("an optimizer run is active on this instance; wait for it to finish")
    with P.using_instance(rb["instance"]):
        cur = P.load_params(rb.get("backend"))
        back, kept = {}, []
        for k, before in rb["before"].items():
            if str(cur.get(k)) == str(rb["after"].get(k)):
                back[k] = before
            else:
                kept.append(k)
        if back:
            P.save_params(dict(cur, **back), rb.get("backend") or cur.get("BACKEND"))
    f.rename(f.with_name(f"rollback-done-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    note = (f"restored {len(back)} setting(s): " + ", ".join(f"{k}={v}" for k, v in back.items())
            if back else "nothing restored")
    if kept:
        note += f"; left as they are (changed since the Apply): {', '.join(kept)}"
    return dict(ok=True, restored=back, kept=kept,
                note=note + (" - takes effect at the next restart" if back else ""))


def apply(run_id, cand_id, target="params", name=None):
    r = read_run(run_id)
    with _lock:
        if _run is not None and _run["instance"] == r["instance"]:
            raise ValueError("an optimizer run is active on this instance; wait for it to finish")
    c = next((x for x in r["candidates"] if x["id"] == cand_id), None)
    if c is None:
        raise ValueError(f"no candidate {cand_id!r} in run {run_id}")
    if c.get("status") != "ok":
        raise ValueError(f"candidate {c['label']} did not complete ({c.get('status')})")
    diffs = dict(c.get("launch") or {}, **(c.get("runtime") or {}))
    with P.using_instance(r["instance"]):
        cur = P.load_params()
        if str(cur.get("MODEL")) != str(r.get("model")):
            raise ValueError("the instance's MODEL changed since this run; results do not transfer")
        vals = dict(cur, **diffs)
        changed = {k: dict(before=cur.get(k), after=v) for k, v in diffs.items()
                   if str(cur.get(k)) != str(v)}
        if target == "params":
            P.save_params(vals, vals.get("BACKEND"))
            # what rollback() puts back (the "Roll back the last Apply" button)
            (_run_dir(run_id) / "rollback.json").write_text(json.dumps(dict(
                instance=r["instance"], candidate=cand_id, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                backend=cur.get("BACKEND"), before={k: v["before"] for k, v in changed.items()},
                after={k: v["after"] for k, v in changed.items()}), indent=1, default=str))
            return dict(ok=True, applied=changed,
                        restart_needed=any(k in FLAG_OF for k in changed),
                        note="saved - launch settings take effect at the next restart")
        if target == "profile":
            res = P.save_profile(name or f"optimized-{run_id[:8]}", vals)
            return dict(ok=True, profile=res["name"], applied=changed)
        if target == "tier":
            P.write_tier(str(name or "normal"), vals)
            return dict(ok=True, tier=name or "normal", applied=changed)
    raise ValueError("target must be params, profile or tier")
