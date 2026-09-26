#!/usr/bin/env python3
"""
Decode rate against context depth, measured (added 2026-09-23).

A single tok/s figure is the shallow best case. Agentic sessions drift across
context sizes, so what the operator actually feels is the CURVE: decode at
8k, 32k, 128k and ~240k, and where it bends. /api/speed-curve fits one from
whatever traffic happened to hit the server, per backend and across models;
this module measures it on purpose, for one instance and one model.

Method (same as the 2026-09-23 script run that produced the first curve):
  * One long prompt of Python stdlib source (code-heavy, like an agent's
    context), tokenized by the server itself so depths are exact.
  * Depths are walked upward with cache_prompt, so each step prefills only the
    new tokens: the whole curve costs one cold prefill of the deepest point.
  * At each depth, REPS generations of N_PREDICT tokens with ignore_eos and the
    server's own sampling. Median decode, MTP acceptance, and the cold prefill
    rate of that step are recorded.

Thermals are watched DURING each request, every 2 s, on every card the
instance uses. The first version of this measured only between requests and
reported 91 C for a run whose junction actually sat at 109 C (crit 110) for
most of a 12-minute prefill. Peak junction/memory per step are stored with the
numbers; a sustained reading over the abort limit drops the connection, which
cancels the request server-side.
"""
import glob, http.client, json, os, statistics, subprocess, threading, time
from pathlib import Path

P = None
O = None                                   # optimizer module: thermal defaults + reader
_lock = threading.RLock()
_run = None
_stop = threading.Event()
_conn = None

DEFAULT_DEPTHS = [8192, 32768, 131072, 240000]
DEFAULT_N_PREDICT = 256
DEFAULT_REPS = 3
THRESHOLD_TPS = 40.0                       # the "bend" line the UI reports against
FILLER_GLOB = "/usr/lib/python3*/**/*.py"


def bind(panel_module, optimizer_module):
    global P, O
    P, O = panel_module, optimizer_module


def _dir(iid):
    d = P.PANEL / "curves" / iid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# thermals
# ---------------------------------------------------------------------------
def _nvidia_temp(pci):
    try:
        r = subprocess.run(["nvidia-smi", f"--id={pci}", "--query-gpu=temperature.gpu,"
                            "temperature.memory,power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        v = [x.strip() for x in r.stdout.strip().split(",")]
        num = lambda s: float(s) if s.replace(".", "", 1).isdigit() else None
        return dict(junction=num(v[0]), mem=num(v[1]) if len(v) > 1 else None,
                    power_w=num(v[2]) if len(v) > 2 else None)
    except Exception:
        return {}


def temps(devices):
    """Hottest junction/memory across the instance's cards, summed power."""
    out = dict(O._gpu_temps([d for d in devices if d != "cpu"]))
    for pci in devices:
        if pci == "cpu" or os.path.isdir(f"/sys/bus/pci/devices/{pci}/hwmon"):
            continue
        t = _nvidia_temp(pci)                   # proprietary driver: no hwmon
        for k in ("junction", "mem"):
            if t.get(k) is not None and (out.get(k) is None or t[k] > out[k]):
                out[k] = t[k]
        if t.get("power_w") is not None:
            out["power_w"] = round((out.get("power_w") or 0) + t["power_w"], 1)
    return out


# ---------------------------------------------------------------------------
# server I/O
# ---------------------------------------------------------------------------
def _target(inst):
    """(host, port, devices) of the instance's RUNNING server."""
    with P.using_instance(inst):
        pid = P.server_pid()
        if not pid:
            raise ValueError(f"{inst['id']} is not running")
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    host = P._argv_get(argv, ("--host",)) or "127.0.0.1"
    port = int(P._argv_get(argv, ("--port",)) or 8080)
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return host, port, list(inst.get("devices") or [inst["device"]]), argv


def _json(host, port, method, path, body=None, timeout=60):
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        c.close()


def _filler_tokens(host, port, need):
    toks, chunk, size = [], [], 0
    for fn in sorted(glob.glob(FILLER_GLOB, recursive=True)):
        try:
            txt = f"\n# ===== {fn} =====\n" + open(fn, errors="ignore").read()
        except OSError:
            continue
        chunk.append(txt)
        size += len(txt)
        if size > 400_000:
            toks += _json(host, port, "POST", "/tokenize", {"content": "".join(chunk)},
                          timeout=300)["tokens"]
            chunk, size = [], 0
            if len(toks) >= need:
                return toks[:need]
    raise RuntimeError(f"not enough filler text for {need} tokens (got {len(toks)})")


def _stream_completion(host, port, prompt, n_predict):
    """Streaming /completion; returns the final chunk's timings. Streaming so a
    dropped connection is noticed and the server cancels the task."""
    global _conn
    c = http.client.HTTPConnection(host, port, timeout=3600)
    with _lock:
        _conn = c
    try:
        c.request("POST", "/completion", body=json.dumps(dict(
            prompt=prompt, n_predict=n_predict, ignore_eos=True, cache_prompt=True,
            stream=True)), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"/completion -> HTTP {r.status}: {r.read()[:200]!r}")
        timings = None
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            if ev.get("timings"):
                timings = ev["timings"]
            if ev.get("stop"):
                break
        if timings is None:
            raise RuntimeError("stream ended without timings (request cancelled?)")
        return timings
    finally:
        with _lock:
            _conn = None
        c.close()


def _drop():
    with _lock:
        c = _conn
    if c is not None and c.sock is not None:
        try:
            import socket
            c.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _summarise(rows, depths):
    out = []
    for d in depths:
        rs = [r for r in rows if r["depth"] == d]
        if not rs:
            continue
        dec = [r["decode_tps"] for r in rs]
        acc = [r["draft_acc"] for r in rs if r.get("draft_acc") is not None]
        cold = next((r for r in rs if r["rep"] == 0), rs[0])
        pk = lambda k: max((r[k] for r in rs if r.get(k) is not None), default=None)
        out.append(dict(depth=d, decode_tps=round(statistics.median(dec), 1),
                        decode_min=round(min(dec), 1), decode_max=round(max(dec), 1),
                        draft_acc=round(statistics.median(acc), 3) if acc else None,
                        prefill_tps=cold.get("prompt_tps"), prefill_tokens=cold.get("prompt_n"),
                        prefill_s=round((cold.get("prompt_ms") or 0) / 1000, 1),
                        peak_junction=pk("peak_junction"), peak_mem=pk("peak_mem"),
                        n=len(rs)))
    return out


def crossing(points, threshold=THRESHOLD_TPS):
    """Depth where decode first falls below `threshold`, linearly interpolated."""
    pts = sorted((p["depth"], p["decode_tps"]) for p in points)
    if not pts or pts[0][1] < threshold:
        return 0 if pts else None
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if v1 < threshold <= v0:
            return int(d0 + (v0 - threshold) * (d1 - d0) / (v0 - v1))
    return None                                  # never falls below within the curve


def _save(run):
    rec = {k: v for k, v in run.items() if not k.startswith("_")}
    rec["summary"] = _summarise(run["rows"], run["depths"])
    rec["below_threshold_at"] = crossing(rec["summary"], run["threshold"])
    tmp = run["_file"].with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, run["_file"])
    return rec


def _peak(cur, t):
    with _lock:
        for k, src in (("peak_junction", "junction"), ("peak_mem", "mem")):
            if t.get(src) is not None and (cur.get(k) is None or t[src] > cur[k]):
                cur[k] = t[src]


def _watch(run, th):
    hot = 0
    while not run.get("_done"):
        t = temps(run["devices"])
        with _lock:
            run["thermal"] = dict(t, at=_now_iso())
            cur = run.get("_cur")
            if cur is not None:
                _peak(cur, t)
                if t.get("power_w") is not None:
                    cur.setdefault("_pw", []).append(t["power_w"])
        over = ((t.get("junction") or 0) >= th["abort_c"]
                or (t.get("mem") or 0) >= th["mem_abort_c"])
        hot = hot + 1 if over else 0
        if hot >= 3 and not run.get("thermal_abort"):
            run["thermal_abort"] = (f"junction {t.get('junction')} C / memory {t.get('mem')} C "
                                    f"held at or above {th['abort_c']}/{th['mem_abort_c']} C")
            _drop()
        time.sleep(2)


def _worker(run, inst, th):
    host, port = run["host"], run["port"]
    try:
        run["step"] = "tokenizing filler text"
        toks = _filler_tokens(host, port, max(run["depths"]) + 16)
        for depth in run["depths"]:
            for rep in range(run["reps"]):
                if _stop.is_set():
                    raise InterruptedError("stopped")
                if run.get("thermal_abort"):
                    raise RuntimeError("thermal abort: " + run["thermal_abort"])
                t = temps(run["devices"])
                if (t.get("junction") or 0) >= th["pause_c"]:
                    run["step"] = f"cooling: junction {t['junction']} C, waiting for {th['resume_c']} C"
                    t0 = time.time()
                    while (temps(run["devices"]).get("junction") or 0) > th["resume_c"] \
                            and time.time() - t0 < 900:
                        if _stop.wait(5):
                            raise InterruptedError("stopped")
                run["step"] = (f"depth {depth:,} rep {rep + 1}/{run['reps']}"
                               + (" (cold prefill)" if rep == 0 else ""))
                cur = dict(depth=depth, rep=rep, started=_now_iso())
                with _lock:
                    run["_cur"] = cur
                _peak(cur, temps(run["devices"]))     # a sub-2 s request can fall
                t0 = time.time()                       # between two watcher samples
                tm = _stream_completion(host, port, toks[:depth], run["n_predict"])
                _peak(cur, temps(run["devices"]))
                dn, da = tm.get("draft_n") or 0, tm.get("draft_n_accepted") or 0
                pw = cur.pop("_pw", [])
                cur.update(prompt_n=tm.get("prompt_n"),
                           prompt_tps=round(tm.get("prompt_per_second") or 0, 1),
                           prompt_ms=round(tm.get("prompt_ms") or 0),
                           predicted_n=tm.get("predicted_n"),
                           decode_tps=round(tm.get("predicted_per_second") or 0, 2),
                           draft_acc=round(da / dn, 3) if dn else None,
                           mean_power_w=round(statistics.mean(pw), 1) if pw else None,
                           wall_s=round(time.time() - t0, 1))
                with _lock:
                    run["_cur"] = None
                    run["rows"].append(cur)
                _save(run)
        run["state"] = "done"
    except Exception as e:
        # a Stop drops the socket mid-request, which surfaces as an I/O error
        if isinstance(e, InterruptedError) or (_stop.is_set() and not run.get("thermal_abort")):
            run["state"] = "stopped"
        else:
            run["state"] = "thermal_abort" if run.get("thermal_abort") else "failed"
            run["error"] = run.get("thermal_abort") or str(e)
    finally:
        run["_done"] = True
        run["finished"] = _now_iso()
        run["step"] = None
        _save(run)


def start(iid, body):
    global _run
    with _lock:
        if _run and not _run.get("_done"):
            raise ValueError(f"a curve run is already in progress on {_run['instance']}")
        if getattr(P, "gputune", None) and P.gputune.busy():
            raise ValueError("a GPU Tuning benchmark is running; wait for it or stop it")
    if (O.status().get("active") or {}).get("state") == "running":
        raise ValueError("the optimizer is running; it restarts the server underneath a curve")
    if getattr(getattr(P, "benchlab", None), "active", lambda: False)():
        raise ValueError("a Bench run is active; wait for it or stop it")
    inst = P.get_instance(iid)
    host, port, devices, argv = _target(inst)
    props = _json(host, port, "GET", "/props", timeout=10)
    n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
    slots = _json(host, port, "GET", "/slots", timeout=10)
    if isinstance(slots, list) and any(s.get("is_processing") for s in slots):
        raise ValueError("the server is busy with a request; try again when it is idle")
    n_predict = int(body.get("n_predict") or DEFAULT_N_PREDICT)
    reps = int(body.get("reps") or DEFAULT_REPS)
    if not 16 <= n_predict <= 2048 or not 1 <= reps <= 10:
        raise ValueError("n_predict 16-2048, reps 1-10")
    want = sorted({int(d) for d in (body.get("depths") or DEFAULT_DEPTHS)})
    limit = n_ctx - n_predict - 512
    if limit < 1024:
        raise ValueError(f"context {n_ctx} is too small to measure")
    depths = sorted({min(d, limit) for d in want if d >= 512})
    clipped = [d for d in want if d > limit]
    th = dict(O.THERMAL_DEFAULTS)
    for k in th:
        if body.get(k) is not None:
            th[k] = float(body[k])
    model = P._argv_get(argv, ("-m", "--model")) or ""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = dict(id=ts, instance=iid, model=model, model_name=Path(model).name,
               host=host, port=port, devices=devices, n_ctx=n_ctx,
               kv=P._argv_get(argv, ("--cache-type-k", "-ctk")),
               spec=P._argv_get(argv, ("--spec-type",)),
               binary=os.path.realpath(argv[0]) if argv else None,
               depths=depths, clipped=clipped, n_predict=n_predict, reps=reps,
               threshold=float(body.get("threshold") or THRESHOLD_TPS), thermal_limits=th,
               started=_now_iso(), state="running", step="starting", rows=[],
               source="panel", _file=_dir(iid) / f"curve_{ts}.json")
    _stop.clear()
    with _lock:
        _run = run
    threading.Thread(target=_worker, args=(run, inst, th), daemon=True).start()
    threading.Thread(target=_watch, args=(run, th), daemon=True).start()
    return public(run)


def stop():
    _stop.set()
    _drop()
    return dict(ok=True)


def public(run):
    if not run:
        return None
    r = {k: v for k, v in run.items() if not k.startswith("_")}
    r["summary"] = _summarise(run["rows"], run["depths"])
    r["below_threshold_at"] = crossing(r["summary"], run["threshold"])
    cur = run.get("_cur")
    if cur:
        r["current"] = {k: v for k, v in cur.items() if not k.startswith("_")}
    return r


def history(iid, limit=20):
    out = []
    for f in sorted(_dir(iid).glob("curve_*.json"), reverse=True)[:limit]:
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def status(iid):
    with _lock:
        live = public(_run) if _run else None
    return dict(active=live, history=history(iid), defaults=dict(
        depths=DEFAULT_DEPTHS, n_predict=DEFAULT_N_PREDICT, reps=DEFAULT_REPS,
        threshold=THRESHOLD_TPS, thermal=O.THERMAL_DEFAULTS))
