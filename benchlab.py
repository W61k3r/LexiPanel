#!/usr/bin/env python3
"""
LexiPanel Bench (added 2026-09-25): measurements that say how sure they are.

Every result carries an interval, and every comparison ends in a verdict from benchstats:
better / worse / small / same / undecided. The platform also measures its own error rates
on this machine (calibrate), so "2.1 % faster" comes with how often it would have said so
by chance.

What makes the numbers trustworthy here:
  * Speed is measured at temperature 0 with a fixed seed on fixed code text. The same tokens
    come out every run, so MTP draft acceptance cannot vary between runs. On the reference box acceptance
    explained ~99 % of run-to-run decode noise (7.6 % -> 0.6 %).
  * Where the workload lives: decode at the depths real requests reach (the workload
    profile's depth mix), a "turn" probe (the typical new prompt appended to a cached deep
    context) and the time a typical request of this workload takes.
  * Comparisons alternate A-B-B-A so heat and drift hit both sides, pair the measurements,
    and stop at pre-registered looks as soon as the answer is clear (benchstats.sequential).
  * Real traffic comes first. A request queued behind ours (llama-server's
    requests_deferred) cancels ours at once; the block is measured again once the server
    is quiet. Our own requests are tagged so the workload profile never learns from them.

Kinds
  traffic    no GPU, instant: request-to-request noise, what it takes to see a 1-5 % change,
             and a verdict with an interval for every configuration change in the history.
  profile    the running configuration, as it is: no restart, no setting touched.
  calibrate  A/A: the saved configuration against itself, `blocks` times, optionally
             restarting between blocks (the honest noise figure, but it restarts the server).
             Ends with a self-test: the false-win rate and the smallest reliable difference.
  compare    A/B: the saved settings against the same plus `candidate` overrides. Restarts
             the server for every switch; the saved files are snapshotted and put back.
  goodput    concurrency: 1..N simultaneous turns on warmed contexts. Per-stream decode,
             time to first token, throughput and how many met the latency target.

Nothing here changes GPU, power or firmware settings. Restarting kinds refuse to start
without allow_restart, and nothing starts while another measurement is running.
"""
import hashlib, http.client, json, math, os, re, shutil, statistics, threading, time
from pathlib import Path

import benchstats as S

P = None             # panel
O = None             # optimizer: thermal limits, live-args check, config file list
W = None             # workload: envelope, requests, live fingerprint, busy_now
DC = None            # depthcurve: instance target, filler tokens
GT = None            # gputune: power/temperature sampler

_lock = threading.RLock()
_run = None
_thread = None
_stop = threading.Event()
_conn = None         # a request in flight (workload.own_inflight looks at it)
_conns = []          # every request in flight (goodput runs several)

KINDS = ("profile", "calibrate", "compare", "goodput")
RESTARTING = ("compare",)
PRESETS = {"workload": "the depths real requests reach (workload profile)",
           "quick": "8k and 32k: minutes, not the whole workload"}
DEFAULTS = dict(preset="workload", n_predict=128, reps=3, seed=4242, turn_tokens=None, margin=0.02,
                alpha=S.ALPHA, looks=[3, 5, 8], blocks=6, restart=False, sampled_reps=0,
                quiet_s=60, max_wait_s=1800, concurrency=[1, 2, 4], goodput_depth=8192,
                slo_ttft_s=10.0, slo_tps=15.0)
# never varied by a comparison: identity, networking, and the draft-model trap
FORBIDDEN = {"PORT", "HOST", "BACKEND", "API_KEY", "ALIAS", "SPEC_DRAFT_MODEL", "MMPROJ",
             "MMPROJ_DEVICE", "SPEC_DRAFT_DEVICE", "LOG_FILE"}
MAX_BLOCK_TRIES = 3
RESTORE_WAIT_S = 600    # at the end, how long a real request may keep the candidate running


class Stopped(Exception):
    pass


class Disturbed(Exception):
    """A real request arrived while we measured: redo the block."""


class Failed(Exception):
    pass


def bind(panel, optimizer, workload, depthcurve, gputune):
    global P, O, W, DC, GT
    P, O, W, DC, GT = panel, optimizer, workload, depthcurve, gputune


def _dir(iid):
    d = P.PANEL / "bench" / iid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def active():
    """A bench run that has not finished (other measuring modules check this)."""
    r = _run
    return bool(r and not r.get("_done"))


def inflight():
    return len(_conns)


def _log(run, msg):
    with _lock:
        run.setdefault("log", []).append(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}")
        del run["log"][:-400]


def _step(run, text):
    with _lock:
        run["step"] = text


def public(run):
    if not run:
        return None
    return {k: v for k, v in run.items() if not k.startswith("_")}


def _save(run):
    _atomic(_dir(run["instance"]) / f"{run['id']}.json", public(run))


# ============================================================================
# target and transport
# ============================================================================
def _target(inst):
    eng = inst.get("engine") or "llama.cpp"
    if eng == "vllm":
        # vLLM retitles its process, so address, name and key come from the instance's settings
        VL = P.vllm_engine
        v = VL.load_params(inst)
        with P.using_instance(inst):
            if not P.server_pid():
                raise ValueError(f"{inst['id']} is not running")
        host = str(v.get("HOST") or "127.0.0.1")
        return dict(engine="vllm", host="127.0.0.1" if host in ("0.0.0.0", "") else host, port=int(v["PORT"]),
                    devices=[d for d in (inst.get("devices") or [inst.get("device")]) if d and d != "cpu"],
                    argv=[], key=VL.api_key(inst),
                    name=str(v.get("VL_SERVED_NAME") or "") or Path(str(v.get("VL_MODEL") or "")).name
                    or str(v.get("VL_MODEL")))
    if eng != "llama.cpp":
        raise ValueError(f"{inst['id']} runs {eng}; Bench measures llama.cpp and vLLM instances")
    host, port, devices, argv = DC._target(inst)
    return dict(engine="llama.cpp", host=host, port=port, devices=[d for d in devices if d and d != "cpu"],
                argv=argv)


def _http(run, method, path, body=None, timeout=60):
    t = run["_t"]
    c = http.client.HTTPConnection(t["host"], t["port"], timeout=timeout)
    hdr = {"Content-Type": "application/json"}
    if t.get("key"):
        hdr["Authorization"] = f"Bearer {t['key']}"
    try:
        c.request(method, path, body=json.dumps(body) if body is not None else None, headers=hdr)
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise Failed(f"{method} {path} -> HTTP {r.status}: {data[:200]!r}")
        return data
    finally:
        c.close()


def _deferred(run):
    """Requests waiting behind ours (llama-server --metrics), or None if unavailable."""
    try:
        txt = _http(run, "GET", "/metrics", timeout=5).decode(errors="ignore")
    except Exception:
        return None
    name = "vllm:num_requests_waiting" if run["_t"].get("engine") == "vllm" else "llamacpp:requests_deferred"
    m = re.search(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.]+)", txt, re.M)
    return int(float(m.group(1))) if m else None


def _foreign(run):
    """Someone else's request is running or waiting."""
    if (W.busy_now(run["instance"]) or 0) > 0:
        return "a real request is running"
    if (_deferred(run) or 0) > 0:
        return "a real request is waiting"
    return None


def _drop_all():
    with _lock:
        cs = list(_conns)
    for c in cs:
        try:
            if c.sock is not None:
                import socket
                c.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _stream(run, prompt, n_predict, temperature=0.0, seed=None, cache=True):
    if run["_t"].get("engine") == "vllm":
        return _stream_openai(run, prompt, n_predict, temperature, seed)
    return _stream_llama(run, prompt, n_predict, temperature, seed, cache)


def _stream_openai(run, prompt, n_predict, temperature=0.0, seed=None):
    """One streaming /v1/completions (vLLM). The server reports no timings, so they are taken
    here: time to first token, and decode as (tokens - 1) over first-to-last token time."""
    global _conn
    t = run["_t"]
    body = dict(model=t["name"], prompt=prompt, max_tokens=int(n_predict), stream=True, ignore_eos=True,
                stream_options=dict(include_usage=True))
    if temperature is not None:
        body["temperature"] = float(temperature)
    if seed is not None:
        body["seed"] = int(seed)
    hdr = {"Content-Type": "application/json"}
    if t.get("key"):
        hdr["Authorization"] = f"Bearer {t['key']}"
    c = http.client.HTTPConnection(t["host"], t["port"], timeout=3600)
    with _lock:
        _conns.append(c)
        _conn = c
    t0 = time.time()
    first = last = None
    h = hashlib.sha1()
    usage = None
    try:
        c.request("POST", "/v1/completions", body=json.dumps(body), headers=hdr)
        r = c.getresponse()
        if r.status != 200:
            raise Failed(f"/v1/completions -> HTTP {r.status}: {r.read()[:200]!r}")
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            for ch in ev.get("choices") or []:
                piece = ch.get("text")
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    h.update(piece.encode())
            if ev.get("usage"):
                usage = ev["usage"]
    except (OSError, http.client.HTTPException) as e:
        if run.get("_disturbed"):
            raise Disturbed(run["_disturbed"])
        if _stop.is_set():
            raise Stopped()
        if run.get("_thermal_abort"):
            raise Failed(run["_thermal_abort"])
        raise Failed(f"request failed: {e}")
    finally:
        with _lock:
            if c in _conns:
                _conns.remove(c)
            _conn = _conns[-1] if _conns else None
        c.close()
    if run.get("_disturbed"):
        raise Disturbed(run["_disturbed"])
    end = time.time()
    gen = int((usage or {}).get("completion_tokens") or 0)
    pn = int((usage or {}).get("prompt_tokens") or 0)
    ttft = (first or end) - t0
    dec = (gen - 1) / (last - first) if gen > 1 and first and last and last > first else None
    return dict(t0=t0, t1=end, wall_s=end - t0, ttft_ms=ttft * 1000, prompt_n=pn, prompt_ms=ttft * 1000,
                prompt_tps=pn / ttft if pn and ttft > 0 else None, gen_n=gen, decode_tps=dec, accept=None,
                hash=h.hexdigest()[:16])


def _stream_llama(run, prompt, n_predict, temperature=0.0, seed=None, cache=True):
    """One streaming /completion. Returns timings plus client-side time to first token, wall
    time and a hash of the generated text (identical text = identical tokens at temp 0)."""
    global _conn
    t = run["_t"]
    body = dict(prompt=prompt, n_predict=int(n_predict), ignore_eos=True, cache_prompt=bool(cache),
                stream=True)
    if temperature is not None:
        body["temperature"] = float(temperature)
    if seed is not None:
        body["seed"] = int(seed)
    c = http.client.HTTPConnection(t["host"], t["port"], timeout=3600)
    with _lock:
        _conns.append(c)
        _conn = c
    t0 = time.time()
    first = None
    h = hashlib.sha1()
    timings = None
    try:
        c.request("POST", "/completion", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise Failed(f"/completion -> HTTP {r.status}: {r.read()[:200]!r}")
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            piece = ev.get("content")
            if piece:
                if first is None:
                    first = time.time()
                h.update(piece.encode())
            if ev.get("timings"):
                timings = ev["timings"]
            if ev.get("stop"):
                break
    except (OSError, http.client.HTTPException) as e:
        if run.get("_disturbed"):
            raise Disturbed(run["_disturbed"])
        if _stop.is_set():
            raise Stopped()
        if run.get("_thermal_abort"):
            raise Failed(run["_thermal_abort"])
        raise Failed(f"request failed: {e}")
    finally:
        with _lock:
            if c in _conns:
                _conns.remove(c)
            _conn = _conns[-1] if _conns else None
        c.close()
    if run.get("_disturbed"):
        raise Disturbed(run["_disturbed"])
    if timings is None:
        if _stop.is_set():
            raise Stopped()
        raise Failed("the stream ended without timings (cancelled or the server died)")
    end = time.time()
    dn, da = timings.get("draft_n") or 0, timings.get("draft_n_accepted") or 0
    return dict(t0=t0, t1=end, wall_s=end - t0, ttft_ms=((first or end) - t0) * 1000,
                prompt_n=timings.get("prompt_n"), prompt_ms=timings.get("prompt_ms"),
                prompt_tps=timings.get("prompt_per_second"), gen_n=timings.get("predicted_n"),
                decode_tps=timings.get("predicted_per_second"),
                accept=(da / dn) if dn else None, hash=h.hexdigest()[:16])


# ============================================================================
# the watcher: heat, power, and real traffic, every second
# ============================================================================
def _watch(run):
    th = run["opts"]["thermal"]
    hot = 0
    while not run.get("_done"):
        try:
            s = GT.sample(run["_t"]["devices"]) if GT and run["_t"]["devices"] else {}
        except Exception:
            s = {}
        s = dict(s or {}, t=time.time())
        with _lock:
            run["_samples"].append(s)
            del run["_samples"][:-7200]
            run["live"] = {k: s.get(k) for k in ("power_w", "junction", "mem", "busy")}
        over = (s.get("junction") or 0) >= th["abort_c"] or (s.get("mem") or 0) >= th["mem_abort_c"]
        hot = hot + 1 if over else 0
        if hot >= 3 and not run.get("_thermal_abort"):
            run["_thermal_abort"] = f"junction {s.get('junction')} C / memory {s.get('mem')} C at the abort limit"
            _drop_all()
        if _conns and not run.get("_disturbed") and not run.get("_goodput"):
            q = _deferred(run)
            if q:
                run["_disturbed"] = "a real request arrived"
                _drop_all()
        time.sleep(1)


def _window(run, t0, t1):
    with _lock:
        return [s for s in run["_samples"] if t0 <= s["t"] <= t1]


def _energy(run, m):
    """Mean power over a request and joules per generated token."""
    xs = [s["power_w"] for s in _window(run, m["t0"], m["t1"]) if s.get("power_w") is not None]
    if not xs or not m.get("gen_n"):
        return None, None
    w = statistics.fmean(xs)
    return round(w, 1), round(w * m["wall_s"] / m["gen_n"], 3)


def _gate(run):
    if _stop.is_set():
        raise Stopped()
    if run.get("_thermal_abort"):
        raise Failed(run["_thermal_abort"])
    th = run["opts"]["thermal"]
    t = O._gpu_temps(run["_t"]["devices"])
    if (t.get("junction") or 0) >= th["pause_c"]:
        _log(run, f"thermal pause at junction {t['junction']} C, waiting for {th['resume_c']} C")
        t0 = time.time()
        while (t.get("junction") or 0) > th["resume_c"] and time.time() - t0 < 900:
            _sleep(5)
            t = O._gpu_temps(run["_t"]["devices"])
        run["_block_notes"].append(f"thermal pause {int(time.time() - t0)} s")
    waited = 0
    while True:
        why = _foreign(run)
        if not why:
            break
        if waited == 0:
            _step(run, f"waiting: {why}")
        if waited >= run["opts"]["max_wait_s"]:
            raise Failed(f"the server stayed busy for {waited} s")
        _sleep(5)
        waited += 5
    if waited:
        run["external_wait_s"] = run.get("external_wait_s", 0) + waited


def _sleep(sec):
    if _stop.wait(sec):
        raise Stopped()


# ============================================================================
# workload shape: where to measure and what a typical request is
# ============================================================================
def shape(inst, opts, n_ctx=None):
    env = {}
    try:
        env = W.envelope(inst) or {}
    except Exception:
        pass
    mix = env.get("mix") or {}
    ctx = int(n_ctx or env.get("slot_ctx") or 0) or 131072
    turn = opts.get("turn_tokens") or int(min(4096, max(16, round(mix.get("prompt_tokens") or 512))))
    out_tok = int(round(mix.get("output_tokens") or 256))
    limit = ctx - int(opts["n_predict"]) - turn - 1024
    if opts.get("depths"):
        depths = [int(d) for d in opts["depths"]]
        weights = [1.0] * len(depths)
        source = "chosen"
    elif opts["preset"] == "workload" and mix.get("depths"):
        depths, weights, source = list(mix["depths"]), list(mix["weights"]), "workload"
    else:
        depths, weights, source = [8192, 32768], [1.0, 1.0], "quick"
    keep = {}
    for d, w in zip(depths, weights):
        d = max(512, min(int(d), limit))
        keep[d] = keep.get(d, 0.0) + float(w)
    ds = sorted(keep)
    tot = sum(keep.values()) or 1.0
    return dict(depths=ds, weights={d: round(keep[d] / tot, 4) for d in ds}, turn_tokens=turn,
                output_tokens=out_tok, source=source, n_ctx=ctx,
                note=None if source != "workload" else f"{env.get('depth', {}).get('n', 0)} real requests")


def _tokens(run, need):
    """Filler tokens (stdlib Python source, tokenized by the server itself), per model."""
    model = next((run["_t"]["argv"][i + 1] for i, a in enumerate(run["_t"]["argv"])
                  if a in ("-m", "--model") and i + 1 < len(run["_t"]["argv"])), "")
    cache = run.setdefault("_toks", {})
    if run["_t"].get("engine") == "vllm":
        model = "vllm:" + run["_t"]["name"]
    if len(cache.get(model) or []) < need:
        _step(run, f"tokenizing {need:,} tokens of filler text")
        cache[model] = (_filler_tokens_vllm(run, need) if run["_t"].get("engine") == "vllm"
                        else DC._filler_tokens(run["_t"]["host"], run["_t"]["port"], need))
    return cache[model]


def _filler_tokens_vllm(run, need):
    """The depth curve's filler text (stdlib Python source), tokenized by vLLM's /tokenize."""
    import glob
    toks, chunk, size = [], [], 0
    for fn in sorted(glob.glob(DC.FILLER_GLOB, recursive=True)):
        try:
            with open(fn, errors="ignore") as fh:
                txt = f"\n# ===== {fn} =====\n" + fh.read()
        except OSError:
            continue
        chunk.append(txt)
        size += len(txt)
        if size > 200_000:
            r = json.loads(_http(run, "POST", "/tokenize", dict(model=run["_t"]["name"], prompt="".join(chunk),
                                                                 add_special_tokens=False), timeout=300))
            toks += r.get("tokens") or []
            chunk, size = [], 0
            if len(toks) >= need:
                return toks[:need]
    raise Failed(f"not enough filler text for {need} tokens (got {len(toks)})")


# ============================================================================
# one measurement block on whatever configuration is running
# ============================================================================
def _block(run, label):
    for attempt in range(1, MAX_BLOCK_TRIES + 1):
        run["_disturbed"] = None
        run["_block_notes"] = []
        try:
            b = _measure(run, label)
            b["attempt"] = attempt
            b["notes"] = run["_block_notes"]
            return b
        except Disturbed as e:
            _log(run, f"{label}: {e} - letting it through, then measuring again")
            _step(run, "a real request arrived: waiting for the server to be quiet")
            _quiet(run)
    raise Failed(f"{label}: disturbed by real traffic {MAX_BLOCK_TRIES} times")


def _quiet(run):
    need = run["opts"]["quiet_s"]
    t0, calm = time.time(), 0
    while calm < need:
        if time.time() - t0 > run["opts"]["max_wait_s"]:
            raise Failed("the server never went quiet")
        _sleep(5)
        calm = calm + 5 if not _foreign(run) else 0


def _measure(run, label):
    o, sh = run["opts"], run["shape"]
    depths, turn = sh["depths"], sh["turn_tokens"]
    toks = _tokens(run, max(depths) + turn + 16)
    per = {}
    for d in depths:
        rows = []
        for i in range(o["reps"]):
            _gate(run)
            _step(run, f"{label}: decode at {d:,} tokens, {i + 1}/{o['reps']}")
            m = _stream(run, toks[:d], o["n_predict"], temperature=0.0, seed=o["seed"])
            m["power_w"], m["j_per_tok"] = _energy(run, m)
            rows.append(m)
        cold = rows[0]
        _gate(run)
        _step(run, f"{label}: turn of {turn:,} new tokens at {d:,}")
        tm = _stream(run, toks[:d + turn], 16, temperature=0.0, seed=o["seed"])
        dec = [r["decode_tps"] for r in rows if r.get("decode_tps")]
        per[str(d)] = dict(
            decode=[round(x, 3) for x in dec], decode_med=round(statistics.median(dec), 3) if dec else None,
            accept=[round(r["accept"], 4) for r in rows if r.get("accept") is not None],
            hashes=[r["hash"] for r in rows], deterministic=len({r["hash"] for r in rows}) == 1,
            prefill_new=cold.get("prompt_n"), prefill_tps=cold.get("prompt_tps"),
            turn_ms=tm.get("prompt_ms"), turn_ttft_ms=round(tm["ttft_ms"], 1),
            power_w=_med([r["power_w"] for r in rows]), j_per_tok=_med([r["j_per_tok"] for r in rows]))
    sampled = None
    if o.get("sampled_reps"):
        d = max(sh["weights"], key=sh["weights"].get)
        acc, dec = [], []
        for i in range(int(o["sampled_reps"])):
            _gate(run)
            _step(run, f"{label}: sampled acceptance at {d:,}, {i + 1}/{o['sampled_reps']}")
            m = _stream(run, toks[:d], o["n_predict"], temperature=None, seed=o["seed"] + 1 + i)
            if m.get("accept") is not None:
                acc.append(m["accept"])
            dec.append(m["decode_tps"])
        sampled = dict(depth=d, accept=S.describe(acc), decode=S.describe(dec))
    return dict(label=label, at=_now(), per_depth=per, sampled=sampled, **_summary(sh, per))


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 3) if xs else None


def _summary(sh, per):
    dec = {int(d): v["decode_med"] for d, v in per.items() if v.get("decode_med")}
    w = {int(d): x for d, x in sh["weights"].items()}
    gen = S.request_seconds(dec, w, sh["output_tokens"])
    turns = [(w[int(d)], v["turn_ms"]) for d, v in per.items() if v.get("turn_ms") is not None]
    turn_s = sum(a * b for a, b in turns) / sum(a for a, _ in turns) / 1000 if turns else 0.0
    eff = None
    if dec and all(dec.get(d) for d in w):
        eff = sum(w.values()) / sum(x / dec[d] for d, x in w.items())
    return dict(request_s=round(gen + turn_s, 4) if gen is not None else None,
                decode_eff=round(eff, 3) if eff else None, turn_s=round(turn_s, 4))


# ============================================================================
# switching configurations (compare): the panel's own save + restart path
# ============================================================================
def check_candidate(cand, base):
    if not isinstance(cand, dict) or not cand:
        raise ValueError("candidate: {\"KEY\": \"value\", ...} with at least one setting")
    out = {}
    for k, v in cand.items():
        k, v = str(k).strip().upper(), str(v).strip()
        if k not in P.DEFAULTS:
            raise ValueError(f"{k}: not a setting of this panel")
        if k in FORBIDDEN:
            raise ValueError(f"{k}: not varied by a comparison")
        if len(v) > 300 or "\n" in v:
            raise ValueError(f"{k}: bad value")
        if k == "MODEL" and not (v.endswith(".gguf") and os.path.isfile(v)):
            raise ValueError("MODEL: an existing .gguf file")
        if str(base.get(k, "")) != v:
            out[k] = v
    if not out:
        raise ValueError("the candidate equals the saved settings")
    return out


def _fits(base, launch):
    vals = dict(base, **launch)
    try:
        est = P.estimate(vals)
        head = est["vram"]["headroom_mib"]
    except Exception as e:
        return False, f"estimator failed: {e}"
    if head < 400:
        return False, f"estimated VRAM headroom {head} MiB < 400 MiB"
    try:
        rb = P.ram_budget(vals, include_others=True)
        if rb.get("verdict") == "impossible":
            return False, "host RAM budget: " + rb.get("detail", "impossible")
    except Exception:
        pass
    return True, f"fits, est. headroom {head} MiB"


def _snapshot(run):
    snap = _dir(run["instance"]) / f"{run['id']}.snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, f in enumerate(O._config_files()):
        if os.path.exists(f):
            shutil.copy2(f, snap / str(i))
            manifest[f] = str(i)
        else:
            manifest[f] = None
    _atomic(snap / "manifest.json", manifest)
    run["snapshot"] = str(snap)


def _restore(snap):
    snap = Path(snap)
    manifest = json.loads((snap / "manifest.json").read_text())
    for f, key in manifest.items():
        if key is None:
            if os.path.exists(f):
                os.unlink(f)
        else:
            shutil.copy2(snap / key, f)


def _wait_idle(run, why, limit):
    """Never restart under someone else's request: wait until no real request is running or
    queued (two checks in a row). Returns the seconds waited, or None if `limit` ran out."""
    t0, calm = time.time(), 0
    while calm < 2:
        if time.time() - t0 > limit:
            return None
        f = _foreign(run)
        if f:
            calm = 0
            _step(run, f"waiting before the restart ({why}): {f}")
        else:
            calm += 1
        if calm < 2:
            _sleep(3)
    return int(time.time() - t0)


def _restart_and_wait(run, what, timeout=900):
    pid0 = P.server_pid()
    inst = P.INST()
    if inst.get("legacy") and P.unit_installed():
        ok, msg = P._systemctl("restart")
    else:
        if pid0:
            P.stop_server()
            _sleep(2)
        ok, msg = P.start_server()
    if not ok:
        raise Failed(f"restart refused: {msg}")
    t0 = time.time()
    _step(run, f"starting {what}")
    while time.time() - t0 < timeout:
        _sleep(3)
        pid = P.server_pid()
        if pid and pid != pid0:
            try:
                if json.loads(_http(run, "GET", "/health", timeout=5)).get("status") == "ok":
                    run["_t"] = _target(run["_inst"])
                    return pid
            except Exception:
                pass
        if (P.unit_state() or {}).get("active") == "failed":
            break
    raise Failed(f"{what} was not healthy after {int(time.time() - t0)} s")


def _apply(run, launch, label):
    """Make the server run the saved settings + `launch`. No-op when already there."""
    if run.get("_applied") == launch:
        return
    vals = dict(run["_base"], **launch)
    w = _wait_idle(run, f"switch to {label}", run["opts"]["max_wait_s"])
    if w is None:
        raise Failed(f"a real request kept the server busy for {run['opts']['max_wait_s']} s; "
                     f"not restarting under it")
    if w > 3:
        _log(run, f"waited {w} s for real traffic to finish before switching to {label}")
    P.save_params(vals, vals.get("BACKEND"))
    run["_applied"] = dict(launch)
    run["applied"] = dict(launch)
    _save(run)
    _restart_and_wait(run, label)
    bad = O._live_matches(vals)
    try:
        P.FAIL_FILE.write_text("0\n")
    except OSError:
        pass
    if bad:
        raise Failed("the server came up with other settings (a fallback tier?): " + "; ".join(bad))


# ============================================================================
# protocols
# ============================================================================
def _profile(run):
    b = _block(run, "profile")
    run["blocks"].append(b)
    run["result"] = dict(kind="profile", request_s=b["request_s"], decode_eff=b["decode_eff"],
                         per_depth={d: dict(decode=S.describe(v["decode"]), deterministic=v["deterministic"],
                                            turn_ms=v["turn_ms"], j_per_tok=v["j_per_tok"],
                                            accept=_med(v["accept"]))
                                    for d, v in b["per_depth"].items()})


def _chi2_lower(p, k):
    """Wilson-Hilferty approximation to the chi-square quantile."""
    z = S.norm_ppf(p)
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def _calibrate(run):
    o = run["opts"]
    for i in range(int(o["blocks"])):
        if o["restart"] and i > 0:
            _restart_and_wait(run, f"the saved settings (A/A block {i + 1})")
        run["blocks"].append(_block(run, f"A/A {i + 1}/{o['blocks']}"))
        _save(run)
    bl = run["blocks"]
    idx = range(0, len(bl) - 1, 2)                  # disjoint pairs (1,2) (3,4) ... stay independent
    diffs = [math.log(bl[i + 1]["request_s"] / bl[i]["request_s"]) for i in idx
             if bl[i].get("request_s") and bl[i + 1].get("request_s")]
    per_depth = {}
    for d in bl[0]["per_depth"]:
        dd = [math.log(bl[i + 1]["per_depth"][d]["decode_med"] / bl[i]["per_depth"][d]["decode_med"])
              for i in idx if bl[i]["per_depth"][d].get("decode_med") and bl[i + 1]["per_depth"][d].get("decode_med")]
        per_depth[d] = dict(between_sd_log=statistics.stdev(dd) if len(dd) > 1 else None, n=len(dd),
                            deterministic=len({h for b in bl for h in b["per_depth"][d]["hashes"]}) == 1)
    within = [v["decode"] for b in bl for v in b["per_depth"].values() if len(v["decode"]) > 1]
    within_sd = statistics.fmean([statistics.stdev([math.log(x) for x in xs]) for xs in within]) if within else None
    res = dict(kind="calibrate", restart=o["restart"], blocks=len(bl), metric="request_s",
               diffs=[round(x, 6) for x in diffs], per_depth=per_depth, within_sd_log=within_sd,
               deterministic_across_blocks=all(v["deterministic"] for v in per_depth.values()),
               scope="between restarts" if o["restart"] else "within one server run (no restart)")
    if len(diffs) >= 2:
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        k = len(diffs) - 1
        sd_up = sd * math.sqrt(k / max(_chi2_lower(0.2, k), 1e-9))     # 80 % upper bound
        res.update(between_sd_log=sd, between_sd_upper=sd_up)
        if len(diffs) >= 12:
            sim, basis = diffs, "the measured differences, resampled"
        else:
            import random
            rng = random.Random(o["seed"])
            sim = [rng.gauss(0, max(sd_up, 1e-4)) for _ in range(400)]
            basis = (f"normal noise at the 80 % upper bound of the spread of {len(diffs)} measured "
                     "differences (conservative until there are 12)")
        st = S.selftest(sim, looks=tuple(o["looks"]), alpha=o["alpha"], margin=o["margin"], trials=600,
                        seed=o["seed"])
        res.update(selftest=st, selftest_basis=basis,
                   pairs_for={f"{p}%": S.pairs_needed(max(sd_up, 1e-4), p / 100) for p in (1, 2, 3, 5)})
    run["result"] = res
    cal = dict(res, id=run["id"], at=_now(), fp=run.get("fp"), config=run.get("config"),
               shape=run["shape"], looks=o["looks"], margin=o["margin"])
    f = _dir(run["instance"]) / "calibration.json"
    hist = _read(f) or {}
    hist.setdefault("history", []).append(cal)
    hist["history"] = hist["history"][-20:]
    hist["latest"] = cal
    _atomic(f, hist)


def _compare(run):
    o = run["opts"]
    cand = run["candidate"]
    looks = tuple(o["looks"])
    order = ["A", "B", "B", "A"]
    a, b = [], []
    pairs = []
    k = 0
    bad = O._live_matches(run["_base"])
    if bad:                                          # the server does not run what is saved
        _log(run, "the running server differs from the saved settings (" + "; ".join(bad) + "): restarting first")
        run["_applied"] = None
    while True:
        side = order[k % 4]
        k += 1
        launch = {} if side == "A" else cand
        _apply(run, launch, "the saved settings (A)" if side == "A" else "the candidate (B)")
        blk = _block(run, f"{side}{(len(a) if side == 'A' else len(b)) + 1}")
        blk["side"] = side
        run["blocks"].append(blk)
        (a if side == "A" else b).append(blk)
        _save(run)
        n = min(len(a), len(b))
        if n and len(a) == len(b):
            pairs.append((a[-1], b[-1]))
            seq = S.sequential([p[0]["request_s"] for p in pairs], [p[1]["request_s"] for p in pairs],
                               looks=looks, alpha=o["alpha"], margin=o["margin"], higher_better=False)
            run["sequential"] = {kk: v for kk, v in seq.items() if kk not in ("looks",)}
            _log(run, f"pair {n}: B request time {seq.get('pct', 0):+.2f} %"
                      + (f" -> {seq['verdict']}" if seq["state"] == "stop" else ""))
            if seq["state"] == "stop":
                break
    per = {}
    for d in pairs[0][0]["per_depth"]:
        pr = S.paired([p[0]["per_depth"][d]["decode_med"] for p in pairs],
                      [p[1]["per_depth"][d]["decode_med"] for p in pairs], alpha=o["alpha"])
        per[d] = dict(pr, verdict=S.verdict(pr.get("mean_log"), pr.get("lo_log"), pr.get("hi_log"), o["margin"]),
                      same_output=all(p[0]["per_depth"][d]["hashes"][0] == p[1]["per_depth"][d]["hashes"][0]
                                      for p in pairs),
                      accept_a=_med([x for p in pairs for x in p[0]["per_depth"][d]["accept"]]),
                      accept_b=_med([x for p in pairs for x in p[1]["per_depth"][d]["accept"]]))
    seq = run["sequential"]
    run["result"] = dict(kind="compare", candidate=cand, pairs=len(pairs), verdict=seq["verdict"],
                         request_time_pct=seq.get("pct"), lo_pct=seq.get("pct_lo"), hi_pct=seq.get("pct_hi"),
                         precision_pct=seq.get("precision_pct"), per_depth=per,
                         note="request time: lower is better; 'better' means B finishes a typical request faster")


def _goodput(run):
    o, sh = run["opts"], run["shape"]
    if run["_t"].get("engine") == "vllm":            # each request may use the whole context; KV is paged
        d = max(256, min(int(o["goodput_depth"]), sh["n_ctx"] - sh["turn_tokens"] - int(o["n_predict"]) - 64))
    else:                                              # llama.cpp slots share one context
        # the same floor as vLLM's, so the two engines are measured at the same depth
        d = max(256, min(int(o["goodput_depth"]), sh["n_ctx"] // max(o["concurrency"]) - sh["turn_tokens"] - 512))
    turn = sh["turn_tokens"]
    levels = sorted({max(1, min(int(x), 32)) for x in o["concurrency"]})
    toks = _tokens(run, max(levels) * (d + turn) + 16)
    run["_goodput"] = True                      # our own parallel streams must not look foreign
    out = []
    for n in levels:
        ctxs = [toks[i * (d + turn):(i + 1) * (d + turn)] for i in range(n)]
        for i, c in enumerate(ctxs):             # warm each context, one at a time
            _gate(run)
            _step(run, f"{n} streams: warming context {i + 1}/{n} ({d:,} tokens)")
            _stream(run, c[:d], 1, temperature=0.0, seed=o["seed"])
        _gate(run)
        _step(run, f"{n} simultaneous turns")
        res, errs = [None] * n, []

        def go(i):
            try:
                res[i] = _stream(run, ctxs[i], o["n_predict"], temperature=0.0, seed=o["seed"])
            except Exception as e:
                errs.append(str(e))
        ths = [threading.Thread(target=go, args=(i,), daemon=True) for i in range(n)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        if errs:
            raise Failed(f"{n} streams: {errs[0]}")
        ok = [r for r in res if r["ttft_ms"] / 1000 <= o["slo_ttft_s"] and (r["decode_tps"] or 0) >= o["slo_tps"]]
        out.append(dict(streams=n, wall_s=round(wall, 2), tokens=sum(r["gen_n"] or 0 for r in res),
                        throughput_tps=round(sum(r["gen_n"] or 0 for r in res) / wall, 2),
                        decode_per_stream=S.describe([r["decode_tps"] for r in res]),
                        ttft_s=S.describe([r["ttft_ms"] / 1000 for r in res]),
                        met_slo=len(ok), goodput=round(len(ok) / n, 3)))
        _save(run)
    run["_goodput"] = False
    run["result"] = dict(kind="goodput", depth=d, turn_tokens=turn, levels=out,
                         slo=dict(ttft_s=o["slo_ttft_s"], decode_tps=o["slo_tps"]),
                         slots=_slots(run))


def _slots(run):
    if run["_t"].get("engine") == "vllm":
        try:
            return int(P.vllm_engine.load_params(run["_inst"]).get("VL_MAX_NUM_SEQS"))
        except Exception:
            return None
    try:
        return len(json.loads(_http(run, "GET", "/slots", timeout=5)))
    except Exception:
        return None


# ============================================================================
# traffic: no GPU, instant
# ============================================================================
def traffic(iid, days=14):
    inst = P.get_instance(iid)
    now = time.time()
    rows = [r for r in W.requests(iid, now - days * 86400) if (r.get("eval_tokens") or 64) >= 64]
    edges = list(P.CURVE_BUCKETS)
    noise = S.traffic_noise(rows, edges)
    out = dict(instance=iid, days=days, requests=len(rows), noise=noise)
    sd = noise.get("adjusted_sd") or noise.get("raw_sd")
    if sd:
        out["requests_per_side"] = {f"{p}%": dict(raw=S.requests_needed(noise.get("raw_sd"), p / 100),
                                                  adjusted=S.requests_needed(sd, p / 100))
                                    for p in (1, 2, 3, 5)}
    cfg = _read(W._dir(iid) / "configs.json") or {}
    seq, changes = [], []
    for r in sorted((r for r in rows if r.get("fp")), key=lambda r: r["t"]):
        if not seq or seq[-1][0] != r["fp"]:
            seq.append((r["fp"], []))
        seq[-1][1].append(r)
    for (fa, ra), (fb, rb) in zip(seq, seq[1:]):
        adj = S.traffic_effect(ra, rb, edges, adjust=True)
        raw = S.traffic_effect(ra, rb, edges, adjust=False)
        v = S.verdict(adj.get("mean_log"), adj.get("lo_log"), adj.get("hi_log"), 0.02) \
            if adj.get("mean_log") is not None else "not enough data"
        changes.append(dict(before=fa, after=fb, at=rb[0]["t"], before_cfg=(cfg.get(fa) or {}).get("summary"),
                            after_cfg=(cfg.get(fb) or {}).get("summary"), adjusted=adj, raw=raw, verdict=v))
    out["changes"] = changes
    out["configs"] = [dict(fp=f, requests=len(rs), summary=(cfg.get(f) or {}).get("summary")) for f, rs in seq]
    return out


# ============================================================================
# control
# ============================================================================
def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _others(iid):
    if getattr(O, "_run", None) is not None and (O._run or {}).get("state") in ("starting", "running", "restoring"):
        return "an optimizer or auto-fit run is active"
    for name, what in (("depthcurve", "a depth-curve run"), ("gputune", "a GPU Tuning benchmark"),
                       ("refusals", "a refusal check")):
        r = getattr(getattr(P, name, None), "_run", None)
        if r and not r.get("_done"):
            return f"{what} is running"
    j = getattr(getattr(P, "fitquant", None), "_job", None)
    if j and j.get("state") == "running":
        return "a Fit job is running"
    return None


def start(body):
    global _run, _thread
    body = body or {}
    kind = str(body.get("kind") or "")
    if kind not in KINDS:
        raise ValueError(f"kind: one of {', '.join(KINDS)} (traffic is GET /api/bench/traffic)")
    iid = str(body.get("instance") or P.INST()["id"])
    inst = P.get_instance(iid)
    with _lock:
        if active():
            raise ValueError("a Bench run is already active")
        why = _others(iid)
        if why:
            raise ValueError(f"{why}; wait for it to finish")
        o = dict(DEFAULTS)
        for k in DEFAULTS:
            if k in body and body[k] is not None:
                o[k] = body[k]
        try:
            o["n_predict"] = max(16, min(int(o["n_predict"]), 1024))
            o["reps"] = max(2, min(int(o["reps"]), 10))
            o["blocks"] = max(2, min(int(o["blocks"]), 24))
            o["sampled_reps"] = max(0, min(int(o["sampled_reps"]), 20))
            o["margin"] = max(0.002, min(float(o["margin"]), 0.5))
            o["alpha"] = max(0.001, min(float(o["alpha"]), 0.2))
            o["quiet_s"] = max(0, min(int(o["quiet_s"]), 3600))
            o["max_wait_s"] = max(60, min(int(o["max_wait_s"]), 7200))
            o["looks"] = sorted({max(2, min(int(x), 24)) for x in o["looks"]})
            o["concurrency"] = sorted({max(1, min(int(x), 32)) for x in o["concurrency"]})
            o["depths"] = [int(x) for x in body["depths"]] if body.get("depths") else None
            o["turn_tokens"] = int(o["turn_tokens"]) if o.get("turn_tokens") else None
            o["seed"] = int(o["seed"])
        except (TypeError, ValueError):
            raise ValueError("options: numbers where numbers are expected")
        if o["preset"] not in PRESETS:
            raise ValueError(f"preset: one of {', '.join(PRESETS)}")
        o["restart"] = bool(o["restart"])
        restarting = kind in RESTARTING or (kind == "calibrate" and o["restart"])
        if restarting and not body.get("allow_restart"):
            raise ValueError("this run restarts the server (a real request in flight would be cut off); "
                             "send allow_restart: true to confirm")
        o["thermal"] = dict(O.THERMAL_DEFAULTS)
        with P.using_instance(inst):
            t = _target(inst)
            base = P.load_params()
            fp, summ = W.live_fp()
        cand = None
        if kind == "compare":
            cand = check_candidate(body.get("candidate"), base)
            with P.using_instance(inst):
                ok, why = _fits(base, cand)
            if not ok:
                raise ValueError(f"the candidate does not fit: {why}")
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run = dict(id=f"{kind}_{ts}", kind=kind, instance=iid, state="running", step="starting",
                   started=_now(), finished=None, opts=o, candidate=cand, fp=fp, config=summ, blocks=[],
                   restarting=restarting, log=[], _t=t, _inst=inst, _base=base, _samples=[],
                   _block_notes=[], _applied={})
        n_ctx = next((int(t["argv"][i + 1]) for i, a in enumerate(t["argv"])
                      if a in ("-c", "--ctx-size") and i + 1 < len(t["argv"])), None)
        if t.get("engine") == "vllm":
            try:
                mm = json.loads(_http(dict(_t=t), "GET", "/v1/models", timeout=10)) or {}
                n_ctx = int(((mm.get("data") or [{}])[0]).get("max_model_len") or 0) or None
            except Exception:
                n_ctx = None
        run["shape"] = shape(inst, o, n_ctx)
        _stop.clear()
        _run = run
    _save(run)
    _log(run, f"{kind} on {iid}: depths {run['shape']['depths']} ({run['shape']['source']}), "
              f"turn {run['shape']['turn_tokens']} tokens, output {run['shape']['output_tokens']}")
    _thread = threading.Thread(target=_worker, args=(run,), daemon=True)
    _thread.start()
    threading.Thread(target=_watch, args=(run,), daemon=True).start()
    return public(run)


def _worker(run):
    inst = run["_inst"]
    try:
        with P.using_instance(inst):
            if run["restarting"]:
                _snapshot(run)
            if run["opts"]["quiet_s"] and _foreign(run):
                _step(run, "waiting for the server to be quiet")
                _quiet(run)
            {"profile": _profile, "calibrate": _calibrate, "compare": _compare,
             "goodput": _goodput}[run["kind"]](run)
            run["state"] = "done"
    except Stopped:
        run["state"] = "stopped"
        _log(run, "stopped")
    except Exception as e:
        run["state"] = "failed"
        run["error"] = str(e)[:500]
        _log(run, f"failed: {e}")
    finally:
        _finish(run)


def _finish(run):
    stopped = _stop.is_set()
    _stop.clear()
    try:
        if run.get("snapshot"):
            with P.using_instance(run["_inst"]):
                _restore(run["snapshot"])
                run["restored"] = True
                if run.get("_applied"):
                    w = _wait_idle(run, "back to the saved settings", RESTORE_WAIT_S)
                    if w is None:
                        _log(run, f"a real request was still running after {RESTORE_WAIT_S} s on the candidate; "
                                  "restarting onto the saved settings anyway")
                    _step(run, "restarting on the saved settings")
                    run["_applied"] = None
                    try:
                        _restart_and_wait(run, "the saved settings")
                        run["applied"] = {}
                    except Exception as e:
                        run["needs_restart"] = True
                        _log(run, f"restart on the saved settings failed: {e}")
    except Exception as e:
        run["restore_error"] = str(e)
        run["needs_restart"] = True
        _log(run, f"RESTORE ERROR: {e}")
    if stopped and run["state"] == "running":
        run["state"] = "stopped"
    run["finished"] = _now()
    run["step"] = None
    run["_done"] = True
    _save(run)


def stop():
    _stop.set()
    _drop_all()
    return dict(ok=True)


def status(iid=None):
    r = _run
    cal = {}
    for d in (P.PANEL / "bench").glob("*") if (P.PANEL / "bench").is_dir() else []:
        c = (_read(d / "calibration.json") or {}).get("latest")
        if c:
            cal[d.name] = {k: c.get(k) for k in ("id", "at", "restart", "blocks", "between_sd_log",
                                                 "selftest", "pairs_for", "deterministic_across_blocks")}
    return dict(active=public(r) if r and not r.get("_done") else None, last=public(r) if r else None,
                calibration=cal, defaults={k: v for k, v in DEFAULTS.items()}, kinds=list(KINDS),
                presets=PRESETS, runs=runs(iid) if iid else None)


def runs(iid, limit=30):
    d = P.PANEL / "bench" / iid
    out = []
    for f in sorted(d.glob("*_*.json"), reverse=True)[:limit] if d.is_dir() else []:
        r = _read(f) or {}
        res = r.get("result") or {}
        out.append(dict(id=r.get("id"), kind=r.get("kind"), state=r.get("state"), started=r.get("started"),
                        finished=r.get("finished"), verdict=res.get("verdict"), candidate=r.get("candidate"),
                        request_s=res.get("request_s"), request_time_pct=res.get("request_time_pct"),
                        blocks=len(r.get("blocks") or [])))
    return out


def read_run(iid, rid):
    if not re.fullmatch(r"[a-z]+_\d{8}_\d{6}", rid or ""):
        raise ValueError("bad run id")
    r = _read(P.PANEL / "bench" / iid / f"{rid}.json")
    if r is None:
        raise ValueError(f"no run {rid}")
    return r


def delete(body):
    iid, rid = str(body.get("instance") or P.INST()["id"]), str(body.get("id") or "")
    read_run(iid, rid)
    if _run and _run.get("id") == rid and not _run.get("_done"):
        raise ValueError("stop the run first")
    (P.PANEL / "bench" / iid / f"{rid}.json").unlink()
    snap = P.PANEL / "bench" / iid / f"{rid}.snapshot"
    if snap.is_dir():
        shutil.rmtree(snap)
    return dict(ok=True)


def recover_on_startup():
    """A run the panel did not finish: put the saved files back, say so."""
    base = P.PANEL / "bench"
    if not base.is_dir():
        return
    for f in base.glob("*/*_*.json"):
        r = _read(f)
        if not r or r.get("state") != "running":
            continue
        if r.get("snapshot") and Path(r["snapshot"]).is_dir():
            try:
                with P.using_instance(P.get_instance(r["instance"])):
                    _restore(r["snapshot"])
                r["restored"] = True
            except Exception as e:
                r["restore_error"] = str(e)
            r["needs_restart"] = bool(r.get("applied"))
        r["state"] = "interrupted"
        r["finished"] = _now()
        r.setdefault("log", []).append("the panel restarted mid-run: saved settings restored"
                                       + ("; the server may still run the candidate - restart it"
                                          if r.get("needs_restart") else ""))
        _atomic(f, r)


def calibration(iid):
    return _read(P.PANEL / "bench" / iid / "calibration.json") or dict(latest=None, history=[])
