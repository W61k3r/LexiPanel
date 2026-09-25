#!/usr/bin/env python3
"""
Workload profile (added 2026-09-25, 1.0.0): what this box's real requests look like, learned
from the traffic it already serves, and how well each llama.cpp instance's settings fit it.
This module only observes; autofit.py acts on what it finds.

  store     every completed request, ingested ONCE from the instance's engine log into
            panel/workload/<id>/requests.jsonl, stamped with when it finished and a
            fingerprint of the configuration that served it: context depth, new prompt
            tokens, generated tokens, prefill and decode rate, draft acceptance. llama-server
            truncates its log on every start, so without this the history dies with each
            restart. The log is read from the last byte offset, so each pass costs only the
            new lines. On first run the archived logs are read once for a head start (their
            records carry the archive's time and no fingerprint).
            Requests sent by LexiPanel's own measurements (optimizer, depth curve, GPU
            benchmark, refusal check, Fit jobs) are tagged "bench" and never counted as
            workload.
            From the panel's 2 s sampler, which already polls /slots: seconds observed, seconds
            busy, seconds with every slot busy and the most slots busy at once, per hour, in
            activity.jsonl. Nothing new is polled.
  envelope  the last WINDOW_DAYS summarised: depth, prompt and output percentiles, requests per
            day, peak concurrency, how busy each hour of the week is (the idle windows), draft
            acceptance, real decode against the measured depth curve, the depth mix (where
            decoding time goes) and the latest configuration change with its effect.
  findings  the envelope against the settings, each with the numbers it rests on and, when
            there is one, a candidate the optimizer can measure.
"""
import hashlib, json, os, statistics, threading, time
from pathlib import Path

P = None
WINDOW_DAYS = 14
RETAIN_DAYS = 60
MIN_REQUESTS = 50            # before a distribution counts as evidence
MIN_SPAN_DAYS = 3
IDLE_FRAC = 0.02             # an hour-of-week busier than this is not idle
IDLE_OBSERVED_H = 1.5        # ... and it must have been watched in (most of) two weeks
INGEST_S = 30
BACKFILL_MAX = 20000
REGRESSION = 0.93            # after/before decode at matched depths
IMPROVED = 1.03
_VOLATILE = {"--port", "--host", "--log-file", "--api-key", "--api-key-file", "--alias", "-a"}
_lock = threading.RLock()
_ingest_lock = threading.Lock()  # one reader of a log at a time, or a record is stored twice
_state = {}                  # iid -> ingest state
_act = {}                    # iid -> current-hour activity
_env_cache = {}
_seen_measuring = {}         # iid -> time a measurement was last seen running


def bind(panel_module):
    global P
    P = panel_module


def _dir(iid):
    d = P.PANEL / "workload" / iid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(f, default):
    try:
        return json.loads(Path(f).read_text())
    except (OSError, ValueError):
        return default


def _write_json(f, obj):
    tmp = Path(str(f) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, f)


def _read_jsonl(f, since=0):
    out = []
    try:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if (r.get("t") if "t" in r else r.get("h", 0) * 3600) >= since:
                    out.append(r)
    except OSError:
        pass
    return out


def llama_instances():
    out = []
    for iid in P.instance_ids():
        try:
            inst = P.get_instance(iid)
        except ValueError:
            continue
        if inst.get("engine") in (None, "", "llama.cpp"):
            out.append(inst)
    return out


# ============================================================================
# what the running server is (fingerprint) and whether LexiPanel is measuring it
# ============================================================================
def live_fp():
    """(fingerprint, summary) of the RUNNING server's configuration, or (None, None).
    Call inside using_instance."""
    argv = P.live_cmdline_args()
    if not argv:
        return None, None
    keep, skip = [], False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a in _VOLATILE:
            skip = True
            continue
        keep.append(a)
    build = P._build_of(argv[0])
    fp = hashlib.sha1("\0".join([build or os.path.basename(argv[0])] + keep).encode()).hexdigest()[:12]
    g = lambda *f: P._argv_get(argv, f)
    summ = dict(model=os.path.basename(g("-m", "--model") or ""), ctx=g("-c", "--ctx-size"),
                parallel=g("-np", "--parallel"), kv=g("--cache-type-k", "-ctk"), spec=g("--spec-type"),
                ubatch=g("-ub", "--ubatch-size"), batch=g("-b", "--batch-size"),
                cache_reuse=g("--cache-reuse"), build=build)
    return fp, summ


def measuring(iid):
    """Is LexiPanel itself sending requests to this instance right now?"""
    O, DC = getattr(P, "optimizer", None), getattr(P, "depthcurve", None)
    GT, RF, FQ = getattr(P, "gputune", None), getattr(P, "refusals", None), getattr(P, "fitquant", None)
    for mod in (O, DC, GT, RF):
        r = getattr(mod, "_run", None) if mod else None
        if r and not r.get("_done") and r.get("instance") == iid:
            return True
    j = getattr(FQ, "_job", None) if FQ else None
    return bool(j and j.get("state") == "running")


# ============================================================================
# ingest: engine log -> requests.jsonl
# ============================================================================
def _parse_lines(lines, cur):
    """The same patterns as panel.parse_timings, fed incrementally. A record is complete at
    its slot-release line (which carries the context depth), or when the next request's
    timing starts. Returns (completed records, the record still being built)."""
    done = []
    for ln in lines:
        m = P.TIMING_RE.search(ln)
        if m:
            if cur.get("decode_tps") is not None:
                done.append(cur)
            cur = dict(prompt_ms=float(m.group(1)), prompt_tokens=int(m.group(2)),
                       prompt_tps=float(m.group(3)))
            continue
        if not cur:
            continue
        m = P.EVAL_RE.search(ln)
        if m:
            cur.update(eval_ms=float(m.group(1)), eval_tokens=int(m.group(2)),
                       decode_tps=float(m.group(3)))
            continue
        m = P.ACCEPT_RE.search(ln)
        if m:
            cur.update(accept=float(m.group(1)), accepted=int(m.group(2)), drafted=int(m.group(3)),
                       mean_len=float(m.group(4)))
            continue
        m = P.RELEASE_RE.search(ln)
        if m and cur.get("decode_tps") is not None:
            cur["depth"] = int(m.group(1))
            done.append(cur)
            cur = {}
    return done, cur


def _tail_sig(path, off):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, off - 64))
            return hashlib.sha1(f.read(min(64, off))).hexdigest()
    except OSError:
        return None


def _same_tail(path, s):
    """Are the 64 bytes before our offset still the ones we read? A log truncated by a
    restart that has already grown past the old offset fails this even though its size
    alone looks fine."""
    off, sig = s.get("off") or 0, s.get("sig")
    return not off or not sig or _tail_sig(path, off) == sig


def _backfill(inst, live_path):
    """First run only: the archived engine logs, for a head start."""
    try:
        files, _unknown = P._log_files_for_curve(None)
    except Exception:
        return []
    out = []
    for f in files:
        if str(f) == str(live_path):
            continue
        try:
            txt = Path(f).read_text(errors="ignore")
            t = int(Path(f).stat().st_mtime)
        except OSError:
            continue
        recs, _c = _parse_lines(txt.splitlines(), {})
        out += [dict(r, t=t, bf=True) for r in recs]
        if len(out) >= BACKFILL_MAX:
            break
    return out[:BACKFILL_MAX]


def ingest(inst, now=None):
    """New completed requests of one instance into its store. Returns how many."""
    with _ingest_lock:
        return _ingest(inst, now)


def _ingest(inst, now):
    iid = inst["id"]
    now = int(now or time.time())
    with P.using_instance(inst):
        path = Path(str(P.ENGINE_LOG))
        with _lock:
            s = _state.get(iid)
            if s is None:
                s = _state[iid] = _read_json(_dir(iid) / "state.json", {})
        new = []
        if not s.get("backfilled"):
            new += _backfill(inst, path)
            s["backfilled"] = True
            s["fresh"] = True
        try:
            st = path.stat()
        except OSError:
            st = None
        if st is not None:
            if s.get("ino") != st.st_ino or st.st_size < s.get("off", 0) or not _same_tail(path, s):
                s.update(ino=st.st_ino, off=0, cur={}, sig=None)  # restarted: the log was truncated
            if st.st_size > s["off"]:
                with open(path, "rb") as f:
                    f.seek(s["off"])
                    data = f.read(min(st.st_size - s["off"], 64 << 20))
                nl = data.rfind(b"\n")
                if nl >= 0:                                      # never split a line
                    s["off"] += nl + 1
                    s["sig"] = _tail_sig(path, s["off"])
                    recs, s["cur"] = _parse_lines(data[:nl + 1].decode(errors="ignore").splitlines(),
                                                  s.get("cur") or {})
                    if recs:
                        fp, summ = live_fp()
                        bench = measuring(iid) or now - _seen_measuring.get(iid, 0) < 2 * INGEST_S
                        slot_ctx = (_act.get(iid) or {}).get("n_ctx")
                        extra = dict(t=now, fp=fp, slot_ctx=slot_ctx)
                        if bench:
                            extra["src"] = "bench"
                        if s.get("fresh"):                       # what the live log held before we watched it
                            extra["bf"] = True
                        new += [dict(r, **extra) for r in recs]
                        if fp and summ:
                            _remember_cfg(iid, fp, summ, now)
        s.pop("fresh", None)
        if measuring(iid):
            _seen_measuring[iid] = now
    if new:
        with open(_dir(iid) / "requests.jsonl", "a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        _env_cache.pop(iid, None)
    _write_json(_dir(iid) / "state.json", {k: v for k, v in s.items() if k != "fresh"})
    return len(new)


def _remember_cfg(iid, fp, summ, now):
    f = _dir(iid) / "configs.json"
    cfg = _read_json(f, {})
    c = cfg.get(fp) or dict(summary=summ, first=now)
    c["last"] = now
    cfg[fp] = c
    _write_json(f, cfg)


def configs(iid):
    return _read_json(_dir(iid) / "configs.json", {})


def _trim(iid, now):
    f = _dir(iid) / "requests.jsonl"
    try:
        if f.stat().st_size < 32 << 20:
            return
    except OSError:
        return
    keep = _read_jsonl(f, now - RETAIN_DAYS * 86400)
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
    os.replace(tmp, f)


# ============================================================================
# activity: the sampler's /slots polls, per hour
# ============================================================================
def own_inflight(iid):
    """How many of LexiPanel's own measurement requests are in flight to this instance.
    Each measuring module holds its open connection in _conn while a request runs."""
    n = 0
    for name in ("optimizer", "depthcurve", "gputune", "refusals"):
        mod = getattr(P, name, None)
        r = getattr(mod, "_run", None) if mod else None
        if r and not r.get("_done") and r.get("instance") == iid and getattr(mod, "_conn", None) is not None:
            n += 1
    return n


def note_slots(iid, busy, n_slots, n_ctx=None, now=None):
    """Called by the panel's sampler on every /slots poll of a running server (~2 s).
    LexiPanel's own measurement requests are not counted as busy: an experiment at 3 am
    must not teach the loop that 3 am is a busy hour."""
    now = now or time.time()
    h = int(now // 3600)
    busy = max(0, int(busy) - own_inflight(iid))
    with _lock:
        a = _act.get(iid)
        if a is None:
            a = _act[iid] = _resume(iid, h, now)
        if a["h"] != h:
            _flush(iid, a)
            a.update(h=h, samples=0.0, busy=0.0, full=0.0, max_busy=0)
        dt = min(10.0, now - a["last"]) if a.get("last") else 2.0
        a["last"] = now
        a["samples"] += dt
        if busy > 0:
            a["busy"] += dt
            a["last_busy"] = now
            if n_slots and busy >= n_slots:
                a["full"] += dt
        a["max_busy"] = max(a["max_busy"], int(busy))
        a["n_slots"] = int(n_slots or 0)
        if n_ctx:
            a["n_ctx"] = int(n_ctx)
        a["busy_now"] = int(busy)
        a["busy_at"] = now


def _resume(iid, h, now):
    cur = _read_json(_dir(iid) / "activity-current.json", None)
    base = dict(h=h, samples=0.0, busy=0.0, full=0.0, max_busy=0, first=now)
    if cur and cur.get("h") == h:
        return dict(cur, first=now, last=None)
    if cur and cur.get("samples"):
        _flush(iid, cur)
    return base


def _flush(iid, a):
    if a.get("samples", 0) <= 0:
        return
    rec = {k: (round(a[k], 1) if isinstance(a.get(k), float) else a.get(k))
           for k in ("h", "samples", "busy", "full", "max_busy", "n_slots")}
    with open(_dir(iid) / "activity.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def checkpoint():
    with _lock:
        for iid, a in _act.items():
            _write_json(_dir(iid) / "activity-current.json",
                        {k: a.get(k) for k in ("h", "samples", "busy", "full", "max_busy", "n_slots", "n_ctx")})


def quiet_s(iid, now=None):
    """Seconds since any slot of this instance was last seen busy (since the panel started
    watching, if never)."""
    now = now or time.time()
    a = _act.get(iid) or {}
    if not a:
        return 0.0
    if a.get("busy_now"):
        return 0.0
    return now - (a.get("last_busy") or a.get("first") or now)


def busy_now(iid, now=None, fresh_s=6):
    """Slots busy with someone else's request at the last poll, if that poll is recent (a
    server that is restarting is not polled, and its last reading must not linger)."""
    a = _act.get(iid) or {}
    if (now or time.time()) - (a.get("busy_at") or 0) > fresh_s:
        return 0
    return int(a.get("busy_now") or 0)


# ============================================================================
# envelope
# ============================================================================
def _q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else None


def _pcts(xs, ps=(50, 90, 99)):
    xs = [x for x in xs if x is not None]
    out = {f"p{p}": _q(xs, p) for p in ps}
    out.update(max=max(xs) if xs else None, n=len(xs))
    return out


def _depth(r):
    return r.get("depth") if r.get("depth") is not None else r.get("prompt_tokens")


def hour_of_week(t):
    lt = time.localtime(t)
    return lt.tm_wday * 24 + lt.tm_hour


def idle_map(act):
    grid = [dict(samples=0.0, busy=0.0) for _ in range(168)]
    for a in act:
        g = grid[hour_of_week(a["h"] * 3600)]
        g["samples"] += a.get("samples") or 0
        g["busy"] += a.get("busy") or 0
    out = []
    for i, g in enumerate(grid):
        frac = g["busy"] / g["samples"] if g["samples"] >= 600 else None
        obs = g["samples"] / 3600
        out.append(dict(how=i, busy=round(frac, 4) if frac is not None else None, observed_h=round(obs, 2),
                        idle=frac is not None and frac <= IDLE_FRAC and obs >= IDLE_OBSERVED_H))
    return out


def compare(before, after, min_each=5):
    """Real decode after a change against before it, at matched depths: the median of each
    depth bucket, weighted by how many requests after the change landed there."""
    bk = P.CURVE_BUCKETS
    def bucket(d):
        for lo, hi in zip(bk, bk[1:]):
            if lo <= d < hi:
                return lo
        return bk[-1]
    by = {}
    for side, rs in (("b", before), ("a", after)):
        for r in rs:
            if r.get("decode_tps") and _depth(r) is not None:
                by.setdefault(bucket(_depth(r)), {"b": [], "a": []})[side].append(r["decode_tps"])
    rows, num, den = [], 0.0, 0
    for lo, v in sorted(by.items()):
        if len(v["b"]) >= min_each and len(v["a"]) >= min_each:
            rb, ra = statistics.median(v["b"]), statistics.median(v["a"])
            rows.append(dict(depth_from=lo, before=round(rb, 2), after=round(ra, 2),
                             n_before=len(v["b"]), n_after=len(v["a"])))
            num += (ra / rb) * len(v["a"])
            den += len(v["a"])
    return dict(ratio=round(num / den, 3) if den else None, n_after=sum(len(v["a"]) for v in by.values()),
                matched=den, buckets=rows)


def config_change(reqs):
    """The latest change of configuration inside the window, and what it did."""
    live = [r for r in reqs if r.get("fp") and not r.get("bf")]
    if len(live) < 2 or len({r["fp"] for r in live}) == 1:
        return None
    last = live[-1]["fp"]
    i = len(live) - 1
    while i >= 0 and live[i]["fp"] == last:
        i -= 1
    if i < 0:
        return None
    prev = live[i]["fp"]
    after, before = live[i + 1:], [r for r in live[:i + 1] if r["fp"] == prev]
    c = compare(before, after)
    return dict(c, at=after[0]["t"], fp_before=prev, fp_after=last, n_before=len(before), n_after=len(after))


def workload_mix(reqs, slot_ctx=None, k=3):
    """Up to k representative depths and the share of generated tokens (decode time) near
    each: the weights auto-fit hands the optimizer."""
    pts = sorted((_depth(r), r.get("eval_tokens") or 0) for r in reqs if _depth(r))
    if len(pts) < 10:
        return None
    lim = (int(slot_ctx) - 2048) if slot_ctx else None
    out = {}
    for i in range(k):
        g = pts[i * len(pts) // k:(i + 1) * len(pts) // k]
        if not g:
            continue
        d = max(1024, int(round(g[len(g) // 2][0] / 1024)) * 1024)
        if lim:
            d = min(d, max(1024, lim))
        out[d] = out.get(d, 0) + (sum(t for _d, t in g) or len(g))
    tot = sum(out.values())
    mean = lambda k: round(statistics.mean(r.get(k) or 0 for r in reqs), 1)
    return dict(depths=list(out), weights=[round(w / tot, 4) for w in out.values()],
                prompt_tokens=mean("prompt_tokens"), output_tokens=mean("eval_tokens"))


def slot_ctx_of(params, live=None):
    """Tokens one conversation can reach: the live server's n_ctx per slot when known,
    otherwise worked out from CTX, PARALLEL and the unified-KV settings."""
    if live:
        return int(live)
    ctx = int(str(params.get("CTX") or 0) or 0)
    par = str(params.get("PARALLEL") or "1").strip()
    par = int(par) if par.lstrip("-").isdigit() else 1
    if par <= 1:
        return ctx
    if str(params.get("KV_UNIFIED") or "") == "on":
        per = str(params.get("KV_UNIFIED_PER_SLOT") or "").strip()
        return int(per) if per.isdigit() and int(per) > 0 else ctx
    return ctx // par


def envelope(inst, days=WINDOW_DAYS, now=None, fresh=False):
    iid = inst["id"]
    now = now or time.time()
    c = _env_cache.get(iid)
    if c and not fresh and now - c[0] < 60 and c[1]["window_days"] == days:
        return c[1]
    since = now - days * 86400
    allr = _read_jsonl(_dir(iid) / "requests.jsonl", since)
    reqs = [r for r in allr if r.get("src") != "bench"]
    act = _read_jsonl(_dir(iid) / "activity.jsonl", since)
    a = _act.get(iid)
    if a and a.get("samples"):
        act = act + [{k: a.get(k) for k in ("h", "samples", "busy", "full", "max_busy", "n_slots")}]
    with P.using_instance(inst):
        params = P.load_params()
        try:
            base = P.measured_baseline(inst)
        except Exception:
            base = None
    ts = [r["t"] for r in reqs]
    span = (max(ts) - min(ts)) / 86400 if len(ts) > 1 else 0.0
    obs_days = sum(x.get("samples") or 0 for x in act) / 86400
    live_ctx = next((r.get("slot_ctx") for r in reversed(reqs) if r.get("slot_ctx")), None) or (a or {}).get("n_ctx")
    slot_ctx = slot_ctx_of(params, live_ctx)
    depths = [_depth(r) for r in reqs]
    hours = idle_map(act)
    # real decode against the measured curve at each request's own depth
    vs = None
    if base and base.get("points"):
        def ratios(rs):
            out = []
            for r in rs:
                e = P.expected_at(base, _depth(r))
                if e and r.get("decode_tps"):
                    out.append(r["decode_tps"] / e)
            return out
        recent = ratios([r for r in reqs if r["t"] >= now - 3 * 86400 and not r.get("bf")])
        older = ratios([r for r in reqs if r["t"] < now - 3 * 86400])
        vs = dict(recent=round(statistics.median(recent), 3) if recent else None, n_recent=len(recent),
                  older=round(statistics.median(older), 3) if older else None, n_older=len(older),
                  curve=base.get("source"), curve_note=base.get("note"))
    bk = P.CURVE_BUCKETS
    hist = []
    for lo, hi in zip(bk, bk[1:]):
        g = [r for r in reqs if lo <= (_depth(r) or 0) < hi]
        if g:
            hist.append(dict(lo=lo, hi=hi, n=len(g), tokens_out=sum(r.get("eval_tokens") or 0 for r in g),
                             decode_p50=_q([r.get("decode_tps") for r in g], 50)))
    acc = [r["accept"] for r in reqs if r.get("accept") is not None]
    samples = sum(x.get("samples") or 0 for x in act)
    env = dict(instance=iid, window_days=days, generated=now, requests=len(reqs),
               requests_live=sum(1 for r in reqs if not r.get("bf")), backfilled=sum(1 for r in reqs if r.get("bf")),
               bench_excluded=len(allr) - len(reqs), span_days=round(span, 1), observed_days=round(obs_days, 2),
               per_day=round(len(reqs) / max(span, 1.0), 1) if reqs else 0.0,
               depth=_pcts(depths), prompt_new=_pcts([r.get("prompt_tokens") for r in reqs]),
               output=_pcts([r.get("eval_tokens") for r in reqs]),
               decode=_pcts([r.get("decode_tps") for r in reqs], (10, 50, 90)),
               prefill=_pcts([r.get("prompt_tps") for r in reqs if (r.get("prompt_tokens") or 0) >= 512], (10, 50, 90)),
               accept=_pcts(acc, (10, 50, 90)) if acc else None,
               slot_ctx=slot_ctx, ctx=int(str(params.get("CTX") or 0) or 0),
               parallel=str(params.get("PARALLEL") or "1"), spec=str(params.get("SPEC_TYPE") or ""),
               concurrency=dict(max=max((x.get("max_busy") or 0 for x in act), default=0),
                                n_slots=next((x.get("n_slots") for x in reversed(act) if x.get("n_slots")), None),
                                busy_frac=round(sum(x.get("busy") or 0 for x in act) / samples, 4) if samples else None,
                                full_min_per_day=round(sum(x.get("full") or 0 for x in act) / 60 / max(obs_days, 1e-9), 1)
                                if obs_days >= 1 else None),
               hours=hours, idle_hours=sum(1 for h in hours if h["idle"]),
               depth_hist=hist, vs_curve=vs, mix=workload_mix(reqs, slot_ctx),
               change=config_change(reqs),
               enough=len(reqs) >= MIN_REQUESTS and span >= MIN_SPAN_DAYS)
    _env_cache[iid] = (now, env)
    return env


def requests(iid, since=0, include_bench=False):
    rs = _read_jsonl(_dir(iid) / "requests.jsonl", since)
    return rs if include_bench else [r for r in rs if r.get("src") != "bench"]


# ============================================================================
# findings
# ============================================================================
def _round_up(n, step=16384):
    return int(-(-int(n) // step) * step)


_find_cache = {}


def findings(inst, env=None):
    env = env or envelope(inst)
    c = _find_cache.get(inst["id"])
    if c and c[0] == env["generated"]:
        return c[1]
    out = _findings(inst, env)
    _find_cache[inst["id"]] = (env["generated"], out)
    return out


def _findings(inst, env):
    with P.using_instance(inst):
        p = P.load_params()
    out = []

    def add(fid, level, title, text, candidate=None, **ev):
        out.append(dict(id=fid, level=level, title=title, text=text, candidate=candidate, evidence=ev))

    n, span, days = env["requests"], env["span_days"], env["window_days"]
    if not env["enough"]:
        add("learning", "info", "Still learning this workload",
            f"{n} requests over {span} days so far. Findings need {MIN_REQUESTS} requests over "
            f"{MIN_SPAN_DAYS} days; everything below that is shown but not acted on.", requests=n, span_days=span)
    dmax, slot = env["depth"]["max"], env["slot_ctx"]
    par = int(env["parallel"]) if env["parallel"].lstrip("-").isdigit() else 1
    unified = str(p.get("KV_UNIFIED") or "") == "on"
    if env["enough"] and dmax and slot and dmax <= slot * 0.5:
        new_slot = max(32768, _round_up(dmax * 1.25))
        new_ctx = new_slot * (par if par > 1 and not unified else 1)
        if new_ctx < env["ctx"]:
            freed = None
            try:
                freed = (P.estimate(p)["vram"]["kv_mib"] - P.estimate(dict(p, CTX=str(new_ctx)))["vram"]["kv_mib"])
            except Exception:
                pass
            add("ctx_unused", "tip", "Context reserved but never reached",
                f"The deepest request in {days} days reached {dmax:,} tokens; each conversation can "
                f"hold {slot:,}. CTX={new_ctx:,} keeps 25% above the deepest"
                + (f" and frees about {freed:,} MiB of KV cache" if freed else "")
                + ": VRAM a better KV type, more offload or a second instance could use. A session "
                  "deeper than that would be cut, so this is only ever a proposal.",
                candidate=dict(label=f"CTX={new_ctx}", launch=dict(CTX=str(new_ctx))),
                deepest=dmax, slot_ctx=slot, freed_mib=freed)
    n_near = 0
    if slot:
        n_near = sum(1 for r in requests(inst["id"], env["generated"] - days * 86400)
                     if (_depth(r) or 0) >= slot * 0.95)
    if n_near >= 3:
        cand = None
        nxt = env["ctx"] + 16384 * (par if par > 1 and not unified else 1)
        try:
            if P.estimate(dict(p, CTX=str(nxt)))["vram"]["headroom_mib"] >= 400:
                cand = dict(label=f"CTX={nxt}", launch=dict(CTX=str(nxt)))
        except Exception:
            pass
        add("ctx_limit", "warn", "Sessions reach the context limit",
            f"{n_near} requests in {days} days ran at 95% or more of the {slot:,} tokens a "
            "conversation can hold: those sessions are being cut or context-shifted. "
            + (f"CTX={nxt:,} fits by the estimator." if cand else
               "More context does not fit by the estimator: a smaller KV type (Optimize) or fewer "
               "layers on the GPU would be needed."), candidate=cand, near_limit=n_near, slot_ctx=slot)
    conc = env["concurrency"]
    if par > 1 and env["observed_days"] >= 7 and conc["max"] <= 1:
        add("parallel_unused", "tip", "Parallel slots never used",
            f"{par} slots, but never more than one busy at once in {env['observed_days']} days of "
            f"watching. Each conversation can reach only {slot:,} tokens because the context is "
            f"split {par} ways; PARALLEL=1 gives one conversation all of it.",
            candidate=dict(label="PARALLEL=1", launch=dict(PARALLEL="1")), slots=par, max_busy=conc["max"])
    # With one slot "all busy" is just "busy"; only several slots all taken says requests queued.
    if (conc.get("n_slots") or par) > 1 and conc.get("full_min_per_day") and conc["full_min_per_day"] >= 30:
        add("slots_full", "info", "Every slot busy at peak times",
            f"All {conc.get('n_slots') or par} slots were busy about {conc['full_min_per_day']} minutes "
            "a day on average: a request arriving then waits for one to free. More slots split the "
            "context between them, so this is reported, not tried.", full_min_per_day=conc["full_min_per_day"])
    spec = env["spec"]
    acc = env["accept"]
    if spec and spec != "none" and acc and acc["n"] >= 50 and acc["p50"] is not None and acc["p50"] < 0.45:
        add("spec_low", "tip", "Speculative decoding may not pay on this workload",
            f"Median draft acceptance is {acc['p50']:.2f} over {acc['n']} requests (it paid at 0.8 on "
            "the workloads it was measured on). Below about half, drafting costs more than it saves. "
            "Worth measuring without it.",
            candidate=dict(label="SPEC_TYPE=none", launch=dict(SPEC_TYPE="none")), accept_p50=acc["p50"])
    vs = env["vs_curve"]
    if vs and vs["recent"] is not None and vs["n_recent"] >= 20 and vs["recent"] < 0.85:
        was = f" It ran at {vs['older']:.0%} before that." if vs.get("older") and vs["n_older"] >= 20 else ""
        add("below_curve", "warn", "Real requests run below the measured curve",
            f"Over the last 3 days real decode ran at {vs['recent']:.0%} of what the measured depth "
            f"curve predicts at the same depths ({vs['n_recent']} requests).{was} Usual causes: the "
            "card throttling (GPU Tuning: temperatures, power cap), another process on the card, or "
            "a build / driver change since the curve was measured (re-run it on the Status tab).",
            ratio=vs["recent"], n=vs["n_recent"])
    ch = env["change"]
    if ch and ch["ratio"] is not None and ch["matched"] >= 20:
        cfg = configs(inst["id"])
        diff = _cfg_diff((cfg.get(ch["fp_before"]) or {}).get("summary"), (cfg.get(ch["fp_after"]) or {}).get("summary"))
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ch["at"]))
        if ch["ratio"] < REGRESSION:
            add("change_slower", "warn", "The last configuration change made real requests slower",
                f"Since {when} ({diff or 'configuration changed'}) real decode runs at {ch['ratio']:.0%} of "
                f"before, at the same depths ({ch['matched']} requests compared).", change=ch)
        elif ch["ratio"] >= IMPROVED:
            add("change_faster", "result", "The last configuration change paid off",
                f"Since {when} ({diff or 'configuration changed'}) real decode runs at {ch['ratio']:.0%} of "
                f"before, at the same depths ({ch['matched']} requests compared).", change=ch)
    return out


def _cfg_diff(a, b):
    if not a or not b:
        return None
    return ", ".join(f"{k} {a.get(k)} -> {b.get(k)}" for k in b if a.get(k) != b.get(k))


# ============================================================================
# worker
# ============================================================================
def tick(now=None):
    now = now or time.time()
    for inst in llama_instances():
        try:
            ingest(inst, now)
            if int(now) % 3600 < INGEST_S:
                _trim(inst["id"], now)
        except Exception as e:
            print(f"workload: ingest {inst['id']} failed: {e}", flush=True)
    checkpoint()


def worker():
    while True:
        try:
            tick()
        except Exception as e:
            print(f"workload: {e}", flush=True)
        time.sleep(INGEST_S)


def status(inst, days=WINDOW_DAYS):
    env = envelope(inst, days)
    return dict(envelope=env, findings=findings(inst, env), configs=configs(inst["id"]))
