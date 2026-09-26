#!/usr/bin/env python3
"""
Auto-fit (added 2026-09-25, 1.0.0): the workload profile's findings measured and acted on,
one change at a time, each one checked afterwards against real traffic.

Off by default, per instance.
  propose  in an idle window, run a small optimizer experiment weighted by this instance's
           real depth mix, and turn a winner into a proposal you apply with one click.
  auto     the same, and apply a winner by itself when all of these hold:
             * it changes speed settings only (AUTO_SAFE: UBATCH, BATCH, CACHE_REUSE,
               SPEC_N_MAX, SPEC_P_MIN, THREADS - none changes what the server can do),
             * a typical request of this workload finishes at least min_gain faster
               (its new prompt at the measured prefill rate plus its output at the decode
               rate over the depth mix),
             * its quality on the task suite is no lower.
           Changes that reshape the server (context size, slots, speculation on or off) are
           always proposals, in either mode. Never a model swap, never a GPU clock change.
Every applied change, whoever applied it, is then compared with the next real requests at
the same depths. One that made them slower is rolled back by itself in auto mode, and
reported with a Roll back button in propose mode.

An experiment starts only when: auto-fit is on; the instance is running; no request for
quiet_min minutes; now is inside an idle window (learned from two weeks of activity, or
hours you set) with an hour of it left; nothing else is measuring or restarting (optimizer,
depth curve, GPU benchmark, refusal check, Fit job); no earlier change is still waiting to
be verified; and fewer than max_per_week experiments ran in the last 7 days. A real
request arriving mid-experiment stops it, and the optimizer puts the saved settings back
byte-for-byte, as after every run. So does running past max_minutes.

Experiments
  tune     the optimizer's launch sweep (speed knobs), weighted by the depth mix. Due when
           the running configuration was never tuned, after 30 days, or when the p90 depth
           moved by half or more since the last tune.
  reshape  the findings' candidates (e.g. CTX=65536, PARALLEL=1, SPEC_TYPE=none) against
           the current settings. Never applied by itself. A change whose point is capacity
           (context freed or gained, one conversation given the whole context) is proposed
           when it is no slower (CAPACITY_SPEED) at no loss of quality; one whose point is
           speed (speculation off) only when it is min_gain faster.
"""
import json, os, re, threading, time

import benchstats as S

P = None
O = None                    # optimizer
W = None                    # workload
DAY = 86400
AUTO_SAFE = ("UBATCH", "BATCH", "CACHE_REUSE", "SPEC_N_MAX", "SPEC_P_MIN", "THREADS")
DEFAULTS = dict(mode="off", window="learned", quiet_min=15, max_per_week=2, min_gain=0.03,
                max_minutes=120, reshape=True, goal="agentic")
VERIFY_MIN = 20             # real requests at matched depths before a verdict
VERIFY_MAX = 400            # ... and never wait for more than this many after the change
VERIFY_DAYS = 7
RETUNE_DAYS = 30
MAX_TRIES = 3               # failed or interrupted attempts per configuration and kind
COOLDOWN_S = 6 * 3600      # after a restart check failed: no automatic experiment for this long
# Findings whose candidate buys capacity, not speed: what it buys, for the proposal's text.
CAPACITY = {"ctx_unused": "frees VRAM the context never used",
            "ctx_limit": "lets sessions run deeper before they are cut",
            "parallel_unused": "gives one conversation the whole context"}
CAPACITY_SPEED = 0.97      # "no slower": within the few % identical runs wobble by
_lock = threading.RLock()
_ext = {}                   # iid -> consecutive polls with someone else's request


def bind(panel_module, optimizer_module, workload_module):
    global P, O, W
    P, O, W = panel_module, optimizer_module, workload_module


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _file(iid):
    return W._dir(iid) / "autofit.json"


def load(iid):
    st = W._read_json(_file(iid), {})
    st["settings"] = dict(DEFAULTS, **(st.get("settings") or {}))
    st.setdefault("experiments", [])
    st.setdefault("dismissed", {})
    return st


def save(iid, st):
    st["experiments"] = st["experiments"][-200:]
    W._write_json(_file(iid), st)


def _log(iid, msg):
    print(f"autofit {iid}: {msg}", flush=True)


# ============================================================================
# settings
# ============================================================================
def set_settings(iid, body):
    with _lock:
        st = load(iid)
        s = dict(st["settings"])
        if "mode" in body:
            if body["mode"] not in ("off", "propose", "auto"):
                raise ValueError("mode: off, propose or auto")
            s["mode"] = body["mode"]
        if "window" in body:
            w = str(body["window"]).strip()
            if w != "learned" and not re.fullmatch(r"([01]?\d|2[0-3])-([01]?\d|2[0-3])", w):
                raise ValueError("window: learned, or local hours like 02-06")
            s["window"] = w
        for k, lo, hi, cast in (("quiet_min", 5, 240, int), ("max_per_week", 0, 14, int),
                                ("min_gain", 0.01, 0.5, float), ("max_minutes", 20, 480, int)):
            if k in body:
                try:
                    v = cast(body[k])
                except (TypeError, ValueError):
                    raise ValueError(f"{k}: a number")
                if not lo <= v <= hi:
                    raise ValueError(f"{k}: {lo} to {hi}")
                s[k] = v
        if "reshape" in body:
            s["reshape"] = bool(body["reshape"])
        if "goal" in body:
            if body["goal"] not in O.GOALS:
                raise ValueError(f"goal: one of {', '.join(O.GOALS)}")
            s["goal"] = body["goal"]
        st["settings"] = s
        save(iid, st)
    _log(iid, f"settings {s}")
    return s


# ============================================================================
# when
# ============================================================================
def in_window(settings, env, now):
    w = settings["window"]
    if w == "learned":
        hours = env["hours"]
        if not any(h["idle"] for h in hours):
            return False, ("no idle window learned yet: it takes two weeks of activity (or set "
                           "hours, e.g. 02-06)")
        how = W.hour_of_week(now)
        if hours[how]["idle"] and hours[(how + 1) % 168]["idle"]:
            return True, "inside a learned idle window"
        return False, f"outside the learned idle windows ({env['idle_hours']} idle hours a week)"
    a, b = (int(x) for x in w.split("-"))
    lt = time.localtime(now)
    cur = lt.tm_hour + lt.tm_min / 60
    end = b if b > a else b + 24
    c = cur if cur >= a else cur + 24
    if a <= c < end - 1:
        return True, f"inside the set window {w}"
    return False, f"outside the set window {w} (it needs an hour left)"


def _others_measuring():
    if O._run is not None:
        return "an optimizer run is active"
    for name, what in (("depthcurve", "a depth-curve run"), ("gputune", "a GPU benchmark"),
                       ("refusals", "a refusal check"), ("benchlab", "a Bench run")):
        r = getattr(getattr(P, name, None), "_run", None)
        if r and not r.get("_done"):
            return f"{what} is running"
    j = getattr(getattr(P, "fitquant", None), "_job", None)
    if j and j.get("state") == "running":
        return "a Fit job is running"
    return None


def can_start(inst, st, env, now, manual=False):
    iid, s = inst["id"], st["settings"]
    with P.using_instance(inst):
        if not P.server_pid():
            return False, "the instance is not running"
    why = _others_measuring()
    if why:
        return False, why
    if W.busy_now(iid, now):
        return False, "a request is being served right now"
    if any(e.get("pending_restart") for e in st["experiments"]):
        return False, "a restart onto changed settings is still pending"
    if now < (st.get("hold_until") or 0) and not manual:
        return False, "cooling down: a restart check failed recently (see the experiment's restart error)"
    if any((e.get("verify") or {}).get("state") in ("waiting-restart", "pending") for e in st["experiments"]):
        return False, "the last change is still being verified on real traffic (one change at a time)"
    if manual:
        return True, "starting on request"
    q = W.quiet_s(iid, now)
    if q < s["quiet_min"] * 60:
        return False, f"waiting for {s['quiet_min']} quiet minutes (last request {int(q // 60)} min ago)"
    ok, why = in_window(s, env, now)
    if not ok:
        return False, why
    n7 = sum(1 for e in st["experiments"] if not e.get("manual") and e.get("started_t", 0) > now - 7 * DAY)
    if n7 >= s["max_per_week"]:
        return False, f"weekly limit reached ({n7} of {s['max_per_week']})"
    return True, why


def _attempts(st, kind, fp, label=None):
    return [e for e in st["experiments"] if e["kind"] == kind and e.get("fp_before") == fp
            and (label is None or label in (e.get("candidates") or []))]


def due(inst, st, env, fp, now):
    """(kind, reason, candidates) of the next experiment, or (None, why nothing is due, None)."""
    if not env["enough"]:
        return None, f"still learning the workload ({env['requests']} requests over {env['span_days']} days)", None
    if fp is None:
        return None, "the instance is not running", None
    exps = st["experiments"]
    if any(e.get("decision") == "proposal" and e.get("proposal") == "pending" for e in exps):
        return None, "a proposal is waiting for you", None
    tries = [e for e in _attempts(st, "tune", fp) if e.get("state") in ("stopped", "failed")]
    tuned = [e for e in exps if e["kind"] == "tune" and e.get("state") == "done"
             and fp in (e.get("fp_before"), e.get("fp_after"))]
    reason = None
    if not tuned:
        reason = "the running configuration has not been tuned for this workload"
    else:
        last = tuned[-1]
        p90, was = env["depth"]["p90"], last.get("p90")
        if now - last.get("finished_t", now) > RETUNE_DAYS * DAY:
            reason = f"{RETUNE_DAYS} days since the last tune"
        elif p90 and was and abs(p90 - was) >= max(8192, 0.5 * was):
            reason = f"the depth mix moved (p90 {was:,} -> {p90:,} tokens)"
    if reason and len(tries) < MAX_TRIES and not (tries and now - tries[-1].get("started_t", 0) < DAY):
        return "tune", reason, None
    if st["settings"]["reshape"]:
        cands = []
        for f in W.findings(inst, env):
            c = f.get("candidate")
            if not c or st["dismissed"].get(c["label"]) == fp:
                continue
            done = [e for e in _attempts(st, "reshape", fp, c["label"])]
            if any(e.get("state") == "done" for e in done) or len(done) >= MAX_TRIES:
                continue
            if done and now - done[-1].get("started_t", 0) < DAY:
                continue
            cands.append(dict(c, finding=f["id"]))
        if cands:
            return "reshape", "findings to measure: " + ", ".join(c["label"] for c in cands), cands[:4]
    return None, "nothing due: the running settings were measured for this workload", None


# ============================================================================
# experiments
# ============================================================================
def _body(inst, st, env, kind, cands):
    s = st["settings"]
    mix = env["mix"]
    depth = env["depth"]["p90"] or 8192
    # The optimizer keeps a knob when its SCORE beats the best so far by min_gain, and speed
    # enters the score as weight * ratio / 2: a knob that is half the auto-apply margin
    # faster moves the score by weight * margin / 4. Never below the optimizer's own
    # noise floor (decode wobbles a few % between identical runs).
    w_speed = O.GOALS.get(s["goal"], O.GOALS["agentic"])["speed"]
    body = dict(goal=s["goal"], budget="quick", phases=["launch"] if kind == "tune" else ["candidates"],
                depth=max(1024, min(int(depth), (env["slot_ctx"] or 16384) - 2048)),
                min_gain=round(max(O.MIN_GAIN, w_speed * s["min_gain"] / 4), 4), source="autofit")
    if mix:
        body["workload"] = mix
    if cands:
        body["candidates"] = [dict(label=c["label"], launch=c["launch"]) for c in cands]
    return body


def start(inst, st, kind, reason, cands, now, manual=False):
    env = W.envelope(inst)
    with P.using_instance(inst):
        fp, summ = W.live_fp()
    body = _body(inst, st, env, kind, cands)
    rid = O.start(inst["id"], body)["id"]
    exp = dict(id=f"x{int(now)}", kind=kind, reason=reason, manual=manual, run_id=rid,
               started=_now(), started_t=int(now), state="running", fp_before=fp, config=summ,
               p90=env["depth"]["p90"], mix=body.get("workload"),
               candidates=[c["label"] for c in cands or []],
               findings={c["label"]: c.get("finding") for c in cands or []})
    st["experiments"].append(exp)
    _log(inst["id"], f"started {kind} ({reason}) as optimizer run {rid}")
    return exp


def _stop(iid, exp, why):
    exp["stop_reason"] = why
    try:
        O.stop()
    except Exception as e:
        _log(iid, f"stop failed: {e}")
    _log(iid, f"stopping {exp['run_id']}: {why}")


def _base_and_rec(r):
    cands = r.get("candidates") or []
    rec = next((c for c in cands if c["id"] == r.get("recommended")), None)
    base = (next((c for c in cands if c["phase"] == "validate" and c["label"].startswith("baseline")
                  and c.get("status") == "ok"), None)
            or next((c for c in cands if c["phase"] == "baseline"), None))
    return base, rec


def decide(inst, st, exp, r, now):
    """Turn a finished optimizer run into: keep, a proposal, or an applied change."""
    iid, s = inst["id"], st["settings"]
    exp.update(finished=_now(), finished_t=int(now), run_state=r.get("state"))
    if r.get("state") != "finished":
        exp.update(state="stopped" if r.get("state") == "stopped" else "failed",
                   outcome=exp.get("stop_reason") or r.get("error") or r.get("state"))
        return
    base, rec = _base_and_rec(r)
    exp["state"] = "done"
    if exp["kind"] == "reshape":
        return _decide_reshape(st, exp, r)
    if not base or not rec or rec is base or rec["phase"] == "baseline" or rec["label"].startswith("baseline"):
        bm = (base or {}).get("metrics") or {}
        exp.update(decision="keep", outcome="the current settings are still the best for this workload",
                   workload_tps=bm.get("workload_tps"))
        return
    bp = r.get("baseline_params") or {}
    diffs = {k: v for k, v in (rec.get("launch") or {}).items() if str(bp.get(k)) != str(v)}
    bm, rm = base.get("metrics") or {}, rec.get("metrics") or {}
    score_gain = (rec["score"] / base["score"] - 1) if base.get("score") else None
    wl = (rm["workload_tps"] / bm["workload_tps"] - 1) if rm.get("workload_tps") and bm.get("workload_tps") else None
    # The margin is on SPEED at this workload's depths: quality dominates the composite score,
    # so a real 10 % speed-up moves the score by only ~2 %. Quality is a separate gate.
    if rec.get("speed_basis") == "workload" and rec.get("speed_ratio"):
        gain = rec["speed_ratio"] - 1                     # a typical request of yours, end to end
    elif wl is not None:
        gain = wl
    else:
        gain = rec["speed_ratio"] - 1 if rec.get("speed_ratio") else None
    q_ok = (rm.get("quality") or 0) >= (bm.get("quality") or 0) - 1e-9
    label = rec["label"].removeprefix("combined: ")          # the optimizer's validation label
    exp.update(cand_id=rec["id"], label=label, diffs=diffs,
               gain=round(gain, 4) if gain is not None else None,
               score_gain=round(score_gain, 4) if score_gain is not None else None,
               workload_gain=round(wl, 4) if wl is not None else None,
               workload_tps=dict(before=bm.get("workload_tps"), after=rm.get("workload_tps")),
               quality=dict(before=bm.get("quality"), after=rm.get("quality")))
    if not diffs:
        exp.update(decision="keep", outcome="the winner is the running configuration")
        return
    safe = all(k in AUTO_SAFE for k in diffs)
    rejected = st.setdefault("rejected", {}).get(label) == exp.get("fp_before")
    if (s["mode"] == "auto" and exp["kind"] == "tune" and safe and q_ok and not rejected
            and gain is not None and gain >= s["min_gain"]):
        _apply(inst, st, exp, by="auto-fit", now=now)
        exp["outcome"] = (f"applied {label}: "
                          + (f"decode at your depths {wl:+.1%}" if wl is not None else f"speed {gain:+.1%}")
                          + ", quality unchanged")
        return
    why = ("changes more than speed settings" if not safe else
           "quality would drop" if not q_ok else
           "it was rolled back once already for making real requests slower" if rejected else
           "gain below the auto-apply margin" if gain is None or gain < s["min_gain"] else
           "auto-fit is in propose mode")
    exp.update(decision="proposal", proposal="pending", outcome=f"proposal: {label} ({why})")


def _decide_reshape(st, exp, r):
    """Each measured candidate against the run's own baseline; the best acceptable one becomes
    the proposal. A reshape is never applied by itself."""
    cands = r.get("candidates") or []
    base = next((c for c in cands if c["phase"] == "baseline" and c.get("status") == "ok"), None)
    if not base:
        exp.update(decision="keep", outcome="the baseline did not complete; nothing to compare with")
        return
    bm, bp = base.get("metrics") or {}, r.get("baseline_params") or {}
    fids = exp.get("findings") or {}
    best, notes = None, []
    for c in cands:
        if c["phase"] != "candidates" or c.get("status") != "ok" or c.get("speed_ratio") is None:
            continue
        m, sr, fid = c.get("metrics") or {}, c["speed_ratio"], fids.get(c["label"])
        q_ok = (m.get("quality") or 0) >= (bm.get("quality") or 0) - 1e-9
        need = CAPACITY_SPEED if fid in CAPACITY else 1 + st["settings"]["min_gain"]
        notes.append(f"{c['label']} {sr:.1%} of current speed" + ("" if q_ok else ", lower quality"))
        if q_ok and sr >= need and (best is None or sr > best["speed_ratio"]):
            best = c
    skipped = [c["label"] for c in cands if c["phase"] == "candidates" and c.get("status") == "skipped"]
    if skipped:
        notes.append("did not fit: " + ", ".join(skipped))
    if best is None:
        exp.update(decision="keep", outcome="measured " + ("; ".join(notes) or "nothing")
                   + ": none is worth proposing")
        return
    m, sr, fid = best.get("metrics") or {}, best["speed_ratio"], fids.get(best["label"])
    diffs = {k: v for k, v in (best.get("launch") or {}).items() if str(bp.get(k)) != str(v)}
    why = (f"{CAPACITY[fid]}, at {sr:.1%} of current speed" if fid in CAPACITY
           else f"a typical request {sr - 1:+.1%}")
    exp.update(cand_id=best["id"], label=best["label"], diffs=diffs, gain=round(sr - 1, 4),
               workload_tps=dict(before=bm.get("workload_tps"), after=m.get("workload_tps")),
               quality=dict(before=bm.get("quality"), after=m.get("quality")),
               decision="proposal", proposal="pending", finding=fid,
               outcome=f"proposal: {best['label']} ({why}; changes what the server can do, so you decide)")


def _R():
    """restarts.py when the panel has it: journaled, verified restarts with a known-good fallback."""
    return getattr(P, "restarts", None)


def _snap(inst, exp, kind):
    R = _R()
    try:
        return R.snapshot_for(inst, f"autofit-{exp['id']}-{kind}") if R else None
    except Exception:
        return None


def _apply(inst, st, exp, by, now):
    snap = _snap(inst, exp, "apply")                      # the known-good, before anything changes
    res = O.apply(exp["run_id"], exp["cand_id"], "params")
    exp.update(decision="applied", applied_by=by, applied_at=int(now), applied=res.get("applied"),
               verify=dict(state="waiting-restart"),
               pending_restart=dict(reason="apply", from_fp=exp.get("fp_before"), since=int(now), tries=0,
                                    snapshot=snap))
    if exp.get("proposal") == "pending":
        exp["proposal"] = "applied"
    _log(inst["id"], f"applied {exp.get('label')} ({by})")


def _restart(inst, reason="restart", snapshot=None):
    R = _R()
    if R is not None:
        rec = R.restart(inst, f"auto-fit {reason}", by="auto-fit", known_good=snapshot)
        return rec["outcome"] == "ok", rec
    with P.using_instance(inst):
        if inst.get("legacy") and P.unit_installed():
            g = P.main_guard(fresh=True)
            if g and g.get("locked"):
                return False, "main's start guard is locked"
            return P._systemctl("restart")
        if P.server_pid():
            P.stop_server()
            time.sleep(2)
        return P.start_server()


def _follow_restart(inst, st, exp, now, force=False):
    pr = exp.get("pending_restart")
    if not pr:
        return
    with P.using_instance(inst):
        fp, _ = W.live_fp()
    if fp and fp != pr["from_fp"]:                       # restarted (by us or by hand) on the new settings
        exp.pop("pending_restart")
        if pr["reason"] == "apply":
            exp.update(fp_after=fp, restarted_at=int(now), verify=dict(state="pending"))
        else:
            exp.setdefault("rollback", {})["restarted_at"] = int(now)
        return
    if pr.get("tries", 0) >= 3:
        exp.pop("pending_restart")
        exp["restart_error"] = pr.get("error") or "the restart did not bring up the new settings"
        if pr["reason"] == "apply":
            exp["verify"] = dict(state="inconclusive", note="never restarted on the new settings")
        return
    if not force and (W.quiet_s(inst["id"], now) < 120 or _others_measuring() or W.busy_now(inst["id"], now)):
        return
    if now - pr.get("last_try", 0) < 300 and not force:
        return
    pr["tries"] = pr.get("tries", 0) + 1
    pr["last_try"] = int(now)
    ok, msg = _restart(inst, pr["reason"], pr.get("snapshot"))
    if isinstance(msg, dict):                             # a safe restart's record (restarts.py)
        rec = msg
        pr.setdefault("restarts", []).append(rec["id"])
        if rec["outcome"] in ("recovered", "failed"):
            exp.pop("pending_restart", None)
            what = "the new settings" if pr["reason"] == "apply" else "the rolled-back settings"
            exp["restart_error"] = (f"{what} failed the restart check ({rec.get('detail')}); "
                                    + ("the previous settings were put back" if rec["outcome"] == "recovered"
                                       else "the known-good settings failed too - check the server"))
            if pr["reason"] == "apply":
                exp["verify"] = dict(state="inconclusive", note="never ran: " + exp["restart_error"])
                st.setdefault("rejected", {})[exp.get("label")] = exp.get("fp_before")
            st["hold_until"] = int(now) + COOLDOWN_S        # no new experiment straight after that
            _log(inst["id"], exp["restart_error"])
            return
        msg = "ok" if ok else (rec.get("detail") or rec["outcome"])
    if not ok:
        pr["error"] = str(msg)[:300]
    _log(inst["id"], f"restart for {pr['reason']}: {'ok' if ok else msg}")


def _follow_verify(inst, st, exp, now):
    v = exp.get("verify") or {}
    if v.get("state") != "pending":
        return
    iid = inst["id"]
    t_apply, t_restart = exp["applied_at"], exp.get("restarted_at", exp["applied_at"])
    reqs = W.requests(iid, t_apply - 14 * DAY)
    before = [r for r in reqs if r["t"] < t_apply and r.get("fp") == exp.get("fp_before")]
    after = [r for r in reqs if r["t"] >= t_restart and r.get("fp") == exp.get("fp_after")]
    c = W.compare(before, after)
    v.update(ratio=c["ratio"], matched=c["matched"], n_after=len(after), n_before=len(before),
             buckets=c["buckets"], checked=_now())
    # Bench statistics: the change as a ratio WITH an interval, from a model of log decode on
    # depth bucket (+ MTP draft acceptance when every request reports it: on the reference box acceptance
    # explained ~80 % of request-to-request noise). The number of requests to wait for is
    # fixed ONCE, from the traffic before the change and the gain that was claimed, so the
    # single look it gets is an honest test however often this runs.
    edges = list(P.CURVE_BUCKETS)
    adjust = bool(before) and all(r.get("accept") is not None for r in before + after)
    if v.get("need") is None and before:
        nz = S.traffic_noise(before, edges)
        sd = nz.get("adjusted_sd") if adjust else nz.get("raw_sd")
        claim = max(float(exp.get("gain") or 0), st["settings"]["min_gain"], 0.02)
        n = S.requests_needed(sd, claim) if sd else None
        v["need"] = max(VERIFY_MIN, min(VERIFY_MAX, n or VERIFY_MIN))
        v["noise_sd"] = round(sd, 5) if sd else None
    need = v.get("need") or VERIFY_MIN
    eff = S.traffic_effect(before, after, edges, adjust=adjust) if after else {}
    if eff.get("mean_log") is not None:
        v["effect"] = dict(pct=round(eff["pct"], 2), lo=round(eff["pct_lo"], 2), hi=round(eff["pct_hi"], 2),
                           adjusted=adjust, verdict=S.verdict(eff["mean_log"], eff["lo_log"], eff["hi_log"],
                                                              st["settings"]["min_gain"]))
    if eff.get("mean_log") is not None and c["matched"] >= VERIFY_MIN and len(after) >= need:
        e = v["effect"]
        if eff["hi_log"] < 0:                       # the whole interval says slower
            v["state"] = "regressed"
            _log(iid, f"{exp.get('label')} made real requests slower ({e['pct']:+.1f} %, "
                      f"95 % interval {e['lo']:+.1f} … {e['hi']:+.1f} %)")
            st.setdefault("rejected", {})[exp.get("label")] = exp.get("fp_before")
            if exp.get("applied_by") == "auto-fit" or st["settings"]["mode"] == "auto":
                rollback(inst, st, exp, by="auto-fit", now=now)
        else:
            v["state"] = "confirmed"
            v["note"] = (f"real traffic {e['pct']:+.1f} % (95 % interval {e['lo']:+.1f} … {e['hi']:+.1f} %), "
                         f"{e['verdict']}" + (", adjusted for draft acceptance" if adjust else ""))
    elif now - t_restart > VERIFY_DAYS * DAY:
        v["state"] = "inconclusive"
        v["note"] = (f"{len(after)} of the {need} real requests needed in {VERIFY_DAYS} days"
                     + (f"; so far {v['effect']['pct']:+.1f} % ({v['effect']['lo']:+.1f} … {v['effect']['hi']:+.1f} %)"
                        if v.get("effect") else ""))


def rollback(inst, st, exp, by, now):
    snap = _snap(inst, exp, "rollback")
    res = O.rollback(exp["run_id"])
    exp["rollback"] = dict(by=by, at=int(now), restored=res.get("restored"), kept=res.get("kept"), note=res.get("note"))
    with P.using_instance(inst):
        fp, _ = W.live_fp()
    exp["pending_restart"] = dict(reason="rollback", from_fp=fp, since=int(now), tries=0, snapshot=snap)
    if (exp.get("verify") or {}).get("state") in ("pending", "waiting-restart"):
        exp["verify"]["state"] = "rolled-back"
    _log(inst["id"], f"rolled back {exp.get('label')} ({by})")
    return res


# ============================================================================
# loop
# ============================================================================
def _running(st):
    return next((e for e in st["experiments"] if e.get("state") == "running"), None)


def tick_one(inst, now):
    """One pass over one instance. Returns True while an experiment runs (poll faster)."""
    iid = inst["id"]
    with _lock:
        st = load(iid)
        exp = _running(st)
        if exp:
            act = O._run
            if act and act.get("id") == exp["run_id"]:
                ext = W.busy_now(iid, now) > 0
                _ext[iid] = _ext.get(iid, 0) + 1 if ext else 0
                if _ext[iid] >= 2 or "someone else's request" in str(act.get("step") or ""):
                    if not exp.get("stop_reason"):
                        _stop(iid, exp, "a real request arrived")
                elif now - exp["started_t"] > st["settings"]["max_minutes"] * 60 and not exp.get("stop_reason"):
                    _stop(iid, exp, f"over the {st['settings']['max_minutes']} minute budget")
                save(iid, st)
                return True
            try:
                r = O.read_run(exp["run_id"])
            except Exception as e:
                r = dict(state="error", error=f"run record unreadable: {e}")
            decide(inst, st, exp, r, now)
            _log(iid, f"{exp['kind']} finished: {exp.get('outcome')}")
        for e in st["experiments"]:
            try:
                _follow_restart(inst, st, e, now)
                _follow_verify(inst, st, e, now)
            except Exception as ex:
                _log(iid, f"follow-up of {e.get('id')} failed: {ex}")
        if st["settings"]["mode"] != "off":
            env = W.envelope(inst)
            with P.using_instance(inst):
                fp, _ = W.live_fp()
            kind, reason, cands = due(inst, st, env, fp, now)
            st["gate"] = dict(at=_now(), due=kind, reason=reason)
            if kind:
                ok, why = can_start(inst, st, env, now)
                st["gate"].update(ok=ok, why=why)
                if ok:
                    try:
                        start(inst, st, kind, reason, cands, now)
                    except ValueError as e:
                        st["gate"]["why"] = f"could not start: {e}"
        save(iid, st)
        return bool(_running(st))


def tick(now=None):
    now = now or time.time()
    busy = False
    for inst in W.llama_instances():
        try:
            busy = tick_one(inst, now) or busy
        except Exception as e:
            _log(inst["id"], f"tick failed: {e}")
    return busy


def worker():
    while True:
        try:
            busy = tick()
        except Exception as e:
            print(f"autofit: {e}", flush=True)
            busy = False
        time.sleep(5 if busy else 60)


# ============================================================================
# what the UI and the API call
# ============================================================================
def _find(st, xid):
    e = next((e for e in st["experiments"] if e["id"] == xid), None)
    if not e:
        raise ValueError(f"no experiment {xid}")
    return e


def run_now(inst, kind):
    if kind not in ("tune", "reshape"):
        raise ValueError("kind: tune or reshape")
    now = time.time()
    with _lock:
        st = load(inst["id"])
        if _running(st):
            raise ValueError("an auto-fit experiment is already running")
        env = W.envelope(inst)
        ok, why = can_start(inst, st, env, now, manual=True)
        if not ok:
            raise ValueError(why)
        cands = None
        if kind == "reshape":
            cands = [dict(f["candidate"], finding=f["id"]) for f in W.findings(inst, env) if f.get("candidate")][:4]
            if not cands:
                raise ValueError("no finding has a candidate to measure")
        exp = start(inst, st, kind, "started by hand", cands, now, manual=True)
        save(inst["id"], st)
    return exp


def stop(inst):
    with _lock:
        st = load(inst["id"])
        exp = _running(st)
        if not exp:
            raise ValueError("no auto-fit experiment is running")
        _stop(inst["id"], exp, "stopped by hand")
        save(inst["id"], st)
    return dict(ok=True)


def apply_proposal(inst, xid, restart=False):
    now = time.time()
    with _lock:
        st = load(inst["id"])
        e = _find(st, xid)
        if e.get("decision") != "proposal" or e.get("proposal") != "pending":
            raise ValueError("that is not a pending proposal")
        _apply(inst, st, e, by="you", now=now)
        if restart:
            _follow_restart(inst, st, e, now, force=True)
        save(inst["id"], st)
    return e


def dismiss(inst, xid):
    with _lock:
        st = load(inst["id"])
        e = _find(st, xid)
        if e.get("proposal") != "pending":
            raise ValueError("that is not a pending proposal")
        e["proposal"] = "dismissed"
        for lab in e.get("candidates") or [e.get("label")]:
            st["dismissed"][lab] = e.get("fp_before")
        save(inst["id"], st)
    return e


def rollback_now(inst, xid, restart=False):
    now = time.time()
    with _lock:
        st = load(inst["id"])
        e = _find(st, xid)
        if e.get("decision") != "applied" or e.get("rollback"):
            raise ValueError("only an applied change that was not rolled back yet")
        rollback(inst, st, e, by="you", now=now)
        if restart:
            _follow_restart(inst, st, e, now, force=True)
        save(inst["id"], st)
    return e


def status(inst):
    now = time.time()
    st = load(inst["id"])
    env = W.envelope(inst)
    with P.using_instance(inst):
        fp, summ = W.live_fp()
    kind, reason, cands = due(inst, st, env, fp, now)
    ok, why = can_start(inst, st, env, now) if kind else (False, reason)
    exp = _running(st)
    act = O._run if exp and O._run and O._run.get("id") == exp["run_id"] else None
    return dict(settings=st["settings"], defaults=DEFAULTS, auto_safe=AUTO_SAFE, goals=list(O.GOALS),
                fp=fp, config=summ, due=dict(kind=kind, reason=reason, candidates=[c["label"] for c in cands or []]),
                gate=dict(ok=ok, why=why),
                running=dict(exp, step=(act or {}).get("step"), run_state=(act or {}).get("state")) if exp else None,
                proposals=[e for e in st["experiments"] if e.get("proposal") == "pending"],
                experiments=list(reversed(st["experiments"][-30:])))
