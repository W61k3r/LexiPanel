#!/usr/bin/env python3
"""
Safe restarts (added 2026-09-25): every planned restart of an inference server through one
journaled, verified procedure. A restart either lands exactly where it was meant to, or puts
the last known-good configuration back - never something in between, silently.

  1 journal   the intent is written before anything changes: who, why, the settings the
              server must run afterwards and which known-good snapshot to fall back to. A
              panel crash at any step is picked up the same way (recover_on_startup).
  2 drain     the gateway holds new requests for the instance (hold_s) instead of failing
              them, and the restart waits for requests in flight or queued to finish (drain_s).
              On a fleet member that shares the instance, the primary's gateway holds it too
              (fleet.announce): its requests go to another replica, or wait when there is none.
              Clients that talk to the engine port directly can only be waited for.
  3 handoff   optional, experimental (kv_handoff, off by default): each slot's KV cache is
              saved before the restart and restored after it when the cache layout did not
              change (same binary, model file, cache types, context, slots, flash attention,
              unified KV), so a pinned agent session does not re-read its whole context. The
              file lives in the server's --slot-save-path, is made 0600 at once and deleted
              after use: it holds conversation tokens.
  4 verify    healthy; running exactly the intended settings (a fallback tier fails this);
              and a canary: a fixed prompt at temperature 0 whose output must match the one
              recorded on this exact configuration fingerprint. A fingerprint is enforced once
              it has reproduced across a restart; until then a mismatch is reported, not acted on.
  5 recover   a failed verify restores the known-good snapshot, restarts once more and
              verifies again. If that fails too the restart ends "failed": it never loops.
  6 report    every restart is a record (timeline, downtime, requests held, outcome) and a
              line in the hash-chained audit log; summary() and export_csv() roll them up.

One restart per instance at a time. Nothing here changes GPU, power or firmware settings.
"""
import csv, hashlib, http.client, io, json, os, re, shutil, threading, time, uuid
from pathlib import Path

P = O = W = G = A = None          # panel, optimizer, workload, gateway, auth
_lock = threading.RLock()
_active = {}                      # iid -> record of the restart in progress
TERMINAL = ("ok", "recovered", "failed", "aborted", "interrupted")
DEFAULTS = dict(kv_handoff=False, hold_s=120, drain_s=600, canary=True)
CANARY_TEXT = ("def merge_intervals(intervals):\n    \"\"\"Merge overlapping [start, end] pairs.\"\"\"\n"
               "    out = []\n    for s, e in sorted(intervals):\n")
CANARY_TOKENS = 24
KV_MIN_FREE_MIB = 6144            # never save a slot when host RAM is tighter than this
KV_FLAGS = (("-ctk", "--cache-type-k"), ("-ctv", "--cache-type-v"), ("-c", "--ctx-size"), ("-np", "--parallel"),
            ("-fa", "--flash-attn"), ("-kvu", "--kv-unified"), ("--no-kv-unified",), ("--swa-full",),
            ("-m", "--model"))


class Abort(Exception):
    pass


class StartFailed(Abort):
    """The server was stopped or restarted and did not come back healthy. Unlike a refusal before
    anything was touched, this is recovered from: the known-good settings go back."""


def bind(panel, optimizer, workload, gateway, auth):
    global P, O, W, G, A
    P, O, W, G, A = panel, optimizer, workload, gateway, auth


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dir(iid, *sub):
    d = P.PANEL.joinpath("restarts", iid, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def settings(iid):
    return dict(DEFAULTS, **(_read(_dir(iid) / "settings.json", {}) or {}))


def set_settings(body):
    iid = str(body.get("instance") or P.INST()["id"])
    P.get_instance(iid)
    s = settings(iid)
    if "kv_handoff" in body:
        s["kv_handoff"] = bool(body["kv_handoff"])
    if "canary" in body:
        s["canary"] = bool(body["canary"])
    for k, lo, hi in (("hold_s", 0, 900), ("drain_s", 0, 3600)):
        if k in body:
            try:
                s[k] = max(lo, min(hi, int(body[k])))
            except (TypeError, ValueError):
                raise ValueError(f"{k}: a whole number of seconds")
    _atomic(_dir(iid) / "settings.json", s)
    return s


# ============================================================================
# the running server
# ============================================================================
def _target(inst):
    with P.using_instance(inst):
        if not P.server_pid():
            return None
        argv = P.live_cmdline_args()
    host = P._argv_get(argv, ("--host",)) or "127.0.0.1"
    port = int(P._argv_get(argv, ("--port",)) or 8080)
    return dict(host="127.0.0.1" if host in ("0.0.0.0", "::", "") else host, port=port, argv=list(argv))


def _http(t, method, path, body=None, timeout=30):
    c = http.client.HTTPConnection(t["host"], t["port"], timeout=timeout)
    try:
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status}: {data[:200]!r}")
        return json.loads(data) if data[:1] in (b"{", b"[") else data.decode(errors="ignore")
    finally:
        c.close()


def _foreign(iid, t):
    if (W.busy_now(iid) or 0) > 0:
        return "a request is running"
    try:
        m = re.search(r"^llamacpp:requests_deferred\s+([0-9.]+)", _http(t, "GET", "/metrics", timeout=5), re.M)
        if m and float(m.group(1)) > 0:
            return "a request is queued"
    except Exception:
        pass
    return None


def _kv_signature(argv):
    def get(names):
        for i, a in enumerate(argv):
            if a in names:
                return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else True
        return None
    sig = {"/".join(f): get(f) for f in KV_FLAGS}
    sig["binary"] = argv[0] if argv else None
    m = sig.get("-m/--model")
    try:
        st = os.stat(m)
        sig["model_file"] = [st.st_size, int(st.st_mtime)]
    except (OSError, TypeError):
        sig["model_file"] = None
    return sig


def _slot_dir(argv):
    for i, a in enumerate(argv):
        if a == "--slot-save-path" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _mem_available_mib():
    try:
        m = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        return int(m.group(1)) // 1024
    except (OSError, AttributeError):
        return None


# ============================================================================
# the procedure
# ============================================================================
def _step(rec, name, **info):
    with _lock:
        rec["state"] = name
        rec["steps"].append(dict(step=name, t=round(time.time(), 2), **info))
        _atomic(_dir(rec["instance"]) / f"{rec['id']}.json", rec)


def _snapshot(inst, dest):
    """Copy the instance's parameter files (the optimizer's list) into `dest`."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    manifest = {}
    with P.using_instance(inst):
        files = O._config_files()
    for i, f in enumerate(files):
        if os.path.exists(f):
            shutil.copy2(f, dest / str(i))
            manifest[f] = str(i)
        else:
            manifest[f] = None
    _atomic(dest / "manifest.json", manifest)
    return str(dest)


def _restore(snap):
    snap = Path(snap)
    for f, key in json.loads((snap / "manifest.json").read_text()).items():
        if key is None:
            if os.path.exists(f):
                os.unlink(f)
        else:
            shutil.copy2(snap / key, f)


def snapshot_for(inst, tag):
    """A caller that is about to change the saved settings (auto-fit apply / rollback)
    snapshots them first; that snapshot is the known-good to fall back to."""
    if not re.fullmatch(r"[\w.-]{1,80}", tag):
        raise ValueError("bad tag")
    return _snapshot(inst, _dir(inst["id"], "snapshots") / tag)


def _drain(rec, inst, s):
    iid = rec["instance"]
    t = _target(inst)
    if G is not None and s["hold_s"] > 0:
        G.hold(iid, s["hold_s"] + s["drain_s"])
        rec["held"] = True
        F = getattr(P, "fleet", None)                 # a fleet member: the primary's gateway holds it too
        try:
            rec["fleet_held"] = bool(F is not None and F.announce("hold", iid, s["hold_s"] + s["drain_s"]))
        except Exception:
            rec["fleet_held"] = False
    if t is None:
        return 0
    t0, calm = time.time(), 0
    while calm < 2:
        if time.time() - t0 > s["drain_s"]:
            rec["notes"].append(f"requests still running after {s['drain_s']} s: restarting anyway")
            break
        f = _foreign(iid, t)
        calm = 0 if f else calm + 1
        if calm < 2:
            time.sleep(2)
    return round(time.time() - t0, 1)


def _kv_save(rec, inst):
    t = _target(inst)
    if t is None:
        return None
    d = _slot_dir(t["argv"])
    if not d:
        rec["notes"].append("kv handoff skipped: the server has no --slot-save-path")
        return None
    free = _mem_available_mib()
    if free is not None and free < KV_MIN_FREE_MIB:
        rec["notes"].append(f"kv handoff skipped: {free} MiB of host RAM available")
        return None
    try:
        slots = _http(t, "GET", "/slots", timeout=10)
    except Exception as e:
        rec["notes"].append(f"kv handoff skipped: /slots: {e}")
        return None
    saved = []
    for s in slots if isinstance(slots, list) else []:
        sid = int(s.get("id", 0))
        name = f"lp-handoff-{rec['id']}-{sid}.bin"          # generated, never user input
        try:
            r = _http(t, "POST", f"/slots/{sid}?action=save", {"filename": name}, timeout=600)
        except Exception as e:
            rec["notes"].append(f"kv handoff: slot {sid} not saved: {e}")
            continue
        path = os.path.join(d, name)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        n = int((r or {}).get("n_saved") or 0)
        if n <= 0:
            _unlink(path)
            continue
        saved.append(dict(slot=sid, file=path, name=name, tokens=n))
    return dict(signature=_kv_signature(t["argv"]), slots=saved) if saved else None


def _unlink(p):
    try:
        os.unlink(p)
    except OSError:
        pass


def _restart_server(inst, timeout=900):
    with P.using_instance(inst):
        pid0 = P.server_pid()
        if inst.get("legacy") and P.unit_installed():
            g = P.main_guard(fresh=True) if hasattr(P, "main_guard") else None
            if g and g.get("locked"):
                raise Abort("main's start guard is locked")
            ok, msg = P._systemctl("restart")
            if not ok:
                raise StartFailed(f"systemctl restart failed: {msg}")
        else:
            if pid0:
                P.stop_server()
                time.sleep(2)
            ok, msg = P.start_server()
            if not ok:                            # stopped already: down until something starts it
                raise (StartFailed if pid0 else Abort)(f"start refused: {msg}")
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(2)
            pid = P.server_pid()
            if pid and pid != pid0:
                t = _target(inst)
                try:
                    if t and (_http(t, "GET", "/health", timeout=5) or {}).get("status") == "ok":
                        return round(time.time() - t0, 1)
                except Exception:
                    pass
            if (P.unit_state() or {}).get("active") == "failed":
                break
    raise StartFailed(f"not healthy after {int(time.time() - t0)} s")


def _canary(inst):
    t = _target(inst)
    r = _http(t, "POST", "/completion", dict(prompt=CANARY_TEXT, n_predict=CANARY_TOKENS, temperature=0.0,
                                               seed=1, ignore_eos=True, cache_prompt=False), timeout=300)
    text = (r or {}).get("content") or ""
    n = ((r or {}).get("timings") or {}).get("predicted_n") or (r or {}).get("tokens_predicted")
    return hashlib.sha256(text.encode()).hexdigest()[:16], text, n


def _verify(rec, inst, intended, s):
    """(ok, detail). Records a new canary on a first sight of a configuration."""
    out = dict()
    with P.using_instance(inst):
        bad = O._live_matches(intended) if intended else []
        try:
            fails = int((P.FAIL_FILE.read_text() or "0").strip() or 0)
        except (OSError, ValueError):
            fails = 0
        fp, summ = W.live_fp()
    out["fingerprint"] = fp
    if bad:
        return False, dict(out, reason="not running the intended settings (a fallback tier?): " + "; ".join(bad))
    if fails > 0:
        out["launch_fails"] = fails
        try:                                   # healthy on the intended settings: don't fall a tier next time
            with P.using_instance(inst):
                P.FAIL_FILE.write_text("0\n")
        except OSError:
            pass
    if not s["canary"]:
        return True, dict(out, canary="off")
    try:
        h, text, n = _canary(inst)
    except Exception as e:
        return False, dict(out, reason=f"canary request failed: {e}")
    if not text.strip() or (n is not None and int(n) < CANARY_TOKENS // 2):
        return False, dict(out, reason=f"canary produced {n} tokens / {len(text)} characters")
    gf = _dir(rec["instance"]) / "golden.json"
    gold = _read(gf, {}) or {}
    g = gold.get(fp)
    if g is None:
        gold[fp] = dict(hash=h, seen=1, mismatches=0, first=_now(), last=_now(), config=summ)
        _atomic(gf, gold)
        return True, dict(out, canary="recorded (first start on this configuration)")
    if g.get("nondeterministic"):
        return True, dict(out, canary="skipped: this configuration's output is not reproducible")
    if h == g["hash"]:
        g.update(seen=g["seen"] + 1, last=_now())
        _atomic(gf, gold)
        return True, dict(out, canary=f"matches ({g['seen']} starts)")
    g["mismatches"] = g.get("mismatches", 0) + 1
    enforced = g["seen"] >= 2
    if not enforced and g["mismatches"] >= 2:
        g["nondeterministic"] = True                      # never reproduced: stop judging it
    _atomic(gf, gold)
    if enforced:
        return False, dict(out, reason="canary output changed on an unchanged configuration "
                                       "(silently wrong: build, driver or hardware)", canary="mismatch")
    return True, dict(out, canary="mismatch, not enforced yet (never reproduced across a restart)")


def _kv_restore(rec, inst, saved):
    t = _target(inst)
    if not saved or t is None:
        return None
    same = _kv_signature(t["argv"]) == saved["signature"]
    restored = []
    for s in saved["slots"]:
        if same:
            try:
                r = _http(t, "POST", f"/slots/{s['slot']}?action=restore", {"filename": s["name"]}, timeout=600)
                restored.append(dict(slot=s["slot"], tokens=int((r or {}).get("n_restored") or 0)))
            except Exception as e:
                rec["notes"].append(f"kv handoff: slot {s['slot']} not restored: {e}")
        _unlink(s["file"])
    if not same:
        rec["notes"].append("kv handoff: the cache layout changed, saved caches discarded")
    return restored


def _finish(rec, outcome, detail=None):
    iid = rec["instance"]
    if rec.get("held") and G is not None:
        h = G.release(iid) or {}
        rec["gateway"] = dict(held=h.get("held", 0), expired=h.get("expired", 0))
    if rec.get("fleet_held"):
        try:
            P.fleet.announce("release", iid)
        except Exception:
            pass
    for s in (rec.get("kv") or {}).get("slots", []):
        _unlink(s["file"])
    rec.update(outcome=outcome, finished=_now(), detail=detail)
    stop = next((s["t"] for s in rec["steps"] if s["step"] == "restarting"), None)
    rec["downtime_s"] = round(time.time() - stop, 1) if stop else 0.0
    _step(rec, outcome)
    code = {"ok": 200, "recovered": 409, "failed": 500, "aborted": 499, "interrupted": 503}[outcome]
    try:
        if A is not None:
            A.audit(rec["by"], "system", "RESTART", f"/restart/{rec['reason']}/{outcome}", iid, code)
    except Exception:
        pass
    with _lock:
        _active.pop(iid, None)
    return rec


def _promote_last_good(inst):
    """The files that just verified become the known-good to fall back to next time."""
    iid = inst["id"]
    new, cur = _dir(iid) / "last-good.new", P.PANEL / "restarts" / iid / "last-good"
    _snapshot(inst, new)
    if cur.exists():
        shutil.rmtree(cur)
    os.replace(new, cur)


def restart(inst, reason, by="operator", intended=None, known_good=None):
    """Run the whole procedure; returns the record. `intended`: the parameters the server must
    run afterwards (default: the saved ones). `known_good`: a snapshot directory to fall back
    to (default: the last snapshot that verified)."""
    iid = inst["id"]
    s = settings(iid)
    with _lock:
        if iid in _active:
            raise ValueError(f"a restart of {iid} is already in progress")
        with P.using_instance(inst):
            intended = dict(intended or P.load_params())
        last_good = _dir(iid) / "last-good"
        kg = known_good or (str(last_good) if (last_good / "manifest.json").exists() else None)
        rec = dict(id=time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:6], instance=iid,
                   reason=str(reason)[:60], by=str(by)[:60], started=_now(), steps=[], notes=[], intended=intended,
                   known_good=kg, settings=s, kv=None, outcome=None)
        _active[iid] = rec
    try:
        _step(rec, "planned")
        _step(rec, "draining")
        rec["drain_s"] = _drain(rec, inst, s)
        if s["kv_handoff"]:
            _step(rec, "saving")
            rec["kv"] = _kv_save(rec, inst)
        _step(rec, "restarting")
        try:
            rec["start_s"] = _restart_server(inst)
            _step(rec, "verifying")
            ok, info = _verify(rec, inst, intended, s)
        except StartFailed as e:                   # e.g. a model that does not fit: the known-good goes back
            ok, info = False, dict(reason=f"it did not come back: {e}")
        rec["verify"] = info
        if ok:
            if rec.get("kv"):
                _step(rec, "restoring_kv")
                rec["kv_restored"] = _kv_restore(rec, inst, rec["kv"])
                rec["kv"]["slots"] = []
            _promote_last_good(inst)
            return _finish(rec, "ok")
        rec["notes"].append("verify failed: " + info.get("reason", "?"))
        if not kg:
            return _finish(rec, "failed", "verify failed and there is no known-good configuration to go back to")
        _step(rec, "recovering", reason=info.get("reason"))
        with P.using_instance(inst):
            _restore(kg)
            good = P.load_params()
        try:
            rec["recover_start_s"] = _restart_server(inst)
            ok2, info2 = _verify(rec, inst, good, s)
        except StartFailed as e:
            ok2, info2 = False, dict(reason=f"it did not come back: {e}")
        rec["verify_recovery"] = info2
        if ok2:
            return _finish(rec, "recovered", info.get("reason"))
        return _finish(rec, "failed", f"{info.get('reason')}; the known-good configuration failed too: "
                                      f"{info2.get('reason')}")
    except Abort as e:
        return _finish(rec, "aborted" if rec["state"] in ("planned", "draining", "saving") else "failed", str(e))
    except Exception as e:
        return _finish(rec, "failed", f"{type(e).__name__}: {e}")


def _measuring():
    """Another LexiPanel job that is using the server (a restart underneath it would spoil it)."""
    B = getattr(P, "benchlab", None)
    if B is not None and B.active():
        return "a Bench run is active"
    r = getattr(O, "_run", None)
    if r and r.get("state") in ("starting", "running", "stopping", "restoring"):
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


def start(body, user=None):
    """POST /api/restarts/start: a safe restart in the background. `user` is the caller the
    panel authenticated (never taken from the body: it goes into the audit log)."""
    iid = str((body or {}).get("instance") or P.INST()["id"])
    inst = P.get_instance(iid)
    with _lock:
        if iid in _active:
            raise ValueError(f"a restart of {iid} is already in progress")
    why = _measuring()
    if why:
        raise ValueError(f"{why}; wait for it to finish (it restarts or measures this server)")
    th = threading.Thread(target=restart, args=(inst, str((body or {}).get("reason") or "operator"),
                                                 user or "operator"), daemon=True)
    th.start()
    time.sleep(0.2)
    return dict(ok=True, active=_active.get(iid))


# ============================================================================
# crash recovery and reporting
# ============================================================================
def recover_on_startup():
    """A restart the panel did not finish. Before the server was touched: aborted, nothing to
    do. After: a background pass waits for the server, verifies it against the journaled
    intent and falls back to the journaled known-good if that fails - the same decision the
    uninterrupted procedure would have made."""
    base = P.PANEL / "restarts"
    if not base.is_dir():
        return
    for f in base.glob("*/*_*.json"):
        rec = _read(f)
        if not rec or rec.get("outcome") or rec.get("state") in TERMINAL:
            continue
        for s in (rec.get("kv") or {}).get("slots", []):
            _unlink(s.get("file", ""))
        rec["notes"] = (rec.get("notes") or []) + ["the panel restarted during this restart"]
        if rec.get("state") in ("planned", "draining", "saving"):
            rec.update(outcome="aborted", finished=_now(), detail="panel restarted before the server was touched")
            _atomic(f, rec)
            try:                                      # a fleet primary may still hold it: let go
                P.fleet.announce("release", rec["instance"])
            except Exception:
                pass
            continue
        rec["kv"] = None
        threading.Thread(target=_resume, args=(rec,), daemon=True).start()


def _resume(rec, wait_s=600):
    iid = rec["instance"]
    try:
        inst = P.get_instance(iid)
    except Exception as e:
        rec.update(outcome="interrupted", finished=_now(), detail=f"instance gone: {e}")
        _atomic(_dir(iid) / f"{rec['id']}.json", rec)
        return
    with _lock:
        if iid in _active:
            return
        _active[iid] = rec
    s = settings(iid)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        t = _target(inst)
        try:
            if t and (_http(t, "GET", "/health", timeout=5) or {}).get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(5)
    try:
        _step(rec, "verifying", resumed=True)
        ok, info = _verify(rec, inst, rec.get("intended") or {}, s)
        rec["verify"] = info
        if ok:
            _promote_last_good(inst)
            return _finish(rec, "ok", "resumed after a panel restart")
        kg = rec.get("known_good")
        if not kg or not (Path(kg) / "manifest.json").exists():
            return _finish(rec, "failed", f"{info.get('reason')}; no known-good configuration to go back to")
        _step(rec, "recovering", reason=info.get("reason"), resumed=True)
        with P.using_instance(inst):
            _restore(kg)
            good = P.load_params()
        rec["recover_start_s"] = _restart_server(inst)
        ok2, info2 = _verify(rec, inst, good, s)
        rec["verify_recovery"] = info2
        return _finish(rec, "recovered" if ok2 else "failed", info.get("reason"))
    except Exception as e:
        return _finish(rec, "failed", f"resume: {type(e).__name__}: {e}")


def _records(iid, since=0):
    d = P.PANEL / "restarts" / iid
    out = []
    for f in sorted(d.glob("*_*.json")) if d.is_dir() else []:
        r = _read(f)
        if r and r.get("started") and time.mktime(time.strptime(r["started"], "%Y-%m-%dT%H:%M:%SZ")) >= since:
            out.append(r)
    return out


def history(iid, limit=50):
    rows = []
    for r in _records(iid)[-limit:][::-1]:
        rows.append({k: r.get(k) for k in ("id", "started", "finished", "reason", "by", "outcome", "detail",
                                           "downtime_s", "drain_s", "start_s", "notes", "gateway")}
                    | dict(canary=(r.get("verify") or {}).get("canary"),
                           kv_tokens=sum(x.get("tokens", 0) for x in (r.get("kv_restored") or []))))
    return rows


def report(iid, rid):
    if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", rid or ""):
        raise ValueError("bad restart id")
    r = _read(P.PANEL / "restarts" / iid / f"{rid}.json")
    if r is None:
        raise ValueError(f"no restart {rid}")
    return r


def summary(iid, days=30):
    rs = [r for r in _records(iid, time.time() - days * 86400) if r.get("outcome")]
    by = {}
    for r in rs:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    down = sorted(r.get("downtime_s") or 0 for r in rs if r["outcome"] in ("ok", "recovered"))
    silent = [r for r in rs if (r.get("verify") or {}).get("reason")]
    return dict(days=days, restarts=len(rs), outcomes=by,
                downtime_median_s=down[len(down) // 2] if down else None,
                downtime_max_s=down[-1] if down else None,
                caught=len(silent), caught_reasons=[r["verify"]["reason"] for r in silent][-10:],
                held=sum((r.get("gateway") or {}).get("held", 0) for r in rs),
                held_expired=sum((r.get("gateway") or {}).get("expired", 0) for r in rs),
                kv_tokens_handed_off=sum(sum(x.get("tokens", 0) for x in (r.get("kv_restored") or [])) for r in rs))


def export_csv(iid, days=90):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "started", "finished", "instance", "reason", "by", "outcome", "downtime_s", "drain_s",
                "start_s", "canary", "held", "held_expired", "kv_tokens", "detail"])
    for r in _records(iid, time.time() - days * 86400):
        g = r.get("gateway") or {}
        w.writerow([r.get("id"), r.get("started"), r.get("finished"), iid, r.get("reason"), r.get("by"),
                    r.get("outcome"), r.get("downtime_s"), r.get("drain_s"), r.get("start_s"),
                    (r.get("verify") or {}).get("canary"), g.get("held", 0), g.get("expired", 0),
                    sum(x.get("tokens", 0) for x in (r.get("kv_restored") or [])), r.get("detail") or ""])
    return buf.getvalue()


def status(iid):
    return dict(active=_active.get(iid), settings=settings(iid), summary=summary(iid), history=history(iid, 20),
                known_good=(_dir(iid) / "last-good" / "manifest.json").exists())
