#!/usr/bin/env python3
"""
Fleet (1.0.0): full LexiPanel on every box, one box the primary (docs/FLEET.md).
Roles in fleet/config.json: standalone (default) | primary | member.
  member   POSTs a report to the primary every 60 s (outbound only), token in fleet/token (0600)
  primary  issues one-time join codes, accepts reports, lists boxes, revokes them
Reports carry what the panel already shows (hardware, instances, speeds, workload counts, alerts),
never prompts, settings or keys.

Remote actions (v2, added 2026-09-26): the primary can start, stop, safely restart and change the
model or sizing of a member's instances, and drain a box out of its gateway. The primary still
never connects to a member: commands ride back in the reply to the member's own report. Each is
  - signed (HMAC-SHA256) with a per-box action key the member receives once and never again: the
    primary keeps only a hash of the report token, and a token seen on the wire must not be
    enough to forge a command;
  - single use and short lived: a random id, refused a second time, expires after ACTION_TTL;
  - opt-in on the member: fleet/actions.json lists the actions and instances it accepts (none
    by default); a plain-http primary is refused unless that box allows it;
  - re-checked by the member's own guard rails (RAM budget, start guard, safe restart with its
    fall-back to the known-good settings, parameter validation). No shell, no file transfer;
  - audited on both boxes, its result returned with the next report.
A member's planned restart (restarts.py) also holds the primary's gateway for that instance:
requests go to another replica meanwhile, or wait when there is none, instead of failing.
"""
import hashlib, hmac, json, os, queue, re, secrets, threading, time, urllib.parse, urllib.request, uuid
from pathlib import Path

P = None
REPORT_S, STALE_N, OFFLINE_S = 60, 3, 600
MAX_BYTES, MIN_GAP_S, JOIN_TTL = 256 * 1024, 20, 1800
ACTION_TTL, SKEW_S, KEEP_ACTIONS = 600, 300, 50
FAST_S = MIN_GAP_S + 2                # report interval while an action is waiting, running or unreported
EVENTS_PER_10MIN, HOLD_MAX_S = 60, 4500   # a restart's hold: hold_s (<= 900) + drain_s (<= 3600)
ACTIONS = ("instance.start", "instance.stop", "instance.restart", "instance.set")
FINAL = ("ok", "failed", "refused", "expired", "cancelled")
# instance.set changes which model runs and how it is sized - never where a server listens, its keys,
# backend, free-form arguments, the draft model (a full-size one locks the host) or a path it writes.
SET_KEYS = frozenset((
    "MODEL", "ALIAS", "CTX", "PARALLEL", "KV_TYPE", "KV_UNIFIED", "FLASH_ATTN", "BATCH", "UBATCH", "NGL",
    "THREADS", "CACHE_RAM", "CACHE_REUSE", "SPEC_TYPE", "SPEC_N_MAX", "N_PREDICT", "TEMP", "TOP_P", "TOP_K",
    "MIN_P", "REASONING_EFFORT", "REASONING_BUDGET",
    "VL_MODEL", "VL_SERVED_NAME", "VL_DTYPE", "VL_MAX_MODEL_LEN", "VL_GPU_MEM_UTIL", "VL_MAX_NUM_SEQS",
    "VL_MAX_NUM_BATCHED_TOKENS", "VL_KV_CACHE_DTYPE", "VL_QUANTIZATION", "VL_ENFORCE_EAGER", "VL_PREFIX_CACHING",
    "OX_MODEL_DIR", "CM_MODEL"))
MODEL_KEYS = frozenset(("MODEL", "VL_MODEL", "OX_MODEL_DIR", "CM_MODEL"))   # must already be on the member
_lock = threading.RLock()
_send_lock = threading.Lock()
_last = {}                            # member: last send result
_wake = threading.Event()             # member: report early (an action finished)
_fast_until = [0.0]
_jobs = queue.Queue()                 # member: accepted actions, run one at a time
_events = queue.Queue()               # member: hold / release notices for the primary, in order
_shared_last = {}                     # member: instance -> its last shared entry
_ev_times = {}                        # primary: box_id -> recent event times
_threads = {}


def bind(panel_module):
    global P
    P = panel_module


def _d():
    d = P.PANEL / "fleet"
    (d / "boxes").mkdir(parents=True, exist_ok=True)
    return d


def _rj(f, default):
    try:
        return json.loads(Path(f).read_text())
    except (OSError, ValueError):
        return default


def _wj(f, obj, mode=0o600):
    tmp = Path(str(f) + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(obj, indent=1))
    os.chmod(tmp, mode)
    os.replace(tmp, f)


def _h(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def _sign(key_hex, cmd):
    body = json.dumps({k: v for k, v in cmd.items() if k != "sig"}, sort_keys=True, separators=(",", ":"))
    return hmac.new(bytes.fromhex(key_hex), body.encode(), hashlib.sha256).hexdigest()


def _fp(key_hex):
    """A key's fingerprint: tells the two boxes apart whether they hold the same key, nothing more."""
    return _h("fleet-action-key:" + key_hex)[:12] if key_hex else None


def _audit(user, path, inst, code):
    A = getattr(P, "auth", None)
    try:
        if A is not None:
            A.audit(str(user)[:80], "fleet", "FLEET", path, inst, code)
    except Exception:
        pass


def config():
    c = _rj(_d() / "config.json", {})
    c.setdefault("role", "standalone")
    c.setdefault("name", os.uname().nodename)
    if not (_d() / "box_id").exists():
        (_d() / "box_id").write_text(str(uuid.uuid4()))
    c["box_id"] = (_d() / "box_id").read_text().strip()
    return c


def set_config(body):
    with _lock:
        c = config()
        role = body.get("role", c["role"])
        if role not in ("standalone", "primary", "member"):
            raise ValueError("role: standalone, primary or member")
        name = str(body.get("name", c["name"]))[:64].strip() or c["name"]
        url = str(body.get("primary_url", c.get("primary_url", "")) or "").strip().rstrip("/")
        if role == "member" and not re.fullmatch(r"https?://[\w.\-\[\]:]+(/[\w.\-/]*)?", url):
            raise ValueError("primary_url: the primary's address, e.g. https://primary-box")
        share = bool(body.get("share", c.get("share", False)))
        c.update(role=role, name=name, primary_url=url, share=share)
        _wj(_d() / "config.json", {k: v for k, v in c.items() if k != "box_id"})
        if role == "member" and body.get("code"):
            _join_primary(c, str(body["code"]).strip())
        return status()


def check_args(action, args):
    """The same shape check on both boxes: the primary refuses early, the member again."""
    if action not in ACTIONS:
        raise ValueError(f"action: one of {', '.join(ACTIONS)}")
    if not isinstance(args, dict):
        raise ValueError("args: an object")
    iid = str(args.get("instance") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", iid):
        raise ValueError("args.instance: an instance id")
    out = dict(instance=iid)
    reason = str(args.get("reason") or "fleet")
    if not re.fullmatch(r"[\w .:/-]{1,60}", reason):
        raise ValueError("args.reason: up to 60 letters, digits and . : / -")
    if action in ("instance.restart", "instance.set"):
        out["reason"] = reason
    if action == "instance.set":
        params = args.get("params")
        if not isinstance(params, dict) or not 0 < len(params) <= 32:
            raise ValueError("args.params: {KEY: value}, 1 to 32 settings")
        bad = sorted(set(params) - SET_KEYS)
        if bad:
            raise ValueError(f"not changeable from the primary: {', '.join(bad)} (model choice and sizing only)")
        for k, v in params.items():
            if isinstance(v, bool) or not isinstance(v, (str, int, float)) or \
                    (isinstance(v, str) and (len(v) > 256 or re.search(r"[\x00-\x1f\x7f]", v))):
                raise ValueError(f"{k}: a plain value (text up to 256 characters or a number)")
        out.update(params={k: params[k] for k in sorted(params)}, restart=args.get("restart", True) is not False)
    return out


def _inst_ok(iid, allowed):
    return "*" in (allowed or []) or iid in (allowed or [])


# ---------------------------------------------------------------- primary side
def new_join_code():
    if config()["role"] != "primary":
        raise ValueError("only the primary issues join codes (set this box's role to primary)")
    code = secrets.token_urlsafe(16)
    with _lock:
        j = _rj(_d() / "joins.json", {})
        now = time.time()
        j = {k: v for k, v in j.items() if v > now}
        j[_h(code)] = now + JOIN_TTL
        _wj(_d() / "joins.json", j)
    return dict(code=code, valid_min=JOIN_TTL // 60)


def join(body):
    if config()["role"] != "primary":
        raise PermissionError("this box is not a fleet primary")
    code, box_id = str(body.get("code") or ""), str(body.get("box_id") or "")
    if not re.fullmatch(r"[0-9a-f-]{36}", box_id):
        raise ValueError("bad box_id")
    with _lock:
        j = _rj(_d() / "joins.json", {})
        exp = j.pop(_h(code), None)
        _wj(_d() / "joins.json", j)
        if not exp or exp < time.time():
            raise PermissionError("join code unknown, used or expired: make a new one on the primary")
        token = secrets.token_urlsafe(32)
        rec = dict(box_id=box_id, name=_cap(body.get("name"), 64), token=_h(token), joined=int(time.time()),
                   last=None, report=None)
        _wj(_d() / "boxes" / f"{box_id}.json", rec)
    return dict(token=token)


def _cap(v, n=200):
    return str(v if v is not None else "")[:n]


def _clean(o, depth=0):
    """Untrusted report -> bounded plain data: strings capped, lists capped, depth capped."""
    if depth > 6:
        return None
    if isinstance(o, dict):
        return {_cap(k, 64): _clean(v, depth + 1) for k, v in list(o.items())[:64]}
    if isinstance(o, list):
        return [_clean(v, depth + 1) for v in o[:64]]
    if isinstance(o, (int, float, bool)) or o is None:
        return o
    return _cap(o)


def _authed(headers, body):
    """(file, record) of the box making this request, once its token checks out."""
    tok = (headers.get("Authorization") or "").removeprefix("Bearer ").strip()
    box_id = str(body.get("box_id") or "")
    f = _d() / "boxes" / f"{box_id}.json"
    if not re.fullmatch(r"[0-9a-f-]{36}", box_id) or not f.exists():
        raise PermissionError("unknown box: join first")
    rec = _rj(f, {})
    if not tok or not secrets.compare_digest(rec.get("token", ""), _h(tok)):
        raise PermissionError("bad token (revoked?)")
    return f, rec


def _box(box_id):
    f = _d() / "boxes" / f"{box_id}.json"
    if not re.fullmatch(r"[0-9a-f-]{36}", str(box_id)) or not f.exists():
        raise ValueError("no such box")
    return f


def intake(headers, raw):
    if config()["role"] != "primary":
        raise PermissionError("this box is not a fleet primary")
    if len(raw) > MAX_BYTES:
        raise ValueError("report too large")
    body = json.loads(raw or b"{}")
    if not isinstance(body, dict):
        raise ValueError("a report is a JSON object")
    results, fa = body.pop("action_results", None), body.pop("fleet_actions", None)
    out = dict(ok=True)
    with _lock:
        f, rec = _authed(headers, body)
        now = time.time()
        if rec.get("last") and now - rec["last"] < MIN_GAP_S:
            raise ValueError("too many reports")
        rec.update(last=int(now), report=_clean(body), name=_cap(body.get("name") or rec.get("name"), 64))
        _take_results(rec, results, now)
        if isinstance(fa, dict):
            allow = [a for a in (fa.get("allow") or [])[:8] if a in ACTIONS]
            rec["member_actions"] = dict(allow=allow, instances=[_cap(i, 32) for i in (fa.get("instances") or [])[:64]],
                                         allow_http=bool(fa.get("allow_http")), key_fp=_cap(fa.get("key_fp"), 12) or None)
            if rec.get("action_key") and rec["member_actions"]["key_fp"] == _fp(rec["action_key"]):
                rec["key_confirmed"] = int(now)
            if allow and not rec.get("key_sent"):
                # once: after this only a re-key on the primary sends a key again
                rec["action_key"] = rec.get("action_key") or secrets.token_hex(32)
                rec["key_sent"] = int(now)
                out["action_key"] = rec["action_key"]
        pend = _deliverable(rec, now)
        if pend:
            out["actions"] = pend
        if pend or any(a["state"] == "running" for a in rec.get("actions") or []):
            out["next_report_s"] = FAST_S
        _wj(f, rec)
    return out


def _deliverable(rec, now):
    """Signed commands the box has not acknowledged yet: sent again with every reply until it does
    (it runs each id once), or until they expire."""
    out = []
    for a in rec.get("actions") or []:
        if a["state"] not in ("queued", "sent"):
            continue
        if now > a["expires"]:
            a.update(state="expired", detail="the box did not pick it up in time", updated=int(now))
            continue
        a.update(state="sent", sent_at=a.get("sent_at") or int(now), updated=int(now))
        out.append({k: a[k] for k in ("id", "box_id", "action", "args", "by", "issued", "expires", "sig")})
    return out


def _take_results(rec, results, now):
    if not isinstance(results, list):
        return
    by_id = {a["id"]: a for a in rec.get("actions") or []}
    for r in results[:64]:
        if not isinstance(r, dict):
            continue
        a = by_id.get(str(r.get("id")))
        st = r.get("state")
        if not a or a["state"] in FINAL or st not in ("running", "ok", "failed", "refused"):
            continue
        a.update(state=st, detail=_cap(r.get("detail"), 300), updated=int(now))
        if isinstance(r.get("restart"), dict):
            a["restart"] = _clean(r["restart"])
        if st in FINAL:
            _audit(f"fleet:{rec.get('name') or rec.get('box_id')}",
                   f"/fleet/{rec.get('box_id')}/{a['action']}/{st}", (a.get("args") or {}).get("instance"),
                   {"ok": 200, "refused": 403}.get(st, 500))


def queue_action(body, by):
    """POST /api/fleet/action {box_id, action, args}: signed now, delivered with the box's next report."""
    if config()["role"] != "primary":
        raise ValueError("only a fleet primary sends actions")
    box_id, action = str(body.get("box_id") or ""), str(body.get("action") or "")
    args = check_args(action, body.get("args") or {})
    with _lock:
        f = _box(box_id)
        rec = _rj(f, {})
        ma, name = rec.get("member_actions") or {}, rec.get("name") or box_id
        if action not in (ma.get("allow") or []):
            raise ValueError(f"{name} does not accept {action}: allow it on that box's own Fleet tab")
        if not _inst_ok(args["instance"], ma.get("instances")):
            raise ValueError(f"{name} does not accept actions for instance {args['instance']}")
        if not rec.get("action_key") or not rec.get("key_sent"):
            raise ValueError(f"{name} has no action key yet: it gets one with its next report")
        now = int(time.time())
        cmd = dict(id="act_" + secrets.token_hex(12), box_id=box_id, action=action, args=args, by=_cap(by, 64),
                   issued=now, expires=now + ACTION_TTL)
        cmd["sig"] = _sign(rec["action_key"], cmd)
        rec["actions"] = ((rec.get("actions") or []) + [dict(cmd, state="queued", updated=now,
                                                          detail="waits for the box's next report")])[-KEEP_ACTIONS:]
        _wj(f, rec)
    return _public_action(cmd | dict(state="queued"))


def cancel_action(body):
    with _lock:
        f = _box(str(body.get("box_id") or ""))
        rec = _rj(f, {})
        for a in rec.get("actions") or []:
            if a["id"] == body.get("id"):
                if a["state"] not in ("queued", "sent"):
                    raise ValueError(f"already {a['state']}: the box has it")
                a.update(state="cancelled", detail="cancelled before the box took it", updated=int(time.time()))
                _wj(f, rec)
                return _public_action(a)
    raise ValueError("no such action")


def drain(body, by=None):
    """POST /api/fleet/drain {box_id, on}: the gateway sends a drained box nothing new."""
    with _lock:
        f = _box(str(body.get("box_id") or ""))
        rec = _rj(f, {})
        rec["drain"] = dict(since=int(time.time()), by=_cap(by, 64)) if body.get("on") else None
        _wj(f, rec)
    return dict(ok=True, drain=rec["drain"])


def rekey(body):
    """A new action key for a box, sent with its next report (the old one stops working)."""
    with _lock:
        f = _box(str(body.get("box_id") or ""))
        rec = _rj(f, {})
        rec.update(action_key=secrets.token_hex(32), key_sent=None, key_confirmed=None)
        _wj(f, rec)
    return dict(ok=True, key_fp=_fp(rec["action_key"]))


def event(headers, raw):
    """POST /api/fleet/event, from a member: a planned restart of one of its shared instances
    begins (hold) or ends (release). Token-checked like a report."""
    if config()["role"] != "primary":
        raise PermissionError("this box is not a fleet primary")
    if len(raw) > 4096:
        raise ValueError("event too large")
    body = json.loads(raw or b"{}")
    if not isinstance(body, dict):
        raise ValueError("an event is a JSON object")
    with _lock:
        _f, rec = _authed(headers, body)
        now = time.time()
        box_id = rec["box_id"]
        ts = [t for t in _ev_times.get(box_id, []) if now - t < 600]
        if len(ts) >= EVENTS_PER_10MIN:
            raise ValueError("too many events")
        _ev_times[box_id] = ts + [now]
    ev, iid = body.get("event"), str(body.get("instance") or "")
    if ev not in ("hold", "release"):
        raise ValueError("event: hold or release")
    G = getattr(P, "gateway", None)
    key = f"box:{box_id}/{iid}"
    if ev == "release":
        h = (G.release(key) if G is not None else None) or {}
        if h:                                       # what this gateway held, on the box's record and the command
            with _lock:
                f, rec = _authed(headers, body)
                held = dict(instance=_cap(iid, 32), at=int(time.time()), held=h.get("held", 0),
                            expired=h.get("expired", 0), s=round(time.time() - h.get("since", time.time()), 1))
                rec["holds"] = ((rec.get("holds") or []) + [held])[-20:]
                a = next((a for a in reversed(rec.get("actions") or []) if a["state"] in ("sent", "running")
                          and a["action"] in ("instance.restart", "instance.set")
                          and (a.get("args") or {}).get("instance") == iid), None)
                if a:
                    a["held_at_primary"] = held["held"]
                _wj(f, rec)
        return dict(ok=True, held=h.get("held", 0))
    s = next((s for s in (rec.get("report") or {}).get("shared") or [] if s.get("instance") == iid), None)
    if not s:
        raise ValueError(f"{iid} is not shared with this primary")
    secs = max(0, min(int(body.get("seconds") or 0), HOLD_MAX_S))
    if G is not None:
        G.hold(key, secs, names={str(n) for n in s.get("names") or []}, replica=(str(s.get("addr")), int(s.get("port"))))
    return dict(ok=True, seconds=secs)


def revoke(box_id):
    f = _d() / "boxes" / f"{box_id}.json"
    if not re.fullmatch(r"[0-9a-f-]{36}", str(box_id)) or not f.exists():
        raise ValueError("no such box")
    f.unlink()
    return dict(ok=True)


def _public_action(a):
    return {k: a.get(k) for k in ("id", "action", "args", "by", "issued", "expires", "state", "detail", "updated",
                                  "restart", "held_at_primary")}


def boxes(now=None):
    now = now or time.time()
    out = []
    for f in sorted((_d() / "boxes").glob("*.json")):
        r = _rj(f, {})
        age = now - r["last"] if r.get("last") else None
        state = ("waiting for its first report" if age is None else "online" if age < STALE_N * REPORT_S
                 else "stale" if age < OFFLINE_S else "offline")
        ma = r.get("member_actions") or {}
        key = ("none" if not r.get("key_sent") else "matches" if ma.get("key_fp") == _fp(r.get("action_key"))
               else ("lost on the box: re-key" if r.get("key_confirmed") else "sent, not confirmed yet")
               if not ma.get("key_fp") else "differs: re-key")
        out.append(dict(box_id=r.get("box_id"), name=r.get("name"), joined=r.get("joined"), last=r.get("last"),
                        age_s=int(age) if age is not None else None, state=state, report=r.get("report"),
                        drain=r.get("drain"), remote=dict(allow=ma.get("allow") or [], instances=ma.get("instances") or [],
                                                          allow_http=ma.get("allow_http"), key=key),
                        actions=[_public_action(a) for a in (r.get("actions") or [])[-10:]][::-1],
                        holds=(r.get("holds") or [])[-5:][::-1]))
    return out


# ---------------------------------------------------------------- member side
def _post(url, obj, token=None, timeout=15):
    rq = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST",
                                headers={"Content-Type": "application/json",
                                         **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def _join_primary(c, code):
    st, r = _post(c["primary_url"] + "/api/fleet/join", dict(code=code, box_id=c["box_id"], name=c["name"]))
    if st != 200 or not r.get("token"):
        raise ValueError(f"the primary refused the join: {r.get('error') or st}")
    f = _d() / "token"
    fd = os.open(str(f) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:                      # 0600 from the start, never briefly readable
        fh.write(r["token"])
    os.replace(str(f) + ".tmp", f)


def build_report():
    c = config()
    try:
        import mcp_server
        ver = mcp_server.VERSION
    except Exception:
        ver = None
    gpus = []
    for d in P.gpu_devices(probe=False):
        if d.get("pci") != "cpu":
            gpus.append(dict(pci=d.get("pci"), name=d.get("name"), vram_mib=d.get("vram_total_mib"),
                             driver=d.get("driver")))
    insts = []
    for iid in P.instance_ids():
        try:
            inst = P.get_instance(iid)
            with P.using_instance(inst):
                pid = P.server_pid()
                p = P.load_params()
            eng = inst.get("engine") or "llama.cpp"
            model = p.get("MODEL") or p.get("VL_MODEL") or p.get("CM_MODEL") or p.get("OX_MODEL_DIR") or ""
            rec = dict(id=iid, name=inst.get("name"), engine=eng, state="running" if pid else "stopped",
                       model=os.path.basename(str(model).rstrip("/")), backend=p.get("BACKEND"),
                       ctx=p.get("CTX") or p.get("VL_MAX_MODEL_LEN"), port=p.get("PORT"))
            if eng == "llama.cpp" and getattr(P, "workload", None):
                e = P.workload.envelope(inst)
                rec["workload"] = dict(requests=e["requests"], per_day=e["per_day"], depth_p90=e["depth"]["p90"],
                                       decode_p50=e["decode"]["p50"], idle_hours=e["idle_hours"])
                pend = [x for x in P.autofit.load(iid)["experiments"] if x.get("proposal") == "pending"]
                rec["autofit"] = dict(mode=P.autofit.load(iid)["settings"]["mode"], pending=len(pend))
            insts.append(rec)
        except Exception as ex:
            insts.append(dict(id=iid, error=str(ex)[:120]))
    shared = []
    if c.get("share"):
        # instances the primary's gateway may route to: running, on the LAN, without an API key
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((urllib.parse.urlparse(c.get("primary_url") or "http://127.0.0.1").hostname, 9))
            addr = s.getsockname()[0]
            s.close()
        except OSError:
            addr = None
        R = getattr(P, "restarts", None)
        restarting = set(getattr(R, "_active", None) or ())
        for iid in P.instance_ids():
            try:
                inst = P.get_instance(iid)
                eng = inst.get("engine") or "llama.cpp"
                if eng not in ("llama.cpp", "onnx", "camelid", "vllm"):
                    continue
                with P.using_instance(inst):
                    if not P.server_pid():
                        if iid in restarting and iid in _shared_last:
                            shared.append(_shared_last[iid])       # mid-restart: keep its place at the primary
                        continue
                    argv, v = P.live_cmdline_args(), P.load_params()
                g = lambda *f: P._argv_get(argv, f)
                if eng == "vllm":                                   # retitles itself: its argv is gone
                    host, port = str(v.get("HOST") or "127.0.0.1"), v.get("PORT")
                else:
                    host = g("--host") or (g("--addr") or "").rpartition(":")[0] or "127.0.0.1"
                    port = g("--port") or (g("--addr") or ":").rpartition(":")[2]
                if host in ("127.0.0.1", "localhost", "::1") or v.get("API_KEY") or v.get("OX_API_KEY") \
                        or v.get("CM_API_KEY") or v.get("VL_API_KEY"):
                    continue
                if eng == "vllm":
                    names = sorted({P.vllm_engine.served_name(v)})
                else:
                    model = g("-m", "--model") or g("--model-dir") or ""
                    names = sorted({n for n in (g("-a", "--alias"), os.path.basename(str(model).rstrip("/"))) if n})
                if addr and port:
                    shared.append(dict(instance=iid, addr=addr, port=int(port), names=names))
            except Exception:
                continue
    _shared_last.clear()
    _shared_last.update({s["instance"]: s for s in shared})
    npus = []
    try:
        npus = [dict(vendor=n.get("vendor"), driver=n.get("driver")) for n in P.onnxrt.npus()]
    except Exception:
        pass
    return dict(v=1, box_id=c["box_id"], name=c["name"], sent=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                lexipanel=ver, os=os.uname().sysname + " " + os.uname().release,
                hardware=dict(ram_mib=P._meminfo_mb("MemTotal"), gpus=gpus, npus=npus), instances=insts,
                shared=shared)


# member: remote actions ---------------------------------------------------
def policy():
    """What this box accepts from its primary (fleet/actions.json). Nothing by default."""
    p = _rj(_d() / "actions.json", {})
    return dict(allow=[a for a in p.get("allow") or [] if a in ACTIONS],
                instances=[str(i) for i in (p.get("instances") if p.get("instances") is not None else ["*"])],
                allow_http=bool(p.get("allow_http")))


def set_policy(body):
    allow = body.get("allow") or []
    if not isinstance(allow, list) or any(a not in ACTIONS for a in allow):
        raise ValueError(f"allow: a list of {', '.join(ACTIONS)}")
    inst = body.get("instances", ["*"])
    if isinstance(inst, str):
        inst = [x.strip() for x in inst.split(",") if x.strip()]
    if not isinstance(inst, list) or any(i != "*" and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", str(i)) for i in inst):
        raise ValueError("instances: instance ids, or * for all")
    with _lock:
        _wj(_d() / "actions.json", dict(allow=sorted(set(allow), key=ACTIONS.index), instances=inst or ["*"],
                                        allow_http=bool(body.get("allow_http"))))
    _wake.set()                                  # tell the primary now (and fetch a key if this enabled one)
    return status()


def _key():
    try:
        k = (_d() / "action_key").read_text().strip()
        return k if re.fullmatch(r"[0-9a-f]{64}", k) else None
    except OSError:
        return None


def _seen(add=None, until=0):
    with _lock:
        now = time.time()
        s = {k: v for k, v in _rj(_d() / "seen.json", {}).items() if v > now}
        if add:
            s[add] = until
            _wj(_d() / "seen.json", s)
        return s


def _log(entry):
    f = _d() / "action-log.jsonl"
    with _lock:
        lines = f.read_text().splitlines()[-499:] if f.exists() else []
        lines.append(json.dumps(dict(entry, at=int(time.time()))))
        tmp = Path(str(f) + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, f)


def action_log(n=20):
    f = _d() / "action-log.jsonl"
    out = []
    for l in (f.read_text().splitlines() if f.exists() else [])[-n:]:
        try:
            out.append(json.loads(l))
        except ValueError:
            continue
    return out[::-1]


def _result(cmd, state, detail, **extra):
    """Queue a result for the next report (fleet/outbox.json survives a panel restart)."""
    r = dict(id=cmd["id"], state=state, detail=str(detail)[:300], **extra)
    with _lock:
        _wj(_d() / "outbox.json", _rj(_d() / "outbox.json", []) + [r])
    _log(dict(r, action=cmd.get("action"), instance=(cmd.get("args") or {}).get("instance"), by=cmd.get("by")))
    _fast_until[0] = time.time() + 300
    _wake.set()


def verify(cmd, c=None, now=None):
    """None when this box should run `cmd`, else the reason it will not ("seen": a copy of one it took)."""
    c, now, pol = c or config(), now or time.time(), policy()
    if not isinstance(cmd, dict) or not re.fullmatch(r"act_[0-9a-f]{24}", str(cmd.get("id") or "")):
        return "malformed"
    if cmd["id"] in _seen():
        return "seen"
    key = _key()
    if not key:
        return "this box has no action key yet"
    if not isinstance(cmd.get("sig"), str) or not hmac.compare_digest(cmd["sig"], _sign(key, cmd)):
        return "bad signature (not from this box's primary, or altered on the way)"
    if cmd.get("box_id") != c["box_id"]:
        return "meant for another box"
    iss, exp = cmd.get("issued"), cmd.get("expires")
    if not (isinstance(iss, int) and isinstance(exp, int)) or exp - iss > 3600 or iss > now + SKEW_S \
            or now > exp + SKEW_S:
        return "expired (or this box's clock is off by more than 5 minutes)"
    if not str(c.get("primary_url") or "").startswith("https://") and not pol["allow_http"]:
        return "the primary is plain http: allow that on this box's Fleet tab to accept actions over it"
    if cmd.get("action") not in pol["allow"]:
        return f"{cmd.get('action')} is not allowed on this box"
    try:
        args = check_args(cmd["action"], cmd.get("args"))
    except ValueError as e:
        return str(e)
    if not _inst_ok(args["instance"], pol["instances"]):
        return f"instance {args['instance']} is not open to the primary on this box"
    return None


def _receive(cmd, c):
    why = verify(cmd, c)
    if why == "seen":
        return                                   # sent again until acknowledged; run once
    if why == "malformed":
        _log(dict(state="refused", detail="malformed command"))
        return
    _seen(cmd["id"], min(int(cmd.get("expires") or 0), int(time.time()) + ACTION_TTL) + SKEW_S)
    iid = (cmd.get("args") or {}).get("instance") if isinstance(cmd.get("args"), dict) else None
    if why:
        _result(cmd, "refused", why)
        _audit(f"fleet:{cmd.get('by') or 'primary'}", f"/fleet/{cmd.get('action')}/refused", iid, 403)
        return
    _result(cmd, "running", "accepted")
    _jobs.put(cmd)


def _restart_result(rec):
    st = "ok" if rec.get("outcome") == "ok" else "failed"
    detail = {"ok": "restarted and verified",
              "recovered": f"the new settings failed ({rec.get('detail')}), so it went back to the known-good ones "
                           "and runs on them"}.get(rec.get("outcome"), str(rec.get("detail") or rec.get("outcome")))
    return st, detail, dict(restart=dict(id=rec.get("id"), outcome=rec.get("outcome"),
                                         downtime_s=rec.get("downtime_s"), held=(rec.get("gateway") or {}).get("held")))


def _model_ok(k, v):
    """A model setting must name something already on this box, under its models folder."""
    base = os.path.realpath(str(getattr(P, "MODELS", "") or ""))
    p = os.path.realpath(str(v))
    if not base or not p.startswith(base + os.sep) or not os.path.exists(p):
        return f"{k}: {v} is not in this box's models folder ({base}); copy it there first"
    return None


def execute(cmd):
    """Run one accepted command through the member's own code paths: (state, detail, extra)."""
    a, args = cmd["action"], check_args(cmd["action"], cmd["args"])
    inst = P.get_instance(args["instance"])
    R = getattr(P, "restarts", None)
    by = f"fleet:{cmd.get('by') or 'primary'}"
    busy = (R is not None and inst["id"] in (getattr(R, "_active", None) or {}) and "a restart of it is in progress") \
        or (R is not None and R._measuring())
    if busy:
        return "refused", f"{busy}; send it again when that is done", {}
    if a in ("instance.restart", "instance.set") and R is None:
        return "refused", "this box has no safe restarts: update LexiPanel", {}
    if a == "instance.start":
        with P.using_instance(inst):
            ok, msg = P.start_server()
        return ("ok" if ok else "failed"), msg, {}
    if a == "instance.stop":
        with P.using_instance(inst):
            ok, msg = P.stop_server()
        return ("ok" if ok else "failed"), msg, {}
    if a == "instance.restart":
        with P.using_instance(inst):
            if not P.server_pid():
                return "refused", "it is not running: start it instead", {}
        return _restart_result(R.restart(inst, args["reason"], by=by))
    params = args["params"]
    for k in MODEL_KEYS & set(params):
        why = _model_ok(k, params[k])
        if why:
            return "refused", why, {}
    snap = R.snapshot_for(inst, f"fleet-{cmd['id']}")           # the known-good, before anything changes
    with P.using_instance(inst):
        P.save_params(dict(params))
        running = P.server_pid()
    if not args["restart"] or not running:
        return "ok", ("saved; it runs with them from its next start" if not running else
                      "saved; it runs with them from its next restart"), {}
    return _restart_result(R.restart(inst, args["reason"], by=by, known_good=snap))


def run_one(cmd):
    _wj(_d() / "running.json", dict(id=cmd["id"], action=cmd["action"], args=cmd["args"], by=cmd.get("by"),
                                    started=int(time.time())))
    try:
        st, detail, extra = execute(cmd)
    except Exception as e:
        st, detail, extra = "failed", f"{type(e).__name__}: {e}", {}
    _result(cmd, st, detail, **extra)
    _audit(f"fleet:{cmd.get('by') or 'primary'}", f"/fleet/{cmd['action']}/{st}", cmd["args"].get("instance"),
           {"ok": 200, "refused": 403}.get(st, 500))
    try:
        (_d() / "running.json").unlink()
    except OSError:
        pass
    return st


def _executor():
    while True:
        run_one(_jobs.get())


def _recover_interrupted():
    """An action the panel was running when it stopped: reported as failed, never re-run (a safe
    restart it started is finished by restarts.recover_on_startup and shows in its history)."""
    r = _rj(_d() / "running.json", None)
    if r and r.get("id"):
        _result(r, "failed", "this box's panel restarted during the action; see its Safe restarts history")
        (_d() / "running.json").unlink()


def announce(event, iid, seconds=0):
    """restarts.py: a planned restart of `iid` begins (hold) or ends (release). A member that
    shares the instance tells its primary, in order, without delaying the restart."""
    c = config()
    if c["role"] != "member" or not (_d() / "token").exists():
        return False
    if event == "hold" and (not c.get("share") or iid not in _shared_last):
        return False                             # (a release is always sent: after a panel restart
                                                 # the shared list is empty, the primary's hold is not)
    _events.put((event, iid, int(seconds)))
    _start("events", _event_sender)
    return True


def _event_sender():
    while True:
        event, iid, seconds = _events.get()
        try:
            c = config()
            _post(c["primary_url"] + "/api/fleet/event", dict(box_id=c["box_id"], event=event, instance=iid,
                                                              seconds=seconds),
                  (_d() / "token").read_text().strip(), timeout=5)
        except Exception:
            pass


def _start(name, fn):
    with _lock:
        if name not in _threads or not _threads[name].is_alive():
            _threads[name] = threading.Thread(target=fn, daemon=True)
            _threads[name].start()


def send_now():
    c = config()
    if c["role"] != "member":
        raise ValueError("this box is not a fleet member")
    try:
        token = (_d() / "token").read_text().strip()
    except OSError:
        raise ValueError("not joined yet: paste a join code from the primary")
    with _send_lock:
        rep, pol, key = build_report(), policy(), _key()
        rep["fleet_actions"] = dict(allow=pol["allow"], instances=pol["instances"], allow_http=pol["allow_http"],
                                    key_fp=_fp(key))
        with _lock:
            out = _rj(_d() / "outbox.json", [])
        if out:
            rep["action_results"] = out
        st, r = _post(c["primary_url"] + "/api/fleet/report", rep, token)
        _last.update(at=int(time.time()), ok=st == 200, status=st, error=None if st == 200 else r.get("error"))
        if st != 200:
            return dict(_last)
        with _lock:
            done = {(x["id"], x["state"]) for x in out}
            _wj(_d() / "outbox.json", [x for x in _rj(_d() / "outbox.json", []) if (x["id"], x["state"]) not in done])
            if isinstance(r.get("action_key"), str) and re.fullmatch(r"[0-9a-f]{64}", r["action_key"]):
                f = _d() / "action_key"
                fd = os.open(str(f) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as fh:
                    fh.write(r["action_key"])
                os.replace(str(f) + ".tmp", f)
        for cmd in (r.get("actions") or [])[:16]:
            _receive(cmd, c)
        if r.get("next_report_s"):
            _fast_until[0] = max(_fast_until[0], time.time() + 120)
    return dict(_last)


def status():
    c = config()
    out = dict(role=c["role"], name=c["name"], box_id=c["box_id"], primary_url=c.get("primary_url", ""),
               share=bool(c.get("share")), actions=list(ACTIONS), set_keys=sorted(SET_KEYS))
    if c["role"] == "primary":
        out["boxes"] = boxes()
    if c["role"] == "member":
        pol, key = policy(), _key()
        out.update(joined=(_d() / "token").exists(), last_send=dict(_last) or None,
                   remote=dict(pol, key_fp=_fp(key), running=_rj(_d() / "running.json", None),
                               queued=_jobs.qsize(), log=action_log()))
    return out


def worker():
    _recover_interrupted()
    _start("executor", _executor)
    while True:
        try:
            if config()["role"] == "member" and (_d() / "token").exists():
                send_now()
        except Exception as e:
            _last.update(at=int(time.time()), ok=False, error=str(e)[:200])
        _wake.clear()
        fast = time.time() < _fast_until[0] or (_d() / "running.json").exists() or _jobs.qsize() \
            or _rj(_d() / "outbox.json", [])
        _wake.wait(FAST_S if fast else REPORT_S + secrets.randbelow(21) - 10)
        gap = time.time() - (_last.get("at") or 0)
        if gap < FAST_S:                          # the primary refuses reports closer than MIN_GAP_S
            time.sleep(FAST_S - gap)
