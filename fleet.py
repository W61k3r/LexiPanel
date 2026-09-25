#!/usr/bin/env python3
"""
Fleet (1.0.0): full LexiPanel on every box, one box the primary (docs/FLEET.md).
Roles in fleet/config.json: standalone (default) | primary | member.
  member   POSTs a report to the primary every 60 s (outbound only), token in fleet/token (0600)
  primary  issues one-time join codes, accepts reports, lists boxes, revokes them
View-only: nothing here lets the primary change a member. Reports carry what the panel already
shows (hardware, instances, speeds, workload counts, alerts), never prompts, settings or keys.
"""
import hashlib, json, os, platform, re, secrets, threading, time, urllib.parse, urllib.request, uuid
from pathlib import Path

P = None
REPORT_S, STALE_N, OFFLINE_S = 60, 3, 600
MAX_BYTES, MIN_GAP_S, JOIN_TTL = 256 * 1024, 20, 1800
_lock = threading.RLock()
_last = {}                        # member: last send result


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
    tmp.write_text(json.dumps(obj, indent=1))
    os.chmod(tmp, mode)
    os.replace(tmp, f)


def _h(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def config():
    c = _rj(_d() / "config.json", {})
    c.setdefault("role", "standalone")
    c.setdefault("name", platform.node())
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


def intake(headers, raw):
    if config()["role"] != "primary":
        raise PermissionError("this box is not a fleet primary")
    if len(raw) > MAX_BYTES:
        raise ValueError("report too large")
    tok = (headers.get("Authorization") or "").removeprefix("Bearer ").strip()
    body = json.loads(raw or b"{}")
    box_id = str(body.get("box_id") or "")
    f = _d() / "boxes" / f"{box_id}.json"
    if not re.fullmatch(r"[0-9a-f-]{36}", box_id) or not f.exists():
        raise PermissionError("unknown box: join first")
    with _lock:
        rec = _rj(f, {})
        if not tok or not secrets.compare_digest(rec.get("token", ""), _h(tok)):
            raise PermissionError("bad token (revoked?)")
        now = time.time()
        if rec.get("last") and now - rec["last"] < MIN_GAP_S:
            raise ValueError("too many reports")
        rec.update(last=int(now), report=_clean(body), name=_cap(body.get("name") or rec.get("name"), 64))
        _wj(f, rec)
    return dict(ok=True)


def revoke(box_id):
    f = _d() / "boxes" / f"{box_id}.json"
    if not re.fullmatch(r"[0-9a-f-]{36}", str(box_id)) or not f.exists():
        raise ValueError("no such box")
    f.unlink()
    return dict(ok=True)


def boxes(now=None):
    now = now or time.time()
    out = []
    for f in sorted((_d() / "boxes").glob("*.json")):
        r = _rj(f, {})
        age = now - r["last"] if r.get("last") else None
        state = ("waiting for its first report" if age is None else "online" if age < STALE_N * REPORT_S
                 else "stale" if age < OFFLINE_S else "offline")
        out.append(dict(box_id=r.get("box_id"), name=r.get("name"), joined=r.get("joined"), last=r.get("last"),
                        age_s=int(age) if age is not None else None, state=state, report=r.get("report")))
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
    f.write_text(r["token"])
    os.chmod(f, 0o600)


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
            model = p.get("MODEL") or p.get("CM_MODEL") or p.get("OX_MODEL_DIR") or ""
            rec = dict(id=iid, name=inst.get("name"), engine=eng, state="running" if pid else "stopped",
                       model=os.path.basename(str(model).rstrip("/")), backend=p.get("BACKEND"),
                       ctx=p.get("CTX"), port=p.get("PORT"))
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
        for iid in P.instance_ids():
            try:
                inst = P.get_instance(iid)
                if (inst.get("engine") or "llama.cpp") not in ("llama.cpp", "onnx", "camelid"):
                    continue
                with P.using_instance(inst):
                    if not P.server_pid():
                        continue
                    argv, v = P.live_cmdline_args(), P.load_params()
                g = lambda *f: P._argv_get(argv, f)
                host = g("--host") or (g("--addr") or "").rpartition(":")[0] or "127.0.0.1"
                if host in ("127.0.0.1", "localhost", "::1") or v.get("API_KEY") or v.get("OX_API_KEY") or v.get("CM_API_KEY"):
                    continue
                port = g("--port") or (g("--addr") or ":").rpartition(":")[2]
                model = g("-m", "--model") or g("--model-dir") or ""
                names = sorted({n for n in (g("-a", "--alias"), os.path.basename(str(model).rstrip("/"))) if n})
                if addr and port:
                    shared.append(dict(instance=iid, addr=addr, port=int(port), names=names))
            except Exception:
                continue
    npus = []
    try:
        npus = [dict(vendor=n.get("vendor"), driver=n.get("driver")) for n in P.onnxrt.npus()]
    except Exception:
        pass
    return dict(v=1, box_id=c["box_id"], name=c["name"], sent=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                lexipanel=ver, os=platform.system() + " " + platform.release(),
                hardware=dict(ram_mib=P._meminfo_mb("MemTotal"), gpus=gpus, npus=npus), instances=insts,
                shared=shared)


def send_now():
    c = config()
    if c["role"] != "member":
        raise ValueError("this box is not a fleet member")
    try:
        token = (_d() / "token").read_text().strip()
    except OSError:
        raise ValueError("not joined yet: paste a join code from the primary")
    st, r = _post(c["primary_url"] + "/api/fleet/report", build_report(), token)
    _last.update(at=int(time.time()), ok=st == 200, status=st, error=None if st == 200 else r.get("error"))
    return dict(_last)


def status():
    c = config()
    out = dict(role=c["role"], name=c["name"], box_id=c["box_id"], primary_url=c.get("primary_url", ""),
               share=bool(c.get("share")))
    if c["role"] == "primary":
        out["boxes"] = boxes()
    if c["role"] == "member":
        out.update(joined=(_d() / "token").exists(), last_send=dict(_last) or None)
    return out


def worker():
    while True:
        try:
            if config()["role"] == "member" and (_d() / "token").exists():
                send_now()
        except Exception as e:
            _last.update(at=int(time.time()), ok=False, error=str(e)[:200])
        time.sleep(REPORT_S + secrets.randbelow(21) - 10)
