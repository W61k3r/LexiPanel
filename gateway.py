#!/usr/bin/env python3
"""
Gateway (1.0.0): one OpenAI-compatible endpoint in front of every running
instance, with per-user quotas and usage. On the panel itself, behind the same login / API keys:
  GET  /v1/models              the running instances, by name
  POST /v1/chat/completions    routed by "model" (instance id, alias or model file name; with one
                               instance running it may be omitted); streaming passes through
Quotas per user (gateway/quotas.json, admin: POST /api/gateway/quota): rpm (requests per minute),
tokens_day, concurrent, models (allowed names). Over a limit: HTTP 429 with the reason. Usage per
day, user and model in gateway/usage.json (60 days): requests, prompt and generated tokens. Counts
only: prompts and replies are passed through, never stored.
"""
import http.client, json, os, threading, time
from pathlib import Path

P = None
_lock = threading.Lock()
_recent, _inflight = {}, {}          # user -> [request times]; user -> open requests


def bind(panel_module):
    global P
    P = panel_module


def _d():
    d = P.PANEL / "gateway"
    d.mkdir(exist_ok=True)
    return d


def _rj(n, dflt):
    try:
        return json.loads((_d() / n).read_text())
    except (OSError, ValueError):
        return dflt


def _wj(n, obj):
    t = _d() / (n + ".tmp")
    t.write_text(json.dumps(obj, indent=1))
    os.replace(t, _d() / n)


def targets():
    """name -> [(host, port, key, where)]: this box's running OpenAI-compatible instances, and on
    a fleet primary the instances online members share (fleet.py, share on)."""
    out = {}
    for iid in P.instance_ids():
        try:
            inst = P.get_instance(iid)
            eng = inst.get("engine") or "llama.cpp"
            if eng not in ("llama.cpp", "onnx", "camelid"):
                continue
            with P.using_instance(inst):
                if not P.server_pid():
                    continue
                argv, v = P.live_cmdline_args(), P.load_params()
            g = lambda *f: P._argv_get(argv, f)
            port = g("--port") or (g("--addr") or ":").rpartition(":")[2] or v.get("PORT")
            key = v.get("API_KEY") or v.get("OX_API_KEY") or v.get("CM_API_KEY") or None
            model = g("-m", "--model") or g("--model-dir") or ""
            t = ("127.0.0.1", int(port), key, iid)
            for n in {iid, g("-a", "--alias"), os.path.basename(str(model).rstrip("/"))}:
                if n:
                    out.setdefault(n, []).append(t)
        except Exception:
            continue
    fl = getattr(P, "fleet", None)
    if fl is not None:
        try:
            if fl.config()["role"] == "primary":
                for b in fl.boxes():
                    if b["state"] != "online":
                        continue
                    for s in (b.get("report") or {}).get("shared") or []:
                        t = (str(s.get("addr")), int(s.get("port")), None, f"{b.get('name')}/{s.get('instance')}")
                        for n in s.get("names") or []:
                            out.setdefault(str(n), []).append(t)
        except Exception:
            pass
    return out


_busy = {}                           # (host, port) -> open requests through the gateway


def choose(reps, down=()):
    """The replica with the fewest open requests, skipping ones that just failed."""
    ok = [r for r in reps if (r[0], r[1]) not in down]
    return min(ok, key=lambda r: _busy.get((r[0], r[1]), 0)) if ok else None


def models():
    data = []
    for n, reps in targets().items():
        data.append(dict(id=n, object="model", owned_by="lexipanel", replicas=[r[3] for r in reps]))
    return dict(object="list", data=data)


def quota_of(user):
    return (_rj("quotas.json", {}) or {}).get(user or "local", {})


def set_quota(body):
    user = str(body.get("user") or "").strip()
    if not user:
        raise ValueError("user")
    q = {}
    for k in ("rpm", "tokens_day", "concurrent"):
        if body.get(k) not in (None, ""):
            v = int(body[k])
            if v < 0:
                raise ValueError(f"{k} must be 0 or more (0 = no limit)")
            if v:
                q[k] = v
    if body.get("models"):
        q["models"] = [str(m)[:128] for m in body["models"]][:64]
    with _lock:
        allq = _rj("quotas.json", {})
        allq[user] = q
        _wj("quotas.json", allq)
    return dict(user=user, quota=q)


def _day():
    return time.strftime("%Y-%m-%d", time.gmtime())


def admit(user, model, now=None):
    """None if the request may go ahead (and it is counted), else why not."""
    now, user = now or time.time(), user or "local"
    q = quota_of(user)
    if q.get("models") and model not in q["models"]:
        return f"model {model!r} is not in your allowed models"
    with _lock:
        r = [t for t in _recent.get(user, []) if now - t < 60]
        if q.get("rpm") and len(r) >= q["rpm"]:
            return f"rate limit: {q['rpm']} requests per minute"
        if q.get("concurrent") and _inflight.get(user, 0) >= q["concurrent"]:
            return f"concurrency limit: {q['concurrent']} at once"
        used = sum(sum(m.get("prompt", 0) + m.get("completion", 0) for m in u.values())
                   for d, du in _rj("usage.json", {}).items() if d == _day() for uu, u in du.items() if uu == user)
        if q.get("tokens_day") and used >= q["tokens_day"]:
            return f"daily token limit: {q['tokens_day']} (used {used}); resets at 00:00 UTC"
        r.append(now)
        _recent[user] = r
        _inflight[user] = _inflight.get(user, 0) + 1
    return None


def record(user, model, prompt, completion):
    user = user or "local"
    with _lock:
        _inflight[user] = max(0, _inflight.get(user, 1) - 1)
        u = _rj("usage.json", {})
        m = u.setdefault(_day(), {}).setdefault(user, {}).setdefault(model, dict(requests=0, prompt=0, completion=0))
        m["requests"] += 1
        m["prompt"] += int(prompt or 0)
        m["completion"] += int(completion or 0)
        for d in sorted(u)[:-60]:
            del u[d]
        _wj("usage.json", u)


def status():
    return dict(models=[m["id"] for m in models()["data"]], quotas=_rj("quotas.json", {}), usage=_rj("usage.json", {}))


def _err(h, code, msg, kind="invalid_request_error"):
    b = json.dumps(dict(error=dict(message=msg, type=kind))).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b)))
    h.end_headers()
    h.wfile.write(b)


def forward(h, host, port, key, body, user, model):
    """Pass one chat completion through to an instance; count its tokens."""
    c = http.client.HTTPConnection(host, port, timeout=3600)
    try:
        c.connect()
    except OSError:
        c.close()
        return False                                   # nothing sent yet: the caller tries another replica
    _busy[(host, port)] = _busy.get((host, port), 0) + 1
    hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    prompt = completion = 0
    try:
        c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=hdr)
        r = c.getresponse()
        h.send_response(r.status)
        h.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
        if not body.get("stream"):
            data = r.read()
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            h.wfile.write(data)
            try:
                u = json.loads(data).get("usage") or {}
                prompt, completion = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            except ValueError:
                pass
        else:
            h.send_header("Cache-Control", "no-cache")
            h.end_headers()
            h.close_connection = True
            for line in r:
                h.wfile.write(line)
                h.wfile.flush()
                if line.startswith(b"data: {"):
                    try:
                        o = json.loads(line[6:])
                    except ValueError:
                        continue
                    if any((ch.get("delta") or {}).get("content") for ch in o.get("choices") or []):
                        completion += 1                      # one chunk ~ one token
                    t, u = o.get("timings") or {}, o.get("usage") or {}
                    if t or u:
                        prompt = u.get("prompt_tokens") or t.get("prompt_n") or prompt
                        completion = u.get("completion_tokens") or t.get("predicted_n") or completion
    except (OSError, http.client.HTTPException) as e:
        try:
            _err(h, 502, f"instance did not answer: {e}", "upstream_error")
        except OSError:
            pass
    finally:
        c.close()
        _busy[(host, port)] = max(0, _busy.get((host, port), 1) - 1)
        record(user, model, prompt, completion)
    return True


def handle_post(h, p, user):
    why = h._foreign("POST")
    if why:
        return _err(h, 403, why)
    if p != "/v1/chat/completions":
        return _err(h, 404, "only /v1/chat/completions and /v1/models")
    try:
        body = json.loads(h.rfile.read(int(h.headers.get("Content-Length") or 0)) or b"{}")
        assert isinstance(body, dict)
    except (ValueError, AssertionError):
        return _err(h, 400, "body must be a JSON object")
    ts = targets()
    where = {r[3] for reps in ts.values() for r in reps}
    name = body.get("model") or (next(iter(ts)) if len(where) == 1 else None)
    if not name or name not in ts:
        return _err(h, 404, f"no running instance serves {name!r}; running: {', '.join(sorted(ts)) or 'none'}")
    no = admit(user, name)
    if no:
        return _err(h, 429, no, "rate_limit_error")
    down = set()
    while True:
        r = choose(ts[name], down)
        if r is None:
            record(user, name, 0, 0)
            return _err(h, 503, f"no replica of {name!r} answered", "upstream_error")
        if forward(h, r[0], r[1], r[2], body, user, name):
            return
        down.add((r[0], r[1]))
