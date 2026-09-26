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
import http.client, json, os, re, threading, time
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
    """name -> [(host, port, key, where, scheme, upstream)]: this box's running OpenAI-compatible
    instances, and on a fleet primary the instances online members share (fleet.py, share on) -
    minus drained boxes and member instances in a planned restart. `upstream`: the model name the
    server itself expects (vLLM refuses any other), or None when any name will do."""
    out = {}
    for iid in P.instance_ids():
        try:
            inst = P.get_instance(iid)
            eng = inst.get("engine") or "llama.cpp"
            if eng not in ("llama.cpp", "onnx", "camelid", "vllm"):
                continue
            with P.using_instance(inst):
                if not P.server_pid():
                    continue
                argv, v = P.live_cmdline_args(), P.load_params()
            g = lambda *f: P._argv_get(argv, f)
            port = g("--port") or (g("--addr") or ":").rpartition(":")[2] or v.get("PORT")
            key = v.get("API_KEY") or v.get("OX_API_KEY") or v.get("CM_API_KEY") or None
            if eng == "vllm":                         # load_params masks it; the file has it
                key = P.vllm_engine.api_key(inst)
            model = g("-m", "--model") or g("--model-dir") or ""
            if eng == "vllm":
                port = port or v.get("PORT")
                model = v.get("VL_SERVED_NAME") or v.get("VL_MODEL") or model
            t = ("127.0.0.1", int(port), key, iid, "http",
                 P.vllm_engine.served_name(v) if eng == "vllm" else None)
            try:                                   # slots it serves at once, for admission
                n = int(v.get("VL_MAX_NUM_SEQS") if eng == "vllm" else
                        (g("-np", "--parallel") or v.get("PARALLEL") or 1))
                _caps[("127.0.0.1", int(port))] = max(1, n)
                if eng == "llama.cpp":             # a shared KV pool: admit by the tokens it can hold
                    c = int(g("-c", "--ctx-size") or v.get("CTX") or 0)
                    uni = "--kv-unified" in argv or "-kvu" in argv
                    _pools[("127.0.0.1", int(port))] = int(c / getattr(P, "KV_POOL_HEADROOM", 1.7)) \
                        if (uni and n > 1 and c) else None
            except (TypeError, ValueError):
                pass
            for n in {iid, g("-a", "--alias"), os.path.basename(str(model).rstrip("/"))}:
                if n:
                    out.setdefault(n, []).append(t)
        except Exception:
            continue
    rm = getattr(P, "remotes", None)                    # endpoints registered by address (remotes.py)
    if rm is not None:
        try:
            for n, ts in rm.gateway_targets().items():
                out.setdefault(n, []).extend(ts)
        except Exception:
            pass
    fl = getattr(P, "fleet", None)
    if fl is not None:
        try:
            if fl.config()["role"] == "primary":
                held = held_keys()
                for b in fl.boxes():
                    if b["state"] != "online" or b.get("drain"):
                        continue
                    for s in (b.get("report") or {}).get("shared") or []:
                        if f"box:{b.get('box_id')}/{s.get('instance')}" in held:
                            continue                   # restarting on purpose: another replica, or wait
                        t = (str(s.get("addr")), int(s.get("port")), None, f"{b.get('name')}/{s.get('instance')}")
                        for n in s.get("names") or []:
                            out.setdefault(str(n), []).append(t)
        except Exception:
            pass
    return out


_busy = {}                           # (host, port) -> open requests through the gateway
_caps = {}                           # (host, port) -> slots a server has, noted by targets()
_pools = {}                          # (host, port) -> KV tokens a shared llama.cpp pool holds at once without failing
DEFAULT_REPLY_TOKENS = 1024          # a request without max_tokens is planned at this many reply tokens


def need_tokens(body):
    """KV cells a chat completion holds at its deepest: the prompt (~3.2 characters a token,
    erring high) plus the longest reply it may write."""
    chars, msgs = 0, body.get("messages") or []
    for m in msgs if isinstance(msgs, list) else []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            chars += sum(len(str(p.get("text") or "")) for p in c if isinstance(p, dict))
    if body.get("tools"):
        chars += len(json.dumps(body["tools"]))
    try:
        out = int(body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or 0)
    except (TypeError, ValueError):
        out = 0
    return int(chars / 3.2) + 8 * len(msgs if isinstance(msgs, list) else []) + (out if out > 0 else DEFAULT_REPLY_TOKENS)

# ---------------------------------------------------------------------------
# admission: a server is never sent more requests at once than it has slots, nor - for a
# llama.cpp server whose slots share one KV pool - more tokens than the pool holds without
# failing requests (panel.KV_POOL_HEADROOM; vLLM pages its cache and needs no such limit). The
# rest wait here, interactive before batch, first-come within a class; a request alone on a
# server always goes (it may use the whole pool).
# ---------------------------------------------------------------------------
PRIO = {"interactive": 0, "batch": 1}
QUEUE_TIMEOUT = {"interactive": 300, "batch": 3600}
_adm = {}                            # (host, port) -> dict(busy, queue, seq, served, waited_s, timeouts)
_adm_cv = threading.Condition()
_cap_cache = {}                      # (host, port) -> (at, cap)


def capacity(host, port):
    """Slots a server serves at once: noted by targets() for local instances, else llama.cpp's
    /slots; None when unknown (no limit is imposed then)."""
    key = (host, int(port))
    if key in _caps:
        return _caps[key]
    hit = _cap_cache.get(key)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    cap = None
    try:
        c = http.client.HTTPConnection(host, int(port), timeout=2)
        c.request("GET", "/slots")
        r = c.getresponse()
        data = r.read(1024 * 1024)
        c.close()
        if r.status == 200:
            slots = json.loads(data)
            cap = len(slots) if isinstance(slots, list) and slots else None
    except (OSError, ValueError, http.client.HTTPException):
        cap = None
    _cap_cache[key] = (time.time(), cap)
    return cap


def admit_slot(host, port, prio="interactive", timeout=None, need=0):
    """Wait for a free slot on this server (and, on a shared KV pool, room for `need` tokens);
    interactive requests go first. Returns seconds waited; raises TimeoutError when the wait
    runs out."""
    key = (host, int(port))
    cap = capacity(host, port)
    rank = PRIO.get(prio, 0)
    timeout = QUEUE_TIMEOUT.get(prio, 300) if timeout is None else timeout
    with _adm_cv:
        a = _adm.setdefault(key, dict(busy=0, queue=[], seq=0, served=0, waited_s=0.0, timeouts=0, tokens=0,
                                      waited_kv=0))
        if cap is None:
            a["busy"] += 1
            a["served"] += 1
            return 0.0
        a["seq"] += 1
        me = (rank, a["seq"])
        a["queue"].append(me)
        a["queue"].sort()
        t0, kv_wait = time.time(), False
        while True:
            pool = _pools.get(key)
            room = pool is None or a["busy"] == 0 or a["tokens"] + need <= pool
            if a["busy"] < cap and a["queue"][0] == me and not room:
                kv_wait = True                     # a free slot, but the pool would overfill
            if a["busy"] < cap and a["queue"][0] == me and room:
                a["queue"].pop(0)
                a["busy"] += 1
                a["tokens"] += need
                a["served"] += 1
                a["waited_kv"] += kv_wait
                w = time.time() - t0
                a["waited_s"] += w
                _adm_cv.notify_all()
                return w
            left = timeout - (time.time() - t0)
            if left <= 0:
                a["queue"].remove(me)
                a["timeouts"] += 1
                _adm_cv.notify_all()
                raise TimeoutError(f"no free slot on {host}:{port} after {int(timeout)} s")
            _adm_cv.wait(min(left, 1.0))


def release_slot(host, port, need=0):
    with _adm_cv:
        a = _adm.get((host, int(port)))
        if a:
            a["busy"] = max(0, a["busy"] - 1)
            a["tokens"] = max(0, a.get("tokens", 0) - need)
        _adm_cv.notify_all()


def admission():
    with _adm_cv:
        return [dict(server=f"{h}:{p}", capacity=_caps.get((h, p)) or (_cap_cache.get((h, p)) or (0, None))[1],
                     busy=a["busy"], waiting=len(a["queue"]),
                     waiting_batch=sum(1 for r, _s in a["queue"] if r >= PRIO["batch"]),
                     served=a["served"], mean_wait_s=round(a["waited_s"] / a["served"], 3) if a["served"] else None,
                     timeouts=a["timeouts"], kv_pool=_pools.get((h, p)), kv_tokens=a.get("tokens", 0),
                     waited_for_kv=a.get("waited_kv", 0)) for (h, p), a in _adm.items()]
_holds = {}                          # instance id (or "box:<box_id>/<instance>") -> hold while it restarts on purpose


def hold(iid, seconds, names=None, replica=None):
    """Hold new requests for `iid` (by every name it answers to) for up to `seconds`, instead of
    failing them while it restarts (restarts.py). A fleet member's instance (`replica`, its
    host and port; fleet.event): its requests go to another replica of the name meanwhile, and
    wait only when there is none."""
    if names is None:
        names = {n for n, reps in targets().items() for r in reps if r[3] == iid} | {iid}
    with _lock:
        _holds[iid] = dict(names=set(names), until=time.time() + max(0, seconds), since=time.time(), held=0,
                           expired=0, replica=replica)


def release(iid):
    with _lock:
        h = _holds.pop(iid, None)
    return h and {k: v for k, v in h.items() if k not in ("names", "replica")}


def held_keys():
    """Fleet members' instances in a planned restart right now."""
    now = time.time()
    with _lock:
        return {k for k, h in _holds.items() if h.get("replica") and now < h["until"]}


def holds():
    now = time.time()
    with _lock:
        return [dict(key=k, names=sorted(h["names"]), since=int(h["since"]), left_s=max(0, int(h["until"] - now)),
                     held=h["held"], expired=h["expired"]) for k, h in _holds.items()]


def wait_if_held(name, poll=0.25):
    """Block while the instance behind `name` is in a planned restart. Returns None (no hold),
    "released" or "expired"."""
    now = time.time()
    with _lock:
        for i in [i for i, h in _holds.items() if h.get("replica") and now > h["until"] + 600]:
            _holds.pop(i)                              # a member that never said "release"
        hit = [(i, h) for i, h in _holds.items() if now < h["until"] and
               (name in h["names"] or (name is None and len(_holds) == 1 and not h.get("replica")))]
    if not hit:
        return None
    if all(h.get("replica") for _i, h in hit) and name in targets():
        return None                                    # a member restarts; another replica serves meanwhile
    iid, h = hit[0]
    with _lock:
        h["held"] += 1
    while True:
        with _lock:
            if iid not in _holds:
                return "released"
            if time.time() >= h["until"]:
                h["expired"] += 1
                return "expired"
        time.sleep(poll)


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


def call(host, port, key, body, scheme="http", timeout=3600, upstream=None):
    """One non-streaming chat completion, answered to the caller (the batch worker) rather than
    to an HTTP client: (status, parsed body)."""
    import ssl
    if upstream:
        body = dict(body, model=upstream)
    c = (http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
         if scheme == "https" else http.client.HTTPConnection(host, port, timeout=timeout))
    hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    try:
        c.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=False)), headers=hdr)
        r = c.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, dict(error=dict(message=data[:500].decode(errors="ignore")))
    finally:
        c.close()


def status():
    return dict(models=[m["id"] for m in models()["data"]], quotas=_rj("quotas.json", {}), usage=_rj("usage.json", {}),
                admission=admission(), holds=holds())


def _err(h, code, msg, kind="invalid_request_error"):
    b = json.dumps(dict(error=dict(message=msg, type=kind))).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b)))
    h.end_headers()
    h.wfile.write(b)


def forward(h, host, port, key, body, user, model, scheme="http", upstream=None):
    """Pass one chat completion through to an instance; count its tokens."""
    if upstream:
        body = dict(body, model=upstream)
    if scheme == "https":
        import ssl
        c = http.client.HTTPSConnection(host, port, timeout=3600, context=ssl.create_default_context())
    else:
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


def _send_json(h, code, obj):
    b = json.dumps(obj).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b)))
    h.end_headers()
    h.wfile.write(b)


def _batches_post(h, p, user):
    B = P.batches
    try:
        body = json.loads(h.rfile.read(int(h.headers.get("Content-Length") or 0)) or b"{}")
        assert isinstance(body, dict)
    except (ValueError, AssertionError):
        return _err(h, 400, "body must be a JSON object")
    try:
        if p == "/v1/batches":
            return _send_json(h, 200, B.create(user, body))
        m = re.fullmatch(r"/v1/batches/(batch_[0-9a-f]{24})/(cancel|delete)", p)
        if not m:
            return _err(h, 404, "POST /v1/batches, /v1/batches/<id>/cancel or /v1/batches/<id>/delete")
        return _send_json(h, 200, (B.cancel if m.group(2) == "cancel" else B.delete)(user, m.group(1)))
    except KeyError as e:
        return _err(h, 404, str(e).strip("'\""))
    except ValueError as e:
        return _err(h, 400, str(e))


def handle_get_batches(h, p, user):
    """GET /v1/batches, /v1/batches/<id>, /v1/batches/<id>/output (JSONL)."""
    B = P.batches
    try:
        if p == "/v1/batches":
            return _send_json(h, 200, B.listing(user))
        m = re.fullmatch(r"/v1/batches/(batch_[0-9a-f]{24})(/output)?", p)
        if not m:
            return _err(h, 404, "no such batch route")
        if m.group(2):
            b = B.output(user, m.group(1)).encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/jsonl")
            h.send_header("Content-Length", str(len(b)))
            h.end_headers()
            h.wfile.write(b)
            return
        return _send_json(h, 200, B.get(user, m.group(1)))
    except KeyError as e:
        return _err(h, 404, str(e).strip("'\""))
    except ValueError as e:
        return _err(h, 400, str(e))


def handle_post(h, p, user):
    why = h._foreign("POST")
    if why:
        return _err(h, 403, why)
    if p == "/v1/batches" or p.startswith("/v1/batches/"):
        return _batches_post(h, p, user)
    if p != "/v1/chat/completions":
        return _err(h, 404, "only /v1/chat/completions, /v1/batches and /v1/models")
    try:
        body = json.loads(h.rfile.read(int(h.headers.get("Content-Length") or 0)) or b"{}")
        assert isinstance(body, dict)
    except (ValueError, AssertionError):
        return _err(h, 400, "body must be a JSON object")
    wait_if_held(body.get("model"))                   # a planned restart: wait instead of failing
    ts = targets()
    where = {r[3] for reps in ts.values() for r in reps}
    name = body.get("model") or (next(iter(ts)) if len(where) == 1 else None)
    if not name or name not in ts:
        return _err(h, 404, f"no running instance serves {name!r}; running: {', '.join(sorted(ts)) or 'none'}")
    no = admit(user, name)
    if no:
        return _err(h, 429, no, "rate_limit_error")
    prio = str(h.headers.get("X-LexiPanel-Priority") or body.pop("priority", None) or "interactive").lower()
    body.pop("priority", None)                         # a LexiPanel field: strict servers would refuse it
    if prio not in PRIO:
        return _err(h, 400, f"priority: one of {', '.join(PRIO)}")
    down, need = set(), need_tokens(body)
    while True:
        r = choose(ts[name], down)
        if r is None:
            record(user, name, 0, 0)
            return _err(h, 503, f"no replica of {name!r} answered", "upstream_error")
        try:
            admit_slot(r[0], r[1], prio, need=need)
        except TimeoutError as e:
            record(user, name, 0, 0)
            return _err(h, 503, str(e), "upstream_error")
        try:
            sent = forward(h, r[0], r[1], r[2], body, user, name, r[4] if len(r) > 4 else "http",
                           r[5] if len(r) > 5 else None)
        finally:
            release_slot(r[0], r[1], need)
        if sent:
            return
        down.add((r[0], r[1]))
