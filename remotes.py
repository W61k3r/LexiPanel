#!/usr/bin/env python3
"""
Remote endpoints (added 2026-09-25): model servers this panel does not run - another box's
llama-server, vLLM, Ollama, LM Studio or any OpenAI-compatible endpoint - registered by
address, so they appear in the server list next to the local ones and, if chosen, answer
through the gateway.

Local servers are found by process (pgrep and /proc), which can never see another machine.
Remote ones are found by address and asked, every few seconds, what they are:
  /health        llama.cpp and vLLM
  /v1/models     any OpenAI-compatible server: the model names
  /props         llama.cpp: context size, model file, build
  /slots         llama.cpp: slots and how many are busy
  /metrics       llama.cpp (--metrics) and vLLM: requests running and waiting, generation rate

Security. Registering and removing are admin actions (the default for a POST). An address is
http(s)://host:port and nothing else: no user info, path, query or fragment. Link-local
(including cloud metadata at 169.254.169.254), multicast, unspecified and reserved addresses
are refused, and the host is resolved and checked again at every poll, because DNS can
change under a name. No redirect is followed, an answer is read up to 2 MB, every call times
out in 3 s. An endpoint's API key is kept in remotes-keys.json (0600), sent only to that
endpoint, and never returned by the API.
"""
import http.client, ipaddress, json, os, re, socket, ssl, threading, time
from pathlib import Path
from urllib.parse import urlsplit

P = None
_lock = threading.RLock()
_cache = {}                      # name -> (at, probe)
CACHE_S = 5
TIMEOUT = 3
MAX_BYTES = 2 * 1024 * 1024
KINDS = ("auto", "llama.cpp", "vllm", "openai")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")


def bind(panel_module):
    global P
    P = panel_module


def _file():
    return P.PANEL / "remotes.json"


def _keys_file():
    return P.PANEL / "remotes-keys.json"


def _read(p, dflt):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return dflt


def _write(p, obj, mode=0o644):
    tmp = Path(str(p) + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(obj, indent=1))
    os.chmod(tmp, mode)
    os.replace(tmp, p)


def load():
    return _read(_file(), {})


def _key(name):
    return (_read(_keys_file(), {}) or {}).get(name)


# ============================================================================
# the address and what it may point at
# ============================================================================
def check_url(url):
    """(scheme, host, port) of an acceptable endpoint address, or ValueError."""
    u = urlsplit(str(url or "").strip())
    if u.scheme not in ("http", "https"):
        raise ValueError("the address must start with http:// or https://")
    if u.username or u.password:
        raise ValueError("no user:password in the address; put an API key in its own field")
    if u.path not in ("", "/") or u.query or u.fragment:
        raise ValueError("the address is http(s)://host:port only (no path, query or fragment)")
    try:
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        raise ValueError("bad port")
    host = u.hostname
    if not host:
        raise ValueError("no host in the address")
    _check_host(host)
    return u.scheme, host, port


def _check_host(host):
    """Resolve and refuse addresses a model server never lives on (metadata, link-local...)."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {host}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved or \
                ip == ipaddress.ip_address("fd00:ec2::254"):                      # AWS IPv6 metadata
            raise ValueError(f"{host} resolves to {ip}, which is not allowed (link-local, metadata, "
                             "multicast, unspecified or reserved)")


# ============================================================================
# asking an endpoint
# ============================================================================
def _get(r, path):
    """(status, body) from GET path on a registered endpoint; body is JSON when it parses."""
    scheme, host, port = r["scheme"], r["host"], r["port"]
    _check_host(host)                                    # every time: DNS can change
    if scheme == "https":
        c = http.client.HTTPSConnection(host, port, timeout=TIMEOUT, context=ssl.create_default_context())
    else:
        c = http.client.HTTPConnection(host, port, timeout=TIMEOUT)
    hdr = {"Accept": "application/json"}
    key = r["_key"] if "_key" in r else _key(r["name"])      # a test passes its key in memory only
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    try:
        c.request("GET", path, headers=hdr)
        resp = c.getresponse()
        data = resp.read(MAX_BYTES + 1)[:MAX_BYTES]     # never follows a redirect
        try:
            body = json.loads(data)
        except ValueError:
            body = data.decode(errors="ignore")
        return resp.status, body
    finally:
        c.close()


def _metric(text, name):
    m = re.search(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)", text or "", re.M)
    return float(m.group(1)) if m else None


def probe(r):
    """What the endpoint says about itself, right now."""
    out = dict(health=None, kind=r.get("kind", "auto"), models=[], ctx=None, model=None, slots=None,
               slots_busy=None, running=None, waiting=None, live_tps=None, error=None, build=None)
    try:
        st, h = _get(r, "/health")
        out["health"] = "ok" if st == 200 else ("loading" if st == 503 else f"HTTP {st}")
    except (OSError, http.client.HTTPException, ValueError) as e:
        out.update(health="no answer", error=str(e)[:200])
        return out
    for path in ("/v1/models", "/props", "/slots", "/metrics"):
        try:
            st, body = _get(r, path)
        except (OSError, http.client.HTTPException, ValueError):
            continue
        if st != 200:
            continue
        if path == "/v1/models" and isinstance(body, dict):
            out["models"] = [str(m.get("id")) for m in body.get("data") or [] if isinstance(m, dict)][:20]
        elif path == "/props" and isinstance(body, dict):
            g = body.get("default_generation_settings") or {}
            out["ctx"] = g.get("n_ctx") or body.get("n_ctx")
            out["model"] = body.get("model_path") or body.get("model_alias")
            out["build"] = body.get("build_info")
            if out["kind"] == "auto":
                out["kind"] = "llama.cpp"
        elif path == "/slots" and isinstance(body, list):
            out["slots"] = len(body)
            out["slots_busy"] = sum(1 for s in body if isinstance(s, dict) and s.get("is_processing"))
        elif path == "/metrics" and isinstance(body, str):
            if "vllm:" in body:
                out["running"] = _metric(body, "vllm:num_requests_running")
                out["waiting"] = _metric(body, "vllm:num_requests_waiting")
                if out["kind"] == "auto":
                    out["kind"] = "vllm"
            else:
                out["running"] = _metric(body, "llamacpp:requests_processing")
                out["waiting"] = _metric(body, "llamacpp:requests_deferred")
                out["live_tps"] = _metric(body, "llamacpp:predicted_tokens_seconds")
    if out["kind"] == "auto":
        out["kind"] = "openai" if out["models"] else "unknown"
    return out


_refreshing = set()


def _refresh(r):
    try:
        p = probe(r)
    except Exception as e:
        p = dict(health="no answer", error=str(e)[:200], models=[], kind=r.get("kind"))
    with _lock:
        _cache[r["name"]] = (time.time(), p)
        _refreshing.discard(r["name"])
    return p


def _cached_probe(r, block=True):
    """The endpoint's last answer. block=False (the status poll) never waits on the network:
    a stale entry is refreshed in the background and the last known state returned."""
    with _lock:
        hit = _cache.get(r["name"])
        fresh = hit and time.time() - hit[0] < CACHE_S
        if fresh:
            return hit[1]
        if not block:
            if r["name"] not in _refreshing:
                _refreshing.add(r["name"])
                threading.Thread(target=_refresh, args=(dict(r),), daemon=True).start()
            return hit[1] if hit else dict(health="checking", models=[], kind=r.get("kind"))
    return _refresh(r)


# ============================================================================
# registry
# ============================================================================
def add(body):
    name = str(body.get("name") or "").strip()
    if not NAME_RE.fullmatch(name):
        raise ValueError("name: 1-40 letters, digits, '.', '_' or '-'")
    kind = str(body.get("kind") or "auto")
    if kind not in KINDS:
        raise ValueError(f"kind: one of {', '.join(KINDS)}")
    scheme, host, port = check_url(body.get("url"))
    with _lock:
        reg = load()
        if name in reg and not body.get("replace"):
            raise ValueError(f"{name} is already registered")
        reg[name] = dict(name=name, url=f"{scheme}://{host}:{port}", scheme=scheme, host=host, port=port,
                         kind=kind, gateway=bool(body.get("gateway")), added=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                            time.gmtime()))
        _write(_file(), reg)
        keys = _read(_keys_file(), {})
        if body.get("api_key"):
            keys[name] = str(body["api_key"])[:500]
        elif name in keys and body.get("api_key") == "":
            keys.pop(name)
        _write(_keys_file(), keys, 0o600)
        _cache.pop(name, None)
    return public(reg[name])


def delete(body):
    name = str(body.get("name") or "")
    with _lock:
        reg = load()
        if name not in reg:
            raise ValueError(f"no remote endpoint {name!r}")
        reg.pop(name)
        _write(_file(), reg)
        keys = _read(_keys_file(), {})
        if keys.pop(name, None) is not None:
            _write(_keys_file(), keys, 0o600)
        _cache.pop(name, None)
    return dict(ok=True)


def test(body):
    """Probe an address without saving it (and without keeping its key)."""
    scheme, host, port = check_url(body.get("url"))
    r = dict(name="__test__", scheme=scheme, host=host, port=port, kind=str(body.get("kind") or "auto"),
             _key=str(body.get("api_key") or "")[:500] or None)
    return probe(r)


def public(r):
    return dict({k: v for k, v in r.items()}, has_key=bool(_key(r["name"])))


def listing(block=True):
    reg = load()
    out = [None] * len(reg)
    items = list(reg.values())

    def one(i):
        out[i] = dict(public(items[i]), status=_cached_probe(items[i], block))
    ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(items))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(TIMEOUT * 6)
    return [x for x in out if x]


def servers():
    """Entries for the server list (the same columns as a local process)."""
    rows = []
    for x in listing(block=False):
        s = x["status"]
        warn = []
        if not x["has_key"] and s.get("health") == "ok":
            warn.append("no API key set: anyone who can reach this address can use it")
        if x["scheme"] == "http" and x["has_key"]:
            warn.append("the API key travels unencrypted (http)")
        model = s.get("model") or (s.get("models") or [None])[0]
        rows.append(dict(pid=None, remote=dict(name=x["name"], url=x["url"], kind=s.get("kind"), gateway=x["gateway"]),
                         model=model or "", model_name=Path(model).name if model else None, alias=x["name"],
                         backend=s.get("kind"), host=x["host"], port=x["port"], ctx=s.get("ctx"),
                         health=s.get("health"), slots=s.get("slots"),
                         slots_busy=s.get("slots_busy") if s.get("slots") is not None else None,
                         live_tps=round(s["live_tps"], 1) if s.get("live_tps") else None,
                         running=s.get("running"), waiting=s.get("waiting"), gpu=[], managed=False,
                         instance=None, warnings=warn + ([s["error"]] if s.get("error") else [])))
    return rows


def gateway_targets():
    """name -> [(host, port, key, where, scheme, upstream)] for the endpoints routed through the gateway."""
    out = {}
    for r in load().values():
        if not r.get("gateway"):
            continue
        p = _cached_probe(r, block=False)
        if p.get("health") != "ok":
            continue
        t = (r["host"], int(r["port"]), _key(r["name"]), f"remote:{r['name']}", r["scheme"])
        ms = list(p.get("models") or [])
        for n in {r["name"], *ms}:
            # by its LexiPanel name: the server's own model name when it has just one (vLLM refuses others)
            up = n if n in ms else (ms[0] if len(ms) == 1 else None)
            out.setdefault(n, []).append(t + (up,))
    return out
