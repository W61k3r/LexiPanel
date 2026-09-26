#!/usr/bin/env python3
"""
Batch jobs (added 2026-09-26): many chat completions submitted at once, run in the background
at batch priority - the gateway's admission queue serves every interactive request first -
and collected as JSONL. The shape follows OpenAI's Batch API; the input is sent inline.

  POST /v1/batches               {"input": [{"custom_id": "...", "body": {chat completion}}, ...],
                                  "model": default model, "metadata": {...}}
                                 or {"input_jsonl": "<one request per line>"}
  GET  /v1/batches               your batches
  GET  /v1/batches/<id>          one batch: status, request_counts
  GET  /v1/batches/<id>/output   results, one JSON line per request, in the order they finished
  POST /v1/batches/<id>/cancel   stop it (finished requests keep their results)
  POST /v1/batches/<id>/delete   remove it and its files

Every request goes through the same quota check, replica choice and slot admission as a live
one, as batch priority. A batch survives a panel restart: it resumes after the last request
that was written.

Privacy. The live gateway stores counts only. A batch cannot work that way: its prompts wait
on disk until they run and its results until they are collected. Both live under
gateway/batches/, readable only by the panel's user (0600), visible only to the user who
submitted them, and deleted RETAIN_DAYS after the batch finished.
"""
import json, os, re, secrets, shutil, threading, time
from pathlib import Path

P = None
G = None                 # gateway
MAX_REQUESTS = 10000
MAX_BYTES = 50 * 1024 * 1024
RETAIN_DAYS = 7
WORKERS = 4              # requests in flight per batch; admission caps what a server gets
_lock = threading.RLock()
_wake = threading.Event()
_cancel = set()


def bind(panel_module, gateway_module):
    global P, G
    P, G = panel_module, gateway_module


def _dir(bid=None):
    d = P.PANEL / "gateway" / "batches"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    if bid:
        d = d / bid
        d.mkdir(exist_ok=True)
        os.chmod(d, 0o700)
    return d


def _write(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def _append(path, line):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(line + "\n")


def _meta(bid):
    try:
        return json.loads((_dir() / bid / "batch.json").read_text())
    except (OSError, ValueError):
        return None


def _save(m):
    tmp = _dir(m["id"]) / "batch.json.tmp"
    _write(tmp, json.dumps(m, indent=1))
    os.replace(tmp, _dir(m["id"]) / "batch.json")


def _public(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def _own(user, bid):
    if not re.fullmatch(r"batch_[0-9a-f]{24}", bid or ""):
        raise ValueError("bad batch id")
    m = _meta(bid)
    if not m or m.get("_user") != (user or "local"):
        raise KeyError(f"no batch {bid}")                 # other users' batches do not exist for you
    return m


# ============================================================================
# API
# ============================================================================
def create(user, body):
    reqs = body.get("input")
    if reqs is None and isinstance(body.get("input_jsonl"), str):
        if len(body["input_jsonl"]) > MAX_BYTES:
            raise ValueError(f"input over {MAX_BYTES // 1048576} MB")
        try:
            reqs = [json.loads(l) for l in body["input_jsonl"].splitlines() if l.strip()]
        except ValueError as e:
            raise ValueError(f"input_jsonl: line not JSON ({e})")
    if not isinstance(reqs, list) or not reqs:
        raise ValueError("input: a list of {custom_id, body} (or input_jsonl)")
    if len(reqs) > MAX_REQUESTS:
        raise ValueError(f"at most {MAX_REQUESTS} requests per batch")
    default_model = body.get("model")
    seen, lines = set(), []
    for i, r in enumerate(reqs):
        if not isinstance(r, dict) or not isinstance(r.get("body"), dict):
            raise ValueError(f"request {i}: {{custom_id, body}} with body a chat completion")
        cid = str(r.get("custom_id") or f"request-{i}")[:128]
        if cid in seen:
            raise ValueError(f"custom_id {cid!r} appears twice")
        seen.add(cid)
        b = dict(r["body"])
        b.setdefault("model", default_model)
        if not b.get("model"):
            raise ValueError(f"request {cid}: no model (set body.model or the batch's model)")
        if not isinstance(b.get("messages"), list):
            raise ValueError(f"request {cid}: body.messages must be a list")
        b.pop("stream", None)
        lines.append(json.dumps(dict(custom_id=cid, body=b)))
    text = "\n".join(lines) + "\n"
    if len(text) > MAX_BYTES:
        raise ValueError(f"input over {MAX_BYTES // 1048576} MB")
    bid = "batch_" + secrets.token_hex(12)
    now = int(time.time())
    m = dict(id=bid, object="batch", endpoint="/v1/chat/completions", status="in_progress", created_at=now,
             in_progress_at=now, completed_at=None, cancelled_at=None, failed_at=None,
             request_counts=dict(total=len(lines), completed=0, failed=0),
             metadata={str(k)[:64]: str(v)[:512] for k, v in (body.get("metadata") or {}).items()} if
             isinstance(body.get("metadata"), dict) else {},
             _user=user or "local", _expires=None)
    _write(_dir(bid) / "input.jsonl", text)
    _save(m)
    _wake.set()
    return _public(m)


def listing(user):
    out = []
    for d in sorted(_dir().glob("batch_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = _meta(d.name)
        if m and m.get("_user") == (user or "local"):
            out.append(_public(m))
    return dict(object="list", data=out[:100])


def get(user, bid):
    return _public(_own(user, bid))


def output(user, bid):
    _own(user, bid)
    f = _dir() / bid / "output.jsonl"
    return f.read_text() if f.exists() else ""


def cancel(user, bid):
    m = _own(user, bid)
    if m["status"] in ("completed", "failed", "cancelled"):
        return _public(m)
    with _lock:
        _cancel.add(bid)
        m = _meta(bid)
        m.update(status="cancelling")
        _save(m)
    _wake.set()
    return _public(m)


def delete(user, bid):
    m = _own(user, bid)
    if m["status"] in ("in_progress", "cancelling"):
        raise ValueError("cancel it first")
    shutil.rmtree(_dir() / bid, ignore_errors=True)
    return dict(id=bid, object="batch.deleted", deleted=True)


def summary():
    """Admin view: counts per status, no contents."""
    by = {}
    for d in _dir().glob("batch_*"):
        m = _meta(d.name)
        if m:
            by[m["status"]] = by.get(m["status"], 0) + 1
    return dict(batches=by, workers=WORKERS, retain_days=RETAIN_DAYS)


# ============================================================================
# worker
# ============================================================================
def _run_one(m, line):
    req = json.loads(line)
    body, cid = req["body"], req["custom_id"]
    user, model = m["_user"], body.get("model")
    no = G.admit(user, model)
    if no:
        return dict(id=f"req_{secrets.token_hex(6)}", custom_id=cid, response=None,
                    error=dict(code="quota", message=no))
    ts = G.targets()
    if model not in ts:
        return dict(id=f"req_{secrets.token_hex(6)}", custom_id=cid, response=None,
                    error=dict(code="model_not_found", message=f"no running instance serves {model!r}"))
    down, need = set(), G.need_tokens(body)
    while True:
        r = G.choose(ts[model], down)
        if r is None:
            G.record(user, model, 0, 0)
            return dict(id=f"req_{secrets.token_hex(6)}", custom_id=cid, response=None,
                        error=dict(code="unavailable", message=f"no replica of {model!r} answered"))
        try:
            G.admit_slot(r[0], r[1], "batch", need=need)
        except TimeoutError as e:
            return dict(id=f"req_{secrets.token_hex(6)}", custom_id=cid, response=None,
                        error=dict(code="timeout", message=str(e)))
        G._busy[(r[0], r[1])] = G._busy.get((r[0], r[1]), 0) + 1
        try:
            status, resp = G.call(r[0], r[1], r[2], body, r[4] if len(r) > 4 else "http",
                                  upstream=r[5] if len(r) > 5 else None)
        except OSError:
            down.add((r[0], r[1]))
            continue
        finally:
            G._busy[(r[0], r[1])] = max(0, G._busy.get((r[0], r[1]), 1) - 1)
            G.release_slot(r[0], r[1], need)
        u = (resp or {}).get("usage") or {}
        G.record(user, model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return dict(id=f"req_{secrets.token_hex(6)}", custom_id=cid,
                    response=dict(status_code=status, body=resp), error=None)


def _process(bid):
    m = _meta(bid)
    if m["status"] == "cancelling":                        # a cancel the panel restarted during
        _cancel.add(bid)
    lines = (_dir() / bid / "input.jsonl").read_text().splitlines()
    out_f = _dir() / bid / "output.jsonl"
    # Resume by what has a result, not by position: with several requests in flight a later
    # one can finish first, so "the next index" would skip one after a crash.
    done = set()
    if out_f.exists():
        for l in out_f.read_text().splitlines():
            try:
                done.add(json.loads(l)["custom_id"])
            except (ValueError, KeyError):
                continue
    todo = [i for i, l in enumerate(lines) if json.loads(l)["custom_id"] not in done]
    idx_lock = threading.Lock()
    state = dict(pos=0)

    def take():
        with idx_lock:
            if bid in _cancel or state["pos"] >= len(todo):
                return None
            i = todo[state["pos"]]
            state["pos"] += 1
            return i

    def worker():
        while True:
            i = take()
            if i is None:
                return
            res = _run_one(m, lines[i])
            with _lock:
                _append(out_f, json.dumps(res))
                cur = _meta(bid)
                ok = res["error"] is None and (res["response"] or {}).get("status_code") == 200
                cur["request_counts"]["completed" if ok else "failed"] += 1
                _save(cur)
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    with _lock:
        cur = _meta(bid)
        now = int(time.time())
        if bid in _cancel:
            _cancel.discard(bid)
            cur.update(status="cancelled", cancelled_at=now)
        else:
            c = cur["request_counts"]
            cur.update(status="completed" if c["completed"] or not c["failed"] else "failed",
                       completed_at=now if c["completed"] or not c["failed"] else None,
                       failed_at=now if not c["completed"] and c["failed"] else None)
        cur["_expires"] = now + RETAIN_DAYS * 86400
        _save(cur)


def _purge():
    now = time.time()
    for d in _dir().glob("batch_*"):
        m = _meta(d.name)
        if m and m.get("_expires") and now > m["_expires"]:
            shutil.rmtree(d, ignore_errors=True)


def worker():
    """One batch at a time, oldest first; purges expired ones."""
    while True:
        try:
            _purge()
            pending = [m for m in (_meta(d.name) for d in _dir().glob("batch_*"))
                       if m and m["status"] in ("in_progress", "cancelling")]
            pending.sort(key=lambda m: m["created_at"])
            if pending:
                _process(pending[0]["id"])
                continue
        except Exception as e:
            print(f"batches: {e}", flush=True)
        _wake.wait(30)
        _wake.clear()
