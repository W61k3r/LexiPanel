#!/usr/bin/env python3
"""
Access (1.0.0): single-user or multi-user mode, users, roles, API keys, audit log.

  single  (default) as before: Caddy's one login in front, the panel trusts what reaches it.
  multi   the panel checks every request itself: HTTP Basic (users in auth/users.json, scrypt)
          or an API key (Authorization: Bearer lp_...). Roles, lowest to highest:
            viewer    every GET except the admin ones below
            operator  + start / stop / restart, benchmarks, optimizer, depth curve, auto-fit runs
            admin     everything: settings, parameters, power, GPU tuning, files, builds, fleet, users
          A POST that is not listed as operator is admin: an unclassified route is never open.
Every POST is written to auth/audit.jsonl, hash-chained (an edited line breaks the chain).

Command line (setup and lockout recovery, as the panel's own account):
  python3 auth.py init --admin NAME      multi-user mode with a first admin (asks the password)
  python3 auth.py mode single            back to single-user mode
  python3 auth.py verify                 check the audit chain
"""
import base64, hashlib, json, os, secrets, sys, threading, time
from pathlib import Path

ROLES = ("viewer", "operator", "admin")
OPERATOR_POST = ("/api/start", "/api/stop", "/api/restart", "/api/main/reset-failed", "/api/optimize/start",
                 "/api/optimize/stop", "/api/curve/start", "/api/curve/stop", "/api/workload/experiment",
                 "/api/workload/stop", "/api/workload/proposal/", "/api/refusals/start", "/api/refusals/stop",
                 "/api/fleet/send-now")
ANY_POST = ("/api/mcp", "/api/mcp/call", "/v1/chat/completions")   # models: any user, within their quota            # MCP tools re-check each inner call with the caller's role
OPEN = ("/api/fleet/report", "/api/fleet/join")     # token / join code checked by fleet.py
ADMIN_GET = ("/api/files", "/api/debug-bundle", "/api/backup", "/api/auth/users", "/api/auth/keys", "/api/auth/audit")
_lock = threading.RLock()
_cache = {}                                         # sha256(header) -> (user, role, until)
INTERNAL = secrets.token_hex(16)                    # the in-process MCP server's calls, as its caller
DIR = None


def bind(panel_dir):
    global DIR
    DIR = Path(panel_dir) / "auth"
    DIR.mkdir(exist_ok=True)
    os.chmod(DIR, 0o700)


def _rj(n, d):
    try:
        return json.loads((DIR / n).read_text())
    except (OSError, ValueError):
        return d


def _wj(n, obj):
    t = DIR / (n + ".tmp")
    t.write_text(json.dumps(obj, indent=1))
    os.chmod(t, 0o600)
    os.replace(t, DIR / n)
    _cache.clear()


def mode():
    return _rj("config.json", {}).get("mode", "single")


def _hash(pw, salt):
    return hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1, dklen=32).hex()


def set_user(name, password=None, role=None):
    name = str(name or "").strip()
    if not name or len(name) > 32 or not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("user name: letters, digits, - _ . (up to 32)")
    with _lock:
        us = _rj("users.json", {})
        u = us.get(name) or {}
        if role is not None:
            if role not in ROLES:
                raise ValueError("role: viewer, operator or admin")
            u["role"] = role
        if password is not None:
            if len(password) < 10:
                raise ValueError("password: at least 10 characters")
            u["salt"] = secrets.token_hex(16)
            u["hash"] = _hash(password, u["salt"])
        if "hash" not in u or "role" not in u:
            raise ValueError("a new user needs a password and a role")
        u.setdefault("created", int(time.time()))
        us[name] = u
        if not any(x.get("role") == "admin" for x in us.values()):
            raise ValueError("there must be at least one admin")
        _wj("users.json", us)
    return dict(user=name, role=u["role"])


def delete_user(name):
    with _lock:
        us = _rj("users.json", {})
        if name not in us:
            raise ValueError("no such user")
        del us[name]
        if mode() == "multi" and not any(x.get("role") == "admin" for x in us.values()):
            raise ValueError("that is the last admin")
        _wj("users.json", us)
        ks = {k: v for k, v in _rj("keys.json", {}).items() if v.get("user") != name}
        _wj("keys.json", ks)
    return dict(ok=True)


def create_key(user, role, days=90, label=""):
    us = _rj("users.json", {})
    if user not in us:
        raise ValueError("no such user")
    if role not in ROLES or ROLES.index(role) > ROLES.index(us[user]["role"]):
        raise ValueError("a key's role cannot be above its user's")
    kid, secret = secrets.token_hex(6), secrets.token_urlsafe(24)
    with _lock:
        ks = _rj("keys.json", {})
        ks[kid] = dict(user=user, role=role, hash=hashlib.sha256(secret.encode()).hexdigest(), label=str(label)[:64],
                       created=int(time.time()), expires=int(time.time() + int(days) * 86400), last_used=None)
        _wj("keys.json", ks)
    return dict(id=kid, key=f"lp_{kid}_{secret}", expires=ks[kid]["expires"], note="shown once: store it now")


def revoke_key(kid):
    with _lock:
        ks = _rj("keys.json", {})
        if ks.pop(kid, None) is None:
            raise ValueError("no such key")
        _wj("keys.json", ks)
    return dict(ok=True)


def identify(headers):
    """(user, role) for a request in multi-user mode, or None."""
    h = headers.get("Authorization") or ""
    internal = headers.get("X-LexiPanel-Internal") or ""
    if internal:
        k, _, who = internal.partition(":")
        us = _rj("users.json", {})
        if secrets.compare_digest(k, INTERNAL) and who in us:
            return who, us[who]["role"]
        return None
    ck = hashlib.sha256(h.encode()).hexdigest()
    c = _cache.get(ck)
    if c and c[2] > time.time():
        return c[0], c[1]
    got = None
    if h.startswith("Bearer lp_"):
        kid, _, secret = h[len("Bearer lp_"):].partition("_")
        k = _rj("keys.json", {}).get(kid)
        if k and k["expires"] > time.time() and secrets.compare_digest(k["hash"], hashlib.sha256(secret.encode()).hexdigest()):
            got = (k["user"], k["role"])
            with _lock:
                ks = _rj("keys.json", {})
                if kid in ks:
                    ks[kid]["last_used"] = int(time.time())
                    _wj("keys.json", ks)
    elif h.startswith("Basic "):
        try:
            name, _, pw = base64.b64decode(h[6:]).decode().partition(":")
        except Exception:
            return None
        u = _rj("users.json", {}).get(name)
        if u and secrets.compare_digest(u["hash"], _hash(pw, u["salt"])):
            got = (name, u["role"])
    if got:
        _cache[ck] = (got[0], got[1], time.time() + 300)
    return got


def needed(method, path):
    """The lowest role a request needs; None = open (token-checked elsewhere)."""
    if method == "POST":
        if path in OPEN:
            return None
        if path in ANY_POST:
            return "viewer"
        return "operator" if any(path == r or (r.endswith("/") and path.startswith(r)) for r in OPERATOR_POST) else "admin"
    if path in OPEN:
        return None
    return "admin" if any(path.startswith(r) for r in ADMIN_GET) else "viewer"


def check(method, path, headers):
    """-> (code, user, role): code 0 allowed, 401 log in, 403 not allowed."""
    if mode() != "multi":
        return 0, "local", "admin"
    need = needed(method, path)
    if need is None:
        return 0, None, None
    who = identify(headers)
    if not who:
        return 401, None, None
    return (0 if ROLES.index(who[1]) >= ROLES.index(need) else 403), who[0], who[1]


def audit(user, role, method, path, inst, code):
    with _lock:
        f = DIR / "audit.jsonl"
        prev = "0" * 64
        try:
            with open(f, "rb") as fh:
                fh.seek(max(0, f.stat().st_size - 4096))
                last = fh.read().splitlines()[-1]
                prev = json.loads(last)["h"]
        except (OSError, IndexError, ValueError, KeyError):
            pass
        rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), user=user, role=role, method=method,
                   path=path, inst=inst, code=code, prev=prev)
        rec["h"] = hashlib.sha256((prev + json.dumps(rec, sort_keys=True)).encode()).hexdigest()
        with open(f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        os.chmod(f, 0o600)


def verify():
    prev, n = "0" * 64, 0
    try:
        lines = (DIR / "audit.jsonl").read_text().splitlines()
    except OSError:
        return dict(ok=True, lines=0)
    for i, ln in enumerate(lines):
        r = json.loads(ln)
        h = r.pop("h")
        if r.get("prev") != prev or hashlib.sha256((prev + json.dumps(r, sort_keys=True)).encode()).hexdigest() != h:
            return dict(ok=False, lines=len(lines), broken_at=i + 1)
        prev, n = h, n + 1
    return dict(ok=True, lines=n)


def status(user=None, role=None):
    us, ks = _rj("users.json", {}), _rj("keys.json", {})
    out = dict(mode=mode(), me=dict(user=user, role=role))
    if role == "admin":
        out["users"] = [dict(user=k, role=v["role"], created=v.get("created")) for k, v in sorted(us.items())]
        out["keys"] = [dict(id=k, **{x: v.get(x) for x in ("user", "role", "label", "created", "expires", "last_used")})
                       for k, v in ks.items()]
        try:
            tail = (DIR / "audit.jsonl").read_text().splitlines()[-50:]
        except OSError:
            tail = []
        out["audit"] = dict(verify(), tail=[json.loads(x) for x in reversed(tail)])
    return out


def set_mode(m, admin_user=None, admin_password=None):
    if m not in ("single", "multi"):
        raise ValueError("mode: single or multi")
    if m == "multi":
        if admin_user:
            set_user(admin_user, admin_password, "admin")
        if not any(x.get("role") == "admin" for x in _rj("users.json", {}).values()):
            raise ValueError("multi-user mode needs an admin first (admin_user, admin_password)")
    with _lock:
        _wj("config.json", dict(_rj("config.json", {}), mode=m))
    return dict(mode=m)


if __name__ == "__main__":
    import getpass
    bind(os.environ.get("INF01_PANEL_DIR") or Path(__file__).resolve().parent)
    a = sys.argv[1:]
    if a[:1] == ["init"] and "--admin" in a:
        name = a[a.index("--admin") + 1]
        pw = getpass.getpass(f"password for {name} (10+ characters): ")
        if pw != getpass.getpass("again: "):
            sys.exit("passwords differ")
        print(set_mode("multi", name, pw))
    elif a[:1] == ["mode"] and len(a) == 2:
        print(set_mode(a[1]))
    elif a[:1] == ["verify"]:
        print(verify())
    else:
        sys.exit(__doc__)
