#!/usr/bin/env python3
"""
Camelid instances (added 2026-09-23): github.com/timtoole02/Camelid, a
Rust-native GGUF chat engine with its own web UI and an OpenAI-compatible API.

An instance whose instance.json says engine="camelid" runs `camelid serve`
through the same unit and launcher as the other engines. What differs lives here:

  * parameters: CM_* keys in the instance's params.env
  * its model catalog (`camelid pull` with no argument prints it, including
    whether each model fits this host) and pulls into ~/camelid/models
  * launch_plan(): argv + env, the API key written to a 0600 file
  * a chat proxy for the Status tab's Chat card

Facts learned bringing it up on LexiPanel (v0.7.8), do not rediscover:
  * GPU paths are CUDA (NVIDIA) and Metal (macOS) only. There is no Vulkan or
    ROCm, so on an AMD card it runs on the CPU; the panel refuses AMD devices.
  * The Linux archive bundles the CUDA runtime (libnvrtc) next to the binary:
    LD_LIBRARY_PATH must point there. BACKEND=cpu also hides CUDA devices so the
    engine never probes the NVIDIA driver.
  * `--addr host:port`, not --port. The default models dir is ./models of the
    working directory, so the plan always passes --models-dir.
  * A non-loopback bind is refused by camelid itself unless there is an API key
    (or --allow-unauthenticated-remote) AND TLS (or --allow-cleartext-remote).
  * /health: {"generation_ready": bool, "generation_readiness_reason": ...}
    while the model warms up.
"""
import json, os, re, shlex, subprocess, threading, time, urllib.error, urllib.request
from pathlib import Path

P = None
E = None

DEFAULTS = dict(
    BACKEND="cpu", PORT=8086, HOST="127.0.0.1", CM_MODEL="", CM_MODELS_DIR="",
    CM_THREADS=6, CM_KV_QUANT="f16", CM_THINKING=0, CM_DETERMINISTIC=0,
    CM_SPEC="", CM_SPEC_DRAFT="", CM_SPEC_TOKENS=5,
    CM_MAX_PROMPT=131072, CM_MAX_GEN=8192, CM_API_KEY="", CM_LAN_CHAT_ONLY=0,
    RAM_FLOOR_MB=2048, CM_EXTRA="",
)
BACKENDS = ("cpu", "cuda")
GROUPS = ["Backend", "Model", "Generation", "Server"]
KV_QUANTS = ["f16", "q8_0", "q4_0", "fp8_e4m3", "fp8_e5m2"]


def bind(panel_module, engines_module):
    global P, E
    P, E = panel_module, engines_module


def home():
    return P.HOME / "camelid"


def models_dir(v=None):
    d = str((v or {}).get("CM_MODELS_DIR") or "").strip()
    return Path(d) if d else home() / "models"


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def _m(group, label, tip, type="text", options=None, strict=False, unit=None, numeric=False):
    d = dict(group=group, label=label, tip=tip, type=type)
    if options is not None:
        d["options"] = options
    if strict:
        d["strict"] = True
    if unit:
        d["unit"] = unit
    if numeric:
        d["numeric"] = True
    return d


def local_models(v=None):
    """GGUFs a Camelid instance could load: its models dir, then ~/models."""
    seen, out = set(), []
    for d in (models_dir(v), P.MODELS):
        try:
            for f in sorted(Path(d).rglob("*.gguf")):
                rp = os.path.realpath(f)
                if rp in seen or "mmproj" in f.name.lower():
                    continue
                seen.add(rp)
                out.append(dict(path=str(f), name=f.name, size=f.stat().st_size,
                                where="camelid" if Path(d) == models_dir(v) else "models"))
        except OSError:
            continue
    return out


def meta():
    flag = lambda f: f"<br><br><span style='opacity:.75'>Flag: {f}</span>"
    files = [""] + [m["path"] for m in local_models()]
    return {
        "BACKEND": _m("Backend", "Compute backend",
                      "<b>cpu</b> runs everywhere and is the only choice on AMD cards: Camelid has "
                      "no Vulkan or ROCm. <b>cuda</b> uses an NVIDIA card (the Linux build ships its "
                      "own CUDA runtime; the NVIDIA driver must be installed)." + flag("--gpu off / on"),
                      "select", list(BACKENDS), strict=True),
        "CM_THREADS": _m("Backend", "CPU threads",
                         "Worker threads for the CPU path. This box has 8 cores; leave some for the "
                         "other servers." + flag("--threads"), "int", numeric=True),
        "CM_MODEL": _m("Model", "Model file",
                       "The GGUF to load at start. Camelid supports a curated list of models (the "
                       "catalog below, with a fit check for this machine); other GGUFs may refuse to "
                       "load. Files from ~/camelid/models and ~/models are offered." + flag("--model"),
                       "select", files),
        "CM_MODELS_DIR": _m("Model", "Models folder",
                            "Where Camelid's catalog downloads go and what its own web UI lists. "
                            "Empty = ~/camelid/models." + flag("--models-dir")),
        "CM_KV_QUANT": _m("Model", "KV cache precision",
                          "Memory used per token of context. <b>f16</b> exact; <b>q8_0</b> and the "
                          "<b>fp8</b> formats halve it; <b>q4_0</b> quarters it. On the CUDA path only "
                          "f16 and q8_0 are honoured." + flag("--kv-quant"), "select", KV_QUANTS,
                          strict=True),
        "CM_THINKING": _m("Generation", "Thinking on by default",
                          "Qwen3 / Gemma 4 think before answering unless a request says otherwise. "
                          "Better answers, slower replies." + flag("--enable-thinking"), "bool"),
        "CM_DETERMINISTIC": _m("Generation", "Deterministic",
                               "Same input gives bit-identical output (order-stable CPU path, GPU off). "
                               "Slower; for testing." + flag("--deterministic"), "bool"),
        "CM_SPEC": _m("Generation", "Speculative decoding",
                      "Faster replies with identical output. <b>ngram</b> guesses ahead from the "
                      "prompt (no extra model). <b>draft</b> uses a small model with the same "
                      "tokenizer (set below). Empty = off." + flag("--spec-decode"),
                      "select", ["", "ngram", "draft"], strict=True),
        "CM_SPEC_DRAFT": _m("Generation", "Draft model",
                            "Small GGUF for draft mode; must share the main model's tokenizer."
                            + flag("--spec-draft-model"), "select", files),
        "CM_SPEC_TOKENS": _m("Generation", "Draft tokens per round",
                             "How far each guess runs ahead." + flag("--spec-draft-tokens"),
                             "int", numeric=True),
        "CM_MAX_PROMPT": _m("Generation", "Max prompt tokens",
                            "Longest prompt accepted." + flag("--max-prompt-tokens"), "int",
                            numeric=True),
        "CM_MAX_GEN": _m("Generation", "Max reply tokens",
                         "Largest max_tokens a request may ask for." + flag("--max-generation-tokens"),
                         "int", numeric=True),
        "RAM_FLOOR_MB": _m("Backend", "Host RAM floor",
                           "The launcher stops Camelid if free host RAM falls below this. The host "
                           "has hard-locked from running out of RAM.", "int", unit="MB", numeric=True),
        "PORT": _m("Server", "Port", "Camelid's API and web UI." + flag("--addr"), "int", numeric=True),
        "HOST": _m("Server", "Listen address",
                   "<b>127.0.0.1</b>: this box only (reach the web UI through an SSH tunnel). "
                   "<b>0.0.0.0</b>: the LAN; then an API key is required and traffic is "
                   "unencrypted." + flag("--addr"), "select", ["127.0.0.1", "0.0.0.0"]),
        "CM_API_KEY": _m("Server", "API key",
                         "Required for a LAN listener. Stored in a 0600 file, passed with "
                         "--api-key-file so it never shows in the process list." + flag("--api-key-file")),
        "CM_LAN_CHAT_ONLY": _m("Server", "LAN: chat only",
                               "On a LAN listener, expose only chat (no model changes, workspace or "
                               "agents)." + flag("--lan-chat-only"), "bool"),
        "CM_EXTRA": _m("Server", "Extra arguments", "Anything else `camelid serve` accepts."),
    }


def _coerce(vals):
    for k, v in list(vals.items()):
        d = DEFAULTS.get(k)
        if isinstance(d, int) and str(v).lstrip("-").isdigit():
            vals[k] = int(v)
    return vals


def load_params(inst):
    cur = dict(DEFAULTS)
    cur.update(P._read_env_file(inst["dir"] / "params.env"))
    return _coerce(cur)


def write_params(inst, vals, note=""):
    body = ["# GENERATED by the admin panel - do not hand-edit.",
            f"# Camelid instance '{inst['id']}'. Read by panel/instance_launch.py.",
            *([f"# {note}"] if note else []), ""]
    for k in DEFAULTS:
        body.append(f"{k}={P._env_quote(vals.get(k, DEFAULTS[k]))}")
    pf = inst["dir"] / "params.env"
    pf.write_text("\n".join(body) + "\n")
    os.chmod(pf, 0o600)                              # it may hold the API key
    kf = _key_file(inst)
    key = str(vals.get("CM_API_KEY") or "")
    if key:                                          # camelid reads it with --api-key-file,
        kf.write_text(key + "\n")                   # so it never appears in argv or the plan
        os.chmod(kf, 0o600)
    elif kf.exists():
        kf.unlink()


def save_params(inst, new):
    cur = load_params(inst)
    unknown = sorted(set(new) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"not Camelid settings: {', '.join(unknown)}")
    cur.update(new)
    cur = _coerce(cur)
    if cur["BACKEND"] not in BACKENDS:
        raise ValueError("BACKEND must be cpu or cuda")
    if cur["CM_KV_QUANT"] not in KV_QUANTS:
        raise ValueError(f"CM_KV_QUANT must be one of {', '.join(KV_QUANTS)}")
    if cur["CM_SPEC"] not in ("", "ngram", "draft"):
        raise ValueError("CM_SPEC must be empty, ngram or draft")
    try:
        port = int(cur["PORT"])
        assert 1024 <= port <= 65535 and port not in (8090, 8091, 8092)
    except (ValueError, AssertionError):
        raise ValueError("PORT must be 1024-65535 and not the panel's own ports")
    for k in ("CM_THREADS", "CM_SPEC_TOKENS", "CM_MAX_PROMPT", "CM_MAX_GEN"):
        if int(cur[k]) < 1:
            raise ValueError(f"{k} must be at least 1")
    try:
        shlex.split(str(cur.get("CM_EXTRA") or ""))
    except ValueError as e:
        raise ValueError(f"CM_EXTRA does not parse: {e}")
    write_params(inst, cur)
    return cur


# --------------------------------------------------------------------------
# catalog + pulls
# --------------------------------------------------------------------------
_cat_cache = {}


def _bindir():
    return E.ENGINES["camelid"].active("linux")


def _run(bindir, *args, timeout=120, cwd=None):
    return subprocess.run([f"{bindir}/camelid", *args], capture_output=True, text=True,
                          timeout=timeout, cwd=cwd or str(home()),
                          env=dict(os.environ, LD_LIBRARY_PATH=bindir, CUDA_VISIBLE_DEVICES=""))


def catalog(force=False):
    """`camelid pull` with no argument: id, quant, size, fit-on-this-host, name."""
    bindir = _bindir()
    if not bindir:
        return dict(models=[], note="no Camelid build installed - install one on the Builds tab")
    c = _cat_cache.get(bindir)
    if c and not force and time.time() - c[0] < 600:
        rows = c[1]
    else:
        home().mkdir(exist_ok=True)
        r = _run(bindir, "pull", timeout=60)
        out = r.stdout + r.stderr                     # the table goes to stderr
        rows, cols = [], None
        for line in out.splitlines():
            if cols is None:
                if re.match(r"^\s+ID\s+QUANT\s+SIZE\s+FIT", line):
                    # fixed-width table: slice by the header's column starts
                    # (the "needs free memory" rows touch NAME with one space)
                    cols = (line.index("QUANT"), line.index("FIT"), line.index("NAME"))
                continue
            if not line.strip() or len(line) < cols[2]:
                continue
            q, f, n = cols
            mid = line[q:f].split()                  # quant, size, "GB"
            if len(mid) < 3:
                continue
            rows.append(dict(id=line[:q].strip(), quant=mid[0], size=f"{mid[1]} {mid[2]}",
                             fit=line[f:n].strip(), name=line[n:].strip()))
        _cat_cache[bindir] = (time.time(), rows)
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    have = [norm(m["name"]) for m in local_models()]
    for r in rows:                     # "Qwen3 0.6B Q8_0" <-> Qwen3-0.6B-Q8_0.gguf
        want = norm(r["name"].split("(")[0])
        r["installed"] = any(want and want in h for h in have)
    return dict(models=rows, models_dir=str(models_dir()), build=bindir)


def pull(model_id, v=None):
    bindir = _bindir()
    if not bindir:
        raise ValueError("no Camelid build installed - install one on the Builds tab")
    cat = {r["id"]: r for r in catalog()["models"]}
    if model_id not in cat:
        raise ValueError(f"{model_id!r} is not in Camelid's catalog")
    for rec in P._downloads.values():
        if rec.get("kind") == "camelid-model" and rec.get("package") == model_id and \
                rec["status"] not in ("done", "failed"):
            raise ValueError(f"{model_id} is already downloading")
    d = models_dir(v)
    d.mkdir(parents=True, exist_ok=True)
    size_gb = float(cat[model_id]["size"].split()[0])
    did = "cmmodel-" + str(int(time.time() * 1000))
    rec = dict(id=did, url=f"camelid:{model_id}", name=f"Camelid model {cat[model_id]['name']}",
               dest=str(d), kind="camelid-model", package=model_id, status="running", pct=0.0,
               downloaded=0, total=int(size_gb * 1e9), started=time.time(), error=None, file=None)
    with P._lock:
        P._downloads[did] = rec

    def run():
        try:
            p = subprocess.Popen([f"{bindir}/camelid", "pull", model_id, "--models-dir", str(d)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(home()),
                                 env=dict(os.environ, LD_LIBRARY_PATH=bindir, CUDA_VISIBLE_DEVICES=""))
            buf, tail = b"", []
            while True:
                ch = p.stdout.read(256)
                if not ch:
                    break
                buf += ch
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for part in parts:
                    line = part.decode(errors="replace").strip()
                    m = re.match(r"^(\d{1,3})\s+[\d.]+[kKMG]?\s", line)
                    if m:
                        rec["pct"] = float(m.group(1))
                        rec["downloaded"] = int(rec["total"] * rec["pct"] / 100)
                    elif line:
                        tail = (tail + [line])[-6:]
                        f = re.search(r"ready at (\S+\.gguf)", line)
                        if f:
                            rec["file"] = f.group(1)
            rc = p.wait()
            if rc != 0:
                raise RuntimeError("; ".join(tail[-3:]) or f"camelid pull exit {rc}")
            rec.update(status="done", pct=100.0)
        except Exception as e:
            rec.update(status="failed", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return {k: v for k, v in rec.items()}


# --------------------------------------------------------------------------
# launch plan
# --------------------------------------------------------------------------
def _key_file(inst):
    return inst["dir"] / "camelid-api-key"


def launch_plan(inst, values=None):
    v = dict(values or load_params(inst))
    errors, warnings = [], []
    backend = str(v.get("BACKEND") or "cpu")
    bindir = _bindir()
    if not bindir:
        errors.append("no Camelid build installed - install one on the Builds tab")
    alld = {d["pci"]: d for d in P.gpu_devices(probe=False)}
    devs = [alld[p] for p in inst["devices"] if p != "cpu" and p in alld]
    for p in inst["devices"]:
        if p != "cpu" and p not in alld:
            errors.append(f"device {p} is not in this machine")
    if any(d.get("vendor") != "nvidia" for d in devs):
        errors.append("Camelid has no Vulkan or ROCm support: it can use an NVIDIA card (cuda) "
                      "or the CPU only")
    if backend == "cuda" and not devs:
        errors.append("BACKEND=cuda needs an NVIDIA card on this instance")
    if backend == "cuda" and devs and devs[0].get("driver") != "nvidia":
        errors.append(f"{devs[0]['name']} is bound to {devs[0].get('driver')}; CUDA needs the "
                      "NVIDIA driver")

    model = str(v.get("CM_MODEL") or "")
    if not model:
        errors.append("no model: pick one on the Parameters tab (or pull one from the catalog)")
    elif not Path(model).is_file():
        errors.append(f"CM_MODEL: {model} does not exist")
    if v.get("CM_SPEC") == "draft":
        if not v.get("CM_SPEC_DRAFT"):
            errors.append("CM_SPEC=draft needs a draft model")
        elif not Path(v["CM_SPEC_DRAFT"]).is_file():
            errors.append(f"CM_SPEC_DRAFT: {v['CM_SPEC_DRAFT']} does not exist")

    size_mib = Path(model).stat().st_size // 1048576 if model and Path(model).is_file() else 0
    avail = P._meminfo_mb("MemAvailable")
    floor = int(v.get("RAM_FLOOR_MB") or 2048)
    if backend == "cpu" and avail and size_mib and size_mib + floor > avail:
        errors.append(f"the model needs ~{size_mib} MiB of host RAM; only {avail} MiB is free with "
                      f"the {floor} MiB floor - this box has hard-locked from running out of RAM")
    elif backend == "cpu" and avail and size_mib and size_mib > avail * 0.6:
        warnings.append(f"the model takes ~{size_mib} MiB of the {avail} MiB host RAM free")

    host = str(v.get("HOST") or "127.0.0.1")
    port = int(v.get("PORT") or 0)
    lan = host not in ("127.0.0.1", "localhost", "::1")
    key = str(v.get("CM_API_KEY") or "")
    if lan and not key:
        errors.append("a LAN listener needs an API key (CM_API_KEY); Camelid refuses to serve the "
                      "network without one")
    for other in P.instance_ids():
        if other == inst["id"]:
            continue
        try:
            o = P.get_instance(other)
        except ValueError:
            continue
        if str(P._read_env_file(o["dir"] / "params.env").get("PORT") or
               (P.DEFAULTS["PORT"] if o["legacy"] else "")) == str(port):
            warnings.append(f"PORT {port} is shared with instance '{other}'")

    md = models_dir(v)
    argv = [f"{bindir or '<no build>'}/camelid", "serve", "--addr", f"{host}:{port}", "--no-open",
            "--models-dir", str(md), "--threads", str(v.get("CM_THREADS") or 6),
            "--kv-quant", str(v.get("CM_KV_QUANT") or "f16"),
            "--gpu", "on" if backend == "cuda" else "off",
            "--max-prompt-tokens", str(v.get("CM_MAX_PROMPT") or 131072),
            "--max-generation-tokens", str(v.get("CM_MAX_GEN") or 8192)]
    if model:
        argv += ["--model", model]
    if _truthy(v.get("CM_THINKING")):
        argv.append("--enable-thinking")
    if _truthy(v.get("CM_DETERMINISTIC")):
        argv.append("--deterministic")
    if v.get("CM_SPEC"):
        argv += ["--spec-decode", v["CM_SPEC"], "--spec-draft-tokens", str(v.get("CM_SPEC_TOKENS") or 5)]
        if v["CM_SPEC"] == "draft" and v.get("CM_SPEC_DRAFT"):
            argv += ["--spec-draft-model", v["CM_SPEC_DRAFT"]]
    if key:
        if not _key_file(inst).is_file():
            errors.append("the API key file is missing - save the parameters again")
        argv += ["--api-key-file", str(_key_file(inst))]
    if lan:
        argv.append("--allow-cleartext-remote")
        warnings.append("Camelid on the LAN is unencrypted HTTP: the API key and chats cross the "
                        "network in clear text")
        if _truthy(v.get("CM_LAN_CHAT_ONLY")):
            argv.append("--lan-chat-only")
    try:
        argv += shlex.split(str(v.get("CM_EXTRA") or ""))
    except ValueError as e:
        errors.append(f"CM_EXTRA does not parse: {e}")
    env = dict(LD_LIBRARY_PATH=bindir or "")
    if backend == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return dict(engine="camelid", argv=argv, env=env, errors=errors, warnings=warnings,
                backend=backend, bindir=bindir, rundir=str(inst["rundir"]),
                device=dict(name=devs[0]["name"] if devs else "CPU",
                            pci=devs[0]["pci"] if devs else "cpu"),
                placement=[backend], weights_mib=size_mib, host_ram_mib=size_mib if backend == "cpu" else 0,
                vulkan_map=[], template_copy=None, api_key=None)


def _truthy(x):
    return str(x).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# the running server
# --------------------------------------------------------------------------
def _addr(argv):
    a = P._argv_get(argv, ("--addr",)) or "127.0.0.1:8181"
    host, _, port = a.rpartition(":")
    return (host or "127.0.0.1"), port


def _http(host, port, path, body=None, timeout=10, key=None):
    h = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://{h}:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET", headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return dict(_error=e.code, detail=json.loads(e.read() or b"{}"))
        except ValueError:
            return dict(_error=e.code, detail={})
    except Exception:
        return None


def describe(pid, argv):
    host, port = _addr(argv)
    h = _http(host, port, "/health", timeout=1.5)
    ms = _http(host, port, "/v1/models", timeout=1.5) or {}
    names = [m.get("id") for m in ms.get("data") or []]
    model = P._argv_get(argv, ("--model",)) or ""
    health = "ok" if isinstance(h, dict) and h.get("generation_ready") else \
        ("no answer" if h is None else "loading")
    exe = os.path.realpath(f"/proc/{pid}/exe") if os.path.exists(f"/proc/{pid}/exe") else argv[0]
    return dict(engine="camelid", model=model, model_name=", ".join(names) or Path(model).name or None,
                binary=exe, host=host, port=int(port) if str(port).isdigit() else port,
                backend="cuda" if P._argv_get(argv, ("--gpu",)) == "on" else "cpu", health=health,
                ctx=(h or {}).get("active_context_length"), kv=P._argv_get(argv, ("--kv-quant",)),
                ngl=None, parallel=None, spec=P._argv_get(argv, ("--spec-decode",)), mmproj=None,
                cache_ram=None, draft=P._argv_get(argv, ("--spec-draft-model",)), alias=None,
                batch=None, ubatch=None, device=None, log_file=None, slots=None, slots_busy=None,
                live_tps=None, visible_devices={})


def _server(inst):
    with P.using_instance(inst):
        pid = P.server_pid()
    if not pid:
        raise ValueError(f"{inst['id']} is not running - press Start")
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    host, port = _addr(argv)
    return host, port, str(load_params(inst).get("CM_API_KEY") or "") or None


def health(inst):
    try:
        host, port, key = _server(inst)
    except ValueError:
        return None
    h = _http(host, port, "/health", timeout=2, key=key)
    if h is None:
        return dict(status="loading")
    if h.get("_error"):
        return dict(status="error", code=h["_error"])
    return dict(status="ok" if h.get("generation_ready") else "loading",
                reason=h.get("generation_readiness_reason"),
                context=h.get("active_context_length"), version=h.get("version"))


def chat(inst, body):
    host, port, key = _server(inst)
    msgs = body.get("messages")
    if not msgs:
        text = str(body.get("text") or "").strip()
        if not text:
            raise ValueError("empty message")
        msgs = [{"role": "user", "content": text}]
    req = dict(messages=msgs, max_tokens=int(body.get("max_tokens") or 512))
    if body.get("temperature") not in (None, ""):
        req["temperature"] = float(body["temperature"])
    t0 = time.time()
    r = _http(host, port, "/v1/chat/completions", req, timeout=900, key=key)
    if r is None:
        raise ValueError("Camelid did not answer")
    if r.get("_error"):
        d = (r.get("detail") or {}).get("error") or {}
        raise ValueError(d.get("message") if isinstance(d, dict) else str(d) or f"HTTP {r['_error']}")
    choice = (r.get("choices") or [{}])[0]
    return dict(reply=(choice.get("message") or {}).get("content"), finish=choice.get("finish_reason"),
                usage=r.get("usage"), model=r.get("model"), seconds=round(time.time() - t0, 2))
