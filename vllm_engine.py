#!/usr/bin/env python3
"""
vLLM instances (added 2026-09-26): serving many requests at once.

llama.cpp gives each slot one contiguous KV region and serves a few conversations well.
vLLM pages the KV cache (PagedAttention: fixed-size blocks handed out as sequences grow,
shared between requests with a common prefix) and schedules every step across all running
requests, so throughput keeps climbing when many requests of different lengths are in
flight. This module runs `vllm serve` as a LexiPanel instance: the same unit (inf01-inst@),
launcher, Status and Parameters tabs, gateway and Bench goodput as every other engine.

Runtime. vLLM lives in its own venv (~/vllm/venv-cu for NVIDIA CUDA, ~/vllm/venv-rocm for
AMD ROCm; `bash install-vllm.sh [--rocm]`), never the system Python: its wheels need Python
3.10-3.13, and the ROCm wheels 3.12. The instance's BACKEND picks the venv; VL_VENV overrides.

Models. A local Hugging Face folder (config.json + safetensors) or a repo id, downloaded
into ~/vllm/hf-cache at the first start. GGUF files are llama.cpp's format, not this one's.

Memory. vLLM takes VL_GPU_MEM_UTIL of the card at start whatever else is on it, so a plan is
refused on a card another running server uses. VL_MAX_MODEL_LEN caps one request's context;
the KV pool is whatever that fraction leaves after the weights.

Secrets and privacy. The API key reaches the server as VLLM_API_KEY, which the launcher asks
this module for (secrets()) after writing its log: never on the command line (readable in /proc
by every local user), in the plan (returned by /api/launch-plan) or in the launch log. It is
stored in a 0600 file and shown as ******** afterwards.
--trust-remote-code (which runs Python from the model repository) is off unless ticked.
VLLM_NO_USAGE_STATS and DO_NOT_TRACK switch off vLLM's usage reporting.
"""
import json, os, re, shlex, subprocess, time
from pathlib import Path

P = None
BACKENDS = ["cuda", "rocm"]
GROUPS = ["Model", "Memory", "Scheduler", "Server"]
DTYPES = ["auto", "float16", "bfloat16", "float32"]
KV_DTYPES = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
QUANTS = ["", "awq", "gptq", "awq_marlin", "gptq_marlin", "compressed-tensors", "fp8", "bitsandbytes", "gguf"]
MASK = "********"
DEFAULTS = dict(VL_MODEL="", VL_SERVED_NAME="", BACKEND="cuda", VL_VENV="", VL_DTYPE="auto", VL_MAX_MODEL_LEN=0,
                VL_GPU_MEM_UTIL="0.85", VL_MAX_NUM_SEQS=32, VL_MAX_NUM_BATCHED_TOKENS=0, VL_KV_CACHE_DTYPE="auto",
                VL_QUANTIZATION="", VL_ENFORCE_EAGER="0", VL_PREFIX_CACHING="1", VL_TP=1, VL_TRUST_REMOTE_CODE="0",
                VL_EXTRA="", VL_API_KEY="", PORT=8084, HOST="127.0.0.1", RAM_FLOOR_MB=4096)
# flags this module sets itself, and flags a free-form VL_EXTRA must never smuggle in
MANAGED = {"--host", "--port", "--api-key", "--served-model-name", "--model", "--trust-remote-code",
           "--dtype", "--max-model-len", "--gpu-memory-utilization", "--max-num-seqs", "--max-num-batched-tokens",
           "--kv-cache-dtype", "--quantization", "--enforce-eager", "--tensor-parallel-size", "--download-dir",
           "--enable-prefix-caching", "--no-enable-prefix-caching", "--ssl-keyfile", "--ssl-certfile",
           "--allowed-local-media-path", "--root-path"}
_rt_cache = {}


def bind(panel_module):
    global P
    P = panel_module


def _m(group, label, tip, kind="str", options=None, **kw):
    return dict(group=group, label=label, tip=tip, type=kind, options=options, **kw)


def meta():
    b = lambda k: "<br><br><span style='opacity:.75'>" + k + "</span>"
    return {
        "VL_MODEL": _m("Model", "Model", "A local Hugging Face model folder (config.json + *.safetensors), or a "
                       "repo id such as <code>Qwen/Qwen3-1.7B</code> (downloaded into ~/vllm/hf-cache at the first "
                       "start). GGUF files belong to llama.cpp instances." + b("vllm serve MODEL")),
        "VL_SERVED_NAME": _m("Model", "Model name", "The name clients use and see in /v1/models. Empty: the "
                             "folder or repo name." + b("--served-model-name")),
        "VL_QUANTIZATION": _m("Model", "Quantization", "Only when the checkpoint needs telling; vLLM reads it from "
                              "the model's config otherwise. AWQ / GPTQ checkpoints fit 4-bit models on small cards."
                              + b("--quantization"), "select", QUANTS),
        "VL_DTYPE": _m("Model", "Compute type", "<b>auto</b> takes the model's; cards without bfloat16 (NVIDIA "
                       "before Ampere, e.g. an RTX 2060) run <b>float16</b>." + b("--dtype"), "select", DTYPES),
        "VL_TRUST_REMOTE_CODE": _m("Model", "Trust remote code", "Runs Python shipped inside the model repository. "
                                   "Only for models you trust; most do not need it." + b("--trust-remote-code"),
                                   "select", ["0", "1"], danger=True),
        "BACKEND": _m("Model", "Runtime", "<b>cuda</b>: NVIDIA (~/vllm/venv-cu). <b>rocm</b>: AMD (~/vllm/venv-rocm)."
                      " Install with <code>bash install-vllm.sh</code> (add <code>--rocm</code> for AMD).",
                      "select", BACKENDS, strict=True),
        "VL_VENV": _m("Model", "Runtime folder", "The venv that has vLLM. Empty: the one for the runtime above."),
        "VL_GPU_MEM_UTIL": _m("Memory", "Card memory share", "The fraction of the card vLLM takes at start: weights, "
                              "activations and the paged KV pool. The card must be free: other servers on it would "
                              "run out." + b("--gpu-memory-utilization")),
        "VL_MAX_MODEL_LEN": _m("Memory", "Context per request", "Longest prompt + answer one request may reach. 0: "
                               "the model's own limit, which must fit the KV pool." + b("--max-model-len"), "int",
                               numeric=True, unit="tokens"),
        "VL_KV_CACHE_DTYPE": _m("Memory", "KV cache type", "<b>fp8</b> doubles the requests the pool holds, at a "
                                "small quality cost; needs a card vLLM supports it on." + b("--kv-cache-dtype"),
                                "select", KV_DTYPES),
        "VL_ENFORCE_EAGER": _m("Memory", "Eager mode", "<b>1</b>: no CUDA graphs. Slower per step but saves the "
                               "memory graph capture takes: useful on small cards." + b("--enforce-eager"),
                               "select", ["0", "1"]),
        "VL_MAX_NUM_SEQS": _m("Scheduler", "Requests at once", "The most requests scheduled in one step: the "
                              "concurrency vLLM batches. More needs more KV pool." + b("--max-num-seqs"), "int",
                              numeric=True),
        "VL_MAX_NUM_BATCHED_TOKENS": _m("Scheduler", "Tokens per step", "Budget of tokens one step processes "
                                        "(prefill chunks + decodes). 0: vLLM's default." +
                                        b("--max-num-batched-tokens"), "int", numeric=True),
        "VL_PREFIX_CACHING": _m("Scheduler", "Prefix caching", "Requests that share a prompt prefix share its KV "
                                "blocks: agent turns and system prompts are not recomputed." +
                                b("--enable-prefix-caching"), "select", ["1", "0"]),
        "VL_TP": _m("Scheduler", "Tensor parallel", "Cards one model is split across (same vendor)." +
                    b("--tensor-parallel-size"), "int", numeric=True),
        "VL_EXTRA": _m("Scheduler", "Other options", "More <code>vllm serve</code> options, as on its command line. "
                       "The ones set above, and address, key and file-access options, are refused."),
        "PORT": _m("Server", "Port", "The OpenAI-compatible API." + b("--port"), "int", numeric=True),
        "HOST": _m("Server", "Listen address", "<b>127.0.0.1</b>: this machine only (the gateway can still serve "
                   "it). <b>0.0.0.0</b>: the LAN; then an API key is required." + b("--host"), "select",
                   ["127.0.0.1", "0.0.0.0"]),
        "VL_API_KEY": _m("Server", "API key", "Stored in a 0600 file and handed to the server in its environment "
                         "(VLLM_API_KEY), never on the command line or in logs. Shown as ******** once set; "
                         "clear the box to remove it."),
        "RAM_FLOOR_MB": _m("Server", "Host RAM floor", "The launcher stops the server if free host RAM falls below "
                           "this.", "int", unit="MB", numeric=True),
    }


# ============================================================================
# settings
# ============================================================================
def _key_file(inst):
    return inst["dir"] / "vllm-api-key"


def api_key(inst):
    try:
        return _key_file(inst).read_text().strip() or None
    except OSError:
        return None


def served_name(v):
    """The name the server answers to (--served-model-name). vLLM refuses a request whose "model"
    is anything else, so the gateway sends exactly this."""
    model = str(v.get("VL_MODEL") or "")
    return str(v.get("VL_SERVED_NAME") or "") or (Path(model).name if model.startswith("/") else model)


def _coerce(vals):
    for k, v in list(vals.items()):
        if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
            vals[k] = int(v)
    return vals


def load_params(inst):
    cur = dict(DEFAULTS)
    cur.update(P._read_env_file(inst["dir"] / "params.env"))
    cur["VL_API_KEY"] = MASK if api_key(inst) else ""
    return _coerce(cur)


def write_params(inst, vals):
    body = ["# GENERATED by the admin panel - do not hand-edit.",
            f"# vLLM instance '{inst['id']}'. Read by panel/instance_launch.py. The API key is in vllm-api-key.", ""]
    body += [f"{k}={P._env_quote('' if k == 'VL_API_KEY' else vals.get(k, DEFAULTS[k]))}" for k in DEFAULTS]
    pf = inst["dir"] / "params.env"
    pf.write_text("\n".join(body) + "\n")
    os.chmod(pf, 0o600)
    key = str(vals.get("VL_API_KEY") or "")
    kf = _key_file(inst)
    if key and key != MASK:
        fd = os.open(kf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key + "\n")
        os.chmod(kf, 0o600)
    elif not key and kf.exists():
        kf.unlink()


def extra_args(v):
    """VL_EXTRA as argv tokens; refuses flags this module manages and anything not flag-shaped."""
    s = str(v.get("VL_EXTRA") or "").strip()
    if not s:
        return []
    if "\n" in s:
        raise ValueError("Other options: one line")
    try:
        toks = shlex.split(s)
    except ValueError as e:
        raise ValueError(f"Other options: {e}")
    for t in toks:
        if t.startswith("-"):
            flag = t.split("=", 1)[0]
            if not re.fullmatch(r"--[a-z0-9][a-z0-9-]*", flag):
                raise ValueError(f"Other options: {flag!r} is not a vllm serve option")
            if flag in MANAGED:
                raise ValueError(f"Other options: {flag} is set by its own field (or refused)")
    return toks


def save_params(inst, new):
    cur = load_params(inst)
    unknown = sorted(set(new) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"not vLLM settings: {', '.join(unknown)}")
    cur = _coerce(dict(cur, **new))
    if cur["BACKEND"] not in BACKENDS:
        raise ValueError(f"BACKEND must be one of {', '.join(BACKENDS)}")
    try:
        port = int(cur["PORT"])
        assert 1024 <= port <= 65535 and port not in (8090, 8091, 8092)
    except (ValueError, AssertionError):
        raise ValueError("PORT must be 1024-65535 and not the panel's own ports")
    if str(cur["HOST"]) not in ("127.0.0.1", "0.0.0.0"):
        raise ValueError("HOST: 127.0.0.1 or 0.0.0.0")
    try:
        u = float(cur["VL_GPU_MEM_UTIL"])
        assert 0.1 <= u <= 0.98
    except (ValueError, AssertionError):
        raise ValueError("Card memory share: 0.1 to 0.98")
    for k in ("VL_MAX_MODEL_LEN", "VL_MAX_NUM_SEQS", "VL_MAX_NUM_BATCHED_TOKENS", "VL_TP", "RAM_FLOOR_MB"):
        if not str(cur[k]).isdigit():
            raise ValueError(f"{k}: a whole number, 0 or more")
    if int(cur["VL_MAX_NUM_SEQS"]) < 1 or int(cur["VL_TP"]) < 1:
        raise ValueError("Requests at once and Tensor parallel start at 1")
    for k, allowed in (("VL_DTYPE", DTYPES), ("VL_KV_CACHE_DTYPE", KV_DTYPES), ("VL_QUANTIZATION", QUANTS),
                       ("VL_ENFORCE_EAGER", ["0", "1"]), ("VL_PREFIX_CACHING", ["0", "1"]),
                       ("VL_TRUST_REMOTE_CODE", ["0", "1"])):
        if str(cur[k]) not in allowed:
            raise ValueError(f"{k}: one of {', '.join(a or '(empty)' for a in allowed)}")
    m = str(cur["VL_MODEL"] or "")
    if m and not (m.startswith("/") or re.fullmatch(r"[\w.-]+/[\w.-]+", m)):
        raise ValueError("Model: an absolute folder path or a Hugging Face repo id (org/name)")
    if cur["VL_SERVED_NAME"] and not re.fullmatch(r"[\w./-]{1,100}", str(cur["VL_SERVED_NAME"])):
        raise ValueError("Model name: letters, digits, '.', '_', '/' and '-'")
    if cur["VL_VENV"] and not str(cur["VL_VENV"]).startswith("/"):
        raise ValueError("Runtime folder: an absolute path")
    extra_args(cur)
    write_params(inst, cur)
    return load_params(inst)


# ============================================================================
# runtime, devices, plan
# ============================================================================
def venv_of(v):
    return Path(str(v.get("VL_VENV") or "") or (P.HOME / "vllm" / ("venv-rocm" if v.get("BACKEND") == "rocm"
                                                                  else "venv-cu")))


def runtime(venv):
    """vLLM / torch versions and the accelerator torch sees (cached 5 min)."""
    c = _rt_cache.get(str(venv))
    if c and time.time() - c[0] < 300:
        return c[1]
    py = Path(venv) / "bin" / "python"
    code = ("import json, importlib.metadata as m, torch\n"
            "print(json.dumps(dict(vllm=m.version('vllm'), torch=torch.__version__, cuda=torch.version.cuda,"
            " hip=getattr(torch.version, 'hip', None), devices=torch.cuda.device_count())))")
    try:
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        out = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else \
            dict(error=(r.stderr or r.stdout).strip()[-300:])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as e:
        out = dict(error=str(e))
    _rt_cache[str(venv)] = (time.time(), out)
    return out


def _norm_pci(a):
    return str(a or "").lower()[-12:]                 # "00000000:07:00.0" -> "0000:07:00.0"


def gpu_index(pci, backend):
    """The index the runtime gives this card, with CUDA_DEVICE_ORDER=PCI_BUS_ID (CUDA) or among
    AMD cards in PCI order (ROCm). None if it is not a card of that vendor."""
    want = _norm_pci(pci)
    if backend == "cuda":
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            ids = [_norm_pci(x.strip()) for x in r.stdout.splitlines() if x.strip()]
        except (OSError, subprocess.TimeoutExpired):
            return None
    else:
        base = Path("/sys/bus/pci/drivers/amdgpu")
        ids = sorted(_norm_pci(p.name) for p in base.iterdir() if re.fullmatch(r"[0-9a-f:.]+", p.name)) \
            if base.is_dir() else []
    return ids.index(want) if want in ids else None


def _card_users(pci, inst):
    """Other running servers on this card (instances of this panel) and, for NVIDIA, any process
    holding memory on it."""
    users = []
    for iid in P.instance_ids():
        if iid == inst["id"]:
            continue
        try:
            other = P.get_instance(iid)
            devs = set(other.get("devices") or [other.get("device")])
            if _norm_pci(pci) not in {_norm_pci(d) for d in devs}:
                continue
            with P.using_instance(other):
                if P.server_pid():
                    users.append(iid)
        except Exception:
            continue
    return users


def _model_size_mib(path):
    p = Path(path)
    if not p.is_dir():
        return 0
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) // 1048576


def launch_plan(inst, values=None):
    v = dict(values or load_params(inst))
    errors, warnings = [], []
    backend = str(v.get("BACKEND") or "cuda")
    venv = venv_of(v)
    vbin = venv / "bin" / "vllm"
    rt = runtime(venv) if vbin.exists() else dict(error=f"{vbin} does not exist")
    if rt.get("error"):
        errors.append(f"no vLLM in {venv}: run  bash install-vllm.sh{' --rocm' if backend == 'rocm' else ''}  "
                      f"({rt['error'][:160]})")
    elif backend == "cuda" and not rt.get("cuda"):
        errors.append(f"the vLLM in {venv} is not a CUDA build")
    elif backend == "rocm" and not rt.get("hip"):
        errors.append(f"the vLLM in {venv} is not a ROCm build")
    model = str(v.get("VL_MODEL") or "")
    local = model.startswith("/")
    if not model:
        errors.append("no model: set Model on the Parameters tab")
    elif local and not (Path(model) / "config.json").is_file():
        errors.append(f"{model} has no config.json: not a Hugging Face model folder"
                      + (" (GGUF files are for llama.cpp instances)" if model.endswith(".gguf") else ""))
    pci = inst.get("device") or ""
    idx = gpu_index(pci, backend) if pci and pci != "cpu" else None
    if idx is None:
        errors.append(f"the instance's device ({pci or 'none'}) is not a{'n NVIDIA' if backend == 'cuda' else 'n AMD'} "
                      f"card; vLLM here needs one (CPU serving is not what vLLM is for)")
    busy = _card_users(pci, inst) if pci else []
    if busy:
        errors.append(f"{pci} is in use by {', '.join(busy)}: vLLM takes {v.get('VL_GPU_MEM_UTIL')} of the card at "
                      "start and the other server would run out")
    try:
        extra = extra_args(v)
    except ValueError as e:
        errors.append(str(e))
        extra = []
    host, port = str(v.get("HOST") or "127.0.0.1"), int(v.get("PORT") or 0)
    key = api_key(inst)
    if host != "127.0.0.1" and not key:
        errors.append("a LAN listener (HOST=0.0.0.0) needs an API key")
    if str(v.get("VL_TRUST_REMOTE_CODE")) == "1":
        warnings.append("trust-remote-code is on: the model repository's Python runs in the server")
    size = _model_size_mib(model) if local else 0
    avail, floor = P._meminfo_mb("MemAvailable"), int(v.get("RAM_FLOOR_MB") or 4096)
    if avail and avail < floor + 2048:
        errors.append(f"only {avail} MiB of host RAM free; vLLM needs a few GB beside the {floor} MiB floor")
    name = served_name(v)
    argv = [str(vbin), "serve", model, "--host", host, "--port", str(port), "--served-model-name", name,
            "--dtype", str(v.get("VL_DTYPE") or "auto"), "--gpu-memory-utilization", str(v.get("VL_GPU_MEM_UTIL")),
            "--max-num-seqs", str(v.get("VL_MAX_NUM_SEQS")), "--tensor-parallel-size", str(v.get("VL_TP") or 1)]
    if int(v.get("VL_MAX_MODEL_LEN") or 0):
        argv += ["--max-model-len", str(v["VL_MAX_MODEL_LEN"])]
    if int(v.get("VL_MAX_NUM_BATCHED_TOKENS") or 0):
        argv += ["--max-num-batched-tokens", str(v["VL_MAX_NUM_BATCHED_TOKENS"])]
    if str(v.get("VL_KV_CACHE_DTYPE") or "auto") != "auto":
        argv += ["--kv-cache-dtype", str(v["VL_KV_CACHE_DTYPE"])]
    if v.get("VL_QUANTIZATION"):
        argv += ["--quantization", str(v["VL_QUANTIZATION"])]
    if str(v.get("VL_ENFORCE_EAGER")) == "1":
        argv += ["--enforce-eager"]
    argv += ["--enable-prefix-caching"] if str(v.get("VL_PREFIX_CACHING")) != "0" else ["--no-enable-prefix-caching"]
    if str(v.get("VL_TRUST_REMOTE_CODE")) == "1":
        argv += ["--trust-remote-code"]
    argv += ["--download-dir", str(P.HOME / "vllm" / "hf-cache")] + extra
    env = dict(VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", PYTHONUNBUFFERED="1",
               HF_HOME=str(P.HOME / "vllm" / "hf-cache"))
    if local:
        env["HF_HUB_OFFLINE"] = "1"
    if backend == "cuda":
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(idx if idx is not None else ""))
    else:
        env.update(HIP_VISIBLE_DEVICES=str(idx if idx is not None else ""))
    dev_name = (P._device_record(pci) or {}).get("name") if pci and hasattr(P, "_device_record") else None
    return dict(engine="vllm", argv=argv, env=env, secret_env_names=["VLLM_API_KEY"] if key else [],
                errors=errors, warnings=warnings, backend=backend, bindir=str(venv / "bin"),
                rundir=str(inst["rundir"]), config=None, config_file=None,
                device=dict(name=dev_name or pci, pci=pci), placement=[f"vllm {backend} gpu{idx}"],
                weights_mib=size, host_ram_mib=0, vulkan_map=[], template_copy=None, api_key=None,
                cmdline=" ".join(argv), runtime=rt, served_name=name)


# ============================================================================
# running server
# ============================================================================
def secrets(inst):
    """Environment the launcher adds at start, after it has written the log. Kept out of the plan,
    because /api/launch-plan returns the plan to the browser."""
    key = api_key(inst)
    return dict(VLLM_API_KEY=key) if key else {}


def _http(host, port, path, timeout=2, key=None, raw=False):
    import urllib.request
    rq = urllib.request.Request(f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}{path}",
                                headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            data = r.read(2 * 1024 * 1024)
            return data.decode(errors="ignore") if raw else (json.loads(data) if data.strip() else {})
    except Exception:
        return None


def _metric(text, *names):
    for n in names:
        m = re.search(rf"^{re.escape(n)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)", text or "", re.M)
        if m:
            return float(m.group(1))
    return None


def _instance_of_pid(pid):
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None
    m = re.search(r"inf01-inst@([\w.-]+)\.service", cg)
    if not m:
        return None
    try:
        return P.get_instance(m.group(1))
    except Exception:
        return None


def describe(pid, argv):
    """A server-list row. vLLM retitles its processes (VLLM::APIServer), so when the command line
    is gone the instance is found by its unit and read from its settings."""
    g = lambda f: P._argv_get(argv, (f,))
    inst = _instance_of_pid(pid)
    v = load_params(inst) if inst and inst.get("engine") == "vllm" else {}
    host = g("--host") or v.get("HOST") or "127.0.0.1"
    port = g("--port") or v.get("PORT")
    key = api_key(inst) if inst else None
    model = (argv[2] if len(argv) > 2 and argv[1:2] == ["serve"] else "") or v.get("VL_MODEL") or ""
    h = _http(host, port, "/health", timeout=1.5, key=key)
    models = _http(host, port, "/v1/models", timeout=1.5, key=key) or {}
    m0 = (models.get("data") or [{}])[0] if isinstance(models, dict) else {}
    met = _http(host, port, "/metrics", timeout=1.5, raw=True) or ""
    running = _metric(met, "vllm:num_requests_running")
    waiting = _metric(met, "vllm:num_requests_waiting")
    kv = _metric(met, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    seqs = g("--max-num-seqs") or v.get("VL_MAX_NUM_SEQS")
    return dict(engine="vllm", model=model, model_name=m0.get("id") or Path(model).name or None, binary=argv[0],
                host=host, port=int(port) if str(port).isdigit() else port, backend=f"vllm {v.get('BACKEND', '')}".strip(),
                health="ok" if h is not None else "loading", ctx=m0.get("max_model_len"), kv=None, ngl=None,
                parallel=str(seqs) if seqs else None, spec=None, mmproj=None, cache_ram=None, draft=None,
                alias=m0.get("id"), batch=None, ubatch=None, device=inst.get("device") if inst else None,
                log_file=None, slots=int(seqs) if str(seqs or "").isdigit() else None,
                slots_busy=int(running) if running is not None else None, live_tps=None, visible_devices={},
                running=running, waiting=waiting, kv_usage=round(kv * 100, 1) if kv is not None else None)


def health(inst):
    v = load_params(inst)
    with P.using_instance(inst):
        pid = P.server_pid()
    if not pid:
        return None
    key = api_key(inst)
    if _http(v.get("HOST"), v.get("PORT"), "/health", key=key) is None:
        return dict(status="loading")
    models = _http(v.get("HOST"), v.get("PORT"), "/v1/models", key=key) or {}
    m0 = (models.get("data") or [{}])[0] if isinstance(models, dict) else {}
    met = _http(v.get("HOST"), v.get("PORT"), "/metrics", raw=True) or ""
    kv = _metric(met, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    return dict(status="ok", model=m0.get("id"), context=m0.get("max_model_len"),
                running=_metric(met, "vllm:num_requests_running"), waiting=_metric(met, "vllm:num_requests_waiting"),
                kv_cache_used_pct=round(kv * 100, 1) if kv is not None else None)


def status(inst=None):
    v = load_params(inst) if inst and inst.get("engine") == "vllm" else DEFAULTS
    out = {}
    for b in BACKENDS:
        venv = venv_of(dict(v, BACKEND=b, VL_VENV=v.get("VL_VENV") if v.get("BACKEND") == b else ""))
        out[b] = dict(venv=str(venv), installed=(venv / "bin" / "vllm").exists(),
                      runtime=runtime(venv) if (venv / "bin" / "vllm").exists() else None)
    return dict(runtimes=out, install="bash install-vllm.sh   (NVIDIA)   |   bash install-vllm.sh --rocm   (AMD)")
