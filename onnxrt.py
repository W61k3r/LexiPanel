#!/usr/bin/env python3
"""
ONNX Runtime GenAI instances (added 2026-09-25): ONNX models (a folder with genai_config.json),
served by onnx_server.py next to this file, on the CPU or an NPU / GPU through ONNX Runtime's
execution providers:
  AMD      VitisAI    Ryzen AI NPU (XDNA; Linux driver amdxdna). Ryzen AI Software supplies the EP.
  Intel    OpenVINO   NPU (Core Ultra; driver intel_vpu), GPU or CPU; device_type picks which.
  Qualcomm QNN        Hexagon NPU (HTP backend) on Snapdragon; backend_path names the QNN library.
  also     cuda (NVIDIA), dml (DirectML, Windows), webgpu.
The runtime lives in its own venv (~/onnxrt/venv; install-onnx.sh), not the system Python.

Parameters are OX_* keys in the instance's params.env. Every generation option here is one the
runtime itself reports (GeneratorParams.get_search_options, onnxruntime-genai 0.16); empty means
"the model's own genai_config.json value". The server sets each at start and reads it back, and
/props says which this build accepted: the suite is re-verified on the machine that runs it.
Execution-provider options are passed through as given; the provider validates them at load.
"""
import glob, json, os, re, subprocess, time
from pathlib import Path

P = None
PROVIDERS = ["cpu", "vitisai", "openvino", "qnn", "cuda", "dml", "webgpu"]
GROUPS = ["Model", "Provider", "Generation", "Session", "Server"]
SEARCH = ["max_length", "min_length", "do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
          "num_beams", "num_return_sequences", "length_penalty", "early_stopping", "no_repeat_ngram_size",
          "diversity_penalty", "past_present_share_buffer", "batch_size", "random_seed", "chunk_size"]
DEFAULTS = dict(OX_MODEL_DIR="", OX_PYTHON="", BACKEND="cpu", OX_DEVICE_TYPE="", OX_QNN_BACKEND="",
                OX_QNN_PERF="", OX_VITIS_CONFIG="", OX_PROVIDER_OPTIONS="", OX_EP_LIBRARY="", OX_THREADS=0,
                **{"OX_" + k.upper(): "" for k in SEARCH},
                PORT=8087, HOST="127.0.0.1", OX_ALIAS="", OX_API_KEY="", RAM_FLOOR_MB=2048)
_rt_cache = {}


def bind(panel_module):
    global P
    P = panel_module


def _m(group, label, tip, kind="str", options=None, **kw):
    return dict(group=group, label=label, tip=tip, type=kind, options=options, **kw)


def _s(label, tip, kind, **kw):
    return _m("Generation", label, tip + "<br><br><span style='opacity:.75'>Empty: the model's genai_config.json "
              "value. Search option, verified at start (Status: /props).</span>", kind, **kw)


def meta():
    b = lambda k: "<br><br><span style='opacity:.75'>" + k + "</span>"
    return {
        "OX_MODEL_DIR": _m("Model", "Model folder",
                           "A folder with <code>genai_config.json</code>, <code>model.onnx</code> (+ .data) and the "
                           "tokenizer files: the ONNX Runtime GenAI format (Olive or the GenAI model builder make "
                           "it; many are on Hugging Face as <i>*-onnx</i>). Build the model for the provider you "
                           "pick: NPU models (Ryzen AI, OpenVINO NPU, QNN) are exported for that NPU."
                           + b("--model-dir")),
        "OX_PYTHON": _m("Model", "Runtime Python",
                        "The Python that has <code>onnxruntime-genai</code> (and the provider's package). Empty: "
                        "~/onnxrt/venv/bin/python from <code>bash install-onnx.sh</code>."),
        "BACKEND": _m("Provider", "Execution provider",
                          "Where it runs. <b>cpu</b> everywhere. <b>vitisai</b>: AMD Ryzen AI NPU (needs Ryzen AI "
                          "Software). <b>openvino</b>: Intel NPU, GPU or CPU (set Device below). <b>qnn</b>: "
                          "Qualcomm Hexagon NPU (set the QNN library). <b>cuda</b>: NVIDIA. <b>dml</b>: DirectML "
                          "(Windows). <b>webgpu</b>: WebGPU." + b("--provider"), "select", PROVIDERS, strict=True),
        "OX_DEVICE_TYPE": _m("Provider", "OpenVINO device",
                             "OpenVINO's <code>device_type</code>: <b>NPU</b>, <b>GPU</b>, <b>CPU</b>, or "
                             "<b>AUTO</b>. Ignored by other providers." + b("provider option device_type"),
                             "select", ["", "NPU", "GPU", "CPU", "AUTO"]),
        "OX_QNN_BACKEND": _m("Provider", "QNN backend library",
                             "QNN's <code>backend_path</code>: <code>libQnnHtp.so</code> (Linux) or "
                             "<code>QnnHtp.dll</code> (Windows) for the Hexagon NPU; <code>libQnnCpu.so</code> for "
                             "QNN's CPU path." + b("provider option backend_path")),
        "OX_QNN_PERF": _m("Provider", "QNN performance mode",
                          "QNN's <code>htp_performance_mode</code>: <b>burst</b> fastest, <b>high_performance</b>, "
                          "<b>balanced</b>, <b>power_saver</b> / <b>low_power_saver</b> for battery. Empty: QNN's "
                          "default." + b("provider option htp_performance_mode"),
                          "select", ["", "burst", "sustained_high_performance", "high_performance", "balanced",
                                     "power_saver", "low_power_saver", "extreme_power_saver"]),
        "OX_VITIS_CONFIG": _m("Provider", "Ryzen AI config file",
                              "VitisAI's <code>config_file</code> (the <code>vaip_config.json</code> Ryzen AI "
                              "Software ships), when your Ryzen AI version asks for one." + b("provider option config_file")),
        "OX_PROVIDER_OPTIONS": _m("Provider", "More provider options",
                                  "Any other option of the chosen provider, as JSON, e.g. "
                                  "<code>{\"cache_dir\": \"/home/admin/onnxrt/cache\"}</code>. Passed as given; the "
                                  "provider rejects what it does not know when the model loads (see the log)."
                                  + b("--provider-options")),
        "OX_EP_LIBRARY": _m("Provider", "Provider plugin library",
                            "For a provider shipped as a plugin: <code>NAME=/path/to/library.so</code>, registered "
                            "before the model loads. Empty for providers built into the runtime." + b("--ep-library")),
        "OX_THREADS": _m("Session", "CPU threads",
                         "ONNX Runtime's <code>intra_op_num_threads</code> for the CPU work (all of it on cpu; "
                         "the parts an NPU does not take otherwise). 0: the runtime decides."
                         + b("session option intra_op_num_threads"), "int", numeric=True),
        "OX_MAX_LENGTH": _s("Max length (prompt + reply)",
                            "Total tokens a conversation may reach. The server caps each request at this or the "
                            "model's context length, whichever is smaller.", "int", numeric=True),
        "OX_MIN_LENGTH": _s("Min length", "Tokens that must exist before end-of-text may end the reply.", "int",
                            numeric=True),
        "OX_DO_SAMPLE": _s("Sample", "On: sample with temperature / top-k / top-p. Off: greedy (always the most "
                           "likely token; deterministic).", "select", options=["", "true", "false"]),
        "OX_TEMPERATURE": _s("Temperature", "Randomness when sampling. Lower is more focused; 0.2-0.7 for code "
                             "and tools, higher for prose.", "float", numeric=True),
        "OX_TOP_K": _s("Top-k", "Sample only among the k most likely tokens. 1 = greedy.", "int", numeric=True),
        "OX_TOP_P": _s("Top-p", "Sample among the smallest set of tokens whose probability adds up to p "
                       "(nucleus sampling).", "float", numeric=True),
        "OX_REPETITION_PENALTY": _s("Repetition penalty", "Above 1 makes already-used tokens less likely; 1 = off.",
                                    "float", numeric=True),
        "OX_NUM_BEAMS": _s("Beams", "Above 1: beam search, keeping that many candidate replies (slower, more "
                           "memory, no sampling). 1 = off.", "int", numeric=True),
        "OX_NUM_RETURN_SEQUENCES": _s("Returned sequences", "Replies generated per request. The server returns "
                                      "the first, so more than 1 only costs time.", "int", numeric=True),
        "OX_LENGTH_PENALTY": _s("Length penalty", "Beam search: above 1 favours longer replies, below 1 shorter.",
                                "float", numeric=True),
        "OX_EARLY_STOPPING": _s("Early stopping", "Beam search: stop once enough finished candidates exist.",
                                "select", options=["", "true", "false"]),
        "OX_NO_REPEAT_NGRAM_SIZE": _s("No-repeat n-gram size", "Forbids repeating any n-gram of this size. 0 = off.",
                                      "int", numeric=True),
        "OX_DIVERSITY_PENALTY": _s("Diversity penalty", "Group beam search: pushes the groups apart. 0 = off.",
                                   "float", numeric=True),
        "OX_PAST_PRESENT_SHARE_BUFFER": _s("Shared KV buffer",
                                           "One KV-cache buffer for past and present: less memory and copying. "
                                           "Only with a model exported for it.", "select", options=["", "true", "false"]),
        "OX_BATCH_SIZE": _s("Batch size", "Sequences per generator. The server answers one request at a time; "
                            "leave 1.", "int", numeric=True),
        "OX_RANDOM_SEED": _s("Random seed", "Fixed seed for reproducible sampling; empty for random. A request's "
                             "<code>seed</code> overrides it.", "int", numeric=True),
        "OX_CHUNK_SIZE": _s("Prefill chunk size", "Feeds a long prompt this many tokens at a time (less peak "
                            "memory). 0 = all at once.", "int", numeric=True),
        "PORT": _m("Server", "Port", "The OpenAI-compatible API (/v1/chat/completions)." + b("--port"), "int",
                   numeric=True),
        "HOST": _m("Server", "Listen address", "<b>127.0.0.1</b>: this machine only. <b>0.0.0.0</b>: the LAN; then "
                   "set an API key (traffic is unencrypted)." + b("--host"), "select", ["127.0.0.1", "0.0.0.0"]),
        "OX_ALIAS": _m("Server", "Model name", "The name clients see in /v1/models. Empty: the folder's name."
                       + b("--alias")),
        "OX_API_KEY": _m("Server", "API key", "Required for a LAN listener. Stored in a 0600 file, passed with "
                         "--api-key-file so it never shows in the process list." + b("--api-key-file")),
        "RAM_FLOOR_MB": _m("Server", "Host RAM floor", "The launcher stops the server if free host RAM falls "
                           "below this.", "int", unit="MB", numeric=True),
    }


def _coerce(vals):
    for k, v in list(vals.items()):
        if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
            vals[k] = int(v)
    return vals


def load_params(inst):
    cur = dict(DEFAULTS)
    cur.update(P._read_env_file(inst["dir"] / "params.env"))
    return _coerce(cur)


def _key_file(inst):
    return inst["dir"] / "onnx-api-key"


def write_params(inst, vals):
    body = ["# GENERATED by the admin panel - do not hand-edit.",
            f"# ONNX Runtime instance '{inst['id']}'. Read by panel/instance_launch.py.", ""]
    body += [f"{k}={P._env_quote(vals.get(k, DEFAULTS[k]))}" for k in DEFAULTS]
    pf = inst["dir"] / "params.env"
    pf.write_text("\n".join(body) + "\n")
    os.chmod(pf, 0o600)
    kf, key = _key_file(inst), str(vals.get("OX_API_KEY") or "")
    if key:
        kf.write_text(key + "\n")
        os.chmod(kf, 0o600)
    elif kf.exists():
        kf.unlink()


def _search(v):
    """OX_* generation values -> {option: typed value}; empty ones left to the model."""
    out = {}
    for k in SEARCH:
        raw = str(v.get("OX_" + k.upper()) or "").strip()
        if raw == "":
            continue
        if k in ("do_sample", "early_stopping", "past_present_share_buffer"):
            if raw.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"OX_{k.upper()} must be true or false")
            out[k] = raw.lower() in ("true", "1")
        elif k in ("temperature", "top_p", "repetition_penalty", "length_penalty", "diversity_penalty"):
            try:
                out[k] = float(raw)
            except ValueError:
                raise ValueError(f"OX_{k.upper()} must be a number")
        else:
            if not re.fullmatch(r"\d{1,9}", raw):
                raise ValueError(f"OX_{k.upper()} must be a whole number")
            out[k] = int(raw)
    for k, lo, hi in (("temperature", 0, 5), ("top_p", 0, 1), ("repetition_penalty", 0.5, 3), ("top_k", 0, 100000),
                      ("num_beams", 1, 64), ("num_return_sequences", 1, 64), ("batch_size", 1, 256)):
        if k in out and not lo <= out[k] <= hi:
            raise ValueError(f"OX_{k.upper()} must be {lo} to {hi}")
    return out


def _provider_options(v):
    pv = str(v.get("BACKEND") or "cpu")
    opts = {}
    raw = str(v.get("OX_PROVIDER_OPTIONS") or "").strip()
    if raw:
        try:
            opts = json.loads(raw)
            assert isinstance(opts, dict)
        except (ValueError, AssertionError):
            raise ValueError("OX_PROVIDER_OPTIONS must be a JSON object")
    if pv == "openvino" and v.get("OX_DEVICE_TYPE"):
        opts["device_type"] = v["OX_DEVICE_TYPE"]
    if pv == "qnn":
        if v.get("OX_QNN_BACKEND"):
            opts["backend_path"] = v["OX_QNN_BACKEND"]
        if v.get("OX_QNN_PERF"):
            opts["htp_performance_mode"] = v["OX_QNN_PERF"]
    if pv == "vitisai" and v.get("OX_VITIS_CONFIG"):
        opts["config_file"] = v["OX_VITIS_CONFIG"]
    if pv == "cpu" and opts:
        raise ValueError("provider options need a provider other than cpu")
    return opts


def save_params(inst, new):
    cur = load_params(inst)
    unknown = sorted(set(new) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"not ONNX Runtime settings: {', '.join(unknown)}")
    cur = _coerce(dict(cur, **new))
    if cur["BACKEND"] not in PROVIDERS:
        raise ValueError(f"BACKEND must be one of {', '.join(PROVIDERS)}")
    try:
        port = int(cur["PORT"])
        assert 1024 <= port <= 65535 and port not in (8090, 8091, 8092)
    except (ValueError, AssertionError):
        raise ValueError("PORT must be 1024-65535 and not the panel's own ports")
    if int(cur["OX_THREADS"]) < 0:
        raise ValueError("OX_THREADS must be 0 or more")
    lib = str(cur.get("OX_EP_LIBRARY") or "")
    if lib and not re.fullmatch(r"[\w.-]+=/\S+", lib):
        raise ValueError("OX_EP_LIBRARY must be NAME=/absolute/path")
    _search(cur)
    _provider_options(cur)
    write_params(inst, cur)
    return cur


def python_of(v):
    return str(v.get("OX_PYTHON") or "") or str(P.HOME / "onnxrt/venv/bin/python")


def runtime(py):
    """What the runtime Python has: version and which providers the build reports (cached 5 min)."""
    c = _rt_cache.get(py)
    if c and time.time() - c[0] < 300:
        return c[1]
    code = ("import json,onnxruntime_genai as og;print(json.dumps(dict(version=og.__version__,"
            "cuda=og.is_cuda_available(),dml=og.is_dml_available(),openvino=og.is_openvino_available(),"
            "qnn=og.is_qnn_available(),webgpu=og.is_webgpu_available())))")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60)
        out = json.loads(r.stdout) if r.returncode == 0 else dict(error=(r.stderr or r.stdout).strip()[-300:])
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        out = dict(error=str(e))
    _rt_cache[py] = (time.time(), out)
    return out


def npus():
    """Neural processors the kernel knows about (/sys/class/accel), and Qualcomm's fastrpc."""
    out = []
    for d in sorted(glob.glob("/sys/class/accel/accel*")):
        try:
            drv = os.path.basename(os.path.realpath(f"{d}/device/driver"))
        except OSError:
            drv = None
        vendor = {"amdxdna": "AMD", "intel_vpu": "Intel", "qaic": "Qualcomm"}.get(drv)
        out.append(dict(node=f"/dev/accel/{os.path.basename(d)}", driver=drv, vendor=vendor,
                        provider={"AMD": "vitisai", "Intel": "openvino", "Qualcomm": "qnn"}.get(vendor)))
    if glob.glob("/dev/fastrpc-*"):
        out.append(dict(node=glob.glob("/dev/fastrpc-*")[0], driver="fastrpc", vendor="Qualcomm", provider="qnn"))
    return out


def launch_plan(inst, values=None):
    v = dict(values or load_params(inst))
    errors, warnings = [], []
    pv = str(v.get("BACKEND") or "cpu")
    py = python_of(v)
    rt = runtime(py) if os.path.exists(py) else dict(error=f"{py} does not exist")
    if rt.get("error"):
        errors.append(f"no ONNX Runtime GenAI in {py}: run  bash install-onnx.sh  ({rt['error'][:160]})")
    elif pv in ("cuda", "dml", "openvino", "qnn", "webgpu") and rt.get(pv) is False and not v.get("OX_EP_LIBRARY"):
        errors.append(f"this onnxruntime-genai build has no {pv} provider: install its package "
                      f"(bash install-onnx.sh --{pv}) or give its plugin library")
    md = str(v.get("OX_MODEL_DIR") or "")
    if not md:
        errors.append("no model folder: set Model folder on the Parameters tab")
    elif not (Path(md) / "genai_config.json").is_file():
        errors.append(f"{md} has no genai_config.json: not an ONNX Runtime GenAI model folder")
    try:
        search, popts = _search(v), _provider_options(v)
    except ValueError as e:
        errors.append(str(e))
        search, popts = {}, {}
    if pv == "openvino" and not popts.get("device_type"):
        warnings.append("OpenVINO without a device: it picks one itself; set NPU to be sure")
    if pv == "qnn" and not popts.get("backend_path"):
        errors.append("QNN needs its backend library (libQnnHtp.so for the NPU)")
    if pv in ("vitisai", "openvino", "qnn") and not any(n["provider"] == pv for n in npus()) \
            and not (pv == "openvino" and popts.get("device_type") in ("CPU", "GPU", "AUTO")):
        warnings.append(f"no NPU for {pv} is visible in /sys/class/accel: is its driver loaded?")
    size = sum(f.stat().st_size for f in Path(md).glob("*") if f.is_file()) // 1048576 if md and Path(md).is_dir() else 0
    avail, floor = P._meminfo_mb("MemAvailable"), int(v.get("RAM_FLOOR_MB") or 2048)
    if avail and size and size + floor > avail:
        errors.append(f"the model needs ~{size} MiB of host RAM; {avail} MiB is free with the {floor} MiB floor")
    host, port = str(v.get("HOST") or "127.0.0.1"), int(v.get("PORT") or 0)
    if host not in ("127.0.0.1", "localhost", "::1") and not v.get("OX_API_KEY"):
        errors.append("a LAN listener (HOST=0.0.0.0) needs an API key")
    so = {"intra_op_num_threads": int(v["OX_THREADS"])} if int(v.get("OX_THREADS") or 0) else {}
    argv = [py, str(Path(__file__).resolve().parent / "onnx_server.py"), "--model-dir", md, "--host", host,
            "--port", str(port), "--provider", pv, "--provider-options", json.dumps(popts),
            "--session-options", json.dumps(so), "--search", json.dumps(search)]
    if v.get("OX_EP_LIBRARY"):
        argv += ["--ep-library", str(v["OX_EP_LIBRARY"])]
    if v.get("OX_ALIAS"):
        argv += ["--alias", str(v["OX_ALIAS"])]
    if v.get("OX_API_KEY"):
        argv += ["--api-key-file", str(_key_file(inst))]
    npu = next((n for n in npus() if n["provider"] == pv), None)
    return dict(engine="onnx", argv=argv, env={}, errors=errors, warnings=warnings, backend=pv,
                bindir=str(Path(py).parent), rundir=str(inst["rundir"]), config=None, config_file=None,
                device=dict(name=(f"{npu['vendor']} NPU ({npu['driver']})" if npu else pv.upper() if pv != "cpu" else "CPU"),
                            pci=npu["node"] if npu else "cpu"),
                placement=[f"onnxruntime {pv}"], weights_mib=size, host_ram_mib=size, vulkan_map=[],
                template_copy=None, api_key=None, cmdline=" ".join(argv), runtime=rt, npus=npus())


def _http(host, port, path, timeout=2, key=None):
    import urllib.request
    rq = urllib.request.Request(f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}{path}",
                                headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception:
        return None


def describe(pid, argv):
    g = lambda f: P._argv_get(argv, (f,))
    host, port = g("--host") or "127.0.0.1", g("--port")
    h = _http(host, port, "/health", timeout=1.5)
    md = g("--model-dir") or ""
    return dict(engine="onnx", model=md, model_name=g("--alias") or Path(md).name or None, binary=argv[0],
                host=host, port=int(port) if str(port).isdigit() else port, backend=g("--provider"),
                health="ok" if h else "loading", ctx=None, kv=None, ngl=None, parallel=None, spec=None,
                mmproj=None, cache_ram=None, draft=None, alias=g("--alias"), batch=None, ubatch=None,
                device=None, log_file=None, slots=None, slots_busy=None, live_tps=None, visible_devices={})


def health(inst):
    v = load_params(inst)
    with P.using_instance(inst):
        pid = P.server_pid()
    if not pid:
        return None
    h = _http(v.get("HOST"), v.get("PORT"), "/health")
    if not h:
        return dict(status="loading")
    pr = _http(v.get("HOST"), v.get("PORT"), "/props", key=v.get("OX_API_KEY") or None) or {}
    ver = pr.get("verified") or {}
    return dict(status="ok", provider=pr.get("provider"), context=pr.get("context_length"),
                verified=sum(1 for x in ver.values() if x.get("ok")), options=len(ver),
                rejected=[k for k, x in ver.items() if not x.get("ok")])


def status(inst=None):
    py = python_of(load_params(inst) if inst and inst.get("engine") == "onnx" else DEFAULTS)
    return dict(runtime=runtime(py) if os.path.exists(py) else dict(error=f"{py} does not exist"), python=py,
                npus=npus(), providers=PROVIDERS)
