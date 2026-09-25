#!/usr/bin/env python3
"""
inf01 inference admin panel - control API.

Stdlib only. This box has no pip, node or docker, and bootstrapping any of
them would be a dependency you have to maintain forever for a personal admin
tool. http.server is enough.

Binds 127.0.0.1 only. Caddy terminates TLS and does auth in front of it.
Never bind this to 0.0.0.0 - it starts processes and downloads files.
"""
import fcntl, hashlib, json, os, re, shlex, shutil, struct, signal, subprocess, sys, threading, time, urllib.error, urllib.request, urllib.parse
import http.server, socketserver
from collections import deque
from pathlib import Path

import hostos                                       # noqa: E402  (Linux/macOS layer)
HOME       = Path.home() if hostos.IS_MAC else Path("/home/smbadmin")
# INF01_PANEL_DIR lets a scratch copy run beside the live panel (on another
# PANEL_PORT) without touching the live panel's files.
PANEL      = Path(os.environ.get("INF01_PANEL_DIR") or HOME / "panel")
LLAMA      = HOME / "llama"
MODELS     = HOME / "models"
LOGS       = HOME / "llama_logs"
SCRIPT     = LLAMA / "run_qwen38_vulkan_inf01.sh"


# ============================================================================
# INSTANCES  (added 2026-09-17)
#
# The panel used to describe exactly one llama-server: one params.env, one
# failure counter, one systemd unit, port 8081, and "the" amdgpu card. Every
# one of those is now resolved through the CURRENT INSTANCE, a thread-local
# set per HTTP request from ?inst=<id> (and per loop iteration in the sampler).
#
#   main    the original server. Keeps every legacy path - panel/params.env,
#           panel/.launch-fails, the inf01-llama SYSTEM unit, the bash launch
#           scripts - so nothing about the production service moves.
#   <id>    panel/instances/<id>/ holds the same set of files, launched by
#           panel/instance_launch.py under the USER unit inf01-inst@<id>,
#           pinned to one PCI device.
#
# Module-level names that used to be constants (PARAMS_ENV, FAIL_FILE, CARD,
# LAUNCH_OUT, CALIB_FILE, _samples, _live) are now proxies that resolve on
# every access, the same trick _EngineLogPath already used, so the hundreds of
# call sites below needed no change.
# ============================================================================
INSTANCES_DIR = PANEL / "instances"
USER_UNIT_DIR = HOME / ".config/systemd/user"
USER_UNIT = "inf01-inst@.service"
_INST_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
_ctx = threading.local()


def _main_device():
    import glob as _glob
    for d in sorted(_glob.glob("/sys/class/drm/card*/device")):
        try:
            if "DRIVER=amdgpu" in Path(d, "uevent").read_text():
                return os.path.basename(os.path.realpath(d))
        except OSError:
            continue
    return ""


_main_dev_cache = []


def _main_device_cached():
    """Resolved on first use, and retried until found: the panel starts ~0.3 s
    before amdgpu registers its DRM node at boot, so an import-time lookup
    came back "" and main showed no GPU (VRAM total 0) until a panel restart."""
    if not _main_dev_cache:
        dev = _main_device()
        if not dev:
            return ""
        _main_dev_cache.append(dev)
    return _main_dev_cache[0]


def get_running_model(pid):
    """Get the model path from a running llama-server process."""
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
        for i, a in enumerate(argv):
            if a in ("-m", "--model", "--diffusion-model"):
                if i + 1 < len(argv):
                    return argv[i + 1]
    except (OSError, IndexError):
        pass
    return None


def instance_ids():
    ids = ["main"]
    try:
        ids += sorted(d.name for d in INSTANCES_DIR.iterdir()
                      if d.name != "main" and _INST_ID.match(d.name)
                      and (d / "instance.json").is_file())
    except OSError:
        pass
    return ids


MAIN_DEVICES_FILE = PANEL / "main-devices.json"


def _main_devices():
    """main's devices, primary first. panel/main-devices.json when the operator
    has picked some; otherwise the amdgpu card, which is what main ran on before
    devices were selectable - so a box without the file behaves exactly as before."""
    try:
        devs = [str(x) for x in json.loads(MAIN_DEVICES_FILE.read_text()).get("devices") or []
                if x]
        if devs:
            return devs
    except (OSError, ValueError, AttributeError):
        pass
    return [_main_device_cached()]


def get_instance(iid=None):
    iid = iid or "main"
    if iid == "main":
        devs = _main_devices()
        return dict(id="main", name="main", legacy=True, dir=PANEL, device=devs[0],
                    engine="llama.cpp",
                    devices=devs, devices_pinned=MAIN_DEVICES_FILE.exists(),
                    unit="inf01-llama.service", scope="system",
                    launch_out=LOGS / "panel_launch.out",
                    rundir=Path("/dev/shm/llama_qwen38"))
    if not _INST_ID.match(str(iid)):
        raise ValueError(f"bad instance id {iid!r}")
    d = INSTANCES_DIR / iid
    try:
        meta = json.loads((d / "instance.json").read_text())
    except (OSError, ValueError):
        raise ValueError(f"no instance {iid!r}")
    devices = [x for x in (meta.get("devices") or [meta.get("device") or "cpu"]) if x]
    return dict(id=iid, name=meta.get("name") or iid, legacy=False, dir=d,
                engine=meta.get("engine") or "llama.cpp",
                device=devices[0], devices=devices, unit=f"inf01-inst@{iid}.service",
                scope="user", launch_out=LOGS / f"instance_{iid}_launch.out",
                rundir=Path(f"/dev/shm/inf01_{iid}"), created=meta.get("created"))


def INST():
    i = getattr(_ctx, "inst", None)
    return i if i is not None else get_instance("main")


class using_instance:
    """with using_instance("gpu2"): ... - everything inside resolves to it."""
    def __init__(self, iid):
        self.i = iid if isinstance(iid, dict) else get_instance(iid)

    def __enter__(self):
        self.prev = getattr(_ctx, "inst", None)
        _ctx.inst = self.i
        return self.i

    def __exit__(self, *a):
        _ctx.inst = self.prev


class _DynPath:
    """A Path that re-resolves against the current instance on every use."""
    def __init__(self, fn):
        self.__dict__["_fn"] = fn
    def _p(self):              return Path(self._fn())
    def __truediv__(self, o):  return self._p() / o
    def __getattr__(self, n):  return getattr(self._p(), n)
    def __fspath__(self):      return str(self._p())
    def __str__(self):         return str(self._p())
    def __repr__(self):        return f"<instance path {self._p()}>"
    def __eq__(self, o):       return str(self) == str(o)
    def __hash__(self):        return hash(str(self))


class _DynStr:
    """A sysfs directory string that re-resolves per instance. Supports the
    f"{CARD}/x", CARD + "/x" and Path(CARD) forms used throughout."""
    def __init__(self, fn):
        self._fn = fn
    def __str__(self):            return self._fn()
    def __format__(self, spec):   return format(self._fn(), spec)
    def __add__(self, o):         return self._fn() + o
    def __radd__(self, o):        return o + self._fn()
    def __fspath__(self):         return self._fn()


PARAMS_ENV = _DynPath(lambda: INST()["dir"] / "params.env")
def live_cmdline_args():
    """argv of the running llama-server, or [] if nothing is up."""
    pid = server_pid()
    if not pid:
        return []
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except OSError:
        return []


def live_arg(flag, default=None):
    argv = live_cmdline_args()
    return argv[argv.index(flag) + 1] if flag in argv else default


def _live_engine_log():
    """Engine log of whatever is ACTUALLY running.

    Read from the process's own --log-file argument. An earlier version listed
    the known rundirs and picked the newest, which went stale the moment a
    script with a different rundir was used - the same hardcoded-path mistake
    as CARD=card0. The process itself is the only authority on where it logs.
    """
    lf = live_arg("--log-file")
    if lf:
        return Path(lf)
    if not INST()["legacy"]:
        return INST()["rundir"] / "telemetry/engine_debug.log"
    cands = [c for c in Path("/dev/shm").glob("*/telemetry/engine_debug.log")
             if not c.parent.parent.name.startswith("inf01_")]
    if cands:
        return max(cands, key=lambda f: f.stat().st_mtime)
    return Path("/dev/shm/llama_qwen38/telemetry/engine_debug.log")


def live_config():
    """What the RUNNING process was actually launched with.

    params.env is what will be used NEXT start; it is not what is running now.
    They diverge whenever someone edits without restarting, or launches a
    different script by hand - and a panel that shows only the saved values is
    reporting a config nobody is running.
    """
    argv = live_cmdline_args()
    if not argv:
        return dict(running=False, drift=[])
    exe = ""
    try:
        exe = os.path.realpath(f"/proc/{server_pid()}/exe")
    except OSError:
        pass
    backend = "rocm" if "rocm" in exe else ("vulkan" if exe else None)
    cfg = dict(running=True, backend=backend, binary=exe,
               log_file=live_arg("--log-file"),
               ctx=live_arg("-c"), kv=live_arg("--cache-type-k"),
               batch=live_arg("--batch-size"), ubatch=live_arg("--ubatch-size"),
               cache_ram=live_arg("--cache-ram"), ngl=live_arg("--n-gpu-layers"),
               spec=live_arg("--spec-type"), port=live_arg("--port"),
               mmproj=("--mmproj" in argv),
               mmproj_offload=("--mmproj" in argv and "--no-mmproj-offload" not in argv),
               cache_reuse=live_arg("--cache-reuse"),
               no_mmap=("--no-mmap" in argv))
    saved = load_params()
    drift = []
    for key, sk in (("ctx", "CTX"), ("kv", "KV_TYPE"), ("batch", "BATCH"),
                    ("ubatch", "UBATCH"), ("cache_ram", "CACHE_RAM"),
                    ("port", "PORT")):
        lv, sv = cfg.get(key), saved.get(sk)
        if lv is not None and str(lv) != str(sv):
            drift.append(f"{sk}: running={lv} saved={sv}")
    if backend and str(saved.get("BACKEND")) != backend:
        drift.append(f"BACKEND: running={backend} saved={saved.get('BACKEND')}")
    cfg["drift"] = drift
    return cfg


# --------------------------------------------------------------------------
# EVERY running llama-server, not just the managed one.
#
# Everything else in this file follows server_pid(), i.e. the FIRST
# llama-server pgrep returns, and talks to 127.0.0.1:8081. With a second model
# up (the RTX 2060, a CPU instance, a hand-started test) that reported one
# server as if it were the only one, and could attribute its port, log and
# numbers to the wrong process. This section discovers all of them from /proc
# and asks each one on its own port.
# --------------------------------------------------------------------------
_srv_prev = {}      # pid -> dict(n_decoded, at), for per-server live t/s

# short and long spellings llama-server accepts for the same option
_ARG_ALIASES = {
    "model": ("-m", "--model"), "ctx": ("-c", "--ctx-size"), "port": ("--port",),
    "host": ("--host",), "ngl": ("-ngl", "--n-gpu-layers", "--gpu-layers"),
    "kv_k": ("-ctk", "--cache-type-k"), "kv_v": ("-ctv", "--cache-type-v"),
    "parallel": ("-np", "--parallel"), "device": ("-dev", "--device"),
    "mmproj": ("-mm", "--mmproj"), "spec": ("--spec-type",),
    "draft": ("-md", "--model-draft", "--spec-draft-model"),
    "log_file": ("--log-file",), "alias": ("-a", "--alias"),
    "batch": ("-b", "--batch-size"), "ubatch": ("-ub", "--ubatch-size"),
    "cache_ram": ("-cram", "--cache-ram"),
}


def _argv_get(argv, names):
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def server_pids():
    # sd-server, audiocpp_server and camelid too: those instances are servers of this panel
    out = sh("pgrep -x 'llama-server|sd-server|audiocpp_server|camelid|onnx-server'")
    return sorted(int(x) for x in out.split() if x.isdigit())


def _proc_uptime_s(pid):
    return hostos.proc_uptime_s(pid)


def _proc_gpu_mem(pid):
    """Per-device resident memory for one process, from DRM fdinfo.

    Keyed by PCI address, so a server on the 2060 and one on the 7900 XTX are
    told apart. Deduplicated by drm-client-id: dup'd fds share a client and
    would otherwise be counted twice.
    """
    devs, seen = {}, set()
    try:
        fds = list(Path(f"/proc/{pid}/fdinfo").iterdir())
    except OSError:
        return []
    for fd in fds:
        try:
            txt = fd.read_text()
        except OSError:
            continue
        if "drm-driver:" not in txt:
            continue
        kv = {}
        for ln in txt.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
        key = (kv.get("drm-pdev"), kv.get("drm-client-id"))
        if key in seen:
            continue
        seen.add(key)
        d = devs.setdefault(kv.get("drm-pdev") or "?",
                            dict(pdev=kv.get("drm-pdev"), driver=kv.get("drm-driver"),
                                 vram_mib=0, gtt_mib=0))

        def kib(field):
            v = kv.get(field, "").split()
            return int(v[0]) // 1024 if v and v[0].isdigit() else 0
        d["vram_mib"] += kib("drm-resident-vram") or kib("drm-memory-vram")
        d["gtt_mib"] += kib("drm-resident-gtt") or kib("drm-memory-gtt")
    return [d for d in devs.values() if d["vram_mib"] or d["gtt_mib"]]


def _server_http(host, port, path, timeout=1.5):
    h = "127.0.0.1" if host in (None, "", "0.0.0.0", "::", "localhost") else host
    try:
        with urllib.request.urlopen(f"http://{h}:{port}{path}", timeout=timeout) as r:
            body = r.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body
    except urllib.error.HTTPError as e:
        return dict(_http_error=e.code)
    except Exception:
        return None


def _mac_unit_of(pid):
    """macOS: which instance's launchd agent owns this server process, in the
    Linux unit-name shape ("inf01-inst@<id>.service") the callers parse."""
    for iid in instance_ids():
        if iid == "main":
            continue
        _st, lp = hostos.la_state(iid)
        if lp and (pid == lp or pid in hostos.children(lp)):
            return f"inf01-inst@{iid}.service"
    return None


def _describe_server(pid):
    try:
        argv = hostos.proc_argv(pid)
    except OSError:
        return None
    if Path(argv[0]).name == "sd-server":
        return _describe_sd_server(pid, argv)
    if Path(argv[0]).name == "audiocpp_server":
        return _describe_sd_server(pid, argv, audiocpp)
    if Path(argv[0]).name == "camelid":
        return _describe_sd_server(pid, argv, camelid)
    if len(argv) > 1 and Path(argv[1]).name == "onnx_server.py":
        return _describe_sd_server(pid, argv, onnxrt)
    try:
        exe = hostos.proc_exe(pid)
    except OSError:
        exe = argv[0] if argv else ""
    a = {k: _argv_get(argv, v) for k, v in _ARG_ALIASES.items()}
    env = {}
    try:
        if hostos.IS_MAC:
            pairs = hostos.proc_environ(pid).items()
        else:
            pairs = (kv.partition("=")[::2] for kv in
                     Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0"))
        for k, v in pairs:
            if k.startswith(("GGML_", "VK_", "CUDA_", "HIP_", "ROCR_", "HSA_")):
                env[k] = v
    except OSError:
        env = None                      # another user's process: not readable
    rss_mb = hostos.proc_rss_mb(pid)
    if hostos.IS_MAC:
        unit = _mac_unit_of(pid)
    else:
        try:
            cg = Path(f"/proc/{pid}/cgroup").read_text()
            # A user unit's cgroup path is .../user@1000.service/app.slice/
            # inf01-inst@x.service: the LAST service is the one that owns the
            # process. Taking the first reported every instance as user@1000.
            units = re.findall(r"([\w@.-]+\.service)", cg)
            unit = units[-1] if units else None
        except OSError:
            unit = None

    low = exe.lower()
    backend = next((b for b in ("rocm", "vulkan", "cuda", "metal", "cpu") if b in low), None)
    port = a["port"] or "8080"
    host = a["host"] or "127.0.0.1"
    ngl = a["ngl"]

    health = _server_http(host, port, "/health")
    slots = _server_http(host, port, "/slots")
    busy = n_slots = None
    tps = None
    if isinstance(slots, list):
        n_slots = len(slots)
        busy = sum(1 for s in slots if s.get("is_processing"))
        nd = 0
        for s in slots:
            nt = s.get("next_token")
            if isinstance(nt, list) and nt:
                nd += nt[0].get("n_decoded", 0) or 0
        now, prev = time.time(), _srv_prev.get(pid)
        if busy and prev and nd > prev["n_decoded"] and now - prev["at"] > 0.2:
            tps = round((nd - prev["n_decoded"]) / (now - prev["at"]), 2)
        _srv_prev[pid] = dict(n_decoded=nd, at=now)

    if isinstance(health, dict) and health.get("status") == "ok":
        state = "ok"
    elif isinstance(health, dict) and health.get("_http_error") == 503:
        state = "loading"
    elif health is None:
        state = "no answer"
    else:
        state = str((health or {}).get("status") or (health or {}).get("_http_error") or "?")

    # The two traps that fail SILENTLY, checked per process, because a second
    # server started by hand never went through save_params' guards.
    warnings = []
    d = a["draft"]
    if d and os.path.exists(d) and os.path.getsize(d) > DRAFT_MAX_BYTES:
        warnings.append(f"draft model is {os.path.getsize(d) // 1048576} MiB: a full-size "
                        "second copy, not a draft. This has hard-locked the host.")
    if backend == "vulkan" and env is not None and env.get("GGML_VK_ALLOW_SYSMEM_FALLBACK") != "0":
        warnings.append("GGML_VK_ALLOW_SYSMEM_FALLBACK is not 0: the model can be served "
                        "from host RAM at ~1/20th speed with no error.")
    if host in ("0.0.0.0", "::"):
        warnings.append(f"bound to {host}:{port} with no auth in front of it.")

    gpu = _proc_gpu_mem(pid)
    model = a["model"] or ""
    return dict(
        pid=pid, model=model, model_name=Path(model).name if model else None,
        alias=a["alias"], binary=exe, backend=backend, host=host, port=int(port) if str(port).isdigit() else port,
        ctx=a["ctx"], kv=(a["kv_k"] if a["kv_k"] == a["kv_v"] or not a["kv_v"]
                          else f"{a['kv_k']}/{a['kv_v']}"),
        batch=a["batch"], ubatch=a["ubatch"], ngl=ngl, parallel=a["parallel"],
        device=a["device"], spec=a["spec"], draft=a["draft"],
        mmproj=a["mmproj"], log_file=a["log_file"], cache_ram=a["cache_ram"],
        visible_devices={k: v for k, v in (env or {}).items() if k.endswith("VISIBLE_DEVICES")},
        uptime_s=_proc_uptime_s(pid), rss_mb=rss_mb, gpu=gpu,
        vram_mib=sum(g["vram_mib"] for g in gpu), gtt_mib=sum(g["gtt_mib"] for g in gpu),
        health=state, slots=n_slots, slots_busy=busy, live_tps=tps,
        unit=unit, managed=(unit == "inf01-llama.service" or
                            bool(unit and unit.startswith("inf01-inst@"))),
        instance=("main" if unit == "inf01-llama.service" else
                  (unit[len("inf01-inst@"):-len(".service")]
                   if unit and unit.startswith("inf01-inst@") else None)),
        warnings=warnings, argv=argv)


def _describe_sd_server(pid, argv, mod=None):
    """/api/servers row for a non-llama.cpp server (sd-server, audiocpp_server)."""
    rec = (mod or sdcpp).describe(pid, argv)
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text()
        units = re.findall(r"([\w@.-]+\.service)", cg)
        unit = units[-1] if units else None
    except OSError:
        unit = None
    try:
        rss_mb = int(re.search(r"VmRSS:\s+(\d+)", Path(f"/proc/{pid}/status").read_text()).group(1)) // 1024
    except Exception:
        rss_mb = None
    gpu = _proc_gpu_mem(pid)
    warnings = []
    if rec["host"] in ("0.0.0.0", "::"):
        warnings.append(f"bound to {rec['host']}:{rec['port']} with no auth in front of it.")
    rec.update(pid=pid, uptime_s=_proc_uptime_s(pid), rss_mb=rss_mb, gpu=gpu,
               vram_mib=sum(g["vram_mib"] for g in gpu), gtt_mib=sum(g["gtt_mib"] for g in gpu),
               unit=unit, managed=bool(unit and unit.startswith("inf01-inst@")),
               instance=(unit[len("inf01-inst@"):-len(".service")]
                         if unit and unit.startswith("inf01-inst@") else None),
               warnings=warnings, argv=argv)
    return rec


def list_servers():
    """All running llama-server processes, queried in parallel.

    Parallel because each is asked over HTTP, and a server that is mid-load or
    wedged would otherwise add its full timeout to /api/status per server.
    """
    pids = server_pids()
    for gone in set(_srv_prev) - set(pids):
        _srv_prev.pop(gone, None)
    res = {}

    def one(p):
        try:
            res[p] = _describe_server(p)
        except Exception as e:
            res[p] = dict(pid=p, error=str(e), warnings=[])
    ts = [threading.Thread(target=one, args=(p,), daemon=True) for p in pids]
    for t in ts:
        t.start()
    for t in ts:
        t.join(6)
    out = [res[p] for p in pids if res.get(p)]
    return sorted(out, key=lambda s: (not s.get("managed"), str(s.get("port"))))


class _EngineLogPath:
    """Resolves on every access, so a backend switch needs no panel restart."""
    def __truediv__(self, other):  return _live_engine_log() / other
    def __getattr__(self, name):   return getattr(_live_engine_log(), name)
    def __fspath__(self):          return str(_live_engine_log())
    def __str__(self):             return str(_live_engine_log())

ENGINE_LOG = _EngineLogPath()
LAUNCH_OUT = _DynPath(lambda: INST()["launch_out"])
def _find_amdgpu_card():
    """Resolve the amdgpu node by DRIVER, never by card number.

    card0 does not exist on this box any more - it was the simple-framebuffer
    handed off to amdgpu at boot - and the numbering moves across boots
    (currently card1=i915 Intel iGPU, card2=amdgpu). Hardcoding card0 made
    every VRAM/GTT reading here silently return 0.
    """
    import glob as _glob
    for d in sorted(_glob.glob("/sys/class/drm/card*/device")):
        try:
            if "DRIVER=amdgpu" in Path(d, "uevent").read_text():
                return d
        except OSError:
            continue
    return "/sys/class/drm/card0/device"      # last-resort, keeps old behaviour

def _inst_card():
    """sysfs dir of the current instance's device. A CPU instance, or a device
    that has gone away, resolves to a path that does not exist, so every
    read_int() on it returns its default instead of another card's numbers."""
    dev = INST().get("device") or ""
    if dev and dev != "cpu" and os.path.isdir(f"/sys/bus/pci/devices/{dev}"):
        return f"/sys/bus/pci/devices/{dev}"
    return "/nonexistent-device"


CARD       = _DynStr(_inst_card)


def api_base():
    """The current instance's own server: the port it is ACTUALLY listening on
    when up, else the port it is configured for."""
    pid = server_pid()
    port = None
    if pid:
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            port = _argv_get(argv, ("--port", "--listen-port"))
        except OSError:
            pass
    if not port:
        port = _read_env_file(PARAMS_ENV).get("PORT") or DEFAULTS["PORT"]
    return f"http://127.0.0.1:{port}"
BIND = os.environ.get("PANEL_BIND", "127.0.0.1")
PORT = int(os.environ.get("PANEL_PORT", "8090"))

DEFAULTS = dict(BACKEND="vulkan",
                CTX=131072, BATCH=512, UBATCH=512, NGL=99, KV_TYPE="q8_0",
                CACHE_RAM=4096, THREADS=4, PORT=8081, SPEC_TYPE="draft-mtp",
                SPEC_N_MAX=2, SPEC_DRAFT_MODEL="", USE_MMPROJ=1, N_PREDICT=16000,
                MMPROJ_OFFLOAD=0, CACHE_REUSE=256, RAM_FLOOR_MB=512,
                # ROCm/HIP tuning. Empty = leave the variable UNSET, which is
                # stock. These are read by the HIP backend, not by the CLI -
                # the ROCm build has an identical flag surface to the Vulkan
                # one, so every backend difference lives in the environment.
                GGML_CUDA_REGISTER_HOST="", GGML_CUDA_NO_PINNED="",
                GGML_CUDA_DISABLE_GRAPHS="", GGML_CUDA_DISABLE_FUSION="",
                GGML_CUDA_ENABLE_UNIFIED_MEMORY="", GGML_CUDA_GRAPH_OPT="",
                GGML_CUDA_DEVICES="", HSA_ENABLE_SDMA="",
                # Vulkan/RADV tuning. Same convention: empty = UNSET = stock.
                GGML_VK_DISABLE_COOPMAT="", GGML_VK_DISABLE_COOPMAT2="",
                GGML_VK_DISABLE_F16="", GGML_VK_DISABLE_BFLOAT16="",
                GGML_VK_DISABLE_FUSION="", GGML_VK_DISABLE_GRAPH_OPTIMIZE="",
                GGML_VK_DISABLE_MMVQ="", GGML_VK_FORCE_MMVQ="",
                GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="",
                GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM="",
                GGML_VK_PREFER_HOST_MEMORY="", GGML_VK_ENABLE_MEMORY_PRIORITY="",
                GGML_VK_FORCE_MAX_ALLOCATION_SIZE="", GGML_VK_SUBALLOCATION_BLOCK_SIZE="",
                GGML_VK_MAX_NODES_PER_SUBMIT="", GGML_VK_DISABLE_ASYNC="",
                GGML_VK_MEMORY_LOGGER="", GGML_VK_PERF_LOGGER="",
                # Sampling defaults. A client that sends its own temperature /
                # top_p per request overrides these; they only cover requests
                # that don't.
                TEMP="1.0", TOP_P="0.95", TOP_K=20, MIN_P="0.0",
                # Reasoning. This template aliases 'high' to 'xhigh' (its max)
                # and injects a think-harder instruction at that level;
                # 'medium' injects nothing, 'low' instructs brevity.
                # REASONING_BUDGET is a hard token cap, -1 = unrestricted.
                REASONING_EFFORT="medium", REASONING_BUDGET="-1",
                PRESERVE_THINKING="true", MAX_TOOL_RESPONSE_CHARS=8000,
                MODEL=str(MODELS / "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"),
                # The projector is paired with MODEL, not global: each repo ships
                # its own. This was hardcoded in the launch script while MODEL was
                # panel-settable, so changing model silently kept the previous
                # repo's projector. Default stays the script's old value, so
                # nothing moves until it is set deliberately.
                MMPROJ=str(MODELS / "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"),
                # Both template keys are panel-owned - see the CHAT TEMPLATES
                # block. Defaults are the launch script's baked-in pair, so
                # nothing moves until a template is picked deliberately.
                TEMPLATE_SRC=str(MODELS / "qwen3.8-safe-v2.jinja"),
                TEMPLATE_SHA256="4ed3960ba9caa33352f417bc6ac2f6e8358c76b4cbdbced9c59e9e16909f794b")

# A draft larger than this is a full-size target, not a draft. Passing one to
# --spec-draft-model loads a second full copy and has hard-locked this box.
DRAFT_MAX_BYTES = 4_000_000_000

# Backends the launch scripts can dispatch to. "cpu" runs the plain
# ubuntu-x64 build with no GPU at all - slow on a 4-core E-2224G, but it is
# the only backend that still serves while the GPU is being reset, swapped
# or driver-debugged.
BACKENDS = ("vulkan", "rocm", "cuda", "cpu")

_lock      = threading.Lock()
_downloads = {}                 # id -> dict
_samples_by = {}                # instance id -> deque, ~4h of 5s VRAM/GTT samples
_live_by    = {}                # instance id -> live decode-rate state


class _SamplesProxy:
    def _d(self):
        return _samples_by.setdefault(INST()["id"], deque(maxlen=2880))
    def __iter__(self):       return iter(list(self._d()))
    def __len__(self):        return len(self._d())
    def __bool__(self):       return len(self._d()) > 0
    def append(self, x):      self._d().append(x)


class _LiveProxy:
    def get(self, k, d=None): return _live_by.get(INST()["id"], {}).get(k, d)
    def __getitem__(self, k): return _live_by.get(INST()["id"], {})[k]


_samples = _SamplesProxy()
_rel_cache = {}                 # GitHub release list, 10 min TTL


def sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def read_int(p, default=0):
    try:
        return int(Path(p).read_text().strip())
    except Exception:
        return default


def api_get(path, timeout=4):
    try:
        with urllib.request.urlopen(api_base() + path, timeout=timeout) as r:
            body = r.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body
    except Exception:
        return None


# --------------------------------------------------------------------------
# sampler: VRAM/GTT single reads are unreliable (idle eviction moves the whole
# model to GTT and the next request pages it back). Keep a rolling series so
# the UI and the debug bundle can show the pattern instead of one misleading
# number.
# --------------------------------------------------------------------------
_live = _LiveProxy()


def sample_live_tps():
    """Derive a live decode rate from /slots while a request is in flight.

    llama-server only prints t/s once a request finishes, so during a long
    generation the panel would otherwise show a stale number. n_decoded moves
    every token, so differencing it gives a real-time rate.
    """
    iid = INST()["id"]
    prev = _live_by.get(iid) or dict(tps=None, processing=False, n_decoded=0, at=0.0)
    sl = api_get("/slots", timeout=2) if server_pid() else None
    if not isinstance(sl, list) or not sl:
        _live_by[iid] = dict(prev, processing=False, tps=None)
        return
    s0 = sl[0]
    proc = bool(s0.get("is_processing"))
    wl = globals().get("workload")
    if wl is not None:
        try:
            wl.note_slots(iid, sum(1 for x in sl if isinstance(x, dict) and x.get("is_processing")),
                          len(sl), s0.get("n_ctx"))
        except Exception:
            pass
    nd = 0
    nt = s0.get("next_token")
    if isinstance(nt, list) and nt:
        nd = nt[0].get("n_decoded", 0) or 0
    now = time.time()
    tps = None
    if proc and prev.get("processing") and nd > prev.get("n_decoded", 0):
        dt = now - prev.get("at", now)
        if dt > 0.2:
            tps = round((nd - prev["n_decoded"]) / dt, 2)
    _live_by[iid] = dict(tps=tps if proc else None, processing=proc, n_decoded=nd, at=now)


def _sample_one():
    pid = server_pid()
    if os.path.exists(f"{CARD}/mem_info_vram_used"):
        # amdgpu: whole-card counters, as before, so main's history is unchanged
        vram = read_int(f"{CARD}/mem_info_vram_used") // 1048576
        gtt = read_int(f"{CARD}/mem_info_gtt_used") // 1048576
    else:
        # nouveau / nvidia / cpu: no card-wide sysfs counters - use the
        # instance's own process residency from DRM fdinfo
        g = _proc_gpu_mem(pid) if pid else []
        vram = sum(x["vram_mib"] for x in g)
        gtt = sum(x["gtt_mib"] for x in g)
    sample_live_tps()
    _samples.append(dict(
        t=int(time.time()), vram=vram, gtt=gtt,
        busy=read_int(f"{CARD}/gpu_busy_percent", -1),
        tps=_live.get("tps"),
        mem_avail=_meminfo_mb("MemAvailable"),
    ))


def sampler():
    while True:
        for iid in instance_ids():
            try:
                inst = get_instance(iid)
                with using_instance(inst):
                    # an instance that has never run has nothing to chart
                    if iid != "main" and not server_pid() and not _samples:
                        continue
                    _sample_one()
            except Exception:
                pass
        time.sleep(2)


# --------------------------------------------------------------------------
# GPU telemetry.
#
# Everything here comes from sysfs and DRM fdinfo, so it works with no extra
# package installed. amdgpu_top, when present, is folded in as a bonus - it is
# never required. fdinfo is the important part: drm-resident-vram is PER
# PROCESS, so it answers "is llama-server's model actually in VRAM" directly,
# rather than inferring it from a whole-GPU counter that idle eviction moves.
# --------------------------------------------------------------------------
def _hwmon():
    for _h in sorted(Path(CARD + "/hwmon").glob("hwmon*")):
        return str(_h)
    return None

_eng_prev = {}


def _cur_clock(f):
    try:
        for line in Path(f"{CARD}/{f}").read_text().splitlines():
            if "*" in line:
                return line.split(":")[1].replace("*", "").strip()
    except Exception:
        pass
    return None


def _hw(f, div=1.0, nd=0):
    hw = _hwmon()
    if not hw:
        return None
    v = read_int(f"{hw}/{f}", -1)
    if v < 0:
        return None
    return round(v / div, nd) if nd else int(v / div)


def drm_clients():
    """Per-process GPU residency and engine time, from DRM fdinfo."""
    out = []
    for pid_dir in Path("/proc").glob("[0-9]*"):
        pid = pid_dir.name
        seen = False
        agg = dict(vram=0, gtt=0, gfx_ns=0, comp_ns=0)
        try:
            for fd in (pid_dir / "fdinfo").iterdir():
                try:
                    txt = fd.read_text()
                except Exception:
                    continue
                # this instance's device only, whatever its driver
                _dev = INST().get("device") or ""
                if not _dev or _dev == "cpu" or f"drm-pdev:\t{_dev}" not in txt:
                    continue
                seen = True
                for ln in txt.splitlines():
                    if ln.startswith("drm-resident-vram:"):
                        agg["vram"] = max(agg["vram"], int(ln.split()[1]) // 1024)
                    elif ln.startswith("drm-resident-gtt:"):
                        agg["gtt"] = max(agg["gtt"], int(ln.split()[1]) // 1024)
                    elif ln.startswith("drm-engine-gfx:"):
                        agg["gfx_ns"] = max(agg["gfx_ns"], int(ln.split()[1]))
                    elif ln.startswith("drm-engine-compute:"):
                        agg["comp_ns"] = max(agg["comp_ns"], int(ln.split()[1]))
        except Exception:
            continue
        if not seen:
            continue
        now = time.time()
        prev = _eng_prev.get(pid)
        gfx = comp = None
        if prev:
            dt = (now - prev["t"]) * 1e9
            if dt > 0:
                gfx = round(min(100, (agg["gfx_ns"] - prev["gfx"]) / dt * 100), 1)
                comp = round(min(100, (agg["comp_ns"] - prev["comp"]) / dt * 100), 1)
        _eng_prev[pid] = dict(t=now, gfx=agg["gfx_ns"], comp=agg["comp_ns"])
        try:
            name = (pid_dir / "comm").read_text().strip()
        except Exception:
            name = "?"
        out.append(dict(pid=int(pid), name=name, vram_mib=agg["vram"],
                        gtt_mib=agg["gtt"], gfx_pct=gfx, compute_pct=comp))
    return sorted(out, key=lambda x: -x["vram_mib"])


def amdgpu_top_json():
    """One-shot amdgpu_top sample.

    NOTE the -n 1. Without it, '--json' streams forever at the refresh interval
    and never exits, so a plain subprocess call hangs until its timeout and the
    parse fails silently.
    """
    if not sh("command -v amdgpu_top"):
        return None
    raw = sh("amdgpu_top --single --json -n 1 2>/dev/null", timeout=20)
    try:
        d = json.loads(raw)
    except Exception:
        return None
    devs = d.get("devices") or []
    if not devs:
        return None
    dev = devs[0]
    # Keep the useful subset; the full payload is ~19KB and mostly noise here.
    return dict(
        version=d.get("amdgpu_top_version"),
        info=dev.get("Info"),
        vram=dev.get("VRAM"),
        sensors=dev.get("Sensors"),
        gpu_activity=dev.get("gpu_activity"),
        grbm=dev.get("GRBM"),
        fdinfo=dev.get("fdinfo"),
        total_fdinfo=dev.get("Total fdinfo"),
    )


def _nouveau_vram_used(pci):
    """Card-wide VRAM in use on a nouveau device, in MiB, or None. nouveau has no
    mem_info_* sysfs, no hwmon under GSP and no fdinfo memory keys; GETPARAM is
    the one unprivileged source. Needs the render group."""
    try:
        fd = os.open(f"/dev/dri/by-path/pci-{pci}-render", os.O_RDWR | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        # DRM_IOWR(DRM_COMMAND_BASE + DRM_NOUVEAU_GETPARAM, {u64 param; u64 value})
        req = (3 << 30) | (16 << 16) | (ord("d") << 8) | 0x40
        buf = fcntl.ioctl(fd, req, struct.pack("QQ", 19, 0))   # NOUVEAU_GETPARAM_VRAM_USED
        return struct.unpack("QQ", buf)[1] // 1048576
    except OSError:
        return None
    finally:
        os.close(fd)


def gpu_summary(pci):
    """Compact live numbers for ONE device, whatever its driver. gpu_info() is
    built around the instance's first card; a multi-device instance needs the
    same headline numbers for each of the others."""
    rec = _device_record(pci) or dict(pci=pci, name=pci)
    d = f"/sys/bus/pci/devices/{pci}"
    out = dict(pci=pci, name=rec.get("name"), vendor=rec.get("vendor"),
               driver=rec.get("driver"), vram_total_mib=rec.get("vram_total_mib") or 0,
               vram_used_mib=None, gtt_used_mib=None, busy_pct=None, temp_c=None,
               power_w=None, pstate=None,
               link=(sh(f"cat {d}/current_link_speed 2>/dev/null").strip() + " x" +
                     sh(f"cat {d}/current_link_width 2>/dev/null").strip()))
    if os.path.exists(f"{d}/mem_info_vram_total"):
        out["vram_total_mib"] = read_int(f"{d}/mem_info_vram_total") // 1048576
        out["vram_used_mib"] = read_int(f"{d}/mem_info_vram_used") // 1048576
        out["gtt_used_mib"] = read_int(f"{d}/mem_info_gtt_used") // 1048576
        b = read_int(f"{d}/gpu_busy_percent", -1)
        out["busy_pct"] = b if b >= 0 else None
        hw = next(iter(sorted(Path(f"{d}/hwmon").glob("hwmon*"))), None)
        if hw:
            t = read_int(f"{hw}/temp2_input", -1)
            pw = read_int(f"{hw}/power1_average", -1)
            out["temp_c"] = t // 1000 if t >= 0 else None
            out["power_w"] = round(pw / 1e6) if pw >= 0 else None
    elif rec.get("driver") == "nvidia":
        r = _nvidia_smi(pci) or {}
        num = lambda k: (float(r[k]) if str(r.get(k, "")).replace(".", "", 1).isdigit() else None)
        out.update(vram_used_mib=int(num("mem_used_mib") or 0) if r else None,
                   vram_total_mib=int(num("mem_total_mib") or out["vram_total_mib"]),
                   busy_pct=num("util_pct"), temp_c=num("temp_c"),
                   power_w=num("power_w"), pstate=r.get("pstate"))
    elif rec.get("driver") == "nouveau":
        out["vram_used_mib"] = _nouveau_vram_used(pci)
    if out["vram_used_mib"] is not None and out["vram_total_mib"]:
        out["vram_pct"] = round(out["vram_used_mib"] * 100 / out["vram_total_mib"], 1)
    return out


def host_gpus(which="instance"):
    """Live per-card numbers for a GPU view: 'instance' (the current instance's
    devices), 'all' (every inference-capable GPU in the box, whoever uses it),
    or one PCI address. Each card says which instances are set to use it."""
    if which == "instance":
        pcis = [p for p in (INST().get("devices") or []) if p and p != "cpu"]
    else:
        pcis = [d["pci"] for d in gpu_devices(probe=False)
                if d["vendor"] in ("amd", "nvidia") and (which == "all" or d["pci"] == which)]
    users = {}
    for iid in instance_ids():
        try:
            inst = get_instance(iid)
        except ValueError:
            continue
        with using_instance(inst):
            up = server_pid() is not None
        for i, p in enumerate(inst.get("devices") or []):
            users.setdefault(p, []).append(dict(id=iid, running=up, primary=i == 0))
    out = []
    for p in pcis:
        g = gpu_summary(p)
        g["instances"] = users.get(p, [])
        out.append(g)
    return out


def instance_gpus():
    inst = INST()
    return [gpu_summary(p) for p in (inst.get("devices") or [inst.get("device")])
            if p and p != "cpu"]


def gpu_info():
    vram_u = read_int(f"{CARD}/mem_info_vram_used") // 1048576
    vram_t = read_int(f"{CARD}/mem_info_vram_total") // 1048576
    if not vram_t:
        # nouveau/nvidia expose no card-wide counters: total from the device
        # probe, used from every process's fdinfo on this device
        vram_t = (_device_record(INST().get("device")) or {}).get("vram_total_mib") or 0
        vram_u = sum(c["vram_mib"] for c in drm_clients())
    gtt_u = read_int(f"{CARD}/mem_info_gtt_used") // 1048576
    gtt_t = read_int(f"{CARD}/mem_info_gtt_total") // 1048576
    at = amdgpu_top_json()
    return dict(
        name=(_device_record(INST().get("device")) or {}).get("name") or "no GPU",
        device=_device_record(INST().get("device")),
        nvidia_smi=_nvidia_smi(INST().get("device")),
        vbios=sh(f"cat {CARD}/vbios_version 2>/dev/null"),
        memory=dict(vram_used_mib=vram_u, vram_total_mib=vram_t,
                    vram_pct=round(vram_u * 100 / vram_t, 1) if vram_t else 0,
                    vis_vram_used_mib=read_int(f"{CARD}/mem_info_vis_vram_used") // 1048576,
                    vis_vram_total_mib=read_int(f"{CARD}/mem_info_vis_vram_total") // 1048576,
                    gtt_used_mib=gtt_u, gtt_total_mib=gtt_t,
                    gtt_pct=round(gtt_u * 100 / gtt_t, 1) if gtt_t else 0),
        activity=dict(gpu_busy_pct=read_int(f"{CARD}/gpu_busy_percent", -1),
                      mem_busy_pct=read_int(f"{CARD}/mem_busy_percent", -1)),
        thermals=dict(edge_c=_hw("temp1_input", 1000), junction_c=_hw("temp2_input", 1000),
                      mem_c=_hw("temp3_input", 1000),
                      power_w=_hw("power1_average", 1e6), power_cap_w=_hw("power1_cap", 1e6),
                      fan_rpm=_hw("fan1_input"), vddgfx_mv=_hw("in0_input")),
        clocks=dict(sclk=_cur_clock("pp_dpm_sclk"), mclk=_cur_clock("pp_dpm_mclk"),
                    fclk=_cur_clock("pp_dpm_fclk")),
        link=dict(speed=sh(f"cat {CARD}/current_link_speed 2>/dev/null"),
                  width=sh(f"cat {CARD}/current_link_width 2>/dev/null")),
        clients=drm_clients(),
        amdgpu_top=at,
        amdgpu_top_available=at is not None,
        devices=instance_gpus(),
    )


_pid_cache = {}


def _find_instance_pid(inst):
    """The llama-server that belongs to this instance.

    First choice is cgroup membership of the instance's unit, which is exact.
    Otherwise (hand-started, or the panel's direct-launch fallback) match the
    configured port - but never claim a process that some OTHER instance's
    unit owns.
    """
    pids = server_pids()
    cg = {}
    for p in pids:
        try:
            cg[p] = Path(f"/proc/{p}/cgroup").read_text()
        except OSError:
            cg[p] = ""
    for p in pids:
        if f"/{inst['unit']}" in cg[p]:
            return p
    port = str(_read_env_file(inst["dir"] / "params.env").get("PORT")
               or (DEFAULTS["PORT"] if inst["legacy"] else ""))
    if not port:
        return None
    for p in pids:
        if "/inf01-llama.service" in cg[p] or "/inf01-inst@" in cg[p]:
            continue
        try:
            argv = Path(f"/proc/{p}/cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue
        addr = _argv_get(argv, ("--addr",))
        if str(_argv_get(argv, ("--port", "--listen-port")) or
               (addr.rpartition(":")[2] if addr else "8080")) == port:
            return p
    return None


def server_pid():
    """PID of the CURRENT INSTANCE's llama-server (not merely the first one)."""
    inst = INST()
    now = time.time()
    c = _pid_cache.get(inst["id"])
    if c and now - c[0] < 1.0 and (c[1] is None or os.path.exists(f"/proc/{c[1]}")):
        return c[1]
    pid = _find_instance_pid(inst)
    _pid_cache[inst["id"]] = (now, pid)
    return pid


def _backend_file(b):
    return INST()["dir"] / f"params-{b}.env"


def active_backend():
    """Backend the scripts will actually use, i.e. whatever params.env says."""
    try:
        for line in PARAMS_ENV.read_text().splitlines():
            if line.startswith("BACKEND="):
                # same unquote as every other reader: settings files are written
                # single-quoted since 2026-09-23, and a bare .strip('"') left
                # "'vulkan'", which sent load_params() to the defaults
                return _env_unquote(line.split("=", 1)[1]) or "vulkan"
    except OSError:
        pass
    return "vulkan"


# params.env, the tier files and builds.env are SOURCED BY BASH (the run_qwen38
# scripts), so a value is written single-quoted: bash takes it literally - no
# $(...), backticks or $VAR expansion, and a " inside survives. The 2026-09-23
# audit found values written raw into double quotes: JSON in EXTRA_ARGS lost
# its quotes, and $(...) ran at the next start. The reader parses the same way
# bash does, so files written before this change still read back as before.
_ENV_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def _env_quote(v):
    s = "" if v is None else str(v)
    if _ENV_CTRL.search(s):
        raise ValueError("a setting may not contain newlines or control characters")
    return "'" + s.replace("'", "'\\''") + "'"


def _env_unquote(raw):
    raw = raw.strip()
    try:
        parts = shlex.split(raw)
    except ValueError:                  # unbalanced quotes: best effort, as before
        return raw.strip('"')
    return " ".join(parts)


def _read_env_file(path):
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = _env_unquote(v)
    except OSError:
        pass
    return out


def load_params(backend=None):
    """Parameters for one backend.

    Vulkan and ROCm keep SEPARATE files, because a value that is right for one
    is often wrong for the other - cache-ram, batch, and every GGML_VK_*
    vs GGML_CUDA_* knob. Sharing one file meant switching backend silently
    carried the other's tuning across.

    params.env stays the single file the launch scripts source; it is
    regenerated from the active backend's file on every save.
    """
    inst = INST()
    if inst.get("engine") in GEN_ENGINES:
        return GEN_ENGINES[inst["engine"]].load_params(inst)
    if not inst["legacy"]:
        # Instance-specific params
        inst_dir = inst["dir"]
        if backend:
            bf = inst_dir / f"params-{backend}.env"
            if not bf.exists():
                return dict(DEFAULTS)
            cur = dict(DEFAULTS)
            cur.update(_read_env_file(bf))
            for k, v in list(cur.items()):
                if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
                    cur[k] = int(v)
            return cur
        # No backend specified - read from params.env
        pf = inst_dir / "params.env"
        if not pf.exists():
            return dict(DEFAULTS)
        cur = dict(DEFAULTS)
        cur.update(_read_env_file(pf))
        for k, v in list(cur.items()):
            if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
                cur[k] = int(v)
        return cur
    b = backend or active_backend()
    cur = dict(DEFAULTS)
    bf = _backend_file(b)
    if not bf.exists() and PARAMS_ENV.exists():
        # First run after the split: seed both backends from the existing file
        # so nothing is lost and the active config keeps working untouched.
        seed = _read_env_file(PARAMS_ENV)
        for other in BACKENDS:
            of = _backend_file(other)
            if not of.exists():
                merged = dict(DEFAULTS)
                merged.update(seed)
                merged["BACKEND"] = other
                _write_env_file(of, merged)
    cur.update(_read_env_file(bf))
    for k, v in list(cur.items()):
        if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
            cur[k] = int(v)
    cur["BACKEND"] = b
    return cur


def _write_env_file(path, values, header=None):
    body = header or [f"# Written by the admin panel. {Path(path).name}",
                      "# Per-backend parameter set; params.env is generated from the",
                      "# active one. Delete to fall back to the script's defaults.", ""]
    for k in DEFAULTS:
        body.append(f"{k}={_env_quote(values.get(k, DEFAULTS[k]))}")
    Path(path).write_text("\n".join(body) + "\n")


def _reject_full_size_draft(cur):
    d = str(cur.get("SPEC_DRAFT_MODEL", "") or "")
    if d and os.path.exists(d) and os.path.getsize(d) > DRAFT_MAX_BYTES:
        raise ValueError(
            f"SPEC_DRAFT_MODEL is {os.path.getsize(d)//1048576}MB. That is a full-size "
            "target, not a draft: it loads a second full copy of the model and will OOM "
            "the host. Leave it empty for the embedded MTP head, or point it at a small "
            "sidecar such as the 862MB FastMTP file.")


def save_params(new, backend=None):
    inst = INST()
    if inst.get("engine") in GEN_ENGINES:
        return GEN_ENGINES[inst["engine"]].save_params(inst, {k: v for k, v in new.items() if k != "id"})
    if not inst["legacy"]:
        # Instance-specific params
        inst_dir = inst["dir"]
        b = str(new.get("BACKEND") or backend or "vulkan")
        if b not in BACKENDS:
            raise ValueError(f"unknown BACKEND {b!r}")
        bf = inst_dir / f"params-{b}.env"
        cur = dict(DEFAULTS)
        if bf.exists():
            cur.update(_read_env_file(bf))
        for k, v in new.items():
            if k in DEFAULTS:
                cur[k] = v
        cur["BACKEND"] = b
        # Validate and save
        _reject_full_size_draft(cur)
        _write_env_file(bf, cur)
        # Regenerate params.env from the backend just saved
        _write_env_file(inst_dir / "params.env", cur,
                        header=["# GENERATED by the admin panel - do not hand-edit.",
                                f"# Copy of params-{b}.env, the active backend.",
                                f"# Instance '{inst['id']}', read by panel/instance_launch.py.",
                                ""])
        return cur
    b = str(new.get("BACKEND") or backend or active_backend())
    if b not in BACKENDS:
        raise ValueError(f"unknown BACKEND {b!r}")
    cur = load_params(b)
    for k, v in new.items():
        if k in DEFAULTS:
            cur[k] = v
    cur["BACKEND"] = b

    inst = INST()
    for _pci in (inst.get("devices") or [inst["device"]]):
        if _pci and _pci != "cpu":
            _dev = _device_record(_pci)
            if _dev and b not in _dev["backends"]:
                raise ValueError(f"BACKEND={b} cannot drive {_dev['name']} "
                                 f"(possible: {', '.join(_dev['backends'])})")
        elif b != "cpu":
            raise ValueError("this instance has no GPU; BACKEND must be cpu")
    if len(inst.get("devices") or []) > 1 and b != "vulkan":
        raise ValueError("multi-device instances are Vulkan-only for now")
    for _o in instance_ids():
        if _o == inst["id"]:
            continue
        _op = _read_env_file(get_instance(_o)["dir"] / "params.env").get("PORT") \
            or (str(DEFAULTS["PORT"]) if _o == "main" else "")
        if str(_op) == str(cur.get("PORT")):
            # main and flash-next are run one at a time and deliberately share
            # 8081 so clients never change. Only a RUNNING owner blocks the port;
            # launch_plan re-checks at start with ss.
            with using_instance(get_instance(_o)):
                _running = server_pid()
            if _running:
                raise ValueError(f"PORT {cur.get('PORT')} is in use by running instance '{_o}' "
                                 f"(pid {_running}) - stop it first")
    try:
        _extra = shlex.split(str(cur.get("EXTRA_ARGS") or ""))
    except ValueError as e:
        raise ValueError(f"EXTRA_ARGS does not parse: {e}")
    _bad = sorted({a.split("=", 1)[0] for a in _extra} & EXTRA_FORBIDDEN)
    if _bad:
        raise ValueError(f"EXTRA_ARGS may not contain {', '.join(_bad)}")

    _reject_full_size_draft(cur)

    if str(cur.get("USE_MMPROJ", 1)) in ("1", "true", "True"):
        mm = str(cur.get("MMPROJ", "") or "")
        if not mm:
            raise ValueError(
                "MMPROJ is empty while vision is enabled. The launch script aborts on "
                "a missing projector - pick a projector file, or turn Enable vision off.")
        if not os.path.exists(mm):
            raise ValueError(f"MMPROJ does not exist: {mm}")
        _a = gguf_header(mm).get("arch")
        if _a != "clip":
            raise ValueError(
                f"MMPROJ is not a vision projector: {Path(mm).name} reports arch "
                f"{_a!r}, not 'clip'.")

    # The launch script aborts on a template/pin mismatch, so the pin is
    # DERIVED here, never typed: whichever template is selected, params.env
    # carries that file's real hash and the check passes for the right reason.
    ts = str(cur.get("TEMPLATE_SRC", "") or "")
    if ts:
        if not os.path.exists(ts):
            raise ValueError(f"TEMPLATE_SRC does not exist: {ts}")
        _ok, _detail, _ = template_lint(Path(ts).read_text(errors="replace"))
        if not _ok:
            raise ValueError(f"TEMPLATE_SRC {Path(ts).name}: {_detail}")
        cur["TEMPLATE_SHA256"] = _sha256_file(ts)

    budget = ram_budget(cur)
    if budget["verdict"] == "impossible":
        raise ValueError(
            "This configuration cannot fit in host RAM. " + budget["detail"] +
            f" Largest CACHE_RAM that fits: {budget['max_safe_cache_ram_mb']} MiB.")

    PANEL.mkdir(exist_ok=True)
    _write_env_file(_backend_file(b), cur)
    # params.env is what the launch scripts source: regenerate it from the
    # backend just saved, so saving is also how you switch backend.
    _write_env_file(PARAMS_ENV, cur,
                    header=["# GENERATED by the admin panel - do not hand-edit.",
                            f"# Copy of params-{b}.env, the active backend.",
                            ("# Sourced by run_qwen38_*_inf01.sh." if inst["legacy"] else
                             f"# Instance '{inst['id']}', read by panel/instance_launch.py."),
                            ""])
    return cur



# ============================================================================
# OPTION SETS. Since 2026-09-09 these are SUGGESTIONS, not a whitelist: every
# select except SPEC_DRAFT_MODEL renders as a combo box and any typed value is
# saved as-is. The lists still matter - they are the discoverable, known-good
# values - but a build that adds a new kv type or spec type no longer has to
# wait for this file to be updated before it can be used.
#
# Nothing here is validated on save. An unknown enum string fails at model load
# (~25 s) with llama-server's own error, which is a better message than any
# guess this panel could make. The one exception is memory: see ram_budget().
#
# Original note: extracted from `llama-server --help` on build b10798, not from
# memory. Regenerate after a llama.cpp upgrade; upstream adds values (spec-type
# gained the ngram-* and draft-eagle3/dflash/dspark families) and a stale list
# silently hides them.
# ============================================================================
OPT_KV_TYPE   = ["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]
OPT_FLASH_ATTN = ["auto", "on", "off"]
OPT_SPEC_TYPE = ["none", "draft-mtp", "draft-simple", "draft-eagle3", "draft-dflash",
                 "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                 "ngram-mod", "ngram-cache",
                 "draft-dflash,ngram-mod", "draft-mtp,ngram-mod", "draft-simple,ngram-mod"]
OPT_ROPE      = ["", "none", "linear", "yarn"]
OPT_SPLITMODE = ["", "none", "layer", "row", "tensor"]
OPT_POOLING   = ["", "none", "mean", "cls", "last", "rank"]
OPT_REASONING = ["auto", "on", "off"]
OPT_REAS_FMT  = ["auto", "none", "deepseek", "deepseek-legacy"]
OPT_REAS_EFF  = ["default", "minimal", "low", "medium", "high", "xhigh", "max"]
OPT_NUMA      = ["", "distribute", "isolate", "numactl"]
OPT_ONOFF     = ["", "0", "1"]

# Context in 16k blocks to 512k. The model's native max is 262144 - larger
# values are offered because RoPE scaling can extend it, but anything above
# that is extrapolation and the calculator will show it blowing the VRAM budget
# long before you get there.
OPT_CTX       = [str(n * 16384) for n in range(1, 33)]
OPT_SPEC_NMAX = ["1", "2", "3", "4", "5"]
OPT_BATCH     = ["128", "256", "512", "1024", "2048", "4096", "8192"]
OPT_UBATCH    = ["64", "128", "256", "512", "1024", "2048"]
OPT_THREADS   = ["1", "2", "3", "4"]          # E-2224G is 4c/4t
OPT_NGL       = ["0", "16", "32", "48", "64", "80", "99"]
OPT_CACHE_RAM = ["0", "-1", "2048", "4096", "8192", "12288", "16384",
                 "20000", "24576", "32768"]
OPT_PARALLEL  = ["1", "2", "4", "8"]

# ============================================================================
# PARAMETER METADATA - drives the settings UI: grouping, input type, and the
# per-parameter tooltip. Every "cost" note here is either measured on this box
# or cited to the launch-script header; none of it is datasheet guesswork.
# ============================================================================
PARAM_META = {
 "BACKEND": dict(group="Backend", label="Compute backend", type="select",
   options=list(BACKENDS), strict=True,
   tip="Which llama.cpp build serves the model. Each backend picks its own build "
       "in the Builds card - see there for what is actually installed, and never "
       "assume a version from this text. <b>vulkan</b> = RADV, the production path "
       "and the only one with measured numbers on this box. "
       "<b>rocm</b> = HIP/gfx1100, downloaded and script-ready but NOT yet "
       "benchmarked here. Switching needs a restart and reloads the full 15.7 GB "
       "model (~25 s). ROCm additionally needs its runtime installed: "
       "<code>sudo apt install -y libamdhip64-7 librocblas5 libhipblas3</code>. "
       "<b>cpu</b> = the plain ubuntu-x64 build, no GPU at all. On this 4-core "
       "E-2224G expect single-digit t/s on a 27B IQ4_XS - it is a fallback for "
       "when the GPU is unavailable, not a serving path. Each backend keeps its "
       "own build selection; see the Builds card."),

 "CTX": dict(group="Context & memory", label="Context size", type="select",
   options=OPT_CTX, unit="tokens",
   tip="Total KV cache depth in tokens. <b>The single biggest VRAM lever.</b> KV scales "
       "linearly: on this model only 16 of 64 layers carry context-scaling KV (48 are "
       "Gated DeltaNet with a fixed recurrent state), which is why 244736 fits at all. "
       "Native max is 262144. Too high and the card runs near-full, which is what makes "
       "amdgpu evict weights into GTT - host RAM - and that is what OOM'd this box on "
       "2026-09-04."),

 "KV_TYPE": dict(group="Context & memory", label="KV cache quant", type="select",
   options=OPT_KV_TYPE,
   tip="Precision of the K and V caches. Cost is linear in bits: q8_0 is ~1.42x the "
       "bytes of q5_1, f16 is ~2.67x. Lower quant buys context depth and pays in "
       "long-chain consistency - the desktop notes record q4_0 'measurably costing "
       "output quality', which is why q8_0 was the compromise there. Applies to both "
       "K and V."),

 "CACHE_RAM": dict(group="Context & memory", label="Prompt cache (host RAM)", type="select",
   options=OPT_CACHE_RAM, unit="MiB",
   tip="<b>Host RAM</b>, not VRAM. Stores KV snapshots of previous prompts so a "
       "returning conversation restores instead of re-prefilling. At ~165k tokens a "
       "re-prefill costs minutes, so this is load-bearing for long agentic loops. "
       "Budget carefully: this box has 30.67 GB and the launch-script header sizes it "
       "at 8192 ('30G box, model mmap is ~16G'). It is only safe at 20000 if VRAM has "
       "headroom - if the GPU evicts to GTT, that eviction plus this cache is what "
       "exhausts RAM."),

 "CACHE_REUSE": dict(group="Context & memory", label="Cache reuse chunk", type="int", unit="tokens",
   tip="Minimum chunk size to salvage from the prompt cache via KV shifting when the "
       "prefix has drifted, instead of re-prefilling from scratch. 0 = off. 256 is the "
       "conventional value. Requires prompt caching (CACHE_RAM &gt; 0). Aimed squarely "
       "at agentic loops where each turn appends to a long, slightly-shifted prompt."),

 "BATCH": dict(group="Throughput", label="Logical batch (-b)", type="select",
   options=OPT_BATCH, unit="tokens",
   tip="How many prompt tokens are submitted per prefill step. Higher = faster prefill, "
       "little VRAM cost (the physical buffer is set by UBATCH, not this). b10766's own "
       "default is 2048; this box ran 512 for a while, which was throttling prefill for "
       "no memory saving."),

 "UBATCH": dict(group="Throughput", label="Physical batch (-ub)", type="select",
   options=OPT_UBATCH, unit="tokens",
   tip="Physical micro-batch actually evaluated at once. <b>This is the one that sizes "
       "the prefill compute buffer</b>, so it is the VRAM knob and the transient-spike "
       "knob. The desktop script's ITEM 6 device-lost crash on deep-context prefill "
       "lists <code>-ub 256</code> as the first mitigation, explicitly not lowering -b. "
       "Leave at 512 unless chasing a device-lost."),

 "NGL": dict(group="Throughput", label="GPU layers", type="select",
   options=OPT_NGL,
   tip="Layers offloaded to the GPU. 99 = all of them. Anything less puts transformer "
       "layers on a 4-core Xeon E-2224G and collapses throughput. Note that pinning "
       "this makes llama.cpp's auto-fitter abort ('n_gpu_layers already set by user'), "
       "so nothing caps an over-budget config for you - size memory by hand."),

 "THREADS": dict(group="Throughput", label="CPU threads", type="select",
   options=OPT_THREADS,
   tip="CPU threads for the non-offloaded path and sampling. This host is a Xeon "
       "E-2224G: 4 cores, 4 threads, no SMT. Above 4 oversubscribes and hurts. Used "
       "for both --threads and --threads-batch."),

 "SPEC_TYPE": dict(group="Speculative decoding", label="Spec type", type="select",
   options=OPT_SPEC_TYPE,
   tip="<b>draft-mtp</b> uses the model's embedded NextN/MTP head - measured 0.82 draft "
       "acceptance, mean accepted length 2.64, for only +370 MiB. Worth roughly 1.8x on "
       "decode. <b>none</b> disables speculation. There is no reason to turn this off on "
       "this model."),

 "SPEC_N_MAX": dict(group="Speculative decoding", label="Draft depth", type="select",
   options=OPT_SPEC_NMAX,
   tip="How many tokens the draft head proposes per step. At the measured 0.82 "
       "acceptance and mean length 2.64 the depth-2 ceiling is hit nearly every draft, "
       "which is the condition under which depth 3 starts paying. The n=2/3/4 sweep is "
       "still an open item from the desktop script."),

 "SPEC_DRAFT_MODEL": dict(group="Speculative decoding", label="Draft model file", type="select",
   danger=True, strict=False, options=[""],  # replaced at request time by draft_options()
   tip="<b>Leave empty.</b> Empty means the embedded MTP head. Pointing this at the "
       "full-size target loads a SECOND complete 15.7 GB copy plus a second KV cache "
       "and has hard-locked this host twice (2026-09-02). Only a small sidecar is ever "
       "valid. Each option below shows its own VRAM cost.<br><br>"
       "<b>On FastMTP:</b> the 862 MB sidecar cannot be used with the prebuilt "
       "binaries. It carries a <i>trimmed 32k draft vocabulary</i> plus a "
       "<code>d2t</code> remap tensor, and stock llama.cpp hard-asserts the full "
       "248320 vocab - hence <code>expected 5120, 248320, got 5120, 32768</code>. "
       "It needs a source build with HauhauCS-FastMTP-llama.cpp.patch. Merging it "
       "into the target GGUF does NOT help: the blocker is the loader, not the file "
       "layout. Panel and launch script both reject anything over 4 GB."),

 "MMPROJ": dict(group="Vision", label="Projector file", type="select",
   strict=True, options=[""],   # replaced at request time by projector_options()
   tip="The vision projector, which must come from the <b>same repo as the model "
       "in MODEL</b>. Hardcoded in the launch script until 2026-09-15, so switching "
       "model used to keep the old projector. A mismatch does <b>not</b> error: "
       "these Qwen3.8 projectors share a shape (qwen3vl_merger, 5120 projection dim, "
       "334 tensors), so the wrong one loads cleanly and only the image answers are "
       "wrong. Options that do not match MODEL are marked [other build]."),

 "USE_MMPROJ": dict(group="Vision", label="Enable vision", type="bool",
   tip="Loads the BF16 multimodal projector so the server accepts images. 1 = on. "
       "Turning it off entirely frees the most VRAM but loses image input."),

 "MMPROJ_OFFLOAD": dict(group="Vision", label="Projector on GPU", type="bool",
   tip="1 = projector on-card (stock). 0 = <code>--no-mmproj-offload</code>, projector "
       "on the CPU. Off-card returns roughly 1.7 GB of VRAM - ~848 MiB for the "
       "projector plus the ~885 MiB mtmd worst-case buffer the desktop measured - and "
       "removes that buffer as a transient spike. Spikes are what trigger the GTT "
       "eviction. Vision still works; image encoding is slower. Cheap trade if images "
       "are occasional and text is constant."),

 "N_PREDICT": dict(group="Generation", label="Max tokens per reply", type="int",
   tip="Server-side ceiling on generated tokens for one request. A client asking for "
       "more is capped here."),

 "TEMP": dict(group="Sampling", label="Temperature", type="float",
   tip="Only a DEFAULT. Any OpenAI-compatible client that sends its own temperature "
       "overrides it. 1.0 with top_p 0.95 / top_k 20 is Qwen's recommended "
       "thinking-mode pairing."),
 "TOP_P": dict(group="Sampling", label="top_p", type="float",
   tip="Nucleus sampling cutoff. Default only - a client's own value wins. Qwen "
       "recommends 0.95 for thinking mode."),
 "TOP_K": dict(group="Sampling", label="top_k", type="int",
   tip="Consider only the k most likely tokens. Default only. Qwen recommends 20 for "
       "thinking mode."),
 "MIN_P": dict(group="Sampling", label="min_p", type="float",
   tip="Drops tokens below this fraction of the top token's probability. 0.0 disables. "
       "Default only."),

 "REASONING_EFFORT": dict(group="Reasoning", label="Reasoning effort", type="select",
   options=OPT_REAS_EFF,
   tip="A <b>chat-template kwarg, not a server flag</b> - it injects an instruction "
       "string into every prompt. This template aliases <b>high to xhigh</b>, its "
       "maximum; there is no separate middle-high rung. xhigh injects a 333-character "
       "think-harder instruction on every prompt, medium injects nothing, low instructs "
       "brevity. The desktop log records xhigh + preserved thinking as the exact "
       "combination that exhausted a 150-iteration agentic run after five compactions. "
       "medium is the right setting for coding loops."),

 "REASONING_BUDGET": dict(group="Reasoning", label="Reasoning budget", type="int",
   tip="Hard cap on thinking tokens. -1 = unrestricted."),

 "PRESERVE_THINKING": dict(group="Reasoning", label="Preserve thinking", type="bool",
   tip="Keeps each turn's chain-of-thought in the context of every later turn. "
       "<b>Quadratic context growth from reasoning alone</b> - the desktop log blames "
       "this for forcing five compactions in one run, and each compaction rewrites the "
       "prefix and destroys the prefix-cache reuse this server otherwise gets at "
       "0.94-0.99 similarity. Improves multi-turn coherence; expensive on long loops."),

 "MAX_TOOL_RESPONSE_CHARS": dict(group="Reasoning", label="Max tool response chars", type="int",
   tip="Truncates tool results when re-rendering history. Matters MORE when thinking is "
       "preserved, not less, because context is scarcer. The desktop log records a "
       "single grep returning 90k tokens with this unset and detonating the context."),

 "RAM_FLOOR_MB": dict(group="Safety", label="Host RAM floor", type="int", unit="MB",
   danger=True,
   tip="The launch script kills llama-server if MemAvailable drops below this, because "
       "a Linux OOM on a swap-thrashing 30 GB host does not reliably kill the offender "
       "before the machine locks up. 0 disables it. <b>Disabling is not advised:</b> it "
       "was set to 0 on 2026-09-04 and the box took a real global kernel OOM 1h43m "
       "later (llama-server killed, 7.8 GB of 8.19 GB swap consumed)."),

 # --- ROCm / HIP -----------------------------------------------------------
 # Verified against this build by extracting the env-var strings from
 # libggml-hip.so and libhsa-runtime64.so, not from documentation. All are
 # ignored under BACKEND=vulkan. Empty = unset = stock.
 "GGML_CUDA_REGISTER_HOST": dict(group="ROCm (HIP)", label="Register host memory", type="text",
   tip="<b>Prime suspect for the 2026-09-04 ROCm prefill stall.</b> Controls whether ggml "
       "registers host buffers with the GPU (hipHostRegister). That registration is what "
       "the kernel was thrashing on when prefill halved batch-over-batch (27.8 → 14.1 t/s) "
       "while logging <code>amdgpu_amdkfd_restore_userptr_worker hogged CPU</code>. "
       "Set <b>0</b> to disable and see whether prefill recovers. "
       "Vulkan equivalent: none - RADV does not register host memory this way. "
       "Empty = unset = stock."),

 "GGML_CUDA_ENABLE_UNIFIED_MEMORY": dict(group="ROCm (HIP)", label="Unified memory", type="text",
   danger=True,
   tip="<b>The HIP analogue of the Vulkan sysmem-fallback trap.</b> Set it and HIP will "
       "silently back device allocations with host memory over PCIe - the same ~20x "
       "slowdown that cost this box 2.79 t/s prefill under Vulkan, with no error logged. "
       "It is opt-in, so the correct value is <b>empty (unset)</b>. "
       "Translation: <code>GGML_VK_ALLOW_SYSMEM_FALLBACK=0</code> under Vulkan is a guard "
       "you must SET; this is a footgun you must NOT set. Opposite polarity - do not "
       "'convert' one to the other."),

 "GGML_CUDA_NO_PINNED": dict(group="ROCm (HIP)", label="Disable pinned host memory", type="text",
   tip="Set to 1 to stop using page-locked host buffers for transfers. Pinned memory makes "
       "host↔device copies faster but pins pages, which interacts with the same KFD "
       "userptr machinery implicated in the prefill stall. Worth trying alongside "
       "REGISTER_HOST=0. Vulkan equivalent: none. Empty = unset = stock."),

 "GGML_CUDA_DISABLE_GRAPHS": dict(group="ROCm (HIP)", label="Disable HIP graphs", type="text",
   tip="Set to 1 to stop capturing the decode step as a replayable HIP graph. Graphs cut "
       "per-token launch overhead, which matters most at high decode rates; disabling is a "
       "diagnostic for graph-capture bugs, not a tuning win. The server log reports "
       "<code>graphs reused = N</code> so you can see whether they are being hit. "
       "Vulkan equivalent: none. Empty = unset = stock."),

 "GGML_CUDA_GRAPH_OPT": dict(group="ROCm (HIP)", label="HIP graph optimisation", type="text",
   tip="Tunes graph-capture optimisation. Leave empty unless chasing a specific graph "
       "problem; DISABLE_GRAPHS is the blunter and better-understood diagnostic. "
       "Empty = unset = stock."),

 "GGML_CUDA_DISABLE_FUSION": dict(group="ROCm (HIP)", label="Disable kernel fusion", type="text",
   tip="Set to 1 to stop fusing adjacent ops into single kernels. Fusion is normally a "
       "win; disabling it isolates whether a fused kernel is miscompiled or slow on "
       "gfx1100. Diagnostic, not tuning. Empty = unset = stock."),

 "GGML_CUDA_DEVICES": dict(group="ROCm (HIP)", label="Device selection", type="text",
   tip="Which HIP devices ggml may use, e.g. <code>0</code>. "
       "Translation: this is the HIP counterpart of <code>GGML_VK_VISIBLE_DEVICES</code>. "
       "Less critical here than under Vulkan - the Intel UHD P630 cannot enumerate under "
       "HSA at all, so there is no wrong-GPU trap to guard against. The launch script also "
       "sets HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES, which are read by the ROCm "
       "runtime rather than by ggml. Empty = unset = stock."),

 "HSA_ENABLE_SDMA": dict(group="ROCm (HIP)", label="SDMA copy engines", type="text",
   tip="Read by the ROCr runtime, not by ggml. Set to 0 to route host↔device copies "
       "through compute kernels instead of the dedicated SDMA engines. SDMA problems show "
       "up as stalled or extremely slow transfers, which is the shape of the prefill "
       "stall - so 0 is a reasonable third thing to try. Vulkan equivalent: none. "
       "Empty = unset = stock."),

 # --- Vulkan / RADV --------------------------------------------------------
 # Names extracted from libggml-vulkan.so in build b10798 (33 GGML_VK_* vars
 # exist; these are the tuning and diagnostic ones). Ignored under
 # BACKEND=rocm. Empty = unset = stock. Most are DISABLE_ switches: stock is
 # the feature ON, and you set 1 only to test whether it is misbehaving.
 "GGML_VK_DISABLE_COOPMAT": dict(group="Vulkan (RADV)", label="Disable coopmat", type="select",
   options=OPT_ONOFF,
   tip="Cooperative-matrix (matrix core) kernels. <code>--list-devices</code> reports this "
       "card as <b>matrix cores: KHR_coopmat</b>, so they are in use. Set 1 to fall back to "
       "plain shaders - a large expected slowdown, useful only to test whether a coopmat "
       "kernel is miscompiled on RADV NAVI31. ROCm equivalent: none."),
 "GGML_VK_DISABLE_COOPMAT2": dict(group="Vulkan (RADV)", label="Disable coopmat2", type="select",
   options=OPT_ONOFF,
   tip="Disables the NV_cooperative_matrix2 path specifically. This card advertises "
       "KHR_coopmat rather than coopmat2, so setting this should change nothing here - "
       "it is listed for completeness and for diffing against other hardware."),
 "GGML_VK_DISABLE_F16": dict(group="Vulkan (RADV)", label="Disable fp16", type="select",
   options=OPT_ONOFF,
   tip="Forces fp32 maths. <code>--list-devices</code> reports <b>fp16: 1</b> on this card, "
       "so fp16 is active and disabling it costs both speed and VRAM. Diagnostic only."),
 "GGML_VK_DISABLE_BFLOAT16": dict(group="Vulkan (RADV)", label="Disable bf16", type="select",
   options=OPT_ONOFF,
   tip="This card reports <b>bf16: 0</b> - bf16 is already unavailable, so this is a no-op "
       "here. Relevant only on hardware that has it."),
 "GGML_VK_DISABLE_FUSION": dict(group="Vulkan (RADV)", label="Disable kernel fusion", type="select",
   options=OPT_ONOFF,
   tip="Stops fusing adjacent ops into one shader. Fusion is normally a win; disabling "
       "isolates a miscompiled fused kernel. Direct counterpart of ROCm's "
       "<code>GGML_CUDA_DISABLE_FUSION</code>."),
 "GGML_VK_DISABLE_GRAPH_OPTIMIZE": dict(group="Vulkan (RADV)", label="Disable graph optimise", type="select",
   options=OPT_ONOFF,
   tip="Turns off compute-graph reordering. Rough counterpart of ROCm's "
       "<code>GGML_CUDA_GRAPH_OPT</code>. Diagnostic."),
 "GGML_VK_DISABLE_MMVQ": dict(group="Vulkan (RADV)", label="Disable MMVQ", type="select",
   options=OPT_ONOFF,
   tip="Quantised matrix-vector kernels - the hot path for <b>decode</b>, where one token "
       "at a time multiplies against quantised weights. Disabling will hurt decode "
       "noticeably. Pair with FORCE_MMVQ to A/B the same path."),
 "GGML_VK_FORCE_MMVQ": dict(group="Vulkan (RADV)", label="Force MMVQ", type="select",
   options=OPT_ONOFF,
   tip="Forces the quantised matrix-vector path even where heuristics would pick a matrix "
       "kernel. Occasionally a win at batch 1; measure, do not assume."),
 "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": dict(group="Vulkan (RADV)", label="Disable int dot", type="select",
   options=OPT_ONOFF,
   tip="Integer dot-product acceleration for quantised weights. This card reports "
       "<b>int dot: 1</b>, so it is in use and matters for an IQ4_XS model. Diagnostic."),
 "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": dict(group="Vulkan (RADV)", label="Disable host-visible VRAM", type="select",
   options=OPT_ONOFF,
   tip="<b>Directly relevant to this box.</b> Resizable BAR makes all 24560 MiB of VRAM "
       "host-visible (confirmed: vis_vram_total == vram_total). This switch makes ggml "
       "stop using that window. If ReBAR ever regresses in BIOS, behaviour here is worth "
       "comparing. Do not set it casually - host-visible VRAM is what makes uploads fast."),
 "GGML_VK_PREFER_HOST_MEMORY": dict(group="Vulkan (RADV)", label="Prefer host memory", type="select",
   options=OPT_ONOFF,
   tip="<b>Do not set this on this box.</b> It biases allocation towards host RAM, which is "
       "the failure mode that cost 2.79 t/s prefill here. It is the opposite of what you "
       "want; <code>GGML_VK_ALLOW_SYSMEM_FALLBACK=0</code> exists to prevent exactly this."),
 "GGML_VK_ENABLE_MEMORY_PRIORITY": dict(group="Vulkan (RADV)", label="Memory priority", type="select",
   options=OPT_ONOFF,
   tip="Requests VK_EXT_memory_priority so the driver is told these allocations matter. "
       "Plausibly useful against the runtime-suspend eviction seen on this host - it is a "
       "hint to the driver, not a guarantee. Untested here; worth an experiment."),
 "GGML_VK_FORCE_MAX_ALLOCATION_SIZE": dict(group="Vulkan (RADV)", label="Max allocation size", type="text",
   tip="Caps a single Vulkan allocation, in bytes. Some drivers fail on very large single "
       "buffers; splitting can work around that. Empty = driver limit."),
 "GGML_VK_SUBALLOCATION_BLOCK_SIZE": dict(group="Vulkan (RADV)", label="Suballocation block", type="text",
   tip="Block size ggml suballocates within, in bytes. Affects fragmentation on a card "
       "already running near full. Empty = default."),
 "GGML_VK_MAX_NODES_PER_SUBMIT": dict(group="Vulkan (RADV)", label="Max nodes per submit", type="text",
   tip="How many graph nodes go into one command-buffer submission. Larger = less "
       "submission overhead, but a longer single GPU job - and an over-long submission is "
       "what trips <code>ring ... timeout</code>, the desktop script's ITEM 6 device-lost. "
       "Lower it if you see ring timeouts. Empty = default."),
 "GGML_VK_DISABLE_ASYNC": dict(group="Vulkan (RADV)", label="Disable async", type="select",
   options=OPT_ONOFF,
   tip="Serialises submissions instead of overlapping them. Costs throughput; useful to "
       "make a device-lost or ordering bug reproducible."),
 "GGML_VK_MEMORY_LOGGER": dict(group="Vulkan (RADV)", label="Memory logger", type="select",
   options=OPT_ONOFF,
   tip="<b>Diagnostic.</b> Logs every allocation and whether it landed on device or host - "
       "the definitive answer to 'is the model actually on the card'. Verbose; turn on to "
       "investigate placement, then off."),
 "GGML_VK_PERF_LOGGER": dict(group="Vulkan (RADV)", label="Perf logger", type="select",
   options=OPT_ONOFF,
   tip="<b>Diagnostic.</b> Per-op GPU timings, which is how you find which kernel is slow "
       "rather than guessing from end-to-end t/s. Adds overhead - measure with it off."),

 "PORT": dict(group="Server", label="API port", type="int",
   tip="Inference API port. Binds 0.0.0.0 and sits OUTSIDE Caddy with no authentication."),
 "MODEL": dict(group="Server", label="Model file", type="text",
   tip="Path to the GGUF served. Changing it needs a restart and a full reload."),

 "TEMPLATE_SRC": dict(group="Server", label="Chat template", type="select",
   strict=True, options=[""],   # replaced at request time by list_templates()
   tip="The Jinja chat template, from <code>~/models/*.jinja</code>. Download one on "
       "the Download tab or paste one on the Templates tab, then pick it here.<br><br>"
       "The launch script SHA-pins the template and <b>aborts</b> on a mismatch. The "
       "panel writes the matching pin (<code>TEMPLATE_SHA256</code>) automatically "
       "whenever you save, so the check keeps protecting you without pinning one "
       "specific file. <b>Edit a selected template on disk and the next launch will "
       "abort</b> - re-save in the panel to re-pin it."),
 "LORA": dict(group="Abliteration", label="LoRA adapter", type="text",
   tip="Path to a LoRA .gguf file to load on top of the base model. "
       "Use for style changes or abliteration adapters (e.g., Heretic uncensoring). "
       "Requires a restart to apply. Upstream: --lora"),

 "LORA_SCALED": dict(group="Abliteration", label="LoRA scaled", type="text",
   tip="LoRA file with strength, e.g. file.gguf:0.5 for half strength. "
       "Useful for tuning abliteration intensity without loading a different adapter. "
       "Upstream: --lora-scaled"),

}

# ============================================================================
# NUMERIC CONTROLS
#
# Which parameters get a real number input - up/down arrows, keyboard step, and
# free typing - instead of a dropdown, plus the step/range for each. Their
# existing option lists stay on as PRESETS in a dropdown beside the box.
#
# This is an explicit list on purpose. Auto-detecting "options all look like
# numbers" would sweep in the GGML_VK_* tri-states, whose options are
# ("", "0", "1"): those are enums meaning unset/off/on, and a spinner that
# walks 0 -> 1 -> 2 on them would be nonsense.
#
# min/max are guard rails against a typo, not policy. Memory sizing is checked
# properly by ram_budget() on save; nothing here is load-bearing for safety.
# ============================================================================
SPIN = {
    "CTX":                     dict(step=1024, min=512),
    "CACHE_RAM":               dict(step=512,  min=-1),
    "CACHE_REUSE":             dict(step=64,   min=0),
    "BATCH":                   dict(step=64,   min=1),
    "UBATCH":                  dict(step=32,   min=1),
    "NGL":                     dict(step=1,    min=0,  max=999),
    "THREADS":                 dict(step=1,    min=1,  max=32),
    "SPEC_N_MAX":              dict(step=1,    min=1,  max=16),
    "N_PREDICT":               dict(step=256,  min=-1),
    "RAM_FLOOR_MB":            dict(step=256,  min=0),
    "TOP_K":                   dict(step=1,    min=0),
    "REASONING_BUDGET":        dict(step=256,  min=-1),
    "MAX_TOOL_RESPONSE_CHARS": dict(step=500,  min=0),
    "PORT":                    dict(step=1,    min=1,  max=65535),
    "TEMP":                    dict(step=0.05, min=0),
    "TOP_P":                   dict(step=0.01, min=0,  max=1),
    "MIN_P":                   dict(step=0.01, min=0,  max=1),
}

# Presets for the numeric parameters that had no option list at all, so the
# dropdown beside the spinner is useful on every one of them rather than only
# on the handful that happened to have an OPT_ list.
EXTRA_PRESETS = {
    "CACHE_REUSE":             ["0", "128", "256", "512", "1024"],
    "N_PREDICT":               ["1024", "2048", "4096", "8192", "16000", "32000"],
    "RAM_FLOOR_MB":            ["512", "1024", "2048", "3072", "4096", "6144"],
    "TOP_K":                   ["0", "20", "40", "64", "100"],
    "REASONING_BUDGET":        ["-1", "1024", "2048", "4096", "8192", "16384"],
    "MAX_TOOL_RESPONSE_CHARS": ["2000", "4000", "8000", "9000", "16000"],
    "PORT":                    ["8081", "8083"],
    "TEMP":                    ["0.6", "0.7", "0.8", "1.0", "1.2"],
    "TOP_P":                   ["0.8", "0.9", "0.95", "1.0"],
    "MIN_P":                   ["0.0", "0.01", "0.05", "0.1"],
}

# ============================================================================
# FLAG SURFACE AUDIT - regenerated against build b10985 on 2026-09-15.
# ----------------------------------------------------------------------------
# Everything below was extracted from THIS box's binaries, not from memory:
#   llama-server --help            (b10985, 0.4.1-dev, commit 760984655)
#   strings libggml-vulkan.so      GGML_VK_*   (33 vars)
#   strings libggml-hip.so         GGML_CUDA_* (10 vars)
# Regenerate after every llama.cpp upgrade with panel/audit-flags.sh, which
# re-runs exactly those three extractions and diffs them against this table.
#
# CONVENTION, unchanged from the original block: "" means LEAVE THE FLAG OFF
# ENTIRELY, which is stock llama.cpp behaviour. Adding a knob here therefore
# cannot move the running config until someone sets it deliberately. The launch
# scripts drop every empty value via their _opt/_tri helpers.
#
# DEPRECATIONS FOUND IN b10985 (deliberately NOT exposed):
#   --defrag-thold / -dt        marked DEPRECATED in --help
#   --no-mmap, --mlock          REMOVED; replaced by -lm/--load-mode
#   --draft, --draft-n,
#   --draft-max, --draft-min    REMOVED; use --spec-draft-n-max/-n-min
#   --spec-ngram-size-n/-m,
#   --spec-ngram-min-hits       REMOVED; use the per-algorithm --spec-ngram-*
# None of these were ever referenced by this panel or the launch scripts, so
# nothing had to be migrated - but they are listed so the next upgrade can tell
# "absent because removed" from "absent because we forgot".
#
# DELIBERATELY OMITTED (present in b10985, meaningless or unsafe on inf01):
#   multi-GPU:   -sm/--split-mode, -ts/--tensor-split, -mg/--main-gpu,
#                GGML_CUDA_P2P, GGML_CUDA_ALLREDUCE   (one card)
#   embeddings:  --embedding, --rerank, --pooling, --embd-normalize
#   router mode: --models-dir, --models-preset, --models-max, --models-autoload
#   downloaders: -hf/-hfr/-hff, -mu, -dr, and every --*-default preset
#   agent mode:  --tools, --tools-runtime, --mcp-servers-*, -ag/--agent,
#                --ui-mcp-proxy   (all flagged "do not enable in untrusted
#                environments" upstream, and 8081 has no auth - see API_KEY)
#   MoE offload: -cmoe/-ncmoe/-ncffn   (Qwen3.8-27B is dense)
#   GGML_VK_ALLOW_SYSMEM_FALLBACK: it is the GUARD from trap 2, pinned to 0 by
#                the launch script. Exposing it in the UI would make the single
#                most expensive misconfiguration on this box one click away.
# ============================================================================
_NEW = {}
def _p(key, default, group, label, type="text", options=None, unit=None,
       danger=False, spin=None, presets=None, tip=""):
    d = dict(group=group, label=label, type=type, tip=tip)
    if options is not None: d["options"] = options
    if unit: d["unit"] = unit
    if danger: d["danger"] = True
    _NEW[key] = (default, d, spin, presets)

OPT_TRI      = ["", "on", "off"]          # "" = don't pass the flag at all
OPT_LOADMODE = ["", "auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]
OPT_LAZYMODE = ["", "auto", "on", "off"]
OPT_MIROSTAT = ["", "0", "1", "2"]
OPT_LOGVERB  = ["", "0", "1", "2", "3", "4", "5"]

# --- Context & memory -------------------------------------------------------
_p("LOAD_MODE", "", "Context & memory", "Model load mode", "select", OPT_LOADMODE,
   tip="<code>-lm/--load-mode</code>, <b>new in b10985</b>. This REPLACES the old "
       "<code>--no-mmap</code> and <code>--mlock</code>, which no longer exist - if you "
       "have notes telling you to pass those, they are stale. <b>auto</b> (stock) mmaps "
       "unless a device cannot. <b>mlock</b> pins the weights in RAM; on a 30 GB host "
       "already running a 15.7 GB model that is a fast route to the swap-thrash OOM of "
       "2026-09-04, so treat it as a danger setting. <b>dio</b> uses DirectIO if "
       "available. Empty leaves the flag off.")
_p("LAZY_MODE", "", "Context & memory", "Lazy tensor reads", "select", OPT_LAZYMODE,
   tip="<code>-lzm/--lazy-mode</code>, new in b10985. Reads oversized tensors (e.g. "
       "per-layer embeddings) from disk on demand instead of keeping them resident; "
       "requires mmap. Stock is <b>auto</b> = on for tensors over 4 GiB. Nothing in "
       "this IQ4_XS file is that large, so this is a no-op here unless the model changes.")
_p("KV_OFFLOAD", "", "Context & memory", "KV cache on GPU", "select", OPT_TRI,
   tip="<code>-kvo/--kv-offload</code> / <code>-nkvo/--no-kv-offload</code>. Stock is "
       "ENABLED - the KV cache lives in VRAM. Turning it off moves the whole cache to "
       "host RAM, which at ctx 245760 is many GB over PCIe every token. It is a "
       "diagnostic, not a tuning knob: use it to prove a hang is KV-related, not to "
       "buy headroom.")
_p("SWA_FULL", "", "Context & memory", "Full-size SWA cache", "select", OPT_TRI,
   tip="<code>--swa-full</code>. Only meaningful for sliding-window-attention models; "
       "Qwen3.8 is not one, so this is inert here. Kept visible because a future model "
       "swap can make it suddenly matter - a wrong default on an SWA model silently "
       "costs either accuracy or a large multiple of KV memory.")
_p("KV_UNIFIED", "", "Context & memory", "Unified KV buffer", "select", OPT_TRI,
   tip="<code>-kvu/--kv-unified</code> / <code>--no-kv-unified</code>. One KV buffer "
       "shared across all sequences instead of one per slot. Stock is enabled when the "
       "slot count is auto. This box runs a single slot (see Server slots), so it "
       "changes nothing until PARALLEL goes above 1.")
_p("KV_UNIFIED_PER_SLOT", "", "Context & memory", "KV per slot", "int", unit="tokens",
   spin=dict(step=1024, min=0), presets=["", "32768", "65536", "131072"],
   tip="<code>--kv-unified-per-slot</code>, new in b10985. Context limit per slot. If "
       "set <b>without</b> -c/--ctx-size the shared KV pool is sized to n_parallel*N. "
       "This panel always passes -c, so here it only caps each slot within the pool "
       "you already sized. Empty = unset = behaviour unchanged.")
_p("CTX_CHECKPOINTS", "", "Context & memory", "Context checkpoints", "int",
   spin=dict(step=1, min=0), presets=["", "0", "4", "8", "16", "32"],
   tip="<code>-ctxcp/--ctx-checkpoints</code> (was --swa-checkpoints). Max context "
       "checkpoints kept per slot; stock is <b>32</b>. Checkpoints are what let the "
       "server rewind instead of re-prefilling, which at 245760 context is the "
       "difference between a resumed turn and a ~5 minute re-read. They are not free "
       "in memory, so this is the first thing to lower if the VRAM estimate is "
       "marginal and you would rather pay in prefill than in headroom.")
_p("CHECKPOINT_MIN_STEP", "", "Context & memory", "Checkpoint spacing", "int", unit="tokens",
   spin=dict(step=1024, min=0), presets=["", "0", "4096", "8192", "16384"],
   tip="<code>-cms/--checkpoint-min-step</code>. Minimum spacing between checkpoints; "
       "stock 8192, 0 = no minimum. Wider spacing = fewer checkpoints = less memory and "
       "coarser rewind granularity.")
_p("CACHE_PROMPT", "", "Context & memory", "Prompt caching", "select", OPT_TRI,
   tip="<code>--cache-prompt</code> / <code>--no-cache-prompt</code>. Stock ENABLED, and "
       "it must stay enabled for CACHE_REUSE to do anything - <code>--cache-reuse</code> "
       "is documented as requiring it. Turning it off makes every request a cold "
       "prefill. Only reason to touch it is to measure what the cache is worth.")
_p("CACHE_IDLE_SLOTS", "", "Context & memory", "Cache idle slots", "select", OPT_TRI,
   tip="<code>--cache-idle-slots</code> / <code>--no-cache-idle-slots</code>, new in "
       "b10985. Saves idle slots into the prompt cache when a new task arrives, and "
       "clears them when using unified KV. Stock enabled, and it requires cache-ram to "
       "be non-zero. Costs host RAM out of the CACHE_RAM budget, which the RAM "
       "calculator on this page already accounts for.")
_p("CONTEXT_SHIFT", "", "Context & memory", "Context shift", "select", OPT_TRI,
   tip="<code>--context-shift</code> / <code>--no-context-shift</code>. <b>Upstream "
       "changed the default to DISABLED</b>, which is why long sessions now stop at the "
       "context limit instead of silently sliding the window and dropping the head of "
       "the conversation. Turning it on trades a hard stop for silent amnesia. With "
       "PRESERVE_THINKING on, what gets dropped first is the oldest reasoning, so the "
       "model loses its own earlier conclusions without saying so.")

# --- Fit / auto-sizing (new in b10985, and ON by default) -------------------
_p("FIT", "", "Context & memory", "Auto-fit to VRAM", "select", OPT_TRI,
   tip="<code>-fit/--fit</code>, <b>new and stock-ENABLED in b10985</b>. llama.cpp will "
       "quietly adjust arguments you did <i>not</i> set so the model fits device memory. "
       "This panel sets ctx, ngl, batch and ubatch explicitly, so there is little left "
       "for it to move - but it is the reason a config can now load where the same "
       "numbers would have failed on b10766. If you are bisecting a memory regression "
       "against the old build, set this to <b>off</b> first so the two builds are "
       "actually comparable.")
_p("FIT_TARGET", "", "Context & memory", "Auto-fit margin", "int", unit="MiB",
   spin=dict(step=128, min=0), presets=["", "512", "1024", "2048"],
   tip="<code>-fitt/--fit-target</code>. Margin per device left free by --fit; stock "
       "1024 MiB. This box runs with roughly that much headroom in total, so the margin "
       "and the working config are the same size - raise it only together with a lower CTX.")
_p("FIT_CTX", "", "Context & memory", "Auto-fit min context", "int", unit="tokens",
   spin=dict(step=1024, min=0), presets=["", "4096", "32768", "131072"],
   tip="<code>-fitc/--fit-ctx</code>. Floor on the context --fit is allowed to shrink to; "
       "stock 4096. Inert while CTX is set explicitly.")

# --- Throughput -------------------------------------------------------------
_p("FLASH_ATTN", "on", "Throughput", "Flash attention", "select", OPT_FLASH_ATTN,
   tip="<code>-fa/--flash-attn</code>. Was <b>hardcoded to 'on'</b> in all three launch "
       "scripts until 2026-09-15 and is now a real setting. It is what makes the "
       "quantised KV types usable: without it the q5_1/q4_1 cache paths fall back and "
       "the memory estimate on this page stops being true. Stock upstream is 'auto'; "
       "this box ships 'on' because that is the configuration every measurement here "
       "was taken under.")
_p("PARALLEL", "1", "Throughput", "Server slots", "select", OPT_PARALLEL,
   tip="<code>-np/--parallel</code>. Was <b>hardcoded to 1</b> until 2026-09-15. Each "
       "slot gets its own share of the KV pool, so going to 2 slots at a fixed CTX "
       "halves the depth each conversation can reach. Upstream's stock is -1 (auto). "
       "Keep it at 1 on this box: 24 GB with ~1.4 GB spare does not have room for a "
       "second full-depth conversation, and the VRAM calculator above assumes one.")
_p("THREADS_HTTP", "", "Throughput", "HTTP threads", "int",
   spin=dict(step=1, min=-1, max=16), presets=["", "-1", "1", "2", "4"],
   tip="<code>--threads-http</code>. Threads serving HTTP, separate from inference "
       "threads. Stock -1 = auto. On a 4-core E-2224G every HTTP thread competes with "
       "the 4 inference threads, so raising it costs decode throughput.")
_p("BACKEND_SAMPLING", "", "Throughput", "Backend sampling", "select", OPT_TRI,
   danger=True,
   tip="<code>-bs/--backend-sampling</code>, new in b10985 and marked "
       "<b>experimental</b> upstream. Moves the sampler onto the GPU. Untested on RADV "
       "here, and this box's failure mode for a bad GPU path is a compute-ring timeout "
       "or a hard lock, not an error message. Leave empty unless you are deliberately "
       "testing it with a short context.")
_p("OP_OFFLOAD", "", "Throughput", "Offload host ops", "select", OPT_TRI,
   tip="<code>--op-offload</code> / <code>--no-op-offload</code>. Whether host tensor "
       "operations are pushed to the device; stock enabled. Disabling moves work back to "
       "the CPU - a diagnostic for suspected backend op bugs, not a speed knob.")
_p("REPACK", "", "Throughput", "Weight repacking", "select", OPT_TRI,
   tip="<code>--repack</code> / <code>-nr/--no-repack</code>. Repacks weights into a "
       "layout the CPU kernels prefer; stock enabled. Matters for BACKEND=cpu, "
       "irrelevant at NGL 99 where nothing runs on the CPU.")
_p("NO_HOST", "", "Throughput", "Bypass host buffer", "select", OPT_TRI,
   tip="<code>--no-host</code>, new in b10985. Bypasses the host buffer so extra buffer "
       "types can be used. Interacts directly with the GTT-placement trap: it changes "
       "where staging buffers live. If you set it, re-check VRAM vs GTT (healthy is "
       "~23,000 MiB VRAM against under ~1,000 MiB GTT) before trusting any timing.")

# --- Long context / RoPE ----------------------------------------------------
_p("ROPE_SCALING", "", "Long context (RoPE)", "RoPE scaling", "select", OPT_ROPE,
   tip="<code>--rope-scaling {none,linear,yarn}</code>. The context dropdown offers up "
       "to 524288 while this model's native training context is <b>262144</b>. Anything "
       "above that is extrapolation and needs RoPE scaling to be coherent rather than "
       "merely allocated - the server will happily allocate a 512k cache and produce "
       "degrading output past the native limit with no warning. yarn is the usual "
       "choice for Qwen. Empty = model default.")
_p("ROPE_SCALE", "", "Long context (RoPE)", "RoPE scale", "text",
   tip="<code>--rope-scale N</code>. Context expansion factor. With linear scaling, "
       "2 doubles the usable context. Empty = model default.")
_p("ROPE_FREQ_BASE", "", "Long context (RoPE)", "RoPE freq base", "text",
   tip="<code>--rope-freq-base N</code>. NTK-aware base frequency. Empty = loaded from "
       "the GGUF, which is almost always what you want.")
_p("ROPE_FREQ_SCALE", "", "Long context (RoPE)", "RoPE freq scale", "text",
   tip="<code>--rope-freq-scale N</code>. Inverse form of --rope-scale (expands by 1/N). "
       "Set one or the other, never both.")
_p("YARN_ORIG_CTX", "", "Long context (RoPE)", "YaRN original ctx", "int", unit="tokens",
   spin=dict(step=1024, min=0), presets=["", "32768", "131072", "262144"],
   tip="<code>--yarn-orig-ctx</code>. The model's <i>training</i> context, which YaRN "
       "scales up from. 0 = read from the model. For this model that is 262144.")
_p("YARN_EXT_FACTOR", "", "Long context (RoPE)", "YaRN ext factor", "text",
   tip="<code>--yarn-ext-factor</code>. Extrapolation mix; -1 = model default, "
       "0.0 = full interpolation.")
_p("YARN_ATTN_FACTOR", "", "Long context (RoPE)", "YaRN attn factor", "text",
   tip="<code>--yarn-attn-factor</code>. Attention magnitude scaling. -1 = default.")
_p("YARN_BETA_FAST", "", "Long context (RoPE)", "YaRN beta fast", "text",
   tip="<code>--yarn-beta-fast</code>. Low correction dim (beta). -1 = default.")
_p("YARN_BETA_SLOW", "", "Long context (RoPE)", "YaRN beta slow", "text",
   tip="<code>--yarn-beta-slow</code>. High correction dim (alpha). -1 = default.")

# --- Speculative decoding ---------------------------------------------------
_p("SPEC_N_MIN", "", "Speculative decoding", "Draft n-min", "int",
   spin=dict(step=1, min=0, max=16), presets=["", "0", "1", "2"],
   tip="<code>--spec-draft-n-min</code>. Floor on drafted tokens per step; stock 0. "
       "Raising it forces the draft head to commit even when its confidence is low, "
       "which on this box shows up as a falling acceptance rate (measured 0.56-0.82, "
       "content-dependent) rather than as an error.")
_p("SPEC_P_MIN", "0", "Speculative decoding", "Draft p-min", "float", spin=dict(step=0.05, min=0, max=1),
   presets=["0", "0.1", "0.5", "0.9"],
   tip="<code>--spec-draft-p-min</code>. Minimum probability before a drafted token is "
       "proposed; stock 0.00 (greedy). The launch scripts have passed 0 explicitly since "
       "before the panel existed and it is now a real setting rather than a literal.")
_p("SPEC_P_SPLIT", "", "Speculative decoding", "Draft p-split", "float", spin=dict(step=0.05, min=0, max=1),
   presets=["", "0.1", "0.5"],
   tip="<code>--spec-draft-p-split</code>. Split probability for the draft tree; "
       "stock 0.10. Only used by tree-style draft types.")
_p("SPEC_DRAFT_KV_TYPE", "", "Speculative decoding", "Draft KV type", "select",
   [""] + OPT_KV_TYPE,
   tip="<code>-ctkd/-ctvd</code> (<code>--spec-draft-type-k/-v</code>), new names in "
       "b10985. KV cache type for the DRAFT context, set independently of the target's "
       "KV_TYPE; stock f16. With <code>draft-mtp</code> the draft context is the ~370 MiB "
       "one created against the target, so this is a small saving - but it is a real one "
       "when headroom is measured in hundreds of MiB. Empty = f16.")
_p("SPEC_DRAFT_NGL", "", "Speculative decoding", "Draft GPU layers", "select",
   ["", "all", "auto", "0"],
   tip="<code>-ngld/--spec-draft-ngl</code>. Accepts a number, <b>auto</b> or <b>all</b>. "
       "The launch scripts pass <code>all</code> automatically whenever a real sidecar is "
       "set in SPEC_DRAFT_MODEL; this overrides that. Irrelevant for the embedded MTP "
       "head, which has no separate layers to place.")
_p("SPEC_DRAFT_BACKEND_SAMPLING", "", "Speculative decoding", "Draft backend sampling",
   "select", OPT_TRI,
   tip="<code>--spec-draft-backend-sampling</code> / <code>--no-...</code>, new in "
       "b10985 and stock <b>enabled</b>. Offloads draft sampling to the backend. Unlike "
       "the target-side <code>-bs</code> this one is on by default, so if you are "
       "chasing a speculative-decoding fault on RADV, turning this OFF is the cheap "
       "first bisect.")

# n-gram speculative decoding. None of these apply to draft-mtp - they exist
# because OPT_SPEC_TYPE offers six ngram-* algorithms that were previously
# selectable with no way to tune any of them.
_p("SPEC_NGRAM_MOD_N_MIN", "", "Speculative (n-gram)", "ngram-mod n-min", "int",
   spin=dict(step=1, min=0), presets=["", "48"],
   tip="<code>--spec-ngram-mod-n-min</code>, stock 48. Only read when SPEC_TYPE is "
       "<b>ngram-mod</b>.")
_p("SPEC_NGRAM_MOD_N_MAX", "", "Speculative (n-gram)", "ngram-mod n-max", "int",
   spin=dict(step=1, min=0), presets=["", "64"],
   tip="<code>--spec-ngram-mod-n-max</code>, stock 64. Only read when SPEC_TYPE is "
       "<b>ngram-mod</b>. This is the flag that replaced the removed --draft-max for "
       "ngram types.")
_p("SPEC_NGRAM_MOD_N_MATCH", "", "Speculative (n-gram)", "ngram-mod match len", "int",
   spin=dict(step=1, min=0), presets=["", "24"],
   tip="<code>--spec-ngram-mod-n-match</code>, stock 24. Lookup length for ngram-mod. "
       "Replaced the removed --spec-ngram-size-n.")
_p("SPEC_NGRAM_SIMPLE_SIZE_N", "", "Speculative (n-gram)", "ngram-simple lookup N", "int",
   spin=dict(step=1, min=0), presets=["", "12"],
   tip="<code>--spec-ngram-simple-size-n</code>, stock 12. Only read when SPEC_TYPE is "
       "<b>ngram-simple</b>.")
_p("SPEC_NGRAM_SIMPLE_SIZE_M", "", "Speculative (n-gram)", "ngram-simple draft M", "int",
   spin=dict(step=1, min=0), presets=["", "48"],
   tip="<code>--spec-ngram-simple-size-m</code>, stock 48.")
_p("SPEC_NGRAM_SIMPLE_MIN_HITS", "", "Speculative (n-gram)", "ngram-simple min hits", "int",
   spin=dict(step=1, min=0), presets=["", "1"],
   tip="<code>--spec-ngram-simple-min-hits</code>, stock 1.")
_p("SPEC_NGRAM_MAP_K_SIZE_N", "", "Speculative (n-gram)", "ngram-map-k lookup N", "int",
   spin=dict(step=1, min=0), presets=["", "12"],
   tip="<code>--spec-ngram-map-k-size-n</code>, stock 12. Only read when SPEC_TYPE is "
       "<b>ngram-map-k</b>.")
_p("SPEC_NGRAM_MAP_K_SIZE_M", "", "Speculative (n-gram)", "ngram-map-k draft M", "int",
   spin=dict(step=1, min=0), presets=["", "48"],
   tip="<code>--spec-ngram-map-k-size-m</code>, stock 48.")
_p("SPEC_NGRAM_MAP_K_MIN_HITS", "", "Speculative (n-gram)", "ngram-map-k min hits", "int",
   spin=dict(step=1, min=0), presets=["", "1"],
   tip="<code>--spec-ngram-map-k-min-hits</code>, stock 1.")
_p("SPEC_NGRAM_MAP_K4V_SIZE_N", "", "Speculative (n-gram)", "ngram-map-k4v lookup N", "int",
   spin=dict(step=1, min=0), presets=["", "12"],
   tip="<code>--spec-ngram-map-k4v-size-n</code>, stock 12. Only read when SPEC_TYPE is "
       "<b>ngram-map-k4v</b>.")
_p("SPEC_NGRAM_MAP_K4V_SIZE_M", "", "Speculative (n-gram)", "ngram-map-k4v draft M", "int",
   spin=dict(step=1, min=0), presets=["", "48"],
   tip="<code>--spec-ngram-map-k4v-size-m</code>, stock 48.")
_p("SPEC_NGRAM_MAP_K4V_MIN_HITS", "", "Speculative (n-gram)", "ngram-map-k4v min hits", "int",
   spin=dict(step=1, min=0), presets=["", "1"],
   tip="<code>--spec-ngram-map-k4v-min-hits</code>, stock 1.")

# --- Vision -----------------------------------------------------------------
_p("IMAGE_MIN_TOKENS", "1024", "Vision", "Image min tokens", "int",
   spin=dict(step=64, min=0), presets=["", "256", "512", "1024", "2048"],
   tip="<code>--image-min-tokens</code>. Floor on tokens spent per image by a "
       "dynamic-resolution vision model. Was <b>hardcoded to 1024</b> in all three "
       "launch scripts until 2026-09-15. Every image now costs at least this much "
       "context, so at 1024 a handful of screenshots is a meaningful bite out of even a "
       "245k window. Empty = read from the model.")
_p("IMAGE_MAX_TOKENS", "", "Vision", "Image max tokens", "int",
   spin=dict(step=64, min=0), presets=["", "1024", "2048", "4096"],
   tip="<code>--image-max-tokens</code>. Ceiling per image. This is the one to set if a "
       "large screenshot is blowing out the context - it caps the cost instead of "
       "letting resolution decide it. Empty = read from the model.")
_p("MTMD_BATCH_MAX_TOKENS", "", "Vision", "Image encode batch", "int",
   spin=dict(step=128, min=0), presets=["", "512", "1024", "2048"],
   tip="<code>--mtmd-batch-max-tokens</code>, stock 1024. Image tokens per encode batch. "
       "This sets the size of a transient VRAM spike during encoding, and transient "
       "spikes are what push amdgpu into evicting weights to GTT on this card. Lower it "
       "before lowering CTX if vision is what destabilises a config.")

# --- Reasoning / chat -------------------------------------------------------
_p("REASONING", "on", "Reasoning", "Reasoning mode", "select", OPT_REASONING,
   tip="<code>-rea/--reasoning</code>. Was <b>hardcoded to 'on'</b> until 2026-09-15. "
       "'auto' detects from the chat template, which for the pinned qwen3.8-safe-v2 "
       "template means thinking stays on anyway. 'off' is the only real way to get "
       "non-thinking replies out of this model without editing the template.")
_p("REASONING_FORMAT", "", "Reasoning", "Reasoning format", "select", OPT_REAS_FMT,
   tip="<code>--reasoning-format</code>. Where thoughts end up in the API response: "
       "<b>none</b> leaves them unparsed inside message.content, <b>deepseek</b> puts "
       "them in message.reasoning_content, <b>deepseek-legacy</b> does both (keeps the "
       "&lt;think&gt; tags in content AND fills reasoning_content). Stock is auto. "
       "Clients that render a collapsible 'thinking' block want deepseek; a client that "
       "shows raw content will display the entire chain of thought if this is none.")
_p("REASONING_BUDGET_MESSAGE", "", "Reasoning", "Budget-exhausted message", "text",
   tip="<code>--reasoning-budget-message</code>, <b>new in b10985</b>. Text injected "
       "immediately before the end-of-thinking tag when REASONING_BUDGET runs out. "
       "Without it the model is cut off mid-thought and then has to answer from a "
       "truncated trace; with it you get a handoff line such as "
       "<i>\"Budget reached - answer now with what you have.\"</i> If you set a finite "
       "REASONING_BUDGET, set this too. Empty = none.")
_p("KEEP", "", "Reasoning", "Tokens to keep", "int",
   spin=dict(step=64, min=-1), presets=["", "0", "-1", "512"],
   tip="<code>--keep</code>. Tokens kept from the start of the prompt when the context "
       "is shifted; stock 0, -1 = all. Only has an effect with CONTEXT_SHIFT on. Set it "
       "to cover the system prompt, or a shifted conversation loses its instructions "
       "first and nothing reports that it happened.")

# --- Sampling (samplers added since the panel's original four) --------------
_p("SAMPLERS", "", "Sampling", "Sampler chain", "text",
   tip="<code>--samplers</code>, ';'-separated and <b>order matters</b>. Stock chain in "
       "b10985 is <code>penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;"
       "temperature</code>. Empty = that stock chain. Anything you omit here is "
       "disabled no matter what its own parameter says, which is the usual reason a "
       "sampler 'does nothing'.")
_p("SEED", "", "Sampling", "RNG seed", "int",
   spin=dict(step=1, min=-1), presets=["", "-1", "0", "42"],
   tip="<code>-s/--seed</code>. -1 (stock) = random per request. Pin it only for "
       "reproducing a specific output; a fixed seed on a shared server makes every "
       "client's generation correlated.")
_p("TOP_N_SIGMA", "", "Sampling", "top_n_sigma", "float", spin=dict(step=0.1, min=-1), presets=["", "-1", "1.0"],
   tip="<code>--top-nsigma</code>, stock -1.0 = disabled. Keeps tokens within N standard "
       "deviations of the top logit. Sits before top_k in the stock chain.")
_p("TYPICAL_P", "", "Sampling", "typical_p", "float", spin=dict(step=0.01, min=0, max=1), presets=["", "1.0", "0.95"],
   tip="<code>--typical</code>, stock 1.0 = disabled. Locally typical sampling.")
_p("REPEAT_PENALTY", "", "Sampling", "repeat_penalty", "float", spin=dict(step=0.01, min=0), presets=["", "1.0", "1.05", "1.1"],
   tip="<code>--repeat-penalty</code>, stock 1.0 = disabled. Qwen thinking mode does not "
       "want this: a repetition penalty applied across a long chain of thought punishes "
       "the model for restating its own premises, which is exactly what reasoning does.")
_p("REPEAT_LAST_N", "", "Sampling", "repeat_last_n", "int",
   spin=dict(step=16, min=0), presets=["", "0", "64", "256"],
   tip="<code>--repeat-last-n</code>, stock 64, 0 = disabled. Window the repetition "
       "penalty looks back over.")
_p("PRESENCE_PENALTY", "", "Sampling", "presence_penalty", "float", spin=dict(step=0.05), presets=["", "0.0", "0.5"],
   tip="<code>--presence-penalty</code>, stock 0.0 = disabled. Default only; a client "
       "sending its own value wins.")
_p("FREQUENCY_PENALTY", "", "Sampling", "frequency_penalty", "float", spin=dict(step=0.05), presets=["", "0.0", "0.5"],
   tip="<code>--frequency-penalty</code>, stock 0.0 = disabled. Default only.")
_p("DRY_MULTIPLIER", "", "Sampling", "DRY multiplier", "float", spin=dict(step=0.05, min=0), presets=["", "0.0", "0.8"],
   tip="<code>--dry-multiplier</code>, stock 0.0 = disabled. DRY suppresses verbatim "
       "repetition of sequences rather than of single tokens, which makes it far safer "
       "on reasoning output than repeat_penalty. It is second in the stock sampler chain.")
_p("DRY_BASE", "", "Sampling", "DRY base", "float", spin=dict(step=0.05, min=0), presets=["", "1.75"],
   tip="<code>--dry-base</code>, stock 1.75. Only read when DRY_MULTIPLIER > 0.")
_p("DRY_ALLOWED_LENGTH", "", "Sampling", "DRY allowed length", "int",
   spin=dict(step=1, min=0), presets=["", "2"],
   tip="<code>--dry-allowed-length</code>, stock 2. Repeats shorter than this are not "
       "penalised. On code output raise it: indentation and closing braces are legitimate "
       "short repeats.")
_p("DRY_PENALTY_LAST_N", "", "Sampling", "DRY window", "int",
   spin=dict(step=16, min=0), presets=["", "0", "64", "512"],
   tip="<code>--dry-penalty-last-n</code>, stock 64, 0 = disabled.")
_p("XTC_PROBABILITY", "", "Sampling", "XTC probability", "float", spin=dict(step=0.05, min=0, max=1), presets=["", "0.0", "0.5"],
   tip="<code>--xtc-probability</code>, stock 0.0 = disabled. Exclude Top Choices drops "
       "high-probability tokens to raise variety. Directly opposed to what you want from "
       "a coding or tool-calling model.")
_p("XTC_THRESHOLD", "", "Sampling", "XTC threshold", "float", spin=dict(step=0.05, min=0, max=1), presets=["", "0.1"],
   tip="<code>--xtc-threshold</code>, stock 0.1, 1.0 = disabled.")
_p("ADAPTIVE_TARGET", "", "Sampling", "adaptive-p target", "float", spin=dict(step=0.05, min=-1, max=1), presets=["", "-1", "0.1"],
   tip="<code>--adaptive-target</code>, <b>new in b10985</b> (upstream PR 17927). "
       "adaptive-p selects tokens near this probability and adapts over time; valid 0.0 "
       "to 1.0, negative = disabled (stock -1.0). New enough that there are no numbers "
       "for it on this box.")
_p("ADAPTIVE_DECAY", "", "Sampling", "adaptive-p decay", "float", spin=dict(step=0.01, min=0, max=0.99), presets=["", "0.90"],
   tip="<code>--adaptive-decay</code>, stock 0.90, valid 0.0-0.99. Lower is more "
       "reactive, higher more stable. Only read when ADAPTIVE_TARGET is enabled.")
_p("DYNATEMP_RANGE", "", "Sampling", "dynatemp range", "float", spin=dict(step=0.05, min=0), presets=["", "0.0", "0.5"],
   tip="<code>--dynatemp-range</code>, stock 0.0 = disabled. Varies temperature with "
       "entropy around TEMP.")
_p("DYNATEMP_EXP", "", "Sampling", "dynatemp exponent", "float", spin=dict(step=0.05, min=0), presets=["", "1.0"],
   tip="<code>--dynatemp-exp</code>, stock 1.0. Only read when DYNATEMP_RANGE > 0.")
_p("MIROSTAT", "", "Sampling", "Mirostat", "select", OPT_MIROSTAT,
   tip="<code>--mirostat</code>, stock 0 = disabled. <b>1 or 2 makes top_k, top_p and "
       "typical_p be ignored entirely</b> - the panel will still show those values and "
       "they will still be sent, and none of them will do anything. Do not enable it and "
       "then tune top_p.")
_p("MIROSTAT_LR", "", "Sampling", "Mirostat eta", "float", spin=dict(step=0.01, min=0), presets=["", "0.1"],
   tip="<code>--mirostat-lr</code>, stock 0.10. Only read when MIROSTAT is 1 or 2.")
_p("MIROSTAT_ENT", "", "Sampling", "Mirostat tau", "float", spin=dict(step=0.1, min=0), presets=["", "5.0"],
   tip="<code>--mirostat-ent</code>, stock 5.00. Only read when MIROSTAT is 1 or 2.")

# --- Server -----------------------------------------------------------------
_p("HOST", "0.0.0.0", "Server", "Bind address", "select", ["0.0.0.0", "127.0.0.1"],
   danger=True,
   tip="<code>--host</code>. Was <b>hardcoded to 0.0.0.0</b> until 2026-09-15, which is "
       "how port 8081 came to be reachable from the whole network with no authentication "
       "in front of it - Caddy only fronts the panel. Setting this to <b>127.0.0.1</b> "
       "is the one-line fix if you do not need remote inference; if you do, set API_KEY. "
       "Needs a restart, and will cut off any remote client immediately.")
_p("API_KEY", "", "Server", "API key", "text", danger=True,
   tip="<code>--api-key</code>, comma-separated for several. <b>Empty means port 8081 "
       "accepts anything that can reach it.</b> That port bypasses Caddy entirely, so "
       "the panel's basicauth does not protect it and neither does anything else on this "
       "box. Setting a key here is the only authentication the inference API has. Note "
       "it is stored in cleartext in params.env and lands in the configuration backup "
       "tarball, which is mode 600 - keep it that way.")
_p("TIMEOUT", "", "Server", "Request timeout", "int", unit="s",
   spin=dict(step=60, min=0), presets=["", "600", "3600", "7200"],
   tip="<code>-to/--timeout</code>, stock 3600. Server read/write timeout. A single "
       "deep-context request on this box can prefill for minutes before the first token; "
       "a client-side timeout shorter than this is the usual cause of a 'hang' that the "
       "server log shows completing normally.")
_p("SSE_PING_INTERVAL", "", "Server", "SSE ping interval", "int", unit="s",
   spin=dict(step=5, min=-1), presets=["", "-1", "15", "30"],
   tip="<code>--sse-ping-interval</code>, stock 30, -1 = disabled. Keepalive on streaming "
       "responses. Raise or disable it only if a proxy is mangling the stream.")
_p("SLEEP_IDLE_SECONDS", "", "Server", "Sleep when idle", "int", unit="s",
   spin=dict(step=60, min=-1), presets=["", "-1", "300", "900", "1800"],
   tip="<code>--sleep-idle-seconds</code>, stock -1 = disabled. Puts the server to sleep "
       "after N idle seconds. Interesting on this box specifically: a 291 W card in a "
       "300 W Dell T40 is the working theory for the silent hard locks, and an idle "
       "server that has released the GPU is one fewer thing holding the rail up. The "
       "cost is that the next request pays a wake-up, and no measurement of that cost "
       "has been taken here yet.")
_p("WEBUI", "", "Server", "Built-in web UI", "select", OPT_TRI,
   tip="<code>--ui/--webui</code> / <code>--no-webui</code>, stock enabled. llama.cpp's "
       "own chat UI, served on this instance's port next to the API (no login of its own). "
       "The Web UI card on the Status tab sets its defaults and can put it behind the "
       "panel's login at <code>/ui/&lt;instance&gt;/</code>.")
_p("UI_CONFIG_FILE", "", "Server", "Web UI defaults file", "text",
   tip="<code>--ui-config-file</code>: the settings a browser starts from in the built-in "
       "web UI. Written by the Web UI card on the Status tab; leave it to that card. A "
       "missing or unreadable file is left out at launch rather than stopping the server.")
_p("UI_MCP_PROXY", "", "Server", "Web UI MCP proxy", "select", OPT_TRI, danger=True,
   tip="<code>--ui-mcp-proxy</code>, stock <b>disabled</b> and marked experimental by "
       "llama.cpp: the server relays the web UI's MCP tool traffic so browsers can reach "
       "MCP servers without CORS. llama.cpp says not to enable it in untrusted "
       "environments: anyone who can reach this port could use the relay.")
_p("PROPS", "", "Server", "Allow POST /props", "select", OPT_TRI, danger=True,
   tip="<code>--props</code>, stock <b>disabled</b>. Lets any client change global server "
       "properties over HTTP. On an unauthenticated port that means anyone who can reach "
       "8081 can reconfigure the server out from under this panel. Leave it off unless "
       "API_KEY is set.")
_p("SLOTS", "", "Server", "Slots endpoint", "select", OPT_TRI,
   tip="<code>--slots/--no-slots</code>, stock enabled. Exposes /slots, which the panel's "
       "own diagnostics read. Turning it off blinds this panel's slot view; it also stops "
       "leaking prompt content to anyone who can reach 8081.")
_p("ALIAS", "", "Server", "Model alias", "text",
   tip="<code>-a/--alias</code>, comma-separated. The model name the API reports and that "
       "OpenAI-compatible clients must send. Empty = the GGUF filename, which for this "
       "model is the 96-character "
       "<code>Qwen3.8-27B-TurboFCFusion-735-882-...-IQ4_XS</code>. Setting a short alias "
       "is usually less trouble than making every client quote that.")
_p("LOG_VERBOSITY", "", "Server", "Log verbosity", "select", OPT_LOGVERB,
   tip="<code>-lv/--log-verbosity</code>, stock 3 (info). 0 generic, 1 error, 2 warning, "
       "3 info, 4 trace, 5 debug. 4 and 5 write a great deal into "
       "<code>/dev/shm/llama_qwen38</code>, which is a tmpfs - it costs host RAM, and "
       "host RAM is what the watchdog kills the server over.")

# --- Vulkan/RADV environment, re-extracted from b10985's libggml-vulkan.so ---
# The original block carried 18 of the 33 GGML_VK_* strings in the library.
# These are the 14 that were missing; the 15th, GGML_VK_ALLOW_SYSMEM_FALLBACK,
# is withheld on purpose (see the header of this section). Same convention:
# "" = variable UNSET = stock.
_p("GGML_VK_VISIBLE_DEVICES", "", "Vulkan (RADV)", "Visible devices", "text",
   tip="Restricts which Vulkan devices ggml will enumerate, by index. The launch script "
       "already pins this to <b>0</b> together with VK_DRIVER_FILES, which is what keeps "
       "the Intel UHD P630 iGPU out of the picture (trap 4). Setting it here overrides "
       "that pin, and getting it wrong means the model is served by the iGPU or fails to "
       "find a device at all.")
_p("GGML_VK_ALLOW_GRAPHICS_QUEUE", "", "Vulkan (RADV)", "Allow graphics queue", "text",
   tip="Permits compute work on the graphics queue when no dedicated compute queue is "
       "usable. Relevant to the ring timeouts recorded on 2026-09-03 "
       "(<code>ring comp_X.Y.Z timeout</code>): those were compute-ring hangs, and this "
       "changes which ring the work lands on. That makes it an experiment worth running "
       "against that specific failure, not a speed setting.")
_p("GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "", "Vulkan (RADV)", "Async transfer queue", "text",
   tip="Uses the dedicated transfer queue for async copies. Pairs with "
       "GGML_VK_DISABLE_ASYNC - disable async entirely, or keep it and move the copies "
       "to their own queue.")
_p("GGML_VK_SERIALIZE_SUBMISSIONS", "", "Vulkan (RADV)", "Serialize submissions", "text",
   tip="Forces command-buffer submissions to be serialised. Slow by design; it is a "
       "debugging aid for exactly the class of fault this box hits, where a GPU hang has "
       "no error attached to it. Set it to 1 to find out whether concurrency is what is "
       "wedging the ring.")
_p("GGML_VK_SYNC_LOGGER", "", "Vulkan (RADV)", "Sync logger", "text",
   tip="Logs synchronisation events. Very high volume; the engine log lives on tmpfs, so "
       "leaving this on will consume host RAM and the RAM_FLOOR_MB watchdog will "
       "eventually kill the server over it.")
_p("GGML_VK_PIPELINE_STATS", "", "Vulkan (RADV)", "Pipeline statistics", "text",
   tip="Dumps per-pipeline statistics at shutdown. Diagnostic only, no runtime cost worth "
       "worrying about.")
_p("GGML_VK_DEBUG_MARKERS", "", "Vulkan (RADV)", "Debug markers", "text",
   tip="Emits Vulkan debug markers for capture tools (RenderDoc, RGP). Useless without a "
       "capture tool attached, and this host is headless.")
_p("GGML_VK_PERF_LOGGER_CONCURRENT", "", "Vulkan (RADV)", "Perf logger concurrent", "text",
   tip="Concurrent variant of GGML_VK_PERF_LOGGER. Only read when the perf logger is on.")
_p("GGML_VK_PERF_LOGGER_FREQUENCY", "", "Vulkan (RADV)", "Perf logger frequency", "text",
   tip="Sampling frequency for the perf logger. Only read when the perf logger is on.")
_p("GGML_VK_DISABLE_MULTI_ADD", "", "Vulkan (RADV)", "Disable multi-add", "text",
   tip="Disables the fused multi-add path. One of the fusion knobs to bisect when output "
       "is subtly wrong rather than absent - a wrong fused kernel produces plausible "
       "garbage, not a crash.")
_p("GGML_VK_DISABLE_DOT2", "", "Vulkan (RADV)", "Disable dot2", "text",
   tip="Disables the 2-wide dot-product path. Companion to "
       "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT, which the panel already exposes.")
_p("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR", "", "Vulkan (RADV)",
   "Disable coopmat2 decode vector", "text",
   tip="Disables the coopmat2 vector path used during decode specifically, leaving it "
       "enabled for prefill. Finer-grained than GGML_VK_DISABLE_COOPMAT2, so it can "
       "separate a decode-only fault from a general coopmat2 problem - useful here, where "
       "prefill (~690 t/s) has always been healthy and decode is where the faults land.")
_p("GGML_VK_FA_SPARSE_DISABLE", "", "Vulkan (RADV)", "Disable sparse flash-attn", "text",
   tip="Disables the sparse flash-attention path. Reach for this before turning FLASH_ATTN "
       "off entirely: it keeps the quantised KV types working while removing only the "
       "sparse kernel.")
_p("GGML_VK_FORCE_MAX_BUFFER_SIZE", "", "Vulkan (RADV)", "Force max buffer size", "text",
   tip="Caps any single Vulkan buffer, in bytes. Distinct from "
       "GGML_VK_FORCE_MAX_ALLOCATION_SIZE, which the panel already exposes: allocation "
       "size bounds a device allocation, this bounds a buffer object within one. Both "
       "matter on RADV, where one oversized buffer fails the whole load.")

# --- ROCm/HIP environment, re-extracted from b10985's libggml-hip.so ---------
# Library exports 10 GGML_CUDA_* strings; the panel carried 7. GGML_CUDA_P2P
# and GGML_CUDA_ALLREDUCE are the other two and are multi-GPU only, so they are
# omitted. That leaves one.
_p("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "", "ROCm (HIP)", "cuBLAS compute type", "text",
   tip="Overrides the compute type hipBLAS uses for matmul, e.g. forcing FP32 accumulation "
       "where the stock path would accumulate in FP16. The usual reason to set it is "
       "numerically wrong output under ROCm that is fine under Vulkan. Ignored entirely "
       "when BACKEND=vulkan.")

# --- per-device offload, for instances on small cards (added 2026-09-17) -----
# The audit above omitted these because Qwen3.8-27B is dense and main has one
# card. An instance on the 6 GB RTX 2060 is exactly where they earn their keep.
_p("OVERRIDE_TENSORS", "", "Offload", "Tensor placement override", "text",
   tip="<code>-ot/--override-tensor</code> <i>pattern=buffer</i>, comma-separated. Pins "
       "matching tensors to a buffer type, e.g. <code>exps=CPU</code> keeps MoE expert "
       "weights in host RAM while attention stays on the card - the standard way to run "
       "a MoE model larger than a 6 GB card. Host RAM is shared with every instance: "
       "check the RAM budget card.")
_p("N_CPU_MOE", "", "Offload", "MoE layers on CPU", "int", spin=dict(min=0, step=1),
   presets=["", "4", "8", "16", "32"],
   tip="<code>-ncmoe/--n-cpu-moe N</code>. Keeps the expert weights of the first N layers "
       "on the CPU. Coarser than OVERRIDE_TENSORS, easier to tune: raise N until the model "
       "fits. Dense models ignore it.")
_p("CPU_MOE", "", "Offload", "All MoE experts on CPU", "select", options=["", "on"],
   tip="<code>-cmoe/--cpu-moe</code>. Every expert on the CPU. The quickest way to get a "
       "large MoE model loading at all on a small card.")
_p("SPEC_DRAFT_DEVICE", "", "Offload", "Draft model device", "text",
   tip="<code>--spec-draft-device</code>, e.g. <code>Vulkan1</code>. Runs the draft (or MTP "
       "sidecar) on another card, so its weights and KV stop competing with the target's "
       "experts for VRAM. Device names are the ones this instance's launch plan lists on the "
       "Status tab - with two drivers loaded the order is the loader's, not the PCI order.")
_p("MMPROJ_DEVICE", "", "Offload", "Vision projector device", "text",
   tip="<code>-mmdev/--mmproj-device</code>, e.g. <code>Vulkan1</code>, or <code>none</code> for "
       "CPU. Only takes effect with MMPROJ_OFFLOAD on.")
_p("SPLIT_MODE", "", "Offload", "Multi-GPU split mode", "select",
   options=["", "none", "layer", "row", "tensor"],
   tip="<code>-sm</code>. <b>layer</b> (stock) pipelines whole layers across the cards and is "
       "the one to use with two different cards. <b>row</b>/<b>tensor</b> split each weight "
       "and need fast links; the RTX 2060 sits in a chipset x4 slot.")
_p("TENSOR_SPLIT", "", "Offload", "Tensor split", "text",
   tip="<code>-ts</code>, e.g. <code>24,6</code>: proportion of offloaded layers per device, in "
       "the instance's device order. Leave empty and place experts explicitly with "
       "OVERRIDE_TENSORS when most of the model lives on the CPU.")
_p("MAIN_GPU", "", "Offload", "Main GPU index", "int", spin=dict(min=0, step=1),
   tip="<code>-mg</code>. Only meaningful with SPLIT_MODE none or row.")
_p("EXTRA_ARGS", "", "Offload", "Extra llama-server arguments", "text", danger=True,
   tip="Appended verbatim (shell-quoted) to the command line, for flags this panel does "
       "not model - e.g. <code>--split-mode none</code>. Applied by instance launches only; "
       "main's legacy scripts ignore it. The panel REFUSES draft-model flags here "
       "(<code>-md</code>, <code>--spec-draft-model</code> - trap 1) and anything the "
       "launch plan owns (port, host, model, log file, API key).")

# ----------------------------------------------------------------------------
# Merge into the structures the rest of the panel already uses. Done as a merge
# rather than by editing the literals above so that the audit table stays one
# reviewable block that can be diffed against a new --help in one pass.
# ----------------------------------------------------------------------------
for _k, (_default, _meta, _spin, _presets) in _NEW.items():
    if _k in DEFAULTS:
        raise RuntimeError(f"parameter {_k} defined twice")
    DEFAULTS[_k] = _default
    PARAM_META[_k] = _meta
    if _spin:
        SPIN[_k] = _spin
    if _presets is not None and not _meta.get("options"):
        EXTRA_PRESETS[_k] = _presets
del _NEW, _p


for _k, _spin in SPIN.items():
    if _k in PARAM_META:
        PARAM_META[_k]["numeric"] = True
        PARAM_META[_k]["spin"] = _spin
        if not PARAM_META[_k].get("options"):
            PARAM_META[_k]["options"] = EXTRA_PRESETS.get(_k, [])

# Groups that only mean something on some backends; the UI hides the rest.
GROUP_BACKENDS = {"Vulkan (RADV)": ["vulkan"], "ROCm (HIP)": ["rocm", "cuda"],
                  "Offload": ["vulkan", "rocm", "cuda", "cpu"]}

PARAM_GROUPS = ["Backend", "Offload", "Context & memory", "Long context (RoPE)", "Throughput",
                "Speculative decoding", "Speculative (n-gram)",
                "ROCm (HIP)", "Vulkan (RADV)", "Vision", "Generation", "Reasoning", "Sampling",
                "Safety", "Server"]


# ============================================================================
# MEMORY CALCULATOR
# ----------------------------------------------------------------------------
# Self-calibrating: the per-token KV cost is solved from a LIVE measurement of
# this box (DRM fdinfo drm-total-vram for the running server) rather than from
# a datasheet, so the estimate tracks reality as the config changes. If nothing
# is running it falls back to a recorded anchor and says so.
# ============================================================================

# Bits per weight including block scale/min overhead.
KV_BPW = {"f32":32.0,"f16":16.0,"bf16":16.0,"q8_0":8.5,"q5_1":6.0,"q5_0":5.5,"q4_1":5.0,
          "q4_0":4.5,"iq4_nl":4.5}

# Measured on this box / cited to the launch-script headers.
MMPROJ_ONCARD_MIB = 1733   # ~848 projector + ~885 mtmd worst-case (desktop v4)
MTP_MIB           = 370    # embedded NextN head, measured
COMPUTE_BASE_MIB  = 400    # graph + prefill compute buffer at ub=512

# Fallback anchor, used only when nothing is running: ctx/kv -> total VRAM MiB.
FALLBACK_ANCHOR = dict(ctx=244736, kv="q5_1", mmproj_on_card=True, spec=True,
                       total_mib=23013)

# Decode t/s vs context depth. MEASURED on this box: Vulkan b10766, q5_1,
# embedded MTP on. Interpolated between these points.
SPEED_VULKAN_DECODE = [(0, 66.0), (20000, 56.0), (50000, 40.0),
                       (117000, 30.0), (156000, 26.0), (244736, 22.0)]
SPEED_VULKAN_PREFILL = 460.0   # t/s, warm, measured range 400-510


def _live_vram_total_mib():
    """drm-total-vram for the running llama-server, in MiB. None if not up."""
    pid = server_pid()
    if not pid:
        return None
    try:
        for fd in Path(f"/proc/{pid}/fdinfo").iterdir():
            try:
                txt = fd.read_text()
            except OSError:
                continue
            if "drm-total-vram" in txt:
                for line in txt.splitlines():
                    if line.startswith("drm-total-vram"):
                        return int(line.split()[1]) // 1024   # KiB -> MiB
    except OSError:
        pass
    return None


CALIB_FILE = _DynPath(lambda: INST()["dir"] / ".calibration.json")


def _load_calib():
    try:
        return json.loads(CALIB_FILE.read_text())
    except Exception:
        return {}


def _save_calib(backend, rec):
    """Remember one backend's measured VRAM anchor so the calculator can model
    it while a DIFFERENT backend is running."""
    d = _load_calib()
    d[backend] = rec
    try:
        CALIB_FILE.write_text(json.dumps(d, indent=1))
    except OSError:
        pass


def _kv_per_token_mib(backend="vulkan"):
    """Solve per-token KV cost at q5_1 for ONE backend.

    Uses the live process when it IS that backend, and the stored calibration
    from that backend's last run otherwise. Anchoring both backends on
    whichever happens to be running gave Vulkan estimates built from ROCm
    measurements - the fixed overheads differ, so the numbers were wrong for
    whichever one was not up.

    Returns (mib_per_token, basis).
    """
    live = _live_vram_total_mib()
    running_backend = None
    pid0 = server_pid()
    if pid0:
        try:
            exe = os.path.realpath(f"/proc/{pid0}/exe")
            running_backend = next((b for b in ("rocm", "cuda", "cpu") if b in exe), "vulkan")
        except OSError:
            pass
    if live and running_backend == backend:
        # Anchor on what the RUNNING process actually launched with, not on the
        # saved params - they drift apart the moment someone edits the form, and
        # attributing the projector to the KV term silently poisons every
        # estimate. Parse the real command line.
        pid = server_pid()
        argv = []
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except OSError:
            pass
        def _arg(flag, default=None):
            return argv[argv.index(flag) + 1] if flag in argv else default
        try:
            ctx = int(_arg("-c", 0) or 0)
        except ValueError:
            ctx = 0
        kv = _arg("--cache-type-k", "q5_1") or "q5_1"
        mm = ("--mmproj" in argv) and ("--no-mmproj-offload" not in argv)
        spec = "draft-" in str(_arg("--spec-type", "") or "")
        if ctx > 0 and kv in KV_BPW:
            weights = _weights_mib()
            kv_mib = live - weights - (MMPROJ_ONCARD_MIB if mm else 0) \
                     - (MTP_MIB if spec else 0) - COMPUTE_BASE_MIB
            if kv_mib > 0:
                per_tok_at_kv = kv_mib / ctx
                _save_calib(backend, dict(per_tok=per_tok_at_kv * 6.0 / KV_BPW[kv],
                                          total_mib=live, ctx=ctx, kv=kv,
                                          mmproj=mm, spec=spec, at=time.time()))
                return (per_tok_at_kv * 6.0 / KV_BPW[kv],
                        f"live process: {live} MiB at ctx={ctx} kv={kv} "
                        f"mmproj={'on-card' if mm else 'off-card'} spec={'on' if spec else 'off'}")
    # Not the running backend: use that backend's own stored calibration.
    rec = _load_calib().get(backend)
    if rec and rec.get("per_tok"):
        import datetime as _dt
        when = _dt.datetime.fromtimestamp(rec["at"]).strftime("%Y-%m-%d %H:%M")
        return (rec["per_tok"],
                f"stored {backend} calibration from {when} "
                f"({rec['total_mib']} MiB at ctx={rec['ctx']} kv={rec['kv']}); "
                f"{running_backend or 'nothing'} is running now")
    if not INST()["legacy"]:
        g = gguf_kv_per_token_mib(load_params().get("MODEL"))
        if g:
            return g, ("from the GGUF header (layers x KV heads x head dim) - not yet "
                       "measured on this instance; it calibrates itself once it runs")
    a = FALLBACK_ANCHOR
    kv_mib = a["total_mib"] - _weights_mib() - MMPROJ_ONCARD_MIB - MTP_MIB - COMPUTE_BASE_MIB
    return (kv_mib / a["ctx"] * 6.0 / KV_BPW[a["kv"]],
            f"recorded default anchor - no {backend} run has been measured yet")


def _arch_sig(model):
    h = gguf_header(model) if model else {}
    return f"{h.get('arch')}:{h.get('blocks')}" if h.get("arch") else None


def _kv_model(model, backend):
    """(q5_1-normalised MiB per token, fixed overhead MiB, basis) for `model`.

    Added 2026-09-23. The older model (_kv_per_token_mib) charged EVERYTHING the
    live process used beyond weights, draft and a 400 MiB constant to the KV
    cache and scaled it with context. On a hybrid model (qwen35: KV only on the
    full-attention layers, a fixed-size recurrent state on the rest) that
    charged ~685 MiB of fixed state and compute buffer per token, so every
    other context size was mis-estimated, and it used the RUNNING model's cost
    even when the form named a different model.

    Now: per-token cost from the header of the model being estimated; the fixed
    part (recurrent state + compute buffers) measured from the live process
    when it runs this same model on this backend, else remembered from the last
    such run, else the old 400 MiB constant.
    """
    hdr = gguf_kv_per_token_mib(model) if model else None
    if not hdr:
        per, basis = _kv_per_token_mib(backend)
        return per, COMPUTE_BASE_MIB, basis
    pid = server_pid()
    live = _live_vram_total_mib() if pid else None
    if live:
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            exe = os.path.realpath(f"/proc/{pid}/exe")
        except OSError:
            argv, exe = [], ""
        lb = next((b for b in ("rocm", "cuda", "cpu") if b in exe), "vulkan")
        lm = _argv_get(argv, ("-m", "--model"))
        try:
            lctx = int(_argv_get(argv, ("-c", "--ctx-size")) or 0)
        except ValueError:
            lctx = 0
        lkv = _argv_get(argv, ("--cache-type-k", "-ctk")) or "f16"
        if lb == backend and lm and os.path.realpath(lm) == os.path.realpath(model) \
                and lctx and lkv in KV_BPW:
            mm = "--mmproj" in argv and "--no-mmproj-offload" not in argv
            spec = "draft-" in str(_argv_get(argv, ("--spec-type",)) or "")
            fixed = live - _weights_mib(model) - (MMPROJ_ONCARD_MIB if mm else 0) \
                - (MTP_MIB if spec else 0) - hdr * KV_BPW[lkv] / 6.0 * lctx
            if fixed > 0:
                d = _load_calib().get(backend) or {}
                d.update(fixed_mib=round(fixed), fixed_model=os.path.realpath(model),
                         fixed_sig=_arch_sig(model), fixed_at=time.time())
                _save_calib(backend, d)
                return hdr, round(fixed), (
                    f"KV from the model header; {round(fixed)} MiB fixed overhead (recurrent "
                    f"state + compute buffers) measured from the running server "
                    f"({live} MiB at ctx={lctx} kv={lkv})")
    rec = _load_calib().get(backend) or {}
    if rec.get("fixed_mib") and rec.get("fixed_model") == os.path.realpath(model):
        return hdr, rec["fixed_mib"], (
            f"KV from the model header; {rec['fixed_mib']} MiB fixed overhead measured on "
            f"this instance's last {backend} run of this model")
    if rec.get("fixed_mib") and rec.get("fixed_sig") and rec["fixed_sig"] == _arch_sig(model):
        # same architecture and depth: same recurrent state, same compute graph
        return hdr, rec["fixed_mib"], (
            f"KV from the model header; {rec['fixed_mib']} MiB fixed overhead measured on "
            f"{Path(rec['fixed_model']).name}, which has the same architecture")
    return hdr, COMPUTE_BASE_MIB, ("KV from the model header; fixed overhead is the "
                                   f"{COMPUTE_BASE_MIB} MiB default until this model runs "
                                   "here (hybrid models add a few hundred MiB)")


_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def model_shards(path):
    """All files of a split GGUF (name-00001-of-00003.gguf), or [path].
    llama.cpp loads the rest itself when given shard 1."""
    p = Path(str(path or ""))
    m = _SHARD_RE.match(p.name)
    if not m:
        return [p]
    n = int(m.group(3))
    return [p.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def model_bytes(path):
    """Total size across shards; None when any shard is missing."""
    total = 0
    for f in model_shards(path):
        try:
            total += f.stat().st_size
        except OSError:
            return None
    return total


def _weights_mib(model=None):
    """Model weight footprint, from the actual file size(s) on disk. `model` is
    the file being ESTIMATED (the form's choice); default the saved one."""
    b = model_bytes(model or load_params().get("MODEL", ""))
    return b // 1048576 if b else 14982


def _interp(table, x):
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


# ============================================================================
# TENSOR PLACEMENT  (added 2026-09-17)
#
# estimate() used to put the whole model file on one pooled "card". That is
# right for main (a dense 27B that fits) and absurd for flash-next: an 81 GB
# MoE whose experts are sent to the CPU by OVERRIDE_TENSORS, with a 27 GB
# per-layer embedding table read lazily from disk, across two GPUs. It showed
# 86,825 MiB against 30,704 and an 86 GB host-RAM worst case - an instant OOM
# on paper for a config that places ~15 GB on the XTX and ~1 GB on the 2060.
#
# This reads every tensor's real size from the GGUF header (all shards) and
# assigns it the way llama.cpp does: -ot patterns first (first match wins),
# input embeddings on the CPU, repeating layers by -ngl and -ts, output with
# the last offloaded layer. Budgets are then checked PER DEVICE, because free
# VRAM on the 2060 does not help an over-full XTX.
# ============================================================================
_layout_cache = {}


def gguf_tensor_layout(path):
    """(kv, {tensor name: bytes}) across every shard, or None.

    Sizes come from the gaps between data offsets, not from a table of ggml
    type sizes, so new quant types cannot make it wrong."""
    shards = model_shards(path)
    try:
        key = tuple((str(s), s.stat().st_mtime_ns) for s in shards)
    except OSError:
        return None
    if key in _layout_cache:
        return _layout_cache[key]
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    kv, sizes = {}, {}
    try:
        for n_shard, shard in enumerate(shards):
            end = shard.stat().st_size
            with open(shard, "rb") as f:
                if f.read(4) != b"GGUF":
                    return None
                f.read(4)
                n_t, n_kv = struct.unpack("<QQ", f.read(16))
                rd_str = lambda: f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
                align = 32
                for _ in range(n_kv):
                    k = rd_str()
                    t = struct.unpack("<I", f.read(4))[0]
                    if t == 8:
                        val = rd_str()
                    elif t == 9:
                        et, ln = struct.unpack("<IQ", f.read(12))
                        if et == 8:
                            for _i in range(ln):
                                rd_str()
                        else:
                            f.read(SZ.get(et, 4) * ln)
                        val = None
                    else:
                        val = int.from_bytes(f.read(SZ.get(t, 4)), "little") if t not in (6, 12) \
                            else f.read(SZ[t]) and None
                    if n_shard == 0:
                        kv[k] = val
                    if k == "general.alignment" and isinstance(val, int) and val:
                        align = val
                infos = []
                for _ in range(n_t):
                    name = rd_str()
                    nd = struct.unpack("<I", f.read(4))[0]
                    f.read(8 * nd)
                    _ty, off = struct.unpack("<IQ", f.read(12))
                    infos.append((off, name))
                data = (f.tell() + align - 1) // align * align
            infos.sort()
            for i, (off, name) in enumerate(infos):
                nxt = infos[i + 1][0] if i + 1 < len(infos) else end - data
                sizes[name] = max(nxt - off, 0)
    except (OSError, struct.error, UnicodeDecodeError):
        return None
    res = (kv, sizes)
    _layout_cache.clear()           # one model at a time is all the form needs
    _layout_cache[key] = res
    return res


_BACKEND_DEV_PREFIX = {"vulkan": "Vulkan", "cuda": "CUDA", "rocm": "ROCm"}


def _instance_dev_names(backend):
    """[(llama.cpp device name, device record)] in instance order.

    Vulkan names come from the build's own enumeration when there is more than
    one card (the loader decides the order, see vulkan_enumeration); anything
    that cannot be resolved falls back to instance order."""
    inst = INST()
    pcis = [p for p in (inst.get("devices") or [inst.get("device")]) if p and p != "cpu"]
    recs = [(_device_record(p) or dict(pci=p, name=p)) for p in pcis]
    prefix = _BACKEND_DEV_PREFIX.get(backend)
    if not prefix:
        return []
    names = [f"{prefix}{i}" for i in range(len(recs))]
    if backend == "vulkan" and len(recs) > 1:
        try:
            icds = []
            for r in recs:
                icd = _icd_for_driver(r.get("driver"))
                if icd and icd not in icds:
                    icds.append(icd)
            bd = active_build("vulkan")
            enum = vulkan_enumeration(bd, icds) if bd and icds else []
            got = []
            for r in recs:
                hit = [e for e in enum if _vk_matches(r, e[1])]
                got.append(f"Vulkan{hit[0][0]}" if len(hit) == 1 else None)
            if all(got) and len(set(got)) == len(got):
                names = got
        except Exception:
            pass
    return list(zip(names, recs))


_INPUT_TENSORS = ("token_embd", "per_layer_token_embd", "token_types", "position_embd")
_BLK_RE = re.compile(r"^blk\.(\d+)\.")


def estimate_placement(params, layout=None, detail=False):
    """Where every weight lands for this parameter set, or None when the model
    header cannot be read. See the block comment above.

    layout=(kv, {name: bytes}) places a model that does not exist yet (the Fit
    solver); detail=True adds where={tensor: device index or None for CPU}."""
    g = lambda k, d="": str(params.get(k, d) if params.get(k) is not None else d).strip()
    lay = layout or gguf_tensor_layout(g("MODEL"))
    if not lay:
        return None
    kv, sizes = lay
    backend = g("BACKEND", "vulkan") or "vulkan"
    devs = _instance_dev_names(backend) if backend != "cpu" else []
    try:
        ngl = int(g("NGL", "99") or 99)
    except ValueError:
        ngl = 99
    if ngl < 0:                                   # -1 / auto: llama.cpp offloads all it can
        ngl = 999
    blocks = {int(m.group(1)) for n in sizes for m in [_BLK_RE.match(n)] if m}
    n_layer = (max(blocks) + 1) if blocks else 0

    # -ts proportions; llama.cpp defaults to free memory, total VRAM is close enough
    try:
        split = [float(x) for x in g("TENSOR_SPLIT").replace("/", ",").split(",") if x.strip()]
    except ValueError:
        split = []
    if len(split) != len(devs) or not sum(split):
        split = [float((r.get("vram_total_mib") or 1)) for _, r in devs]
    if g("SPLIT_MODE") == "none" and devs:
        mg = int(g("MAIN_GPU", "0") or 0) if g("MAIN_GPU").isdigit() else 0
        split = [1.0 if i == mg else 0.0 for i in range(len(devs))]
    tot = sum(split) or 1.0
    cum, acc = [], 0.0
    for s in split:
        acc += s / tot
        cum.append(acc)
    i_gpu_start = max(n_layer - ngl, 0) if devs else n_layer
    n_gpu = n_layer - i_gpu_start

    def layer_dev(il):
        """index into devs, or None for CPU."""
        if not devs or il < i_gpu_start:
            return None
        frac = (il - i_gpu_start) / max(n_gpu, 1)
        for i, c in enumerate(cum):
            if frac < c - 1e-9:
                return i
        return len(devs) - 1

    out_dev = layer_dev(n_layer - 1) if (devs and ngl > n_layer) else None

    # -ot: "regex=buft,regex=buft", first match wins, same as llama.cpp
    rules = []
    for part in [p for p in g("OVERRIDE_TENSORS").split(",") if "=" in p]:
        pat, _, buft = part.rpartition("=")
        try:
            rx = re.compile(pat)
        except re.error:
            continue
        b = buft.strip()
        if b.upper() == "CPU" or b.endswith("_Host") or b.startswith("CPU"):
            rules.append((rx, None))
        else:
            idx = next((i for i, (n, _) in enumerate(devs) if n == b), "unknown")
            rules.append((rx, idx))
    if str(g("CPU_MOE")) in ("1", "true", "on"):
        rules.append((re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"), None))
    try:
        ncm = int(g("N_CPU_MOE", "0") or 0)
    except ValueError:
        ncm = 0
    for il in range(ncm):
        rules.insert(0, (re.compile(rf"blk\.{il}\.ffn_(up|down|gate|gate_up)_(ch|)exps"), None))

    lazy_on = g("LAZY_MODE", "auto") or "auto"
    lm = g("LOAD_MODE")
    mapped = lm not in ("none", "mlock", "mmap+mlock", "dio")
    MIB = 1048576
    dev_w = [0] * len(devs)
    dev_layers = [set() for _ in devs]
    cpu_resident = cpu_mapped = lazy = 0
    cpu_layer_exps = {}            # layer -> CPU expert bytes (op-offload staging)
    unknown_dev = []
    where = {}
    for name, b in sizes.items():
        m = _BLK_RE.match(name)
        il = int(m.group(1)) if m else None
        hit = next((d for rx, d in rules if rx.search(name)), "none")
        if hit == "unknown":
            unknown_dev.append(name)
            hit = 0 if devs else None
        if hit != "none":
            d = hit
        elif il is not None:
            d = layer_dev(il)
        elif name.split(".")[0] in _INPUT_TENSORS:
            d = None
        else:
            d = out_dev
        where[name] = d
        if d is None:
            big_lazy = name.startswith("per_layer_token_embd") and (
                lazy_on == "on" or (lazy_on == "auto" and b > 4 * 1024 * MIB))
            if big_lazy and mapped:
                lazy += b
            elif mapped:
                cpu_mapped += b
            else:
                cpu_resident += b
            if il is not None and "_exps" in name:
                cpu_layer_exps[il] = cpu_layer_exps.get(il, 0) + b
        else:
            dev_w[d] += b
            if il is not None and "_exps" not in name:
                dev_layers[d].add(il)

    # KV and recurrent state live with the layer's attention, not its experts
    kv_frac = []
    layered = sum(len(s) for s in dev_layers)
    for s in dev_layers:
        kv_frac.append(len(s) / n_layer if n_layer else 0.0)
    return dict(
        devices=[dict(name=n, pci=r.get("pci"), label=r.get("name"),
                      vendor=r.get("vendor"), driver=r.get("driver"),
                      vram_total_mib=r.get("vram_total_mib") or 0,
                      weights_mib=round(dev_w[i] / MIB), layers=len(dev_layers[i]),
                      kv_frac=kv_frac[i])
                 for i, (n, r) in enumerate(devs)],
        cpu_kv_frac=max(1.0 - sum(kv_frac), 0.0) if n_layer else 1.0,
        n_layer=n_layer, ngl=ngl, gpu_layers=layered,
        cpu_resident_mib=round(cpu_resident / MIB), cpu_mapped_mib=round(cpu_mapped / MIB),
        lazy_mib=round(lazy / MIB),
        max_cpu_layer_exps_mib=round(max(cpu_layer_exps.values(), default=0) / MIB),
        model_mib=round(sum(sizes.values()) / MIB),
        unknown_device_tensors=len(unknown_dev),
        simple=(len(devs) == 1 and not rules and ngl >= n_layer),
        arch=kv.get("general.architecture"),
        **(dict(where=where) if detail else {}))


def _device_footprints(params, pl):
    """What this instance allocates on each device of a placement estimate:
    weights, KV share, compute buffer, op-offload staging, projector and draft
    context. Shared by the VRAM calculator and the host-RAM budget, so the card
    view and the eviction view cannot disagree about the same allocation."""
    g = lambda k, d=None: params.get(k, load_params().get(k, d))
    ctx     = int(g("CTX", 131072) or 131072)
    kvt     = str(g("KV_TYPE", "q5_1") or "q5_1")
    use_mm  = str(g("USE_MMPROJ", 1)) in ("1", "true", "True")
    mm_card = str(g("MMPROJ_OFFLOAD", 0)) in ("1", "true", "True")
    spec    = any(t.strip().startswith("draft-")
                  for t in str(g("SPEC_TYPE", "") or "").split(","))
    per_tok, _ = _kv_per_token_mib(str(g("BACKEND", "vulkan") or "vulkan"))
    kv_mib  = ctx * per_tok * (KV_BPW.get(kvt, 6.0) / 6.0)
    draft_path = str(g("SPEC_DRAFT_MODEL", "") or "")
    mtp = 0
    if spec:
        try:
            mtp = draft_vram_mib(os.path.getsize(draft_path)) if draft_path else MTP_MIB
        except OSError:
            mtp = MTP_MIB
    try:
        ub = int(g("UBATCH", 512) or 512)
    except ValueError:
        ub = 512
    op_offload = str(g("OP_OFFLOAD", "") or "") not in ("0", "off", "false")
    names = [d["name"] for d in pl["devices"]]
    def _dev_idx(key, default=0):
        n = str(g(key, "") or "")
        return names.index(n) if n in names else default
    main_i = next((i for i, d in enumerate(pl["devices"]) if d["layers"]), 0)
    mm_i = _dev_idx("MMPROJ_DEVICE", main_i)
    draft_i = _dev_idx("SPEC_DRAFT_DEVICE", main_i)
    mm_mib = 0
    if use_mm and mm_card:
        try:
            mm_mib = os.path.getsize(str(g("MMPROJ", ""))) // 1048576 + 885
        except OSError:
            mm_mib = MMPROJ_ONCARD_MIB
    devices = []
    for i, d in enumerate(pl["devices"]):
        w = d["weights_mib"]
        kv_d = kv_mib * d["kv_frac"]
        # graph buffer scales with ubatch. A card holding only overridden expert
        # tensors runs a fraction of the graph, so it gets half.
        comp = COMPUTE_BASE_MIB * ub / 512 * (1.0 if d["layers"] else 0.5)
        # op offload: a large batch runs CPU experts on the main GPU by staging
        # one layer's expert tensors in its compute buffer
        stage = pl["max_cpu_layer_exps_mib"] if (op_offload and i == main_i) else 0
        mm_d = mm_mib if i == mm_i else 0
        mtp_d = mtp if (spec and i == draft_i) else 0
        if not w and not d["layers"] and not mm_d and not mtp_d:
            comp = 0
        t = w + kv_d + comp + stage + mm_d + mtp_d
        vt = d["vram_total_mib"] or 1
        devices.append(dict(name=d["name"], label=d["label"], vendor=d["vendor"],
                            pci=d.get("pci"),
                            weights_mib=w, kv_mib=round(kv_d), compute_mib=round(comp),
                            offload_stage_mib=stage, mmproj_mib=mm_d, mtp_mib=mtp_d,
                            total_mib=round(t), vram_total_mib=vt,
                            headroom_mib=round(vt - t), pct_used=round(100.0 * t / vt, 1)))
    return devices


def estimate(params):
    """VRAM/RAM breakdown + expected speed for a candidate parameter set."""
    g = lambda k, d=None: params.get(k, load_params().get(k, d))

    ctx     = int(g("CTX", 131072) or 131072)
    kvt     = str(g("KV_TYPE", "q5_1") or "q5_1")
    use_mm  = str(g("USE_MMPROJ", 1)) in ("1", "true", "True")
    mm_card = str(g("MMPROJ_OFFLOAD", 0)) in ("1", "true", "True")
    spec_types = [t.strip() for t in str(g("SPEC_TYPE", "") or "").split(",") if t.strip()]
    # ngram-* speculation is a host-side token lookup with no weights and no
    # device memory. Only draft-* types load or build a draft context.
    spec    = any(t.startswith("draft-") for t in spec_types)
    backend = str(g("BACKEND", "vulkan") or "vulkan")
    cache_ram = int(g("CACHE_RAM", 8192) or 8192)

    model   = str(g("MODEL", "") or "")
    per_tok, fixed_mib, kv_basis = _kv_model(model, backend)
    if kvt not in KV_BPW:
        kv_basis += f"; KV type {kvt} is not in the size table, counted as q5_1"
    kv_mib   = ctx * per_tok * (KV_BPW.get(kvt, 6.0) / 6.0)
    weights  = _weights_mib(model)
    mmproj   = MMPROJ_ONCARD_MIB if (use_mm and mm_card) else 0
    # A separate sidecar replaces the embedded head and costs its own weights.
    draft_path = str(g("SPEC_DRAFT_MODEL", "") or "")
    if spec and draft_path:
        try:
            mtp = draft_vram_mib(os.path.getsize(draft_path))
        except OSError:
            mtp = MTP_MIB
    else:
        mtp = MTP_MIB if spec else 0

    total    = weights + mmproj + mtp + kv_mib + fixed_mib
    vram_tot = sum((_device_record(p) or {}).get("vram_total_mib") or 0
                   for p in (INST().get("devices") or [INST().get("device")])) \
        or read_int(f"{CARD}/mem_info_vram_total") // 1048576 or 24560
    # Other servers already on these cards (another instance, an image server)
    # take their share first - the calculator used to assume a card to itself.
    mine = server_pid()
    my_devs = set(INST().get("devices") or [INST().get("device")])
    others, others_by = 0, []
    for srv in list_servers():
        if srv.get("pid") == mine:
            continue
        use = sum(gg.get("vram_mib") or 0 for gg in srv.get("gpu") or [] if gg.get("pdev") in my_devs)
        if use:
            others += use
            others_by.append(f"{srv.get('instance') or 'pid ' + str(srv.get('pid'))} {use} MiB")
    headroom = vram_tot - total - others
    pct_used = 100.0 * (total + others) / vram_tot if vram_tot else 0.0

    # Host RAM. The failure mode this box actually hit: VRAM near-full makes
    # amdgpu evict weights into GTT, which is host RAM, and that plus the
    # prompt cache is what exhausts 30.67 GB.
    try:
        ram_tot = int([l for l in Path("/proc/meminfo").read_text().splitlines()
                       if l.startswith("MemTotal")][0].split()[1]) // 1024
    except Exception:
        ram_tot = 30674
    ram_used_est = cache_ram + 1000 + 2000          # cache + llama base + system
    evict_risk   = pct_used >= 90.0
    ram_worst    = ram_used_est + (weights if evict_risk else 500)

    warnings, notes, devices = [], [], []
    ram_extra = {}
    pl = None
    if not INST()["legacy"]:
        try:
            pl = estimate_placement({**load_params(), **params})
        except Exception as e:                  # never let the calculator 500 the form
            notes.append(f"placement model failed ({e}); showing the whole-file estimate")
    if pl and not pl["simple"] and pl["devices"]:
        # ---- per-device budget: each card has to fit on its own
        op_offload = str(g("OP_OFFLOAD", "") or "") not in ("0", "off", "false")
        names = [d["name"] for d in pl["devices"]]
        main_i = next((i for i, d in enumerate(pl["devices"]) if d["layers"]), 0)
        devices = _device_footprints(params, pl)
        weights = sum(d["weights_mib"] for d in devices)
        mmproj = sum(d["mmproj_mib"] for d in devices)
        kv_mib = sum(d["kv_mib"] for d in devices)
        comp_all = sum(d["compute_mib"] + d["offload_stage_mib"] for d in devices)
        total = sum(d["total_mib"] for d in devices)
        vram_tot = sum(d["vram_total_mib"] for d in devices)
        # the tightest card is the one that fails, so it drives the headline numbers
        tight = min(devices, key=lambda d: d["headroom_mib"])
        headroom = tight["headroom_mib"]
        pct_used = max(d["pct_used"] for d in devices)
        for d in devices:
            short = d["name"] + " (" + ("7900 XTX" if "Navi 31" in str(d["label"]) else
                                        "RTX 2060" if "TU106" in str(d["label"]) else
                                        str(d["label"])[:40]) + ")"
            if d["headroom_mib"] < 0:
                warnings.append(f"{short}: OVER BUDGET by {-d['headroom_mib']} MiB "
                                f"({d['total_mib']}/{d['vram_total_mib']} MiB).")
            elif d["pct_used"] >= 90:
                warnings.append(f"{short}: {d['pct_used']}% of VRAM. Above ~90% a prefill "
                                "spike can push the rest into eviction or an allocation failure.")
        # host RAM: only the GPU part of the weights can be evicted to GTT, and only by
        # amdgpu; experts read through mmap are page cache, which the kernel reclaims
        gtt_cap = _gtt_ceiling_mb() or 15706
        evict = sum(min(d["weights_mib"], gtt_cap) for d in devices
                    if d["vendor"] == "amd" and d["pct_used"] >= 90)
        mm_host = 896 if (use_mm and not mm_card) else 0
        ram_used_est = cache_ram + 1000 + 2000 + mm_host + pl["cpu_resident_mib"] \
            + round(kv_mib * pl["cpu_kv_frac"])
        evict_risk = evict > 0
        ram_worst = ram_used_est + (evict if evict_risk else 500)
        ram_extra = dict(cpu_mapped_mb=pl["cpu_mapped_mib"], lazy_mb=pl["lazy_mib"],
                         cpu_resident_mb=pl["cpu_resident_mib"])
        if pl["cpu_mapped_mib"]:
            spare = ram_tot - ram_used_est
            notes.append(f"{pl['cpu_mapped_mib']} MiB of weights stay on the CPU, read through "
                         f"mmap. That is page cache the kernel can drop, so it is not counted "
                         f"as used. Only ~{max(spare, 0)} MB of it can be cached at once; the "
                         f"rest is read from NVMe when routed to, which costs speed, not memory.")
        if pl["lazy_mib"]:
            notes.append(f"{pl['lazy_mib']} MiB per-layer embedding table is read row by row "
                         f"from disk (LAZY_MODE), not kept resident.")
        if pl["max_cpu_layer_exps_mib"] and op_offload:
            notes.append(f"{pl['max_cpu_layer_exps_mib']} MiB on {names[main_i]} is the op-offload "
                         "staging area: large prefill batches copy one layer's CPU experts to "
                         "the GPU to run them. OP_OFFLOAD=off removes it and makes prefill slower.")
        if pl["unknown_device_tensors"]:
            warnings.append(f"OVERRIDE_TENSORS names a device this instance does not have; "
                            f"{pl['unknown_device_tensors']} tensors were counted on "
                            f"{names[0]}. llama.cpp will refuse to start.")
        if spec_types and not spec:
            notes.append(f"SPEC_TYPE={','.join(spec_types)} is host-side n-gram lookup: "
                         "no draft weights, no VRAM.")
        notes.append("Placement is read from the GGUF header and the override rules, not "
                     "measured. Compute buffers are scaled from main's measured "
                     f"{COMPUTE_BASE_MIB} MiB at ub=512 and are the least certain part.")
        mtp = sum(d["mtp_mib"] for d in devices)
        compute_show = comp_all
    else:
        compute_show = fixed_mib
    if others:
        warnings.insert(0, f"{others} MiB of this card is already used by "
                           f"{', '.join(others_by)}. This instance only gets what is left"
                           + (" - stop the other first." if headroom < 0 else "."))

    if not devices:
        if headroom < 0:
            warnings.append(f"OVER BUDGET by {abs(headroom):.0f} MiB - this will fail to load, "
                            "or load into GTT and crawl.")
        elif pct_used >= 90:
            warnings.append(f"{pct_used:.1f}% of VRAM. Above ~90% a transient spike (prefill "
                            "compute buffer, mtmd buffer) makes amdgpu evict weights into GTT "
                            "- host RAM - which is what OOM'd this box on 2026-09-04.")
    if ram_worst > ram_tot:
        warnings.append(f"Host RAM worst case ~{ram_worst:.0f} MB of {ram_tot} MB. If the "
                        "GPU evicts, CACHE_RAM at this size does not fit.")
    if spec and draft_path:
        _d = next((o for o in draft_options() if o["value"] == draft_path), None)
        if _d and not _d["ok"]:
            warnings.append(f"SPEC_DRAFT_MODEL: {_d['note']}")
    if backend == "rocm":
        warnings.append("ROCm has no measured numbers on this box yet - the speed figures "
                        "below are the Vulkan measurements and do NOT describe ROCm.")

    # Prefer the curve fitted to this box's own request history; fall back to
    # the hand-derived table only when there is not enough logged data.
    curve = observed_curve(backend)
    if len(curve["points"]) >= 3:
        table = curve["points"]
        fitted = True
    else:
        table = SPEED_VULKAN_DECODE
        fitted = False
    decode  = _interp(table, ctx)
    # Use the rate for a prefill THIS SIZE, not the all-sizes median.
    pf_pts = curve.get("prefill_points") or []
    if len(pf_pts) >= 2:
        prefill = _interp(pf_pts, ctx)
        prefill_basis = (f"fitted by prompt size across {len(pf_pts)} buckets "
                         f"({curve['n_prefill']} samples)")
    else:
        prefill = curve["prefill_tps"] or SPEED_VULKAN_PREFILL
        prefill_basis = "all-sizes median (not enough data to fit by size)"

    if not fitted and backend != "vulkan":
        # No ROCm history yet: show the Vulkan curve but say plainly that it
        # does not describe ROCm.
        vc = observed_curve("vulkan")
        if len(vc["points"]) >= 3:
            table = vc["points"]
            decode = _interp(table, ctx)
            prefill = vc["prefill_tps"] or SPEED_VULKAN_PREFILL
        basis = (f"No {backend} requests logged on this box yet. Figures shown are the "
                 f"VULKAN history and do NOT describe {backend} - run bench_backends.sh.")
    elif fitted:
        basis = (f"Fitted to {backend} logs on this box: {curve['n_samples']} completed "
                 f"requests from {curve['n_logs']} log(s), {len(curve['points'])} depth "
                 f"buckets, {curve['n_prefill']} prefill samples. Updates as it serves "
                 f"more traffic. Includes MTP speculative decoding, unlike llama-bench.")
    else:
        basis = ("Hand-derived table - not enough logged requests to fit a curve "
                 "yet. Serve some traffic and this becomes measured.")

    speed = dict(
        decode_tps_at_ctx=round(decode, 1),
        decode_tps_shallow=round(_interp(table, 0), 1),
        prefill_tps=round(prefill, 0),
        full_prefill_s=round(ctx / prefill, 0) if prefill else None,
        measured=(backend == "vulkan" and fitted),
        fitted=fitted, n_samples=curve["n_samples"],
        prefill_basis=prefill_basis,
        prefill_curve=curve.get("prefill_buckets", []),
        curve=curve["buckets"],
        basis=basis)

    return dict(
        vram=dict(weights_mib=weights, mmproj_mib=mmproj, mtp_mib=mtp,
                  kv_mib=round(kv_mib), compute_mib=round(compute_show),
                  others_mib=others, others=others_by,
                  total_mib=round(total), card_total_mib=vram_tot,
                  headroom_mib=round(headroom), pct_used=round(pct_used, 1),
                  devices=devices),
        ram=dict(cache_ram_mb=cache_ram, est_normal_mb=ram_used_est,
                 est_worst_mb=round(ram_worst), total_mb=ram_tot,
                 evict_risk=evict_risk, **ram_extra),
        speed=speed, warnings=warnings, notes=notes,
        kv_basis=kv_basis, backend=backend)



# ============================================================================
# TIERS - saved parameter sets, and the fallback the launch script selects.
# ============================================================================
TIER_NAMES = ["normal", "safe", "minimal"]
FAIL_FILE = _DynPath(lambda: INST()["dir"] / ".launch-fails")


def tier_file(name):
    return INST()["dir"] / f"params-tier-{name}.env"


def read_tier(name):
    if name not in TIER_NAMES:
        raise ValueError(f"unknown tier {name!r}")
    f = tier_file(name)
    if not f.exists():
        return None
    cur = dict(DEFAULTS)
    cur.update(_read_env_file(f))
    for k, v in list(cur.items()):
        if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
            cur[k] = int(v)
    return cur


def write_tier(name, values):
    if name not in TIER_NAMES:
        raise ValueError(f"unknown tier {name!r}")
    cur = dict(DEFAULTS)
    cur.update({k: v for k, v in values.items() if k in DEFAULTS})
    e = estimate(cur)
    v, r = e["vram"], e["ram"]
    hdr = [f"# Tier '{name}' - saved from the panel {time.strftime('%Y-%m-%d %H:%M')}Z",
           f"# Estimated VRAM {v['total_mib']}/{v['card_total_mib']} MiB ({v['pct_used']}%), "
           f"headroom {v['headroom_mib']} MiB.",
           f"# Estimated host RAM worst case {r['est_worst_mb']}/{r['total_mb']} MB.",
           "# Sourced by the launch script when the failure counter selects this tier.", ""]
    _write_env_file(tier_file(name), cur, header=hdr)
    return cur


def fail_count():
    try:
        return int(FAIL_FILE.read_text().strip() or 0)
    except Exception:
        return 0


def tier_status():
    n = fail_count()
    active = "normal" if n == 0 else ("safe" if n == 1 else "minimal")
    out = {}
    for t in TIER_NAMES:
        v = read_tier(t)
        out[t] = None if v is None else {k: v[k] for k in
                 ("CTX", "KV_TYPE", "CACHE_RAM", "BATCH", "UBATCH",
                  "USE_MMPROJ", "MMPROJ_OFFLOAD", "SPEC_TYPE")}
    return dict(fails=n, next_tier=active, tiers=out,
                note=("The launch script picks a tier from consecutive failed starts: "
                      "0 = your saved params, 1 = safe, 2+ = minimal. The counter clears "
                      "once a start stays up 120s."))

# ============================================================================
# MODEL PROFILES - named parameter sets saved per model file.
#
# Tiers are per failure count and backends are per backend; neither remembers
# that one model wants ctx 245760/q4_1 and another 131072/q8_0, so switching
# model meant retyping the form. A profile is a full env file under
# profiles/<model basename>/<name>.env. Loading one fills the FORM only, the
# same as a tier: nothing runs until Save, and Save still applies every guard.
# ============================================================================
PROFILES = PANEL / "profiles"
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")
_PROFILE_SUMMARY = ("BACKEND", "CTX", "KV_TYPE", "CACHE_RAM", "BATCH", "UBATCH", "NGL",
                    "SPEC_TYPE", "SPEC_N_MAX", "USE_MMPROJ", "MMPROJ_OFFLOAD",
                    "TEMP", "REASONING_EFFORT")


def _profile_dir(model):
    base = Path(str(model or "")).name
    if not base:
        raise ValueError("a profile needs a MODEL")
    return PROFILES / re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _profile_path(model, name):
    name = str(name or "").strip()
    if not _PROFILE_NAME.match(name):
        raise ValueError("profile name: 1-64 of letters, digits, space, . _ -; "
                         "starting with a letter or digit")
    return _profile_dir(model) / f"{name}.env"


def _read_profile(f):
    cur = dict(DEFAULTS)
    cur.update(_read_env_file(f))
    for k, v in list(cur.items()):
        if isinstance(DEFAULTS.get(k), int) and str(v).lstrip("-").isdigit():
            cur[k] = int(v)
    return cur


def list_profiles(model=None):
    """Profiles, newest first; only `model`'s when given.

    Each carries `missing`: files it names that are no longer on disk. A
    profile is not a deletion guard (it would pin every model it was ever
    saved for), so it has to say when it has gone stale instead.
    """
    if not PROFILES.is_dir():
        return []
    dirs = [_profile_dir(model)] if model else sorted(PROFILES.iterdir())
    out = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.glob("*.env"):
            v = _read_profile(f)
            missing = [k for k in ("MODEL", "MMPROJ", "SPEC_DRAFT_MODEL", "TEMPLATE_SRC")
                       if v.get(k) and not (k == "MMPROJ" and str(v.get("USE_MMPROJ")) in ("0", "false"))
                       and not os.path.exists(str(v[k]))]
            out.append(dict(name=f.stem, model=v.get("MODEL"),
                            model_name=Path(str(v.get("MODEL") or "")).name,
                            saved=int(f.stat().st_mtime), missing=missing,
                            summary={k: v.get(k) for k in _PROFILE_SUMMARY}))
    return sorted(out, key=lambda x: -x["saved"])


def save_profile(name, values):
    cur = dict(DEFAULTS)
    cur.update({k: v for k, v in (values or {}).items() if k in DEFAULTS})
    model = str(cur.get("MODEL") or "")
    if not model or not os.path.exists(model):
        raise ValueError(f"MODEL does not exist: {model or '(empty)'}")
    # Same guard as Save. A profile is only a form fill, but one carrying a
    # full-size draft is a hard lock waiting for someone to press Save.
    _reject_full_size_draft(cur)
    f = _profile_path(model, name)
    hdr = [f"# Model profile '{f.stem}' for {Path(model).name}",
           f"# Saved from the panel {time.strftime('%Y-%m-%d %H:%M')}Z."]
    try:
        e = estimate(cur)
        hdr.append(f"# Estimated VRAM {e['vram']['total_mib']}/{e['vram']['card_total_mib']} MiB, "
                   f"host RAM worst case {e['ram']['est_worst_mb']}/{e['ram']['total_mb']} MB.")
    except Exception:
        pass
    hdr += ["# Loading it only fills the panel form; Save applies it.", ""]
    f.parent.mkdir(parents=True, exist_ok=True)
    _write_env_file(f, cur, header=hdr)
    return dict(ok=True, name=f.stem, model=model, params=cur)


def load_profile(model, name):
    f = _profile_path(model, name)
    if not f.exists():
        raise ValueError(f"no profile '{name}' for {Path(str(model)).name}")
    return dict(ok=True, name=f.stem, params=_read_profile(f))


def delete_profile(model, name):
    f = _profile_path(model, name)
    if not f.exists():
        raise ValueError(f"no profile '{name}' for {Path(str(model)).name}")
    f.unlink()
    try:
        f.parent.rmdir()                # only succeeds once the last one is gone
    except OSError:
        pass
    return dict(ok=True, deleted=f.stem)


# ============================================================================
# PROFILE MANAGEMENT - copy / rename / new for every saved parameter set.
#
# Three kinds of saved set exist, and each used to have its own partial verbs:
#   tier:<name>              per instance, picked by the failure counter
#   profile:<model>:<name>   per model file, shared by all instances
#   iprofile:<name>          a whole instance: every backend file, every tier,
#                            and the device list it was saved from
# resolve_source() turns any of those strings (plus defaults / current /
# instance:<id>) into a full parameter set, so every "copy X into Y" is one
# resolve plus one existing, guarded writer. Nothing here restarts anything.
# ============================================================================
# Keys that belong to the instance a set is applied TO, not the one it came
# from. Copying main's PORT into a second instance's tier would make that
# instance's fallback start collide with main on 8081.
_INSTANCE_LOCAL_KEYS = ("PORT", "HOST")


def _keep_local(vals, into):
    out = dict(vals)
    for k in _INSTANCE_LOCAL_KEYS:
        if k in into:
            out[k] = into[k]
    return out


def resolve_source(src):
    """Full parameter set for a source string, read in the CURRENT instance."""
    src = str(src or "current").strip()
    if src == "defaults":
        return dict(DEFAULTS)
    if src == "current":
        return load_params()
    kind, _, rest = src.partition(":")
    if kind == "tier":
        v = read_tier(rest)
        if v is None:
            raise ValueError(f"tier '{rest}' has not been saved on {INST()['id']}")
        return v
    if kind == "profile":
        model, _, name = rest.rpartition(":")      # names cannot hold ':'; paths might
        return load_profile(model, name)["params"]
    if kind == "instance":
        with using_instance(rest):
            return load_params()
    if kind == "iprofile":
        ip = read_instance_profile(rest)
        return dict(ip["params"][ip["meta"]["backend"]])
    raise ValueError(f"unknown source {src!r}: use defaults, current, tier:<name>, "
                     "profile:<model>:<name>, instance:<id> or iprofile:<name>")


def new_profile(name, src="current", model=None):
    vals = resolve_source(src)
    if model:
        vals["MODEL"] = str(model)
    elif src == "defaults":
        # DEFAULTS carries the original bring-up model; a new profile belongs to
        # whatever this instance runs now
        vals["MODEL"] = load_params().get("MODEL") or vals.get("MODEL")
    f = _profile_path(vals.get("MODEL"), name)
    if f.exists():
        raise ValueError(f"profile '{name}' already exists for {Path(str(vals.get('MODEL'))).name}")
    return save_profile(name, vals)


def copy_profile(model, name, new_name, new_model=None):
    vals = load_profile(model, name)["params"]
    if new_model:
        vals["MODEL"] = str(new_model)
    if _profile_path(vals["MODEL"], new_name).exists():
        raise ValueError(f"profile '{new_name}' already exists for {Path(str(vals['MODEL'])).name}")
    return save_profile(new_name, vals)


def rename_profile(model, name, new_name):
    if str(new_name).strip() == str(name).strip():
        raise ValueError("the new name is the same as the old one")
    r = copy_profile(model, name, new_name)
    delete_profile(model, name)
    return r


def delete_tier(name):
    f = tier_file(name) if name in TIER_NAMES else None
    if f is None:
        raise ValueError(f"unknown tier {name!r}")
    if not f.exists():
        raise ValueError(f"tier '{name}' is not saved on {INST()['id']}")
    f.unlink()
    return dict(ok=True, deleted=name,
                note="a start that selects this tier now keeps params.env instead")


def copy_into_tier(name, src):
    return write_tier(name, _keep_local(resolve_source(src), load_params()))


# --- instance profiles ------------------------------------------------------
IPROFILES = PANEL / "instance-profiles"


def _iprofile_dir(name):
    name = str(name or "").strip()
    if not _PROFILE_NAME.match(name):
        raise ValueError("profile name: 1-64 of letters, digits, space, . _ -; "
                         "starting with a letter or digit")
    return IPROFILES / name


def read_instance_profile(name):
    d = _iprofile_dir(name)
    try:
        meta = json.loads((d / "profile.json").read_text())
    except (OSError, ValueError):
        raise ValueError(f"no instance profile '{name}'")
    params = {b: _read_profile(d / f"params-{b}.env") for b in BACKENDS
              if (d / f"params-{b}.env").exists()}
    tiers = {t: (_read_profile(d / f"params-tier-{t}.env")
                 if (d / f"params-tier-{t}.env").exists() else None) for t in TIER_NAMES}
    if meta.get("backend") not in params:
        raise ValueError(f"instance profile '{name}' is missing its active backend file")
    return dict(meta=meta, params=params, tiers=tiers)


def list_instance_profiles():
    out = []
    if not IPROFILES.is_dir():
        return out
    for d in sorted(IPROFILES.iterdir()):
        try:
            ip = read_instance_profile(d.name)
        except ValueError:
            continue
        v = ip["params"][ip["meta"]["backend"]]
        missing = [k for k in ("MODEL", "MMPROJ", "SPEC_DRAFT_MODEL", "TEMPLATE_SRC")
                   if v.get(k) and not (k == "MMPROJ" and str(v.get("USE_MMPROJ")) in ("0", "false"))
                   and not os.path.exists(str(v[k]))]
        out.append(dict(ip["meta"], name=d.name, missing=missing,
                        backends=sorted(ip["params"]),
                        tiers=[t for t, tv in ip["tiers"].items() if tv],
                        model_name=Path(str(v.get("MODEL") or "")).name,
                        summary={k: v.get(k) for k in _PROFILE_SUMMARY}))
    return sorted(out, key=lambda x: -int(x.get("saved_ts") or 0))


def save_instance_profile(name, iid=None, overwrite=False):
    """Snapshot an instance: every backend file, every saved tier, its devices."""
    import shutil
    inst = get_instance(iid or INST()["id"])
    d = _iprofile_dir(name)
    if d.exists() and not overwrite:
        raise ValueError(f"instance profile '{d.name}' already exists")
    tmp = IPROFILES / f".tmp-{d.name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with using_instance(inst):
            active = active_backend()
            for b in BACKENDS:
                if b == active or _backend_file(b).exists():
                    v = load_params(b)
                    _reject_full_size_draft(v)
                    _write_env_file(tmp / f"params-{b}.env", v, header=[
                        f"# Instance profile '{d.name}', backend {b}, from instance '{inst['id']}'.", ""])
            for t in TIER_NAMES:
                tv = read_tier(t)
                if tv is not None:
                    _write_env_file(tmp / f"params-tier-{t}.env", tv, header=[
                        f"# Instance profile '{d.name}', tier {t}, from instance '{inst['id']}'.", ""])
            model = load_params(active).get("MODEL")
        (tmp / "profile.json").write_text(json.dumps(dict(
            saved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), saved_ts=int(time.time()),
            source_instance=inst["id"], source_name=inst["name"],
            devices=inst.get("devices") or [inst["device"]],
            device_names=[(_device_record(p) or {}).get("name") or p
                          for p in (inst.get("devices") or [inst["device"]])],
            backend=active, model=model), indent=1) + "\n")
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(ok=True, name=d.name)


def copy_instance_profile(name, new_name):
    import shutil
    src, dst = _iprofile_dir(name), _iprofile_dir(new_name)
    read_instance_profile(name)
    if dst.exists():
        raise ValueError(f"instance profile '{dst.name}' already exists")
    shutil.copytree(src, dst)
    meta = json.loads((dst / "profile.json").read_text())
    meta.update(saved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                saved_ts=int(time.time()), copied_from=src.name)
    (dst / "profile.json").write_text(json.dumps(meta, indent=1) + "\n")
    return dict(ok=True, name=dst.name)


def rename_instance_profile(name, new_name):
    src, dst = _iprofile_dir(name), _iprofile_dir(new_name)
    read_instance_profile(name)
    if dst.exists():
        raise ValueError(f"instance profile '{dst.name}' already exists")
    src.rename(dst)
    return dict(ok=True, name=dst.name)


def delete_instance_profile(name, confirm):
    import shutil
    d = _iprofile_dir(name)
    read_instance_profile(name)
    if confirm != d.name:
        raise ValueError("confirm must equal the profile name")
    shutil.rmtree(d)
    return dict(ok=True, deleted=d.name)


def apply_instance_profile(name, iid=None, parts=("params", "tiers")):
    """Write a saved instance into an existing instance. PORT/HOST stay the
    target's own. Parameters go through save_params, so every guard applies;
    devices go through update_instance, which refuses while running."""
    ip = read_instance_profile(name)
    inst = get_instance(iid or INST()["id"])
    parts = set(parts or ())
    done = []
    if "devices" in parts:
        update_instance(inst["id"], dict(devices=ip["meta"]["devices"]))
        inst = get_instance(inst["id"])
        done.append("devices")
    with using_instance(inst):
        here = load_params()
        if "params" in parts:
            active = ip["meta"]["backend"]
            for b, v in ip["params"].items():
                if b != active:
                    _reject_full_size_draft(v)
                    _write_env_file(_backend_file(b), _keep_local(dict(v, BACKEND=b), here))
            save_params(_keep_local(ip["params"][active], here), active)
            done.append("params")
        if "tiers" in parts:
            for t, tv in ip["tiers"].items():
                if tv is not None:
                    write_tier(t, _keep_local(tv, here))
            done.append("tiers")
    return dict(ok=True, applied=done, instance=inst["id"],
                note="nothing was restarted; restart the instance to run it")


# --------------------------------------------------------------------------
# usage statistics
# --------------------------------------------------------------------------
def parse_metrics():
    """Prometheus counters from llama-server --metrics."""
    raw = api_get("/metrics")
    if not isinstance(raw, str):
        return {}
    out = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0].split("{")[0]] = float(parts[-1])
            except ValueError:
                pass
    return out


TIMING_RE = re.compile(
    r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
EVAL_RE = re.compile(
    r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
ACCEPT_RE = re.compile(
    r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")
# Context depth of a finished request. It lands on the slot-release line, not on
# any print_timing line, which is why the per-request history had throughput but
# no depth to plot it against.
RELEASE_RE = re.compile(r"stop processing: n_tokens =\s*(\d+)")


def parse_timings(limit=400):
    """Per-request history scraped from the engine log."""
    if not ENGINE_LOG.exists():
        return []
    try:
        lines = ENGINE_LOG.read_text(errors="ignore").splitlines()
    except Exception:
        return []
    recs, cur = [], {}
    for ln in lines:
        m = TIMING_RE.search(ln)
        if m:
            if cur.get("decode_tps") is not None:
                recs.append(cur)
            cur = dict(prompt_ms=float(m.group(1)), prompt_tokens=int(m.group(2)),
                       prompt_tps=float(m.group(3)))
            continue
        m = EVAL_RE.search(ln)
        if m and cur:
            cur.update(eval_ms=float(m.group(1)), eval_tokens=int(m.group(2)),
                       decode_tps=float(m.group(3)))
            continue
        m = ACCEPT_RE.search(ln)
        if m and cur:
            cur.update(accept=float(m.group(1)), accepted=int(m.group(2)),
                       drafted=int(m.group(3)), mean_len=float(m.group(4)))
            continue
        m = RELEASE_RE.search(ln)
        if m and cur:
            cur.update(depth=int(m.group(1)))
    if cur.get("decode_tps") is not None:
        recs.append(cur)
    return recs[-limit:]



# ============================================================================
# OBSERVED SPEED CURVE
# ----------------------------------------------------------------------------
# The estimate's speed figures started life as constants typed in from reading
# logs by hand. This replaces them with the real thing: every completed request
# in the live log AND the archived logs is a (context depth, decode t/s) sample,
# so the projection is fitted to what this box has actually done and improves as
# it serves more traffic. Falls back to the hand-derived table only when there
# is not enough data to fit.
# ============================================================================
CURVE_BUCKETS = [0, 10000, 25000, 50000, 80000, 120000, 160000, 200000, 262144]
# Prefill rate depends on PROMPT SIZE: small warm prompts are overhead-bound,
# large cold ones reach the real streaming rate. Fit them separately.
PREFILL_BUCKETS = [128, 512, 2048, 8192, 32768, 131072, 524288]
_curve_cache = {}


BANNER_RE = re.compile(r"### inf01-engine BACKEND=(\w+)")


def log_backend(path):
    """Which backend produced this engine log: 'vulkan', 'rocm' or None.

    Order of evidence:
      1. the .meta sidecar written at startup
      2. the '### inf01-engine BACKEND=' banner inside the log
      3. the filename convention
    Archives named *_inf01_* predate the vulkan/rocm split and are all Vulkan -
    that is the entire recorded history of this box, so it is safe to claim.
    """
    meta = Path(str(path) + ".meta")
    for src in (meta, path):
        try:
            if not src.exists():
                continue
            head = src.read_text(errors="ignore")[:4000]
            tail = src.read_text(errors="ignore")[-4000:]
            m = BANNER_RE.search(head) or BANNER_RE.search(tail)
            if m:
                return m.group(1)
        except OSError:
            continue
    # A live log carries no banner until the next restart writes one; the
    # running process's binary path settles it definitively.
    try:
        if Path(path).resolve() == Path(str(_live_engine_log())).resolve():
            pid = server_pid()
            exe = os.path.realpath(f"/proc/{pid}/exe") if pid else ""
            if "rocm" in exe:
                return "rocm"
            if exe:
                return "vulkan"
    except (OSError, ValueError):
        pass
    n = Path(path).name
    if "_rocm_" in n:
        return "rocm"
    if "_vulkan_" in n or "_inf01_" in n:
        return "vulkan"
    # Live logs have no banner until the next restart writes one, and their
    # filename is just engine_debug.log. The rundirs are backend-specific by
    # design (ROCm script DELTA R3), so the path settles it.
    sp = str(path)
    if "/llama_qwen38_rocm/" in sp:
        return "rocm"
    if "/llama_qwen38/" in sp:
        return "vulkan"
    return None


def _log_files_for_curve(backend=None):
    """Logs for one backend. Live log first, then archives newest-first.

    Filtering by backend is not cosmetic: fitting one curve across mixed Vulkan
    and ROCm samples would silently average two different machines' worth of
    throughput into a number that describes neither.
    """
    cands = []
    if ENGINE_LOG.exists():
        cands.append(ENGINE_LOG)
    legacy = INST()["legacy"]
    if legacy:
        for rd in ("/dev/shm/llama_qwen38_rocm/telemetry/engine_debug.log",):
            if Path(rd).exists():
                cands.append(Path(rd))
    pat = "engine_debug_*.log" if legacy else f"engine_debug_inst-{INST()['id']}_*.log"
    try:
        cands.extend(sorted((f for f in LOGS.glob(pat)
                             if not (legacy and f.name.startswith("engine_debug_inst-"))),
                            key=lambda f: f.stat().st_mtime, reverse=True))
    except OSError:
        pass
    out, unknown = [], 0
    for f in cands:
        b = log_backend(f)
        if backend and b != backend:
            if b is None:
                unknown += 1
            continue
        out.append(f)
        if len(out) >= 13:
            break
    return out, unknown


def observed_curve(backend="vulkan", max_age_s=120):
    """Median decode t/s per depth bucket, plus prefill median, from real logs.

    Scoped to ONE backend - see _log_files_for_curve.
    """
    now = time.time()
    ck = _curve_cache.get(backend)
    if ck and now - ck["t"] < max_age_s:
        return ck["val"]

    files, unknown = _log_files_for_curve(backend)
    samples, prefill = [], []
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        cur = {}
        for ln in txt.splitlines():
            m = TIMING_RE.search(ln)
            if m:
                cur = {"prompt_tps": float(m.group(3)),
                       "prompt_tokens": int(m.group(2))}
                continue
            m = EVAL_RE.search(ln)
            if m and cur:
                cur["decode_tps"] = float(m.group(3))
                continue
            m = RELEASE_RE.search(ln)
            if m and cur.get("decode_tps") is not None:
                samples.append((int(m.group(1)), cur["decode_tps"]))
                # Keep the SIZE with the rate. A single median over all
                # prefills is dominated by hundreds of small warm prompts whose
                # fixed overhead makes them look slow, which understated deep
                # cold prefill by ~1.7x - the exact case worth planning around.
                pt = cur.get("prompt_tokens", 0)
                if pt >= 128:
                    prefill.append((pt, cur["prompt_tps"]))
                cur = {}

    buckets = []
    for lo, hi in zip(CURVE_BUCKETS, CURVE_BUCKETS[1:]):
        vals = sorted(v for d, v in samples if lo <= d < hi)
        if len(vals) >= 3:
            buckets.append(dict(depth=(lo + hi) // 2, tps=round(pct(vals, 50), 1),
                                n=len(vals), lo=lo, hi=hi))

    # Prefill fitted by PROMPT SIZE, same way decode is fitted by depth.
    pf_buckets = []
    for lo, hi in zip(PREFILL_BUCKETS, PREFILL_BUCKETS[1:]):
        vals = sorted(v for n, v in prefill if lo <= n < hi)
        if len(vals) >= 3:
            pf_buckets.append(dict(size=(lo + hi) // 2, tps=round(pct(vals, 50), 1),
                                   n=len(vals), lo=lo, hi=hi))

    out = dict(points=[(b["depth"], b["tps"]) for b in buckets],
               prefill_points=[(b["size"], b["tps"]) for b in pf_buckets],
               prefill_buckets=pf_buckets,
               buckets=buckets, n_samples=len(samples), backend=backend,
               n_logs=len(files), n_unknown_logs=unknown,
               logs=[Path(f).name for f in files],
               prefill_tps=round(pct(sorted(v for _, v in prefill), 50), 0)
                            if len(prefill) >= 3 else None,
               n_prefill=len(prefill))
    _curve_cache[backend] = dict(t=now, val=out)
    return out


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(len(s) * p / 100))], 2)


def headline(n=40):
    """Compact throughput summary + series for the panel's charts."""
    recs = parse_timings(limit=n)
    base = measured_baseline()
    drecs = [r for r in recs if r.get("decode_tps")]
    dec = [r["decode_tps"] for r in drecs]
    dep = [r.get("depth") for r in drecs]
    exp = [expected_at(base, d) for d in dep]
    pre = [r.get("prompt_tps") for r in recs
           if r.get("prompt_tps") and r.get("prompt_tokens", 0) > 200]
    acc = [r.get("accept") for r in recs if r.get("accept") is not None]
    return dict(
        decode_last=dec[-1] if dec else None,
        decode_median=pct(dec, 50),
        prefill_last=pre[-1] if pre else None,
        prefill_median=pct(pre, 50),
        accept_last=acc[-1] if acc else None,
        accept_median=pct(acc, 50),
        # separate series - decode (~45 t/s) and prefill (~650 t/s) must never
        # share an axis; they are plotted as two charts.
        decode_series=dec[-40:],
        prefill_series=pre[-40:],
        accept_series=acc[-40:],
        # each request judged at its own context depth against the measured curve
        decode_depth_series=dep[-40:],
        decode_expected_series=exp[-40:],
        decode_depth_last=dep[-1] if dep else None,
        decode_expected_last=exp[-1] if exp else None,
        decode_vs_expected=(round(dec[-1] / exp[-1], 3) if dec and exp and exp[-1] else None),
        baseline=base,
    )


def stats():
    recs = parse_timings()
    m = parse_metrics()
    dec = [r["decode_tps"] for r in recs if r.get("decode_tps")]
    pre = [r["prompt_tps"] for r in recs if r.get("prompt_tps") and r.get("prompt_tokens", 0) > 200]
    acc = [r["accept"] for r in recs if r.get("accept") is not None]
    mlen = [r["mean_len"] for r in recs if r.get("mean_len") is not None]
    # the context each request ran at - NOT prompt_tokens, which is only the part
    # of the prompt that was new (a 130k agent turn usually adds a few hundred)
    depth = [r.get("depth") if r.get("depth") is not None else r.get("prompt_tokens", 0) for r in recs]
    base = measured_baseline()
    return dict(
        requests=len(recs),
        tokens_in=sum(r.get("prompt_tokens", 0) for r in recs),
        tokens_out=sum(r.get("eval_tokens", 0) for r in recs),
        decode_tps=dict(last=round(dec[-1], 2) if dec else None,
                        median=pct(dec, 50), p10=pct(dec, 10), p90=pct(dec, 90)),
        prefill_tps=dict(last=round(pre[-1], 2) if pre else None,
                         median=pct(pre, 50), p10=pct(pre, 10), p90=pct(pre, 90),
                         note="only prompts >200 tokens; short ones are all overhead"),
        draft_acceptance=dict(last=round(acc[-1], 3) if acc else None,
                              median=pct(acc, 50), mean_len_median=pct(mlen, 50),
                              samples=len(acc),
                              note="workload dependent: higher on documents and code, lower on short chat"),
        context_depth=dict(max=max(depth) if depth else 0, median=pct(depth, 50),
                           deepest_tested=max(depth) if depth else 0),
        prometheus={k: v for k, v in m.items() if k.startswith("llamacpp:")},
        history=[dict(r, expected_tps=expected_at(base, r.get("depth"))) for r in recs[-60:]],
        baseline=base,
        # Why there is nothing to show, when there is nothing to show. Zero
        # requests is indistinguishable from a broken parser unless we say so:
        # llama-server TRUNCATES --log-file on open, so every restart wipes the
        # history, and a record only exists once a request COMPLETES. After a
        # restart, the first deep request prefills for minutes before it counts.
        empty_reason=(None if recs else _stats_empty_reason()),
    )


def _stats_empty_reason():
    pid = server_pid()
    if not pid:
        return "llama-server is not running, so there is nothing to measure."
    try:
        up = int(time.time() - Path(f"/proc/{pid}").stat().st_ctime)
    except OSError:
        up = None
    busy = False
    s = api_get("/slots")
    if isinstance(s, list) and s:
        busy = bool(s[0].get("is_processing"))
    msg = ("No request has COMPLETED since the server started"
           + (f" {up // 60}m {up % 60}s ago" if up is not None else "")
           + ". The engine log is truncated on every start, so restarting "
             "clears this history.")
    if busy:
        msg += (" One request is in flight right now - at this context depth a "
                "prefill can run for several minutes before it is counted.")
    return msg


# --------------------------------------------------------------------------
# diagnostics
#
# Each check is {id, status, evidence, meaning, fix}. Structured findings with
# the evidence attached are what make this useful to a model - a raw log dump
# forces it to re-derive what we already know. Every trap below cost us real
# time or a hard lock on this box.
# --------------------------------------------------------------------------
def tail(path, n=4000):
    try:
        return Path(path).read_text(errors="ignore")[-n:]
    except Exception:
        return ""


def newest(glob):
    fs = sorted(LOGS.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return fs[0] if fs else None


def checks():
    if INST().get("engine") in GEN_ENGINES:
        return _sd_checks()
    out = []
    pid = server_pid()
    cmdline, env = "", {}
    if pid:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            env = dict(l.split("=", 1) for l in
                       Path(f"/proc/{pid}/environ").read_bytes().decode(errors="ignore").split("\0")
                       if "=" in l)
        except Exception:
            pass

    launch_log = (str(newest("panel_launch.out") or newest("run_*.out") or "")
                  if INST()["legacy"] else (str(LAUNCH_OUT) if LAUNCH_OUT.exists() else ""))
    _exe = os.path.realpath(f"/proc/{pid}/exe") if pid else ""
    _be = next((b for b in ("rocm", "cuda", "cpu") if b in _exe), "vulkan") if _exe else \
        str(load_params().get("BACKEND"))
    ltxt = tail(launch_log, 60000) if launch_log else ""
    etxt = tail(ENGINE_LOG, 60000)

    def add(cid, ok, evidence, meaning, fix="", warn=False):
        out.append(dict(id=cid, status="ok" if ok else ("warn" if warn else "fail"),
                        evidence=evidence, meaning=meaning, fix=fix))

    # main's start guard: lockout, starts used in the window, fallback tier and
    # the last refusals - the things systemd enforces without telling anyone.
    g = main_guard(fresh=True)
    if g:
        mins = max(1, g["interval_s"] // 60)
        ev = [f"unit {g['unit']}: {g['active']}/{g['sub']}, result {g['result']}",
              f"starts in the last {mins} min: {g['starts_in_window']} of {g['burst']}",
              f"failure counter {g['fails']}, so the next start uses the '{g['next_tier']}' tier"]
        if g["locked"]:
            ev.append("LOCKED until about " + (time.strftime("%H:%M", time.localtime(g["unlocks_at"]))
                                               if g["unlocks_at"] else "the window passes"))
        ev += [f"{r['at']} {r['code']}: {r['msg'][:160]}" for r in g["recent"][-3:]]
        near = g["starts_in_window"] >= g["burst"] - 1 or g["fails"] > 0
        add("start_guard", not g["locked"] and not near, "; ".join(ev),
            f"systemd allows main {g['burst']} starts per {mins} min (StartLimitBurst, a "
            f"crash-loop guard in the unit file). When it trips, every Start and Save-and-restart "
            f"is refused until it clears. A start that dies before it has been up "
            f"{g['ok_after_s']} s also counts as a failed start and moves the next start to the "
            f"safe, then minimal, tier.",
            fix=(g["clear_cmd"] if g["locked"]
                 else "let a start finish loading before restarting again"),
            warn=not g["locked"])

    # 0. does it start at all? Every check below reads the logs of the LAST
    # run, so a server that cannot start any more used to show all green.
    if not pid:
        us = unit_state() or {}
        fails = fail_count()
        err = last_start_error()
        missing = [d for d in INST()["devices"]
                   if d != "cpu" and d not in _present_gpus()]
        bad = us.get("active") == "failed" or bool(missing) or (fails > 0 and err)
        add("start_failures", not bad,
            "; ".join(filter(None, [
                f"unit {us.get('unit')} is {us.get('active')}" if us else "no unit state",
                f"{fails} consecutive failed starts" if fails else None,
                f"devices not in this machine: {', '.join(missing)}" if missing else None,
                f"last refusal: {err}" if err else None])),
            "The server is not running and its last start attempts failed. The checks "
            "below describe the last run that did start, not the current configuration.",
            ("Fix the device list on the Status tab (Change devices)" if missing else
             "Read the launch log (Logs tab) for the refusal") +
            ", then reset the failed-start counter and start it again.")

    # 1. the double-load trap - this one hard-locked the box twice
    dbl = "loading draft model" in ltxt
    add("spec_double_load", not dbl,
        "found 'loading draft model'" if dbl else
        ("found 'creating MTP draft context'" if "creating MTP draft context" in ltxt
         else "neither marker present (spec off, or log rotated)"),
        "--spec-draft-model pointed at a full-size target loads a SECOND full copy "
        "of the model. Two 27B copies + two KV caches exceed 30GB host / 24.5GB VRAM.",
        "Leave SPEC_DRAFT_MODEL empty to use the embedded NextN head. Only a small "
        "sidecar (e.g. the 862MB FastMTP file) is a valid value.")

    # 2. VRAM vs GTT placement
    fb = env.get("GGML_VK_ALLOW_SYSMEM_FALLBACK")
    if _be == "cuda":
        _um = env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY")
        add("cuda_unified_memory", not _um,
            f"GGML_CUDA_ENABLE_UNIFIED_MEMORY={_um!r} in the live process env",
            "The CUDA counterpart of the Vulkan sysmem trap: with it set, weights that do "
            "not fit are served from host RAM over PCIe instead of failing to load.",
            "Leave it unset unless you have chosen that trade deliberately.", warn=True)
    elif _be == "vulkan" or not pid:
      add("vk_sysmem_fallback", fb == "0" or not pid,
        (f"GGML_VK_ALLOW_SYSMEM_FALLBACK={fb!r} in the live process env" if pid else
         "not running - checked against the live process env once it starts"),
        "ggml-vulkan defaults to silently serving the model from GTT (host RAM over "
        "PCIe) if it will not place in VRAM. Measured cost: 2.79 t/s prefill and "
        "5.86 t/s decode, with no error logged. ~20x slowdown, silent.",
        "Set GGML_VK_ALLOW_SYSMEM_FALLBACK=0. It also converts an over-budget config "
        "into a loud load-time failure instead of a silent crawl.")
    

    # 3. fit-params
    fit = "failed to fit params" in ltxt
    add("fit_params_abort", not fit,
        "common_fit_params aborted" if fit else "no fit-params abort",
        "The auto-fitter gave up because n_gpu_layers was pinned, so nothing caps "
        "an over-budget config. Not fatal alone, but it means no guard rails.",
        "Expected when NGL is set explicitly. Treat as a reminder to size memory by hand.",
        warn=True)

    # 4. GPU reset / device lost
    dl = re.search(r"device lost|ErrorDeviceLost|context is lost", etxt + ltxt, re.I)
    add("gpu_device_lost", not dl, dl.group(0) if dl else "none in current logs",
        "VK_ERROR_DEVICE_LOST is a kernel-level GPU hang. It is terminal: llama-server "
        "has no recovery path and every later request fails within milliseconds while "
        "the server still looks alive.",
        "Restart the server. Then check dmesg: a VM_L2/page-fault means VRAM OOM at "
        "prefill peak (lower ubatch or ctx); a 'ring timeout' means raise "
        "amdgpu.lockup_timeout instead.")

    # 5. template
    tp = "failed to parse" in etxt or "failed to parse" in ltxt
    add("chat_template", not tp, "parse error present" if tp else "no parse error",
        "minja is a C++ Jinja subset, so a template that parses under Python Jinja2 "
        "can still fail here.", "Check the template against minja's supported subset.")

    # 6. MTP actually engaged
    ignored = ltxt.count("nextn") and "ignoring" in ltxt
    add("mtp_engaged", not ignored,
        "nextn tensors reported 'unused ... ignoring'" if ignored else "NextN head in use",
        "The model's embedded MTP head is being discarded, so speculative decoding is "
        "off. Costs roughly 1.5-1.7x on decode.",
        "Set SPEC_TYPE=draft-mtp with SPEC_DRAFT_MODEL empty.", warn=True)

    # 6b. projector paired with the model it was built for
    _mmw = projector_mismatch()
    add("mmproj_pairing", not _mmw,
        _mmw or "projector and model report the same build",
        "Every model repo ships its own mmproj. A projector from a different "
        "finetune has the same type, projection dim and tensor count, so it loads "
        "with no error and the server looks healthy - image understanding is just "
        "wrong. Text is unaffected.",
        "Set MMPROJ to the projector that shipped with the model in MODEL.",
        warn=True)

    # 6c. template pin still matches the file it was taken from
    _tsrc = str(load_params().get("TEMPLATE_SRC", "") or "")
    _tpin = str(load_params().get("TEMPLATE_SHA256", "") or "")
    if _tsrc and os.path.exists(_tsrc):
        _tnow = _sha256_file(_tsrc)
        add("template_pin", _tnow == _tpin,
            f"{Path(_tsrc).name}: on disk {_tnow[:16]}, pinned {_tpin[:16] or '(none)'}",
            "The launch script copies the template and compares it against "
            "TEMPLATE_SHA256, aborting on a mismatch. A template edited after the "
            "last panel save will not start the server at all.",
            "Re-save parameters in the panel to re-pin the template to its "
            "current contents.")

    # 7. memory placement, from the rolling series rather than one sample
    s = list(_samples)[-60:]
    if s:
        peak = max(x["vram"] for x in s)
        gpeak = max(x["gtt"] for x in s)
        # Idle evicts everything to GTT, so a low peak only means something if we
        # have a decent window AND the server has actually served in it. Otherwise
        # this is unknown, not broken - reporting it as a failure is a false alarm.
        served = bool(parse_timings(limit=5))
        # THIS instance's process, not whichever llama-server fdinfo lists
        proc_vram = sum(x["vram_mib"] for x in _proc_gpu_mem(pid)) if pid else 0
        want = 15000 if INST()["legacy"] else int(_weights_mib() * 0.8)
        if proc_vram > want:
            add("vram_residency", True,
                f"llama-server holds {proc_vram} MiB resident VRAM (per-process, "
                f"DRM fdinfo); whole-GPU peak {peak} MiB / GTT {gpeak} MiB",
                "Per-process residency from fdinfo is the reliable signal - the "
                "whole-GPU counter swings with idle eviction.",
                "")
        elif len(s) < 12 or not served:
            add("vram_residency", True,
                f"peak VRAM {peak} MiB / GTT {gpeak} MiB over {len(s)} samples "
                f"- INCONCLUSIVE (need >=12 samples and traffic; idle evicts to GTT)",
                "Cannot judge placement from an idle window.",
                "Send a request, then re-check.", warn=True)
        else:
            add("vram_residency", peak > want,
                f"peak VRAM over last {len(s)} samples = {peak} MiB (GTT peak {gpeak} MiB)",
                "A single VRAM read is unreliable: amdgpu evicts to GTT when idle and "
                "pages back on the next request. Judge by the peak across a window, or t/s.",
                "Low peak while actively serving means placement is broken - check "
                "GGML_VK_ALLOW_SYSMEM_FALLBACK=0.")

    # 8. host memory headroom
    avail = s[-1]["mem_avail"] if s else 0
    add("host_ram_headroom", avail > 3000, f"MemAvailable = {avail} MB",
        "A host OOM on this 30GB box does not reliably kill the offender before the "
        "machine locks up. It has been power-cycled twice from this.",
        "The launch script has a watchdog that kills llama-server below 2048MB.",
        warn=True)
    return out, cmdline, env, launch_log


def _sd_checks():
    out = []
    pid = server_pid()

    def add(cid, ok, evidence, meaning, fix="", warn=False):
        out.append(dict(id=cid, status="ok" if ok else ("warn" if warn else "fail"),
                        evidence=evidence, meaning=meaning, fix=fix))
    if not pid:
        us = unit_state() or {}
        fails, err = fail_count(), last_start_error()
        add("start_failures", not (us.get("active") == "failed" or (fails and err)),
            "; ".join(filter(None, [f"unit {us.get('unit')} is {us.get('active')}" if us else None,
                                    f"{fails} consecutive failed starts" if fails else None,
                                    f"last refusal: {err}" if err else None])) or "not running",
            "Whether the last start attempts failed.", "Read the launch log (Logs tab).")
    plan = launch_plan()
    add("launch_plan", not plan["errors"],
        "; ".join(plan["errors"]) or "files present, build active, devices present",
        "What a start would be refused for.", "Fix the parameters or apply a preset.")
    add("plan_warnings", not plan["warnings"], "; ".join(plan["warnings"]) or "none",
        "Things that may make a start fail or run slowly.", "", warn=True)
    env = {}
    if pid:
        try:
            env = dict(l.split("=", 1) for l in Path(f"/proc/{pid}/environ").read_bytes()
                       .decode(errors="ignore").split("\0") if "=" in l)
        except OSError:
            pass
        add("vk_sysmem_fallback", env.get("GGML_VK_ALLOW_SYSMEM_FALLBACK") == "0" or
            load_params().get("BACKEND") != "vulkan",
            f"GGML_VK_ALLOW_SYSMEM_FALLBACK={env.get('GGML_VK_ALLOW_SYSMEM_FALLBACK')}",
            "Without it a Vulkan allocation that does not fit silently lands in host RAM.")
    cmd = ""
    if pid:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            pass
    return out, cmd, env, str(LAUNCH_OUT) if LAUNCH_OUT.exists() else ""


# --------------------------------------------------------------------------
# measured baselines
#
# "Expected" decode speed comes from the newest depth curve measured for the
# SAME model file (any instance - main and main-test can front the same server),
# interpolated at each request's own context depth. Decode at 136k is not
# comparable to decode at 8k, and one fixed number (the old hard-coded
# 2026-09-02 baseline: 44.92 t/s, a different model and build) made healthy
# deep requests look like regressions. A curve measured with another KV type,
# build or speculative setup still counts, but is reported as approximate,
# with the differences named, so nobody reads it as exact.
# Prefill and draft acceptance get no baseline: they depend on prompt size and
# content far more than on the card.
# --------------------------------------------------------------------------
_base_cache = {}


def _build_of(binary):
    m = re.search(r"/(b\d+)[-/]", str(binary or ""))
    return m.group(1) if m else (Path(str(binary)).parent.name if binary else None)


def _server_identity(inst):
    with using_instance(inst):
        pid = server_pid()
        p = load_params()
    ident = dict(model=Path(str(p.get("MODEL") or "")).name or None, kv=p.get("KV_TYPE"),
                 spec=p.get("SPEC_TYPE") or None, build=None, running=bool(pid))
    if pid:
        try:
            argv = hostos.proc_argv(pid)
            m = _argv_get(argv, ("-m", "--model"))
            ident.update(model=Path(m).name if m else ident["model"],
                         kv=_argv_get(argv, ("--cache-type-k", "-ctk")) or "f16",
                         spec=_argv_get(argv, ("--spec-type",)) or None,
                         build=_build_of(argv[0]))
        except OSError:
            pass
    return ident


def measured_baseline(inst=None):
    """The depth curve to judge this instance's decode speed against, or a note saying none exists."""
    inst = inst or INST()
    c = _base_cache.get(inst["id"])
    if c and time.time() - c[0] < 30:
        return c[1]
    me = _server_identity(inst)
    best = None
    for iid in instance_ids():
        for cv in depthcurve.history(iid, limit=20):
            if cv.get("state") != "done" or not cv.get("summary") or Path(str(cv.get("model") or "")).name != me["model"]:
                continue
            same = dict(kv=cv.get("kv") == me["kv"], build=_build_of(cv.get("binary")) == me["build"],
                        spec=(cv.get("spec") or None) == me["spec"])
            key = (sum(same.values()), cv.get("finished") or "")
            if best is None or key > best[0]:
                best = (key, cv, same)
    if not best:
        out = dict(source=None, model=me["model"],
                   note="No depth curve has been measured for this model yet. Run one on the Status "
                        "tab (Decode vs context depth) to get an expected-speed line.")
    else:
        _, cv, same = best
        differs = []
        if not same["kv"]:
            differs.append(f"KV cache {cv.get('kv')} (running {me['kv']})")
        if not same["build"]:
            differs.append(f"build {_build_of(cv.get('binary'))} (running {me['build']})")
        if not same["spec"]:
            differs.append(f"speculative {cv.get('spec') or 'off'} (running {me['spec'] or 'off'})")
        out = dict(source="depth-curve", curve=cv.get("id"), instance=cv.get("instance"),
                   measured=cv.get("finished"), model=me["model"], exact=not differs, differs=differs,
                   points=[dict(depth=q["depth"], decode_tps=q["decode_tps"], decode_min=q.get("decode_min"),
                                decode_max=q.get("decode_max")) for q in cv["summary"]],
                   note=("Measured with the same model, KV cache, build and speculative setup."
                         if not differs else "Approximate - measured with " + ", ".join(differs)
                         + ". Re-run the depth curve for an exact baseline."))
    _base_cache[inst["id"]] = (time.time(), out)
    return out


def expected_at(base, depth):
    """Expected decode t/s at a context depth: linear between measured depths, flat beyond them."""
    pts = sorted((q["depth"], q["decode_tps"]) for q in (base or {}).get("points") or [])
    if not pts or depth is None:
        return None
    if depth <= pts[0][0]:
        return pts[0][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if depth <= d1:
            return round(v0 + (v1 - v0) * (depth - d0) / (d1 - d0), 1)
    return pts[-1][1]


def _running_build():
    """Build string for the llama-server that is ACTUALLY running.

    Hardcoded to b10766 until 2026-09-16, so after the b10985 upgrade this field
    reported a build that had not run for days - and it is the field a reader is
    most likely to trust without checking. Ground truth, in order: the running
    server's own /props, then the binary the live process was exec'd from, then
    whatever builds.env currently marks active.
    """
    props = api_get("/props")
    if isinstance(props, dict) and props.get("build_info"):
        return str(props["build_info"])
    pid = server_pid()
    if pid:
        try:
            exe = os.path.realpath(f"/proc/{pid}/exe")
            d = os.path.dirname(exe)
            return sh(f"LD_LIBRARY_PATH={d} {exe} --version 2>&1 | head -1")
        except OSError:
            pass
    bd = active_build(active_backend())
    if bd:
        return sh(f"LD_LIBRARY_PATH={bd} {bd}/llama-server --version 2>&1 | head -1")
    return "<unknown: nothing running and no active build in builds.env>"


_CLEAN_END = ("Journal stopped", "systemd-shutdown", "Reached target reboot.target",
              "Reached target poweroff.target", "Reached target halt.target")


def _read_or_why(path, n):
    try:
        return Path(path).read_text(errors="ignore")[-n:]
    except PermissionError:
        return "(root-only: sudo cat " + str(path) + ")"
    except OSError as e:
        return f"(unreadable: {e})"


def crash_logs(boots=8):
    """How recent boots ended, plus whatever a crash left behind.

    A silent hard lock leaves no kernel message, so the useful signal is the
    SHAPE of the journal: a boot that ends without systemd-shutdown/'Journal
    stopped' ended by freeze or power loss. For those, the last lines before
    the gap are included. kdump dumps (/var/crash) and pstore records are
    root-owned; the panel runs as smbadmin (group adm), so it lists them and
    includes contents only where it can read them.
    """
    out = dict(note="ended='abrupt' means no clean shutdown was logged: freeze, "
                    "panic without kdump, reset button or power loss. Timestamps UTC.")
    try:
        lst = json.loads(sh("journalctl --list-boots -o json --no-pager", timeout=20))
    except ValueError:
        lst = []
    recs = []
    for b in lst[-boots:]:
        bid, cur = b["boot_id"], b["index"] == 0
        fmt = lambda us: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(us / 1e6))
        last = sh(f"journalctl -b {bid} -n 12 --no-pager -o short-iso", timeout=20)
        ended = "running" if cur else ("clean" if any(m in last for m in _CLEAN_END)
                                       else "abrupt")
        r = dict(boot_id=bid, start=fmt(b["first_entry"]), end=fmt(b["last_entry"]),
                 minutes=round((b["last_entry"] - b["first_entry"]) / 6e7, 1),
                 ended=ended,
                 kernel_warnings=sh(f"journalctl -b {bid} -k -p warning --no-pager "
                                    f"-o short-iso | tail -40", timeout=20))
        if ended == "abrupt":
            r["last_lines_before_gap"] = sh(f"journalctl -b {bid} -n 60 --no-pager "
                                            f"-o short-iso", timeout=20)
        recs.append(r)
    out["boots"] = recs
    kd = []
    for d in sorted(Path("/var/crash").glob("2*"), reverse=True)[:5]:
        e = dict(dir=str(d), files=sorted(f.name for f in d.iterdir()) if d.is_dir() else [])
        for f in sorted(d.glob("dmesg.*")):
            e["dmesg_tail"] = _read_or_why(f, 12000)
        kd.append(e)
    out["kdump"] = dict(
        state=sh("kdump-config show 2>/dev/null | grep -i 'current state'")
              .split(":", 1)[-1].strip(),
        crash_kernel_loaded=read_int("/sys/kernel/kexec_crash_loaded"),
        softlockup_panic=read_int("/proc/sys/kernel/softlockup_panic"),
        hardlockup_panic=read_int("/proc/sys/kernel/hardlockup_panic"),
        nmi_watchdog=read_int("/proc/sys/kernel/nmi_watchdog"),
        dumps=kd or "none")
    ps = []
    for d in sorted(Path("/var/lib/systemd/pstore").glob("*"),
                    key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
        ps.append(dict(dir=str(d), saved=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime(d.stat().st_mtime)),
                       dmesg_tail=_read_or_why(d / "dmesg.txt", 6000)))
    out["pstore"] = ps or "none"
    return out


def debug_bundle():
    """Everything a model needs to debug this box, in one structured object."""
    ck, cmdline, env, launch_log = checks()
    s = list(_samples)
    return dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        purpose="Diagnostic bundle for inf01 llama.cpp Vulkan inference. "
                "Findings are pre-analysed: read 'checks' first.",
        host=dict(
            hostname=sh("hostname"), kernel=sh("uname -r"),
            os=sh(". /etc/os-release && echo $PRETTY_NAME"),
            cpu=sh("lscpu | awk -F: '/Model name/{print $2}' | head -1").strip(),
            cores=sh("nproc"),
            ram_total_mb=int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) // 1024,
            gpu=sh("lspci -nn | grep -i 'navi\\|vga' | head -2"),
            mesa=sh("apt list --installed mesa-vulkan-drivers 2>/dev/null | tail -1"),
            vram_total_mib=read_int(f"{CARD}/mem_info_vram_total") // 1048576,
            vram_visible_mib=read_int(f"{CARD}/mem_info_vis_vram_total") // 1048576,
        ),
        server=dict(
            pid=server_pid(), running=server_pid() is not None,
            build=_running_build(),
            cmdline=cmdline,
            relevant_env={k: v for k, v in env.items()
                          if k.startswith(("GGML_", "VK_", "LD_LIBRARY"))},
            props=api_get("/props"),
            slots=api_get("/slots"),
        ),
        params=load_params(),
        gpu=gpu_info(),
        checks=ck,
        stats=stats(),
        memory_series=dict(
            note="VRAM/GTT in MiB, 5s interval. Single reads are misleading: amdgpu "
                 "evicts to GTT when idle and pages back on the next request.",
            samples=s[-180:],
        ),
        measured_baseline=measured_baseline(),
        integrity=dict(
            model_sha_state="verified against repo SHA256SUMS at download",
            template_sha=sh(f"sha256sum {MODELS}/qwen3.8-safe-v2.jinja | cut -d' ' -f1"),
            template_expected="4ed3960ba9caa33352f417bc6ac2f6e8358c76b4cbdbced9c59e9e16909f794b",
        ),
        logs=dict(
            launch_log_path=launch_log,
            engine_log_tail=tail(ENGINE_LOG, 8000),
            launch_log_tail=tail(launch_log, 8000) if launch_log else "",
            dmesg_amdgpu=sh("dmesg 2>/dev/null | grep -iE 'amdgpu|ring|reset|VM_L2' | tail -30")
                         or "(dmesg restricted to root)",
        ),
        crashes=crash_logs(),
        known_traps=[
            "--spec-draft-model must never point at a full-size target: it loads a "
            "second full copy and OOM-locks the host. Embedded MTP takes NO draft model.",
            "GGML_VK_ALLOW_SYSMEM_FALLBACK must be 0 or the model is silently served "
            "from GTT at ~1/20th speed with no error.",
            "A single mem_info_vram_used read is unreliable; use the series or t/s.",
            "Draft acceptance is workload dependent: ~0.82 on documents, ~0.48 on chat.",
        ],
    )


# --------------------------------------------------------------------------
# server control + downloads
# --------------------------------------------------------------------------
UNIT = "inf01-llama"          # main's system unit; sudoers names it exactly so


def unit_installed(inst=None):
    inst = inst or INST()
    if inst["scope"] == "system":
        return os.path.exists(f"/etc/systemd/system/{UNIT}.service")
    return (USER_UNIT_DIR / USER_UNIT).exists() and user_manager()["bus"]


def _systemctl(verb, inst=None):
    inst = inst or INST()
    if inst["scope"] == "system":
        r = subprocess.run(["sudo", "-n", "/usr/bin/systemctl", verb, UNIT],
                           capture_output=True, text=True, timeout=60)
    else:
        r = subprocess.run(["systemctl", "--user", verb, inst["unit"]], env=_user_env(),
                           capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def unit_state(inst=None):
    inst = inst or INST()
    if not unit_installed(inst):
        return None
    if inst["scope"] == "system":
        cmd, env = ["systemctl"], None
    else:
        cmd, env = ["systemctl", "--user"], _user_env()
    name = UNIT if inst["scope"] == "system" else inst["unit"]
    r = subprocess.run(cmd + ["is-active", name], capture_output=True, text=True, env=env)
    e = subprocess.run(cmd + ["is-enabled", name], capture_output=True, text=True, env=env)
    return dict(active=r.stdout.strip(), enabled=e.stdout.strip(), unit=name,
                scope=inst["scope"],
                linger=None if inst["scope"] == "system" else user_manager()["linger"])


# --------------------------------------------------------------------------
# main's start guard (added 2026-09-24).
# main's system unit caps starts at StartLimitBurst per StartLimitIntervalSec -
# a crash-loop guard, because restart storms have hard-locked this box. systemd
# enforces it silently ("Start request repeated too quickly") and the panel used
# to report only "systemctl failed", so a lockout looked like a broken Save.
# This reads the guard's state so every start/restart can explain a refusal,
# and so a restart while main is still loading (which kills it mid-load and
# counts as a failed start) is refused unless forced.
# --------------------------------------------------------------------------
_guard_cache = [0.0, None]
_guard_log = deque(maxlen=50)          # refusals and lockout clears, newest last
_reset_ok_cache = [0.0, None]


def _unit_props(names):
    r = subprocess.run(["systemctl", "show", UNIT, "-p", ",".join(names)],
                       capture_output=True, text=True, timeout=10)
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        out[k] = v
    return out


def _usec_to_s(v, default):
    """systemd prints StartLimitIntervalUSec as '10min', '1min 30s' or plain usec."""
    v = (v or "").strip()
    if v.isdigit():
        return int(v) // 1000000
    total, units = 0, dict(h=3600, min=60, s=1, ms=0.001, us=0.000001)
    for num, unit in re.findall(r"([\d.]+)\s*(h|min|ms|us|s)", v):
        total += float(num) * units[unit]
    return int(total) or default


def reset_failed_allowed():
    """Whether a PASSWORDLESS sudo rule lets the panel run reset-failed on main
    (cached 60 s). `sudo -l <cmd>` is no use here: an (ALL : ALL) ALL entry that
    needs a password makes it answer yes for every command, and the panel runs
    sudo -n. So read the NOPASSWD entries and look for the exact command."""
    if time.time() - _reset_ok_cache[0] < 60:
        return _reset_ok_cache[1]
    r = subprocess.run(["sudo", "-n", "-l"], capture_output=True, text=True, timeout=10)
    text = " ".join(r.stdout.split())
    ok = False
    for part in text.split("NOPASSWD:")[1:]:
        entry = re.split(r"\(\w+(?: : \w+)?\)", part)[0]      # up to the next runas block
        if re.search(rf"/usr/bin/systemctl reset-failed {re.escape(UNIT)}(\.service)?(,|$|\s)", entry):
            ok = True
    _reset_ok_cache[:] = [time.time(), ok]
    return ok


def main_guard(fresh=False):
    """State of main's start guard, or None when the current instance is not main."""
    inst = INST()
    if not inst["legacy"] or not unit_installed(inst):
        return None
    if not fresh and _guard_cache[1] is not None and time.time() - _guard_cache[0] < 4:
        return _guard_cache[1]
    pr = _unit_props(["ActiveState", "SubState", "Result", "StartLimitBurst",
                      "StartLimitIntervalUSec", "ActiveEnterTimestamp"])
    burst = int(pr.get("StartLimitBurst") or 5)
    interval = _usec_to_s(pr.get("StartLimitIntervalUSec"), 10)
    now = time.time()
    txt = sh(f"journalctl -u {UNIT} --since @{int(now - interval)} -o short-unix --no-pager "
             f"2>/dev/null | grep ' systemd\\[1\\]: Started '", timeout=10) or ""
    starts = sorted(float(l.split()[0]) for l in txt.splitlines() if l[:1].isdigit())
    active = pr.get("ActiveState")
    # systemd refuses a start once `burst` starts fall inside the window. Its
    # Result does not always say so (seen here: it stayed 'exit-code' after
    # "Start request repeated too quickly"), so count the starts instead.
    unlocks_at = starts[-burst] + interval if len(starts) >= burst else None
    locked = active not in ("active", "activating") and (
        (unlocks_at is not None and unlocks_at > now) or pr.get("Result") == "start-limit-hit")
    if not locked:
        unlocks_at = None
    pid = server_pid()
    up_s = None
    if active in ("active", "activating"):
        ts = sh(f"date -d {shlex.quote(pr.get('ActiveEnterTimestamp') or '')} +%s 2>/dev/null").strip()
        up_s = int(now - int(ts)) if ts.isdigit() else None
    health = api_get("/health", timeout=2) if pid else None
    ready = isinstance(health, dict) and health.get("status") == "ok"
    loading = active in ("active", "activating") and not ready
    fails = fail_count()
    g = dict(unit=UNIT, active=active, sub=pr.get("SubState"), result=pr.get("Result"),
             burst=burst, interval_s=interval, starts_in_window=len(starts),
             locked=locked, unlocks_at=unlocks_at, loading=loading, ready=ready, up_s=up_s,
             fails=fails, next_tier="normal" if fails == 0 else ("safe" if fails == 1 else "minimal"),
             ok_after_s=120, can_reset=reset_failed_allowed(),
             clear_cmd=f"sudo systemctl reset-failed {UNIT} && sudo systemctl start {UNIT}",
             recent=list(_guard_log)[-5:])
    _guard_cache[:] = [now, g]
    return g


def _guard_note(code, text):
    _guard_log.append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code=code,
                           msg=text))
    print(f"[start-guard] {code}: {text}", flush=True)


def guard_refusal(code, g, systemd_msg=None):
    """The API answer for a refused start/restart of main: a plain reason, the
    guard state for the popup, and systemd's own words when it spoke."""
    mins = max(1, g["interval_s"] // 60)
    if code == "start_limit":
        when = (time.strftime("%H:%M", time.localtime(g["unlocks_at"])) if g["unlocks_at"]
                else f"{mins} min after the last start")
        msg = (f"main was NOT started: systemd's crash-loop guard for {UNIT} is active "
               f"({g['burst']} starts within {mins} min; StartLimitBurst in the unit file). "
               f"Your settings are saved. The guard clears on its own at about {when}, "
               f"or now with: {g['clear_cmd']}")
    else:
        up = f" (up {g['up_s']} s)" if g.get("up_s") is not None else ""
        tier = "safe" if g["fails"] == 0 else "minimal"
        msg = (f"main was NOT restarted: it is still loading{up}. Restarting now kills it "
               f"mid-load, counts as a failed start (the next start would use the '{tier}' "
               f"tier) and uses one of {g['burst']} starts allowed per {mins} min. Wait until "
               f"it reports ready, or choose Restart anyway.")
    _guard_note(code, msg + (f" | systemd: {systemd_msg}" if systemd_msg else ""))
    return dict(ok=False, code=code, msg=msg, systemd=systemd_msg, guard=g)


def reset_main_lockout():
    if not INST()["legacy"]:
        raise ValueError("only main has a system-unit start guard")
    r = subprocess.run(["sudo", "-n", "/usr/bin/systemctl", "reset-failed", UNIT],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        g = main_guard(fresh=True)
        msg = ("the sudo rule does not allow reset-failed yet; run it once by hand: "
               + (g["clear_cmd"] if g else f"sudo systemctl reset-failed {UNIT}"))
        _guard_note("reset_refused", msg + f" | sudo: {(r.stderr or r.stdout).strip()[:200]}")
        return dict(ok=False, msg=msg, guard=g)
    _guard_note("reset", f"lockout on {UNIT} cleared from the panel")
    return dict(ok=True, msg="lockout cleared", guard=main_guard(fresh=True))


def start_server():
    """Prefer systemd when the unit is installed.

    If systemd owns the process, starting it any other way means two owners:
    the unit's Restart policy would fight the panel's stop, and the process
    would come back seconds after being stopped.
    """
    if server_pid():
        return False, "already running"
    inst = INST()
    # Host RAM is shared by every instance. Refuse a start whose worst case,
    # added to what is already running, cannot fit - that combination is the
    # hard lock this box has already taken three times.
    if inst.get("engine") not in GEN_ENGINES:
        rb = ram_budget(load_params(), include_others=True)
        if rb["verdict"] == "impossible":
            return False, "refused - " + rb["detail"]
    if not inst["legacy"]:
        plan = launch_plan()
        if plan["errors"]:
            return False, "refused - " + " | ".join(plan["errors"])
        ensure_user_unit()
        if unit_installed(inst):
            _systemctl("reset-failed", inst)
            ok, msg = _systemctl("start", inst)
            return ok, ("starting via systemd --user" if ok else f"systemctl --user failed: {msg}")
        LOGS.mkdir(exist_ok=True)
        subprocess.Popen(["/usr/bin/python3", str(PANEL / "instance_launch.py"), inst["id"]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         cwd=str(LLAMA), start_new_session=True)
        return True, ("starting by direct launch - no systemd --user bus, so it dies with "
                      "the panel. Fix: " + (user_manager()["fix"] or "log in once"))
    if unit_installed():
        g = main_guard(fresh=True)
        if g and g["locked"]:
            return False, guard_refusal("start_limit", g)["msg"]
        ok, msg = _systemctl("start")
        if not ok:
            g = main_guard(fresh=True)
            if g and (g["locked"] or "too quickly" in msg or "start-limit" in msg):
                return False, guard_refusal("start_limit", g, msg)["msg"]
        return ok, ("starting via systemd" if ok else f"systemctl failed: {msg}")
    LOGS.mkdir(exist_ok=True)
    with open(LAUNCH_OUT, "wb") as f:
        subprocess.Popen(["/bin/bash", str(SCRIPT)], stdout=f, stderr=subprocess.STDOUT,
                         cwd=str(LLAMA), start_new_session=True)
    return True, "starting (no systemd unit; direct launch)"


def stop_server():
    if unit_installed() and (INST()["legacy"] or _systemctl("is-active")[0]):
        ok, msg = _systemctl("stop")
        return ok, ("stopped via systemd" if ok else f"systemctl failed: {msg}")
    pid = server_pid()
    if not pid:
        return False, "not running"
    # Signal THIS instance's process only. The old 'pkill -x llama-server'
    # would have taken down every instance on the box.
    if INST()["legacy"]:
        subprocess.run(["pkill", "-f", "run_qwen38_vulkan_inf01"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", f"instance_launch.py {INST()['id']}$"], capture_output=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    for _ in range(30):
        _pid_cache.pop(INST()["id"], None)
        if not os.path.exists(f"/proc/{pid}"):
            return True, "stopped"
        time.sleep(1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True, "force-killed"


def gguf_header(path):
    """architecture / tensor count / name from a GGUF header, without loading it.

    Classifying by filename alone is what made `usable_as_draft` size-only: a
    0.93 GB projector passed the size test and was reported as a valid draft,
    which it is not. The header settles what a file actually is.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            ver = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def rd_str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            SZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            kv = {}
            for _ in range(min(n_kv, 512)):
                k = rd_str()
                t = struct.unpack("<I", f.read(4))[0]
                if t == 8:
                    kv[k] = rd_str()
                elif t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    ln = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:
                        for _i in range(ln):
                            rd_str()
                    else:
                        f.read(SZ.get(et, 4) * ln)
                    kv[k] = f"<array x{ln}>"
                else:
                    raw = f.read(SZ.get(t, 4))
                    kv[k] = int.from_bytes(raw, "little") if len(raw) <= 8 else None
            return dict(version=ver, n_tensors=n_tensors,
                        arch=kv.get("general.architecture"),
                        name=kv.get("general.name"),
                        finetune=kv.get("general.finetune"),
                        ctx_train=kv.get(
                            f"{kv.get('general.architecture')}.context_length"),
                        blocks=kv.get("general.architecture", "") and
                               kv.get(f"{kv.get('general.architecture')}.block_count"))
    except (OSError, struct.error, UnicodeDecodeError):
        return {}


def gguf_kv_per_token_mib(path):
    """KV cost per token in MiB, normalised to q5_1 (6 bpw) like the calibration.

    Read from the header: block_count x head_count_kv x (key_length +
    value_length). Hybrid / linear-attention models only keep KV on their full
    attention layers, so full_attention_interval divides it when present.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)
            f.read(8)
            n_kv = struct.unpack("<Q", f.read(8))[0]
            SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            kv = {}
            def rd_str():
                return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
            for _ in range(min(n_kv, 1024)):
                k = rd_str()
                t = struct.unpack("<I", f.read(4))[0]
                if t == 8:
                    kv[k] = rd_str()
                elif t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    ln = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:
                        for _i in range(ln):
                            rd_str()
                        kv[k] = None
                    elif et in (4, 5) and ln <= 4096:
                        kv[k] = list(struct.unpack(f"<{ln}{'I' if et == 4 else 'i'}",
                                                   f.read(4 * ln)))
                    else:
                        f.read(SZ.get(et, 4) * ln)
                        kv[k] = None
                elif t == 6:
                    kv[k] = struct.unpack("<f", f.read(4))[0]
                else:
                    kv[k] = int.from_bytes(f.read(SZ.get(t, 4)), "little")
    except (OSError, struct.error, TypeError):
        return None
    a = kv.get("general.architecture")
    g = lambda s: kv.get(f"{a}.{s}")
    n_layer, n_embd, n_head = g("block_count"), g("embedding_length"), g("attention.head_count")
    if not isinstance(n_layer, int) or not n_layer:
        return None
    hkv = g("attention.head_count_kv")
    if isinstance(hkv, list):
        heads = sum(hkv)
    else:
        if isinstance(n_head, list):
            n_head = max(n_head) or None
        heads = (hkv or n_head or 0) * n_layer
    hd = (n_embd // n_head) if isinstance(n_embd, int) and isinstance(n_head, int) and n_head else 0
    k_len, v_len = g("attention.key_length") or hd, g("attention.value_length") or hd
    if not heads or not (k_len and v_len):
        return None
    fai = g("full_attention_interval")
    if isinstance(fai, int) and fai > 1:
        heads = heads / fai
    bytes_f16 = heads * (k_len + v_len) * 2
    return bytes_f16 / 1048576 * 6.0 / 16.0


# A draft's own VRAM cost is roughly its weights plus a small KV cache of its
# own. The sidecar KV is tiny next to the target's, so weights dominate.
def draft_vram_mib(size_bytes):
    return int(size_bytes / 1048576 * 1.06) + 64


# ============================================================================
# DISK  (added 2026-09-16)
# ----------------------------------------------------------------------------
# Everything on this box shares ONE filesystem: / on /dev/sda2. Models, builds,
# backups and the OS all draw from the same pool, so "space left" is a single
# number and any one of them can starve the others.
#
# /dev/shm is deliberately reported separately and labelled: it is a tmpfs
# carved out of the 31.4 GB of host RAM, NOT disk. The engine log and slot
# saves live there, so filling it spends the same RAM the prompt cache and the
# RAM_FLOOR_MB watchdog are already fighting over. Reporting it next to the SSD
# without that label is how someone ends up pointing --log-prompts-dir at it.
# ============================================================================
def _dir_bytes(p):
    """Bytes used by a directory tree. du, not a Python walk: du counts blocks
    actually allocated, which is what the filesystem will hand back."""
    out = sh(f"du -sb {shlex.quote(str(p))} 2>/dev/null", timeout=60)
    try:
        return int(out.split()[0])
    except (ValueError, IndexError):
        return 0


def disk_info():
    try:
        st = os.statvfs(str(MODELS))
    except OSError as e:
        return dict(error=str(e))
    total = st.f_blocks * st.f_frsize
    avail = st.f_bavail * st.f_frsize
    used = total - (st.f_bfree * st.f_frsize)
    shm = {}
    try:
        s2 = os.statvfs("/dev/shm")
        shm = dict(total_gb=round(s2.f_blocks * s2.f_frsize / 1e9, 1),
                   avail_gb=round(s2.f_bavail * s2.f_frsize / 1e9, 1),
                   note="tmpfs - this is HOST RAM, not disk. Filling it competes "
                        "with the prompt cache and the RAM_FLOOR_MB watchdog.")
    except OSError:
        pass
    breakdown = {}
    for label, path in (("models", MODELS), ("llama_builds", LLAMA),
                        ("backups", HOME / "backups"), ("llama_logs", HOME / "llama_logs")):
        if Path(path).exists():
            breakdown[label] = round(_dir_bytes(path) / 1e9, 2)
    return dict(
        mount=sh("df --output=source / | tail -1").strip() or "/",
        total_gb=round(total / 1e9, 1),
        used_gb=round(used / 1e9, 1),
        avail_gb=round(avail / 1e9, 1),
        pct_used=int(round(100 * used / total)) if total else 0,
        breakdown_gb=breakdown,
        shm=shm,
    )


def model_references():
    """Every model/template path any config could load -> why it is held.

    Resolves each backend against DEFAULTS rather than reading its env file
    raw. That distinction is load-bearing: params-rocm.env carries
    USE_MMPROJ="1" but no MMPROJ key at all (it predates the per-backend
    split), so the projector ROCm would actually load comes from DEFAULTS and
    appears nowhere in the file. Reading files alone reported that projector as
    unreferenced, i.e. safe to delete, which would have broken the ROCm backend
    the next time anyone switched to it.

    The three fallback tiers get the same treatment, because a file held only
    by the 'minimal' tier is a booby trap that fires days later during a
    fallback, when nobody is watching.

    Deliberately does NOT call load_params(): that seeds missing backend files
    as a side effect, and a reporting function must not write config.
    """
    refs = {}
    KEYS = ("MODEL", "MMPROJ", "SPEC_DRAFT_MODEL", "TEMPLATE_SRC")

    def add(v, why):
        v = str(v or "").strip()
        if v:
            try:
                refs.setdefault(os.path.realpath(v), []).append(why)
            except OSError:
                pass

    for iid in instance_ids():
        idir = get_instance(iid)["dir"]
        tag = "" if iid == "main" else f"instance {iid} "
        for b in BACKENDS:
            bf = idir / f"params-{b}.env"
            if iid != "main" and not bf.exists():
                continue
            if iid == "main" and b == "cuda" and not bf.exists():
                continue                       # main never had a cuda file
            vals = dict(DEFAULTS)
            vals.update(_read_env_file(bf))
            for k in KEYS:
                add(vals.get(k), f"{tag}backend {b}:{k}")
        for env in sorted(idir.glob("params-tier-*.env")):
            vals = dict(DEFAULTS)
            vals.update(_read_env_file(env))
            for k in KEYS:
                add(vals.get(k), f"{tag}{env.name}:{k}")
        if (idir / "params.env").exists():
            vals = dict(DEFAULTS)
            vals.update(_read_env_file(idir / "params.env"))
            for k in KEYS:
                add(vals.get(k), f"{tag}params.env:{k}")
    for pid in server_pids():
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue
        for a in argv:
            if a.endswith((".gguf", ".jinja")):
                add(a, f"running server pid {pid}")
    return refs


def delete_model(path):
    """Delete a model file. Refuses anything referenced by any config or in use.

    Deliberately stricter than delete_build(): these files are 15 GB and take
    hours to re-download over this link, and the failure mode of getting it
    wrong is a server that will not start. Every guard below refuses loudly
    rather than deleting something that MIGHT be needed.
    """
    p = Path(path).resolve()
    if p.parent != MODELS.resolve():
        raise ValueError(f"refusing to delete outside {MODELS}")
    if p.suffix != ".gguf":
        raise ValueError("refusing to delete anything that is not a .gguf")
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    refs = model_references().get(str(p))
    if refs:
        raise ValueError(f"{p.name} is still referenced by: {', '.join(sorted(set(refs)))}. "
                         "Point those at another file first.")
    size = p.stat().st_size
    p.unlink()
    return dict(deleted=str(p), freed_gb=round(size / 1e9, 2), disk=disk_info())


def list_models():
    out = []
    # Build identity of the model currently selected, so the UI can mark each
    # projector paired/not in one call instead of re-deriving it per row.
    try:
        _want = _norm_build(gguf_header(load_params().get("MODEL", "")).get("name"))
    except Exception:
        _want = ""
    _refs = model_references()
    for f in sorted(MODELS.glob("*.gguf")):
        _sm = _SHARD_RE.match(f.name)
        if _sm and _sm.group(2) != "00001":
            continue                     # listed as part of shard 1's row
        sz = model_bytes(f) or f.stat().st_size
        h = gguf_header(f)
        n_t = h.get("n_tensors") or 0
        is_proj = f.name.startswith("mmproj") or (h.get("arch") or "") == "clip"
        # A real draft sidecar is a SMALL file with few tensors - an MTP head,
        # not a whole model. The target has hundreds of tensors; this has ~19.
        is_sidecar = (not is_proj) and sz <= DRAFT_MAX_BYTES and 0 < n_t <= 128
        kind = "projector" if is_proj else ("draft-sidecar" if is_sidecar else "target")

        note, ok = "", is_sidecar
        if is_proj:
            note = "vision projector - NOT a draft model"
        elif not is_sidecar and sz > DRAFT_MAX_BYTES:
            note = ("full-size target: as a draft this loads a SECOND complete copy "
                    "and has hard-locked this host twice")
        elif is_sidecar and "FastMTP" in f.name:
            note = ("needs a PATCHED llama.cpp build. It carries a trimmed 32k draft "
                    "vocab plus a d2t remap tensor, and stock builds hard-assert the "
                    "full 248320 vocab: 'expected 5120, 248320, got 5120, 32768'. "
                    "MERGING IT INTO THE TARGET GGUF DOES NOT HELP - the blocker is "
                    "the loader, not the file layout, so one file or two is rejected "
                    "identically. Needs a source build at commit 4df29be4 with "
                    "HauhauCS-FastMTP-llama.cpp.patch (one file, src/models/qwen35.cpp).")
            ok = False
        _shards = model_shards(f)
        out.append(dict(
            name=f.name, path=str(f), size_gb=round(sz / 1e9, 2),
            shards=len(_shards) if _sm else None,
            shards_missing=[x.name for x in _shards if not x.exists()] if _sm else [],
            kind=kind, arch=h.get("arch"), n_tensors=n_t,
            vram_mib=draft_vram_mib(sz) if kind != "target" else None,
            usable_as_draft=ok, note=note,
            build=h.get("name"),
            # None for anything that is not a projector - only a projector can
            # be silently paired with the wrong model.
            matches_model=((bool(_want) and _norm_build(h.get("name")) == _want)
                           if is_proj else None),
            # Populated from EVERY params*.env plus the live cmdline, so a file
            # held only by a fallback tier still reports as undeletable.
            referenced_by=sorted(set(_refs.get(str(f.resolve()), []))),
        ))
    return out


def draft_options():
    """Values for the SPEC_DRAFT_MODEL dropdown, with their memory cost."""
    opts = [dict(value="", label="(embedded MTP head - recommended)", vram_mib=370,
                 note="Uses the NextN head already inside the target GGUF. "
                      "Measured 0.82 acceptance, mean length 2.64, for +370 MiB.",
                 ok=True)]
    for m in list_models():
        if m["kind"] == "target":
            continue
        opts.append(dict(value=m["path"], label=f"{m['name']}  ({m['size_gb']} GB)",
                         vram_mib=m["vram_mib"], note=m["note"],
                         ok=m["usable_as_draft"]))
    return opts


def _norm_build(v):
    """Build identity reduced to comparable form: 'Qwen3.8-27B-X' == 'Qwen3.8 27B X'."""
    return re.sub(r"[^a-z0-9]", "", (v or "").lower())


def projector_mismatch(params=None):
    """'' if MODEL and MMPROJ are the same build, else why they are not.

    Added 2026-09-15, when a second model repo landed. Nothing errors on a
    mismatch: these Qwen3.8 projectors share a shape (qwen3vl_merger, 5120
    projection dim, 334 tensors, 27 vision blocks), so a projector from another
    finetune loads cleanly and only the image answers are wrong. Silent wrong
    output is the failure mode this box has been bitten by before, so it gets a
    check rather than a comment.
    """
    p = params or load_params()
    if str(p.get("USE_MMPROJ", 1)) not in ("1", "true", "True"):
        return ""
    mp, pp = str(p.get("MODEL", "") or ""), str(p.get("MMPROJ", "") or "")
    if not (mp and pp and os.path.exists(mp) and os.path.exists(pp)):
        return ""
    mn = gguf_header(mp).get("name")
    pn = gguf_header(pp).get("name")
    if not (mn and pn) or _norm_build(mn) == _norm_build(pn):
        return ""
    return (f"model reports build '{mn}' but projector "
            f"{Path(pp).name} reports '{pn}'")


def projector_options():
    """Values for the MMPROJ dropdown, flagged against the model in MODEL."""
    want = _norm_build(gguf_header(load_params().get("MODEL", "")).get("name"))
    opts = []
    for m in list_models():
        if m["kind"] != "projector":
            continue
        nm = gguf_header(m["path"]).get("name") or "unknown build"
        match = bool(want) and _norm_build(nm) == want
        opts.append(dict(
            value=m["path"],
            label=f"{m['name']}  ({m['size_gb']} GB)" + ("" if match else "  [other build]"),
            vram_mib=m["vram_mib"],
            note=(f"built for '{nm}'" if match else
                  f"built for '{nm}', which is NOT the model in MODEL. It will load "
                  "without an error and return wrong image results."),
            ok=match))
    return opts


# ============================================================================
# CHAT TEMPLATES  (added 2026-09-15)
#
# The launch script SHA-pins the template and ABORTS on a mismatch. That is
# exactly why a template could not be swapped before: any file other than
# qwen3.8-safe-v2.jinja fails the pin and the launch dies at line 405.
#
# Rather than weaken that check, the panel now owns BOTH keys. The script
# assigns TEMPLATE_SRC/TEMPLATE_SHA256 at lines 146-148 and sources params.env
# at 214, so the panel's pin replaces the baked-in one. The integrity check
# still does its job - it catches the file changing between save and launch -
# it just no longer hardcodes a single template. No launch-script edit needed.
#
# Templates live in ~/models beside the GGUFs on purpose: downloads already
# land there, and make-backup.sh copies every non-.gguf file out of that
# directory, so a pasted template is captured by the existing backup.
# ============================================================================
TEMPLATE_MAX_BYTES = 1 << 20


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _template_features(text):
    """What a template appears to support, for the picker. Advisory only."""
    return [label for label, needle in
            (("reasoning_effort", "reasoning_effort"), ("xhigh", "xhigh"),
             ("tools", "tools"), ("vision", "image"),
             ("thinking", "enable_thinking"))
            if needle in text]


def template_lint(text):
    """(ok, detail, features) for a candidate chat template.

    Jinja2 is a SUPERSET of minja, the C++ subset llama.cpp actually uses, so a
    clean parse here is necessary but not sufficient - the authoritative result
    is the 'failed to parse' line in the server log, which the chat_template
    check in checks() already reads. Catching gross errors before a restart
    still beats discovering them in a failed launch.
    """
    if not text.strip():
        return False, "template is empty", []
    if len(text.encode("utf-8", "replace")) > TEMPLATE_MAX_BYTES:
        return False, f"over {TEMPLATE_MAX_BYTES // 1024} KiB", []
    if "{%" not in text and "{{" not in text:
        return False, "no Jinja tags - this does not look like a chat template", []
    try:
        import jinja2
    except ImportError:
        return True, "saved WITHOUT a parse check (jinja2 not installed)", _template_features(text)
    try:
        jinja2.Environment().parse(text)
    except Exception as e:
        return False, f"Jinja parse error: {e}", []
    return True, "parses under Jinja2 (minja is a subset - confirm on load)", _template_features(text)


def list_templates():
    """Every *.jinja in ~/models, with the pin the launch script will check."""
    p = load_params()
    cur, pin = str(p.get("TEMPLATE_SRC", "") or ""), str(p.get("TEMPLATE_SHA256", "") or "")
    out = []
    for f in sorted(MODELS.glob("*.jinja")):
        try:
            text = f.read_text(errors="replace")
        except OSError as e:
            out.append(dict(name=f.name, path=str(f), error=str(e), parses=False))
            continue
        ok, detail, feats = template_lint(text)
        sha = _sha256_file(f)
        in_use = str(f) == cur
        out.append(dict(
            name=f.name, path=str(f), bytes=f.stat().st_size, sha256=sha,
            parses=ok, detail=detail, features=feats, in_use=in_use,
            # A selected template edited after the save would abort the launch
            # on the pin. Surface it here rather than at 3am in a launch log.
            pin_ok=(sha == pin) if in_use else None,
            preview=text[:600]))
    return out


def gguf_chat_template(path):
    """The chat template embedded in a GGUF, or '' if it carries none.

    Both models on this box ship one (~8.9 KB), so 'use the template this model
    was built with' does not have to mean hunting for it on Hugging Face.
    Structure mirrors gguf_header(); only the value we want is kept.
    """
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            f.read(4)                                   # version
            f.read(8)                                   # tensor count
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def rd_str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            for _ in range(n_kv):
                k = rd_str()
                t = struct.unpack("<I", f.read(4))[0]
                if t == 8:
                    v = rd_str()
                    if k == "tokenizer.chat_template":
                        return v
                elif t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    ln = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:
                        for _i in range(ln):
                            rd_str()
                    else:
                        f.read(SZ.get(et, 4) * ln)
                else:
                    f.read(SZ.get(t, 4))
    except (OSError, struct.error):
        return ""
    return ""


def extract_template(model_path, name=None):
    """Lift a GGUF's own template into ~/models so it can be selected."""
    mp = str(model_path or load_params().get("MODEL", ""))
    if not mp or not os.path.exists(mp):
        raise ValueError(f"model not found: {mp}")
    text = gguf_chat_template(mp)
    if not text:
        raise ValueError(f"{Path(mp).name} carries no embedded chat template")
    if not name:
        name = re.sub(r"[^A-Za-z0-9._-]", "-", Path(mp).stem)[:60] + "-embedded.jinja"
    return save_template(name, text)


def save_template(name, content):
    """Write a pasted template into ~/models. Returns its row from list_templates()."""
    name = (name or "").strip()
    if name.endswith(".j2"):
        name = name[:-3] + ".jinja"
    if not name.endswith(".jinja"):
        name += ".jinja"
    # This directory is read by the launch script, so no traversal and no
    # surprises: a strict allowlist, not a blocklist.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.jinja", name):
        raise ValueError("bad name - use letters, digits, dot, dash, underscore")
    ok, detail, _ = template_lint(content)
    if not ok:
        raise ValueError(detail)
    (MODELS / name).write_text(content)
    for t in list_templates():
        if t["name"] == name:
            return t
    raise ValueError("template written but not found on rescan")


def _curl_tail(path, n=400):
    """Last n chars of curl's stderr log, for the failure message."""
    try:
        return path.read_text(errors="ignore")[-n:].strip()
    except OSError:
        return ""


def _verify_download(dest, rec):
    """Did we get the file, or a web page wearing its name?

    `curl -f` only catches HTTP error status. A Hugging Face *repo* URL returns
    200 with HTML, so on 2026-09-15 856 KB of markup landed in models/ named
    after the repo and was reported done, 100%. Content decides, not status.

    Returns (ok, why, resumable). `resumable` separates "this is the right file,
    just unfinished" - where the bytes on disk are what `curl -C -` needs - from
    "this is the wrong kind of file entirely", which the caller deletes.
    """
    try:
        size = dest.stat().st_size
        with open(dest, "rb") as f:
            head = f.read(512)
    except OSError as e:
        return False, str(e), True

    # Anchored at the start only. A substring test would misfire on a real GGUF
    # that happens to carry markup in an early metadata string.
    if head.lstrip()[:5].lower() in (b"<!doc", b"<html"):
        return False, ("server returned an HTML page, not a file. For Hugging "
                       "Face use the /resolve/main/<file> URL, not the repo "
                       "page."), False
    if dest.name.lower().endswith(".gguf") and size >= 4 and head[:4] != b"GGUF":
        return False, "missing GGUF magic - this is not a GGUF file.", False
    if rec["total"] and size != rec["total"]:
        return False, (f"incomplete: got {size} bytes, expected {rec['total']}. "
                       "Re-running the download resumes from here."), True
    return True, "", True


HF_TOKEN_FILE = HOME / ".cache/huggingface/token"


def _hf_auth_header_file(url):
    """For huggingface.co URLs with a token saved, a 0600 header file for curl.

    Passed as -H @file so the token never appears in /proc/<pid>/cmdline, and
    kept outside panel/ so make-backup.sh does not sweep it into a tarball.
    curl drops a custom Authorization header when a redirect leaves the host,
    so the signed CDN URL HuggingFace redirects to never sees it.
    """
    if not re.match(r"^https://huggingface\.co/", url or ""):
        return None
    try:
        tok = HF_TOKEN_FILE.read_text().strip()
    except OSError:
        return None
    if not tok:
        return None
    hf = HOME / ".cache/huggingface" / f".curl-auth-{os.getpid()}-{threading.get_ident()}"
    fd = os.open(hf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"Authorization: Bearer {tok}\n")
    return hf


def hf_token_status():
    try:
        return dict(present=bool(HF_TOKEN_FILE.read_text().strip()), path=str(HF_TOKEN_FILE))
    except OSError:
        return dict(present=False, path=str(HF_TOKEN_FILE))


def save_hf_token(token):
    token = str(token or "").strip()
    if token and not re.match(r"^hf_[A-Za-z0-9]{20,}$", token):
        raise ValueError("that does not look like a HuggingFace token (hf_...)")
    HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not token:
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
        return hf_token_status()
    fd = os.open(HF_TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return hf_token_status()


def start_download(url, name=None, subdir=None):
    name = name or url.split("/")[-1].split("?")[0]
    if "/" in name or name.startswith("."):
        raise ValueError("bad filename")
    if subdir not in (None, "sd"):
        raise ValueError("bad subdir")
    did = str(int(time.time() * 1000))
    dest = (MODELS / subdir / name) if subdir else MODELS / name
    rec = dict(id=did, url=url, name=name, dest=str(dest), status="running",
               pct=0.0, downloaded=0, total=0, started=time.time(), error=None)
    with _lock:
        _downloads[did] = rec

    def run():
        # curl's stderr goes to a FILE, never a pipe. With stderr=PIPE and a
        # loop that only polls, curl blocks in write() as soon as the 64 KiB
        # pipe buffer fills with progress output and then never exits - so
        # poll() never returns either and both sides wait forever. That
        # deadlocked a 14.3 GiB fetch at 13% on 2026-09-15. --no-progress-meter
        # removes the trigger; the log file removes the whole failure mode.
        log = dest.parent / f".{name}.curl.log"
        auth = _hf_auth_header_file(url)
        try:
            with open(log, "wb") as errf:
                p = subprocess.Popen(
                    ["curl", "-fL", "-C", "-", "--no-progress-meter"]
                    + (["-H", f"@{auth}"] if auth else []) + [
                     # A stalled socket used to hang here indefinitely: curl has
                     # no stall timeout unless you ask for one.
                     "--speed-limit", "102400", "--speed-time", "120",
                     # Plain --retry covers curl's transient set, which
                     # includes the timeout (28) that --speed-time raises, so a
                     # stall recovers. --retry-all-errors is deliberately NOT
                     # used: it would retry a 404 ten times before giving up.
                     "--retry", "10", "--retry-delay", "5",
                     "-o", str(dest), url],
                    stderr=errf, stdout=subprocess.DEVNULL)
                if auth:
                    # curl has read the header file by the time it is transferring
                    time.sleep(3)
                    try:
                        auth.unlink()
                    except OSError:
                        pass
                while p.poll() is None:
                    time.sleep(2)
                    try:
                        cur = dest.stat().st_size
                        rec["downloaded"] = cur
                        if rec["total"]:
                            rec["pct"] = round(cur * 100 / rec["total"], 1)
                    except FileNotFoundError:
                        pass
            if p.returncode != 0:
                rec["status"] = "failed"
                rec["error"] = _curl_tail(log) or f"curl exit {p.returncode}"
                return
            ok, why, resumable = _verify_download(dest, rec)
            rec["status"] = "done" if ok else "failed"
            rec["error"] = None if ok else why
            if not ok and not resumable:
                # Wrong kind of file. Leaving it in models/ is what let 856 KB
                # of HTML sit there looking like a model; a partial download is
                # kept instead, because -C - resumes from exactly those bytes.
                try:
                    dest.unlink()
                    rec["error"] += " Discarded."
                except OSError:
                    pass
        except Exception as e:
            rec["status"], rec["error"] = "failed", str(e)
        finally:
            for _f in (log, auth):
                try:
                    if _f:
                        _f.unlink()
                except OSError:
                    pass

    try:
        req = urllib.request.Request(url, method="HEAD")
        try:
            _tok = HF_TOKEN_FILE.read_text().strip() if url.startswith("https://huggingface.co/") else ""
        except OSError:
            _tok = ""
        if _tok:
            req.add_header("Authorization", f"Bearer {_tok}")
        with urllib.request.urlopen(req, timeout=10) as r:
            rec["total"] = int(r.headers.get("Content-Length", 0))
    except Exception:
        pass
    threading.Thread(target=run, daemon=True).start()
    return rec


# ============================================================================
# BUILD REGISTRY  (added 2026-09-09)
#
# Until now LIB_DIR was hardcoded in each launch script, so "update llama.cpp"
# meant hand-editing a 34 KB shell script and remembering to do it again for
# the other backend. Builds now live side by side under ~/llama and the panel
# records which one each backend uses in builds.env, which the scripts source.
# Nothing is ever overwritten in place: switching build is a dropdown, and
# rolling back is the same dropdown.
# ============================================================================
LLAMA_REPO = "ggml-org/llama.cpp"
BUILDS_ENV = PANEL / "builds.env"

# Upstream release assets that this box can actually run. Matched by name
# against the GitHub release; verified against the b10883 asset list on
# 2026-09-09. Everything else upstream ships (win-*, macos-*, arm64, s390x,
# sycl, openvino, cuda) is either the wrong OS or the wrong silicon.
ASSET_RE = {
    "vulkan": re.compile(r"^llama-b(\d+)-bin-ubuntu-vulkan-x64\.tar\.gz$"),
    "rocm":   re.compile(r"^llama-b(\d+)-bin-ubuntu-rocm-[\d.]+-x64\.tar\.gz$"),
    "cpu":    re.compile(r"^llama-b(\d+)-bin-ubuntu-x64\.tar\.gz$"),
    # Upstream started shipping Ubuntu CUDA builds (seen b11010). 12.8 is
    # preferred over 13.x: it is the conservative choice for a Turing card
    # (RTX 2060, compute 7.5). The runtime libraries come in a separate
    # cudart-*.tar.gz that is unpacked into the same directory.
    "cuda":   re.compile(r"^llama-b(\d+)-bin-ubuntu-cuda-12\.[\d.]+-x64\.tar\.gz$"),
}

# A build is classified by the ggml backend library it ships, NOT by its
# directory name. b10766/ predates the -<flavour> convention and would
# otherwise be unclassifiable; a name-based guess would also silently
# mislabel a directory someone renamed.
FLAVOUR_LIB = (("libggml-vulkan.so", "vulkan"), ("libggml-hip.so", "rocm"),
               ("libggml-cuda.so", "cuda"))


def _bindir(root):
    """Directory holding llama-server inside an unpacked build, or None.

    The tarballs unpack to a single llama-b<NNNNN>/ top level, but accept a
    flat layout too so a hand-unpacked build still registers.
    """
    root = Path(root)
    if (root / "llama-server").is_file():
        return root
    try:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and (sub / "llama-server").is_file():
                return sub
    except OSError:
        pass
    return None


def _flavour_of(bindir):
    names = {p.name for p in bindir.iterdir()} if bindir else set()
    for lib, flav in FLAVOUR_LIB:
        if lib in names:
            return flav
    return "cpu" if names else None


def _dir_bytes(p):
    total = 0
    for dirpath, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def scan_builds():
    """Every unpacked llama.cpp build under ~/llama, newest build first."""
    out = []
    active = _read_env_file(BUILDS_ENV) if BUILDS_ENV.exists() else {}
    for d in sorted(LLAMA.glob("b*")):
        if not d.is_dir():
            continue
        bd = _bindir(d)
        if not bd:
            continue
        m = re.search(r"b(\d+)", bd.name) or re.search(r"b(\d+)", d.name)
        flav = _flavour_of(bd)
        rec = dict(dir=str(d), bindir=str(bd), name=d.name,
                   build=int(m.group(1)) if m else 0,
                   flavour=flav, size_mb=round(_dir_bytes(d) / 1048576, 1),
                   active=False)
        rec["active"] = (active.get(f"LIB_DIR_{(flav or '').upper()}") == str(bd))
        out.append(rec)
    out.sort(key=lambda r: (-r["build"], r["flavour"] or ""))
    return out


def active_build(backend):
    """bindir this backend will launch from, or None to let the script decide."""
    return (_read_env_file(BUILDS_ENV) or {}).get(f"LIB_DIR_{backend.upper()}") or None


def set_active_build(backend, bindir):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    bd = Path(bindir)
    if not (bd / "llama-server").is_file():
        raise ValueError(f"{bindir} has no llama-server")
    flav = _flavour_of(bd)
    if flav != backend:
        raise ValueError(
            f"that build is a {flav} build; it cannot serve BACKEND={backend}. "
            f"Pick a build whose flavour matches, or change the backend.")
    cur = _read_env_file(BUILDS_ENV) if BUILDS_ENV.exists() else {}
    cur[f"LIB_DIR_{backend.upper()}"] = str(bd)
    PANEL.mkdir(exist_ok=True)
    BUILDS_ENV.write_text(
        "# GENERATED by the admin panel - do not hand-edit.\n"
        "# Per-backend llama.cpp build selection, sourced by run_qwen38_*_inf01.sh.\n"
        "# Each value is the directory containing llama-server.\n\n"
        + "".join(f"{k}={_env_quote(v)}\n" for k, v in sorted(cur.items()) if k.startswith("LIB_DIR_")))
    return cur


def delete_build(bindir):
    """Remove an unpacked build. Refuses to delete one that is in use."""
    bd = Path(bindir).resolve()
    if LLAMA not in bd.parents:
        raise ValueError("build is not under ~/llama")
    for b in BACKENDS:
        if active_build(b) == str(bd):
            raise ValueError(f"that build is the active {b} build; switch {b} to "
                             "another build first")
    if str(bd) in " ".join(live_cmdline_args()):
        raise ValueError("that build is what the running server was launched from")
    root = bd if bd.parent == LLAMA else bd.parent
    if root.parent != LLAMA:
        raise ValueError("refusing to delete outside ~/llama")
    subprocess.run(["rm", "-rf", str(root)], check=True)
    return dict(deleted=str(root))


def github_releases(limit=12):
    """Recent llama.cpp releases and which of our three flavours each carries.

    Unauthenticated GitHub API: 60 requests/hour per IP. The panel caches for
    10 minutes so an open browser tab cannot burn the quota.
    """
    now = time.time()
    with _lock:
        c = _rel_cache.get("data")
        if c and now - _rel_cache.get("at", 0) < 600:
            return c
    url = f"https://api.github.com/repos/{LLAMA_REPO}/releases?per_page={int(limit)}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "inf01-panel"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = json.load(r)
    out = []
    for rel in raw:
        flavours = {}
        for a in rel.get("assets", []):
            for flav, rx in ASSET_RE.items():
                if rx.match(a["name"]):
                    flavours[flav] = dict(name=a["name"],
                                          url=a["browser_download_url"],
                                          size_mb=round(a["size"] / 1048576, 1))
        if "cuda" in flavours:
            ver = re.search(r"cuda-([\d.]+)-x64", flavours["cuda"]["name"]).group(1)
            rt = next((a for a in rel.get("assets", [])
                       if a["name"] == f"cudart-llama-{rel.get('tag_name')}-bin-ubuntu-cuda-{ver}-x64.tar.gz"
                       or re.match(rf"^cudart-llama-.*ubuntu-cuda-{re.escape(ver)}-x64\.tar\.gz$", a["name"])),
                      None)
            if rt:
                flavours["cuda"]["runtime_url"] = rt["browser_download_url"]
                flavours["cuda"]["size_mb"] = round(flavours["cuda"]["size_mb"] + rt["size"] / 1048576, 1)
            else:
                flavours.pop("cuda")
        if flavours:
            m = re.match(r"b(\d+)", rel.get("tag_name", ""))
            out.append(dict(tag=rel.get("tag_name"),
                            build=int(m.group(1)) if m else 0,
                            published=rel.get("published_at"),
                            flavours=flavours))
    with _lock:
        _rel_cache.update(data=out, at=now)
    return out


def start_build_install(url, flavour, build, runtime_url=None):
    """Download a release tarball and unpack it to ~/llama/b<build>-<flavour>/.

    Unpacks to a staging directory first: a half-extracted build left behind by
    a dropped connection would otherwise register in scan_builds() and be
    selectable. Only a staging tree that yields a working llama-server is
    promoted.
    """
    if flavour not in ASSET_RE:
        raise ValueError(f"unknown flavour {flavour!r}")
    if not url.startswith("https://"):
        raise ValueError("url must be https")
    for _u in filter(None, (url, runtime_url)):
        if not re.match(r"^https://(github\.com|objects\.githubusercontent\.com)/", _u):
            raise ValueError("url must be a github.com release asset")
    if flavour == "cuda" and not runtime_url:
        raise ValueError("a CUDA build needs its cudart runtime tarball as well")
    dest_dir = LLAMA / f"b{int(build)}-{flavour}"
    if dest_dir.exists():
        raise ValueError(f"{dest_dir.name} already exists - delete it first")

    did = "build-" + str(int(time.time() * 1000))
    tarball = LLAMA / f".dl-{did}.tar.gz"
    stage = LLAMA / f".stage-{did}"
    rec = dict(id=did, url=url, name=f"b{build}-{flavour}", dest=str(dest_dir),
               kind="build", flavour=flavour, build=int(build), status="running",
               pct=0.0, downloaded=0, total=0, started=time.time(), error=None)
    with _lock:
        _downloads[did] = rec

    def run():
        try:
            p = subprocess.Popen(["curl", "-fL", "--retry", "3", "-o", str(tarball), url],
                                 stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
            while p.poll() is None:
                time.sleep(2)
                try:
                    cur = tarball.stat().st_size
                    rec["downloaded"] = cur
                    if rec["total"]:
                        rec["pct"] = round(cur * 100 / rec["total"], 1)
                except FileNotFoundError:
                    pass
            if p.returncode != 0:
                raise RuntimeError((p.stderr.read().decode(errors="ignore")[-400:]
                                    if p.stderr else f"curl exit {p.returncode}"))
            rec.update(status="unpacking", pct=100.0)
            stage.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "xzf", str(tarball), "-C", str(stage)],
                           check=True, capture_output=True)
            bd = _bindir(stage)
            if not bd:
                raise RuntimeError("archive contains no llama-server")
            if runtime_url:
                rec.update(status="fetching CUDA runtime")
                rt = LLAMA / f".dl-{did}-cudart.tar.gz"
                try:
                    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(rt), runtime_url],
                                   check=True, capture_output=True, timeout=3600)
                    rs = stage / ".cudart"
                    rs.mkdir()
                    subprocess.run(["tar", "xzf", str(rt), "-C", str(rs)], check=True,
                                   capture_output=True)
                    for so in rs.rglob("*.so*"):
                        os.replace(so, bd / so.name)
                    subprocess.run(["rm", "-rf", str(rs)], check=False)
                finally:
                    try:
                        rt.unlink()
                    except OSError:
                        pass
            os.chmod(bd / "llama-server", 0o755)
            got = _flavour_of(bd)
            if got != flavour:
                raise RuntimeError(f"archive is a {got} build, expected {flavour}")
            stage.rename(dest_dir)
            rec.update(status="done", bindir=str(_bindir(dest_dir)))
        except Exception as e:
            rec.update(status="failed", error=str(e))
            subprocess.run(["rm", "-rf", str(stage)], check=False)
        finally:
            try:
                tarball.unlink()
            except OSError:
                pass

    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "inf01-panel"})
        with urllib.request.urlopen(req, timeout=10) as r:
            rec["total"] = int(r.headers.get("Content-Length", 0))
    except Exception:
        pass
    threading.Thread(target=run, daemon=True).start()
    return rec


# ============================================================================
# HOST-RAM BUDGET  (added 2026-09-09)
#
# Written after reconstructing the Sep 4-8 failures: ten watchdog kills, two
# kernel OOMs and three host hard locks, all from one cause the settings UI
# could not see. When VRAM runs near full, amdgpu evicts model weights into
# GTT - host RAM pinned by the driver, invisible to ps RSS and to the cgroup.
# GTT ceiling + prompt cache + process overhead has to fit in physical RAM,
# and at CACHE_RAM=16384 it provably did not:
#
#     15.70 GB GTT ceiling + 16.00 GB cache = 31.70 GB  >  30.67 GB total
#
# The panel let that be saved because nothing here modelled host RAM at all;
# every existing estimate was about VRAM. This function is the missing half.
# ============================================================================
def _meminfo_mb(key):
    return hostos.meminfo_mb(key)            # /proc/meminfo on Linux, vm_stat/sysctl on macOS


def _gtt_ceiling_mb():
    """How much host RAM the amdgpu driver may pin as GTT."""
    try:
        card = Path(CARD)
        v = (card / "mem_info_gtt_used")
        if v.exists():
            return int(v.read_text().strip()) // 1048576
    except (OSError, ValueError):
        pass
    return 0


# Resident set of everything that is not llama-server and not reclaimable:
# panel, caddy, both ttyd units, whatever is running inside them. Measured at
# ~1.2 GB with a Claude Code session open in /terminal/; 1536 is that rounded
# up so the check does not pass by a hair on an idle box and then fail in use.
OTHER_SERVICES_MB = 1536
LLAMA_BASE_MB = 2048          # llama-server's own working set, model excluded


def _ram_demand(params, card_dir):
    """(own MB, gtt key, gtt MB) for one instance's params on one device."""
    def _int(k, d=0):
        try:
            return int(str(params.get(k, d)).strip() or d)
        except ValueError:
            return d
    total = _meminfo_mb("MemTotal:")
    cache = _int("CACHE_RAM", 0)
    cache_eff = max(total - LLAMA_BASE_MB - OTHER_SERVICES_MB, 0) if cache < 0 else cache
    mmproj = 0 if _int("MMPROJ_OFFLOAD", 0) else (896 if _int("USE_MMPROJ", 1) else 0)
    on_gpu = str(params.get("BACKEND", "vulkan")) != "cpu" and _int("NGL", 99) > 0
    gtts = {}
    if on_gpu:
        # A driver can only evict what this instance allocated on the card, so
        # each card's exposure is min(driver ceiling, instance footprint there).
        # Charging nouveau the whole 6.4 GB 2060 for ~2.4 GB of blk.64 + projector
        # turned main's budget IMPOSSIBLE on 2026-09-17. No readable placement
        # (bad header, unknown device) keeps the full ceiling: fail closed.
        foot = {}
        try:
            pl = estimate_placement({**load_params(), **params})
            if pl and pl["devices"]:
                foot = {d["pci"]: d["total_mib"]
                        for d in _device_footprints(params, pl) if d.get("pci")}
        except Exception:
            foot = {}
        for pci in (INST().get("devices") or [os.path.basename(str(card_dir))]):
            cd = f"/sys/bus/pci/devices/{pci}"
            f = Path(cd) / "mem_info_gtt_used"
            if f.exists():
                gtts[cd] = read_int(f) // 1048576
            elif os.path.realpath(Path(cd) / "driver").endswith("/nouveau"):
                # nouveau/TTM can evict to system RAM too; bound it by the card size
                gtts[cd] = (_device_record(pci) or {}).get("vram_total_mib") or 0
            if cd in gtts and pci in foot:
                gtts[cd] = min(gtts[cd], foot[pci])
    return LLAMA_BASE_MB + cache_eff + mmproj, gtts, sum(gtts.values())


def others_ram(exclude=None):
    """Worst-case host RAM of every OTHER running instance. A GTT ceiling is a
    per-card limit, so two instances on one card count it once."""
    me = exclude or INST()["id"]
    own, gtts, names = 0, {}, []
    for iid in instance_ids():
        if iid == me:
            continue
        try:
            inst = get_instance(iid)
        except ValueError:
            continue
        with using_instance(inst):
            if not server_pid():
                continue
            mb, cards, _gtt = _ram_demand(load_params(), str(CARD))
        own += mb
        for card, g in cards.items():
            gtts[card] = max(gtts.get(card, 0), g)
        names.append(iid)
    return own, gtts, names


def ram_budget(params, include_others=False):
    """Can this config's host-RAM demand fit in physical RAM?

    Returns a verdict, the arithmetic behind it, and the largest CACHE_RAM that
    would fit. 'impossible' is refused on save; 'risk' warns and saves.
    """
    total = _meminfo_mb("MemTotal:")
    gtt = _gtt_ceiling_mb()

    def _int(k, d=0):
        try:
            return int(str(params.get(k, d)).strip() or d)
        except ValueError:
            return d

    # GTT only exists as a risk when the GPU actually holds the weights. Under
    # BACKEND=cpu, or NGL=0, nothing is on the card, so there is nothing for
    # amdgpu to evict and the ceiling does not apply. The weights are then in
    # page cache via mmap, which IS reclaimable and so is not counted here.
    on_gpu = str(params.get("BACKEND", "vulkan")) != "cpu" and _int("NGL", 99) > 0
    if not on_gpu:
        gtt = 0
    else:
        gtt = _ram_demand(params, str(CARD))[2]

    # Without mmap, every weight that is not on the card is anonymous memory:
    # not reclaimable, not lazily paged. At minimum that is the model minus the
    # card's whole VRAM. For a 79 GiB model with a 54 GiB engram table on a
    # 24 GB card, that alone exceeds this host - under mmap the same table is
    # read row-by-row from disk and costs almost nothing resident.
    lm = str(params.get("LOAD_MODE", "") or "")
    pinned_weights = 0
    if lm in ("none", "mlock", "mmap+mlock", "dio"):
        # an incomplete split model still counts the parts that exist (fail closed)
        mb = (model_bytes(params.get("MODEL")) or sum(
            f.stat().st_size for f in model_shards(params.get("MODEL")) if f.exists())) // 1048576
        on_card = 0
        if on_gpu:
            on_card = (_device_record(INST().get("device")) or {}).get("vram_total_mib") or 0
        pinned_weights = max(mb - on_card, 0)
        # With a readable header, count exactly what stays on the CPU instead of
        # "model minus one card" - that ignored the second card and -ot entirely.
        try:
            pl = None if INST()["legacy"] else estimate_placement({**load_params(), **params})
        except Exception:
            pl = None
        if pl and on_gpu:
            pinned_weights = pl["cpu_resident_mib"] + pl["cpu_mapped_mib"] + pl["lazy_mib"]
    cache = _int("CACHE_RAM", 0)
    # -1 means "let llama.cpp size it", which in practice tracks free RAM and
    # is the same unbounded exposure as a very large explicit value.
    cache_eff = max(total - LLAMA_BASE_MB - OTHER_SERVICES_MB, 0) if cache < 0 else cache
    # A projector left on the CPU is host RAM for the life of the process.
    mmproj = 0 if _int("MMPROJ_OFFLOAD", 0) else (896 if _int("USE_MMPROJ", 1) else 0)
    fixed = LLAMA_BASE_MB + OTHER_SERVICES_MB + mmproj + pinned_weights
    others_mb, others = 0, []
    if include_others:
        o_own, o_gtt, others = others_ram()
        mine = _ram_demand(params, str(CARD))[1] if on_gpu else {}
        # a card shared with a running instance: its ceiling counts once, not twice
        gtt = sum(g for c, g in mine.items() if c not in o_gtt) + \
            sum(max(0, g - o_gtt[c]) for c, g in mine.items() if c in o_gtt)
        others_mb = o_own + sum(o_gtt.values())
        fixed += others_mb
    worst = fixed + cache_eff + gtt
    headroom = total - worst
    floor = _int("RAM_FLOOR_MB", 0)

    if total and worst > total:
        verdict = "impossible"
        detail = ((f"Running instance(s) {', '.join(others)} already account for "
                   f"{others_mb} MB of that. " if others else "") +
                  f"Worst case needs {worst} MB but the host has {total} MB. If the GPU "
                  f"evicts to GTT while the prompt cache is full, this config cannot fit "
                  f"in RAM - that is the livelock that hard-locked this box on Sep 4, 8 "
                  f"and 9.")
    elif total and headroom < 2048:
        verdict = "risk"
        detail = (f"Only {headroom} MB spare in the worst case. The watchdog polls every "
                  f"2 s and has been observed arriving at 74-898 MB, well past its floor, "
                  f"so a margin this thin is not reliably survivable.")
    else:
        verdict = "ok"
        detail = f"{headroom} MB spare with GTT fully evicted and the prompt cache full."

    max_cache = max(total - fixed - gtt - 2048, 0) if total else 0
    notes = []
    if pinned_weights:
        notes.append(f"LOAD_MODE={lm} keeps weights off the card resident in RAM: at least "
                     f"{pinned_weights} MB (model size minus the whole card). Use auto/mmap to "
                     "let the kernel page them - and LAZY_MODE needs mmap anyway.")
    if cache < 0:
        notes.append("CACHE_RAM=-1 lets llama.cpp size the prompt cache from free RAM, "
                     "so it is modelled here as taking everything left.")
    if floor and floor < 2048:
        notes.append(f"RAM_FLOOR_MB={floor} is below the 2048 MB the watchdog needs to "
                     "act before the kernel has to; every observed kill landed under it.")
    return dict(verdict=verdict, detail=detail, total_mb=total, gtt_ceiling_mb=gtt,
                cache_ram_mb=cache, cache_effective_mb=cache_eff, mmproj_host_mb=mmproj,
                base_mb=LLAMA_BASE_MB, other_services_mb=OTHER_SERVICES_MB,
                worst_case_mb=worst, headroom_mb=headroom,
                others_mb=others_mb, others=others, pinned_weights_mb=pinned_weights,
                max_safe_cache_ram_mb=max_cache, notes=notes)


# ============================================================================
# CRASH FORENSICS  (added 2026-09-09)
#
# /api/diagnostics only ever described the LIVE process, so two minutes after a
# hard lock it reported every check green - which is exactly what it did on the
# morning this was written, on a box that had just taken its third lock in six
# days. Post-mortem evidence lives in the journal and in the params history,
# and nothing in the panel read either.
#
# A host hard lock leaves NO shutdown sequence in its boot, so the reliable
# signal is negative: a boot that ended with no poweroff/reboot markers and is
# followed by another boot died on its feet. That is the test used here.
# ============================================================================
EVENT_RE = (
    r"Reached target (Power-Off|Poweroff|Reboot|Halt)|systemd-shutdown"
    r"|\[WATCHDOG\]|oom-kill|Out of memory: Killed"
    r"|GPU reset|ring .* timeout|device wedged|device lost|ErrorDeviceLost"
    r"|Linux version [0-9]"
)
_forensics_cache = {}


def _params_timeline():
    """When each parameter set was written, so a crash can be attributed.

    The panel writes a params.env.bak-* on every save, so file mtimes are a
    real changelog of what this box was configured with and when.
    """
    out = []
    for f in list(PANEL.glob("params.env.bak-*")) + list(PANEL.glob("params.env.*.bak")) \
            + list(PANEL.glob("params.env.presplit-*")) + list(PANEL.glob("params-*.env")) \
            + [PANEL / "params.env"]:
        try:
            vals = _read_env_file(f)
            out.append(dict(file=f.name, at=f.stat().st_mtime,
                            CTX=vals.get("CTX"), CACHE_RAM=vals.get("CACHE_RAM"),
                            KV_TYPE=vals.get("KV_TYPE"), BATCH=vals.get("BATCH"),
                            UBATCH=vals.get("UBATCH"),
                            RAM_FLOOR_MB=vals.get("RAM_FLOOR_MB")))
        except OSError:
            pass
    out.sort(key=lambda r: r["at"])
    return out


def _params_at(ts, timeline):
    """The most recent parameter set written at or before ts."""
    prev = None
    for r in timeline:
        if r["at"] <= ts:
            prev = r
        else:
            break
    return prev


def collect_boots(max_boots=12):
    """Per-boot event counts and a lived/died verdict.

    One journalctl pass over every boot, bucketed by _BOOT_ID. Per-boot calls
    took seconds each on this journal; this takes one.
    """
    boots = []
    try:
        raw = subprocess.run(["journalctl", "--list-boots", "-o", "json", "--no-pager"],
                             capture_output=True, text=True, timeout=30).stdout
        boots = json.loads(raw or "[]")
    except Exception:
        return []
    boots = boots[-max_boots:]
    ids = {b.get("boot_id"): b for b in boots}
    for b in boots:
        b["events"] = dict(shutdown=0, watchdog=0, oom=0, gpu=0)
        b["kernel"] = None
        b["samples"] = []

    since = min((b.get("first_entry", 0) or 0) for b in boots) // 1_000_000 if boots else 0
    cmd = ["journalctl", "--no-pager", "-o", "json",
           "--output-fields=MESSAGE,_BOOT_ID,_TRANSPORT", "--case-sensitive=false",
           "-g", EVENT_RE]
    if since:
        cmd += ["--since", "@%d" % since]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        proc = None
    if proc:
        for line in proc.stdout.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            b = ids.get(rec.get("_BOOT_ID"))
            if not b:
                continue
            msg = rec.get("MESSAGE") or ""
            if isinstance(msg, list):
                msg = "".join(chr(c) for c in msg if isinstance(c, int))
            low = msg.lower()
            kern = rec.get("_TRANSPORT") == "kernel"
            # The launch script echoes "[CHECK] prior GPU resets since boot" and
            # the panel echoes its own diagnostic prose, both of which contain
            # the very strings searched for here. Counting those made a clean
            # boot report 21 GPU faults.
            if "[check]" in low or "want none" in low:
                continue
            if "reached target power" in low or "reached target reboot" in low \
               or "reached target halt" in low or "systemd-shutdown" in low:
                b["events"]["shutdown"] += 1
            if "[watchdog]" in low:
                b["events"]["watchdog"] += 1
                m = re.search(r"MemAvailable (\d+)MB", msg)
                if m:
                    b["samples"].append(int(m.group(1)))
            # One OOM writes several lines (invoked oom-killer, oom-kill:constraint,
            # Out of memory: Killed). Only the last is one-per-kill.
            if "out of memory: killed" in low:
                b["events"]["oom"] += 1
            # A real GPU fault comes from the kernel, or from ggml itself. Any
            # other process merely talking about one does not count.
            engine_fault = "ggml_vulkan" in low or "errordevicelost" in low
            if (kern or engine_fault) and (
                    "gpu reset" in low or "device wedged" in low or "device lost" in low
                    or "errordevicelost" in low or re.search(r"ring .* timeout", low)):
                b["events"]["gpu"] += 1
            if b["kernel"] is None:
                m = re.search(r"Linux version (\S+)", msg)
                if m:
                    b["kernel"] = m.group(1)

    out = []
    prev_last = 0.0
    for i, b in enumerate(boots):
        first = (b.get("first_entry") or 0) / 1e6
        last = (b.get("last_entry") or 0) / 1e6
        # This box boots with an unset RTC and lets chrony correct it, so a boot
        # can claim a first entry weeks before the boot that preceded it. Trust
        # the previous boot's end instead; 43-day uptimes are a clock artefact.
        clock_skew = bool(prev_last and first < prev_last)
        if clock_skew:
            first = prev_last
        prev_last = last
        is_current = (i == len(boots) - 1)
        ev = b["events"]
        if is_current:
            verdict, why = "running", "current boot"
        elif ev["shutdown"]:
            verdict, why = "clean", "shutdown sequence present"
        else:
            verdict, why = "hard-lock", ("boot ended with no poweroff or reboot markers - "
                                         "the host stopped executing rather than shutting down")
        out.append(dict(index=b.get("index"), boot_id=b.get("boot_id"),
                        first=first, last=last, duration_s=max(int(last - first), 0),
                        clock_skew=clock_skew,
                        kernel=b["kernel"], verdict=verdict, why=why,
                        watchdog_kills=ev["watchdog"], oom_kills=ev["oom"],
                        gpu_faults=ev["gpu"], shutdown_markers=ev["shutdown"],
                        watchdog_lows=sorted(b["samples"])[:5]))
    return out


def crash_report(force=False):
    """Ranked, evidence-backed causes for this host's failures."""
    now = time.time()
    if not force and _forensics_cache.get("at", 0) > now - 300:
        return _forensics_cache["data"]

    boots = collect_boots()
    timeline = _params_timeline()
    findings = []

    locks = [b for b in boots if b["verdict"] == "hard-lock"]
    ram_boots = [b for b in boots if b["watchdog_kills"] or b["oom_kills"]]
    tot_wd = sum(b["watchdog_kills"] for b in boots)
    tot_oom = sum(b["oom_kills"] for b in boots)
    tot_gpu = sum(b["gpu_faults"] for b in boots)

    if tot_wd or tot_oom:
        # Attribute to config: compare CACHE_RAM in the boots that failed
        # against the boots that did not.
        # Only boots that ran long enough to exercise the prompt cache tell you
        # anything: a 16-second boot with no RAM event is not evidence that its
        # CACHE_RAM was safe.
        bad, good = set(), set()
        for b in boots:
            if b["duration_s"] < 3600:
                continue
            p = _params_at(b["last"] or now, timeline)
            c = (p or {}).get("CACHE_RAM")
            if c is None:
                continue
            (bad if (b["watchdog_kills"] or b["oom_kills"]) else good).add(str(c))
        ev = [f"{tot_wd} watchdog kills and {tot_oom} kernel OOM kills, in boot" +
              ("s " if len(ram_boots) != 1 else " ") +
              ", ".join(str(b["index"]) for b in ram_boots)]
        lows = sorted({x for b in boots for x in b["watchdog_lows"]})[:6]
        if lows:
            ev.append("watchdog fired at MemAvailable " +
                      ", ".join(f"{x}MB" for x in lows) +
                      " - consistently far below its own floor, meaning the 2 s poll "
                      "arrives after the collapse, not before it")
        if bad:
            ev.append("CACHE_RAM in effect during failing boots: " + ", ".join(sorted(bad)))
        if good:
            ev.append("CACHE_RAM during boots over an hour with no RAM event: " +
                      ", ".join(sorted(good)))
        budget = ram_budget(load_params())
        ev.append(f"current config worst case {budget['worst_case_mb']} MB vs "
                  f"{budget['total_mb']} MB physical ({budget['verdict']})")
        findings.append(dict(
            id="host_ram_exhaustion", severity="critical",
            title="Host RAM exhaustion from prompt cache plus GTT eviction",
            evidence=ev,
            meaning="When VRAM runs near full amdgpu evicts model weights into GTT, "
                    "which is host RAM pinned by the driver and invisible to ps RSS "
                    "and to the cgroup. GTT ceiling plus the host prompt cache plus "
                    "process overhead has to fit in physical RAM. When it does not, "
                    "the box goes into swap thrash and stops scheduling - which is "
                    "why the fatal ones leave no log at all.",
            fix=f"Keep CACHE_RAM at or below {budget['max_safe_cache_ram_mb']} MiB, and "
                f"raise RAM_FLOOR_MB to at least 3072 so the watchdog can act while "
                f"the kernel can still schedule the kill."))

    if tot_gpu:
        gb = [b for b in boots if b["gpu_faults"]]
        findings.append(dict(
            id="gpu_fault", severity="high",
            title="GPU compute hang (device lost / ring timeout)",
            evidence=[f"{tot_gpu} GPU fault lines across boots " +
                      ", ".join(str(b["index"]) for b in gb)],
            meaning="VK_ERROR_DEVICE_LOST is a kernel-level GPU hang. llama-server has "
                    "no recovery path: every later request fails in milliseconds while "
                    "the server still looks alive.",
            fix="A VM_L2 or page fault means VRAM OOM at the prefill peak - lower UBATCH "
                "or CTX. A ring timeout means raising amdgpu.lockup_timeout instead."))

    # A boot that died within three minutes cannot be a memory or a thermal
    # story: the model is not loaded yet and the host is idle. Restricted to
    # the last 3 days because this box's bring-up week is full of locks with a
    # known cause (--spec-draft-model double-loads) that are not open questions.
    cutoff = now - 3 * 86400
    early = [b for b in locks if b["duration_s"] < 180 and not b["watchdog_kills"]
             and not b["oom_kills"] and not b["gpu_faults"] and b["last"] >= cutoff]
    if early:
        ev, kernel_note = [], ""
        for b in early:
            ev.append(f"boot {b['index']} lasted {b['duration_s']}s and ended with no "
                      f"shutdown markers, no watchdog kill, no OOM and no GPU fault")
            prior = [x for x in boots if x["index"] == (b["index"] or 0) - 1]
            if prior and prior[0]["kernel"] and b["kernel"] \
                    and prior[0]["kernel"] != b["kernel"]:
                kernel_note = (f" Boot {b['index']} was also the first boot on kernel "
                               f"{b['kernel']}; the boot before it ran "
                               f"{prior[0]['kernel']}.")
                ev.append(f"kernel changed at boot {b['index']}: "
                          f"{prior[0]['kernel']} -> {b['kernel']}")
        findings.append(dict(
            id="early_boot_lock", severity="high",
            title=f"Host locked during boot ({len(early)} in the last 3 days)",
            evidence=ev,
            meaning="A lock this early happens before the model is loaded and with the "
                    "host essentially idle, so neither host RAM exhaustion nor sustained "
                    "GPU load can explain it." + kernel_note,
            fix=("Boot the previous kernel from the GRUB menu to separate a kernel "
                 "regression from hardware." if kernel_note else
                 "Watch for a repeat. A single early lock with no logged cause is not "
                 "enough to attribute.")))

    if not findings:
        findings.append(dict(
            id="none", severity="ok", title="No crash signature in the retained journal",
            evidence=[f"{len(boots)} boots examined, all accounted for"],
            meaning="", fix=""))

    order = dict(critical=0, high=1, medium=2, ok=9)
    findings.sort(key=lambda f: order.get(f["severity"], 5))
    data = dict(generated=now, boots=boots, findings=findings,
                params_timeline=timeline[-12:],
                totals=dict(boots=len(boots), hard_locks=len(locks),
                            clean=len([b for b in boots if b["verdict"] == "clean"]),
                            watchdog_kills=tot_wd, oom_kills=tot_oom, gpu_faults=tot_gpu),
                budget=ram_budget(load_params()))
    _forensics_cache.update(at=now, data=data)
    return data



# ============================================================================
# DEVICES  (added 2026-09-17)
#
# Discovered from PCI sysfs, never by card number (card numbering moved again
# with the XPS 8930 move: the 7900 XTX is card3 now). Each record says which
# backends can drive it with the driver that is bound RIGHT NOW - the RTX 2060
# on nouveau can run Vulkan (Mesa NVK) but not CUDA, which needs the NVIDIA
# driver bound instead.
# ============================================================================
_VENDORS = {"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel"}
DEVICE_CACHE = PANEL / ".device-cache.json"
_dev_cache = {}


def _lspci_name(pci):
    out = sh(f"lspci -s {shlex.quote(pci)} 2>/dev/null")
    return out.split(": ", 1)[1].strip() if ": " in out else pci


def _nvidia_smi(pci=None):
    """nvidia-smi for one device, or None. Only exists with the NVIDIA driver."""
    if not os.path.exists("/usr/bin/nvidia-smi"):
        return None
    raw = sh("nvidia-smi --query-gpu=pci.bus_id,name,driver_version,memory.total,"
             "memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu,"
             "pstate --format=csv,noheader,nounits", timeout=10)
    keys = ("bus_id", "name", "driver", "mem_total_mib", "mem_used_mib", "temp_c",
            "power_w", "power_limit_w", "util_pct", "pstate")
    rows = [dict(zip(keys, [x.strip() for x in ln.split(",")])) for ln in raw.splitlines()
            if ln.count(",") >= 9]
    if pci is None:
        return rows
    for r in rows:
        # nvidia-smi prints 00000000:07:00.0; sysfs says 0000:07:00.0
        if r["bus_id"].lower().endswith(pci.lower()[-7:]):
            return r
    return None


def _probe_vulkan_vram(pci, driver):
    """VRAM of a device that exposes no sysfs counter (nouveau), by asking the
    Vulkan build to enumerate through that driver's ICD. Loads no model.
    Cached on disk: the answer only changes when the hardware does."""
    key = f"{pci}:{driver}"
    if not _dev_cache:
        try:
            _dev_cache.update(json.loads(DEVICE_CACHE.read_text()))
        except (OSError, ValueError):
            pass
    if key in _dev_cache:
        return _dev_cache[key]
    icd = _icd_for_driver(driver)
    bd = active_build("vulkan")
    if not icd or not bd:
        return None
    out = sh(f"VK_DRIVER_FILES={icd} VK_ICD_FILENAMES={icd} LD_LIBRARY_PATH={bd} "
             f"timeout 60 {bd}/llama-server --list-devices 2>&1", timeout=70)
    m = re.findall(r"Vulkan\d+: .*?\((\d+) MiB", out)
    val = int(m[0]) if len(m) == 1 else None
    if val:
        _dev_cache[key] = val
        try:
            DEVICE_CACHE.write_text(json.dumps(_dev_cache, indent=1))
        except OSError:
            pass
    return val


def _icd_for_driver(driver):
    name = {"amdgpu": "radeon_icd", "nouveau": "nouveau_icd", "nvidia": "nvidia_icd",
            "i915": "intel_icd", "xe": "intel_icd"}.get(driver or "")
    if not name:
        return None
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        # NOT the .x86_64.json spelling: it does not exist on Ubuntu 26.04 (trap 4)
        for fn in (f"{name}.json", f"{name}.x86_64.json"):
            if os.path.isfile(f"{d}/{fn}"):
                return f"{d}/{fn}"
    return None


_vk_enum_cache = {}


def vulkan_enumeration(bindir, icds, fresh=False):
    """[(index, name, MiB)] exactly as this build enumerates with these ICDs.

    Needed because with two ICDs loaded, which card is Vulkan0 and which is
    Vulkan1 is the loader's business, not ours - so -dev, -ot buffer names and
    --spec-draft-device are resolved against what the build actually reports.
    Loads no model.
    """
    key = (str(bindir), tuple(icds))
    if not fresh and key in _vk_enum_cache:
        return _vk_enum_cache[key]
    joined = ":".join(icds)
    # env -u: a caller that already pinned GGML_VK_VISIBLE_DEVICES (main's
    # launch script does, before asking which card is which) would otherwise
    # hide every device but the first from this enumeration
    out = sh(f"env -u GGML_VK_VISIBLE_DEVICES "
             f"VK_DRIVER_FILES={shlex.quote(joined)} VK_ICD_FILENAMES={shlex.quote(joined)} "
             f"LD_LIBRARY_PATH={shlex.quote(str(bindir))} timeout 60 "
             f"{shlex.quote(str(bindir))}/llama-server --list-devices 2>&1", timeout=70)
    res = [(int(m.group(1)), m.group(2).strip(), int(m.group(3)))
           for m in re.finditer(r"Vulkan(\d+): (.*?) \((\d+) MiB", out)]
    if res:
        _vk_enum_cache[key] = res
    return res


def _vk_matches(dev, name):
    """Does a Vulkan device name belong to this PCI device? By vendor, which is
    unambiguous while every device in an instance has a distinct driver."""
    n = name.lower()
    return {"amd": ("amd" in n or "radeon" in n), "nvidia": ("nvidia" in n or "geforce" in n),
            "intel": "intel" in n}.get(dev.get("vendor"), False)


def gpu_devices(probe=True):
    out = []
    try:
        pdevs = sorted(Path("/sys/bus/pci/devices").iterdir())
    except OSError:
        pdevs = []
    for d in pdevs:
        try:
            if not (d / "class").read_text().strip().startswith("0x03"):
                continue
            vendor = _VENDORS.get((d / "vendor").read_text().strip(), "other")
        except OSError:
            continue
        drv = os.path.basename(os.path.realpath(d / "driver")) if (d / "driver").exists() else None
        vram = None
        if drv == "amdgpu":
            vram = read_int(d / "mem_info_vram_total") // 1048576 or None
        elif drv == "nvidia":
            r = _nvidia_smi(d.name)
            vram = int(float(r["mem_total_mib"])) if r and r.get("mem_total_mib") else None
        elif drv == "nouveau" and probe:
            vram = _probe_vulkan_vram(d.name, drv)
        backends, notes = ["cpu"], []
        if vendor == "amd" and drv == "amdgpu":
            backends = ["vulkan", "rocm", "cpu"]
        elif vendor == "nvidia":
            backends = ["vulkan", "cuda", "cpu"]
            if drv != "nvidia":
                notes.append(f"bound to {drv or 'no driver'}: Vulkan works through Mesa NVK, "
                             "CUDA needs the NVIDIA driver bound instead.")
        elif vendor == "intel":
            notes.append("integrated GPU - never an inference device on this box (trap 4).")
        out.append(dict(pci=d.name, vendor=vendor, driver=drv, name=_lspci_name(d.name),
                        vram_total_mib=vram, backends=backends,
                        cuda_ready=(drv == "nvidia"),
                        usable=vendor in ("amd", "nvidia") and drv is not None,
                        notes=notes))
    out.append(dict(pci="cpu", vendor="cpu", driver=None, name="CPU only (no GPU)",
                    vram_total_mib=0, backends=["cpu"], cuda_ready=False, usable=True,
                    notes=[]))
    return out


def _device_record(pci):
    if not pci:
        return None
    for d in gpu_devices():
        if d["pci"] == pci:
            return d
    return None


# ============================================================================
# LAUNCH PLAN  (added 2026-09-17)
#
# The single place a parameter set becomes an argv + environment for an
# instance. instance_launch.py executes exactly what this returns, and the
# panel shows the same plan before anything starts, so what you review is
# what runs.
#
# NOTE ON main: the legacy bash scripts still launch main, and they read only
# the ~30 original keys. legacy_ignored_keys() says which saved keys they drop
# on the floor, so the UI can mark them instead of implying they apply.
# ============================================================================
# (KEY, kind, flag[, off_flag])
#   val     pass "flag value" when the value is non-empty
#   tri     "on" -> flag, "off" -> off_flag, "" -> nothing (stock)
#   switch  "on" -> flag, anything else -> nothing (flag has no negation)
LAUNCH_FLAGS = [
    ("LOAD_MODE", "val", "--load-mode"), ("LAZY_MODE", "val", "--lazy-mode"),
    ("LORA", "val", "--lora"), ("LORA_SCALED", "val", "--lora-scaled"),
    ("KV_OFFLOAD", "tri", "--kv-offload", "--no-kv-offload"),
    ("SWA_FULL", "switch", "--swa-full"),
    ("KV_UNIFIED", "tri", "--kv-unified", "--no-kv-unified"),
    ("KV_UNIFIED_PER_SLOT", "val", "--kv-unified-per-slot"),
    ("CTX_CHECKPOINTS", "val", "--ctx-checkpoints"),
    ("CHECKPOINT_MIN_STEP", "val", "--checkpoint-min-step"),
    ("CACHE_PROMPT", "tri", "--cache-prompt", "--no-cache-prompt"),
    ("CACHE_IDLE_SLOTS", "tri", "--cache-idle-slots", "--no-cache-idle-slots"),
    ("CONTEXT_SHIFT", "tri", "--context-shift", "--no-context-shift"),
    ("FIT", "val", "--fit"), ("FIT_TARGET", "val", "--fit-target"),
    ("FIT_CTX", "val", "--fit-ctx"),
    ("FLASH_ATTN", "val", "--flash-attn"), ("PARALLEL", "val", "--parallel"),
    ("THREADS_HTTP", "val", "--threads-http"),
    ("BACKEND_SAMPLING", "switch", "--backend-sampling"),
    ("OP_OFFLOAD", "tri", "--op-offload", "--no-op-offload"),
    ("REPACK", "tri", "--repack", "--no-repack"),
    ("NO_HOST", "switch", "--no-host"),
    ("OVERRIDE_TENSORS", "val", "--override-tensor"),
    ("SPLIT_MODE", "val", "--split-mode"), ("TENSOR_SPLIT", "val", "--tensor-split"),
    ("MAIN_GPU", "val", "--main-gpu"),
    ("N_CPU_MOE", "val", "--n-cpu-moe"), ("CPU_MOE", "switch", "--cpu-moe"),
    ("ROPE_SCALING", "val", "--rope-scaling"), ("ROPE_SCALE", "val", "--rope-scale"),
    ("ROPE_FREQ_BASE", "val", "--rope-freq-base"),
    ("ROPE_FREQ_SCALE", "val", "--rope-freq-scale"),
    ("YARN_ORIG_CTX", "val", "--yarn-orig-ctx"),
    ("YARN_EXT_FACTOR", "val", "--yarn-ext-factor"),
    ("YARN_ATTN_FACTOR", "val", "--yarn-attn-factor"),
    ("YARN_BETA_FAST", "val", "--yarn-beta-fast"),
    ("YARN_BETA_SLOW", "val", "--yarn-beta-slow"),
    ("REASONING_BUDGET_MESSAGE", "val", "--reasoning-budget-message"),
    ("KEEP", "val", "--keep"),
    ("SAMPLERS", "val", "--samplers"), ("SEED", "val", "--seed"),
    ("TOP_N_SIGMA", "val", "--top-nsigma"), ("TYPICAL_P", "val", "--typical"),
    ("REPEAT_PENALTY", "val", "--repeat-penalty"),
    ("REPEAT_LAST_N", "val", "--repeat-last-n"),
    ("PRESENCE_PENALTY", "val", "--presence-penalty"),
    ("FREQUENCY_PENALTY", "val", "--frequency-penalty"),
    ("DRY_MULTIPLIER", "val", "--dry-multiplier"), ("DRY_BASE", "val", "--dry-base"),
    ("DRY_ALLOWED_LENGTH", "val", "--dry-allowed-length"),
    ("DRY_PENALTY_LAST_N", "val", "--dry-penalty-last-n"),
    ("XTC_PROBABILITY", "val", "--xtc-probability"),
    ("XTC_THRESHOLD", "val", "--xtc-threshold"),
    ("ADAPTIVE_TARGET", "val", "--adaptive-target"),
    ("ADAPTIVE_DECAY", "val", "--adaptive-decay"),
    ("DYNATEMP_RANGE", "val", "--dynatemp-range"),
    ("DYNATEMP_EXP", "val", "--dynatemp-exp"),
    ("MIROSTAT", "val", "--mirostat"), ("MIROSTAT_LR", "val", "--mirostat-lr"),
    ("MIROSTAT_ENT", "val", "--mirostat-ent"),
    ("TIMEOUT", "val", "--timeout"), ("SSE_PING_INTERVAL", "val", "--sse-ping-interval"),
    ("SLEEP_IDLE_SECONDS", "val", "--sleep-idle-seconds"),
    ("WEBUI", "tri", "--webui", "--no-webui"), ("PROPS", "switch", "--props"),
    ("UI_CONFIG_FILE", "val", "--ui-config-file"),
    ("UI_MCP_PROXY", "tri", "--ui-mcp-proxy", "--no-ui-mcp-proxy"),
    ("SLOTS", "tri", "--slots", "--no-slots"), ("ALIAS", "val", "--alias"),
    ("LOG_VERBOSITY", "val", "--log-verbosity"),
]
SPEC_FLAGS = [
    ("SPEC_N_MIN", "val", "--spec-draft-n-min"), ("SPEC_P_SPLIT", "val", "--spec-draft-p-split"),
    ("SPEC_DRAFT_DEVICE", "val", "--spec-draft-device"),
    ("SPEC_DRAFT_BACKEND_SAMPLING", "tri", "--spec-draft-backend-sampling",
     "--no-spec-draft-backend-sampling"),
] + [(k, "val", "--" + k.lower().replace("_", "-").replace("spec-ngram-", "spec-ngram-"))
     for k in ("SPEC_NGRAM_MOD_N_MIN", "SPEC_NGRAM_MOD_N_MAX", "SPEC_NGRAM_MOD_N_MATCH",
               "SPEC_NGRAM_SIMPLE_SIZE_N", "SPEC_NGRAM_SIMPLE_SIZE_M",
               "SPEC_NGRAM_SIMPLE_MIN_HITS", "SPEC_NGRAM_MAP_K_SIZE_N",
               "SPEC_NGRAM_MAP_K_SIZE_M", "SPEC_NGRAM_MAP_K_MIN_HITS",
               "SPEC_NGRAM_MAP_K4V_SIZE_N", "SPEC_NGRAM_MAP_K4V_SIZE_M",
               "SPEC_NGRAM_MAP_K4V_MIN_HITS")]
VISION_FLAGS = [("MMPROJ_DEVICE", "val", "--mmproj-device"),
                ("IMAGE_MAX_TOKENS", "val", "--image-max-tokens"),
                ("MTMD_BATCH_MAX_TOKENS", "val", "--mtmd-batch-max-tokens")]

# EXTRA_ARGS may not smuggle in what the plan owns or what the guards exist
# to stop. The draft-model spellings are trap 1; the rest would silently
# override the instance's identity (port, log, model) and break the panel.
EXTRA_FORBIDDEN = {"-md", "--model-draft", "--spec-draft-model", "-hfd", "-hfrd",
                   "--hf-repo-draft", "--spec-draft-hf", "-m", "--model", "--port",
                   "--host", "--log-file", "--api-key", "--api-key-file",
                   "--slot-save-path", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url"}
ENV_PREFIX = {"vulkan": ("GGML_VK_",), "rocm": ("GGML_CUDA_", "HSA_"),
              "cuda": ("GGML_CUDA_",), "cpu": ()}
_flag_cache = {}


def build_flags(bindir):
    """Every --flag / -f the build's --help accepts. Cached per binary mtime."""
    exe = Path(bindir) / "llama-server"
    try:
        key = (str(exe), exe.stat().st_mtime)
    except OSError:
        return set()
    if key not in _flag_cache:
        raw = sh(f"LD_LIBRARY_PATH={shlex.quote(str(bindir))} {shlex.quote(str(exe))} --help 2>&1",
                 timeout=60)
        # Left column only: descriptions mention other flags in prose. Aliases
        # are written "-b,    --batch-size N", so collapse ", <spaces>" first or
        # the column split cuts after the short form.
        left = "\n".join(re.split(r"\s{2,}", re.sub(r",\s+", ", ", ln.strip()), maxsplit=1)[0]
                         for ln in raw.splitlines() if ln.lstrip().startswith("-"))
        _flag_cache[key] = set(re.findall(r"(?<![\w-])(--?[a-zA-Z][\w-]*)", left))
    return _flag_cache[key]


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def vulkan_pinning(devs, bindir, check_runtime=True):
    """(env, [(VulkanN, device record)], errors) that pin a Vulkan process to
    exactly these devices, primary first. The one implementation behind both
    launch_plan() and main_devices.py, so main and the other instances cannot
    disagree about which card is which."""
    errors, icds, vk_map = [], [], []
    alldev = {d["pci"]: d for d in gpu_devices(probe=False)}
    for d in devs:
        icd = _icd_for_driver(d.get("driver"))
        if d.get("vendor") == "intel":
            errors.append("refusing the Intel iGPU as a Vulkan device (trap 4)")
        if not icd:
            errors.append(f"no Vulkan ICD for driver {d.get('driver')!r} ({d.get('name')})")
        elif icd not in icds:
            icds.append(icd)
        same = [x for x in alldev.values() if x.get("driver") == d.get("driver")
                and x["pci"] != "cpu"]
        if len(same) > 1:
            errors.append(f"{len(same)} devices share the {d.get('driver')} ICD; pinning "
                          "by ICD cannot tell them apart")
    if len({d["pci"] for d in devs}) != len(devs):
        errors.append("the same device is listed twice")
    env = dict(VK_DRIVER_FILES=":".join(icds), VK_ICD_FILENAMES=":".join(icds),
               # trap 2 - never negotiable, never exposed
               GGML_VK_ALLOW_SYSMEM_FALLBACK="0")
    if len(devs) == 1:
        env["GGML_VK_VISIBLE_DEVICES"] = "0"
    elif bindir and icds and check_runtime and not errors:
        enum = vulkan_enumeration(bindir, icds)
        if len(enum) != len(devs):
            errors.append(f"expected {len(devs)} Vulkan devices with these drivers, the build "
                          f"enumerates {len(enum)}: {', '.join(n for _, n, _ in enum) or 'none'}")
        else:
            for d in devs:
                hit = [e for e in enum if _vk_matches(d, e[1])]
                if len(hit) != 1:
                    errors.append(f"cannot identify {d.get('name')} among: "
                                  f"{', '.join(n for _, n, _ in enum)}")
                    break
                vk_map.append((f"Vulkan{hit[0][0]}", d))
    return env, vk_map, errors


def launch_plan(values=None, check_runtime=True):
    if INST().get("engine") in GEN_ENGINES:
        return GEN_ENGINES[INST()["engine"]].launch_plan(INST(), values)
    return _llama_launch_plan(values, check_runtime)


def _llama_launch_plan(values=None, check_runtime=True):
    """argv + env for the CURRENT instance, with every refusal spelled out.

    Returns dict(argv, env, errors, warnings, rundir, template, api_key, ...).
    errors non-empty means the launcher will refuse to start.
    """
    inst = INST()
    v = dict(DEFAULTS)
    v.update(values if values is not None else load_params())
    errors, warnings = [], []
    backend = str(v.get("BACKEND") or "vulkan")
    alldev = {d["pci"]: d for d in gpu_devices(probe=False)}
    devs = []
    for pci in inst.get("devices") or [inst["device"]]:
        d = alldev.get(pci)
        if not d:
            errors.append(f"device {pci} is not present on this host")
            d = dict(pci=pci, driver=None, backends=[], vendor="?", name=pci)
        elif backend not in d["backends"]:
            errors.append(f"BACKEND={backend} cannot drive {d['name']} "
                          f"(possible: {', '.join(d['backends'])})")
        devs.append(d)
    dev = devs[0]
    if len(devs) > 1 and backend != "vulkan":
        errors.append(f"multi-device instances are Vulkan-only for now; BACKEND={backend} "
                      "spans one device")

    bindir = active_build(backend)
    if not bindir or not os.path.exists(f"{bindir}/llama-server"):
        errors.append(f"no {backend} llama.cpp build is selected - install one on the Builds tab")
        bindir = bindir or ""
    rundir = inst["rundir"]
    env = {"LD_LIBRARY_PATH": bindir}

    # ---- device pinning, per backend
    vk_map = []                         # [(VulkanN, device record)] in instance order
    if backend == "vulkan":
        venv, vk_map, verr = vulkan_pinning(devs, bindir, check_runtime=check_runtime and not errors)
        env.update(venv)
        errors += verr
    elif backend == "cuda":
        if dev.get("driver") != "nvidia":
            errors.append(f"CUDA needs the NVIDIA driver; {dev.get('name', 'this device')} is "
                          f"bound to {dev.get('driver')!r}. Use BACKEND=vulkan (NVK) until "
                          "the driver is installed.")
        nv = sorted(d["pci"] for d in gpu_devices(probe=False) if d.get("driver") == "nvidia")
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=str(nv.index(dev["pci"])) if dev["pci"] in nv else "")
    elif backend == "rocm":
        amd = sorted(d["pci"] for d in gpu_devices(probe=False) if d.get("driver") == "amdgpu")
        idx = str(amd.index(dev["pci"])) if dev["pci"] in amd else ""
        env.update(HIP_VISIBLE_DEVICES=idx, ROCR_VISIBLE_DEVICES=idx)
    for k, val in v.items():
        if any(k.startswith(p) for p in ENV_PREFIX.get(backend, ())) and str(val) != "" \
                and k != "GGML_VK_ALLOW_SYSMEM_FALLBACK":
            env[k] = str(val)
    if env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY"):
        warnings.append("GGML_CUDA_ENABLE_UNIFIED_MEMORY is set: an over-budget model spills "
                        "into host RAM at a fraction of the speed instead of failing to load.")

    # ---- files
    model = str(v.get("MODEL") or "")
    if not model or not os.path.isfile(model):
        errors.append(f"MODEL does not exist: {model or '(empty)'}")
    elif model_bytes(model) is None:
        missing = [f.name for f in model_shards(model) if not f.exists()]
        errors.append(f"split model is incomplete, missing: {', '.join(missing)}")
    try:
        _reject_full_size_draft(v)
    except ValueError as e:
        errors.append(str(e))
    use_mm = _truthy(v.get("USE_MMPROJ"))
    mm = str(v.get("MMPROJ") or "")
    if use_mm and not os.path.isfile(mm):
        errors.append(f"vision is on but MMPROJ does not exist: {mm or '(empty)'}")
    tsrc = str(v.get("TEMPLATE_SRC") or "")
    tcopy = None
    if tsrc:
        if not os.path.isfile(tsrc):
            errors.append(f"TEMPLATE_SRC does not exist: {tsrc}")
        elif _sha256_file(tsrc) != str(v.get("TEMPLATE_SHA256") or ""):
            errors.append(f"{Path(tsrc).name} changed since it was pinned - re-save parameters")
        tcopy = str(rundir / Path(tsrc).name)

    ngl = "0" if backend == "cpu" else str(v.get("NGL"))
    kv = str(v.get("KV_TYPE"))
    argv = [f"{bindir}/llama-server", "-m", model]
    if use_mm:
        argv += ["--mmproj", mm, "--image-min-tokens", str(v.get("IMAGE_MIN_TOKENS") or 1024)]
        if not _truthy(v.get("MMPROJ_OFFLOAD")):
            argv.append("--no-mmproj-offload")
        for key, _k, flag in VISION_FLAGS:
            if str(v.get(key, "")) != "":
                argv += [flag, str(v[key])]
    if vk_map:
        # primary device first: with the default layer split it also hosts the
        # output and, unless moved, the draft and projector
        argv += ["--device", ",".join(n for n, _ in vk_map)]
    argv += ["-c", str(v.get("CTX")), "--n-gpu-layers", ngl,
             "--cache-type-k", kv, "--cache-type-v", kv,
             "--cache-ram", str(v.get("CACHE_RAM"))]
    if int(str(v.get("CACHE_REUSE") or 0)) > 0:
        argv += ["--cache-reuse", str(v["CACHE_REUSE"])]
    argv.append("--jinja")
    if tcopy:
        argv += ["--chat-template-file", tcopy]
    pt = "true" if _truthy(v.get("PRESERVE_THINKING")) else "false"
    argv += ["--chat-template-kwargs",
             '{"reasoning_effort":"%s","preserve_thinking":%s,"max_tool_response_chars":%s}'
             % (v.get("REASONING_EFFORT"), pt, int(str(v.get("MAX_TOOL_RESPONSE_CHARS") or 0))),
             "--reasoning-preserve" if pt == "true" else "--no-reasoning-preserve",
             "--reasoning", str(v.get("REASONING") or "on"),
             "--reasoning-format", str(v.get("REASONING_FORMAT") or "deepseek"),
             "--reasoning-budget", str(v.get("REASONING_BUDGET"))]
    spec = str(v.get("SPEC_TYPE") or "")
    if spec and spec != "none":
        argv += ["--spec-type", spec, "--spec-draft-n-max", str(v.get("SPEC_N_MAX")),
                 "--spec-draft-p-min", str(v.get("SPEC_P_MIN") or 0)]
        d = str(v.get("SPEC_DRAFT_MODEL") or "")
        if d:
            argv += ["--spec-draft-model", d, "--spec-draft-ngl",
                     str(v.get("SPEC_DRAFT_NGL") or "all")]
        if str(v.get("SPEC_DRAFT_KV_TYPE") or ""):
            argv += ["--spec-draft-type-k", v["SPEC_DRAFT_KV_TYPE"],
                     "--spec-draft-type-v", v["SPEC_DRAFT_KV_TYPE"]]
        # a draft device only means something when there is a draft to place
        has_draft = bool(str(v.get("SPEC_DRAFT_MODEL") or "")) or "draft-mtp" in spec
        argv += _render_flags({k: val for k, val in v.items()
                              if has_draft or k != "SPEC_DRAFT_DEVICE"}, SPEC_FLAGS)
    argv += ["--temp", str(v.get("TEMP")), "--top-p", str(v.get("TOP_P")),
             "--top-k", str(v.get("TOP_K")), "--min-p", str(v.get("MIN_P")),
             "--metrics", "--no-warmup",
             "--host", str(v.get("HOST") or "127.0.0.1"), "--port", str(v.get("PORT")),
             "--slot-save-path", str(rundir / "slots"),
             "--log-file", str(rundir / "telemetry/engine_debug.log"),
             "--batch-size", str(v.get("BATCH")), "--ubatch-size", str(v.get("UBATCH")),
             "--threads", str(v.get("THREADS")), "--threads-batch", str(v.get("THREADS")),
             "--n-predict", str(v.get("N_PREDICT")), "--cont-batching"]
    argv += _render_flags(v, LAUNCH_FLAGS)
    if "--ui-config-file" in argv:
        # llama-server refuses to start on a missing or malformed UI config; the UI's
        # defaults are never worth an outage, so leave the flag out and say so
        _i = argv.index("--ui-config-file")
        try:
            json.loads(Path(argv[_i + 1]).read_text())
        except (OSError, ValueError, IndexError) as _e:
            warnings.append(f"web UI defaults file left out: {_e}")
            del argv[_i:_i + 2]
    api_key = str(v.get("API_KEY") or "")
    if api_key:
        # a key on the command line is readable by every local user via /proc
        argv += ["--api-key-file", str(rundir / "api-keys")]
    try:
        extra = shlex.split(str(v.get("EXTRA_ARGS") or ""))
    except ValueError as e:
        extra = []
        errors.append(f"EXTRA_ARGS does not parse: {e}")
    bad = sorted({a.split("=", 1)[0] for a in extra} & EXTRA_FORBIDDEN)
    if bad:
        errors.append(f"EXTRA_ARGS may not contain {', '.join(bad)} - the plan owns those, "
                      "and a draft model there would bypass the full-size-draft guard")
    argv += extra

    if str(v.get("LAZY_MODE") or "") == "on" and str(v.get("LOAD_MODE") or "") in ("none", "mlock", "dio"):
        errors.append(f"LAZY_MODE=on reads rows from disk on demand and requires mmap; "
                      f"LOAD_MODE={v.get('LOAD_MODE')} disables it, so the whole table would be "
                      "loaded resident instead")
    if str(v.get("SLOTS")) == "off":
        warnings.append("SLOTS=off hides /slots: the panel's live t/s and busy state go blank.")
    host = str(v.get("HOST") or "127.0.0.1")
    if host in ("0.0.0.0", "::") and not api_key:
        warnings.append(f"HOST={host} with no API_KEY: anyone on the LAN can use this instance.")

    # ---- flags the selected build does not know
    if bindir and check_runtime:
        known = build_flags(bindir)
        if known:
            unknown = sorted({a.split("=", 1)[0] for a in argv[1:]
                              if a.startswith("-") and not re.match(r"^-\d", a)} - known)
            if unknown:
                errors.append(f"the {backend} build does not accept: {', '.join(unknown)}")

    # ---- collisions with other instances, and host RAM across all of them
    port = str(v.get("PORT"))
    for other in instance_ids():
        if other == inst["id"]:
            continue
        try:
            op = _read_env_file(get_instance(other)["dir"] / "params.env").get("PORT") \
                or (str(DEFAULTS["PORT"]) if other == "main" else "")
        except ValueError:
            continue
        if str(op) == port:
            with using_instance(get_instance(other)):
                other_pid = server_pid()
            if other_pid:
                errors.append(f"PORT {port} is in use by running instance '{other}' "
                              f"(pid {other_pid}) - stop it first")
            else:
                warnings.append(f"PORT {port} is shared with instance '{other}'; only one of "
                                "them can run at a time")
    if check_runtime and not server_pid():
        busy = sh(f"ss -Hltn 'sport = :{int(port)}' 2>/dev/null") if port.isdigit() else ""
        if busy and not busy.startswith("<error"):
            errors.append(f"port {port} is already in use by another process")
    rb = ram_budget(v, include_others=True)
    if rb["verdict"] == "impossible":
        errors.append("host RAM: " + rb["detail"])
    elif rb["verdict"] == "risk":
        warnings.append("host RAM: " + rb["detail"])

    for key in ("SPEC_DRAFT_DEVICE", "MMPROJ_DEVICE", "MAIN_GPU", "TENSOR_SPLIT", "SPLIT_MODE",
                "OVERRIDE_TENSORS"):
        val = str(v.get(key) or "")
        named = set(re.findall(r"Vulkan\d+", val))
        known = {n for n, _ in vk_map} if vk_map else ({"Vulkan0"} if backend == "vulkan" else set())
        if named - known and backend == "vulkan":
            errors.append(f"{key} names {', '.join(sorted(named - known))}, which is not one of "
                          f"this instance's devices ({', '.join(sorted(known)) or 'none'})")
    if str(v.get("MMPROJ_DEVICE") or "") not in ("", "none") and use_mm \
            and not _truthy(v.get("MMPROJ_OFFLOAD")):
        warnings.append("MMPROJ_DEVICE is set but MMPROJ_OFFLOAD is off, so the projector stays on "
                        "the CPU and the device choice is ignored")
    return dict(instance=inst["id"], backend=backend, device=dev, bindir=bindir,
                devices=devs, vulkan_map=[dict(name=n, pci=d["pci"], device=d.get("name"))
                                          for n, d in vk_map],
                argv=argv, env=env, errors=errors, warnings=warnings,
                rundir=str(rundir), template_src=tsrc, template_copy=tcopy,
                api_key=api_key, ram=rb)


def _render_flags(v, table):
    out = []
    for row in table:
        key, kind, flag = row[0], row[1], row[2]
        val = str(v.get(key, "") if v.get(key) is not None else "")
        if val == "":
            continue
        if kind == "val":
            out += [flag, val]
        elif kind == "tri":
            if val == "on":
                out.append(flag)
            elif val == "off":
                out.append(row[3])
        elif kind == "switch" and val == "on":
            out.append(flag)
    return out


_legacy_keys_cache = {}


def legacy_ignored_keys():
    """Saved keys the legacy bash launchers (main) never read."""
    texts = []
    for f in LLAMA.glob("run_qwen38_*_inf01.sh"):
        try:
            texts.append((f.stat().st_mtime, f.read_text(errors="ignore")))
        except OSError:
            pass
    key = tuple(t[0] for t in texts)
    if key not in _legacy_keys_cache:
        body = "\n".join(t[1] for t in texts)
        _legacy_keys_cache.clear()
        _legacy_keys_cache[key] = sorted(
            k for k in DEFAULTS
            if not re.search(r"(\$\{?|\b_v in |\s)" + re.escape(k) + r"\b", body))
    return _legacy_keys_cache[key]


# ============================================================================
# INSTANCE CONTROL
# ============================================================================
USER_UNIT_TEXT = """[Unit]
Description=inf01 llama.cpp inference instance %i
Documentation=file:///home/smbadmin/AI-INSTRUCTIONS.md
# Same throttle as inf01-llama: a crash loop on a big model is its own outage.
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory=/home/smbadmin/llama
ExecStart=/usr/bin/python3 /home/smbadmin/panel/instance_launch.py %i
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=45
Restart=on-failure
RestartSec=20
# 78 = the launch plan refused (bad config). Retrying cannot fix that.
RestartPreventExitStatus=78

[Install]
WantedBy=default.target
"""


def _user_env():
    rt = f"/run/user/{os.getuid()}"
    return dict(os.environ, XDG_RUNTIME_DIR=rt, DBUS_SESSION_BUS_ADDRESS=f"unix:path={rt}/bus")


def user_manager():
    """Is there a systemd --user manager the panel can talk to, and will it
    survive logout / come up at boot (linger)?"""
    bus = os.path.exists(f"/run/user/{os.getuid()}/bus")
    linger = os.path.exists(f"/var/lib/systemd/linger/{os.environ.get('USER') or 'smbadmin'}") \
        or os.path.exists("/var/lib/systemd/linger/smbadmin")
    return dict(bus=bus, linger=linger,
                fix=None if linger else "sudo loginctl enable-linger smbadmin")


def ensure_user_unit():
    f = USER_UNIT_DIR / USER_UNIT
    if f.exists() and f.read_text() == USER_UNIT_TEXT:
        return False
    USER_UNIT_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(USER_UNIT_TEXT)
    if user_manager()["bus"]:
        subprocess.run(["systemctl", "--user", "daemon-reload"], env=_user_env(),
                       capture_output=True, timeout=30)
    return True


def _instance_record(iid, with_plan=False):
    inst = get_instance(iid)
    with using_instance(inst):
        p = load_params()
        pid = server_pid()
        dev = _device_record(inst["device"]) if inst["device"] != "cpu" else None
        # Detect running model from process, fall back to configured model
        running_model = get_running_model(pid) if pid else None
        model_path = running_model or p.get("MODEL") or p.get("SD_DIFFUSION_MODEL") or p.get("SD_MODEL")
        rec = dict(id=inst["id"], name=inst["name"], legacy=inst["legacy"],
                   devices=inst.get("devices") or [inst["device"]],
                   device_names=[(_device_record(p) or {}).get("name") or p
                                 for p in (inst.get("devices") or [inst["device"]])],
                   device=inst["device"], device_name=(dev or {}).get("name") or
                   ("CPU only" if inst["device"] == "cpu" else inst["device"]),
                   device_driver=(dev or {}).get("driver"),
                   backend=p.get("BACKEND"), port=p.get("PORT"), host=p.get("HOST"),
                   model=model_path, model_name=Path(str(model_path or "")).name,
                   running=pid is not None, pid=pid, unit=inst["unit"], scope=inst["scope"],
                   unit_state=unit_state(), live_tps=_live.get("tps"),
                   generating=bool(_live.get("processing")),
                   engine=inst.get("engine") or "llama.cpp")
        rec.update(_instance_problems(inst, rec, p))
        if with_plan:
            rec["plan"] = launch_plan(p)
    return rec


def _saved_device_names(inst):
    """Names recorded when the devices were picked - the only name left for a
    card that has since been pulled from the machine."""
    if inst["legacy"]:
        try:
            meta = json.loads(MAIN_DEVICES_FILE.read_text())
            return dict(zip(meta.get("devices") or [], meta.get("names") or []))
        except (OSError, ValueError):
            return {}
    try:
        meta = json.loads((inst["dir"] / "instance.json").read_text())
        return dict(zip(meta.get("devices") or [], meta.get("device_names") or []))
    except (OSError, ValueError):
        return {}


def _instance_problems(inst, rec, p):
    """Things that stop an instance starting, or make two of them collide.

    Added 2026-09-23 after main failed 18 starts in a row (its device list
    still named an RTX 2060 that had been removed) while the Diagnostics
    checks and the instance card all read green.
    """
    present = _present_gpus() | {"cpu"}
    saved = _saved_device_names(inst)
    missing = [pci for pci in rec["devices"] if pci not in present]
    names = []
    for pci, nm in zip(rec["devices"], rec["device_names"]):
        names.append(saved.get(pci) or nm if pci in missing else nm)
    warnings = []
    for pci in missing:
        warnings.append(f"device {pci} ({_short_gpu(saved.get(pci)) or 'unknown card'}) is not "
                        "in this machine; the instance cannot start until it is back or is "
                        "removed from the device list")
    port = str(rec.get("port") or "")
    clash = []
    for other in instance_ids():
        if other == inst["id"]:
            continue
        try:
            o = get_instance(other)
            op = str(_read_env_file(o["dir"] / "params.env").get("PORT") or
                     (DEFAULTS["PORT"] if o["legacy"] else ""))
        except (ValueError, OSError):
            continue
        if port and op == port:
            clash.append(other)
    if clash:
        mine_boot = (rec.get("unit_state") or {}).get("enabled") == "enabled"
        boot = []
        for other in clash:
            try:
                us = unit_state(get_instance(other)) or {}
            except ValueError:
                us = {}
            if us.get("enabled") == "enabled":
                boot.append(other)
        warnings.append(f"port {port} is also {', '.join(clash)}'s port"
                        + (f"; this one and {', '.join(boot)} both start at boot, and "
                           "whichever binds first wins" if mine_boot and boot else
                           "; only one of them can run at a time"))
    fails = 0
    try:
        with using_instance(inst):
            fails = fail_count()
    except Exception:
        pass
    us = rec.get("unit_state") or {}
    last_error = None
    if not rec["running"] and (us.get("active") == "failed" or fails):
        last_error = last_start_error(inst)
    return dict(device_names=names, missing_devices=missing, port_conflicts=clash,
                warnings=warnings, launch_fails=fails, last_start_error=last_error)


def _present_gpus():
    """PCI addresses of display-class devices on the bus. sysfs only: cheap
    enough for every status poll (gpu_devices() may shell out to nvidia-smi)."""
    out = set()
    try:
        for d in Path("/sys/bus/pci/devices").iterdir():
            try:
                if (d / "class").read_text().startswith("0x03"):
                    out.add(d.name)
            except OSError:
                pass
    except OSError:
        pass
    return out


def _short_gpu(name):
    n = str(name or "")
    return "7900 XTX" if "Navi 31" in n else ("RTX 2060" if "TU106" in n else n or None)


def last_start_error(inst=None):
    """The reason the last start was refused or died, from the launch output.

    main's script logs to the journal of inf01-llama (readable here); instances
    log to their launch .out file. Only lines the launchers mark as fatal.
    """
    inst = inst or INST()
    c = _start_err_cache.get(inst["id"])
    if c and time.time() - c[0] < 15:           # /api/status polls every 5 s
        return c[1]
    err = _last_start_error(inst)
    _start_err_cache[inst["id"]] = (time.time(), err)
    return err


_start_err_cache = {}


def _last_start_error(inst):
    if inst["legacy"]:
        txt = sh(f"journalctl -u {UNIT} -n 120 --no-pager -o cat 2>/dev/null", timeout=10)
    else:
        txt = tail(inst["launch_out"], 20000)
    lines = [l.strip() for l in (txt or "").splitlines()]
    idx = [i for i, l in enumerate(lines)
           if l.startswith(("[ABORT]", "[inf01-inst] ABORT", "ABORT")) or "[ABORT]" in l]
    if not idx:
        return None
    i = idx[-1]
    block = [lines[i]]
    for l in lines[i + 1:i + 3]:                  # the launchers put the fix on the next lines
        if l and not l.startswith(("[", "Started", "Stopped")) and "systemd" not in l \
                and ".service:" not in l:
            block.append(l)
    # and the reasons printed just above it
    for l in reversed(lines[max(0, i - 4):i]):
        if l and ("not present" in l or "cannot" in l or "refus" in l):
            block.insert(0, l)
    return " ".join(block)[:600]


def list_instances():
    out = []
    for iid in instance_ids():
        try:
            out.append(_instance_record(iid))
        except Exception as e:
            out.append(dict(id=iid, error=str(e)))
    return out


def _next_free_port():
    used = set()
    for iid in instance_ids():
        try:
            used.add(str(_read_env_file(get_instance(iid)["dir"] / "params.env").get("PORT")
                         or DEFAULTS["PORT"]))
        except ValueError:
            pass
    for port in range(8083, 8190):
        if port in (8090, 8091, 8092):          # panel, ttyd, shell
            continue
        if str(port) not in used and not sh(f"ss -Hltn 'sport = :{port}' 2>/dev/null"):
            return port
    raise ValueError("no free port between 8083 and 8189")


def create_instance(body):
    name = str(body.get("name") or "").strip()
    iid = str(body.get("id") or re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-"))[:24]
    engine = str(body.get("engine") or "llama.cpp")
    if engine not in ("llama.cpp", *GEN_ENGINES):
        raise ValueError(f"unknown engine {engine!r}")
    if not _INST_ID.match(iid or ""):
        raise ValueError("instance id: lowercase letters, digits and '-', max 24")
    if iid == "main" or (INSTANCES_DIR / iid).exists():
        raise ValueError(f"instance {iid!r} already exists")
    devices = [str(x) for x in (body.get("devices") or [body.get("device")]) if x]
    if not devices:
        raise ValueError("pick at least one device")
    alld = {d["pci"]: d for d in gpu_devices()}
    for pci in devices:
        if pci not in alld or not alld[pci]["usable"]:
            raise ValueError(f"device {pci!r} is not usable for inference")
    if len(set(devices)) != len(devices):
        raise ValueError("the same device is listed twice")
    if "cpu" in devices and len(devices) > 1:
        raise ValueError("CPU-only cannot be combined with GPUs; the CPU is always available")
    device, dev = devices[0], alld[devices[0]]
    backend = str(body.get("backend") or ("vulkan" if len(devices) > 1 else dev["backends"][0]))
    for pci in devices:
        if backend not in alld[pci]["backends"]:
            raise ValueError(f"{backend} cannot drive {alld[pci]['name']}")
    if len(devices) > 1 and backend != "vulkan":
        raise ValueError("multi-device instances are Vulkan-only for now")
    if engine == "sd.cpp":
        return _create_sd_instance(iid, name, devices, alld, backend, body)
    if engine == "audio.cpp":
        return _create_ac_instance(iid, name, devices, alld, backend, body)
    if engine == "camelid":
        return _create_cm_instance(iid, name, devices, alld, backend, body)
    if engine == "onnx":
        return _create_ox_instance(iid, name, devices, alld, body)

    # Start from a copy of another instance or a profile when asked; otherwise
    # from DEFAULTS with the main model's specifics cleared - its 245k context,
    # 11 GB prompt cache and Qwen template are wrong for a small card.
    # A copy of an instance or instance profile brings every backend file and
    # tier with it; a model profile is one parameter set, so only that.
    src = str(body.get("copy_from") or "")
    per_backend, tiers = {}, {}
    if src.startswith("profile:"):
        vals = resolve_source(src)
    elif src.startswith("iprofile:"):
        ip = read_instance_profile(src.split(":", 1)[1])
        per_backend, tiers = ip["params"], ip["tiers"]
        vals = dict(per_backend[ip["meta"]["backend"]])
    elif src:
        with using_instance(src.split(":", 1)[1] if src.startswith("instance:") else src):
            vals = load_params()
            per_backend = {b: load_params(b) for b in BACKENDS if _backend_file(b).exists()}
            tiers = {t: read_tier(t) for t in TIER_NAMES}
    else:
        vals = dict(DEFAULTS)
        vals.update(MODEL="", USE_MMPROJ=0, MMPROJ="", TEMPLATE_SRC="", TEMPLATE_SHA256="",
                    SPEC_TYPE="", CTX=16384, CACHE_RAM=1024, KV_TYPE="q8_0",
                    RAM_FLOOR_MB=2048, N_PREDICT=4096, REASONING_EFFORT="medium")
    if per_backend.get(backend):
        vals = dict(per_backend[backend])
    vals.update(BACKEND=backend, PORT=int(body.get("port") or _next_free_port()),
                HOST=str(body.get("host") or "127.0.0.1"))
    _reject_full_size_draft(vals)
    if backend == "cpu":
        vals["NGL"] = 0

    d = INSTANCES_DIR / iid
    d.mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps(dict(
        name=name or iid, device=device, devices=devices, engine=engine,
        device_names=[alld[p]["name"] for p in devices],
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        indent=1))
    with using_instance(iid):
        for b in BACKENDS:
            base = per_backend.get(b) if b != backend else None
            _write_env_file(_backend_file(b), _keep_local(dict(base or vals, BACKEND=b), vals))
        for t, tv in tiers.items():
            if tv is not None:
                _write_env_file(tier_file(t), _keep_local(tv, vals), header=[
                    f"# Tier '{t}' copied from {src} at instance creation.", ""])
        _write_env_file(PARAMS_ENV, vals, header=[
            "# GENERATED by the admin panel - do not hand-edit.",
            f"# Instance '{iid}'. Copy of params-{backend}.env, the active backend.",
            "# Read by panel/instance_launch.py.", ""])
        FAIL_FILE.write_text("0\n")
    ensure_user_unit()
    return _instance_record(iid)


def _create_sd_instance(iid, name, devices, alld, backend, body):
    """A stable-diffusion.cpp instance: its own SD_* parameter set, no tiers."""
    if backend not in sdcpp.BACKENDS:
        raise ValueError(f"stable-diffusion.cpp has no {backend} build; use vulkan, rocm or cpu")
    src = str(body.get("copy_from") or "")
    if src.startswith("instance:") or (src and not src.startswith(("profile:", "iprofile:"))):
        other = get_instance(src.split(":", 1)[-1])
        if other.get("engine") != "sd.cpp":
            raise ValueError("copy from a stable-diffusion.cpp instance, not a llama.cpp one")
        vals = sdcpp.load_params(other)
    elif src:
        raise ValueError("profiles hold llama.cpp settings; start a stable-diffusion.cpp "
                         "instance from a preset or another stable-diffusion.cpp instance")
    else:
        vals = dict(sdcpp.DEFAULTS)
    vals.update(BACKEND=backend, PORT=int(body.get("port") or _next_free_port()),
                HOST=str(body.get("host") or "127.0.0.1"))
    d = INSTANCES_DIR / iid
    d.mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps(dict(
        name=name or iid, device=devices[0], devices=devices, engine="sd.cpp",
        device_names=[alld[p]["name"] for p in devices],
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    inst = get_instance(iid)
    if body.get("preset"):
        vram = alld[devices[0]].get("vram_total_mib") if devices[0] != "cpu" else None
        vals, _st = sdcpp.apply_preset(str(body["preset"]), vals, vram)
    sdcpp.write_params(inst, vals, note=f"created from preset {body.get('preset')}"
                       if body.get("preset") else "")
    (d / ".launch-fails").write_text("0\n")
    ensure_user_unit()
    return _instance_record(iid)


def _create_ac_instance(iid, name, devices, alld, backend, body):
    """An audio.cpp instance: AC_* parameters plus its own model list."""
    if backend not in audiocpp.BACKENDS:
        raise ValueError(f"audio.cpp has no {backend} build; use vulkan or cpu")
    if len(devices) > 1:
        raise ValueError("audio.cpp drives one device per server; pick one device")
    src = str(body.get("copy_from") or "")
    models = []
    if src.startswith("instance:") or (src and not src.startswith(("profile:", "iprofile:"))):
        other = get_instance(src.split(":", 1)[-1])
        if other.get("engine") != "audio.cpp":
            raise ValueError("copy from an audio.cpp instance")
        vals, models = audiocpp.load_params(other), audiocpp.load_models(other)
    elif src:
        raise ValueError("profiles hold llama.cpp settings; start an audio.cpp instance fresh "
                         "or from another audio.cpp instance")
    else:
        vals = dict(audiocpp.DEFAULTS)
    vals.update(BACKEND=backend, PORT=int(body.get("port") or _next_free_port()),
                HOST=str(body.get("host") or "127.0.0.1"))
    d = INSTANCES_DIR / iid
    d.mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps(dict(
        name=name or iid, device=devices[0], devices=devices, engine="audio.cpp",
        device_names=[alld[p]["name"] for p in devices],
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    inst = get_instance(iid)
    audiocpp.write_params(inst, vals)
    if models:
        audiocpp.save_models(inst, models)
    (d / ".launch-fails").write_text("0\n")
    ensure_user_unit()
    return _instance_record(iid)


def _create_ox_instance(iid, name, devices, alld, body):
    """An ONNX Runtime GenAI instance: OX_* parameters. The execution provider (CPU, an NPU, a GPU)
    is a parameter, so the device picked here is only a label; one per server."""
    if len(devices) > 1:
        raise ValueError("an ONNX Runtime server drives one device; pick one")
    src = str(body.get("copy_from") or "")
    if src:
        other = get_instance(src.split(":", 1)[-1])
        if other.get("engine") != "onnx":
            raise ValueError("copy from an ONNX Runtime instance, or start fresh")
        vals = onnxrt.load_params(other)
    else:
        vals = dict(onnxrt.DEFAULTS)
    vals.update(PORT=int(body.get("port") or _next_free_port()), HOST=str(body.get("host") or "127.0.0.1"))
    d = INSTANCES_DIR / iid
    d.mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps(dict(
        name=name or iid, device=devices[0], devices=devices, engine="onnx",
        device_names=[alld[p]["name"] for p in devices],
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    onnxrt.write_params(get_instance(iid), vals)
    (d / ".launch-fails").write_text("0\n")
    ensure_user_unit()
    return _instance_record(iid)


def _create_cm_instance(iid, name, devices, alld, backend, body):
    """A Camelid instance: CM_* parameters. CPU or one NVIDIA card (no AMD)."""
    if backend not in camelid.BACKENDS:
        raise ValueError("Camelid runs on cpu or cuda (NVIDIA); it has no Vulkan or ROCm")
    if len(devices) > 1:
        raise ValueError("Camelid drives one device per server; pick one")
    if devices[0] != "cpu" and alld[devices[0]].get("vendor") != "nvidia":
        raise ValueError("Camelid cannot use AMD or Intel GPUs; pick CPU (or an NVIDIA card)")
    src = str(body.get("copy_from") or "")
    if src.startswith("instance:") or (src and not src.startswith(("profile:", "iprofile:"))):
        other = get_instance(src.split(":", 1)[-1])
        if other.get("engine") != "camelid":
            raise ValueError("copy from a Camelid instance")
        vals = camelid.load_params(other)
    elif src:
        raise ValueError("profiles hold llama.cpp settings; start a Camelid instance fresh")
    else:
        vals = dict(camelid.DEFAULTS)
    vals.update(BACKEND=backend, PORT=int(body.get("port") or _next_free_port()),
                HOST=str(body.get("host") or "127.0.0.1"))
    d = INSTANCES_DIR / iid
    d.mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps(dict(
        name=name or iid, device=devices[0], devices=devices, engine="camelid",
        device_names=[alld[p]["name"] for p in devices],
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    camelid.write_params(get_instance(iid), vals)
    (d / ".launch-fails").write_text("0\n")
    ensure_user_unit()
    return _instance_record(iid)


def update_instance(iid, body):
    inst = get_instance(iid)
    if inst["legacy"]:
        other = sorted(set(body) - {"id", "devices", "device"})
        if other:
            raise ValueError("main is defined by the inf01-llama unit and its scripts; only its "
                             f"devices and parameters are editable (not {', '.join(other)})")
        meta = {}
    else:
        meta = json.loads((inst["dir"] / "instance.json").read_text())
    if "name" in body:
        meta["name"] = str(body["name"]).strip() or iid
    want = body.get("devices") or ([body["device"]] if body.get("device") else None)
    if want and list(want) != inst["devices"]:
        with using_instance(inst):
            if server_pid():
                raise ValueError("stop the instance before changing its devices")
            b = load_params().get("BACKEND")
        alld = {d["pci"]: d for d in gpu_devices()}
        if len(set(want)) != len(want):
            raise ValueError("the same device is listed twice")
        for pci in want:
            dev = alld.get(pci)
            if not dev or not dev["usable"]:
                raise ValueError(f"device {pci!r} is not usable")
            if b not in dev["backends"]:
                raise ValueError(f"current BACKEND={b} cannot drive {dev['name']} - "
                                 f"switch backend first ({', '.join(dev['backends'])})")
        if len(want) > 1 and b != "vulkan":
            raise ValueError("multi-device instances are Vulkan-only for now")
        if "cpu" in want and len(want) > 1:
            raise ValueError("CPU-only cannot be combined with GPUs; the CPU is always available")
        if inst["legacy"]:
            if "cpu" in want:
                raise ValueError("main runs CPU-only through BACKEND=cpu, not a device choice")
            with using_instance(inst):
                venv, _m, verr = vulkan_pinning([alld[p] for p in want], active_build("vulkan"))
            if b == "vulkan" and verr:
                raise ValueError("; ".join(verr))
        meta["device"], meta["devices"] = want[0], list(want)
        # kept so a card that is later pulled can still be named in warnings
        meta["device_names"] = [alld[p]["name"] for p in want]
    if inst["legacy"]:
        if meta:
            MAIN_DEVICES_FILE.write_text(json.dumps(dict(
                devices=meta["devices"],
                names=[alld[p]["name"] for p in meta["devices"]],
                saved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1) + "\n")
        return _instance_record(iid)
    (inst["dir"] / "instance.json").write_text(json.dumps(meta, indent=1))
    if "autostart" in body:
        ok, msg = _systemctl("enable" if body["autostart"] else "disable", inst)
        if not ok:
            raise ValueError(msg)
    return _instance_record(iid)


def delete_instance(iid, confirm):
    inst = get_instance(iid)
    if inst["legacy"]:
        raise ValueError("main cannot be deleted from the panel")
    if confirm != iid:
        raise ValueError("confirm must equal the instance id")
    with using_instance(inst):
        if server_pid():
            raise ValueError("stop the instance first")
    if user_manager()["bus"]:
        subprocess.run(["systemctl", "--user", "disable", inst["unit"]], env=_user_env(),
                       capture_output=True, timeout=30)
    import shutil
    shutil.rmtree(inst["dir"])
    _pid_cache.pop(iid, None)
    _samples_by.pop(iid, None)
    _live_by.pop(iid, None)
    return dict(ok=True, deleted=iid)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
# Endpoints that model llama-server (KV cache, prompt cache, tiers, speculative
# decoding...). On a stable-diffusion.cpp instance they answer na=True instead
# of a llama estimate for a process that is not llama.
_LLAMA_ONLY_GET = {"/api/estimate", "/api/ram-budget", "/api/tiers", "/api/speed-curve",
                   "/api/curve", "/api/stats", "/api/profiles", "/api/workload", "/api/hermes"}
_LLAMA_ONLY_POST = {"/api/estimate", "/api/ram-budget", "/api/tier/load", "/api/tier/save",
                    "/api/profile/save", "/api/profile/load", "/api/profile/new",
                    "/api/tier/copy", "/api/tier/delete", "/api/optimize/start",
                    "/api/curve/start", "/api/instance-profile/save",
                    "/api/instance-profile/apply", "/api/workload/settings",
                    "/api/workload/experiment", "/api/workload/stop", "/api/workload/proposal/apply",
                    "/api/workload/proposal/dismiss", "/api/workload/rollback"}


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "inf01-panel"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = (json.dumps(obj, indent=2, default=str).encode()
                if ctype == "application/json" else obj.encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, ctype="application/octet-stream", filename=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _disposition(self, name):
        ascii_name = re.sub(r'[^A-Za-z0-9._ ()-]', "_", name) or "download"
        return (f'attachment; filename="{ascii_name}"; '
                f"filename*=UTF-8''{urllib.parse.quote(name)}")

    def _send_file(self, path):
        """Stream a file (Files tab downloads can be many GB)."""
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", self._disposition(path.name))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def _send_zip(self, rels):
        """Stream a zip of files/folders. No Content-Length: the connection
        closes at the end (HTTP/1.0), which is how the browser knows it's done."""
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", self._disposition(filemgr.zip_name(rels)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        filemgr.stream_zip(rels, self.wfile)

    # Browsers attach the Caddy basic-auth login to requests from ANY site, so
    # without this a page elsewhere could drive the whole API (2026-09-23
    # audit, confirmed). The UI only talks to its own origin. Refuse what the
    # browser marks as coming from another site, a foreign Origin when the
    # browser sends no Sec-Fetch-Site, and on POST the form/text content types
    # that browsers send cross-site without a CORS preflight. curl and scripts
    # send none of these headers and are unaffected (POST with JSON).
    _SIMPLE_CT = ("application/x-www-form-urlencoded", "multipart/form-data", "text/plain")

    def _foreign(self, method):
        h = self.headers
        site = (h.get("Sec-Fetch-Site") or "").lower()
        if site in ("cross-site", "same-site"):
            return f"refused: request from another site (Sec-Fetch-Site: {site})"
        origin = h.get("Origin")
        if not site and origin:
            if origin == "null" or urllib.parse.urlsplit(origin).netloc != (h.get("Host") or ""):
                return f"refused: request from origin {origin}"
        if method == "POST":
            ct = (h.get("Content-Type") or "").split(";")[0].strip().lower()
            if ct in self._SIMPLE_CT:
                return f"refused: Content-Type {ct}; send application/json"
        return None

    def _bind_instance(self):
        q = urllib.parse.parse_qs(self.path.partition("?")[2])
        _ctx.inst = get_instance(q.get("inst", ["main"])[0] or "main")

    def _authorize(self, method, p):
        """Multi-user mode: who is asking and may they (auth.py). Every POST is audited."""
        code, user, role = auth.check(method, p, self.headers)
        self.who = (user, role)
        mcp_server._caller.who = user
        if method == "POST" and p not in auth.OPEN:
            q = urllib.parse.parse_qs(self.path.partition("?")[2])
            auth.audit(user or "-", role or "-", method, p, (q.get("inst") or ["main"])[0],
                       "allowed" if code == 0 else "denied")
        if code == 401:
            b = b'{"error": "log in (multi-user mode)"}'
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="LexiPanel"')
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return False
        if code == 403:
            self._send(dict(error=f"your role ({role}) cannot do this; it needs {auth.needed(method, p)}"), 403)
            return False
        return True

    def do_GET(self):
        p = self.path.split("?")[0]
        if not self._authorize("GET", p):
            return
        if p == "/v1/models":
            return self._send(gateway.models())
        why = self._foreign("GET") if p.startswith("/api/") else None
        if why:
            return self._send(dict(error=why), 403)
        try:
            self._bind_instance()
        except ValueError as e:
            return self._send(dict(error=str(e)), 404)
        try:
            if p in ("/", "/index.html"):
                return self._send((PANEL / "static/index.html").read_text(),
                                  ctype="text/html; charset=utf-8")
            if INST().get("engine") in GEN_ENGINES and p in _LLAMA_ONLY_GET:
                return self._send(dict(na=True, error=f"not used by {INST()['engine']} "
                                       "instances", engine=INST()["engine"]))
            if p == "/api/flags":
                try:
                    return self._send(flagcatalog.catalog(INST()))
                except Exception as e:
                    return self._send(dict(error=f"could not read the build's --help: {e}",
                                           sections=[]), 500)
            if p == "/api/sd/presets":
                vals = load_params() if INST().get("engine") == "sd.cpp" else {}
                return self._send(dict(presets=[sdcpp.preset_status(n, vals) for n in sdcpp.PRESETS],
                                       files=sdcpp.model_files()))
            if p == "/api/sd/job":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    return self._send(sdcpp.job(q.get("id", [""])[0]))
                except ValueError as e:
                    return self._send(dict(error=str(e)), 404)
            if p.startswith("/api/cm/"):
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    if p == "/api/cm/catalog":
                        return self._send(camelid.catalog(force=q.get("refresh", ["0"])[0] == "1"))
                    if p == "/api/cm/local":
                        v = camelid.load_params(INST()) if INST().get("engine") == "camelid" else None
                        return self._send(dict(models=camelid.local_models(v),
                                               models_dir=str(camelid.models_dir(v))))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/files/"):
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    if p == "/api/files/list":
                        return self._send(filemgr.listing(q.get("path", [""])[0]))
                    if p == "/api/files/download":
                        f = filemgr.resolve(q.get("path", [""])[0])
                        if f.is_dir():
                            filemgr.check_zip([q.get("path", [""])[0]])
                            return self._send_zip([q.get("path", [""])[0]])
                        return self._send_file(f)
                    if p == "/api/files/zip":
                        paths = q.get("path", [])
                        filemgr.check_zip(paths)
                        return self._send_zip(paths)
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if p.startswith("/api/ac/"):
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    if p == "/api/ac/catalog":
                        return self._send(audiocpp.catalog())
                    if INST().get("engine") != "audio.cpp":
                        raise ValueError("select an audio.cpp instance first")
                    if p == "/api/ac/models":
                        return self._send(dict(models=audiocpp.load_models(INST())))
                    if p == "/api/ac/job":
                        return self._send(audiocpp.job(q.get("id", [""])[0]))
                    if p == "/api/ac/outputs":
                        return self._send(audiocpp.outputs(INST()["id"]))
                    if p == "/api/ac/voices":
                        return self._send(dict(voices=audiocpp.voices(INST(), q.get("model", [""])[0])))
                    if p == "/api/ac/file":
                        f = audiocpp.output_path(INST()["id"], q.get("file", [""])[0])
                        return self._send_bytes(f.read_bytes(),
                                                "audio/wav" if f.suffix == ".wav" else "text/plain; charset=utf-8",
                                                filename=f.name if q.get("dl") else None)
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p == "/api/sd/gallery":
                return self._send(sdcpp.gallery(INST()["id"]))
            if p == "/api/sd/image":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    f = sdcpp.image_path(INST()["id"], q.get("file", [""])[0])
                except ValueError as e:
                    return self._send(dict(error=str(e)), 404)
                self.send_response(200)
                data = f.read_bytes()
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
                return
            if p == "/api/status":
                pid = server_pid()
                s = list(_samples)[-1] if _samples else {}
                return self._send(dict(
                    running=pid is not None, pid=pid,
                    health=(GEN_ENGINES[INST()["engine"]].health(INST())
                            if INST().get("engine") in GEN_ENGINES
                            else api_get("/health")), params=load_params(),
                    engine=INST().get("engine") or "llama.cpp",
                    vram_mib=s.get("vram"), gtt_mib=s.get("gtt"),
                    mem_avail_mb=s.get("mem_avail"),
                    vram_total_mib=read_int(f"{CARD}/mem_info_vram_total") // 1048576,
                    live_tps=_live.get("tps"), generating=_live.get("processing"),
                    unit=unit_state(),
                    guard=main_guard(),
                    throughput=headline(),
                    live=live_config(),
                    servers=list_servers(),
                    instance=_instance_record(INST()["id"]),
                    gpus=instance_gpus(),
                    instances=list_instances(),
                    uptime=sh(f"ps -o etimes= -p {pid}").strip() if pid else None))
            if p == "/api/servers":
                return self._send(list_servers())
            if p == "/api/instances":
                return self._send(dict(instances=list_instances(), devices=gpu_devices(),
                                       user_manager=user_manager()))
            if p == "/api/gpu-power":
                return self._send(gpupower.status())
            if p == "/api/gpus":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                return self._send(host_gpus(q.get("which", ["instance"])[0]))
            if p == "/api/devices":
                return self._send(gpu_devices())
            if p == "/api/hf-token":
                return self._send(hf_token_status())
            if p == "/api/launch-plan":
                return self._send(launch_plan())
            if p == "/api/profiles":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                return self._send(list_profiles(q.get("model", [None])[0]))
            if p == "/api/instance-profiles":
                return self._send(list_instance_profiles())
            if p == "/api/optimize/status":
                return self._send(optimizer.status())
            if p == "/api/optimize/runs":
                return self._send(optimizer.list_runs())
            if p == "/api/optimize/run":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                return self._send(optimizer.read_run(q.get("id", [""])[0]))
            if p == "/api/optimize/suite":
                return self._send(optimizer.describe_suite())
            if p == "/api/refusals/status":
                return self._send(refusals.status(INST()["id"]))
            if p == "/api/webui":
                try:
                    return self._send(webui.status())
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/power"):
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    if p == "/api/power":
                        return self._send(poweropts.status())
                    if p == "/api/power/stability":
                        return self._send(poweropts.stability(force=q.get("force", [""])[0] == "1"))
                    if p == "/api/power/budget":
                        return self._send(dict(budget=poweropts.budget(), ups=poweropts.ups_status()))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/fit/"):
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    if p == "/api/fit/status":
                        return self._send(fitquant.status())
                    if p == "/api/fit/job":
                        return self._send(fitquant.job(q.get("id", [""])[0]))
                    if p == "/api/fit/plan":
                        return self._send(fitquant.plan(q.get("id", [""])[0]))
                    if p == "/api/fit/target":
                        return self._send(fitquant._public_target(fitquant.target(
                            q.get("instance", ["main"])[0])))
                    if p == "/api/fit/recipes":
                        return self._send(dict(recipes=fitrecipe.list_recipes(),
                                               models=fitrecipe.explain_models(),
                                               base_menu=fitrecipe.BASE_MENU))
                    if p == "/api/fit/recipe":
                        rid = q.get("id", [""])[0]
                        return self._send(fitrecipe.export(rid) if q.get("export", [""])[0] == "1"
                                          else fitrecipe.get(rid))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/gpu-tune"):
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    if p == "/api/gpu-tune":
                        return self._send(gputune.status())
                    if p == "/api/gpu-tune/bench":
                        return self._send(gputune.bench_status())
                    if p == "/api/gpu-tune/vbios/file":
                        return self._send_file(gputune.vbios_file(q.get("name", [""])[0]))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p == "/api/auth":
                return self._send(auth.status(*self.who))
            if p == "/api/gateway":
                return self._send(gateway.status())
            if p == "/api/fleet":
                return self._send(fleet.status())
            if p == "/api/hermes":
                return self._send(hermes.setup(INST(), self.headers.get("Host")))
            if p == "/api/workload":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    days = int(q.get("days", [str(workload.WINDOW_DAYS)])[0])
                except ValueError:
                    days = workload.WINDOW_DAYS
                days = max(1, min(days, workload.RETAIN_DAYS))
                return self._send(dict(workload.status(INST(), days), autofit=autofit.status(INST())))
            if p == "/api/mcp/tools":
                return self._send(dict(tools=mcp_server.list_tools()))
            if p == "/api/mcp":
                return self._send(dict(error="MCP here is POST only (Streamable HTTP, JSON responses; "
                                             "no SSE stream)"), 405)
            if p == "/api/gpu":
                return self._send(gpu_info())
            if p == "/api/stats":
                return self._send(stats())
            if p == "/api/diagnostics":
                ck, cmdline, env, log = checks()
                return self._send(dict(checks=ck, cmdline=cmdline, env=env,
                                       launch_log=log,
                                       samples=list(_samples)[-180:],
                                       baseline=measured_baseline()))
            if p == "/api/debug-bundle":
                return self._send(debug_bundle())
            if p == "/api/backup":
                # Rebuild the config tarball on demand and hand it to the browser.
                # Excludes *.gguf and the llama.cpp binaries; everything else on
                # this box is hand-tuned and expensive to reproduce.
                r = subprocess.run([str(PANEL / "make-backup.sh")],
                                   capture_output=True, text=True, timeout=180)
                if r.returncode != 0:
                    return self._send(
                        dict(error=(r.stderr or "backup failed")[-500:]), 500)
                tb = Path(r.stdout.strip())
                if not tb.is_file():
                    return self._send(dict(error=f"no tarball at {tb}"), 500)
                return self._send_bytes(tb.read_bytes(), "application/gzip", tb.name)
            if p == "/api/disk":
                return self._send(disk_info())
            if p == "/api/models":
                return self._send(list_models())
            if p == "/api/templates":
                return self._send(list_templates())
            if p == "/api/params":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                b = q.get("backend", [None])[0]
                return self._send(load_params(b))
            if p == "/api/tiers":
                return self._send(tier_status())
            if p == "/api/param-meta" and INST().get("engine") == "onnx":
                return self._send(dict(meta=onnxrt.meta(), groups=onnxrt.GROUPS, defaults=onnxrt.DEFAULTS,
                                       drafts=[], projectors=[], engine="onnx",
                                       instance=dict(id=INST()["id"], legacy=False, device=None),
                                       group_backends={}, ignored=[]))
            if p == "/api/onnx/status":
                return self._send(onnxrt.status(INST()))
            if p == "/api/param-meta" and INST().get("engine") == "camelid":
                _dev = _device_record(INST()["device"]) if INST()["device"] != "cpu" else None
                m = camelid.meta()
                m["BACKEND"]["options"] = ["cpu"] + (["cuda"] if (_dev or {}).get("vendor") == "nvidia" else [])
                return self._send(dict(meta=m, groups=camelid.GROUPS, defaults=camelid.DEFAULTS,
                                       drafts=[], projectors=[], engine="camelid",
                                       instance=dict(id=INST()["id"], legacy=False, device=_dev),
                                       group_backends={}, ignored=[]))
            if p == "/api/param-meta" and INST().get("engine") == "audio.cpp":
                _dev = _device_record(INST()["device"]) if INST()["device"] != "cpu" else None
                m = audiocpp.meta()
                m["BACKEND"]["options"] = [b for b in audiocpp.BACKENDS
                                           if b in ((_dev or {}).get("backends") or ["cpu"])]
                return self._send(dict(meta=m, groups=audiocpp.GROUPS, defaults=audiocpp.DEFAULTS,
                                       drafts=[], projectors=[], engine="audio.cpp",
                                       instance=dict(id=INST()["id"], legacy=False, device=_dev),
                                       group_backends={}, ignored=[]))
            if p == "/api/param-meta" and INST().get("engine") == "sd.cpp":
                _dev = _device_record(INST()["device"]) if INST()["device"] != "cpu" else None
                m = sdcpp.meta()
                m["BACKEND"]["options"] = [b for b in sdcpp.BACKENDS
                                           if b in ((_dev or {}).get("backends") or ["cpu"])]
                return self._send(dict(meta=m, groups=sdcpp.GROUPS, defaults=sdcpp.DEFAULTS,
                                       drafts=[], projectors=[], engine="sd.cpp",
                                       instance=dict(id=INST()["id"], legacy=False, device=_dev),
                                       group_backends={}, ignored=[]))
            if p == "/api/param-meta":
                # SPEC_DRAFT_MODEL's options are whatever is on disk right now,
                # so a freshly downloaded sidecar appears without a restart.
                meta = {k: dict(v) for k, v in PARAM_META.items()}
                if "TEMPLATE_SHA256" not in meta and "TEMPLATE_SRC" in meta:
                    meta["TEMPLATE_SHA256"] = dict(
                        group=meta["TEMPLATE_SRC"]["group"], label="Template checksum",
                        type="text", tip="A fingerprint of the chat-template file. The launcher "
                        "refuses to start if the file on disk no longer matches it, so a template "
                        "that was edited or corrupted by accident is caught. It is filled in "
                        "automatically when you pick a template - you never need to type it.")
                # plain words first, the original technical note after
                for k, txt in flaghelp.PARAM_PLAIN.items():
                    if k in meta:
                        meta[k]["tip"] = (f"{txt}<br><br><span style='opacity:.75'>Technical: "
                                          f"{meta[k].get('tip') or ''}</span>")
                dopts = draft_options()
                meta["SPEC_DRAFT_MODEL"]["options"] = [o["value"] for o in dopts]
                meta["SPEC_DRAFT_MODEL"]["option_meta"] = dopts
                popts = projector_options()
                meta["MMPROJ"]["options"] = [o["value"] for o in popts]
                meta["MMPROJ"]["option_meta"] = popts
                topts = list_templates()
                meta["TEMPLATE_SRC"]["options"] = [t["path"] for t in topts]
                meta["TEMPLATE_SRC"]["option_meta"] = topts
                _dev = _device_record(INST()["device"]) if INST()["device"] != "cpu" else None
                meta["BACKEND"]["options"] = (_dev or {}).get("backends") or ["cpu"]
                return self._send(dict(meta=meta, groups=PARAM_GROUPS,
                                       defaults=DEFAULTS, drafts=dopts,
                                       projectors=popts,
                                       instance=dict(id=INST()["id"], legacy=INST()["legacy"],
                                                     device=_dev),
                                       group_backends=GROUP_BACKENDS,
                                       ignored=(legacy_ignored_keys() if INST()["legacy"]
                                                else [])))
            if p == "/api/estimate":
                # GET form estimates the CURRENT saved params.
                return self._send(estimate(load_params()))
            if p == "/api/ram-budget":
                return self._send(ram_budget(load_params(), include_others=True))
            if p == "/api/crash-report":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                return self._send(crash_report(force=q.get("force", ["0"])[0] == "1"))
            if p == "/api/builds":
                return self._send(dict(
                    builds=scan_builds(), backends=list(BACKENDS),
                    active={b: active_build(b) for b in BACKENDS},
                    running=live_arg("-m") and next(
                        (a for a in live_cmdline_args() if a.endswith("llama-server")), None)))
            if p == "/api/builds/releases":
                try:
                    return self._send(dict(releases=github_releases()))
                except Exception as e:
                    return self._send(dict(error=f"GitHub API unreachable: {e}",
                                           releases=[]), 502)
            if p == "/api/engines":
                return self._send(dict(engines=engines.describe_all()))
            if p == "/api/engines/releases":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    return self._send(dict(releases=engines.releases(q.get("engine", [""])[0])))
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
                except Exception as e:
                    return self._send(dict(error=f"GitHub API unreachable: {e}",
                                           releases=[]), 502)
            if p == "/api/curve":
                return self._send(depthcurve.status(INST()["id"]))
            if p == "/api/speed-curve":
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                b = (q.get("backend", ["vulkan"])[0] or "vulkan")
                return self._send(observed_curve(b))
            if p == "/api/downloads":
                with _lock:
                    return self._send(list(_downloads.values()))
            if p == "/api/logs":
                # Caller picks how far back to go. Default is 10x the old
                # fixed 20k; capped so a huge log cannot wedge the panel.
                q = urllib.parse.parse_qs(self.path.partition("?")[2])
                try:
                    n = int(q.get("n", ["200000"])[0])
                except ValueError:
                    n = 200000
                n = max(1000, min(n, 4_000_000))
                return self._send(dict(engine=tail(ENGINE_LOG, n),
                                       launch=tail(LAUNCH_OUT, n), n=n))
            return self._send(dict(error="not found"), 404)
        except ValueError as e:                      # a bad request, as on POST
            return self._send(dict(error=str(e)), 400)
        except Exception as e:
            return self._send(dict(error=str(e)), 500)

    def do_POST(self):
        p = self.path.split("?")[0]
        if not self._authorize("POST", p):
            return
        if p.startswith("/v1/"):
            return gateway.handle_post(self, p, self.who[0])
        why = self._foreign("POST")
        if why:
            return self._send(dict(error=why), 403)
        try:
            self._bind_instance()
        except ValueError as e:
            return self._send(dict(error=str(e)), 404)
        if p == "/api/files/upload":
            # raw body streamed to disk; never through the JSON path below
            q = urllib.parse.parse_qs(self.path.partition("?")[2])
            try:
                return self._send(filemgr.receive(
                    self.rfile, int(self.headers.get("Content-Length", -1)),
                    q.get("dir", [""])[0], q.get("name", [""])[0],
                    overwrite=q.get("overwrite", ["0"])[0] == "1"))
            except ValueError as e:
                self.close_connection = True       # an unread body may remain
                return self._send(dict(error=str(e)), 400)
        if p == "/api/mcp":
            try:
                msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
            except ValueError:
                return self._send(dict(jsonrpc="2.0", id=None,
                                       error=dict(code=-32700, message="parse error")), 400)
            resp = mcp_server.handle(msg)
            if resp is None:                   # notifications only: accepted, no body
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return self._send(resp)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        try:
            if INST().get("engine") in GEN_ENGINES and p in _LLAMA_ONLY_POST:
                return self._send(dict(na=True, error=f"not used by {INST()['engine']} "
                                       "instances", engine=INST()["engine"]))
            if p.startswith("/api/cm/"):
                try:
                    if p == "/api/cm/pull":
                        v = camelid.load_params(INST()) if INST().get("engine") == "camelid" else None
                        return self._send(camelid.pull(str(body.get("id") or ""), v))
                    if INST().get("engine") != "camelid":
                        raise ValueError("select a Camelid instance first")
                    if p == "/api/cm/chat":
                        return self._send(camelid.chat(INST(), body))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/files/"):
                try:
                    if p == "/api/files/mkdir":
                        return self._send(filemgr.mkdir(body.get("path", ""), body.get("name")))
                    if p == "/api/files/rename":
                        return self._send(filemgr.rename(body.get("path", ""), body.get("name")))
                    if p == "/api/files/delete":
                        return self._send(filemgr.delete(body.get("paths") or []))
                    if p == "/api/files/paste":
                        return self._send(filemgr.paste(str(body.get("op") or ""),
                                                        body.get("paths") or [],
                                                        body.get("dest", "")))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
                except OSError as e:
                    return self._send(dict(error=f"{e.strerror or e}"), 500)
            if p.startswith("/api/ac/"):
                try:
                    if p == "/api/ac/install":
                        return self._send(audiocpp.install_package(str(body.get("package") or "")))
                    if p == "/api/ac/uninstall":
                        return self._send(audiocpp.uninstall_package(str(body.get("package") or "")))
                    if p == "/api/ac/sizes":
                        return self._send(audiocpp.check_sizes(str(body.get("family") or "")))
                    if INST().get("engine") != "audio.cpp":
                        raise ValueError("select an audio.cpp instance first")
                    if p == "/api/ac/models":
                        return self._send(audiocpp.save_models(INST(), body.get("models")))
                    if p == "/api/ac/upload":
                        return self._send(audiocpp.upload(INST(), body))
                    if p == "/api/ac/run":
                        return self._send(audiocpp.run(INST(), body))
                    if p == "/api/ac/unload":
                        return self._send(audiocpp.unload_all(INST()))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/sd/"):
                try:
                    if INST().get("engine") != "sd.cpp" and p != "/api/sd/download":
                        raise ValueError("select a stable-diffusion.cpp instance first")
                    if p == "/api/sd/generate":
                        return self._send(sdcpp.generate(INST(), body))
                    if p == "/api/sd/cancel":
                        return self._send(sdcpp.cancel(str(body.get("id") or "")))
                    if p == "/api/sd/preset/apply":
                        dev = _device_record(INST()["device"]) if INST()["device"] != "cpu" else None
                        vals, st = sdcpp.apply_preset(str(body.get("name")), load_params(),
                                                      (dev or {}).get("vram_total_mib"))
                        sdcpp.save_params(INST(), vals)
                        return self._send(dict(ok=True, status=st, params=load_params()))
                    if p == "/api/sd/download":
                        return self._send(sdcpp.download_missing(
                            str(body.get("name")), bool(body.get("optional"))))
                    return self._send(dict(error="not found"), 404)
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/engines/") or p.startswith("/api/curve/"):
                try:
                    if p == "/api/engines/install":
                        return self._send(engines.install(body.get("engine"), body.get("tag"),
                                                          body.get("flavour")))
                    if p == "/api/engines/activate":
                        return self._send(engines.activate(body.get("engine"), body.get("flavour"),
                                                           body.get("bindir")))
                    if p == "/api/engines/delete":
                        return self._send(engines.delete(body.get("engine"), body.get("bindir")))
                    if p == "/api/engines/policy":
                        return self._send(dict(policy=engines.set_policy(body.get("engine"), body)))
                    if p == "/api/engines/check":
                        return self._send(engines.run_update_async(body.get("engine"), "manual"))
                    if p == "/api/curve/start":
                        return self._send(depthcurve.start(INST()["id"], body))
                    if p == "/api/curve/stop":
                        return self._send(depthcurve.stop())
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
                return self._send(dict(error="not found"), 404)
            if p == "/api/start":
                ok, msg = start_server()
                resp = dict(ok=ok, msg=msg)
                if INST()["legacy"]:
                    g = main_guard(fresh=True)
                    resp["guard"] = g
                    if not ok and g and g["locked"]:
                        resp["code"] = "start_limit"
                return self._send(resp)
            if p == "/api/main/reset-failed":
                try:
                    return self._send(reset_main_lockout())
                except ValueError as e:
                    return self._send(dict(error=str(e)), 400)
            if p == "/api/stop":
                ok, msg = stop_server()
                return self._send(dict(ok=ok, msg=msg))
            if p == "/api/restart":
                if not INST()["legacy"]:
                    if server_pid():
                        stop_server(); time.sleep(2)
                    ok, msg = start_server()
                    return self._send(dict(ok=ok, msg=msg))
                if unit_installed():
                    g = main_guard(fresh=True)
                    if g["locked"]:
                        return self._send(guard_refusal("start_limit", g))
                    if g["loading"] and not body.get("force"):
                        return self._send(guard_refusal("loading", g))
                    ok, msg = _systemctl("restart")
                    if not ok:
                        g = main_guard(fresh=True)
                        if g["locked"] or "too quickly" in msg or "start-limit" in msg:
                            return self._send(guard_refusal("start_limit", g, msg))
                    return self._send(dict(ok=ok, msg="restarting via systemd" if ok
                                           else f"systemctl failed: {msg}",
                                           guard=main_guard(fresh=True)))
                stop_server(); time.sleep(2)
                ok, msg = start_server()
                return self._send(dict(ok=ok, msg=msg))
            if p == "/api/params":
                return self._send(dict(ok=True, params=save_params(body)))
            if p == "/api/estimate":
                # POST form estimates a CANDIDATE set without saving it, so the
                # calculator updates live as the form is edited.
                return self._send(estimate(body))
            if p == "/api/tier/load":
                # Return a tier's values for the FORM. Does not save - you still
                # press Save, so a misclick cannot change what is running.
                t = str(body.get("tier") or "normal")
                v = read_tier(t)
                if v is None:
                    return self._send(dict(error=f"tier '{t}' has not been saved yet"), 404)
                return self._send(dict(ok=True, tier=t, params=v))
            if p == "/api/tier/save":
                t = str(body.get("tier") or "normal")
                return self._send(dict(ok=True, tier=t,
                                       params=write_tier(t, body.get("params") or {})))
            if p == "/api/profile/save":
                return self._send(save_profile(body.get("name"), body.get("params")))
            if p == "/api/profile/load":
                return self._send(load_profile(body.get("model"), body.get("name")))
            if p == "/api/profile/delete":
                return self._send(delete_profile(body.get("model"), body.get("name")))
            if p == "/api/profile/new":
                return self._send(new_profile(body.get("name"), body.get("source") or "current",
                                              body.get("model")))
            if p == "/api/profile/copy":
                return self._send(copy_profile(body.get("model"), body.get("name"),
                                               body.get("new_name"), body.get("new_model")))
            if p == "/api/profile/rename":
                return self._send(rename_profile(body.get("model"), body.get("name"),
                                                 body.get("new_name")))
            if p == "/api/tier/delete":
                return self._send(delete_tier(str(body.get("tier") or "")))
            if p == "/api/tier/copy":
                t = str(body.get("tier") or "")
                return self._send(dict(ok=True, tier=t,
                                       params=copy_into_tier(t, body.get("source"))))
            if p == "/api/instance-profile/save":
                return self._send(save_instance_profile(body.get("name"), body.get("instance"),
                                                        bool(body.get("overwrite"))))
            if p == "/api/instance-profile/copy":
                return self._send(copy_instance_profile(body.get("name"), body.get("new_name")))
            if p == "/api/instance-profile/rename":
                return self._send(rename_instance_profile(body.get("name"), body.get("new_name")))
            if p == "/api/instance-profile/delete":
                return self._send(delete_instance_profile(str(body.get("name") or ""),
                                                          str(body.get("confirm") or "")))
            if p == "/api/instance-profile/apply":
                return self._send(apply_instance_profile(body.get("name"), body.get("instance"),
                                                         body.get("parts") or ["params", "tiers"]))
            if p == "/api/optimize/start":
                return self._send(optimizer.start(INST()["id"], body))
            if p == "/api/optimize/stop":
                return self._send(optimizer.stop())
            if p == "/api/refusals/start":
                return self._send(refusals.start(str(body.get("instance") or INST()["id"]), body))
            if p == "/api/refusals/stop":
                return self._send(refusals.stop())
            if p.startswith("/api/webui/"):
                webui_post = {"/api/webui/defaults": webui.save_defaults,
                              "/api/webui/flags": webui.set_flags,
                              "/api/webui/caddy-snippet": webui.write_snippet}
                if p in webui_post:
                    return self._send(webui_post[p](body))
                return self._send(dict(error="not found"), 404)
            if p.startswith("/api/power/"):
                power_post = {
                    "/api/power/apply": poweropts.apply, "/api/power/persist": poweropts.persist,
                    "/api/power/unpersist": poweropts.unpersist,
                    "/api/power/profile/save": poweropts.save_profile,
                    "/api/power/profile/capture": poweropts.capture,
                    "/api/power/profile/delete": poweropts.delete_profile,
                    "/api/power/settings": poweropts.save_settings,
                    "/api/power/annotate": poweropts.annotate,
                }
                if p in power_post:
                    return self._send(power_post[p](body))
                if p == "/api/power/recheck":
                    poweropts._drop("helper", "view", "ups", "peaks")
                    return self._send(poweropts.helper_info())
                return self._send(dict(error="not found"), 404)
            if p.startswith("/api/gpu-tune/"):
                tune_post = {"/api/gpu-tune/apply": gputune.apply,
                             "/api/gpu-tune/bench/start": gputune.bench_start,
                             "/api/gpu-tune/bench/delete": gputune.bench_delete,
                             "/api/gpu-tune/vbios/backup": gputune.vbios_backup,
                             "/api/gpu-tune/rom-check": gputune.rom_check,
                             "/api/gpu-tune/profile/save": gputune.profile_save,
                             "/api/gpu-tune/profile/apply": gputune.profile_apply,
                             "/api/gpu-tune/profile/boot": gputune.profile_persist,
                             "/api/gpu-tune/profile/unboot": gputune.profile_unpersist,
                             "/api/gpu-tune/profile/delete": gputune.profile_delete}
                if p in tune_post:
                    return self._send(tune_post[p](body))
                if p == "/api/gpu-tune/bench/stop":
                    return self._send(gputune.bench_stop())
                return self._send(dict(error="not found"), 404)
            if p == "/api/gateway/quota":
                return self._send(gateway.set_quota(body))
            if p.startswith("/api/auth/"):
                if p == "/api/auth/mode":
                    return self._send(auth.set_mode(str(body.get("mode") or ""), body.get("admin_user"),
                                                    body.get("admin_password")))
                if p == "/api/auth/users":
                    return self._send(auth.set_user(body.get("user"), body.get("password") or None, body.get("role") or None))
                if p == "/api/auth/users/delete":
                    return self._send(auth.delete_user(str(body.get("user") or "")))
                if p == "/api/auth/keys/create":
                    return self._send(auth.create_key(str(body.get("user") or ""), str(body.get("role") or "viewer"),
                                                      int(body.get("days") or 90), body.get("label") or ""))
                if p == "/api/auth/keys/revoke":
                    return self._send(auth.revoke_key(str(body.get("id") or "")))
                return self._send(dict(error="not found"), 404)
            if p.startswith("/api/fleet/"):
                try:
                    if p == "/api/fleet/report":
                        return self._send(fleet.intake(self.headers, json.dumps(body).encode()))
                    if p == "/api/fleet/join":
                        return self._send(fleet.join(body))
                    if p == "/api/fleet/settings":
                        return self._send(fleet.set_config(body))
                    if p == "/api/fleet/join-code":
                        return self._send(fleet.new_join_code())
                    if p == "/api/fleet/revoke":
                        return self._send(fleet.revoke(str(body.get("box_id") or "")))
                    if p == "/api/fleet/send-now":
                        return self._send(fleet.send_now())
                except PermissionError as e:
                    return self._send(dict(error=str(e)), 403)
                return self._send(dict(error="not found"), 404)
            if p.startswith("/api/workload/"):
                xid = str(body.get("id") or "")
                if p == "/api/workload/settings":
                    return self._send(dict(settings=autofit.set_settings(INST()["id"], body)))
                if p == "/api/workload/experiment":
                    return self._send(autofit.run_now(INST(), str(body.get("kind") or "")))
                if p == "/api/workload/stop":
                    return self._send(autofit.stop(INST()))
                if p == "/api/workload/proposal/apply":
                    return self._send(autofit.apply_proposal(INST(), xid, bool(body.get("restart"))))
                if p == "/api/workload/proposal/dismiss":
                    return self._send(autofit.dismiss(INST(), xid))
                if p == "/api/workload/rollback":
                    return self._send(autofit.rollback_now(INST(), xid, bool(body.get("restart"))))
                return self._send(dict(error="not found"), 404)
            if p == "/api/mcp/call":
                try:
                    return self._send(dict(result=mcp_server.call_tool(str(body.get("name") or ""),
                                                                        body.get("arguments"))))
                except KeyError:
                    return self._send(dict(error=f"unknown tool {body.get('name')!r}"), 404)
                except mcp_server.ToolError as e:
                    return self._send(dict(error=str(e)), 400)
            if p.startswith("/api/fit/"):
                fit_post = {
                    "/api/fit/solve": fitquant.solve, "/api/fit/build": fitquant.build,
                    "/api/fit/bench": fitquant.bench, "/api/fit/verify": fitquant.verify,
                    "/api/fit/imatrix": fitquant.imatrix, "/api/fit/convert": fitquant.convert,
                    "/api/fit/delete": fitquant.delete, "/api/fit/profile": fitquant.save_profile,
                    "/api/fit/source": fitquant.fetch_source,
                    "/api/fit/explain": fitrecipe.explain, "/api/fit/recipe/check": fitrecipe.check,
                    "/api/fit/recipe/build": fitrecipe.build,
                    "/api/fit/recipe/import": fitrecipe.import_recipe,
                    "/api/fit/recipe/delete": fitrecipe.delete,
                    "/api/fit/imatrix/import": fitrecipe.import_imatrix,
                }
                if p in fit_post:
                    return self._send(fit_post[p](body))
                if p == "/api/fit/check":
                    return self._send(fitquant.check_plan(str(body.get("plan") or "")))
                if p == "/api/fit/cancel":
                    return self._send(fitquant.cancel())
                if p == "/api/fit/updates":
                    return self._send(fitquant.check_updates())
                return self._send(dict(error="not found"), 404)
            if p == "/api/optimize/apply":
                return self._send(optimizer.apply(str(body.get("run") or ""),
                                                  str(body.get("candidate") or ""),
                                                  str(body.get("target") or "params"),
                                                  body.get("name")))
            if p == "/api/optimize/rollback":
                return self._send(optimizer.rollback(str(body.get("run") or "")))
            if p == "/api/tier/reset-fails":
                FAIL_FILE.write_text("0\n")
                return self._send(dict(ok=True, fails=0))
            if p == "/api/builds/install":
                return self._send(start_build_install(
                    (body.get("url") or "").strip(),
                    (body.get("flavour") or "").strip(),
                    body.get("build") or 0,
                    (body.get("runtime_url") or "").strip() or None))
            if p == "/api/builds/activate":
                return self._send(dict(builds_env=set_active_build(
                    (body.get("backend") or "").strip(),
                    (body.get("bindir") or "").strip())))
            if p == "/api/builds/delete":
                return self._send(delete_build((body.get("bindir") or "").strip()))
            if p == "/api/model/delete":
                # Irreversible and ~15 GB. The client must echo the exact
                # filename back in `confirm`, so a mis-routed or replayed POST
                # cannot delete a model the caller never named.
                _path = (body.get("path") or "").strip()
                if (body.get("confirm") or "").strip() != os.path.basename(_path):
                    raise ValueError("confirm must equal the file's basename")
                return self._send(delete_model(_path))
            if p == "/api/ram-budget":
                return self._send(ram_budget(body or {}, include_others=True))
            if p == "/api/launch-plan":
                return self._send(launch_plan(body or None))
            if p == "/api/hf-token":
                return self._send(save_hf_token(body.get("token")))
            if p == "/api/gpu-power/set":
                return self._send(gpupower.set_cap(str(body.get("pci") or ""), body.get("watts")))
            if p == "/api/gpu-power/clear":
                return self._send(gpupower.clear(str(body.get("pci") or "")))
            if p == "/api/instance/create":
                return self._send(create_instance(body))
            if p == "/api/instance/update":
                return self._send(update_instance(str(body.get("id") or ""), body))
            if p == "/api/instance/delete":
                return self._send(delete_instance(str(body.get("id") or ""),
                                                  str(body.get("confirm") or "")))
            if p == "/api/template":
                return self._send(save_template(body.get("name"),
                                                body.get("content") or ""))
            if p == "/api/template/extract":
                return self._send(extract_template(body.get("model"),
                                                   body.get("name")))
            if p == "/api/download":
                url = (body.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    return self._send(dict(error="url must be http(s)"), 400)
                return self._send(start_download(url, body.get("name")))
            return self._send(dict(error="not found"), 404)
        except ValueError as e:
            return self._send(dict(error=str(e)), 400)
        except Exception as e:
            return self._send(dict(error=str(e)), 500)


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


import optimizer                                    # noqa: E402  (needs the names above)
optimizer.bind(sys.modules[__name__])
import engines                                      # noqa: E402
engines.bind(sys.modules[__name__])
import depthcurve                                   # noqa: E402
depthcurve.bind(sys.modules[__name__], optimizer)
import sdcpp                                        # noqa: E402
sdcpp.bind(sys.modules[__name__], engines)
import audiocpp                                     # noqa: E402
audiocpp.bind(sys.modules[__name__], engines)
import camelid                                      # noqa: E402
camelid.bind(sys.modules[__name__], engines)
import onnxrt                                       # noqa: E402  (ONNX Runtime GenAI: CPU / NPU)
onnxrt.bind(sys.modules[__name__])
GEN_ENGINES = {"sd.cpp": sdcpp, "audio.cpp": audiocpp, "camelid": camelid, "onnx": onnxrt}   # non-llama.cpp engines
import flaghelp                                     # noqa: E402
import gpupower                                     # noqa: E402
gpupower.bind(sys.modules[__name__])
import filemgr                                      # noqa: E402
filemgr.bind(sys.modules[__name__])
import flagcatalog                                  # noqa: E402
flagcatalog.bind(sys.modules[__name__])
import fitquant                                     # noqa: E402  (LexiPanel Fit, phases C-F)
fitquant.bind(sys.modules[__name__], engines, optimizer)
import fitrecipe                                    # noqa: E402  (Fit recipes: explain + replicate)
fitrecipe.bind(sys.modules[__name__], fitquant)
import webui                                        # noqa: E402  (llama.cpp built-in web UI)
webui.bind(sys.modules[__name__])
import poweropts                                     # noqa: E402  (Power options tab)
poweropts.bind(sys.modules[__name__])
import refusals                                     # noqa: E402  (uncensoring, phase 0)
refusals.bind(sys.modules[__name__])
import gputune                                      # noqa: E402  (GPU Tuning tab)
gputune.bind(sys.modules[__name__], poweropts, depthcurve, optimizer)
import workload                                     # noqa: E402  (workload profile, 1.0.0)
workload.bind(sys.modules[__name__])
import autofit                                      # noqa: E402  (auto-fit loop, 1.0.0)
autofit.bind(sys.modules[__name__], optimizer, workload)
import auth                                         # noqa: E402  (single / multi-user access, 1.0.0)
auth.bind(PANEL)
import gateway                                      # noqa: E402  (one /v1 endpoint, quotas, usage)
gateway.bind(sys.modules[__name__])
import fleet                                        # noqa: E402  (fleet: one primary, many boxes)
fleet.bind(sys.modules[__name__])
import hermes                                       # noqa: E402  (Hermes Agent setup)
hermes.bind(sys.modules[__name__])
import mcp_server                                   # noqa: E402  (MCP tools: /api/mcp and stdio)
mcp_server.bind(sys.modules[__name__])


if __name__ == "__main__":
    optimizer.recover_on_startup()
    fitquant.recover_on_startup()
    threading.Thread(target=sampler, daemon=True).start()
    threading.Thread(target=engines.scheduler, daemon=True).start()
    threading.Thread(target=gpupower.enforcer, daemon=True).start()
    threading.Thread(target=fitquant.scheduler, daemon=True).start()
    threading.Thread(target=poweropts.worker, daemon=True).start()
    threading.Thread(target=workload.worker, daemon=True).start()
    threading.Thread(target=autofit.worker, daemon=True).start()
    threading.Thread(target=fleet.worker, daemon=True).start()
    print(f"inf01 panel API on http://{BIND}:{PORT} (localhost only; Caddy fronts it)")
    Threaded((BIND, PORT), Handler).serve_forever()
