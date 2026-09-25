#!/usr/bin/env python3
"""
Host operating system layer (added 2026-09-23).

The few questions LexiPanel asks the OS, answered for Linux and macOS:
memory, process details, listening ports, the GPU, where run folders live,
and the service manager that keeps instances alive (systemd --user on Linux,
launchd agents on macOS).

  Linux  is the tested platform. Every Linux branch here returns exactly what
         the panel computed inline before this module existed, so behaviour on
         Linux does not change.
  macOS  is EXPERIMENTAL: written from Apple's documentation and the upstream
         release layouts, never run on a real Mac by the authors. Apple
         Silicon only is expected to work (Metal builds of llama.cpp,
         stable-diffusion.cpp, audio.cpp and Camelid). Linux-only features
         (AMD power caps, sysfs thermals and clocks, DRM residency, journald
         crash triage, Vulkan pinning) report "not available on macOS".
"""
import ctypes, ctypes.util, os, platform, plistlib, re, shlex, subprocess, sys, tempfile, time
from pathlib import Path

IS_MAC = sys.platform == "darwin"
ARCH = platform.machine()                      # "arm64" on Apple Silicon, "x86_64"
MAC_ONLY_NOTE = "not available on macOS"


def _out(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------
_MAC_MEM = {}


def _mac_mem():
    """MiB figures in the shape of /proc/meminfo keys. Cached for 2 s."""
    c = _MAC_MEM.get("at")
    if c and time.time() - c < 2:
        return _MAC_MEM["v"]
    v = {}
    total = _out(["sysctl", "-n", "hw.memsize"]).strip()
    if total.isdigit():
        v["MemTotal"] = int(total) // 1048576
    vm = _out(["vm_stat"])
    m = re.search(r"page size of (\d+) bytes", vm)
    page = int(m.group(1)) if m else 16384
    pages = {k.strip().lower(): int(n) for k, n in re.findall(r"^Pages ([^:]+):\s+(\d+)\.", vm, re.M)}
    # what can be handed out without swapping: free + inactive + speculative + purgeable
    avail = sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))
    v["MemAvailable"] = avail * page // 1048576
    v["MemFree"] = pages.get("free", 0) * page // 1048576
    sw = _out(["sysctl", "-n", "vm.swapusage"])
    t = re.search(r"total = ([\d.]+)M", sw)
    f = re.search(r"free = ([\d.]+)M", sw)
    if t:
        v["SwapTotal"] = int(float(t.group(1)))
    if f:
        v["SwapFree"] = int(float(f.group(1)))
    _MAC_MEM.update(at=time.time(), v=v)
    return v


def meminfo_mb(key):
    """/proc/meminfo value in MiB (Linux), or its macOS equivalent."""
    if IS_MAC:
        return _mac_mem().get(key.rstrip(":"), 0)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


# --------------------------------------------------------------------------
# processes
# --------------------------------------------------------------------------
def _procargs2(pid):
    """macOS KERN_PROCARGS2: (exec_path, argv, env). Same user only."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    CTL_KERN, KERN_ARGMAX, KERN_PROCARGS2 = 1, 8, 49
    argmax = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(argmax))
    mib = (ctypes.c_int * 2)(CTL_KERN, KERN_ARGMAX)
    if libc.sysctl(mib, 2, ctypes.byref(argmax), ctypes.byref(size), None, 0) != 0:
        raise OSError("sysctl KERN_ARGMAX failed")
    buf = ctypes.create_string_buffer(argmax.value)
    size = ctypes.c_size_t(argmax.value)
    mib3 = (ctypes.c_int * 3)(CTL_KERN, KERN_PROCARGS2, int(pid))
    if libc.sysctl(mib3, 3, buf, ctypes.byref(size), None, 0) != 0:
        raise OSError(f"no such process or not ours: {pid}")
    raw = buf.raw[:size.value]
    argc = int.from_bytes(raw[:4], sys.byteorder)
    rest = raw[4:]
    exe, _, rest = rest.partition(b"\0")
    rest = rest.lstrip(b"\0")
    parts = rest.split(b"\0")
    argv = [p.decode(errors="replace") for p in parts[:argc]]
    env = {}
    for p in parts[argc:]:
        if not p:
            break
        k, _, val = p.decode(errors="replace").partition("=")
        env[k] = val
    return exe.decode(errors="replace"), argv, env


def proc_argv(pid):
    """argv of a process as a list; raises OSError when it cannot be read."""
    if not IS_MAC:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    try:
        return _procargs2(pid)[1] + [""]           # same trailing "" shape as /proc
    except (OSError, AttributeError):
        out = _out(["ps", "-ww", "-o", "command=", "-p", str(pid)]).strip()
        if not out:
            raise OSError(f"no process {pid}")
        return shlex.split(out) + [""]


def proc_environ(pid):
    if not IS_MAC:
        return dict(l.split("=", 1) for l in Path(f"/proc/{pid}/environ").read_bytes()
                    .decode(errors="ignore").split("\0") if "=" in l)
    return _procargs2(pid)[2]


def proc_exe(pid):
    if not IS_MAC:
        return os.path.realpath(f"/proc/{pid}/exe")
    try:
        return os.path.realpath(_procargs2(pid)[0])
    except (OSError, AttributeError):
        return proc_argv(pid)[0]


def pid_alive(pid):
    if not IS_MAC:
        return os.path.exists(f"/proc/{pid}")
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def proc_uptime_s(pid):
    if IS_MAC:
        et = _out(["ps", "-o", "etime=", "-p", str(pid)]).strip()      # [[dd-]hh:]mm:ss
        if not et:
            return None
        days, _, hms = et.rpartition("-")
        parts = [int(x) for x in hms.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    try:
        start = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        up = float(Path("/proc/uptime").read_text().split()[0])
        return int(up - start / os.sysconf("SC_CLK_TCK"))
    except Exception:
        return None


def proc_rss_mb(pid):
    if IS_MAC:
        kb = _out(["ps", "-o", "rss=", "-p", str(pid)]).strip()
        return int(kb) // 1024 if kb.isdigit() else None
    try:
        return int(re.search(r"VmRSS:\s+(\d+)", Path(f"/proc/{pid}/status").read_text()).group(1)) // 1024
    except Exception:
        return None


def children(pid):
    out = _out(["pgrep", "-P", str(pid)])
    return [int(x) for x in out.split() if x.isdigit()]


def open_paths(pid):
    """Paths a process holds open, mapped, or as cwd (macOS: lsof)."""
    out = _out(["lsof", "-n", "-P", "-F", "n", "-p", str(pid)], timeout=20)
    return [l[1:] for l in out.splitlines() if l.startswith("n/")]


# --------------------------------------------------------------------------
# ports
# --------------------------------------------------------------------------
def port_listening(port):
    """Non-empty string if something listens on TCP `port` (the ss/lsof line)."""
    port = int(port)
    if IS_MAC:
        return _out(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]).strip()
    return subprocess.run(f"ss -Hltn 'sport = :{port}' 2>/dev/null", shell=True,
                          capture_output=True, text=True, timeout=15).stdout.strip()


# --------------------------------------------------------------------------
# GPU (macOS: the Metal device; Linux keeps its sysfs/PCI discovery in panel.py)
# --------------------------------------------------------------------------
_MAC_GPU = {}


def mac_gpu():
    """Device record for the Mac's GPU in panel.gpu_devices() shape. Unified
    memory: Metal may use roughly 75% of RAM by default (recommendedMaxWorkingSetSize)."""
    if "rec" in _MAC_GPU:
        return _MAC_GPU["rec"]
    name, cores = "Apple GPU", None
    try:
        import json
        d = json.loads(_out(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30) or "{}")
        g = (d.get("SPDisplaysDataType") or [{}])[0]
        name = g.get("sppci_model") or g.get("_name") or name
        cores = g.get("sppci_cores")
    except ValueError:
        pass
    total = meminfo_mb("MemTotal")
    rec = dict(pci="apple-gpu", vendor="apple", driver="metal",
               name=f"{name}" + (f" ({cores}-core GPU)" if cores else ""),
               vram_total_mib=total * 3 // 4 if total else None,
               backends=["metal", "cpu"], cuda_ready=False, usable=True,
               notes=["Unified memory: the GPU shares system RAM; macOS lets Metal use about "
                      "75% of it by default. Experimental: LexiPanel on macOS is untested."])
    _MAC_GPU["rec"] = rec
    return rec


# --------------------------------------------------------------------------
# run folders
# --------------------------------------------------------------------------
def run_base():
    """Where instance run folders (logs, telemetry, generated configs) live."""
    if IS_MAC:
        d = Path(tempfile.gettempdir()) / "lexipanel"
        d.mkdir(exist_ok=True)
        return d
    return Path("/dev/shm")


# --------------------------------------------------------------------------
# launchd agents (macOS service manager for instances)
# --------------------------------------------------------------------------
def la_label(iid):
    return f"com.lexipanel.inst.{iid}"


def la_plist(iid):
    return Path.home() / "Library/LaunchAgents" / f"{la_label(iid)}.plist"


def _gui():
    return f"gui/{os.getuid()}"


def la_write(iid, panel_dir, workdir, logs_dir):
    """The agent that runs instance_launch.py for one instance. Not started at
    login (like the Linux units, which are started on demand, not enabled)."""
    p = la_plist(iid)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(Label=la_label(iid),
                ProgramArguments=[sys.executable, str(Path(panel_dir) / "instance_launch.py"), iid],
                WorkingDirectory=str(workdir), RunAtLoad=True, KeepAlive=False,
                ProcessType="Interactive",
                StandardOutPath=str(Path(logs_dir) / f"launchd_{iid}.out"),
                StandardErrorPath=str(Path(logs_dir) / f"launchd_{iid}.out"),
                EnvironmentVariables=dict(PATH=os.environ.get("PATH", "/usr/bin:/bin")))
    data = plistlib.dumps(body)
    if not p.exists() or p.read_bytes() != data:
        p.write_bytes(data)
    return p


def la_state(iid):
    """(active, pid) for an instance's agent."""
    r = subprocess.run(["launchctl", "print", f"{_gui()}/{la_label(iid)}"],
                       capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        return "inactive", None
    m = re.search(r"^\s*pid = (\d+)", r.stdout, re.M)
    st = re.search(r"^\s*state = (\S+)", r.stdout, re.M)
    return ("active" if m else (st.group(1) if st else "inactive")), (int(m.group(1)) if m else None)


def la_start(iid):
    p = la_plist(iid)
    subprocess.run(["launchctl", "bootout", f"{_gui()}/{la_label(iid)}"],
                   capture_output=True, timeout=30)                     # stale load, if any
    r = subprocess.run(["launchctl", "bootstrap", _gui(), str(p)],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def la_stop(iid):
    r = subprocess.run(["launchctl", "bootout", f"{_gui()}/{la_label(iid)}"],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def la_remove(iid):
    la_stop(iid)
    try:
        la_plist(iid).unlink()
    except OSError:
        pass
