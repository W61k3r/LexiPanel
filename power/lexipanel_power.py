#!/usr/bin/python3 -I
"""
LexiPanel power options helper (added 2026-09-24).

Installed as /usr/local/sbin/lexipanel-power (root:root 0755) and run by the panel
through ONE sudoers line. The panel also imports this same file, unprivileged, for its
read-only view, so the two can never disagree about what a setting is.

This file is the security boundary. Every target (a PCI address, an NVMe controller,
"all" CPUs) must be one this helper enumerated itself, and every value must be in that
target's own list of choices or inside its own range. Nothing from the caller becomes a
path, a shell word or file content. The only files it writes are:
  sysfs attributes and PCI config space (the settings themselves),
  /etc/lexipanel/power.json          the boot profile (validated before writing),
  /run/lexipanel/*.json              this boot's baseline and boot-apply log,
  two systemd drop-ins it owns       (watchdog, journald sync) and one modules-load file.
It removes only files it created.

  lexipanel-power status             everything readable, as JSON
  lexipanel-power apply   < json     {"settings":[{"knob","target","ident","value"}...]}
  lexipanel-power persist < json     {"name", "settings":[...]} -> the boot profile
  lexipanel-power unpersist          no boot profile
  lexipanel-power vbios   < json     {"target": "<pci>"} -> that GPU's VBIOS image (read only)
  lexipanel-power boot               (boot unit) record this boot's baseline, apply the profile

Version 2 (2026-09-25) adds the amdgpu OverDrive clock and voltage-offset knobs (GPU
Tuning tab) and the read-only vbios verb. Nothing in this helper writes GPU firmware.
"""
import json, os, re, struct, subprocess, sys, time

VERSION = 2

# Roots (tests point these at a fake tree)
SYS, PROC, ETC, RUN = "/sys", "/proc", "/etc", "/run"
BIN = dict(systemctl="/usr/bin/systemctl", modprobe="/usr/sbin/modprobe",
           nvidia_smi="/usr/bin/nvidia-smi", analyze="/usr/bin/systemd-analyze")

STATE_DIR = f"{ETC}/lexipanel"
PROFILE_FILE = f"{STATE_DIR}/power.json"
RUN_DIR = f"{RUN}/lexipanel"
BASELINE_FILE = f"{RUN_DIR}/power-baseline.json"
BOOTLOG_FILE = f"{RUN_DIR}/power-boot.json"
WD_DROPIN = f"{ETC}/systemd/system.conf.d/90-lexipanel-watchdog.conf"
WD_MODLOAD = f"{ETC}/modules-load.d/lexipanel-watchdog.conf"
JD_DROPIN = f"{ETC}/systemd/journald.conf.d/90-lexipanel-sync.conf"
MAX_INPUT = 256 * 1024
ASPM = {"off": 0, "l0s": 1, "l1": 2, "l0s+l1": 3}
ASPM_NAME = {v: k for k, v in ASPM.items()}
WATCHDOG_MODULES = {"GenuineIntel": ["iTCO_wdt"], "AuthenticAMD": ["sp5100_tco"]}


# ============================================================================
# small helpers
# ============================================================================
def _r(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _w(path, text):
    with open(path, "w") as f:
        f.write(text)


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _run(argv, timeout=20, stdin=None):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


def is_root():
    return os.geteuid() == 0


def boot_id():
    return _r(f"{PROC}/sys/kernel/random/boot_id")


# ============================================================================
# PCI
# ============================================================================
def pci_devices():
    base = f"{SYS}/bus/pci/devices"
    out = {}
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for a in names:
        d = f"{base}/{a}"
        cls = _r(f"{d}/class")
        if not cls:
            continue
        real = os.path.realpath(d)
        drv = f"{d}/driver"
        out[a] = dict(addr=a, cls=int(cls, 16),
                      vendor=(_r(f"{d}/vendor") or "0x")[2:], device=(_r(f"{d}/device") or "0x")[2:],
                      sub=f"{(_r(f'{d}/subsystem_vendor') or '0x')[2:]}:{(_r(f'{d}/subsystem_device') or '0x')[2:]}",
                      driver=os.path.basename(os.path.realpath(drv)) if os.path.exists(drv) else None,
                      parent=os.path.basename(os.path.dirname(real)))
    return out


def ident(dev):
    return f"{dev['vendor']}:{dev['device']}:{dev['sub']}"


def _cfg(addr):
    try:
        with open(f"{SYS}/bus/pci/devices/{addr}/config", "rb") as f:
            return f.read(256)
    except OSError:
        return b""


def _pcie_cap(cfg):
    """Offset of the PCI Express capability, or None (not PCIe, or not readable:
    unprivileged reads of config space stop at 64 bytes)."""
    if len(cfg) < 64 or not (cfg[6] & 0x10):
        return None
    p, seen = cfg[0x34] & 0xFC, set()
    while p and p not in seen and p + 2 <= len(cfg):
        seen.add(p)
        if cfg[p] == 0x10:
            return p if p + 0x12 <= len(cfg) else None
        p = cfg[p + 1] & 0xFC
    return None


def aspm_regs(addr):
    """dict(ptype, sup, ctl) from the PCIe capability, or None if unreadable."""
    cfg = _cfg(addr)
    cp = _pcie_cap(cfg)
    if cp is None:
        return None
    flags = struct.unpack_from("<H", cfg, cp + 2)[0]
    lnkcap = struct.unpack_from("<I", cfg, cp + 0x0C)[0]
    lnkctl = struct.unpack_from("<H", cfg, cp + 0x10)[0]
    return dict(ptype=(flags >> 4) & 0xF, sup=(lnkcap >> 10) & 3, ctl=lnkctl & 3, cp=cp)


def _set_aspm_bits(addr, bits):
    cfg = _cfg(addr)
    cp = _pcie_cap(cfg)
    if cp is None:
        raise OSError(f"{addr}: PCIe capability not readable")
    off = cp + 0x10
    cur = struct.unpack_from("<H", cfg, off)[0]
    new = (cur & ~3) | bits
    if new != cur:
        fd = os.open(f"{SYS}/bus/pci/devices/{addr}/config", os.O_WRONLY)
        try:
            os.pwrite(fd, struct.pack("<H", new), off)
        finally:
            os.close(fd)


def _cls_kind(cls):
    top = cls >> 16
    if top == 0x03:
        return "gpu"
    if cls >> 8 == 0x0108:
        return "nvme"
    if top == 0x01:
        return "storage"
    if top == 0x02:
        return "network"
    if cls >> 8 == 0x0c03:
        return "usb"
    if cls >> 8 == 0x0604:
        return "bridge"
    return "other"


def pcie_links(devs=None):
    """One entry per real PCIe link, named by the device below it (function 0).
    A switch's internal link (upstream port -> its downstream port) is left out."""
    devs = devs or pci_devices()
    out = []
    for a, d in devs.items():
        if not a.endswith(".0"):
            continue
        port = d["parent"]
        pd = devs.get(port)
        if not pd or pd["cls"] >> 8 != 0x0604:
            continue
        pr = aspm_regs(port)
        if pr is not None:
            if pr["ptype"] not in (4, 6):          # root port / switch downstream port
                continue
        elif d["cls"] >> 8 == 0x0604 and devs.get(pd["parent"], {}).get("cls", 0) >> 8 == 0x0604:
            continue                                # unprivileged guess: switch-internal
        fns = sorted(x for x in devs if x.rsplit(".", 1)[0] == a.rsplit(".", 1)[0])
        kind = _cls_kind(d["cls"])
        if kind == "bridge":                    # a switch (e.g. on the GPU card): name it after
            below, todo = [], [a]               # what sits behind it
            while todo:
                x = todo.pop()
                kids = [y for y, dd in devs.items() if dd["parent"] == x]
                below += [_cls_kind(devs[y]["cls"]) for y in kids if _cls_kind(devs[y]["cls"]) != "bridge"]
                todo += [y for y in kids if _cls_kind(devs[y]["cls"]) == "bridge"]
            main = sorted(set(below) - {"other"}) or sorted(set(below))
            kind = f"{'/'.join(main) or 'empty'} switch"
        out.append(dict(target=a, port=port, fns=fns, kind=kind, ident=ident(d),
                        driver=d["driver"]))
    return out


# ============================================================================
# knobs: targets, read, choices, write
# ============================================================================
def _cpu_dirs():
    base = f"{SYS}/devices/system/cpu"
    try:
        return sorted(f"{base}/{c}" for c in os.listdir(base) if re.fullmatch(r"cpu\d+", c))
    except OSError:
        return []


class Knob:
    name = ""
    label = ""
    order = 50

    def targets(self, ctx):          # [dict(target, ident, label)]
        return []

    def read(self, ctx, t):
        return None

    def choices(self, ctx, t):       # list of strings, or dict(kind="range"...) for free values
        return []

    def valid(self, ctx, t, v):
        ch = self.choices(ctx, t)
        return isinstance(ch, list) and v in ch

    def write(self, ctx, t, v):
        raise NotImplementedError

    def same(self, ctx, t, want, got):
        return str(want) == str(got)


class CpuGovernor(Knob):
    name, label, order = "cpu.governor", "CPU frequency governor", 10

    def targets(self, ctx):
        c0 = f"{SYS}/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        return [dict(target="all", label="all CPUs")] if _r(c0) else []

    def read(self, ctx, t):
        vals = {_r(f"{c}/cpufreq/scaling_governor") for c in _cpu_dirs()} - {None}
        return vals.pop() if len(vals) == 1 else ("mixed" if vals else None)

    def choices(self, ctx, t):
        return (_r(f"{SYS}/devices/system/cpu/cpu0/cpufreq/scaling_available_governors") or "").split()

    def write(self, ctx, t, v):
        for c in _cpu_dirs():
            p = f"{c}/cpufreq/scaling_governor"
            if os.path.exists(p):
                _w(p, v)


class CpuEpp(Knob):
    name, label, order = "cpu.epp", "CPU energy/performance preference", 11

    def targets(self, ctx):
        c0 = f"{SYS}/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
        return [dict(target="all", label="all CPUs")] if _r(c0) else []

    def read(self, ctx, t):
        vals = {_r(f"{c}/cpufreq/energy_performance_preference") for c in _cpu_dirs()} - {None}
        return vals.pop() if len(vals) == 1 else ("mixed" if vals else None)

    def choices(self, ctx, t):
        return (_r(f"{SYS}/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences")
                or "").split()

    def write(self, ctx, t, v):
        gov = CpuGovernor().read(ctx, "all")
        if gov == "performance" and v != "performance" and \
                _r(f"{SYS}/devices/system/cpu/intel_pstate/status") == "active":
            raise OSError("intel_pstate pins the preference to 'performance' under the "
                          "performance governor; change the governor first")
        for c in _cpu_dirs():
            p = f"{c}/cpufreq/energy_performance_preference"
            if os.path.exists(p):
                _w(p, v)


class CpuIdle(Knob):
    """Deepest C-state allowed: every deeper state is disabled on every CPU."""
    name, label, order = "cpu.idle", "Deepest CPU idle state allowed", 12

    def _states(self, cpu=f"cpu0"):
        base = f"{SYS}/devices/system/cpu/{cpu}/cpuidle"
        try:
            ss = sorted((int(s[5:]), s) for s in os.listdir(base) if re.fullmatch(r"state\d+", s))
        except OSError:
            return []
        return [(i, _r(f"{base}/{s}/name"), f"{base}/{s}") for i, s in ss]

    def targets(self, ctx):
        return [dict(target="all", label="all CPUs")] if len(self._states()) > 1 else []

    def read(self, ctx, t):
        ss = self._states()
        dis = [_r(f"{p}/disable") == "1" for _i, _n, p in ss]
        if not any(dis):
            return "all"
        first = dis.index(True)
        if all(dis[first:]) and first > 0:
            return ss[first - 1][1]
        return "custom"

    def choices(self, ctx, t):
        return [n for _i, n, _p in self._states()[1:-1]] + ["all"]

    def write(self, ctx, t, v):
        names = [n for _i, n, _p in self._states()]
        cut = len(names) if v == "all" else names.index(v) + 1
        for c in _cpu_dirs():
            for i, _n, p in self._states(os.path.basename(c)):
                if os.path.exists(f"{p}/disable"):
                    _w(f"{p}/disable", "1" if i >= cut else "0")


class CpuTurbo(Knob):
    name, label, order = "cpu.turbo", "CPU turbo / boost", 13

    def _file(self):
        for p, inv in ((f"{SYS}/devices/system/cpu/intel_pstate/no_turbo", True),
                       (f"{SYS}/devices/system/cpu/cpufreq/boost", False)):
            if _r(p) is not None:
                return p, inv
        return None, None

    def targets(self, ctx):
        return [dict(target="all", label="all CPUs")] if self._file()[0] else []

    def read(self, ctx, t):
        p, inv = self._file()
        v = _r(p) if p else None
        return None if v is None else ("on" if (v == "0") == inv else "off")

    def choices(self, ctx, t):
        return ["on", "off"]

    def write(self, ctx, t, v):
        p, inv = self._file()
        _w(p, ("0" if v == "on" else "1") if inv else ("1" if v == "on" else "0"))


class AspmPolicy(Knob):
    name, label, order = "pcie.aspm_policy", "PCIe ASPM kernel policy", 20

    def _f(self):
        return f"{SYS}/module/pcie_aspm/parameters/policy"

    def targets(self, ctx):
        return [dict(target="global", label="kernel")] if _r(self._f()) else []

    def read(self, ctx, t):
        m = re.search(r"\[(\w+)\]", _r(self._f()) or "")
        return m.group(1) if m else None

    def choices(self, ctx, t):
        return re.sub(r"[\[\]]", "", _r(self._f()) or "").split()

    def write(self, ctx, t, v):
        _w(self._f(), v)


class LinkAspm(Knob):
    """Link power states on one PCIe link, written on both ends (every function of
    the device, and the port above it) the way the kernel orders it."""
    name, label, order = "pcie.aspm", "PCIe link power saving (ASPM)", 21

    def targets(self, ctx):
        out = []
        for l in ctx.links():
            d = ctx.devs()[l["target"]]
            out.append(dict(target=l["target"], ident=l["ident"], kind=l["kind"],
                            label=f"{l['kind']} {l['target']} (port {l['port']})",
                            driver=d["driver"]))
        return out

    def _link(self, ctx, t):
        return next((l for l in ctx.links() if l["target"] == t), None)

    def read(self, ctx, t):
        l = self._link(ctx, t)
        dr, pr = (aspm_regs(t), aspm_regs(l["port"])) if l else (None, None)
        if dr is None or pr is None:
            return None                        # needs root to read config space
        if dr["ctl"] == pr["ctl"]:
            return ASPM_NAME[dr["ctl"]]
        return f"mixed (device {ASPM_NAME[dr['ctl']]}, port {ASPM_NAME[pr['ctl']]})"

    def choices(self, ctx, t):
        l = self._link(ctx, t)
        dr, pr = (aspm_regs(t), aspm_regs(l["port"])) if l else (None, None)
        if dr is None or pr is None:
            return ["off"]                     # always safe; others need the root view
        sup = dr["sup"] & pr["sup"]
        return [n for n, b in ASPM.items() if b & ~sup == 0]

    def write(self, ctx, t, v):
        l = self._link(ctx, t)
        bits = ASPM[v]
        fns = [f for f in l["fns"] if aspm_regs(f) is not None]
        cur = aspm_regs(t)["ctl"]
        low = cur & bits
        for f in fns:                          # 1. drop states being removed: device first
            _set_aspm_bits(f, low)
        _set_aspm_bits(l["port"], low)
        _set_aspm_bits(l["port"], bits)        # 2. add states: port first
        for f in fns:
            _set_aspm_bits(f, bits)


class NvmeApst(Knob):
    name, label, order = "nvme.apst", "NVMe autonomous power states (APST)", 30

    def _ctrls(self):
        base = f"{SYS}/class/nvme"
        try:
            return sorted(c for c in os.listdir(base) if re.fullmatch(r"nvme\d+", c))
        except OSError:
            return []

    def targets(self, ctx):
        out = []
        for c in self._ctrls():
            if _r(f"{SYS}/class/nvme/{c}/power/pm_qos_latency_tolerance_us") is None:
                continue
            addr = os.path.basename(os.path.realpath(f"{SYS}/class/nvme/{c}/device"))
            d = ctx.devs().get(addr)
            out.append(dict(target=c, ident=ident(d) if d else None, pci=addr,
                            label=f"{c} {(_r(f'{SYS}/class/nvme/{c}/model') or '').strip()}"))
        return out

    def read(self, ctx, t):
        v = _int(_r(f"{SYS}/class/nvme/{t}/power/pm_qos_latency_tolerance_us"))
        return None if v is None else ("off" if v == 0 else "on")

    def choices(self, ctx, t):
        return ["on", "off"]

    def write(self, ctx, t, v):
        on = 100000
        b = (ctx.baseline_raw() or {}).get(f"nvme.apst.us:{t}")
        if _int(b):
            on = _int(b)
        _w(f"{SYS}/class/nvme/{t}/power/pm_qos_latency_tolerance_us", "0" if v == "off" else str(on))


class RuntimePm(Knob):
    name, label, order = "pci.runpm", "PCI runtime power management", 31

    def targets(self, ctx):
        out = []
        for a, d in ctx.devs().items():
            if _cls_kind(d["cls"]) in ("gpu", "nvme", "network") and \
                    _r(f"{SYS}/bus/pci/devices/{a}/power/control") in ("auto", "on"):
                out.append(dict(target=a, ident=ident(d), kind=_cls_kind(d["cls"]),
                                label=f"{_cls_kind(d['cls'])} {a} ({d['driver'] or 'no driver'})"))
        return out

    def read(self, ctx, t):
        return _r(f"{SYS}/bus/pci/devices/{t}/power/control")

    def choices(self, ctx, t):
        return ["auto", "on"]

    def write(self, ctx, t, v):
        _w(f"{SYS}/bus/pci/devices/{t}/power/control", v)


def _amd_gpus(ctx):
    return [(a, d) for a, d in ctx.devs().items() if _cls_kind(d["cls"]) == "gpu" and d["driver"] == "amdgpu"]


class GpuPerfLevel(Knob):
    name, label, order = "gpu.perf_level", "GPU performance level (amdgpu)", 40

    def targets(self, ctx):
        return [dict(target=a, ident=ident(d), label=f"GPU {a}") for a, d in _amd_gpus(ctx)
                if _r(f"{SYS}/bus/pci/devices/{a}/power_dpm_force_performance_level")]

    def read(self, ctx, t):
        return _r(f"{SYS}/bus/pci/devices/{t}/power_dpm_force_performance_level")

    def choices(self, ctx, t):
        return ["auto", "low", "high"]

    def write(self, ctx, t, v):
        _w(f"{SYS}/bus/pci/devices/{t}/power_dpm_force_performance_level", v)


class GpuFan(Knob):
    """amdgpu OverDrive fan curve: 'auto' (firmware) or
    'T:P T:P T:P T:P T:P zero_rpm=0|1' (hotspot C : fan %)."""
    name, label, order = "gpu.fan", "GPU fan curve (amdgpu OverDrive)", 41

    def _d(self, t):
        return f"{SYS}/bus/pci/devices/{t}/gpu_od/fan_ctrl"

    def _curve(self, t):
        txt = _r(f"{self._d(t)}/fan_curve") or ""
        pts = [(int(a), int(b)) for a, b in re.findall(r"^\s*\d+:\s*(\d+)C\s+(\d+)%", txt, re.M)]
        tr = re.search(r"temp\):\s*(\d+)C\s+(\d+)C", txt)
        sr = re.search(r"speed\):\s*(\d+)%\s+(\d+)%", txt)
        return pts, (tuple(map(int, tr.groups())) if tr else None), (tuple(map(int, sr.groups())) if sr else None)

    def _zero(self, t):
        m = re.search(r"FAN_ZERO_RPM_ENABLE:\s*(\d)", _r(f"{self._d(t)}/fan_zero_rpm_enable") or "")
        return int(m.group(1)) if m else None

    def targets(self, ctx):
        return [dict(target=a, ident=ident(d), label=f"GPU {a}") for a, d in _amd_gpus(ctx)
                if _r(f"{self._d(a)}/fan_curve")]

    def read(self, ctx, t):
        pts, _tr, _sr = self._curve(t)
        z = self._zero(t)
        if not pts:
            return None
        if all(p == (0, 0) for p in pts):
            return "auto"
        return " ".join(f"{a}:{b}" for a, b in pts) + (f" zero_rpm={z}" if z is not None else "")

    def choices(self, ctx, t):
        pts, tr, sr = self._curve(t)
        return dict(kind="fan_curve", points=len(pts), temp=tr, speed=sr,
                    zero_rpm=self._zero(t) is not None, example="45:30 60:45 70:65 80:85 90:100 zero_rpm=0")

    def _parse(self, ctx, t, v):
        if v == "auto":
            return "auto", None
        m = re.fullmatch(r"((?:\d{1,3}:\d{1,3}\s*){2,8})(?:zero_rpm=([01]))?\s*", str(v))
        if not m:
            return None
        pts = [tuple(map(int, x.split(":"))) for x in m.group(1).split()]
        n, tr, sr = (lambda c: (len(c[0]), c[1], c[2]))(self._curve(t))
        if len(pts) != n or not tr or not sr:
            return None
        if any(not (tr[0] <= a <= tr[1] and sr[0] <= b <= sr[1]) for a, b in pts):
            return None
        if any(pts[i][0] > pts[i + 1][0] or pts[i][1] > pts[i + 1][1] for i in range(n - 1)):
            return None                        # a curve never goes down
        return pts, (int(m.group(2)) if m.group(2) is not None else None)

    def valid(self, ctx, t, v):
        return self._parse(ctx, t, v) is not None

    def write(self, ctx, t, v):
        pts, z = self._parse(ctx, t, v)
        d = self._d(t)
        if pts == "auto":
            _w(f"{d}/fan_curve", "r")
            _w(f"{d}/fan_curve", "c")
            if self._zero(t) is not None:
                _w(f"{d}/fan_zero_rpm_enable", "1")
                _w(f"{d}/fan_zero_rpm_enable", "c")
            return
        for i, (a, b) in enumerate(pts):
            _w(f"{d}/fan_curve", f"{i} {a} {b}")
        _w(f"{d}/fan_curve", "c")
        if z is not None and self._zero(t) is not None:
            _w(f"{d}/fan_zero_rpm_enable", str(z))
            _w(f"{d}/fan_zero_rpm_enable", "c")

    def same(self, ctx, t, want, got):
        norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip()
        return norm(want) == norm(got) or (want == "auto" and got == "auto")


class GpuPowerCap(Knob):
    """Board power limit in watts: amdgpu power1_cap, or NVIDIA through nvidia-smi."""
    name, label, order = "gpu.power_cap", "GPU power cap (W)", 42

    def _hwmon(self, a):
        base = f"{SYS}/bus/pci/devices/{a}/hwmon"
        try:
            for h in sorted(os.listdir(base)):
                if os.path.exists(f"{base}/{h}/power1_cap"):
                    return f"{base}/{h}"
        except OSError:
            pass
        return None

    def _nv(self, ctx):
        if "nv" not in ctx.cache:
            ctx.cache["nv"] = {}
            if os.path.exists(BIN["nvidia_smi"]):
                rc, out = _run([BIN["nvidia_smi"], "--query-gpu=pci.bus_id,power.limit,power.min_limit,"
                                "power.max_limit", "--format=csv,noheader,nounits"], timeout=10)
                for line in out.splitlines() if rc == 0 else []:
                    f = [x.strip() for x in line.split(",")]
                    if len(f) == 4:
                        ctx.cache["nv"][f[0].lower()[-12:]] = dict(bus=f[0], limit=f[1], min=f[2], max=f[3])
        return ctx.cache["nv"]

    def targets(self, ctx):
        out = []
        for a, d in ctx.devs().items():
            if _cls_kind(d["cls"]) != "gpu":
                continue
            if (d["driver"] == "amdgpu" and self._hwmon(a)) or (d["driver"] == "nvidia" and a[-12:] in self._nv(ctx)):
                out.append(dict(target=a, ident=ident(d), label=f"GPU {a} ({d['driver']})"))
        return out

    def _range(self, ctx, t):
        h = self._hwmon(t)
        if h:
            f = lambda n: (_int(_r(f"{h}/{n}")) or 0) // 1000000
            return f("power1_cap_min"), f("power1_cap_max"), f("power1_cap")
        nv = self._nv(ctx).get(t[-12:])
        if nv:
            g = lambda x: int(float(x)) if re.fullmatch(r"[\d.]+", x or "") else None
            return g(nv["min"]), g(nv["max"]), g(nv["limit"])
        return None, None, None

    def read(self, ctx, t):
        v = self._range(ctx, t)[2]
        return None if v is None else ("unset" if v == 0 else str(v))

    def choices(self, ctx, t):
        lo, hi, _ = self._range(ctx, t)
        return dict(kind="watts", min=lo, max=hi)

    def valid(self, ctx, t, v):
        lo, hi, _ = self._range(ctx, t)
        return bool(re.fullmatch(r"\d{1,4}", str(v))) and lo is not None and lo <= int(v) <= hi

    def write(self, ctx, t, v):
        h = self._hwmon(t)
        if h:
            _w(f"{h}/power1_cap", str(int(v) * 1000000))
            return
        nv = self._nv(ctx).get(t[-12:])
        rc, out = _run([BIN["nvidia_smi"], "-i", nv["bus"], "-pl", str(int(v))], timeout=20)
        ctx.cache.pop("nv", None)
        if rc != 0:
            raise OSError(out.strip()[-300:])


def od_table(addr):
    """amdgpu OverDrive table from pp_od_clk_voltage, or None when OverDrive is off.

    RDNA2/RDNA3 (SMU11 sienna_cichlid, SMU13) print a min/max pair per clock:
        OD_SCLK:  0: 500Mhz  1: 2500Mhz       OD_MCLK:  0: 97Mhz  1: 1250MHz
        OD_VDDGFX_OFFSET:  0mV
        OD_RANGE: SCLK: 500Mhz 3150Mhz  MCLK: 97Mhz 1500Mhz  VDDGFX_OFFSET: -450mv 0mv
    Older chips (Vega, Navi1x) print a voltage curve instead; they get no clock
    entries here, so these knobs list no target for them rather than guess."""
    txt = _r(f"{SYS}/bus/pci/devices/{addr}/pp_od_clk_voltage")
    if not txt:
        return None
    sec, out = None, dict(sclk={}, mclk={}, vo=None, range={})
    for line in txt.splitlines():
        line = line.strip()
        m = re.fullmatch(r"(OD_[A-Z_]+):", line)
        if m:
            sec = m.group(1)
            continue
        if sec in ("OD_SCLK", "OD_MCLK"):
            m = re.fullmatch(r"(\d+):\s*(\d+)\s*mhz", line, re.I)
            if m:
                out["sclk" if sec == "OD_SCLK" else "mclk"][int(m.group(1))] = int(m.group(2))
        elif sec == "OD_VDDGFX_OFFSET":
            m = re.fullmatch(r"(-?\d+)\s*mv", line, re.I)
            if m:
                out["vo"] = int(m.group(1))
        elif sec == "OD_RANGE":
            m = re.fullmatch(r"(\w+):\s*(-?\d+)\s*(?:mhz|mv)\s+(-?\d+)\s*(?:mhz|mv)", line, re.I)
            if m:
                out["range"][m.group(1).upper()] = (int(m.group(2)), int(m.group(3)))
    return out


class _GpuOd(Knob):
    """One amdgpu OverDrive value. Live only unless it is put in a boot profile; a
    reboot always brings back the card's stock table. The kernel re-checks every value
    against its own limits; this knob refuses anything outside the range it prints."""
    key, cmd, rng, unit = "", "", "", ""

    def _od(self, t):
        return od_table(t)

    def targets(self, ctx):
        out = []
        for a, d in _amd_gpus(ctx):
            od = self._od(a)
            if od and self._has(od):
                out.append(dict(target=a, ident=ident(d), label=f"GPU {a}"))
        return out

    def _has(self, od):
        return bool(od[self.key]) and self.rng in od["range"]

    def _limits(self, t):
        od = self._od(t)
        return od["range"].get(self.rng) if od else None

    def read(self, ctx, t):
        od = self._od(t)
        if not od or not od[self.key]:
            return None
        return str(od[self.key][max(od[self.key])])       # the top of the min/max pair

    def choices(self, ctx, t):
        lo, hi = self._limits(t) or (None, None)
        return dict(kind="range", min=lo, max=hi, unit=self.unit)

    def valid(self, ctx, t, v):
        lim = self._limits(t)
        return bool(lim and re.fullmatch(r"\d{2,5}", str(v)) and lim[0] <= int(v) <= lim[1])

    def write(self, ctx, t, v):
        od = self._od(t)
        f = f"{SYS}/bus/pci/devices/{t}/pp_od_clk_voltage"
        _w(f, f"{self.cmd} {max(od[self.key])} {int(v)}")
        _w(f, "c")


class GpuOdSclk(_GpuOd):
    name, label, order = "gpu.od_sclk", "GPU max core clock (amdgpu OverDrive, MHz)", 43
    key, cmd, rng, unit = "sclk", "s", "SCLK", "MHz"


class GpuOdMclk(_GpuOd):
    name, label, order = "gpu.od_mclk", "GPU max memory clock (amdgpu OverDrive, MHz)", 44
    key, cmd, rng, unit = "mclk", "m", "MCLK", "MHz"


class GpuOdVoltage(_GpuOd):
    """Core voltage offset in mV. Undervolt only (never above 0): RDNA3 firmware
    refuses a positive offset anyway, and RDNA2 prints no range to check one against."""
    name, label, order = "gpu.od_voltage", "GPU core voltage offset (amdgpu OverDrive, mV)", 45
    unit = "mV"
    FLOOR = -300                     # when the kernel prints no range (RDNA2)

    def _has(self, od):
        return od["vo"] is not None

    def _limits(self, t):
        od = self._od(t)
        if not od or od["vo"] is None:
            return None
        lo, hi = od["range"].get("VDDGFX_OFFSET", (self.FLOOR, 0))
        return lo, min(hi, 0)

    def read(self, ctx, t):
        od = self._od(t)
        return None if not od or od["vo"] is None else str(od["vo"])

    def valid(self, ctx, t, v):
        lim = self._limits(t)
        return bool(lim and re.fullmatch(r"-?\d{1,4}", str(v)) and lim[0] <= int(v) <= lim[1])

    def write(self, ctx, t, v):
        f = f"{SYS}/bus/pci/devices/{t}/pp_od_clk_voltage"
        _w(f, f"vo {int(v)}")
        _w(f, "c")


# ============================================================================
# VBIOS image (read only)
# ============================================================================
VBIOS_MAX = 16 * 1024 * 1024


def vbios(target, ctx=None):
    """The VBIOS image of one GPU this helper enumerated, base64 in JSON.

    Read only. amdgpu's debugfs copy is the image the driver actually loaded; the
    PCI ROM BAR (sysfs rom: enable, read, disable) is the fallback for every other
    driver. The flash chip is never written here or anywhere else in LexiPanel."""
    import base64
    ctx = ctx or Ctx()
    d = ctx.devs().get(str(target or ""))
    if not d or _cls_kind(d["cls"]) != "gpu":
        raise ValueError(f"{str(target)[:40]!r} is not a GPU on this box")
    a = d["addr"]
    tried, data = [], None
    names = [a]
    try:
        names += sorted(n for n in os.listdir(f"{SYS}/bus/pci/devices/{a}/drm") if n.startswith("card"))
    except OSError:
        pass
    for n in names:
        p = f"{SYS}/kernel/debug/dri/{n[4:] if n.startswith('card') else n}/amdgpu_vbios"
        if os.path.exists(p):
            tried.append(p)
            try:
                with open(p, "rb") as f:
                    data = f.read(VBIOS_MAX + 1)
                if data:
                    break
            except OSError:
                data = None
    if not data:
        rom = f"{SYS}/bus/pci/devices/{a}/rom"
        tried.append(rom)
        if os.path.exists(rom):
            try:
                _w(rom, "1")
                with open(rom, "rb") as f:
                    data = f.read(VBIOS_MAX + 1)
            except OSError as e:
                raise ValueError(f"reading {rom} failed: {e.strerror or e}")
            finally:
                try:
                    _w(rom, "0")
                except OSError:
                    pass
    if not data:
        raise ValueError("no readable VBIOS image (tried " + ", ".join(tried) + ")")
    if len(data) > VBIOS_MAX:
        raise ValueError("VBIOS image larger than 16 MiB: refusing")
    return dict(target=a, ident=ident(d), source=tried[-1], size=len(data),
                data=base64.b64encode(data).decode())


def _effective(conf, key):
    rc, out = _run([BIN["analyze"], "cat-config", conf], timeout=10)
    val = None
    for line in out.splitlines() if rc == 0 else []:
        m = re.match(rf"\s*{key}\s*=\s*(\S*)", line)
        if m:
            val = m.group(1)
    return val


class Watchdog(Knob):
    """systemd pets the hardware watchdog; a frozen box reboots itself."""
    name, label, order = "recovery.watchdog", "Hardware watchdog (systemd)", 90

    def _devs(self):
        base = f"{SYS}/class/watchdog"
        try:
            return sorted(w for w in os.listdir(base) if re.fullmatch(r"watchdog\d+", w))
        except OSError:
            return []

    def _module(self):
        for w in self._devs():
            drv = os.path.realpath(f"{SYS}/class/watchdog/{w}/device/driver")
            if os.path.exists(drv):
                return os.path.basename(drv)
        vendor = re.search(r"vendor_id\s*:\s*(\S+)", _r(f"{PROC}/cpuinfo") or "")
        return (WATCHDOG_MODULES.get(vendor.group(1) if vendor else "", []) or [None])[0]

    def targets(self, ctx):
        return [dict(target="system", label=(self._module() or "no watchdog driver found"))]

    def read(self, ctx, t):
        rc, out = _run([BIN["systemctl"], "show", "-p", "RuntimeWatchdogUSec", "--value"], timeout=10)
        v = out.strip() if rc == 0 else ""
        if not v:
            return None
        if v in ("0", "infinity"):
            return "off"
        # Configured but no watchdog device means nothing is watching (Ubuntu deny-lists
        # iTCO_wdt, so modules-load.d skips it at boot): say "off", so applying "on" at boot
        # loads the driver explicitly and re-executes systemd to arm it.
        return "on" if self._devs() else "off"

    def choices(self, ctx, t):
        return ["on", "off"] if self._module() else ["off"]

    def write(self, ctx, t, v):
        if v == "on":
            mod = self._module()
            if not self._devs():
                rc, out = _run([BIN["modprobe"], mod])
                if rc != 0 or not self._devs():
                    raise OSError(f"loading {mod} gave no /dev/watchdog: {out.strip()[-200:]}")
            os.makedirs(os.path.dirname(WD_DROPIN), exist_ok=True)
            _w(WD_DROPIN, "# written by LexiPanel power options\n[Manager]\n"
                          "RuntimeWatchdogSec=30s\nRebootWatchdogSec=5min\n")
            os.makedirs(os.path.dirname(WD_MODLOAD), exist_ok=True)
            _w(WD_MODLOAD, f"# written by LexiPanel power options\n{mod}\n")
        else:
            for f in (WD_DROPIN, WD_MODLOAD):
                if os.path.exists(f):
                    os.remove(f)
        rc, out = _run([BIN["systemctl"], "daemon-reexec"], timeout=60)
        if rc != 0:
            raise OSError(out.strip()[-200:])
        time.sleep(1)
        if v == "off" and self.read(ctx, t) == "on":
            raise OSError("still on: another drop-in in /etc/systemd/system.conf.d sets it; "
                          "LexiPanel only removes its own")


class JournalSync(Knob):
    """How often journald forces the log to disk. On a box that dies without warning,
    the default (5 min) loses the last minutes before the stop."""
    name, label, order = "recovery.journal_sync", "Journal sync to disk", 91

    def targets(self, ctx):
        return [dict(target="journald", label="systemd-journald")] if os.path.exists(BIN["analyze"]) else []

    def read(self, ctx, t):
        v = _effective("systemd/journald.conf", "SyncIntervalSec")
        return v or "default"

    def choices(self, ctx, t):
        return ["5s", "default"]

    def write(self, ctx, t, v):
        if v == "5s":
            os.makedirs(os.path.dirname(JD_DROPIN), exist_ok=True)
            _w(JD_DROPIN, "# written by LexiPanel power options\n[Journal]\nSyncIntervalSec=5s\n")
        elif os.path.exists(JD_DROPIN):
            os.remove(JD_DROPIN)
        rc, out = _run([BIN["systemctl"], "restart", "systemd-journald"], timeout=60)
        if rc != 0:
            raise OSError(out.strip()[-200:])
        if v == "default" and self.read(ctx, t) != "default":
            raise OSError("still set: another journald drop-in sets SyncIntervalSec; "
                          "LexiPanel only removes its own")

    def same(self, ctx, t, want, got):
        return str(want) == str(got)


KNOBS = {k.name: k for k in (CpuGovernor(), CpuEpp(), CpuIdle(), CpuTurbo(), AspmPolicy(),
                             LinkAspm(), NvmeApst(), RuntimePm(), GpuPerfLevel(), GpuFan(),
                             GpuPowerCap(), GpuOdSclk(), GpuOdMclk(), GpuOdVoltage(),
                             Watchdog(), JournalSync())}


class Ctx:
    """Per-call cache: one PCI scan, one nvidia-smi query."""
    def __init__(self):
        self.cache = {}

    def devs(self):
        if "devs" not in self.cache:
            self.cache["devs"] = pci_devices()
        return self.cache["devs"]

    def links(self):
        if "links" not in self.cache:
            self.cache["links"] = pcie_links(self.devs())
        return self.cache["links"]

    def baseline_raw(self):
        try:
            with open(BASELINE_FILE) as f:
                return json.load(f).get("raw") or {}
        except (OSError, ValueError):
            return {}


# ============================================================================
# table / apply / persist / boot
# ============================================================================
def table(ctx=None):
    ctx = ctx or Ctx()
    rows = []
    for k in sorted(KNOBS.values(), key=lambda k: k.order):
        try:
            ts = k.targets(ctx)
        except Exception as e:           # a broken knob must not hide the others
            rows.append(dict(knob=k.name, label=k.label, target=None, error=str(e)[:200]))
            continue
        for t in ts:
            try:
                val, ch = k.read(ctx, t["target"]), k.choices(ctx, t["target"])
            except Exception as e:
                val, ch = None, []
                t = dict(t, error=str(e)[:200])
            rows.append(dict(t, knob=k.name, label=k.label, device=t.get("label"), value=val, choices=ch))
    return rows


def _resolve(ctx, knob, target, idt):
    ts = knob.targets(ctx)
    hit = next((t for t in ts if t["target"] == target), None)
    if hit and (not idt or not hit.get("ident") or hit.get("ident") == idt):
        return hit["target"]
    if idt:                               # the card moved: same model, new address
        same = [t for t in ts if t.get("ident") == idt]
        if len(same) == 1:
            return same[0]["target"]
    return None


def _check_settings(settings, need_targets=True, ctx=None):
    ctx = ctx or Ctx()
    if not isinstance(settings, list) or len(settings) > 400:
        raise ValueError("settings: a list of at most 400")
    out = []
    for s in settings:
        if not isinstance(s, dict):
            raise ValueError("each setting is an object")
        k = KNOBS.get(str(s.get("knob")))
        if not k:
            raise ValueError(f"unknown setting {str(s.get('knob'))[:40]!r}")
        target, idt, v = str(s.get("target") or ""), s.get("ident"), str(s.get("value") or "")
        if not re.fullmatch(r"[\w:.+-]{1,64}", target) or len(v) > 200 or \
                (idt is not None and not re.fullmatch(r"[0-9a-f:]{1,40}", str(idt))):
            raise ValueError(f"{k.name}: bad target, ident or value")
        rt = _resolve(ctx, k, target, idt)
        if rt is None:
            if need_targets:
                out.append(dict(knob=k.name, target=target, ident=idt, value=v, missing=True))
                continue
        elif not k.valid(ctx, rt, v):
            raise ValueError(f"{k.name} {rt}: {v!r} is not an allowed value here")
        out.append(dict(knob=k.name, target=target, ident=idt, value=v, resolved=rt))
    return out


def apply(settings, ctx=None):
    ctx = ctx or Ctx()
    checked = _check_settings(settings, ctx=ctx)
    results = []
    for s in sorted(checked, key=lambda s: KNOBS[s["knob"]].order):
        k = KNOBS[s["knob"]]
        if s.get("missing"):
            results.append(dict(s, ok=False, msg="no such device right now"))
            continue
        t = s["resolved"]
        before = k.read(ctx, t)
        if k.same(ctx, t, s["value"], before):
            results.append(dict(s, before=before, after=before, ok=True, msg="already set"))
            continue
        try:
            k.write(ctx, t, s["value"])
            after = k.read(ctx, t)
            ok = k.same(ctx, t, s["value"], after)
            results.append(dict(s, before=before, after=after, ok=ok,
                                msg="" if ok else f"reads back {after!r}"))
        except (OSError, ValueError, KeyError, TypeError) as e:
            results.append(dict(s, before=before, after=k.read(ctx, t), ok=False,
                                msg=str(getattr(e, "strerror", None) or e)[:300]))
    return results


def persist(prof, ctx=None):
    name = str(prof.get("name") or "")[:100]
    checked = _check_settings(prof.get("settings") or [], ctx=ctx)
    clean = [{k: s[k] for k in ("knob", "target", "ident", "value")} for s in checked]
    os.makedirs(STATE_DIR, mode=0o755, exist_ok=True)
    tmp = PROFILE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(name=name, id=str(prof.get("id") or "")[:120], settings=clean,
                       saved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f, indent=1)
    os.chmod(tmp, 0o644)
    os.replace(tmp, PROFILE_FILE)
    return dict(ok=True, file=PROFILE_FILE, settings=len(clean))


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def snapshot(ctx=None):
    ctx = ctx or Ctx()
    rows = table(ctx)
    raw = {}
    for t in NvmeApst().targets(ctx):
        raw[f"nvme.apst.us:{t['target']}"] = _r(f"{SYS}/class/nvme/{t['target']}/power/pm_qos_latency_tolerance_us")
    return dict(boot_id=boot_id(), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                settings=[dict(knob=r["knob"], target=r["target"], ident=r.get("ident"), value=r["value"])
                          for r in rows if r.get("target") and r.get("value") is not None],
                raw=raw)


def boot(wait_s=90):
    os.makedirs(RUN_DIR, mode=0o755, exist_ok=True)
    base = _load(BASELINE_FILE)
    if not base or base.get("boot_id") != boot_id():
        with open(BASELINE_FILE, "w") as f:
            json.dump(snapshot(), f, indent=1)
    prof = _load(PROFILE_FILE)
    log = dict(boot_id=boot_id(), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               profile=(prof or {}).get("name"), profile_id=(prof or {}).get("id"), results=[])
    pending = [s for s in (prof or {}).get("settings") or []
               if isinstance(s, dict) and s.get("knob") in KNOBS]
    deadline = time.time() + wait_s
    while pending:
        ctx = Ctx()
        ready = [s for s in pending
                 if _resolve(ctx, KNOBS[s["knob"]], str(s.get("target") or ""), s.get("ident"))]
        pending = [s for s in pending if s not in ready]
        for s in sorted(ready, key=lambda s: KNOBS[s["knob"]].order):
            try:                          # one bad value must not stop the rest
                log["results"] += apply([s], ctx)
            except ValueError as e:
                log["results"].append(dict(s, ok=False, msg=str(e)))
        if not pending or time.time() > deadline:
            break
        time.sleep(3)                     # GPU hwmon / OverDrive / nvidia-smi come up late
    log["missing"] = pending
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(BOOTLOG_FILE, "w") as f:
        json.dump(log, f, indent=1)
    for r in log["results"]:
        print(f"lexipanel-power: {r.get('knob')} {r.get('resolved') or r.get('target')} -> "
              f"{r.get('value')}: {'ok' if r.get('ok') else 'FAILED ' + str(r.get('msg'))}")
    return log


def pstore_count():
    try:
        return len(os.listdir(f"{SYS}/fs/pstore"))
    except OSError:
        return None


def status():
    ctx = Ctx()
    return dict(version=VERSION, root=is_root(), boot_id=boot_id(), table=table(ctx),
                links=ctx.links(), persisted=_load(PROFILE_FILE), baseline=_load(BASELINE_FILE),
                bootlog=_load(BOOTLOG_FILE), pstore=pstore_count(),
                watchdog_devices=[dict(dev=w, identity=_r(f"{SYS}/class/watchdog/{w}/identity"),
                                       state=_r(f"{SYS}/class/watchdog/{w}/state"),
                                       timeout=_int(_r(f"{SYS}/class/watchdog/{w}/timeout")))
                                  for w in Watchdog()._devs()])


def _stdin_json():
    data = sys.stdin.read(MAX_INPUT + 1)
    if len(data) > MAX_INPUT:
        raise ValueError("input too large")
    return json.loads(data or "{}")


def main(argv):
    verb = argv[1] if len(argv) > 1 else "status"
    try:
        if verb == "status":
            res = status()
        elif verb in ("apply", "persist", "unpersist", "boot", "vbios") and not is_root():
            raise ValueError(f"{verb} needs root (sudo)")
        elif verb == "vbios":
            res = vbios((_stdin_json() or {}).get("target"))
        elif verb == "apply":
            res = dict(results=apply((_stdin_json() or {}).get("settings") or []))
        elif verb == "persist":
            res = persist(_stdin_json() or {})
        elif verb == "unpersist":
            if os.path.exists(PROFILE_FILE):
                os.remove(PROFILE_FILE)
            res = dict(ok=True)
        elif verb == "boot":
            res = boot()
        else:
            raise ValueError(f"unknown verb {verb!r}: status, apply, persist, unpersist, vbios, boot")
    except ValueError as e:
        print(json.dumps(dict(error=str(e))))
        return 2
    print(json.dumps(res, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
