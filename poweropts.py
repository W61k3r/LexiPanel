#!/usr/bin/env python3
"""
Power options (added 2026-09-24): the equipment that decides whether this box stays up,
and the power settings that keep it up.

  equipment   CPU, GPUs, disks, network, PCIe links, UPS, PSU, watchdog, crash capture.
              Found by PCI/USB id, so it works on any box and survives cards moving.
  settings    one table of every power setting the root helper knows (power/lexipanel_power.py):
              current value, this boot's firmware/kernel default, the boot profile's value.
  profiles    named sets of settings. Built-ins: "Stable performance" (no deep CPU idle, no
              link or disk power saving, watchdog on: the set that ended inf01's crashes on
              2026-09-24) and "Boot defaults" (what firmware and kernel chose at this boot).
              "Capture live" saves whatever is set now.
  apply       through the helper (sudo, 5 verbs). Every change goes to an audit log with the
              value before and after. A boot profile is applied by the helper's boot unit
              before the inference servers start; drift is shown and can be re-applied.
  stability   boots, their uptime and how they ended, per profile: evidence, not hope.
  power       UPS through NUT (local or on another machine), the PSU and UPS ratings you
              enter, the measured GPU peaks, and a heartbeat to machines sharing the UPS so
              a shared power cut can be told apart from a fault of this box.

Unprivileged by design: without the helper this module still shows everything it can read.
"""
import calendar, hashlib, importlib.util, json, os, re, shutil, subprocess, threading, time
from pathlib import Path

P = None
HELPER = "/usr/local/sbin/lexipanel-power"
SRC = Path(__file__).resolve().parent / "power" / "lexipanel_power.py"
INSTALL = Path(__file__).resolve().parent / "power" / "install-power.sh"
_lock = threading.RLock()
_cache = {}
_lp_mod = None
UPS_USB_VENDORS = {"051d": "APC", "0764": "CyberPower", "0463": "Eaton/MGE", "09ae": "Tripp Lite",
                   "10af": "Liebert/Vertiv", "06da": "Phoenixtec", "0665": "Cypress-based UPS",
                   "0925": "Richcomm UPS", "2b2d": "Ablerex", "0001": "Fry's/generic UPS"}
_TS = re.compile(r'\{"ts":\s*"([^"]+)"')
_HOST = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$|^\[?[0-9a-fA-F:]{2,39}\]?$")


def bind(panel_module):
    global P
    P = panel_module


def _dir(*sub):
    d = P.PANEL.joinpath("power-state", *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def _atomic(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def _cached(key, ttl, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > time.time() - ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def _drop(*keys):
    with _lock:
        for k in keys:
            _cache.pop(k, None)


# ============================================================================
# the helper
# ============================================================================
def lp():
    """The helper's code, imported unprivileged for the read-only view."""
    global _lp_mod
    m = SRC.stat().st_mtime_ns
    if _lp_mod is None or getattr(_lp_mod, "_mtime", None) != m:
        spec = importlib.util.spec_from_file_location("lexipanel_power", SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._mtime = m
        _lp_mod = mod
    return _lp_mod


def _sha(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def helper_info():
    def get():
        inst = os.path.exists(HELPER)
        ok, err = False, None
        if inst:
            try:
                r = subprocess.run(["sudo", "-n", HELPER, "status"], capture_output=True, text=True,
                                   timeout=30)
                ok = r.returncode == 0 and r.stdout.lstrip().startswith("{")
                err = None if ok else (r.stderr or r.stdout).strip()[-200:]
            except (OSError, subprocess.SubprocessError) as e:
                err = str(e)
        return dict(installed=inst, sudo_ok=ok, error=err,
                    current=inst and _sha(HELPER) == _sha(SRC),
                    install_cmd=f"sudo bash {INSTALL}", remove_cmd=f"sudo bash {INSTALL} --remove",
                    unit=os.path.exists("/etc/systemd/system/lexipanel-power.service"))
    return _cached("helper", 60, get)


def _sudo(verb, payload=None, timeout=180):
    try:
        r = subprocess.run(["sudo", "-n", HELPER, verb], capture_output=True, text=True, timeout=timeout,
                           input=json.dumps(payload) if payload is not None else "")
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"the power helper did not run: {e}")
    try:
        out = json.loads(r.stdout or "{}")
    except ValueError:
        raise ValueError(f"the power helper failed: {(r.stderr or r.stdout).strip()[-300:]}")
    if r.returncode != 0 or out.get("error"):
        raise ValueError(out.get("error") or (r.stderr or "").strip()[-300:] or f"exit {r.returncode}")
    return out


def view(fresh=False):
    """The helper's status: as root when the helper is installed (link states need it),
    otherwise the same code run unprivileged."""
    if fresh:
        _drop("view")

    def get():
        h = helper_info()
        if h["installed"] and h["sudo_ok"]:
            try:
                return _sudo("status", timeout=60)
            except ValueError as e:
                v = lp().status()
                v["helper_error"] = str(e)
                return v
        return lp().status()
    return _cached("view", 5, get)


# ============================================================================
# equipment
# ============================================================================
def _r(p):
    try:
        return Path(p).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _pci_names():
    def get():
        out = {}
        try:
            r = subprocess.run(["lspci", "-D", "-mm"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                f = re.findall(r'"([^"]*)"|(\S+)', line)
                vals = [a or b for a, b in f]
                if len(vals) >= 4:
                    out[vals[0]] = f"{vals[2]} {vals[3]}".strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return out
    return _cached("pcinames", 600, get)


def _root_disk_pci():
    try:
        st = os.stat("/")
        real = os.path.realpath(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
        m = re.findall(r"/(\d{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d)", real)
        return m[-1] if m else None
    except OSError:
        return None


def _usb_ups():
    out = []
    base = Path("/sys/bus/usb/devices")
    for d in sorted(base.glob("*")) if base.is_dir() else []:
        v = _r(d / "idVendor")
        if v in UPS_USB_VENDORS and v != "0001":
            out.append(dict(usb=d.name, vendor=UPS_USB_VENDORS[v], product=_r(d / "product"),
                            id=f"{v}:{_r(d / 'idProduct')}"))
    return out


def _rigmon_peaks(hours=24):
    """Per-GPU board power over the last `hours` from rigmon's 5 s log."""
    def get():
        logs = sorted((P.HOME / "rigmon" / "logs").glob("rigmon-*.jsonl"))[-2:]
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600))
        vals = {}
        for f in logs:
            try:
                with open(f) as fh:
                    for line in fh:
                        m = _TS.match(line)             # cheap skip before parsing the JSON
                        if not m or m.group(1) < cutoff:
                            continue
                        try:
                            d = json.loads(line)
                        except ValueError:
                            continue
                        for k, h in (d.get("hwmon") or {}).items():
                            if k.startswith("amdgpu@") and h.get("power:PPT") is not None:
                                vals.setdefault(k.split("@", 1)[1], []).append(float(h["power:PPT"]))
            except OSError:
                continue
        out = {}
        for pci, xs in vals.items():
            xs.sort()
            out[pci] = dict(samples=len(xs), max_w=round(xs[-1]), p99_w=round(xs[int(len(xs) * 0.99) - 1]),
                            median_w=round(xs[len(xs) // 2]),
                            over_cap_pct=None)
        return out
    return _cached("peaks", 300, get)


def _cpu():
    info = _r("/proc/cpuinfo") or ""
    m = re.search(r"model name\s*:\s*(.+)", info)
    v = re.search(r"vendor_id\s*:\s*(\S+)", info)
    pl = {}
    for z in sorted(Path("/sys/class/powercap").glob("*-rapl:[0-9]")) if Path("/sys/class/powercap").is_dir() else []:
        for i in range(3):
            n = _r(z / f"constraint_{i}_name")
            w = _r(z / f"constraint_{i}_power_limit_uw")
            if n and w and w.isdigit():
                pl[n] = int(w) // 1000000
    return dict(model=m.group(1).strip() if m else None, vendor=v.group(1) if v else None,
                threads=os.cpu_count(), driver=_r("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
                power_limits_w=pl or None)


def equipment(v=None):
    v = v or view()
    names = _pci_names()
    root_pci = _root_disk_pci()
    peaks = _rigmon_peaks()
    tab = v.get("table") or []
    val = lambda knob, target: next((r.get("value") for r in tab if r.get("knob") == knob
                                     and r.get("target") == target), None)
    items = []
    c = _cpu()
    items.append(dict(group="CPU", name=c["model"] or "CPU", id="cpu", detail=dict(
        driver=c["driver"], threads=c["threads"], power_limits_w=c["power_limits_w"],
        governor=val("cpu.governor", "all"), deepest_idle=val("cpu.idle", "all"), turbo=val("cpu.turbo", "all")),
        notes=(["Deep idle states are allowed. Waking from them is a large, fast power step; on boxes "
                "that die when load starts or while idle, limiting them is the first test."]
               if val("cpu.idle", "all") == "all" else [])))
    for g in P.gpu_devices(probe=False):
        if g.get("pci") in (None, "cpu"):
            continue
        pk = peaks.get(g["pci"]) or {}
        cap = val("gpu.power_cap", g["pci"])
        notes = []
        if pk.get("max_w") and cap not in (None, "unset") and pk["max_w"] > int(cap) * 1.15:
            notes.append(f"Peaks reach {pk['max_w']} W against a {cap} W cap: a cap limits the average, "
                         "not the sub-second spikes a PSU or UPS has to ride through.")
        items.append(dict(group="GPU", name=names.get(g["pci"]) or g.get("name"), id=g["pci"], detail=dict(
            driver=g.get("driver"), power_cap_w=cap, peak_24h_w=pk.get("max_w"), p99_24h_w=pk.get("p99_w"),
            fan=val("gpu.fan", g["pci"]), perf_level=val("gpu.perf_level", g["pci"]),
            runtime_pm=val("pci.runpm", g["pci"])), notes=notes))
    for t in [r for r in tab if r.get("knob") == "nvme.apst"]:
        pci = t.get("pci")
        is_root = bool(root_pci and pci and root_pci == pci)
        link = next((r for r in tab if r.get("knob") == "pcie.aspm" and r.get("target") == pci), None)
        notes = []
        if is_root and (t.get("value") == "on" or (link and link.get("value") not in (None, "off"))):
            notes.append("Power saving is on for the disk the logs are written to. If it drops off the "
                         "bus at idle, the box looks dead and nothing can be logged about it.")
        items.append(dict(group="Storage", name=(names.get(pci) or t.get("label")), id=t["target"], detail=dict(
            pci=pci, root_disk=is_root, apst=t.get("value"), link_aspm=link.get("value") if link else None),
            notes=notes))
    for l in v.get("links") or []:
        if l["kind"] == "network":
            items.append(dict(group="Network", name=names.get(l["target"]) or l["target"], id=l["target"],
                              detail=dict(driver=l.get("driver"), link_aspm=val("pcie.aspm", l["target"])),
                              notes=[]))
    ups = ups_status()
    for u in ups.get("ups") or []:
        items.append(dict(group="Power", name=f"UPS {u['name']}@{u['host']}", id=f"ups:{u['name']}@{u['host']}",
                          detail={k: u.get(k) for k in ("status", "charge_pct", "load_pct", "runtime_s",
                                                         "nominal_w", "input_v", "model", "error")},
                          notes=[x for x in [u.get("error")] if x]))
    for u in ups.get("usb") or []:
        items.append(dict(group="Power", name=f"{u['vendor']} UPS on USB {u['usb']}", id=f"usb:{u['usb']}",
                          detail=dict(product=u["product"], id=u["id"]),
                          notes=[] if ups.get("nut") else ["Plugged in, but NUT is not installed, so the "
                                                          "panel cannot read it: sudo apt install nut"]))
    s = settings()
    if not ups.get("ups") and not ups.get("usb"):
        items.append(dict(group="Power", name="UPS", id="ups:none", detail=dict(
            rating_w=s.get("ups_w"), rating_va=s.get("ups_va"), other_loads=len(s.get("other_loads") or [])),
            notes=["No UPS is visible to this box (no USB link, no NUT server configured). An overloaded "
                   "UPS cuts power to everything on it without warning or a log line."]))
    items.append(dict(group="Power", name="Power supply", id="psu", detail=dict(rating_w=s.get("psu_w")),
                      notes=[] if s.get("psu_w") else ["Its rating is not reported by the firmware: enter it "
                                                       "below for the power budget."]))
    wd = v.get("watchdog_devices") or []
    items.append(dict(group="Recovery", name="Hardware watchdog", id="watchdog", detail=dict(
        devices=", ".join(f"{w['dev']} {w['identity']} ({w['state']}, {w['timeout']} s)" for w in wd) or "none",
        systemd=val("recovery.watchdog", "system")),
        notes=[] if val("recovery.watchdog", "system") == "on" else
        ["Off: a box that freezes stays frozen until someone presses the button."]))
    kd = _r("/sys/kernel/kexec_crash_loaded")
    lock = {k: _r(f"/proc/sys/kernel/{k}") for k in ("softlockup_panic", "hardlockup_panic", "panic")}
    ras = subprocess.run(["systemctl", "is-active", "rasdaemon"], capture_output=True, text=True).stdout.strip()
    rig = sorted((P.HOME / "rigmon" / "logs").glob("rigmon-*.jsonl"))
    rig_age = round(time.time() - rig[-1].stat().st_mtime) if rig else None
    items.append(dict(group="Recovery", name="Crash capture", id="capture", detail=dict(
        kdump_loaded=kd == "1", lockup_panic=lock, journal_sync=val("recovery.journal_sync", "journald"),
        rasdaemon=ras or "absent", pstore_records=v.get("pstore"),
        telemetry=(f"rigmon, last sample {rig_age} s ago" if rig_age is not None else "none")),
        notes=(["The journal is synced every 5 minutes by default: a hard stop loses the minutes before it."]
               if val("recovery.journal_sync", "journald") == "default" else [])))
    return items


# ============================================================================
# settings file (ratings, UPS, peers, auto re-apply)
# ============================================================================
def settings():
    return _read(_dir() / "settings.json", {}) or {}


def save_settings(body):
    s = settings()

    def num(k, lo, hi):
        v = body.get(k)
        if v in (None, ""):
            s.pop(k, None)
            return
        try:
            v = int(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"{k}: a number")
        if not lo <= v <= hi:
            raise ValueError(f"{k}: {lo}-{hi}")
        s[k] = v

    for k, lo, hi in (("psu_w", 100, 5000), ("ups_w", 100, 20000), ("ups_va", 100, 30000),
                      ("platform_w", 0, 1000)):
        if k in body:
            num(k, lo, hi)

    def rows(key, fields, maxn=16):
        if key not in body:
            return
        out = []
        for r in (body.get(key) or [])[:maxn]:
            row = {}
            for f, kind in fields:
                x = str((r or {}).get(f) or "").strip()
                if kind == "host":
                    if not _HOST.match(x):
                        raise ValueError(f"{key}: {x[:60]!r} is not a host name or address")
                elif kind == "w":
                    if not re.fullmatch(r"\d{1,5}", x):
                        raise ValueError(f"{key}: watts as a whole number")
                    x = int(x)
                elif kind == "name":
                    if not re.fullmatch(r"[\w .+-]{1,40}", x):
                        raise ValueError(f"{key}: names are letters, digits, space and ._+-")
                row[f] = x
            out.append(row)
        s[key] = out

    rows("other_loads", (("name", "name"), ("w", "w")))
    rows("ups", (("name", "name"), ("host", "host")))
    rows("peers", (("name", "name"), ("host", "host")))
    if "auto_reapply" in body:
        s["auto_reapply"] = bool(body.get("auto_reapply"))
    _atomic(_dir() / "settings.json", s)
    _drop("ups")
    return s


# ============================================================================
# profiles
# ============================================================================
def _builtins(v):
    tab = v.get("table") or []
    rows = lambda knob: [r for r in tab if r.get("knob") == knob and r.get("target")]
    st = []

    def add(knob, value, only_if_choice=True):
        for r in rows(knob):
            ch = r.get("choices")
            if only_if_choice and isinstance(ch, list) and value not in ch:
                continue
            st.append(dict(knob=knob, target=r["target"], ident=r.get("ident"), value=value))

    add("cpu.governor", "performance")
    for r in rows("cpu.idle"):
        ch = [c for c in (r.get("choices") or []) if c != "all"]
        if ch:
            st.append(dict(knob="cpu.idle", target="all", value=ch[0]))
    for r in rows("pcie.aspm"):
        st.append(dict(knob="pcie.aspm", target=r["target"], ident=r.get("ident"), value="off"))
    add("nvme.apst", "off")
    add("recovery.watchdog", "on")
    add("recovery.journal_sync", "5s")
    out = [dict(id="builtin-stable", builtin=True, name="Stable performance",
                description="CPU on the performance governor with only the shallowest idle state, no PCIe "
                            "link power saving, no NVMe power saving, the hardware watchdog armed and the "
                            "journal synced every 5 s. On inf01 this set ended the silent stops on "
                            "2026-09-24. GPU caps and fan curves are per card: add them with Capture.",
                settings=st)]
    base = v.get("baseline")
    if base and base.get("boot_id") == v.get("boot_id"):
        out.append(dict(id="builtin-defaults", builtin=True, name="Boot defaults",
                        description=f"What firmware and kernel chose at the start of this boot "
                                    f"(recorded {base.get('at')}), before any profile was applied.",
                        settings=[s for s in base.get("settings") or [] if s.get("value") not in (None, "mixed", "custom")]))
    return out


def profiles(v=None):
    v = v or view()
    saved = [p for p in (_read(f) for f in sorted(_dir("profiles").glob("*.json"), reverse=True)) if p]
    return _builtins(v) + saved


def get_profile(pid, v=None):
    p = next((x for x in profiles(v) if x["id"] == pid), None)
    if not p:
        raise ValueError(f"no profile {pid}")
    return p


def _clean_settings(st):
    if not isinstance(st, list) or not st or len(st) > 400:
        raise ValueError("a profile needs 1-400 settings")
    knobs = lp().KNOBS
    out = []
    for s in st:
        k, t, v = str(s.get("knob") or ""), str(s.get("target") or ""), str(s.get("value") or "")
        if k not in knobs or not re.fullmatch(r"[\w:.+-]{1,64}", t) or not v or len(v) > 200:
            raise ValueError(f"bad setting {k} {t}")
        idt = s.get("ident")
        if idt is not None and not re.fullmatch(r"[0-9a-f:]{1,40}", str(idt)):
            raise ValueError("bad device id")
        out.append(dict(knob=k, target=t, ident=idt, value=v))
    return out


def save_profile(body):
    name = str(body.get("name") or "").strip()
    if not re.fullmatch(r"[\w .,:()+-]{1,60}", name):
        raise ValueError("name: 1-60 letters, digits, spaces and .,:()+-")
    st = _clean_settings(body.get("settings"))
    pid = str(body.get("id") or "")
    if pid.startswith("builtin-"):
        pid = ""
    if pid and not re.fullmatch(r"[\w-]{1,80}", pid):
        raise ValueError("bad id")
    if not pid:
        pid = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + re.sub(r"[^\w]+", "-", name).strip("-").lower()[:40]
    p = dict(id=pid, name=name, description=str(body.get("description") or "")[:500], settings=st,
             source=str(body.get("source") or "custom")[:20], saved=_now())
    _atomic(_dir("profiles") / f"{pid}.json", p)
    return p


def capture(body):
    v = view(fresh=True)
    pick = body.get("rows")                    # optional [[knob, target], ...]
    want = {(a, b) for a, b in pick} if isinstance(pick, list) else None
    st, skipped = [], 0
    for r in v.get("table") or []:
        if not r.get("target") or (want is not None and (r["knob"], r["target"]) not in want):
            continue
        val = r.get("value")
        if val in (None, "mixed", "custom", "unset") or str(val).startswith("mixed"):
            skipped += 1
            continue
        st.append(dict(knob=r["knob"], target=r["target"], ident=r.get("ident"), value=str(val)))
    if not st:
        raise ValueError("nothing readable to capture" + ("" if v.get("root") else
                         " (install the helper: PCIe link states need it)"))
    p = save_profile(dict(name=body.get("name") or f"Captured {time.strftime('%Y-%m-%d %H:%M')}",
                          description=f"Captured from the live settings at {_now()}"
                                      + (f"; {skipped} unreadable settings left out" if skipped else "")
                                      + ("" if v.get("root") else "; PCIe links need the helper to be read"),
                          settings=st, source="captured"))
    return p


def delete_profile(body):
    pid = str(body.get("id") or "")
    if pid.startswith("builtin-") or not re.fullmatch(r"[\w-]{1,80}", pid):
        raise ValueError("built-in profiles cannot be deleted")
    f = _dir("profiles") / f"{pid}.json"
    if not f.exists():
        raise ValueError("no such profile")
    if (settings().get("boot_profile") or {}).get("id") == pid:
        raise ValueError("this is the boot profile: choose another boot profile first")
    f.unlink()
    return dict(ok=True)


# ============================================================================
# apply / boot profile / drift / audit
# ============================================================================
def _audit(entry):
    entry = dict(entry, at=_now())
    with open(_dir() / "audit.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def audit(limit=100):
    try:
        lines = (_dir() / "audit.jsonl").read_text().splitlines()[-limit:]
    except OSError:
        return []
    out = []
    for l in reversed(lines):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def _need_helper():
    h = helper_info()
    if not h["installed"]:
        raise ValueError(f"the power helper is not installed. Run once: {h['install_cmd']}")
    if not h["sudo_ok"]:
        raise ValueError(f"the helper is installed but sudo refuses it ({h['error'] or 'no rule'}). "
                         f"Re-run: {h['install_cmd']}")
    if not h["current"]:
        raise ValueError(f"the installed helper is older than the panel's. Update it: {h['install_cmd']}")


def _sync_gpupower(results):
    """The GPU tab re-applies its saved amdgpu caps every minute; keep it in step with a
    cap set here, or the two would undo each other."""
    gp = getattr(P, "gpupower", None)
    if not gp:
        return
    try:
        saved = gp._saved()
        changed = False
        for r in results:
            if r.get("knob") == "gpu.power_cap" and r.get("ok") and gp._hwmon(r.get("resolved") or "") \
                    and str(r.get("value")).isdigit():
                pci = r["resolved"]
                if (saved.get(pci) or {}).get("watts") != int(r["value"]):
                    saved[pci] = dict(watts=int(r["value"]), set_at=_now(), name=(saved.get(pci) or {}).get("name"))
                    changed = True
        if changed:
            gp._write_saved(saved)
    except Exception as e:
        print(f"poweropts: GPU tab cap sync failed: {e}", flush=True)


def apply(body, why="apply"):
    _need_helper()
    v = view()
    if body.get("profile"):
        p = get_profile(str(body["profile"]), v)
        st, label = p["settings"], p["name"]
    else:
        st, label = _clean_settings(body.get("settings")), str(body.get("label") or "selected settings")[:80]
    res = _sudo("apply", dict(settings=st))["results"]
    _sync_gpupower(res)
    _audit(dict(action=why, profile=label, profile_id=body.get("profile"),
                results=[{k: r.get(k) for k in ("knob", "target", "resolved", "value", "before", "after", "ok", "msg")}
                         for r in res]))
    _drop("view")
    return dict(results=res, ok=all(r.get("ok") for r in res))


def persist(body):
    _need_helper()
    v = view()
    p = get_profile(str(body.get("profile") or ""), v)
    out = _sudo("persist", dict(name=p["name"], id=p["id"], settings=p["settings"]))
    s = settings()
    s["boot_profile"] = dict(id=p["id"], name=p["name"], at=_now())
    _atomic(_dir() / "settings.json", s)
    _audit(dict(action="boot-profile", profile=p["name"], profile_id=p["id"], settings=len(p["settings"])))
    _drop("view")
    res = dict(out, profile=p["name"])
    if body.get("apply_now"):
        res["applied"] = apply(dict(profile=p["id"]), why="apply")
    return res


def unpersist(body=None):
    _need_helper()
    _sudo("unpersist")
    s = settings()
    s.pop("boot_profile", None)
    _atomic(_dir() / "settings.json", s)
    _audit(dict(action="boot-profile", profile=None))
    _drop("view")
    return dict(ok=True)


def _same(knob, want, got):
    if got is None:
        return None
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip()
    return norm(want) == norm(got)


def drift(v=None):
    v = v or view()
    prof = v.get("persisted")
    if not prof:
        return None
    tab = v.get("table") or []
    out = []
    for s in prof.get("settings") or []:
        row = next((r for r in tab if r.get("knob") == s["knob"] and r.get("target") == s["target"]
                    and (not s.get("ident") or not r.get("ident") or r.get("ident") == s["ident"])), None)
        if row is None and s.get("ident"):
            same = [r for r in tab if r.get("knob") == s["knob"] and r.get("ident") == s["ident"]]
            row = same[0] if len(same) == 1 else None
        if row is None:
            out.append(dict(s, live=None, state="absent"))
            continue
        ok = _same(s["knob"], s["value"], row.get("value"))
        if ok is False:
            out.append(dict(s, live=row.get("value"), resolved=row["target"], state="drift"))
        elif ok is None:
            out.append(dict(s, live=None, resolved=row["target"], state="unreadable"))
    return dict(profile=prof.get("name"), id=prof.get("id"), saved=prof.get("saved"), items=out,
                drifted=sum(1 for x in out if x["state"] == "drift"))


def _ingest_bootlog(v):
    bl = v.get("bootlog")
    if not bl or not bl.get("boot_id"):
        return
    seen = _read(_dir() / "boots-seen.json", []) or []
    if bl["boot_id"] in seen:
        return
    _audit(dict(action="boot-apply", boot_id=bl["boot_id"], profile=bl.get("profile"),
                profile_id=bl.get("profile_id"), started=bl.get("started"),
                results=[{k: r.get(k) for k in ("knob", "target", "resolved", "value", "before", "after", "ok", "msg")}
                         for r in bl.get("results") or []],
                missing=[f"{m.get('knob')} {m.get('target')}" for m in bl.get("missing") or []]))
    _atomic(_dir() / "boots-seen.json", (seen + [bl["boot_id"]])[-100:])


# ============================================================================
# stability by profile
# ============================================================================
def annotations():
    return _read(_dir() / "annotations.json", {}) or {}


def annotate(body):
    bid = str(body.get("boot_id") or "")
    note = str(body.get("note") or "")
    if not re.fullmatch(r"[0-9a-f-]{8,40}", bid):
        raise ValueError("bad boot id")
    if note not in ("", "manual-off", "power-cut", "crash"):
        raise ValueError("note: manual-off, power-cut, crash or empty")
    a = annotations()
    if note:
        a[bid] = dict(note=note, at=_now())
    else:
        a.pop(bid, None)
    _atomic(_dir() / "annotations.json", a)
    _drop("stability")
    return dict(ok=True)


def _ts(s):
    try:
        return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def stability(force=False):
    if force:
        _drop("stability")

    def get():
        boots = P.collect_boots(max_boots=30)
        ev = list(reversed(audit(2000)))
        ann = annotations()
        peers = _peer_events()
        rows = []
        for i, b in enumerate(boots):
            bid = (b.get("boot_id") or "").replace("-", "")
            first, last = b.get("first") or 0, b.get("last") or 0
            prof = None
            for e in ev:                     # boot-apply of this boot, then applies inside it
                t = _ts(e.get("at")) or 0
                if e.get("action") == "boot-apply" and (e.get("boot_id") or "").replace("-", "") == bid:
                    prof = e.get("profile") or "(no boot profile)"
                elif e.get("action") in ("apply", "auto-reapply") and first <= t <= last and e.get("profile"):
                    prof = e["profile"] if not prof else (prof if prof == e["profile"] else f"{prof} → {e['profile']}")
            note = (ann.get(bid) or ann.get(b.get("boot_id") or "") or {}).get("note")
            ended = "running" if b.get("index") == 0 else (
                "clean" if b.get("verdict") != "hard-lock" else (note or "unclean"))
            pe = []
            if ended not in ("running", "clean"):
                nxt = boots[i + 1].get("first") if i + 1 < len(boots) else None   # oldest first
                for p in peers:
                    t = p["t"]
                    if last - 300 <= t <= last + 30 and p["state"] == "down":
                        pe.append(f"{p['name']} went down {int(last - t)} s before this box's last log")
                    if nxt and nxt <= t <= nxt + 900 and p.get("first_probe"):
                        pe.append(f"{p['name']} was {p['state']} when this box came back")
            rows.append(dict(index=b.get("index"), boot_id=b.get("boot_id"), start=first, end=last,
                             hours=round((b.get("duration_s") or 0) / 3600, 1), ended=ended,
                             profile=prof, note=note, peers=pe))
        agg = {}
        for r in rows:
            a = agg.setdefault(r["profile"] or "(before power options)", dict(boots=0, hours=0.0, stops=0, manual=0))
            a["boots"] += 1
            a["hours"] = round(a["hours"] + r["hours"], 1)
            if r["ended"] in ("unclean", "crash", "power-cut"):
                a["stops"] += 1
            elif r["ended"] == "manual-off":
                a["manual"] += 1
        for a in agg.values():
            a["hours_per_stop"] = round(a["hours"] / a["stops"], 1) if a["stops"] else None
        return dict(boots=list(reversed(rows)), by_profile=agg)       # newest first
    return _cached("stability", 120, get)


# ============================================================================
# UPS (NUT) and the power budget
# ============================================================================
def _upsc(target):
    try:
        r = subprocess.run(["upsc", target], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[-200:]
    return dict(re.findall(r"^([\w.]+):\s*(.*)$", r.stdout, re.M)), None


def ups_status():
    def get():
        nut = shutil.which("upsc") is not None
        cfg = list(settings().get("ups") or [])
        if nut and not cfg:
            try:
                r = subprocess.run(["upsc", "-l"], capture_output=True, text=True, timeout=5)
                cfg = [dict(name=n, host="localhost") for n in r.stdout.split() if re.fullmatch(r"[\w.-]+", n)]
            except (OSError, subprocess.SubprocessError):
                pass
        out = []
        for u in cfg if nut else []:
            d, err = _upsc(f"{u['name']}@{u['host']}")
            g = lambda k: (d or {}).get(k)
            f = lambda k: float(g(k)) if g(k) and re.fullmatch(r"[\d.]+", g(k)) else None
            nominal = f("ups.realpower.nominal") or ((f("ups.power.nominal") or 0) * 0.6 or None)
            load = f("ups.load")
            out.append(dict(name=u["name"], host=u["host"], error=err, status=g("ups.status"),
                            charge_pct=f("battery.charge"), runtime_s=f("battery.runtime"), load_pct=load,
                            nominal_w=round(nominal) if nominal else None,
                            load_w=round(f("ups.realpower") or (load * nominal / 100 if load and nominal else 0)) or None,
                            input_v=f("input.voltage"), model=" ".join(x for x in (g("device.mfr"), g("device.model")) if x) or None,
                            on_battery="OB" in (g("ups.status") or "").split()))
        return dict(nut=nut, ups=out, usb=_usb_ups())
    return _cached("ups", 10, get)


def budget(v=None):
    v = v or view()
    s = settings()
    peaks = _rigmon_peaks()
    tab = v.get("table") or []
    lines, sus, peak = [], 0, 0
    for g in P.gpu_devices(probe=False):
        if g.get("pci") in (None, "cpu") or g.get("vendor") not in ("amd", "nvidia"):
            continue
        cap = next((r.get("value") for r in tab if r.get("knob") == "gpu.power_cap" and r.get("target") == g["pci"]), None)
        pk = peaks.get(g["pci"]) or {}
        cap_w = int(cap) if cap and str(cap).isdigit() else None
        s_w = cap_w or pk.get("p99_w")
        p_w = pk.get("max_w") or cap_w
        lines.append(dict(item=f"GPU {g['pci']} {g.get('name') or ''}".strip(), sustained_w=s_w, peak_w=p_w,
                          basis=("cap" if cap_w else "measured p99") + (", peak measured (24 h)" if pk.get("max_w") else ", peak unknown")))
        sus += s_w or 0
        peak += p_w or 0
    c = _cpu()
    pl = c.get("power_limits_w") or {}
    cs, cp = pl.get("long_term"), pl.get("short_term") or pl.get("long_term")
    lines.append(dict(item=f"CPU {c.get('model') or ''}".strip(), sustained_w=cs, peak_w=cp,
                      basis="RAPL power limits" if pl else "unknown: enter the platform figure"))
    sus += cs or 0
    peak += cp or 0
    plat = s.get("platform_w", 60)
    lines.append(dict(item="Board, RAM, disks, fans", sustained_w=plat, peak_w=plat, basis="your figure" if "platform_w" in s else "default estimate"))
    sus += plat
    peak += plat
    warn = []
    psu = s.get("psu_w")
    if psu and peak > psu * 0.9:
        warn.append(f"Peak draw ({peak} W) is {'over' if peak > psu else 'within 10% of'} the PSU rating ({psu} W).")
    others = sum(int(o.get("w") or 0) for o in s.get("other_loads") or [])
    ups_w = s.get("ups_w") or (round(s["ups_va"] * 0.6) if s.get("ups_va") else None)
    u = ups_status()
    live = next((x for x in u.get("ups") or [] if x.get("nominal_w")), None)
    if live and not ups_w:
        ups_w = live["nominal_w"]
    if ups_w and peak + others > ups_w * 0.8:
        warn.append(f"This box's peak ({peak} W) plus the other loads on the UPS ({others} W) is "
                    f"{round((peak + others) / ups_w * 100)}% of the UPS's {ups_w} W. An overloaded UPS cuts "
                    "everything on it at once, with no log line on any of them.")
    if live and live.get("on_battery"):
        warn.append(f"UPS {live['name']} is ON BATTERY now.")
    return dict(lines=lines, sustained_w=sus, peak_w=peak, other_loads_w=others, psu_w=psu, ups_w=ups_w,
                ups_basis=("your rating" if s.get("ups_w") else "VA × 0.6" if s.get("ups_va") else
                           "reported by NUT" if live else None), warnings=warn)


# ============================================================================
# peers: machines on the same UPS
# ============================================================================
_peer_state = {}


def _peer_log(entry):
    with open(_dir() / "peers.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _peer_events():
    try:
        lines = (_dir() / "peers.jsonl").read_text().splitlines()[-5000:]
    except OSError:
        return []
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def _ping(host):
    try:
        return subprocess.run(["ping", "-n", "-c", "1", "-W", "1", host], capture_output=True,
                              timeout=4).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def peers_tick():
    for p in settings().get("peers") or []:
        up = _ping(p["host"])
        key = p["host"]
        prev = _peer_state.get(key)
        state = "up" if up else "down"
        if prev is None or prev["state"] != state:
            _peer_log(dict(t=time.time(), name=p["name"], host=key, state=state, first_probe=prev is None))
        _peer_state[key] = dict(state=state, since=time.time() if not prev or prev["state"] != state else prev["since"],
                                name=p["name"], checked=time.time())


def peers_now():
    return [dict(name=v["name"], host=k, state=v["state"], since=v["since"], checked=v["checked"])
            for k, v in _peer_state.items()]


# ============================================================================
# background: boot-log ingest, auto re-apply, peers
# ============================================================================
def minute_tick():
    """Record the boot unit's results once per boot, and re-apply drifted settings when
    the operator turned that on. Returns what it did (for tests and the log)."""
    h = helper_info()
    if not (h["installed"] and h["sudo_ok"]):
        return None
    v = view(fresh=True)
    _ingest_bootlog(v)
    d = drift(v)
    if d and d["drifted"] and settings().get("auto_reapply") and h["current"]:
        items = [dict(knob=x["knob"], target=x["target"], ident=x.get("ident"), value=x["value"])
                 for x in d["items"] if x["state"] == "drift"]
        res = apply(dict(settings=items, label=f"{d['profile']} (drifted)"), why="auto-reapply")
        print(f"poweropts: re-applied {len(items)} drifted setting(s) of {d['profile']}", flush=True)
        return res
    return None


def worker():
    n = 0
    while True:
        try:
            peers_tick()
            if n % 12 == 0:                       # every minute
                minute_tick()
        except Exception as e:
            print(f"poweropts: worker: {e}", flush=True)
        n += 1
        time.sleep(5)


# ============================================================================
# status for the tab
# ============================================================================
def status():
    v = view()
    h = helper_info()
    s = settings()
    return dict(helper=h, root=v.get("root"), helper_error=v.get("helper_error"), boot_id=v.get("boot_id"),
                table=v.get("table") or [], links=v.get("links") or [], baseline_at=(v.get("baseline") or {}).get("at")
                if (v.get("baseline") or {}).get("boot_id") == v.get("boot_id") else None,
                baseline={f"{x['knob']}|{x['target']}": x.get("value")
                          for x in ((v.get("baseline") or {}).get("settings") or [])}
                if (v.get("baseline") or {}).get("boot_id") == v.get("boot_id") else {},
                persisted=v.get("persisted"), drift=drift(v), bootlog=v.get("bootlog"),
                equipment=equipment(v), profiles=profiles(v), settings=s, audit=audit(60),
                peers=peers_now())
