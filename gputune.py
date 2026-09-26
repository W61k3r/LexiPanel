#!/usr/bin/env python3
"""
GPU Tuning (added 2026-09-25): every card in detail, its clocks, voltage and power
changed through the root helper, and each change measured on this box's own workload.

  cards     identity (PCI ids, board, VBIOS, VRAM vendor), PCIe link, DPM clock tables,
            sensors (edge / junction / memory, fan, power against its cap, core voltage)
            and the OverDrive table with the range the kernel accepts. Plain sysfs, no
            root; nvidia-smi for the proprietary driver.
  settings  max core clock, max memory clock and voltage offset (amdgpu OverDrive), power
            cap, performance level and fan curve. All are knobs of the power helper
            (power/lexipanel_power.py): range-checked there, read back after the write, and
            logged in the Power options audit. Changes made here are LIVE ONLY: a reboot
            brings back the stock table unless a boot profile (Power options) says otherwise,
            so an unstable overclock is undone by restarting the box.
  bench     one fixed llama.cpp request, repeated, against a running instance, with each of
            its cards' power, clocks, temperatures and load sampled every second. Decode
            tokens/s, tokens per joule, peaks, and kernel GPU resets during the run.
            "Apply and test" applies settings, runs the bench, and puts the previous values
            back if the run fails, the server dies, the driver resets the GPU, or a thermal
            limit trips.
  vbios     back up a card's VBIOS image (read through the helper), and check a ROM file
            against the card before a flash. LexiPanel never writes firmware: the check ends
            with the vendor tool's command, to run yourself in the terminal.

The first attempt at this tab (a development build before 1.0.0) flashed with `flashrom -p internal`,
which programs the MOTHERBOARD's firmware chip, not the GPU's; it was never wired to a route
and is gone.
"""
import base64, hashlib, http.client, json, os, re, statistics, subprocess, threading, time
from pathlib import Path

P = None            # panel
PO = None           # poweropts (the helper's front end)
DC = None           # depthcurve (server I/O helpers)
O = None            # optimizer (thermal limits)
SYS = "/sys"        # tests point this at a fake tree

TUNE_KNOBS = ("gpu.od_sclk", "gpu.od_mclk", "gpu.od_voltage", "gpu.power_cap",
              "gpu.perf_level", "gpu.fan")
OD_KNOBS = TUNE_KNOBS[:3]
BENCH_DEFAULTS = dict(depth=2048, n_predict=256, reps=5)
ROM_MAX = 16 * 1024 * 1024
RESET_PAT = re.compile(r"amdgpu.*(GPU reset|ring \S+ timeout|GPU recovery|VRAM is lost)|"
                       r"NVRM: Xid|GPU has fallen off the bus", re.I)
_lock = threading.RLock()
_run = None
_stop = threading.Event()
_conn = None


def bind(panel_module, poweropts_module, depthcurve_module, optimizer_module):
    global P, PO, DC, O
    P, PO, DC, O = panel_module, poweropts_module, depthcurve_module, optimizer_module


def _dir(*sub):
    d = P.PANEL.joinpath(*sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _r(p):
    try:
        return Path(p).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _num(p, div=1, nd=None):
    v = _r(p)
    try:
        x = int(v) / div
    except (TypeError, ValueError):
        return None
    return round(x, nd) if nd is not None else (int(x) if div == 1 else x)


# ============================================================================
# cards
# ============================================================================
def _dpm(txt):
    """pp_dpm_sclk & co -> [dict(level, mhz, current)]."""
    out = []
    for line in (txt or "").splitlines():
        m = re.match(r"\s*(\d+):\s*(\d+)\s*mhz\s*(\*)?", line, re.I)
        if m:
            out.append(dict(level=int(m.group(1)), mhz=int(m.group(2)), current=bool(m.group(3))))
    return out


def _hwmon(dev):
    base = f"{dev}/hwmon"
    try:
        for h in sorted(os.listdir(base)):
            if os.path.exists(f"{base}/{h}/name"):
                return f"{base}/{h}"
    except OSError:
        pass
    return None


def _amd_sensors(dev):
    h = _hwmon(dev)
    s = dict(temps={}, crit={}, power_w=None, cap_w=None, cap_min_w=None, cap_max_w=None,
             cap_default_w=None, fan_rpm=None, fan_max_rpm=None, vddgfx_mv=None,
             sclk_mhz=None, mclk_mhz=None,
             busy_pct=_num(f"{dev}/gpu_busy_percent"), mem_busy_pct=_num(f"{dev}/mem_busy_percent"))
    if not h:
        return s
    for i in range(1, 6):
        v = _num(f"{h}/temp{i}_input", 1000, 1)
        if v is None:
            continue
        lab = _r(f"{h}/temp{i}_label") or f"temp{i}"
        s["temps"][lab] = v
        c = _num(f"{h}/temp{i}_crit", 1000, 1)
        if c:
            s["crit"][lab] = c
    w = _num(f"{h}/power1_average", 1e6, 1)
    s["power_w"] = w if w is not None else _num(f"{h}/power1_input", 1e6, 1)
    for k, f in (("cap_w", "power1_cap"), ("cap_min_w", "power1_cap_min"),
                 ("cap_max_w", "power1_cap_max"), ("cap_default_w", "power1_cap_default")):
        s[k] = _num(f"{h}/{f}", 1e6, 0)
    s["fan_rpm"], s["fan_max_rpm"] = _num(f"{h}/fan1_input"), _num(f"{h}/fan1_max")
    s["vddgfx_mv"] = _num(f"{h}/in0_input")
    s["sclk_mhz"] = _num(f"{h}/freq1_input", 1e6, 0)
    s["mclk_mhz"] = _num(f"{h}/freq2_input", 1e6, 0)
    return s


def _profile_mode(dev):
    for line in (_r(f"{dev}/pp_power_profile_mode") or "").splitlines():
        m = re.match(r"\s*(\d+)\s+([A-Z0-9_]+)\s*\*", line)
        if m:
            return m.group(2)
    return None


def _overdrive_mask():
    v = _r(f"{SYS}/module/amdgpu/parameters/ppfeaturemask")
    try:
        return bool(int(v, 16 if v.lower().startswith("0x") else 10) & 0x4000)
    except (AttributeError, TypeError, ValueError):
        return None


def _nv(pci):
    """nvidia-smi fields for one card, or {}."""
    if not os.path.exists("/usr/bin/nvidia-smi"):
        return {}
    keys = ["pci.bus_id", "name", "vbios_version", "driver_version", "memory.total", "memory.used",
            "temperature.gpu", "temperature.memory", "power.draw", "power.limit",
            "power.min_limit", "power.max_limit", "clocks.gr", "clocks.mem", "clocks.max.gr",
            "clocks.max.mem", "fan.speed", "pstate", "pcie.link.gen.current", "pcie.link.gen.max",
            "pcie.link.width.current", "pcie.link.width.max", "utilization.gpu", "utilization.memory"]
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(keys)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in r.stdout.splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) == len(keys) and v[0].lower().endswith(pci.lower()[-7:]):
            num = lambda x: float(x) if re.fullmatch(r"-?[\d.]+", x) else None
            return dict(zip(keys, [num(x) if num(x) is not None else (None if x.startswith(("[", "N/A")) else x)
                                   for x in v]))
    return {}


def card(d, rows=()):
    """Everything readable about one GPU. d is a panel gpu_devices() record."""
    pci = d["pci"]
    dev = f"{SYS}/bus/pci/devices/{pci}"
    rid = lambda f: (_r(f"{dev}/{f}") or "0x")[2:] or None
    c = dict(pci=pci, name=d.get("name"), vendor=d.get("vendor"), driver=d.get("driver"),
             ids=dict(vendor=rid("vendor"), device=rid("device"),
                      subsystem=f"{rid('subsystem_vendor')}:{rid('subsystem_device')}"),
             board=_r(f"{dev}/product_name"), vbios=_r(f"{dev}/vbios_version"),
             vram=dict(total_mib=(_num(f"{dev}/mem_info_vram_total") or 0) // 1048576 or d.get("vram_total_mib"),
                       used_mib=(_num(f"{dev}/mem_info_vram_used") or 0) // 1048576 or None,
                       vendor=_r(f"{dev}/mem_info_vram_vendor")),
             link=dict(speed=_r(f"{dev}/current_link_speed"), width=_r(f"{dev}/current_link_width"),
                       max_speed=_r(f"{dev}/max_link_speed"), max_width=_r(f"{dev}/max_link_width")),
             clocks={}, sensors={}, perf_level=None, power_profile=None,
             overdrive=dict(available=False, table=None, hint=None),
             settings=[r for r in rows if r.get("target") == pci], notes=list(d.get("notes") or []))
    if d.get("driver") == "amdgpu":
        c["clocks"] = {k: _dpm(_r(f"{dev}/pp_dpm_{k}")) for k in ("sclk", "mclk", "fclk", "socclk")}
        c["clocks"] = {k: v for k, v in c["clocks"].items() if v}
        c["sensors"] = _amd_sensors(dev)
        c["perf_level"] = _r(f"{dev}/power_dpm_force_performance_level")
        c["power_profile"] = _profile_mode(dev)
        od = PO.lp().od_table(pci) if PO else None
        if od:
            c["overdrive"] = dict(available=True, table=od, hint=None)
        else:
            mask = _overdrive_mask()
            c["overdrive"]["hint"] = (
                "OverDrive is off (amdgpu.ppfeaturemask lacks bit 0x4000), so clocks and voltage are "
                "locked. Copy systemd/99-amdgpu-overdrive.cfg to /etc/default/grub.d/, run "
                "sudo update-grub and reboot." if mask is False else
                "This card prints no OverDrive clock table (older generation, or OverDrive off).")
    elif d.get("driver") == "nvidia":
        n = _nv(pci)
        if n:
            c["vbios"] = n.get("vbios_version") or c["vbios"]
            c["vram"].update(total_mib=int(n.get("memory.total") or 0) or c["vram"]["total_mib"],
                             used_mib=int(n["memory.used"]) if n.get("memory.used") is not None else None)
            c["sensors"] = dict(temps={k: v for k, v in (("gpu", n.get("temperature.gpu")),
                                                          ("mem", n.get("temperature.memory"))) if v is not None},
                                crit={}, power_w=n.get("power.draw"), cap_w=n.get("power.limit"),
                                cap_min_w=n.get("power.min_limit"), cap_max_w=n.get("power.max_limit"),
                                sclk_mhz=n.get("clocks.gr"), mclk_mhz=n.get("clocks.mem"),
                                sclk_max_mhz=n.get("clocks.max.gr"), mclk_max_mhz=n.get("clocks.max.mem"),
                                fan_pct=n.get("fan.speed"), busy_pct=n.get("utilization.gpu"),
                                mem_busy_pct=n.get("utilization.memory"), pstate=n.get("pstate"))
            c["link"].update(gen=n.get("pcie.link.gen.current"), max_gen=n.get("pcie.link.gen.max"),
                             width=c["link"]["width"] or n.get("pcie.link.width.current"),
                             max_width=c["link"]["max_width"] or n.get("pcie.link.width.max"))
            c["driver_version"] = n.get("driver_version")
        c["overdrive"]["hint"] = ("NVIDIA clocks and voltage are not changed from the panel; the power "
                                  "cap is (Power cap row, through nvidia-smi).")
    elif d.get("vendor") == "intel":
        c["overdrive"]["hint"] = "Integrated GPU: never an inference device here, nothing to tune."
    return c


def sample(pcis):
    """Quick live numbers across an instance's cards: summed power, hottest temps,
    first card's clocks and load. Used every second during a bench."""
    out = dict(power_w=None, junction=None, mem=None, edge=None, sclk=None, mclk=None, busy=None, fan=None)
    for i, pci in enumerate(pcis):
        dev = f"{SYS}/bus/pci/devices/{pci}"
        if _hwmon(dev):
            s = _amd_sensors(dev)
            t = s["temps"]
            vals = dict(power_w=s["power_w"], junction=t.get("junction"), mem=t.get("mem"),
                        edge=t.get("edge"), sclk=s["sclk_mhz"], mclk=s["mclk_mhz"],
                        busy=s["busy_pct"], fan=s["fan_rpm"])
        else:
            n = _nv(pci)
            vals = dict(power_w=n.get("power.draw"), junction=n.get("temperature.gpu"),
                        mem=n.get("temperature.memory"), sclk=n.get("clocks.gr"),
                        mclk=n.get("clocks.mem"), busy=n.get("utilization.gpu"))
        if vals.get("power_w") is not None:
            out["power_w"] = round((out["power_w"] or 0) + vals["power_w"], 1)
        for k in ("junction", "mem", "edge"):
            if vals.get(k) is not None and (out[k] is None or vals[k] > out[k]):
                out[k] = vals[k]
        if i == 0:
            for k in ("sclk", "mclk", "busy", "fan"):
                out[k] = vals.get(k)
    return out


def _rows(fresh=False):
    v = PO.view(fresh=fresh)
    return v, [r for r in v.get("table") or [] if r.get("knob") in TUNE_KNOBS and r.get("target")]


def status():
    v, rows = _rows()
    base = {}
    for s in ((v.get("baseline") or {}).get("settings") or []) if (v.get("baseline") or {}).get("boot_id") == v.get("boot_id") else []:
        base[f"{s['knob']}|{s['target']}"] = s.get("value")
    cards = [card(d, rows) for d in P.gpu_devices(probe=False)
             if d["pci"] != "cpu" and d.get("vendor") in ("amd", "nvidia", "intel")]
    h = PO.helper_info()
    hist = history()
    with _lock:
        live = public(_run) if _run else None
    return dict(cards=cards, baseline=base, helper=dict(h, ready=bool(h["installed"] and h["sudo_ok"] and h["current"])),
                bench=dict(active=live, defaults=BENCH_DEFAULTS, instances=_bench_targets()),
                profiles=_gpu_profiles(),
                history=hist[:30], advice=advice(cards, hist, h), backups=vbios_list())


def _bench_targets():
    out = []
    for iid in P.instance_ids():
        try:
            inst = P.get_instance(iid)
        except ValueError:
            continue
        if inst.get("engine") not in (None, "", "llama.cpp"):
            continue
        with P.using_instance(inst):
            up = P.server_pid() is not None
        out.append(dict(id=iid, running=up, devices=[x for x in (inst.get("devices") or [inst.get("device")])
                                                     if x and x != "cpu"]))
    return out


# ============================================================================
# settings (through the power helper)
# ============================================================================
def _clean(settings):
    if not isinstance(settings, list) or not settings or len(settings) > 24:
        raise ValueError("settings: a list of 1-24 changes")
    out = []
    for s in settings:
        if not isinstance(s, dict) or s.get("knob") not in TUNE_KNOBS:
            raise ValueError(f"not a GPU tuning setting: {str((s or {}).get('knob'))[:40]!r}")
        out.append(dict(knob=s["knob"], target=str(s.get("target") or ""),
                        ident=s.get("ident"), value=str(s.get("value") if s.get("value") is not None else "")))
    return out


def apply(body):
    """Live change of GPU tuning knobs. Validation, write and read-back happen in the helper."""
    return PO.apply(dict(settings=_clean(body.get("settings")),
                         label=str(body.get("label") or "GPU Tuning")[:80]), why="gpu-tune")


# ============================================================================
# profiles: a named fan / voltage / clock set, stored and applied through Power
# options (so it can be the boot profile, and every change is in the same audit).
# ============================================================================
def _gpu_profiles():
    """Power-options profiles made entirely of GPU tuning knobs (the ones this tab saves),
    each with the card(s) it targets and whether it is the boot profile."""
    boot = (PO.settings().get("boot_profile") or {}).get("id")
    out = []
    for p in PO.profiles():
        st = p.get("settings") or []
        if p.get("builtin") or not st or any(s.get("knob") not in TUNE_KNOBS for s in st):
            continue
        out.append(dict(id=p["id"], name=p["name"], description=p.get("description"),
                        settings=st, targets=sorted({s["target"] for s in st}), boot=p["id"] == boot))
    return out


def profile_save(body):
    st = _clean(body.get("settings"))
    return PO.save_profile(dict(name=body.get("name"), description=str(body.get("description") or "")[:500],
                                settings=st, source="gpu-tune", id=body.get("id")))


def profile_delete(body):
    return PO.delete_profile(body)


def profile_apply(body):
    return PO.apply(dict(profile=str(body.get("id") or "")), why="gpu-tune profile")


def profile_persist(body):
    return PO.persist(dict(profile=str(body.get("id") or ""), apply_now=bool(body.get("apply_now"))))


def profile_unpersist(body=None):
    return PO.unpersist()


def _current(settings):
    """[setting with the value that is live now] for the same knob/targets, or raise."""
    _v, rows = _rows(fresh=True)
    live = {f"{r['knob']}|{r['target']}": r for r in rows}
    out, skipped = [], []
    for s in settings:
        r = live.get(f"{s['knob']}|{s['target']}")
        val = r and r.get("value")
        if val in (None, "unset", "mixed", "custom"):
            skipped.append(f"{s['knob']} {s['target']} (now {val!r}: cannot be put back)")
            continue
        out.append(dict(knob=s["knob"], target=s["target"], ident=r.get("ident"), value=str(val)))
    return out, skipped


# ============================================================================
# bench
# ============================================================================
def busy():
    with _lock:
        return bool(_run and not _run.get("_done"))


def _stream(host, port, prompt, n_predict, cache=True):
    """Streaming /completion on our own connection (so Stop can drop it); returns timings."""
    global _conn
    c = http.client.HTTPConnection(host, port, timeout=1800)
    with _lock:
        _conn = c
    try:
        c.request("POST", "/completion", body=json.dumps(dict(
            prompt=prompt, n_predict=n_predict, ignore_eos=True, cache_prompt=cache, stream=True)),
            headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"/completion -> HTTP {r.status}: {r.read()[:200]!r}")
        timings = None
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            if ev.get("timings"):
                timings = ev["timings"]
            if ev.get("stop"):
                break
        if timings is None:
            raise RuntimeError("the stream ended without timings (request cancelled or server died)")
        return timings
    finally:
        with _lock:
            _conn = None
        c.close()


def _drop():
    with _lock:
        c = _conn
    if c is not None and c.sock is not None:
        try:
            import socket
            c.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _journal_cursor():
    """Where the kernel log stands now, so a run reads only its own lines (a timestamp
    would also catch the previous run's reset when two runs start in the same second)."""
    try:
        r = subprocess.run(["journalctl", "-k", "-n", "1", "-o", "cat", "--show-cursor", "--no-pager"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"^-- cursor: (\S+)", r.stdout, re.M)
    return m.group(1) if m else None


def _resets(cursor, since):
    """Kernel GPU reset / hang lines logged during the run, or None if unreadable."""
    where = ["--after-cursor", cursor] if cursor else ["--since", f"@{int(since)}"]
    try:
        r = subprocess.run(["journalctl", "-k", "-o", "cat", "--no-pager"] + where,
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [l[:240] for l in r.stdout.splitlines() if RESET_PAT.search(l)][:20]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 1) if xs else None


def _peak(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _summary(run):
    rows = [r for r in run["rows"] if r.get("decode_tps")]
    if not rows:
        return None
    dec = [r["decode_tps"] for r in rows]
    med = statistics.median(dec)
    s = [x for r in rows for x in r.get("samples") or []]
    pw = _mean([x["power_w"] for x in s])
    cap = run.get("cap_w")
    return dict(decode_tps=round(med, 2), decode_min=round(min(dec), 2), decode_max=round(max(dec), 2),
                spread_pct=round((max(dec) - min(dec)) * 100 / med, 1) if med else None,
                prefill_tps=run.get("prefill_tps"), mean_power_w=pw, peak_power_w=_peak([x["power_w"] for x in s]),
                tok_per_j=round(med / pw, 3) if pw else None,
                j_per_tok=round(pw / med, 2) if pw and med else None,
                power_frac=round(pw / cap, 3) if pw and cap else None,
                peak_junction=_peak([x["junction"] for x in s]), peak_mem=_peak([x["mem"] for x in s]),
                mean_sclk=_mean([x["sclk"] for x in s]), mean_mclk=_mean([x["mclk"] for x in s]),
                mean_busy=_mean([x["busy"] for x in s]), reps=len(rows))


def public(run):
    if not run:
        return None
    r = {k: v for k, v in run.items() if not k.startswith("_")}
    r["rows"] = [{k: v for k, v in x.items() if k != "samples"} for x in run["rows"]]
    r["summary"] = _summary(run)
    r["series"] = run.get("_series", [])[-600:]
    return r


def _save(run):
    rec = public(run)
    f = run["_file"]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, f)
    return rec


def history(limit=60):
    out = []
    for f in sorted(_dir("gpu-tune").glob("bench_*.json"), reverse=True)[:limit]:
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def _cap(pci):
    dev = f"{SYS}/bus/pci/devices/{pci}"
    if _hwmon(dev):
        return _amd_sensors(dev)["cap_w"]
    return _nv(pci).get("power.limit")


def _alive(inst):
    with P.using_instance(inst):
        return P.server_pid() is not None


def _sampler(run, th):
    hot = 0
    while not run.get("_done"):
        t = sample(run["devices"])
        t["t"] = round(time.time() - run["_t0"], 1)
        with _lock:
            run["live"] = t
            run.setdefault("_series", []).append(t)
            if len(run["_series"]) > 3600:
                del run["_series"][:600]
            cur = run.get("_cur")
            if cur is not None:
                cur.setdefault("samples", []).append(t)
        over = (t.get("junction") or 0) >= th["abort_c"] or (t.get("mem") or 0) >= th["mem_abort_c"]
        hot = hot + 1 if over else 0
        if hot >= 3 and not run.get("thermal_abort"):
            run["thermal_abort"] = (f"junction {t.get('junction')} C / memory {t.get('mem')} C held at or "
                                    f"above {th['abort_c']}/{th['mem_abort_c']} C")
            _drop()
        time.sleep(1)


def _worker(run, inst, th):
    host, port = run["host"], run["port"]
    applied = False
    try:
        if run.get("trial"):
            run["step"] = "applying the trial settings"
            res = PO.apply(dict(settings=run["trial"], label="GPU Tuning: apply and test"), why="gpu-tune trial")
            run["trial_results"] = res["results"]
            applied = True
            if not res["ok"]:
                raise RuntimeError("the helper refused or could not read back: " + "; ".join(
                    f"{x.get('knob')} {x.get('msg')}" for x in res["results"] if not x.get("ok")))
            time.sleep(2)
        run["step"] = "tokenizing the prompt"
        toks = DC._filler_tokens(host, port, run["depth"] + 16)[:run["depth"]]
        run["step"] = "warm-up (cold prefill)"
        tm = _stream(host, port, toks, 16, cache=False)
        if (tm.get("prompt_n") or 0) >= len(toks) * 0.9:        # a cached prompt says nothing
            run["prefill_tps"] = round(tm.get("prompt_per_second") or 0, 1)
        for rep in range(run["reps"]):
            if _stop.is_set():
                raise InterruptedError("stopped")
            if run.get("thermal_abort"):
                raise RuntimeError("thermal abort: " + run["thermal_abort"])
            run["step"] = f"run {rep + 1} of {run['reps']}"
            cur = dict(rep=rep, started=_now())
            with _lock:
                run["_cur"] = cur
            t0 = time.time()
            tm = _stream(host, port, toks, run["n_predict"])
            with _lock:
                run["_cur"] = None
            cur.update(decode_tps=round(tm.get("predicted_per_second") or 0, 2),
                       predicted_n=tm.get("predicted_n"), wall_s=round(time.time() - t0, 1))
            run["rows"].append(cur)
            _save(run)
        run["state"] = "done"
    except Exception as e:
        if isinstance(e, InterruptedError) or (_stop.is_set() and not run.get("thermal_abort")):
            run["state"] = "stopped"
        else:
            run["state"] = "thermal_abort" if run.get("thermal_abort") else "failed"
            run["error"] = run.get("thermal_abort") or str(e)[:500]
    finally:
        with _lock:
            run["_cur"] = None
        run["server_alive"] = _alive(inst)
        run["gpu_resets"] = _resets(run.get("_cursor"), run["_t0"])
        if run["state"] == "done" and (not run["server_alive"] or run["gpu_resets"]):
            run["state"] = "failed"
            run["error"] = ("the server died during the run" if not run["server_alive"]
                            else f"the kernel logged {len(run['gpu_resets'])} GPU reset/hang line(s)")
        run["verdict"] = ("stable" if run["state"] == "done" else
                          "stopped" if run["state"] == "stopped" else "unstable")
        if applied and run["state"] != "done" and run.get("revert"):
            run["step"] = "putting the previous values back"
            try:
                res = PO.apply(dict(settings=run["revert"], label="GPU Tuning: revert after a failed test"),
                               why="gpu-tune revert")
                run["reverted"] = dict(ok=res["ok"], results=res["results"])
            except Exception as e:
                run["reverted"] = dict(ok=False, error=str(e)[:300])
        run["_done"] = True
        run["finished"] = _now()
        run["step"] = None
        _save(run)


def bench_start(body):
    global _run
    body = body or {}
    with _lock:
        if _run and not _run.get("_done"):
            raise ValueError("a GPU benchmark is already running")
    if DC._run and not DC._run.get("_done"):
        raise ValueError("a depth-curve run is in progress; wait for it or stop it")
    if (O.status().get("active") or {}).get("state") == "running":
        raise ValueError("the optimizer is running (it restarts the server); wait for it to finish")
    if getattr(getattr(P, "benchlab", None), "active", lambda: False)():
        raise ValueError("a Bench run is active; wait for it or stop it")
    iid = str(body.get("instance") or "main")
    inst = P.get_instance(iid)
    if inst.get("engine") not in (None, "", "llama.cpp"):
        raise ValueError(f"the benchmark drives llama.cpp; {iid} runs {inst['engine']}")
    host, port, devices, argv = DC._target(inst)
    gpus = [d for d in devices if d and d != "cpu"]
    if not gpus:
        raise ValueError(f"{iid} runs on the CPU: nothing to tune")
    slots = DC._json(host, port, "GET", "/slots", timeout=10)
    if isinstance(slots, list) and any(s.get("is_processing") for s in slots):
        raise ValueError("the server is busy with a request; try again when it is idle")
    props = DC._json(host, port, "GET", "/props", timeout=10)
    n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
    try:
        depth = int(body.get("depth") or BENCH_DEFAULTS["depth"])
        n_predict = int(body.get("n_predict") or BENCH_DEFAULTS["n_predict"])
        reps = int(body.get("reps") or BENCH_DEFAULTS["reps"])
    except (TypeError, ValueError):
        raise ValueError("depth, n_predict and reps are whole numbers")
    if not (256 <= depth <= 65536 and 32 <= n_predict <= 2048 and 1 <= reps <= 30):
        raise ValueError("depth 256-65536, n_predict 32-2048, reps 1-30")
    if n_ctx and depth + n_predict + 64 > n_ctx:
        raise ValueError(f"depth + n_predict must fit the context ({n_ctx} tokens)")
    trial = revert = None
    skipped = []
    if body.get("trial"):
        trial = _clean(body["trial"])
        bad = [s for s in trial if s["target"] not in gpus]
        if bad:
            raise ValueError(f"{bad[0]['target']} is not a card of {iid} (it uses {', '.join(gpus)})")
        PO._need_helper()
        PO.lp()._check_settings(trial)          # the helper's own checks, before anything runs
        revert, skipped = _current(trial)
    _v, rows = _rows(fresh=bool(trial))
    in_force = {f"{r['knob']}|{r['target']}": r.get("value") for r in rows if r["target"] in gpus}
    if trial:
        in_force.update({f"{s['knob']}|{s['target']}": s["value"] for s in trial})
    cap = sum(_cap(p) or 0 for p in gpus) or None
    model = P._argv_get(argv, ("-m", "--model")) or ""
    ts = time.strftime("%Y%m%d_%H%M%S")
    th = dict(O.THERMAL_DEFAULTS)
    run = dict(id=ts, instance=iid, model_name=Path(model).name, host=host, port=port, devices=gpus,
               depth=depth, n_predict=n_predict, reps=reps, label=str(body.get("label") or "")[:80],
               settings=in_force, cap_w=cap, trial=trial, revert=revert, revert_skipped=skipped,
               thermal_limits=th, started=_now(), state="running", step="starting", rows=[],
               _t0=time.time(), _cursor=_journal_cursor(), _file=_dir("gpu-tune") / f"bench_{ts}.json")
    _stop.clear()
    with _lock:
        _run = run
    threading.Thread(target=_worker, args=(run, inst, th), daemon=True).start()
    threading.Thread(target=_sampler, args=(run, th), daemon=True).start()
    return public(run)


def bench_stop():
    _stop.set()
    _drop()
    return dict(ok=True)


def bench_status():
    with _lock:
        live = public(_run) if _run else None
    return dict(active=live, history=history()[:30], defaults=BENCH_DEFAULTS)


def bench_delete(body):
    rid = str(body.get("id") or "")
    if not re.fullmatch(r"\d{8}_\d{6}", rid):
        raise ValueError("bad run id")
    f = _dir("gpu-tune") / f"bench_{rid}.json"
    if not f.exists():
        raise ValueError("no such run")
    with _lock:
        if _run and _run["id"] == rid and not _run.get("_done"):
            raise ValueError("that run is still going")
    f.unlink()
    return dict(ok=True)


# ============================================================================
# advice: every line quotes the number it rests on
# ============================================================================
def advice(cards, hist, helper):
    out = []
    add = lambda pci, level, text: out.append(dict(card=pci, level=level, text=text))
    if not (helper["installed"] and helper["sudo_ok"] and helper["current"]):
        add(None, "info", "Settings are read-only until the power helper is installed or updated: "
                          f"{helper['install_cmd']}")
    for c in cards:
        if c["vendor"] == "intel":
            continue
        s = c.get("sensors") or {}
        if c["driver"] == "amdgpu" and not c["overdrive"]["available"] and c["overdrive"]["hint"]:
            add(c["pci"], "info", c["overdrive"]["hint"])
        j, jc = (s.get("temps") or {}).get("junction"), (s.get("crit") or {}).get("junction")
        if j is not None and jc and j >= jc - 10:
            add(c["pci"], "warn", f"Junction is {j} °C now, within 10 °C of its {jc} °C limit. Fix cooling "
                                  "(fan curve, lower power cap) before raising clocks.")
        runs = [h for h in hist if c["pci"] in (h.get("devices") or []) and h.get("summary")
                and h.get("verdict") == "stable"]
        if not runs:
            add(c["pci"], "info", "No benchmark on this card yet. Run one at stock first: it is the "
                                  "baseline every change is compared against.")
            continue
        last = runs[0]["summary"]
        if last.get("power_frac") and last["power_frac"] >= 0.95:
            add(c["pci"], "tip", f"Power-limited: the last run averaged {last['mean_power_w']} W, "
                                 f"{round(last['power_frac'] * 100)}% of the cap. More core clock cannot help "
                                 "while the cap holds it; a negative voltage offset gets more clock out of the "
                                 "same watts, or raise the cap if the junction has headroom.")
        if last.get("peak_junction") and last["peak_junction"] >= O.THERMAL_DEFAULTS["resume_c"]:
            add(c["pci"], "warn", f"The last run peaked at {last['peak_junction']} °C junction. Thermal "
                                  "headroom is gone: lower the cap or steepen the fan curve before going faster.")
        if last.get("mean_busy") is not None and last["mean_busy"] < 70:
            add(c["pci"], "tip", f"The card was only {last['mean_busy']}% busy during decode: the limit is "
                                 "elsewhere (CPU offload, N_CPU_MOE, host RAM). Clock changes will do little.")
        if last.get("spread_pct") and last["spread_pct"] > 8:
            add(c["pci"], "warn", f"Decode varied {last['spread_pct']}% between repeats in the last run: "
                                  "throttling or other load. Compare runs only when this is low.")
        same = [h for h in runs if h.get("model_name") == runs[0].get("model_name")
                and h.get("depth") == runs[0].get("depth") and h.get("n_predict") == runs[0].get("n_predict")]
        if len(same) >= 2:
            fast = max(same, key=lambda h: h["summary"]["decode_tps"])
            eff = max((h for h in same if h["summary"].get("tok_per_j")), key=lambda h: h["summary"]["tok_per_j"],
                      default=None)
            add(c["pci"], "result", f"Fastest of {len(same)} comparable runs: {fast['id']} "
                                    f"({fast['summary']['decode_tps']} t/s{', ' + fast['label'] if fast.get('label') else ''}).")
            if eff and eff is not fast:
                add(c["pci"], "result", f"Most efficient: {eff['id']} ({eff['summary']['tok_per_j']} tokens/J, "
                                        f"{eff['summary']['decode_tps']} t/s{', ' + eff['label'] if eff.get('label') else ''}).")
    return out


# ============================================================================
# VBIOS: back up, inspect, check a file before a flash (never flash)
# ============================================================================
CODE_TYPES = {0: "x86 BIOS", 1: "Open Firmware", 2: "PA-RISC", 3: "UEFI"}


def parse_rom(data):
    """PCI option ROM: the chain of 0x55AA images with their PCIR data structures."""
    start = 0
    if data[:2] != b"\x55\xaa":                     # nvflash dumps can carry a header first
        m = re.search(rb"\x55\xaa", data[:0x10000])
        start = m.start() if m else 0
    images, off = [], start
    while off + 0x1a <= len(data) and len(images) < 8:
        if data[off:off + 2] != b"\x55\xaa":
            break
        pcir = off + int.from_bytes(data[off + 0x18:off + 0x1a], "little")
        if data[pcir:pcir + 4] != b"PCIR":
            break
        ln = int.from_bytes(data[pcir + 0x10:pcir + 0x12], "little") * 512
        ct = data[pcir + 0x14]
        images.append(dict(offset=off, vendor=f"{int.from_bytes(data[pcir + 4:pcir + 6], 'little'):04x}",
                           device=f"{int.from_bytes(data[pcir + 6:pcir + 8], 'little'):04x}",
                           length=ln, code_type=CODE_TYPES.get(ct, f"type {ct}")))
        if data[pcir + 0x15] & 0x80 or ln == 0:
            break
        off += ln
    head = data[start:start + 0x10000]
    part = re.search(rb"113-[A-Z0-9]{3,12}-[A-Z0-9]{2,6}", head)
    date = re.search(rb"(20\d\d/\d\d/\d\d \d\d:\d\d)", head)
    used = (images[-1]["offset"] + images[-1]["length"] - start) if images else 0
    return dict(valid=bool(images), size=len(data), used_bytes=used, header_bytes=start,
                sha256=hashlib.sha256(data).hexdigest(), images=images,
                vendor=images[0]["vendor"] if images else None, device=images[0]["device"] if images else None,
                uefi=any(i["code_type"] == "UEFI" for i in images),
                part_number=part.group(0).decode() if part else None,
                build_date=date.group(1).decode() if date else None)


def _gpu_record(pci):
    d = next((x for x in P.gpu_devices(probe=False) if x["pci"] == pci and x["pci"] != "cpu"), None)
    if not d:
        raise ValueError(f"no GPU at {pci}")
    return d


def vbios_list():
    folder = _dir("gpu-bios")
    try:                                           # where the Files tab (and the ROM check) sees it
        rel = folder.resolve().relative_to(P.filemgr._root().resolve()).as_posix()
    except (ValueError, OSError, AttributeError):
        rel = None
    out = []
    for f in sorted(folder.glob("*.json"), reverse=True):
        try:
            b = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        b["files_path"] = f"{rel}/{b['file']}" if rel is not None else None
        out.append(b)
    return out


def vbios_backup(body):
    pci = str(body.get("target") or "")
    d = _gpu_record(pci)
    PO._need_helper()
    got = PO._sudo("vbios", dict(target=pci), timeout=60)
    data = base64.b64decode(got["data"])
    info = parse_rom(data)
    ver = (_r(f"{SYS}/bus/pci/devices/{pci}/vbios_version") or info.get("part_number") or "unknown")
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = re.sub(r"[^\w.-]+", "_", f"{pci.replace(':', '-')}_{ver}_{ts}")
    folder = _dir("gpu-bios")
    (folder / f"{stem}.rom").write_bytes(data)
    meta = dict(info, file=f"{stem}.rom", pci=pci, name=d.get("name"), vbios=ver, source=got.get("source"),
                ident=got.get("ident"), saved=_now())
    (folder / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    PO._audit(dict(action="vbios-backup", target=pci, file=meta["file"], size=len(data), sha256=info["sha256"]))
    return meta


def vbios_file(name):
    name = str(name or "")
    if not re.fullmatch(r"[\w.-]{1,160}\.rom", name):
        raise ValueError("bad file name")
    f = _dir("gpu-bios") / name
    if not f.is_file():
        raise ValueError("no such backup")
    return f


def rom_check(body):
    """Is this ROM file plausibly right for this card? Reads only; flashes nothing."""
    pci = str(body.get("target") or "")
    d = _gpu_record(pci)
    f = P.filemgr.resolve(str(body.get("path") or ""))
    if not f.is_file():
        raise ValueError("pick a file, not a folder")
    if f.stat().st_size > ROM_MAX:
        raise ValueError("larger than 16 MiB: not a GPU ROM")
    data = f.read_bytes()
    rom = parse_rom(data)
    dev = f"{SYS}/bus/pci/devices/{pci}"
    want_v, want_d = (_r(f"{dev}/vendor") or "0x")[2:], (_r(f"{dev}/device") or "0x")[2:]
    backups = [b for b in vbios_list() if b.get("pci") == pci]
    ref = backups[0] if backups else None
    checks = []
    add = lambda ok, text: checks.append(dict(ok=ok, text=text))
    add(rom["valid"], "a PCI option ROM (0x55AA signature and PCIR structure)" if rom["valid"]
        else "not a PCI option ROM: no 0x55AA image with a PCIR structure")
    if rom["valid"]:
        add(rom["vendor"] == want_v, f"vendor id {rom['vendor']} (card: {want_v})")
        add(rom["device"] == want_d, f"device id {rom['device']} (card: {want_d})"
            + ("" if rom["device"] == want_d else ": this image is for a different GPU"))
    if ref:
        add(rom["size"] == ref["size"], f"size {rom['size']:,} bytes (backup of this card: {ref['size']:,})")
        if ref.get("uefi"):
            add(rom["uefi"], "has a UEFI (GOP) image, like the card's own" if rom["uefi"]
                else "no UEFI (GOP) image, the card's own has one: no display before the OS on a UEFI boot")
        if ref.get("part_number") and rom.get("part_number"):
            same = ref["part_number"] == rom["part_number"]
            add(True if same else None, f"board part number {rom['part_number']} (card: {ref['part_number']})"
                + ("" if same else ": another board's image; right only if you mean to cross-flash"))
        add(True if rom["sha256"] != ref["sha256"] else False,
            "differs from the backup" if rom["sha256"] != ref["sha256"] else "identical to the backup: nothing to flash")
    else:
        add(False, "no VBIOS backup of this card yet: back it up first and keep a copy off this box")
    blocked = any(c["ok"] is False for c in checks)
    vendor = d.get("vendor")
    fname = str(f)
    if vendor == "amd":
        cmd = ["sudo amdvbflash -i                  # find the adapter number of " + pci,
               f"sudo amdvbflash -s <adapter> backup-{pci.replace(':', '-')}.rom",
               f"sudo amdvbflash -p <adapter> '{fname}'"]
        notes = ["RDNA2 and newer only accept images signed by AMD; a modified image is refused or leaves "
                 "the card unbootable.", "Linux amdvbflash builds lag behind new cards; use the board "
                 "vendor's tool if yours is not listed."]
    elif vendor == "nvidia":
        cmd = ["sudo nvflash --list", f"sudo nvflash --index=<n> --save backup-{pci.replace(':', '-')}.rom",
               f"sudo nvflash --index=<n> '{fname}'"]
        notes = ["Turing and newer only accept NVIDIA-signed images."]
    else:
        cmd, notes = [], ["Intel GPU firmware updates come through the driver / fwupd, not a ROM flash."]
    notes += ["Stop every instance on this card first, and have a second display path (iGPU or another "
              "card) so a bad flash can be recovered. LexiPanel does not run these commands."]
    return dict(target=pci, file=fname, rom=rom, checks=checks,
                verdict="blocked" if blocked else ("review" if any(c["ok"] is None for c in checks) else "ok"),
                commands=cmd if not blocked else [], notes=notes)
