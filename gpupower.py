#!/usr/bin/env python3
"""
GPU power caps (added 2026-09-23).

One cap per GPU, set from the GPU tab, kept in panel/gpu-power.json and
re-applied by the panel at startup and every ENFORCE_S seconds, so it survives
reboots and GPU resets without touching /etc beyond a one-time udev grant.

  amdgpu   hwmon power1_cap (microwatts). The kernel enforces its own
           power1_cap_min..power1_cap_max, and so does this module.
           PPT is an AVERAGED limit: it pulls the sustained draw down, but
           the sub-second transients (450 W+ observed on the XTX) still go
           through. Do not sell a cap as a transient fix.
  others   listed, not settable here (nvidia-smi -pl needs root on every
           call; nouveau exposes no cap).

power1_cap is root:root 0644 by default. The panel runs as admin, so it
needs the udev rule in GRANT_CMD once. Until then status() says so and hands
over the command instead of failing silently.

A cap of 0 read back from sysfs is what LexiPanel's XTX reports at boot; sustained
30 s averages of up to 329 W were measured in that state, so 0 is NOT the
291 W default being enforced. It is shown as "not set", and clearing a saved
cap leaves the live value alone until the next reboot rather than guessing
what "stock" means.
"""
import json, os, threading, time
from pathlib import Path

P = None                        # the panel module, bound by panel.py
CAPS_FILE = None
ENFORCE_S = 60
_lock = threading.RLock()
_last = {}                      # pci -> dict(at, ok, msg) of the last enforce write

RULE_FILE = "/etc/udev/rules.d/99-LexiPanel-gpu-powercap.rules"
RULE = ('ACTION=="add|change", SUBSYSTEM=="hwmon", ATTR{name}=="amdgpu", '
        'RUN+="/bin/sh -c \'chgrp admin /sys%p/power1_cap && chmod g+w /sys%p/power1_cap\'"')
GRANT_CMD = (f"echo '{RULE.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}' | sudo tee {RULE_FILE} "
             "&& sudo udevadm control --reload "
             "&& sudo udevadm trigger --action=change --subsystem-match=hwmon")


def bind(panel_module):
    global P, CAPS_FILE
    P = panel_module
    CAPS_FILE = P.PANEL / "gpu-power.json"


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _saved():
    try:
        return json.loads(CAPS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_saved(d):
    tmp = CAPS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, CAPS_FILE)


def _hwmon(pci):
    """The hwmon dir of this PCI device that carries power1_cap, or None."""
    for h in sorted(Path(f"/sys/bus/pci/devices/{pci}/hwmon").glob("hwmon*")):
        if (h / "power1_cap").exists():
            return h
    return None


def _watts(h, f):
    try:
        return round(int((h / f).read_text().strip()) / 1e6)
    except (OSError, ValueError):
        return None


def _device(d, saved):
    pci = d["pci"]
    rec = dict(pci=pci, name=d.get("name"), driver=d.get("driver"), supported=False,
               reason=None, cap_w=None, min_w=None, max_w=None, default_w=None,
               power_w=None, writable=False, saved_w=(saved.get(pci) or {}).get("watts"),
               saved_at=(saved.get(pci) or {}).get("set_at"), in_effect=None,
               last_enforce=_last.get(pci))
    h = _hwmon(pci) if d.get("driver") == "amdgpu" else None
    if h is None:
        rec["reason"] = {"nvidia": "NVIDIA caps need root on every change (nvidia-smi -pl); "
                                   "not settable from the panel.",
                         "nouveau": "nouveau exposes no power cap.",
                         None: "no driver bound."}.get(d.get("driver"),
                                                      "this driver exposes no power1_cap.")
        return rec
    cap = _watts(h, "power1_cap")
    rec.update(supported=True, cap_w=cap or None, min_w=_watts(h, "power1_cap_min"),
               max_w=_watts(h, "power1_cap_max"), default_w=_watts(h, "power1_cap_default"),
               power_w=_watts(h, "power1_average") or _watts(h, "power1_input"),
               writable=os.access(h / "power1_cap", os.W_OK))
    if rec["saved_w"] is not None:
        rec["in_effect"] = cap == rec["saved_w"]
    if not rec["writable"]:
        rec["reason"] = "power1_cap is root-only; run the one-time grant below."
    return rec


def status():
    saved = _saved()
    devs = [_device(d, saved) for d in P.gpu_devices(probe=False)
            if d["pci"] != "cpu" and d.get("vendor") in ("amd", "nvidia")]
    present = {d["pci"] for d in devs}
    # A saved cap for a card that is not in the box right now stays on file and
    # is re-applied when the card comes back; list it so it is not invisible.
    for pci, s in saved.items():
        if pci not in present:
            devs.append(dict(pci=pci, name=s.get("name"), driver=None, supported=False,
                             reason="not present in lspci right now; the saved cap is kept "
                                    "and re-applied if it returns.",
                             saved_w=s.get("watts"), saved_at=s.get("set_at"),
                             writable=False, in_effect=None, last_enforce=_last.get(pci)))
    return dict(devices=devs, grant_cmd=GRANT_CMD, rule_file=RULE_FILE,
                grant_needed=any(d["supported"] and not d["writable"] for d in devs),
                enforce_every_s=ENFORCE_S)


def _write(h, watts):
    (h / "power1_cap").write_text(str(int(watts) * 1000000))
    return _watts(h, "power1_cap")


def set_cap(pci, watts):
    d = next((x for x in status()["devices"] if x["pci"] == pci), None)
    if d is None:
        raise ValueError(f"no GPU at {pci}")
    if not d["supported"]:
        raise ValueError(d["reason"])
    try:
        watts = int(round(float(watts)))
    except (TypeError, ValueError):
        raise ValueError("watts must be a number")
    if not d["min_w"] <= watts <= d["max_w"]:
        raise ValueError(f"{watts} W is outside this card's range "
                         f"{d['min_w']}-{d['max_w']} W")
    if not d["writable"]:
        raise ValueError("power1_cap is root-only. Run this once, then press Apply again:\n"
                         + GRANT_CMD)
    with _lock:
        got = _write(_hwmon(pci), watts)
        if got != watts:
            raise ValueError(f"wrote {watts} W but the card reads back {got} W")
        saved = _saved()
        saved[pci] = dict(watts=watts, set_at=_now_iso(), name=d["name"])
        _write_saved(saved)
        _last[pci] = dict(at=_now_iso(), ok=True, msg=f"set to {watts} W from the panel")
    print(f"gpupower: {pci} cap set to {watts} W", flush=True)
    return status()


def clear(pci):
    with _lock:
        saved = _saved()
        if saved.pop(pci, None) is None:
            raise ValueError(f"no saved cap for {pci}")
        _write_saved(saved)
        _last.pop(pci, None)
    print(f"gpupower: {pci} saved cap cleared (live value kept until reboot)", flush=True)
    return status()


def enforce():
    """Re-apply every saved cap whose live value has drifted (reboot, GPU
    reset, card re-seated). Quiet when nothing changed."""
    with _lock:
        for pci, s in _saved().items():
            h = _hwmon(pci)
            want = s.get("watts")
            if h is None or want is None or _watts(h, "power1_cap") == want:
                continue
            try:
                got = _write(h, want)
                ok = got == want
                msg = f"re-applied {want} W" + ("" if ok else f", reads back {got} W")
            except OSError as e:
                ok, msg = False, f"could not re-apply {want} W: {e.strerror or e}"
            prev = _last.get(pci) or {}
            if prev.get("msg") != msg:
                print(f"gpupower: {pci} {msg}", flush=True)
            _last[pci] = dict(at=_now_iso(), ok=ok, msg=msg)


def enforcer():
    while True:
        try:
            enforce()
        except Exception as e:
            print(f"gpupower: enforce failed: {e}", flush=True)
        time.sleep(ENFORCE_S)
