#!/usr/bin/env python3
"""
main_devices.py <LIB_DIR> <BACKEND>

Device pinning for main, printed as shell assignments for
run_llama_vulkan.sh to eval. Only called when panel/main-devices.json
exists; without it the script keeps its original RADV-only pinning.

Uses panel.vulkan_pinning(), the same function every other instance's launch
plan uses, so main and the panel cannot disagree about which card is Vulkan0.

Prints:
  MAIN_VK_ICDS        colon-joined ICD list for VK_DRIVER_FILES/VK_ICD_FILENAMES
  MAIN_VK_VISIBLE     "0" for a single device, empty for several (unset it)
  MAIN_VK_DEVICE      "--device" value, e.g. "Vulkan1,Vulkan0"; empty for one device
  MAIN_AMD_ONLY       "1" when the selection is the single amdgpu card - the only
                      layout the script's XTX placement assertion applies to
  MAIN_DEVICE_NAMES   human-readable, for the launch banner

Exit 3 with the reasons on stderr when the selection cannot be pinned; the
script aborts rather than silently falling back to a different card.
"""
import shlex, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P                                   # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: main_devices.py <LIB_DIR> <BACKEND>")
    bindir, backend = sys.argv[1], sys.argv[2]
    inst = P.get_instance("main")
    alldev = {d["pci"]: d for d in P.gpu_devices(probe=False)}
    errors, devs = [], []
    for pci in inst["devices"]:
        d = alldev.get(pci)
        if not d:
            errors.append(f"device {pci} is not present on this host")
            continue
        if backend not in d["backends"]:
            errors.append(f"BACKEND={backend} cannot drive {d['name']}")
        devs.append(d)
    if backend != "vulkan" and len(inst["devices"]) > 1:
        errors.append(f"main has {len(inst['devices'])} devices selected; that needs "
                      f"BACKEND=vulkan, not {backend}")
    if backend != "vulkan" or errors:
        if errors:
            print("\n".join(errors), file=sys.stderr)
            sys.exit(3)
        return
    with P.using_instance(inst):
        env, vk_map, verr = P.vulkan_pinning(devs, bindir)
    if verr:
        print("\n".join(verr), file=sys.stderr)
        sys.exit(3)
    out = dict(MAIN_VK_ICDS=env["VK_DRIVER_FILES"],
               MAIN_VK_VISIBLE=env.get("GGML_VK_VISIBLE_DEVICES", ""),
               MAIN_VK_DEVICE=",".join(n for n, _ in vk_map),
               MAIN_AMD_ONLY="1" if [d["driver"] for d in devs] == ["amdgpu"] else "0",
               MAIN_DEVICE_NAMES="; ".join(
                   (f"{n}=" if vk_map else "") + d["name"]
                   for n, d in (vk_map or [("", d) for d in devs])))
    for k, v in out.items():
        print(f"{k}={shlex.quote(v)}")


if __name__ == "__main__":
    main()
