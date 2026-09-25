"""A fake /sys for tests: one RDNA3 card (7900 XTX shaped) with OverDrive, and a
kernel stand-in for pp_od_clk_voltage that behaves like amdgpu's SMU13 code:
edits are staged, 'c' commits them, 'r' restores the stock table, values outside
OD_RANGE are refused with EINVAL."""
import errno, os, re
from pathlib import Path

PCI = "0000:03:00.0"

STOCK = dict(sclk=(500, 2500), mclk=(97, 1250), vo=0)
RANGE = dict(SCLK=(500, 3150), MCLK=(97, 1500), VDDGFX_OFFSET=(-450, 0))


def od_text(t):
    return (f"OD_SCLK:\n0: {t['sclk'][0]}Mhz\n1: {t['sclk'][1]}Mhz\n"
            f"OD_MCLK:\n0: {t['mclk'][0]}Mhz\n1: {t['mclk'][1]}MHz\n"
            f"OD_VDDGFX_OFFSET:\n{t['vo']}mV\n"
            "OD_RANGE:\n"
            f"SCLK: {RANGE['SCLK'][0]:>7}Mhz {RANGE['SCLK'][1]:>10}Mhz\n"
            f"MCLK: {RANGE['MCLK'][0]:>7}Mhz {RANGE['MCLK'][1]:>10}Mhz\n"
            f"VDDGFX_OFFSET: {RANGE['VDDGFX_OFFSET'][0]:>7}mv {RANGE['VDDGFX_OFFSET'][1]:>10}mv\n")


def build(root, od=True, vbios=b""):
    """Write the tree under root; returns the device dir."""
    root = Path(root)
    d = root / "bus/pci/devices" / PCI
    (d / "hwmon/hwmon3").mkdir(parents=True, exist_ok=True)
    drv = root / "bus/pci/drivers/amdgpu"
    drv.mkdir(parents=True, exist_ok=True)
    if not (d / "driver").exists():
        os.symlink(drv, d / "driver")
    files = {
        "class": "0x030000", "vendor": "0x1002", "device": "0x744c",
        "subsystem_vendor": "0x1002", "subsystem_device": "0x0e3b",
        "vbios_version": "113-D7020100-102", "product_name": "Radeon RX 7900 XTX",
        "mem_info_vram_total": str(24 << 30), "mem_info_vram_used": str(18 << 30),
        "mem_info_vram_vendor": "hynix", "gpu_busy_percent": "97", "mem_busy_percent": "41",
        "current_link_speed": "16.0 GT/s PCIe", "current_link_width": "16",
        "max_link_speed": "16.0 GT/s PCIe", "max_link_width": "16",
        "power_dpm_force_performance_level": "auto",
        "pp_dpm_sclk": "0: 500Mhz\n1: 1650Mhz\n2: 2482Mhz *\n",
        "pp_dpm_mclk": "0: 96Mhz\n1: 456Mhz\n2: 772Mhz\n3: 1249Mhz *\n",
        "pp_dpm_fclk": "0: 800Mhz\n1: 1900Mhz *\n",
        "pp_power_profile_mode": "PROFILE_INDEX(NAME)\n  0 BOOTUP_DEFAULT*:\n  1 3D_FULL_SCREEN:\n  5 COMPUTE:\n",
        "hwmon/hwmon3/name": "amdgpu",
        "hwmon/hwmon3/temp1_input": "61000", "hwmon/hwmon3/temp1_label": "edge",
        "hwmon/hwmon3/temp2_input": "84000", "hwmon/hwmon3/temp2_label": "junction",
        "hwmon/hwmon3/temp2_crit": "110000",
        "hwmon/hwmon3/temp3_input": "76000", "hwmon/hwmon3/temp3_label": "mem",
        "hwmon/hwmon3/power1_average": "288000000", "hwmon/hwmon3/power1_cap": "303000000",
        "hwmon/hwmon3/power1_cap_min": "202000000", "hwmon/hwmon3/power1_cap_max": "347000000",
        "hwmon/hwmon3/power1_cap_default": "303000000",
        "hwmon/hwmon3/fan1_input": "1850", "hwmon/hwmon3/fan1_max": "3300",
        "hwmon/hwmon3/in0_input": "1012", "hwmon/hwmon3/freq1_input": "2482000000",
        "hwmon/hwmon3/freq2_input": "1249000000",
    }
    if od:
        files["pp_od_clk_voltage"] = od_text(STOCK)
    for k, v in files.items():
        (d / k).write_text(v)
    if vbios:
        (d / "rom").write_bytes(vbios)
    return d


class OdKernel:
    """Wraps lexipanel_power._w so writes to pp_od_clk_voltage act like the kernel."""
    def __init__(self, mod):
        self.mod, self.real = mod, mod._w
        self.live = dict(STOCK)
        self.staged = dict(STOCK)
        self.writes = []

    def __call__(self, path, text):
        if not path.endswith("pp_od_clk_voltage"):
            return self.real(path, text)
        self.writes.append(text)
        m = re.fullmatch(r"([sm]) (\d) (\d+)", text) or re.fullmatch(r"(vo) (-?\d+)", text)
        if text == "c":
            self.live = dict(self.staged)
        elif text == "r":
            self.staged = dict(STOCK)
        elif m and m.group(1) in "sm":
            key, idx, v = ("sclk" if m.group(1) == "s" else "mclk"), int(m.group(2)), int(m.group(3))
            lo, hi = RANGE["SCLK" if key == "sclk" else "MCLK"]
            if not lo <= v <= hi:
                raise OSError(errno.EINVAL, "Invalid argument")
            pair = list(self.staged[key]); pair[idx] = v; self.staged[key] = tuple(pair)
        elif m:
            v = int(m.group(2))
            lo, hi = RANGE["VDDGFX_OFFSET"]
            if not lo <= v <= hi:
                raise OSError(errno.EINVAL, "Invalid argument")
            self.staged["vo"] = v
        else:
            raise OSError(errno.EINVAL, "Invalid argument")
        self.real(path, od_text(self.live))


def rom_image(vendor=0x1002, device=0x744c, size=0x40000, efi=True, part=b"113-D7020100-102"):
    """A minimal PCI option ROM: legacy x86 image + optional EFI image, each with
    0x55AA, a PCIR structure and the 'last image' bit on the final one."""
    def image(code_type, last, length):
        b = bytearray(length)
        b[0:2] = b"\x55\xaa"
        b[2] = length // 512 & 0xff
        b[0x18:0x1a] = (0x40).to_bytes(2, "little")
        p = 0x40
        b[p:p + 4] = b"PCIR"
        b[p + 4:p + 6] = vendor.to_bytes(2, "little")
        b[p + 6:p + 8] = device.to_bytes(2, "little")
        b[p + 0x10:p + 0x12] = (length // 512).to_bytes(2, "little")
        b[p + 0x14] = code_type
        b[p + 0x15] = 0x80 if last else 0
        if code_type == 0:
            b[0x80:0x80 + len(part)] = part
            b[0x100:0x113] = b"2022/11/07 21:14   "
        return bytes(b)
    legacy = image(0, not efi, 0x10000)
    out = legacy + (image(3, True, 0x10000) if efi else b"")
    return out + b"\xff" * (size - len(out))
