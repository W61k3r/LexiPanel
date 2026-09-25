"""GPU Tuning: card reading, ROM parsing / checks and advice, on a fake /sys.
Run from the panel folder:  python3 -m unittest discover tests"""
import importlib.util, sys, tempfile, types, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fakesys  # noqa: E402
import gputune  # noqa: E402


def helper(sysroot):
    spec = importlib.util.spec_from_file_location("lexipanel_power_g", HERE.parent / "power" / "lexipanel_power.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.SYS = sysroot
    return m


DEV = dict(pci=fakesys.PCI, vendor="amd", driver="amdgpu", name="Navi 31 [Radeon RX 7900 XTX]",
           vram_total_mib=24560, notes=[])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.sys = root / "sys"
        self.home = root / "home"
        (self.home / "panel").mkdir(parents=True)
        fakesys.build(self.sys)
        lp = helper(str(self.sys))
        self.files = {}
        P = types.SimpleNamespace(
            PANEL=self.home / "panel",
            gpu_devices=lambda probe=True: [DEV, dict(pci="cpu", vendor="cpu")],
            filemgr=types.SimpleNamespace(resolve=lambda rel: self.home / rel, _root=lambda: self.home))
        PO = types.SimpleNamespace(lp=lambda: lp)
        O = types.SimpleNamespace(THERMAL_DEFAULTS=dict(pause_c=109, resume_c=100, abort_c=112, mem_abort_c=106))
        gputune.bind(P, PO, None, O)
        gputune.SYS = str(self.sys)

    def tearDown(self):
        self.tmp.cleanup()


class Cards(Base):
    def test_amd_card_reads_identity_sensors_clocks_and_overdrive(self):
        c = gputune.card(DEV)
        self.assertEqual((c["ids"]["vendor"], c["ids"]["device"], c["ids"]["subsystem"]), ("1002", "744c", "1002:0e3b"))
        self.assertEqual((c["board"], c["vbios"], c["vram"]["vendor"]), ("Radeon RX 7900 XTX", "113-D7020100-102", "hynix"))
        s = c["sensors"]
        self.assertEqual(s["temps"], dict(edge=61.0, junction=84.0, mem=76.0))
        self.assertEqual((s["crit"]["junction"], s["power_w"], s["cap_w"], s["cap_max_w"]), (110.0, 288.0, 303, 347))
        self.assertEqual((s["vddgfx_mv"], s["sclk_mhz"], s["mclk_mhz"]), (1012, 2482, 1249))
        self.assertEqual([x["mhz"] for x in c["clocks"]["sclk"] if x["current"]], [2482])
        self.assertTrue(c["overdrive"]["available"])
        self.assertEqual(c["overdrive"]["table"]["range"]["SCLK"], (500, 3150))
        self.assertEqual(c["power_profile"], "BOOTUP_DEFAULT")

    def test_overdrive_off_explains_how_to_turn_it_on(self):
        (self.sys / "bus/pci/devices" / fakesys.PCI / "pp_od_clk_voltage").unlink()
        (self.sys / "module/amdgpu/parameters").mkdir(parents=True)
        (self.sys / "module/amdgpu/parameters/ppfeaturemask").write_text("0xfff7bfff")
        c = gputune.card(DEV)
        self.assertFalse(c["overdrive"]["available"])
        self.assertIn("99-amdgpu-overdrive.cfg", c["overdrive"]["hint"])

    def test_sample_sums_power_and_keeps_the_hottest(self):
        t = gputune.sample([fakesys.PCI])
        self.assertEqual((t["power_w"], t["junction"], t["sclk"]), (288.0, 84.0, 2482))

    def test_only_tuning_knobs_are_accepted(self):
        with self.assertRaises(ValueError):
            gputune._clean([dict(knob="cpu.governor", target="all", value="performance")])
        with self.assertRaises(ValueError):
            gputune._clean([])
        self.assertEqual(gputune._clean([dict(knob="gpu.od_sclk", target=fakesys.PCI, value=2600)])[0]["value"], "2600")


class Profiles(Base):
    def setUp(self):
        super().setUp()
        self.saved, self.boot = {}, {}
        PO = gputune.PO
        PO.save_profile = lambda body: self.saved.setdefault(
            body.get("id") or body["name"], dict(id=body.get("id") or body["name"], name=body["name"],
            description=body.get("description"), settings=body["settings"], source=body.get("source")))
        PO.profiles = lambda v=None: [dict(id="builtin-stable", name="Stable", builtin=True,
                                           settings=[dict(knob="cpu.governor", target="all", value="performance")])] + list(self.saved.values())
        PO.settings = lambda: dict(boot_profile=self.boot)
        PO.delete_profile = lambda body: self.saved.pop(body["id"], None) and dict(ok=True)
        PO.apply = lambda body, why=None: dict(ok=True, applied=body)
        PO.persist = lambda body: self.boot.update(id=body["profile"]) or dict(ok=True)
        PO.unpersist = lambda body=None: self.boot.clear() or dict(ok=True)

    def test_save_lists_only_gpu_profiles_and_marks_the_boot_one(self):
        gputune.profile_save(dict(name="quiet uv", settings=[
            dict(knob="gpu.od_voltage", target=gputune.PCI if hasattr(gputune, "PCI") else "0000:03:00.0", value="-60"),
            dict(knob="gpu.od_mclk", target="0000:03:00.0", value="1400")]))
        ps = gputune._gpu_profiles()
        self.assertEqual([p["name"] for p in ps], ["quiet uv"])   # the cpu builtin is filtered out
        self.assertEqual(ps[0]["targets"], ["0000:03:00.0"])
        self.assertFalse(ps[0]["boot"])
        gputune.profile_persist(dict(id=ps[0]["id"]))
        self.assertTrue(gputune._gpu_profiles()[0]["boot"])

    def test_save_refuses_a_non_gpu_knob(self):
        with self.assertRaises(ValueError):
            gputune.profile_save(dict(name="x", settings=[dict(knob="cpu.governor", target="all", value="performance")]))

    def test_a_profile_mixing_in_a_non_gpu_knob_is_not_listed(self):
        self.saved["m"] = dict(id="m", name="mixed", settings=[
            dict(knob="gpu.od_mclk", target="0000:03:00.0", value="1400"),
            dict(knob="cpu.idle", target="all", value="C1")])
        self.assertEqual(gputune._gpu_profiles(), [])


class Roms(Base):
    def test_parse_rom_walks_the_image_chain(self):
        r = gputune.parse_rom(fakesys.rom_image())
        self.assertTrue(r["valid"])
        self.assertEqual((r["vendor"], r["device"], r["uefi"]), ("1002", "744c", True))
        self.assertEqual([i["code_type"] for i in r["images"]], ["x86 BIOS", "UEFI"])
        self.assertEqual((r["part_number"], r["build_date"]), ("113-D7020100-102", "2022/11/07 21:14"))

    def test_parse_rom_finds_the_image_after_a_dump_header_and_rejects_junk(self):
        r = gputune.parse_rom(b"NVGI" + b"\0" * 0x3fc + fakesys.rom_image(vendor=0x10de, device=0x2684, efi=False))
        self.assertEqual((r["valid"], r["vendor"], r["header_bytes"]), (True, "10de", 0x400))
        self.assertFalse(gputune.parse_rom(b"\xff" * 4096)["valid"])

    def _backup(self, data):
        folder = self.home / "panel" / "gpu-bios"
        folder.mkdir(exist_ok=True)
        meta = dict(gputune.parse_rom(data), file="b.rom", pci=fakesys.PCI)
        (folder / "b.json").write_text(__import__("json").dumps(meta))

    def _check(self, data):
        (self.home / "new.rom").write_bytes(data)
        return gputune.rom_check(dict(target=fakesys.PCI, path="new.rom"))

    def test_wrong_gpu_is_blocked_and_gets_no_command(self):
        self._backup(fakesys.rom_image())
        r = self._check(fakesys.rom_image(device=0x73bf))
        self.assertEqual(r["verdict"], "blocked")
        self.assertEqual(r["commands"], [])
        self.assertTrue(any("different GPU" in c["text"] for c in r["checks"]))

    def test_no_backup_blocks(self):
        self.assertEqual(self._check(fakesys.rom_image(part=b"113-D7020100-103"))["verdict"], "blocked")

    def test_matching_image_gets_the_vendor_command_and_is_never_flashed_here(self):
        self._backup(fakesys.rom_image())
        r = self._check(fakesys.rom_image(part=b"113-D7020100-102") + b"")
        # identical to the backup -> nothing to flash
        self.assertEqual(r["verdict"], "blocked")
        other = bytearray(fakesys.rom_image()); other[0x200] = 1
        r = self._check(bytes(other))
        self.assertEqual(r["verdict"], "ok", r["checks"])
        self.assertTrue(any("amdvbflash -p" in c for c in r["commands"]))

    def test_rom_file_name_cannot_escape_the_backup_folder(self):
        for bad in ("../../etc/passwd", "x.rom/../../y.rom", "a b.rom", ""):
            with self.assertRaises(ValueError):
                gputune.vbios_file(bad)


class Advice(Base):
    HELPER = dict(installed=True, sudo_ok=True, current=True, install_cmd="x")

    def run_(self, **summary):
        base = dict(decode_tps=40, mean_power_w=290, power_frac=0.97, peak_junction=95, mean_busy=97, spread_pct=2)
        base.update(summary)
        return dict(id="20260925_000000", devices=[fakesys.PCI], verdict="stable", summary=base,
                    model_name="m", depth=2048, n_predict=256)

    def texts(self, hist):
        return " | ".join(a["text"] for a in gputune.advice([gputune.card(DEV)], hist, self.HELPER))

    def test_power_limited_run_suggests_undervolt(self):
        self.assertIn("Power-limited", self.texts([self.run_()]))

    def test_low_load_says_the_limit_is_elsewhere(self):
        self.assertIn("only 40% busy", self.texts([self.run_(mean_busy=40, power_frac=0.5)]))

    def test_no_runs_asks_for_a_stock_baseline(self):
        self.assertIn("Run one at stock first", self.texts([]))

    def test_unstable_runs_are_not_evidence(self):
        r = self.run_(); r["verdict"] = "unstable"
        self.assertIn("Run one at stock first", self.texts([r]))


if __name__ == "__main__":
    unittest.main()
