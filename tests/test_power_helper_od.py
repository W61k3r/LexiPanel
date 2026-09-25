"""The root helper's amdgpu OverDrive knobs and vbios verb, on a fake /sys.
Run from the panel folder:  python3 -m unittest discover tests"""
import base64, importlib.util, os, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fakesys  # noqa: E402


def load_helper():
    spec = importlib.util.spec_from_file_location("lexipanel_power_t",
                                                  HERE.parent / "power" / "lexipanel_power.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class OverDriveKnobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lp = load_helper()
        self.lp.SYS = self.tmp.name
        fakesys.build(self.tmp.name)
        self.kernel = fakesys.OdKernel(self.lp)
        self.lp._w = self.kernel

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        return {r["knob"]: r for r in self.lp.table() if r.get("target") == fakesys.PCI}

    def test_table_lists_od_knobs_with_kernel_ranges(self):
        r = self.rows()
        self.assertEqual(r["gpu.od_sclk"]["value"], "2500")
        self.assertEqual(r["gpu.od_sclk"]["choices"], dict(kind="range", min=500, max=3150, unit="MHz"))
        self.assertEqual(r["gpu.od_mclk"]["value"], "1250")
        self.assertEqual(r["gpu.od_mclk"]["choices"]["max"], 1500)
        self.assertEqual(r["gpu.od_voltage"]["value"], "0")
        self.assertEqual(r["gpu.od_voltage"]["choices"], dict(kind="range", min=-450, max=0, unit="mV"))

    def test_apply_writes_top_of_pair_then_commits_and_reads_back(self):
        res = self.lp.apply([dict(knob="gpu.od_sclk", target=fakesys.PCI, value="2750"),
                             dict(knob="gpu.od_voltage", target=fakesys.PCI, value="-60")])
        self.assertTrue(all(x["ok"] for x in res), res)
        self.assertEqual(self.kernel.writes, ["s 1 2750", "c", "vo -60", "c"])
        r = self.rows()
        self.assertEqual((r["gpu.od_sclk"]["value"], r["gpu.od_voltage"]["value"]), ("2750", "-60"))

    def test_out_of_range_and_garbage_are_refused_before_any_write(self):
        for knob, v in (("gpu.od_sclk", "3151"), ("gpu.od_sclk", "499"), ("gpu.od_mclk", "2500 c"),
                        ("gpu.od_voltage", "10"), ("gpu.od_voltage", "-451"), ("gpu.od_sclk", "$(reboot)")):
            with self.assertRaises(ValueError, msg=f"{knob}={v}"):
                self.lp.apply([dict(knob=knob, target=fakesys.PCI, value=v)])
        self.assertEqual(self.kernel.writes, [])

    def test_unknown_target_is_not_written(self):
        res = self.lp.apply([dict(knob="gpu.od_sclk", target="0000:09:00.0", value="2600")])
        self.assertFalse(res[0]["ok"])
        self.assertEqual(self.kernel.writes, [])

    def test_no_overdrive_means_no_targets(self):
        os.remove(Path(self.tmp.name) / "bus/pci/devices" / fakesys.PCI / "pp_od_clk_voltage")
        self.assertNotIn("gpu.od_sclk", self.rows())

    def test_rdna2_without_voltage_range_is_capped_at_undervolt_only(self):
        f = Path(self.tmp.name) / "bus/pci/devices" / fakesys.PCI / "pp_od_clk_voltage"
        f.write_text("\n".join(l for l in f.read_text().splitlines() if not l.startswith("VDDGFX_OFFSET")))
        ch = self.rows()["gpu.od_voltage"]["choices"]
        self.assertEqual((ch["min"], ch["max"]), (self.lp.GpuOdVoltage.FLOOR, 0))


class VbiosVerb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lp = load_helper()
        self.lp.SYS = self.tmp.name
        self.rom = fakesys.rom_image()
        fakesys.build(self.tmp.name, vbios=self.rom)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_rom_bar_and_leaves_it_disabled(self):
        writes = []
        real = self.lp._w
        self.lp._w = lambda p, t: writes.append((os.path.basename(p), t)) if p.endswith("/rom") else real(p, t)
        out = self.lp.vbios(fakesys.PCI)
        self.assertEqual(base64.b64decode(out["data"]), self.rom)
        self.assertEqual(writes, [("rom", "1"), ("rom", "0")])

    def test_prefers_amdgpu_debugfs_copy(self):
        dbg = Path(self.tmp.name) / "kernel/debug/dri" / fakesys.PCI
        dbg.mkdir(parents=True)
        (dbg / "amdgpu_vbios").write_bytes(b"\x55\xaa" + b"x" * 30)
        out = self.lp.vbios(fakesys.PCI)
        self.assertTrue(out["source"].endswith("amdgpu_vbios"))

    def test_refuses_non_gpu_and_unknown_targets(self):
        for t in ("0000:09:00.0", "../../etc", ""):
            with self.assertRaises(ValueError):
                self.lp.vbios(t)


if __name__ == "__main__":
    unittest.main()
