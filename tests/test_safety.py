"""Safety tripwires: what LexiPanel must never do, checked in the source on every run.
Broken tuning code in a development build before 1.0.0 ran a MOTHERBOARD flasher (`flashrom -p internal`); these fail
the build if anything like it comes back.  python3 -m unittest discover tests"""
import ast, re, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
FLASHERS = re.compile(r"\b(flashrom|amdvbflash|atiflash|atiwinflash|nvflash(64)?|amdgpuflash)\b", re.I)
# May NAME a flasher, as text for you to run yourself; must never execute anything.
DISPLAY_ONLY = {("gputune.py", "rom_check")}
# May write under /sys: the root helper, and the range-checked power cap after its one-time grant.
SYS_WRITERS = {("power/lexipanel_power.py", "*"), ("gpupower.py", "_write")}
SYSLIKE = re.compile(r"/sys\b|\bSYS\b|\bCARD\b|hwmon|pp_od_clk|power1_cap|/dev/(mem|port|mtd)")
EXEC_ATTRS = {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
              "system", "popen", "execv", "execvp", "execve", "spawnv", "create_subprocess_exec",
              "create_subprocess_shell"}
EXEC_NAMES = {"sh", "_sh", "_run", "_sudo", "_systemctl"}
HELPER_VERBS = {"status", "apply", "persist", "unpersist", "vbios"}      # the sudoers rule's five


def shipped(*globs):
    out = []
    for g in globs:
        out += [p for p in ROOT.rglob(g) if "tests" not in p.relative_to(ROOT).parts
                and ".git" not in p.parts and "node_modules" not in p.parts]
    return sorted(out)


def functions(tree):
    """(qualname, node) of every function, plus ('<module>', tree)."""
    out = [("<module>", tree)]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((n.name, n))
    return out


def own_nodes(fn):
    """Nodes of fn, not of functions nested in it; docstrings left out."""
    doc = set()
    for n in [fn] + [x for x in ast.walk(fn) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
        b = getattr(n, "body", None)
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant):
            doc.add(id(b[0].value))
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if id(n) not in doc:
            yield n
        stack.extend(ast.iter_child_nodes(n))


def is_exec(call):
    f = call.func
    return (isinstance(f, ast.Attribute) and f.attr in EXEC_ATTRS | EXEC_NAMES) or \
           (isinstance(f, ast.Name) and f.id in EXEC_NAMES | EXEC_ATTRS)


def py_violations(path, rel):
    tree = ast.parse(path.read_text())
    bad = []
    for name, fn in functions(tree):
        nodes = list(own_nodes(fn))
        strs = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if any(FLASHERS.search(s) for s in strs):
            if (rel, name) not in DISPLAY_ONLY:
                bad.append(f"{rel}:{name} names a firmware flasher")
            elif any(isinstance(n, ast.Call) and is_exec(n) for n in nodes):
                bad.append(f"{rel}:{name} names a flasher AND executes commands")
        assigned = {}
        for n in nodes:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                assigned[n.targets[0].id] = ast.unparse(n.value)
        for n in nodes:
            if not isinstance(n, ast.Call):
                continue
            f, target = n.func, None
            if isinstance(f, ast.Name) and f.id == "open" and n.args:
                mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
                if isinstance(mode, ast.Constant) and re.search(r"[wax+]", str(mode.value)):
                    target = n.args[0]
            elif isinstance(f, ast.Attribute) and f.attr in ("write_text", "write_bytes"):
                target = f.value
            if target is not None:
                src = ast.unparse(target)
                src += " " + " ".join(assigned.get(x.id, "") for x in ast.walk(target) if isinstance(x, ast.Name))
                if SYSLIKE.search(src) and (rel, "*") not in SYS_WRITERS and (rel, name) not in SYS_WRITERS:
                    bad.append(f"{rel}:{name} writes {ast.unparse(target)} (sysfs/device) outside the helper")
            if is_exec(n):
                txt = " ".join(s.value for s in ast.walk(n) if isinstance(s, ast.Constant) and isinstance(s.value, str))
                if re.search(r"(\btee\b|>)\s*\S*/sys/", txt) and not rel.startswith("power/"):
                    bad.append(f"{rel}:{name} writes /sys through a shell command")
    return bad


def sh_violations(path, rel):
    bad = []
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r"\s#.*$", "", s)
        if FLASHERS.search(s) and not re.match(r"(echo|printf)\b", s):
            bad.append(f"{rel}:{i} runs a firmware flasher: {s[:80]}")
        if re.search(r"(\btee\b[^|]*|>)\s*/sys/", s) and not rel.startswith("power/"):
            bad.append(f"{rel}:{i} writes /sys: {s[:80]}")
    return bad


def sudoers_rules(path):
    txt = re.sub(r"\\\n", " ", path.read_text())
    return [l.strip() for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]


class Tripwires(unittest.TestCase):
    def test_no_flasher_runs_and_sysfs_is_written_only_by_the_helper(self):
        bad = []
        for p in shipped("*.py"):
            bad += py_violations(p, p.relative_to(ROOT).as_posix())
        for p in shipped("*.sh", "*.service"):
            bad += sh_violations(p, p.relative_to(ROOT).as_posix())
        self.assertEqual(bad, [])

    def test_the_tripwires_catch_what_they_are_for(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.py"
            f.write_text("import subprocess\ndef flash(p):\n    subprocess.run(['flashrom', '-p', 'internal', '-w', p])\n"
                         "def oc(card):\n    path = f'{CARD}/pp_od_clk_voltage'\n    open(path, 'w').write('s 1 3200')\n"
                         "def rom_check(b):\n    import os\n    os.system('amdvbflash -p 0 x.rom')\n")
            got = py_violations(f, "x.py")
            self.assertEqual(len([g for g in got if "flasher" in g]), 2, got)
            self.assertTrue(any("writes path" in g for g in got), got)
            s = Path(d) / "x.sh"
            s.write_text("# flashrom is never used\necho 'run: sudo amdvbflash -p 0 f.rom'\nsudo flashrom -p internal -w bios.bin\n"
                         "echo 3200 | sudo tee /sys/class/drm/card1/device/pp_od_clk_voltage\n")
            self.assertEqual(len(sh_violations(s, "x.sh")), 2)

    def test_sudoers_grant_exact_commands_only(self):
        for p in shipped("*.sudoers"):
            for rule in sudoers_rules(p):
                self.assertIn("NOPASSWD:", rule, p)
                cmds = [c.strip() for c in rule.split("NOPASSWD:", 1)[1].split(",")]
                for c in cmds:
                    self.assertNotIn("*", c, f"{p.name}: wildcard in {c!r}")
                    self.assertNotEqual(c, "ALL", f"{p.name}: grants ALL")
                    ok = (re.fullmatch(r"/usr/local/sbin/lexipanel-power (\w+)", c)
                          and c.split()[1] in HELPER_VERBS) or \
                         re.fullmatch(r"/usr/bin/systemctl (start|stop|restart|reset-failed) [\w@.-]+", c)
                    self.assertTrue(ok, f"{p.name}: {c!r} is not one of the allowed commands")

    def test_helper_verbs_line_up(self):
        granted = {c.split()[1] for r in sudoers_rules(ROOT / "power/lexipanel-power.sudoers")
                   for c in r.split("NOPASSWD:", 1)[1].split(",")}
        self.assertEqual(granted, HELPER_VERBS)                      # "boot" is for the unit, never sudo
        called = set()
        for p in shipped("*.py"):
            for n in ast.walk(ast.parse(p.read_text())):
                if (isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
                        and getattr(n.func, "attr", getattr(n.func, "id", "")) == "_sudo"
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    called.add(n.args[0].value)
        self.assertTrue(called <= granted, f"code calls helper verbs sudo does not grant: {called - granted}")
        src = (ROOT / "power/lexipanel_power.py").read_text()
        for v in granted | {"boot"}:
            self.assertRegex(src, rf'verb (== |in \([^)]*)"{v}"', f"the helper does not dispatch {v}")


class FleetDirection(unittest.TestCase):
    """The primary never connects to a member: every request fleet code makes is a member posting
    to its primary's join, report or event route (remote actions ride back in the replies)."""
    ALLOWED = {"/api/fleet/join", "/api/fleet/report", "/api/fleet/event"}

    def test_fleet_only_posts_to_the_primarys_member_routes(self):
        tree = ast.parse((ROOT / "fleet.py").read_text())
        paths, calls = set(), 0
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_post":
                calls += 1
                url = ast.unparse(n.args[0])
                self.assertTrue(url.startswith("c['primary_url'] + "), url)
                paths |= {s.value for s in ast.walk(n.args[0]) if isinstance(s, ast.Constant) and isinstance(s.value, str)
                          and s.value.startswith("/")}
            if isinstance(n, ast.Attribute) and n.attr in ("urlopen", "HTTPConnection", "HTTPSConnection", "create_connection"):
                self.assertTrue(any(isinstance(f, ast.FunctionDef) and f.name == "_post" and n in ast.walk(f)
                                    for f in ast.walk(tree)), f"a connection outside _post: {ast.unparse(n)}")
        self.assertGreaterEqual(calls, 3)
        self.assertEqual(paths, self.ALLOWED)


class DangerousValues(unittest.TestCase):
    """The helper, not the UI, is the last line: it refuses these whatever the kernel prints."""
    def setUp(self):
        import fakesys
        from test_power_helper_od import load_helper
        self.fakesys, self.tmp = fakesys, tempfile.TemporaryDirectory()
        self.lp = load_helper()
        self.lp.SYS = self.tmp.name
        fakesys.build(self.tmp.name)
        self.kernel = fakesys.OdKernel(self.lp)
        self.lp._w = self.kernel
        self.od = Path(self.tmp.name) / "bus/pci/devices" / fakesys.PCI / "pp_od_clk_voltage"

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, knob, value):
        try:
            self.lp.apply([dict(knob=knob, target=self.fakesys.PCI, value=value)])
        except ValueError:
            return True
        return False

    def test_no_overvolt_even_if_the_kernel_range_allows_it(self):
        txt = self.od.read_text()
        self.od.write_text(re.sub(r"(VDDGFX_OFFSET:\s*-?\d+mv\s+)0mv", r"\g<1>100mv", txt))
        self.assertNotEqual(txt, self.od.read_text())              # the kernel now "allows" +100 mV
        self.assertTrue(self.refused("gpu.od_voltage", "50"))
        self.assertEqual(self.kernel.writes, [])

    def test_clock_past_the_kernel_range_and_shell_tricks_are_refused_unwritten(self):
        for knob, v in (("gpu.od_sclk", "99999"), ("gpu.od_sclk", "2500; reboot"), ("gpu.od_mclk", "-1"),
                        ("gpu.od_voltage", "-5000"), ("gpu.od_voltage", "0 && c")):
            self.assertTrue(self.refused(knob, v), (knob, v))
        self.assertEqual(self.kernel.writes, [])


if __name__ == "__main__":
    unittest.main()
