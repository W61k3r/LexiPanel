"""Fleet: join codes, reports, revoke, stale / offline, untrusted input.
Run from the panel folder:  python3 -m unittest discover tests"""
import json, sys, tempfile, time, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fleet as F  # noqa: E402


class Primary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        F.bind(types.SimpleNamespace(PANEL=Path(self.tmp.name)))
        F.set_config(dict(role="primary", name="prime"))
        self.box = "12345678-1234-1234-1234-123456789abc"

    def tearDown(self):
        self.tmp.cleanup()

    def joined(self):
        return F.join(dict(code=F.new_join_code()["code"], box_id=self.box, name="rig-2"))["token"]

    def report(self, token, **kw):
        body = json.dumps(dict(dict(box_id=self.box, name="rig-2", instances=[dict(id="main", state="running")]), **kw))
        return F.intake({"Authorization": f"Bearer {token}"}, body.encode())

    def test_join_report_list_revoke(self):
        tok = self.joined()
        self.assertTrue(self.report(tok)["ok"])
        b = F.boxes()[0]
        self.assertEqual((b["name"], b["state"]), ("rig-2", "online"))
        self.assertEqual(b["report"]["instances"][0]["state"], "running")
        self.assertNotIn(tok, (Path(self.tmp.name) / "fleet/boxes" / f"{self.box}.json").read_text())  # hash only
        F.revoke(self.box)
        with self.assertRaises(PermissionError):
            self.report(tok)

    def test_codes_are_single_use_and_expire(self):
        code = F.new_join_code()["code"]
        F.join(dict(code=code, box_id=self.box))
        with self.assertRaises(PermissionError):
            F.join(dict(code=code, box_id=self.box))
        code = F.new_join_code()["code"]
        j = json.loads((Path(self.tmp.name) / "fleet/joins.json").read_text())
        (Path(self.tmp.name) / "fleet/joins.json").write_text(json.dumps({k: 1 for k in j}))
        with self.assertRaises(PermissionError):
            F.join(dict(code=code, box_id=self.box))

    def test_bad_tokens_floods_and_oversized_reports_are_refused(self):
        tok = self.joined()
        with self.assertRaises(PermissionError):
            self.report("nope")
        self.report(tok)
        with self.assertRaises(ValueError):
            self.report(tok)                                       # again within 20 s
        with self.assertRaises(ValueError):
            F.intake({"Authorization": f"Bearer {tok}"}, b"x" * (F.MAX_BYTES + 1))

    def test_untrusted_strings_are_capped_and_state_ages(self):
        tok = self.joined()
        self.report(tok, name="n" * 500, junk=["x" * 1000] * 100)
        b = F.boxes()[0]
        self.assertEqual(len(b["name"]), 64)
        self.assertEqual(len(b["report"]["junk"]), 64)
        self.assertEqual(len(b["report"]["junk"][0]), 200)
        self.assertEqual(F.boxes(now=time.time() + 400)[0]["state"], "stale")
        self.assertEqual(F.boxes(now=time.time() + 700)[0]["state"], "offline")

    def test_only_a_primary_accepts(self):
        F.set_config(dict(role="standalone"))
        with self.assertRaises(ValueError):
            F.new_join_code()
        with self.assertRaises(PermissionError):
            F.join(dict(code="x", box_id=self.box))


if __name__ == "__main__":
    unittest.main()
