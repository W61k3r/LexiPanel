"""Access: modes, roles on every POST route, users, API keys, the audit chain.
Run from the panel folder:  python3 -m unittest discover tests"""
import base64, json, re, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import auth as A  # noqa: E402

basic = lambda u, p: {"Authorization": "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()}


class Access(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        A.bind(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def multi(self):
        A.set_mode("multi", "root-admin", "correct horse battery")
        A.set_user("vic", "viewer-pass-123", "viewer")
        A.set_user("oli", "operator-pass-1", "operator")

    def test_single_mode_changes_nothing(self):
        self.assertEqual(A.check("POST", "/api/params", {}), (0, "local", "admin"))

    def test_every_post_route_has_a_role_and_none_is_open_to_viewers(self):
        src = (ROOT / "panel.py").read_text()
        post = src[src.index("    def do_POST(self):"):]
        routes = set(re.findall(r'p == "(/api/[^"]+)"', post)) | set(re.findall(r'p\.startswith\("(/api/[^"]+)"\)', post))
        self.assertGreater(len(routes), 100)
        for r in routes:
            need = A.needed("POST", r)
            if r in A.OPEN:
                self.assertIsNone(need, r)
            elif r in A.ANY_POST:
                self.assertEqual(need, "viewer", r)
            else:
                self.assertIn(need, ("operator", "admin"), r)
        for r in ("/api/params", "/api/power/apply", "/api/gputune/apply", "/api/files/delete", "/api/auth/users",
                  "/api/auth/mode", "/api/fleet/settings", "/api/instance/delete", "/api/builds/install"):
            self.assertEqual(A.needed("POST", r), "admin", r)

    def test_roles_and_logins(self):
        self.multi()
        self.assertEqual(A.check("GET", "/api/status", {})[0], 401)
        self.assertEqual(A.check("GET", "/api/status", basic("vic", "wrong-password"))[0], 401)
        self.assertEqual(A.check("GET", "/api/status", basic("vic", "viewer-pass-123"))[0], 0)
        self.assertEqual(A.check("POST", "/api/start", basic("vic", "viewer-pass-123"))[0], 403)
        self.assertEqual(A.check("POST", "/api/start", basic("oli", "operator-pass-1"))[0], 0)
        self.assertEqual(A.check("POST", "/api/params", basic("oli", "operator-pass-1"))[0], 403)
        self.assertEqual(A.check("GET", "/api/files", basic("oli", "operator-pass-1"))[0], 403)
        self.assertEqual(A.check("POST", "/api/params", basic("root-admin", "correct horse battery"))[0], 0)
        self.assertEqual(A.check("POST", "/api/fleet/report", {})[0], 0)            # token-checked by fleet.py

    def test_api_keys_are_capped_expire_and_revoke(self):
        self.multi()
        with self.assertRaises(ValueError):
            A.create_key("oli", "admin")                                          # above the user's role
        k = A.create_key("oli", "viewer", days=1)
        h = {"Authorization": "Bearer " + k["key"]}
        self.assertEqual(A.check("GET", "/api/status", h)[:3], (0, "oli", "viewer"))
        self.assertEqual(A.check("POST", "/api/start", h)[0], 403)
        A.revoke_key(k["id"])
        A._cache.clear()
        self.assertEqual(A.check("GET", "/api/status", h)[0], 401)
        self.assertNotIn(k["key"].split("_")[-1], (Path(self.tmp.name) / "auth/keys.json").read_text() if
                         (Path(self.tmp.name) / "auth/keys.json").exists() else "")

    def test_no_lockout_by_accident(self):
        with self.assertRaises(ValueError):
            A.set_mode("multi")                                                   # no admin yet
        self.multi()
        with self.assertRaises(ValueError):
            A.delete_user("root-admin")
        with self.assertRaises(ValueError):
            A.set_user("x", "short", "viewer")
        self.assertNotIn("correct horse", (Path(self.tmp.name) / "auth/users.json").read_text())

    def test_audit_chain_detects_an_edit(self):
        for i in range(5):
            A.audit("oli", "operator", "POST", "/api/start", "main", "allowed")
        self.assertEqual(A.verify(), dict(ok=True, lines=5))
        f = Path(self.tmp.name) / "auth/audit.jsonl"
        lines = f.read_text().splitlines()
        r = json.loads(lines[2]); r["user"] = "someone-else"; lines[2] = json.dumps(r)
        f.write_text("\n".join(lines) + "\n")
        self.assertEqual(A.verify()["broken_at"], 3)

    def test_internal_calls_act_as_their_caller_only_with_the_process_key(self):
        self.multi()
        self.assertEqual(A.check("POST", "/api/start", {"X-LexiPanel-Internal": f"{A.INTERNAL}:vic"})[0], 403)
        self.assertEqual(A.check("GET", "/api/status", {"X-LexiPanel-Internal": "guess:root-admin"})[0], 401)


if __name__ == "__main__":
    unittest.main()
