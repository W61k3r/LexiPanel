#!/usr/bin/env python3
"""The panel reads a request body into memory, so its declared size is checked first: a false
Content-Length must not make it read gigabytes, least of all on the fleet routes that answer
without a login. panel.body_limit is pure; it is lifted out of panel.py so no panel starts."""
import ast, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = {}
for node in ast.parse((ROOT / "panel.py").read_text()).body:
    if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("MAX_BODY", "OPEN_MAX_BODY")) or \
            (isinstance(node, ast.FunctionDef) and node.name == "body_limit"):
        exec(compile(ast.Module([node], []), "panel.py", "exec"), NS)
limit, MAX, OPEN_MAX = NS["body_limit"], NS["MAX_BODY"], NS["OPEN_MAX_BODY"]
OPEN = ("/api/fleet/report", "/api/fleet/join", "/api/fleet/event")


class BodyLimit(unittest.TestCase):
    def test_open_routes_take_a_report_and_no_more(self):
        for p in OPEN:
            self.assertIsNone(limit(p, str(OPEN_MAX), OPEN))
            self.assertEqual(limit(p, str(OPEN_MAX + 1), OPEN)[0], 413)
            self.assertEqual(limit(p, "99999999999", OPEN)[0], 413)

    def test_logged_in_routes_have_a_ceiling_too(self):
        self.assertIsNone(limit("/v1/batches", str(50 * 1024 * 1024 + 4096), OPEN))    # a full batch input
        self.assertEqual(limit("/api/params", str(MAX + 1), OPEN)[0], 413)

    def test_nonsense_lengths(self):
        for bad in ("-1", "abc", "1e9"):
            self.assertEqual(limit("/api/fleet/report", bad, OPEN)[0], 400, bad)   # -1 would read to EOF
        self.assertIsNone(limit("/api/start", None, OPEN))
        self.assertIsNone(limit("/api/start", "", OPEN))

    def test_uploads_stream_with_their_own_checks(self):
        self.assertIsNone(limit("/api/files/upload", str(20 * 1024 ** 3), OPEN))

    def test_the_handler_checks_before_reading_or_authorizing(self):
        src = (ROOT / "panel.py").read_text()
        post = src[src.index("    def do_POST(self):"):]
        self.assertLess(post.index("body_limit("), post.index("self._authorize("))
        self.assertLess(post.index("body_limit("), post.index("self.rfile.read("))


if __name__ == "__main__":
    unittest.main()
