"""Gateway: quotas, usage, and a real pass-through to a stand-in OpenAI server (plain and streaming).
Run from the panel folder:  python3 -m unittest discover tests"""
import http.server, io, json, sys, tempfile, threading, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gateway as G  # noqa: E402


class Upstream(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen = (self.headers.get("Authorization"), body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for w in ("Hel", "lo"):
                self.wfile.write(f'data: {json.dumps(dict(choices=[dict(delta=dict(content=w))]))}\n\n'.encode())
            self.wfile.write(b'data: {"choices": [], "timings": {"prompt_n": 7, "predicted_n": 2}}\n\ndata: [DONE]\n\n')
        else:
            b = json.dumps(dict(choices=[dict(message=dict(content="hi"))], usage=dict(prompt_tokens=5, completion_tokens=3))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)


class Client:
    """Just enough of BaseHTTPRequestHandler for forward()."""
    def __init__(self):
        self.wfile, self.code, self.close_connection = io.BytesIO(), None, False

    def send_response(self, c):
        self.code = c

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass


class Gateway(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        G.bind(types.SimpleNamespace(PANEL=Path(self.tmp.name)))
        G._recent.clear(); G._inflight.clear()
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.tmp.cleanup()

    def test_pass_through_counts_tokens_plain_and_streaming(self):
        port = self.srv.server_address[1]
        c = Client()
        G.forward(c, "127.0.0.1", port, "k1", dict(messages=[]), "oli", "qwen")
        self.assertEqual(c.code, 200)
        self.assertEqual(json.loads(c.wfile.getvalue())["choices"][0]["message"]["content"], "hi")
        self.assertEqual(Upstream.seen[0], "Bearer k1")                       # the instance's own key
        c = Client()
        G.forward(c, "127.0.0.1", port, None, dict(messages=[], stream=True), "oli", "qwen")
        self.assertIn(b"[DONE]", c.wfile.getvalue())
        u = G._rj("usage.json", {})[G._day()]["oli"]["qwen"]
        self.assertEqual(u, dict(requests=2, prompt=12, completion=5))

    def test_least_busy_replica_and_failover_to_the_next(self):
        port = self.srv.server_address[1]
        dead = ("127.0.0.1", 9, None, "gone/main")                         # nothing listens on port 9
        live = ("127.0.0.1", port, None, "rig-2/main")
        G._busy.clear()
        G._busy[("127.0.0.1", 9)] = 0
        G._busy[("127.0.0.1", port)] = 3
        self.assertEqual(G.choose([dead, live]), dead)                       # fewest open requests first
        c = Client()
        self.assertFalse(G.forward(c, "127.0.0.1", 9, None, dict(messages=[]), "oli", "qwen"))
        self.assertEqual(G.choose([dead, live], down={("127.0.0.1", 9)}), live)
        self.assertTrue(G.forward(c, "127.0.0.1", port, None, dict(messages=[]), "oli", "qwen"))
        self.assertEqual(c.code, 200)

    def test_quotas(self):
        G.set_quota(dict(user="oli", rpm=2, concurrent=1, tokens_day=10, models=["qwen"]))
        self.assertIn("allowed models", G.admit("oli", "other"))
        self.assertIsNone(G.admit("oli", "qwen"))
        self.assertIn("concurrency", G.admit("oli", "qwen"))                  # the first is still open
        G.record("oli", "qwen", 4, 4)
        self.assertIsNone(G.admit("oli", "qwen"))
        G.record("oli", "qwen", 1, 1)
        self.assertIn("rate limit", G.admit("oli", "qwen"))
        G._recent.clear()
        self.assertIn("daily token limit", G.admit("oli", "qwen"))            # 10 used
        self.assertIsNone(G.admit("someone-else", "anything"))                # no quota, no limit
        with self.assertRaises(ValueError):
            G.set_quota(dict(user="oli", rpm=-1))


if __name__ == "__main__":
    unittest.main()
