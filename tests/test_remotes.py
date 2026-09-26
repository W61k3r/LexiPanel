#!/usr/bin/env python3
"""remotes.py: servers on other machines, registered by address. Fake endpoints: a llama.cpp
server that wants an API key, a vLLM-style server, and one that redirects to the cloud
metadata address (never followed)."""
import http.server, io, json, os, socketserver, stat, sys, tempfile, threading, time, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import remotes as RM  # noqa: E402
import gateway as GW  # noqa: E402

KEY = "s3cret-remote-key"


def serve(handler):
    class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    s = Srv(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


class Base(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class Llama(Base):
    seen_auth = []

    def do_GET(self):
        Llama.seen_auth.append(self.headers.get("Authorization"))
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self.send(dict(error="unauthorized"), 401)
        if self.path == "/health":
            return self.send(dict(status="ok"))
        if self.path == "/v1/models":
            return self.send(dict(data=[dict(id="qwen-remote")]))
        if self.path == "/props":
            return self.send(dict(default_generation_settings=dict(n_ctx=65536), model_path="/m/qwen.gguf", build_info="b1"))
        if self.path == "/slots":
            return self.send([dict(id=0, is_processing=True), dict(id=1, is_processing=False)])
        if self.path == "/metrics":
            return self.send("llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\n"
                             "llamacpp:predicted_tokens_seconds 42.5\n", ctype="text/plain")
        self.send(dict(error="nf"), 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Llama.posted = (self.headers.get("Authorization"), body)
        self.send(dict(choices=[dict(message=dict(content="hi from remote"))], usage=dict(prompt_tokens=4, completion_tokens=3)))


class Vllm(Base):
    def do_GET(self):
        if self.path == "/health":
            return self.send("", 200, "text/plain")
        if self.path == "/v1/models":
            return self.send(dict(data=[dict(id="meta/llama-3-8b")]))
        if self.path == "/metrics":
            return self.send("vllm:num_requests_running 2.0\nvllm:num_requests_waiting 1.0\n", ctype="text/plain")
        self.send(dict(error="nf"), 404)


class Redirect(Base):
    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
        self.send_header("Content-Length", "0")
        self.end_headers()


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.P = types.SimpleNamespace(PANEL=Path(self.tmp.name), instance_ids=lambda: [])
        RM.bind(self.P)
        RM._cache.clear(); RM._refreshing.clear()
        self.llama, self.vllm, self.redir = serve(Llama), serve(Vllm), serve(Redirect)
        Llama.seen_auth = []

    def tearDown(self):
        for s in (self.llama, self.vllm, self.redir):
            s.shutdown(); s.server_close()
        self.tmp.cleanup()

    def url(self, s):
        return f"http://127.0.0.1:{s.server_address[1]}"


class Remotes(Env):
    def test_address_rules(self):
        bad = {"ftp://127.0.0.1:21": "http:// or https://",
               "http://user:pw@127.0.0.1:8080": "user:password",
               "http://127.0.0.1:8080/v1": "no path",
               "http://127.0.0.1:8080?x=1": "no path",
               "http://169.254.169.254:80": "not allowed",
               "http://[fe80::1]:80": "not allowed",
               "http://0.0.0.0:8080": "not allowed",
               "http://no-such-host.invalid:8080": "cannot resolve"}
        for u, msg in bad.items():
            with self.assertRaisesRegex(ValueError, msg, msg=u):
                RM.check_url(u)
        self.assertEqual(RM.check_url("http://127.0.0.1:8080"), ("http", "127.0.0.1", 8080))
        self.assertEqual(RM.check_url("https://localhost")[2], 443)

    def test_llama_endpoint_with_key(self):
        RM.add(dict(name="box2", url=self.url(self.llama), api_key=KEY))
        mode = stat.S_IMODE(os.stat(self.P.PANEL / "remotes-keys.json").st_mode)
        self.assertEqual(mode, 0o600)
        [x] = RM.listing()
        self.assertTrue(x["has_key"])
        self.assertNotIn(KEY, json.dumps(x))                               # never returned
        s = x["status"]
        self.assertEqual((s["health"], s["kind"], s["ctx"], s["slots"], s["slots_busy"], s["live_tps"]),
                         ("ok", "llama.cpp", 65536, 2, 1, 42.5))
        self.assertEqual(s["models"], ["qwen-remote"])
        self.assertTrue(all(a == f"Bearer {KEY}" for a in Llama.seen_auth))

    def test_without_the_key_it_is_unauthorized(self):
        RM.add(dict(name="box2", url=self.url(self.llama)))
        self.assertEqual(RM.listing()[0]["status"]["health"], "HTTP 401")

    def test_vllm_is_recognised(self):
        RM.add(dict(name="v", url=self.url(self.vllm)))
        s = RM.listing()[0]["status"]
        self.assertEqual((s["kind"], s["running"], s["waiting"], s["models"]), ("vllm", 2.0, 1.0, ["meta/llama-3-8b"]))

    def test_redirects_are_not_followed(self):
        RM.add(dict(name="r", url=self.url(self.redir)))
        self.assertEqual(RM.listing()[0]["status"]["health"], "HTTP 302")

    def test_server_rows_and_warnings(self):
        RM.add(dict(name="box2", url=self.url(self.llama), api_key=KEY, gateway=True))
        RM.listing()                                                        # warm the cache
        [row] = RM.servers()
        self.assertEqual(row["remote"]["name"], "box2")
        self.assertEqual((row["health"], row["ctx"], row["slots"], row["slots_busy"], row["managed"]), ("ok", 65536, 2, 1, False))
        self.assertTrue(any("unencrypted" in w for w in row["warnings"]))

    def test_status_poll_never_waits_on_a_dead_endpoint(self):
        RM.add(dict(name="dead", url="http://127.0.0.1:9"))                 # nothing listens on :9
        t0 = time.time()
        [row] = RM.servers()
        self.assertLess(time.time() - t0, 2.0)                      # never the 3 s network timeout
        self.assertEqual(row["health"], "checking")
        for _ in range(100):
            time.sleep(0.05)
            if RM.servers()[0]["health"] != "checking":
                break
        self.assertEqual(RM.servers()[0]["health"], "no answer")

    def test_test_does_not_keep_the_key(self):
        r = RM.test(dict(url=self.url(self.llama), api_key=KEY))
        self.assertEqual(r["health"], "ok")
        self.assertFalse((self.P.PANEL / "remotes-keys.json").exists())

    def test_delete_forgets_the_key(self):
        RM.add(dict(name="box2", url=self.url(self.llama), api_key=KEY))
        RM.delete(dict(name="box2"))
        self.assertEqual(RM.listing(), [])
        self.assertNotIn(KEY, (self.P.PANEL / "remotes-keys.json").read_text())
        with self.assertRaises(ValueError):
            RM.delete(dict(name="box2"))

    def test_names_and_kinds_validated(self):
        for body, msg in ((dict(name="../x", url=self.url(self.vllm)), "name"),
                          (dict(name="ok", url=self.url(self.vllm), kind="docker"), "kind")):
            with self.assertRaisesRegex(ValueError, msg):
                RM.add(body)
        RM.add(dict(name="dup", url=self.url(self.vllm)))
        with self.assertRaisesRegex(ValueError, "already registered"):
            RM.add(dict(name="dup", url=self.url(self.vllm)))


class ThroughTheGateway(Env):
    def test_gateway_routes_to_the_remote_with_its_key(self):
        RM.add(dict(name="box2", url=self.url(self.llama), api_key=KEY, gateway=True))
        RM.add(dict(name="v", url=self.url(self.vllm)))                     # not exposed
        RM.listing()
        self.P.remotes = RM
        GW.bind(self.P)
        ts = GW.targets()
        self.assertIn("box2", ts)
        self.assertIn("qwen-remote", ts)                                    # by model name too
        self.assertNotIn("v", ts)
        host, port, key, where, scheme, upstream = ts["qwen-remote"][0]
        self.assertEqual((key, where, scheme, upstream), (KEY, "remote:box2", "http", "qwen-remote"))
        self.assertEqual(ts["box2"][0][5], "qwen-remote")                  # its own name for it, sent upstream

        class Client:
            def __init__(self):
                self.wfile, self.code, self.close_connection = io.BytesIO(), None, False

            def send_response(self, c):
                self.code = c

            def send_header(self, *a):
                pass

            def end_headers(self):
                pass
        c = Client()
        self.assertTrue(GW.forward(c, host, port, key, dict(model="qwen-remote", messages=[]), "alice", "qwen-remote", scheme))
        self.assertEqual(c.code, 200)
        self.assertIn(b"hi from remote", c.wfile.getvalue())
        self.assertEqual(Llama.posted[0], f"Bearer {KEY}")


if __name__ == "__main__":
    unittest.main()
