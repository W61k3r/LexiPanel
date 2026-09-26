#!/usr/bin/env python3
"""Gateway admission (never more requests at once than a server has slots; interactive before
batch) and /v1/batches (run, owner-only, 0600, cancel, resume after a crash, retention),
against a stand-in OpenAI server that records how many requests it held at once."""
import http.server, io, json, os, socketserver, stat, sys, tempfile, threading, time, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gateway as G  # noqa: E402
import batches as B  # noqa: E402


class Upstream:
    def __init__(self, delay=0.05):
        self.now = self.peak = self.count = 0
        self.lock = threading.Lock()
        self.bodies = []
        self.delay = delay
        up = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with up.lock:
                    up.now += 1
                    up.count += 1
                    up.peak = max(up.peak, up.now)
                    up.bodies.append(body)
                time.sleep(up.delay)
                with up.lock:
                    up.now -= 1
                b = json.dumps(dict(choices=[dict(message=dict(content="ok " + str(body["messages"][0]["content"])))],
                                    usage=dict(prompt_tokens=3, completion_tokens=2))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64
        self.srv = Srv(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Handler:
    """Just enough of BaseHTTPRequestHandler for gateway.handle_post."""
    def __init__(self, body, headers=None):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = dict({"Content-Length": str(len(raw))}, **(headers or {}))
        self.code, self.close_connection = None, False

    def _foreign(self, m):
        return None

    def send_response(self, c):
        self.code = c

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.P = types.SimpleNamespace(PANEL=Path(self.tmp.name))
        G.bind(self.P)
        B.bind(self.P, G)
        self.P.batches = B
        self.up = Upstream()
        self.key = ("127.0.0.1", self.up.port)
        self._targets = G.targets
        G.targets = lambda: {"m": [("127.0.0.1", self.up.port, None, "main")]}
        G._adm.clear(); G._caps.clear(); G._cap_cache.clear(); G._recent.clear(); G._inflight.clear(); G._busy.clear()
        G._pools.clear()
        B._cancel.clear()

    def tearDown(self):
        G.targets = self._targets
        self.up.close()
        self.tmp.cleanup()


class Admission(Base):
    def test_never_more_than_the_slots(self):
        G._caps[self.key] = 2
        hs = [Handler(dict(model="m", messages=[dict(role="user", content=str(i))])) for i in range(6)]
        ths = [threading.Thread(target=G.handle_post, args=(h, "/v1/chat/completions", "alice")) for h in hs]
        for t in ths:
            t.start()
        for t in ths:
            t.join(10)
        self.assertEqual([h.code for h in hs], [200] * 6)
        self.assertEqual(self.up.peak, 2)                               # the server never held a third
        self.assertEqual(G.admission()[0]["served"], 6)

    def test_interactive_jumps_the_batch_queue(self):
        G._caps[self.key] = 1
        order = []
        G.admit_slot("127.0.0.1", self.up.port)                         # the only slot: busy
        def wait(prio, tag):
            G.admit_slot("127.0.0.1", self.up.port, prio)
            order.append(tag)
            G.release_slot("127.0.0.1", self.up.port)
        tb = threading.Thread(target=wait, args=("batch", "batch"))
        tb.start()
        time.sleep(0.2)                                                 # the batch request queued first
        ti = threading.Thread(target=wait, args=("interactive", "interactive"))
        ti.start()
        time.sleep(0.2)
        G.release_slot("127.0.0.1", self.up.port)
        tb.join(5); ti.join(5)
        self.assertEqual(order, ["interactive", "batch"])

    def test_a_shared_kv_pool_admits_by_tokens(self):
        G._caps[self.key] = 8
        G._pools[self.key] = 1000                                       # tokens it holds without failing
        self.up.delay = 0.2
        body = lambda i: dict(model="m", max_tokens=100, messages=[dict(role="user", content=f"{i}" + "x" * 950)])
        self.assertEqual(G.need_tokens(body(0)), int(951 / 3.2) + 8 + 100)          # ~405 tokens each
        hs = [Handler(body(i)) for i in range(6)]
        ths = [threading.Thread(target=G.handle_post, args=(h, "/v1/chat/completions", "alice")) for h in hs]
        for t in ths:
            t.start()
        for t in ths:
            t.join(10)
        self.assertEqual([h.code for h in hs], [200] * 6)
        self.assertEqual(self.up.peak, 2)                               # 8 slots free, but only 2 fit the pool
        row = G.admission()[0]
        self.assertEqual((row["kv_pool"], row["kv_tokens"]), (1000, 0))
        self.assertGreaterEqual(row["waited_for_kv"], 1)

    def test_a_request_bigger_than_the_pool_still_goes_alone(self):
        G._caps[self.key] = 8
        G._pools[self.key] = 100
        self.assertLess(G.admit_slot("127.0.0.1", self.up.port, need=5000), 0.1)
        with self.assertRaises(TimeoutError):
            G.admit_slot("127.0.0.1", self.up.port, need=10, timeout=0.3)          # waits for it to finish
        G.release_slot("127.0.0.1", self.up.port, 5000)
        self.assertLess(G.admit_slot("127.0.0.1", self.up.port, need=10), 0.1)

    def test_targets_note_the_pool_from_the_live_command_line(self):
        argv = {"a": ["llama-server", "-m", "/m/a.gguf", "--port", "8085", "-c", "8192", "-np", "16", "--kv-unified"],
                "b": ["llama-server", "-m", "/m/b.gguf", "--port", "8086", "-c", "8192", "-np", "16"]}
        cur = {}
        P = types.SimpleNamespace(PANEL=self.P.PANEL, KV_POOL_HEADROOM=1.7, instance_ids=lambda: ["a", "b"],
                                  get_instance=lambda i: dict(id=i, engine="llama.cpp"),
                                  using_instance=lambda inst: types.SimpleNamespace(
                                      __enter__=lambda s: cur.update(i=inst["id"]), __exit__=lambda s, *a: None),
                                  server_pid=lambda: 1, live_cmdline_args=lambda: argv[cur["i"]],
                                  load_params=lambda: dict(PARALLEL="1", CTX="4096"),
                                  _argv_get=lambda a, names: next((a[i + 1] for i, x in enumerate(a)
                                                                   if x in names and i + 1 < len(a)), None))
        import contextlib

        @contextlib.contextmanager
        def using(inst):
            cur.update(i=inst["id"])
            yield inst
        P.using_instance = using
        G.targets = self._targets
        G.bind(P)
        G.targets()
        self.assertEqual((G._caps[("127.0.0.1", 8085)], G._pools[("127.0.0.1", 8085)]), (16, int(8192 / 1.7)))
        self.assertEqual((G._caps[("127.0.0.1", 8086)], G._pools[("127.0.0.1", 8086)]), (16, None))  # separate KV

    def test_wait_times_out_cleanly(self):
        G._caps[self.key] = 1
        G.admit_slot("127.0.0.1", self.up.port)
        with self.assertRaises(TimeoutError):
            G.admit_slot("127.0.0.1", self.up.port, "interactive", timeout=0.3)
        self.assertEqual(G.admission()[0]["waiting"], 0)
        self.assertEqual(G.admission()[0]["timeouts"], 1)

    def test_unknown_capacity_is_not_limited(self):
        self.assertEqual(G.admit_slot("127.0.0.1", 9), 0.0)             # /slots unreachable: no limit

    def test_the_servers_own_model_name_goes_upstream(self):
        G.targets = lambda: {"vl-2060": [("127.0.0.1", self.up.port, None, "vl-2060", "http", "qwen3-1.7b")]}
        h = Handler(dict(model="vl-2060", messages=[dict(role="user", content="x")]))
        G.handle_post(h, "/v1/chat/completions", "alice")
        self.assertEqual((h.code, self.up.bodies[-1]["model"]), (200, "qwen3-1.7b"))    # vLLM refuses any other
        st, _r = G.call("127.0.0.1", self.up.port, None, dict(model="vl-2060", messages=[dict(role="user", content="y")]),
                        upstream="qwen3-1.7b")
        self.assertEqual((st, self.up.bodies[-1]["model"]), (200, "qwen3-1.7b"))
        self.assertEqual(G.admission()[0]["served"], 1)

    def test_priority_field_never_reaches_the_server(self):
        h = Handler(dict(model="m", messages=[dict(role="user", content="x")], priority="batch"))
        G.handle_post(h, "/v1/chat/completions", "alice")
        self.assertEqual(h.code, 200)
        self.assertNotIn("priority", self.up.bodies[-1])
        bad = Handler(dict(model="m", messages=[]), headers={"X-LexiPanel-Priority": "urgent"})
        G.handle_post(bad, "/v1/chat/completions", "alice")
        self.assertEqual(bad.code, 400)


def reqs(n):
    return [dict(custom_id=f"r{i}", body=dict(messages=[dict(role="user", content=f"q{i}")])) for i in range(n)]


class Batches(Base):
    def test_run_to_completion(self):
        b = B.create("alice", dict(model="m", input=reqs(12), metadata=dict(job="nightly")))
        self.assertEqual((b["status"], b["request_counts"]["total"]), ("in_progress", 12))
        B._process(b["id"])
        got = B.get("alice", b["id"])
        self.assertEqual((got["status"], got["request_counts"]), ("completed", dict(total=12, completed=12, failed=0)))
        out = [json.loads(l) for l in B.output("alice", b["id"]).splitlines()]
        self.assertEqual(sorted(o["custom_id"] for o in out), sorted(f"r{i}" for i in range(12)))
        self.assertTrue(all(o["response"]["status_code"] == 200 for o in out))
        d = self.P.PANEL / "gateway" / "batches" / b["id"]
        for f in ("input.jsonl", "output.jsonl", "batch.json"):
            self.assertEqual(stat.S_IMODE((d / f).stat().st_mode), 0o600, f)
        self.assertEqual(stat.S_IMODE(d.stat().st_mode), 0o700)

    def test_owner_only(self):
        b = B.create("alice", dict(model="m", input=reqs(1)))
        for fn in (B.get, B.output, B.cancel):
            with self.assertRaises(KeyError):
                fn("mallory", b["id"])
        self.assertEqual(B.listing("mallory")["data"], [])
        self.assertEqual(len(B.listing("alice")["data"]), 1)

    def test_validation(self):
        for body, msg in ((dict(input=[]), "input"), (dict(input=[dict(body="x")]), "request 0"),
                          (dict(input=reqs(1)), "no model"),
                          (dict(model="m", input=[dict(custom_id="a", body=dict(messages=[])),
                                                  dict(custom_id="a", body=dict(messages=[]))]), "twice"),
                          (dict(model="m", input=[dict(body=dict(messages="hi"))]), "messages")):
            with self.assertRaisesRegex(ValueError, msg):
                B.create("alice", body)

    def test_cancel_mid_way(self):
        self.up.delay = 0.2
        b = B.create("alice", dict(model="m", input=reqs(40)))
        th = threading.Thread(target=B._process, args=(b["id"],))
        th.start()
        time.sleep(0.5)
        B.cancel("alice", b["id"])
        th.join(20)
        got = B.get("alice", b["id"])
        self.assertEqual(got["status"], "cancelled")
        self.assertLess(got["request_counts"]["completed"], 40)

    def test_resume_after_a_crash_runs_only_the_missing(self):
        b = B.create("alice", dict(model="m", input=reqs(10)))
        out = self.P.PANEL / "gateway" / "batches" / b["id"] / "output.jsonl"
        for cid in ("r0", "r3", "r7"):                                  # finished before the crash, out of order
            B._append(out, json.dumps(dict(id="x", custom_id=cid, response=dict(status_code=200, body={}), error=None)))
        B._process(b["id"])
        ran = sorted(x["messages"][0]["content"] for x in self.up.bodies)
        self.assertEqual(ran, sorted(f"q{i}" for i in range(10) if i not in (0, 3, 7)))
        self.assertEqual(len(B.output("alice", b["id"]).splitlines()), 10)

    def test_quota_refusal_is_an_error_line(self):
        G.set_quota(dict(user="alice", models=["other"]))
        b = B.create("alice", dict(model="m", input=reqs(2)))
        B._process(b["id"])
        out = [json.loads(l) for l in B.output("alice", b["id"]).splitlines()]
        self.assertTrue(all(o["error"]["code"] == "quota" for o in out))
        self.assertEqual(B.get("alice", b["id"])["status"], "failed")

    def test_expired_batches_are_deleted(self):
        b = B.create("alice", dict(model="m", input=reqs(1)))
        B._process(b["id"])
        m = B._meta(b["id"])
        m["_expires"] = time.time() - 1
        B._save(m)
        B._purge()
        self.assertFalse((self.P.PANEL / "gateway" / "batches" / b["id"]).exists())

    def test_routes(self):
        h = Handler(dict(model="m", input=reqs(2)))
        G.handle_post(h, "/v1/batches", "alice")
        self.assertEqual(h.code, 200)
        bid = json.loads(h.wfile.getvalue())["id"]
        g = Handler({})
        G.handle_get_batches(g, f"/v1/batches/{bid}", "alice")
        self.assertEqual((g.code, json.loads(g.wfile.getvalue())["id"]), (200, bid))
        n = Handler({})
        G.handle_get_batches(n, f"/v1/batches/{bid}", "mallory")
        self.assertEqual(n.code, 404)


if __name__ == "__main__":
    unittest.main()
