"""The MCP server's JSON-RPC handling and tool mapping, against a fake panel API.
Run from the panel folder:  python3 -m unittest discover tests"""
import io, json, os, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mcp_server as M  # noqa: E402


class FakePanel:
    def __init__(self):
        self.calls = []
        self.params = {"main": dict(BACKEND="rocm", CTX=131072, NGL=99, KV_TYPE="q8_0")}

    def __call__(self, method, path, query=None, body=None, raw=False, timeout=120):
        self.calls.append((method, path, dict(query or {}), body))
        inst = (query or {}).get("inst", "main")
        if path == "/api/params" and method == "GET":
            return dict(self.params[inst])
        if path == "/api/params" and method == "POST":
            self.params[inst] = dict(body)
            return dict(ok=True, params=dict(body))
        if path == "/api/files/download":
            return b"line one\nline two\n" if query["path"].endswith(".txt") else b"\x00\x01binary"
        if path == "/api/status":
            return dict(running=True, pid=1, params={"huge": "x" * 10000}, instances=[], engine="llama.cpp")
        if path == "/api/start":
            return dict(ok=False, msg="refused: plan does not fit")
        raise M.ToolError(f"panel said HTTP 404: not found {path}")


def rpc(method, params=None, mid=1):
    m = dict(jsonrpc="2.0", method=method, params=params or {})
    if mid is not None:
        m["id"] = mid
    return M.handle(m)


def call(name, **args):
    r = rpc("tools/call", dict(name=name, arguments=args))
    res = r["result"]
    return res["isError"], (res["content"][0]["text"] if res["isError"] else json.loads(res["content"][0]["text"]))


class Protocol(unittest.TestCase):
    def test_initialize_negotiates_a_known_version(self):
        self.assertEqual(rpc("initialize", dict(protocolVersion="2025-03-26"))["result"]["protocolVersion"], "2025-03-26")
        r = rpc("initialize", dict(protocolVersion="1999-01-01"))["result"]
        self.assertEqual(r["protocolVersion"], M.PROTOCOLS[0])
        self.assertIn("tools", r["capabilities"])

    def test_notifications_get_no_answer_and_unknown_methods_an_error(self):
        self.assertIsNone(rpc("notifications/initialized", mid=None))
        self.assertEqual(rpc("resources/list")["error"]["code"], -32601)
        self.assertEqual(M.handle({"id": 1, "method": "ping"})["error"]["code"], -32600)
        self.assertEqual(rpc("ping")["result"], {})

    def test_batch(self):
        out = M.handle([dict(jsonrpc="2.0", id=1, method="ping"), dict(jsonrpc="2.0", method="notifications/x")])
        self.assertEqual(len(out), 1)

    def test_every_tool_has_a_schema_and_honest_annotations(self):
        tools = rpc("tools/list")["result"]["tools"]
        self.assertGreaterEqual(len(tools), 20)
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])
            self.assertIn("readOnlyHint", t["annotations"], t["name"])
        names = {t["name"] for t in tools}
        self.assertNotIn("write_file", names)
        writers = {t["name"] for t in tools if not t["annotations"]["readOnlyHint"]}
        self.assertTrue({"set_params", "stop_instance", "restart_instance", "apply_power_profile"} <= writers)

    def test_stdio_loop(self):
        out = io.StringIO()
        M.serve_stdio(io.StringIO('{"jsonrpc":"2.0","id":7,"method":"ping"}\nnot json\n\n'), out)
        lines = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual((lines[0]["id"], lines[1]["error"]["code"]), (7, -32700))


class Tools(unittest.TestCase):
    def setUp(self):
        self.fake = FakePanel()
        self.real, M.api = M.api, self.fake
        os.environ.pop("LEXIPANEL_MCP_READONLY", None)

    def tearDown(self):
        M.api = self.real
        os.environ.pop("LEXIPANEL_MCP_READONLY", None)

    def test_set_params_merges_and_keeps_the_backend(self):
        err, out = call("set_params", params=dict(CTX=65536))
        self.assertFalse(err, out)
        self.assertEqual(out["changed"], dict(CTX=dict(before=131072, after=65536)))
        saved = self.fake.params["main"]
        self.assertEqual((saved["BACKEND"], saved["NGL"], saved["CTX"]), ("rocm", 99, 65536))

    def test_set_params_refuses_unknown_settings_before_saving(self):
        err, msg = call("set_params", params=dict(NOPE=1))
        self.assertTrue(err)
        self.assertIn("NOPE", msg)
        self.assertFalse(any(c[0] == "POST" for c in self.fake.calls))

    def test_bad_instance_id_never_reaches_the_panel(self):
        err, msg = call("get_status", instance="../../etc")
        self.assertTrue(err)
        self.assertEqual(self.fake.calls, [])

    def test_status_is_trimmed(self):
        err, out = call("get_status")
        self.assertFalse(err)
        self.assertNotIn("params", out)

    def test_refusals_come_back_as_results(self):
        err, out = call("start_instance", instance="main")
        self.assertEqual(out["msg"], "refused: plan does not fit")

    def test_read_text_file_refuses_binary(self):
        self.assertEqual(call("read_text_file", path="notes.txt")[1]["text"], "line one\nline two\n")
        self.assertTrue(call("read_text_file", path="model.gguf")[0])

    def test_panel_errors_are_tool_errors_not_protocol_errors(self):
        r = rpc("tools/call", dict(name="list_models", arguments={}))
        self.assertTrue(r["result"]["isError"])

    def test_read_only_mode_hides_writers(self):
        os.environ["LEXIPANEL_MCP_READONLY"] = "1"
        names = {t["name"] for t in rpc("tools/list")["result"]["tools"]}
        self.assertNotIn("stop_instance", names)
        self.assertEqual(rpc("tools/call", dict(name="stop_instance", arguments={}))["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main()
