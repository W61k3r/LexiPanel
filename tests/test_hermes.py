"""Hermes Agent setup: the checks and the config.yaml it writes.
Run from the panel folder:  python3 -m unittest discover tests"""
import contextlib, sys, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hermes as H  # noqa: E402
import workload  # noqa: E402

ARGV = ["llama-server", "-m", "/m/Qwen3.8-27B.gguf", "-c", "131072", "--jinja", "--host", "0.0.0.0", "--port", "8081"]


def fake(argv=ARGV, params=None, props=None, running=True):
    p = dict(dict(CTX="131072", PARALLEL="1", PORT="8081", HOST="0.0.0.0", MODEL="/m/Qwen3.8-27B.gguf"), **(params or {}))
    get = lambda a, flags: next((a[a.index(f) + 1] for f in flags if f in a and a.index(f) + 1 < len(a)), None)
    return types.SimpleNamespace(
        using_instance=lambda inst: contextlib.nullcontext(), load_params=lambda: p,
        server_pid=lambda: 42 if running else None, live_cmdline_args=lambda: argv,
        api_get=lambda path, timeout=3: props if props is not None else
        dict(chat_template="{% if tools %}...{% endif %}", default_generation_settings=dict(n_ctx=131072)),
        workload=workload, _argv_get=get, HOME=Path("/home/admin"), PANEL=Path("/home/admin/panel"))


class Setup(unittest.TestCase):
    def run_with(self, **kw):
        H.bind(fake(**kw))
        return H.setup(dict(id="main"), "box.lan:443")

    def ids(self, d):
        return {c["id"]: c["ok"] for c in d["checks"]}

    def test_a_good_instance_is_ready_and_the_yaml_points_at_it(self):
        d = self.run_with()
        self.assertTrue(d["ready"], d["checks"])
        self.assertIn("base_url: http://box.lan:8081/v1", d["yaml"])
        self.assertIn('default: "Qwen3.8-27B.gguf"', d["yaml"])
        self.assertIn("context_length: 131072", d["yaml"])
        self.assertIn('args: ["/home/admin/panel/mcp_server.py"]', d["yaml"])
        self.assertIn('LEXIPANEL_MCP_READONLY: "1"', d["yaml"])
        self.assertNotIn("api_key", d["yaml"])

    def test_what_hermes_cannot_use_is_flagged(self):
        d = self.run_with(argv=[a for a in ARGV if a != "--jinja"],
                          params=dict(PARALLEL="4", API_KEY="s3cret"),
                          props=dict(chat_template="{{ messages }}", default_generation_settings=dict(n_ctx=32768)))
        ok = self.ids(d)
        self.assertFalse(d["ready"])
        self.assertEqual((ok["jinja"], ok["template"], ok["context"]), (False, False, False))
        self.assertIn("${env:LEXIPANEL_LLM_KEY}", d["yaml"])
        self.assertNotIn("s3cret", d["yaml"])                         # the key itself is never printed

    def test_localhost_only_means_same_machine(self):
        argv = [a if a != "0.0.0.0" else "127.0.0.1" for a in ARGV]
        d = self.run_with(argv=argv, params=dict(HOST="127.0.0.1"))
        self.assertIn("base_url: http://127.0.0.1:8081/v1", d["yaml"])
        self.assertIsNone(self.ids(d)["reachable"])

    def test_stopped_instance_is_not_ready(self):
        d = self.run_with(running=False)
        self.assertFalse(d["ready"])
        self.assertIn("context_length: 131072", d["yaml"])          # from the saved settings


if __name__ == "__main__":
    unittest.main()
