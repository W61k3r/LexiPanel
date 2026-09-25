"""ONNX Runtime GenAI engine: the parameter suite, its validation, the launch plan, and (when
onnxruntime-genai and a model are present) the suite verified against the real runtime.
  python3 -m unittest discover tests
  LEXIPANEL_ONNX_TEST_MODEL=/path/to/genai-model python3 -m unittest tests.test_onnxrt"""
import contextlib, json, os, sys, tempfile, types, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import onnxrt as X  # noqa: E402
import onnx_server as S  # noqa: E402


class Suite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "m").mkdir()
        (d / "m" / "genai_config.json").write_text(json.dumps(dict(model=dict(context_length=4096), search={})))
        self.inst = dict(id="ox", dir=d, rundir=d / "run", engine="onnx")
        env = lambda f: dict(l.split("=", 1) for l in Path(f).read_text().splitlines() if "=" in l and not l.startswith("#")) \
            if Path(f).exists() else {}
        X.bind(types.SimpleNamespace(_read_env_file=env, _env_quote=lambda v: str(v), HOME=d, _meminfo_mb=lambda k: 64000,
                                     using_instance=lambda i: contextlib.nullcontext(), server_pid=lambda: None))
        X.runtime = lambda py: dict(version="0.16.0", cuda=False, dml=False, openvino=True, qnn=True, webgpu=False)
        X.npus = lambda: [dict(node="/dev/accel/accel0", driver="amdxdna", vendor="AMD", provider="vitisai")]
        self.model = str(d / "m")

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_runtime_search_option_is_a_parameter_with_a_tooltip(self):
        self.assertEqual(set(X.SEARCH), set(S.SEARCH_TYPES))          # the 17 onnxruntime-genai 0.16 reports
        meta = X.meta()
        self.assertEqual(set(meta), set(X.DEFAULTS))
        for k, m in meta.items():
            self.assertGreater(len(m["tip"]), 40, k)

    def test_bad_values_are_refused(self):
        for bad in (dict(OX_TOP_P="1.5"), dict(OX_TEMPERATURE="hot"), dict(OX_TOP_K="-1"), dict(OX_DO_SAMPLE="maybe"),
                    dict(BACKEND="tpu"), dict(PORT=8090), dict(OX_PROVIDER_OPTIONS="[1]"),
                    dict(OX_PROVIDER_OPTIONS='{"device_type": "NPU"}'), dict(OX_EP_LIBRARY="nopath"), dict(NOT_A_KEY=1)):
            with self.assertRaises(ValueError, msg=bad):
                X.save_params(self.inst, bad)

    def test_the_plan_carries_provider_session_and_search_options(self):
        v = X.save_params(self.inst, dict(OX_MODEL_DIR=self.model, OX_PYTHON=sys.executable, BACKEND="openvino",
                                          OX_DEVICE_TYPE="NPU", OX_THREADS=4, OX_TEMPERATURE="0.3", OX_DO_SAMPLE="true",
                                          OX_MAX_LENGTH="2048"))
        plan = X.launch_plan(self.inst, v)
        self.assertEqual(plan["errors"], [])
        a = plan["argv"]
        self.assertEqual(a[a.index("--provider") + 1], "openvino")
        self.assertEqual(json.loads(a[a.index("--provider-options") + 1]), dict(device_type="NPU"))
        self.assertEqual(json.loads(a[a.index("--session-options") + 1]), dict(intra_op_num_threads=4))
        self.assertEqual(json.loads(a[a.index("--search") + 1]), dict(max_length=2048, temperature=0.3, do_sample=True))
        self.assertIn("no NPU for openvino", " ".join(plan["warnings"]))  # only an AMD NPU is present

    def test_qnn_needs_its_library_and_a_missing_provider_is_an_error(self):
        v = X.save_params(self.inst, dict(OX_MODEL_DIR=self.model, OX_PYTHON=sys.executable, BACKEND="qnn"))
        self.assertIn("QNN needs its backend library", " ".join(X.launch_plan(self.inst, v)["errors"]))
        v = X.save_params(self.inst, dict(BACKEND="cuda"))
        self.assertIn("no cuda provider", " ".join(X.launch_plan(self.inst, v)["errors"]))
        v = X.save_params(self.inst, dict(BACKEND="vitisai", OX_VITIS_CONFIG="/opt/ryzen/vaip_config.json"))
        a = X.launch_plan(self.inst, v)["argv"]
        self.assertEqual(json.loads(a[a.index("--provider-options") + 1]), dict(config_file="/opt/ryzen/vaip_config.json"))

    def test_the_api_key_never_reaches_argv(self):
        v = X.save_params(self.inst, dict(OX_MODEL_DIR=self.model, OX_PYTHON=sys.executable, HOST="0.0.0.0", OX_API_KEY="s3cret"))
        plan = X.launch_plan(self.inst, v)
        self.assertNotIn("s3cret", " ".join(plan["argv"]))
        self.assertIn("--api-key-file", plan["argv"])


@unittest.skipUnless(os.environ.get("LEXIPANEL_ONNX_TEST_MODEL"), "set LEXIPANEL_ONNX_TEST_MODEL to a genai model folder")
class RealRuntime(unittest.TestCase):
    def test_every_search_option_is_accepted_and_read_back(self):
        a = types.SimpleNamespace(ep_library=None, model_dir=os.environ["LEXIPANEL_ONNX_TEST_MODEL"], provider="cpu",
                                  provider_options="{}", session_options='{"intra_op_num_threads": 2}')
        og, model, tok = S.load(a)
        vals = dict(max_length=512, min_length=0, do_sample=True, temperature=0.7, top_k=40, top_p=0.9,
                    repetition_penalty=1.1, num_beams=1, num_return_sequences=1, length_penalty=1.0, early_stopping=True,
                    no_repeat_ngram_size=0, diversity_penalty=0.0, past_present_share_buffer=True, batch_size=1,
                    random_seed=42, chunk_size=0)
        self.assertEqual(set(vals), set(S.SEARCH_TYPES))
        res = S.verify(og, model, vals)
        self.assertEqual([k for k, r in res.items() if not r["ok"]], [])
        self.assertEqual(set(og.GeneratorParams(model).get_search_options()), set(S.SEARCH_TYPES))  # nothing missing


if __name__ == "__main__":
    unittest.main()
