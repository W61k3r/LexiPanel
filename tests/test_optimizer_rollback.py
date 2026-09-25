"""Optimizer: Apply records what it replaced; Roll back puts it back, and leaves alone
anything changed by hand since.  python3 -m unittest discover tests"""
import contextlib, json, sys, tempfile, types, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optimizer  # noqa: E402


class Rollback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        optimizer.RUNS_DIR = Path(self.tmp.name)
        self.params = dict(BACKEND="rocm", MODEL="/m/a.gguf", BATCH=512, UBATCH=512, CTX=131072)
        self.saves = []
        optimizer.P = types.SimpleNamespace(
            using_instance=lambda iid: contextlib.nullcontext(),
            load_params=lambda b=None: dict(self.params),
            save_params=lambda vals, b=None: (self.saves.append((dict(vals), b)), self.params.update(vals)))
        run = dict(id="r1", instance="main", model="/m/a.gguf", candidates=[
            dict(id="c1", status="ok", label="b2048", launch=dict(BATCH=2048, UBATCH=1024))])
        (optimizer.RUNS_DIR / "r1").mkdir()
        (optimizer.RUNS_DIR / "r1" / "run.json").write_text(json.dumps(run))
        optimizer._run = None

    def tearDown(self):
        self.tmp.cleanup()

    def test_apply_then_rollback_restores_the_previous_values(self):
        optimizer.apply("r1", "c1")
        self.assertEqual((self.params["BATCH"], self.params["UBATCH"]), (2048, 1024))
        r = optimizer.rollback("r1")
        self.assertEqual((self.params["BATCH"], self.params["UBATCH"], self.params["BACKEND"]), (512, 512, "rocm"))
        self.assertEqual(r["kept"], [])
        with self.assertRaises(ValueError):            # used up
            optimizer.rollback("r1")

    def test_a_setting_changed_by_hand_since_is_left_alone(self):
        optimizer.apply("r1", "c1")
        self.params["UBATCH"] = 256                     # the operator changed it afterwards
        r = optimizer.rollback("r1")
        self.assertEqual((self.params["BATCH"], self.params["UBATCH"]), (512, 256))
        self.assertEqual(r["kept"], ["UBATCH"])

    def test_nothing_applied_means_nothing_to_roll_back(self):
        with self.assertRaises(ValueError):
            optimizer.rollback("r1")
        with self.assertRaises(ValueError):
            optimizer.rollback("../etc")


if __name__ == "__main__":
    unittest.main()
