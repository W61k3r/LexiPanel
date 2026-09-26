#!/usr/bin/env python3
"""Slot planning (panel.slot_plan): what CTX and PARALLEL mean for conversations served at once.
The shared-pool rule is the one measured on the RTX 2060 on 2026-09-26: 16 streams of 448 tokens
failed in an 8192 pool and ran clean in 12288. The function is pure; it is lifted out of panel.py
so the test does not start a panel."""
import ast, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = {}
for node in ast.parse((ROOT / "panel.py").read_text()).body:
    if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "KV_POOL_HEADROOM") or \
            (isinstance(node, ast.FunctionDef) and node.name == "slot_plan"):
        exec(compile(ast.Module([node], []), "panel.py", "exec"), NS)
plan = NS["slot_plan"]
KV = 0.109375                  # MiB per token: Qwen3-1.7B, f16 KV (28 layers x 8 heads x 128 x 2 x 2 bytes)


class SlotPlan(unittest.TestCase):
    def test_one_slot(self):
        self.assertEqual(plan(8192, 1, False, KV, 4000, 2000)["safe_streams"], 1)
        self.assertEqual(plan(8192, 1, False, KV, 9000, 2000)["safe_streams"], 0)
        self.assertEqual(plan(8192, 1, True, KV, 4000, 2000)["unified"], False)       # meaningless with one

    def test_separate_kv_caps_each_conversation(self):
        p = plan(12288, 16, False, KV, 448, 2000)
        self.assertEqual((p["per_slot_max"], p["safe_streams"]), (768, 16))
        p = plan(12288, 16, False, KV, 1000, 2000)
        self.assertEqual(p["safe_streams"], 0)                                     # deeper than CTX / slots
        self.assertEqual(p["for_all"]["ctx"], 16384)

    def test_shared_pool_matches_the_measurement(self):
        p = plan(8192, 16, True, KV, 448, 2000)                                    # failed on the card
        self.assertEqual((p["per_slot_max"], p["safe_streams"]), (8192, 10))
        self.assertEqual(p["for_all"]["ctx"], 12288)                               # ran clean on the card
        self.assertEqual(plan(12288, 16, True, KV, 448, 2000)["safe_streams"], 16)
        self.assertEqual(plan(8192, 16, True, KV, 8000, 2000)["safe_streams"], 1)  # one deep one, alone

    def test_per_slot_cap_and_memory(self):
        p = plan(65536, 4, True, KV, 10000, 2000, per_slot_cap=8192)
        self.assertEqual((p["per_slot_max"], p["safe_streams"]), (8192, 0))
        p = plan(8192, 16, True, KV, 448, 2000)
        self.assertEqual(p["kv_mib_per_stream"], 49)
        self.assertEqual((p["for_all"]["kv_mib"], p["for_all"]["extra_mib"], p["for_all"]["fits"]), (1344, 448, True))
        afford = 8192 + (2000 - 1024) / KV
        self.assertEqual(p["max_streams"], dict(separate=int(afford // 448), shared=int(afford // (448 * 1.7))))
        self.assertFalse(plan(8192, 16, True, KV, 448, 1200)["for_all"]["fits"])  # 448 MiB more, 176 spare


if __name__ == "__main__":
    unittest.main()
