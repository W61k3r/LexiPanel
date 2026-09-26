#!/usr/bin/env python3
"""benchstats: distributions against published values, the decision rule, and the property
the platform promises (a false-win rate within alpha however the sequential test peeks)."""
import math, os, random, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchstats as S  # noqa: E402


class Distributions(unittest.TestCase):
    def test_t_quantiles_match_tables(self):
        for (p, df), v in {(0.975, 1): 12.7062, (0.975, 2): 4.3027, (0.975, 5): 2.5706,
                           (0.975, 10): 2.2281, (0.975, 30): 2.0423, (0.975, 100): 1.9840,
                           (0.8, 4): 0.9410}.items():
            self.assertAlmostEqual(S.t_ppf(p, df), v, places=3, msg=(p, df))

    def test_t_df2_closed_form(self):
        # F(t) = 1/2 + t / (2 sqrt(2 + t^2)) for two degrees of freedom
        for p in (0.9, 0.99375, 0.999):
            r = 2 * p - 1
            self.assertAlmostEqual(S.t_ppf(p, 2), math.sqrt(2 * r * r / (1 - r * r)), places=6)

    def test_t_cdf_and_normal(self):
        self.assertAlmostEqual(S.t_cdf(2.0, 10), 0.963306, places=6)
        self.assertAlmostEqual(S.norm_ppf(0.975), 1.959964, places=6)
        self.assertAlmostEqual(S.norm_ppf(1e-6), -4.753424, places=5)
        self.assertAlmostEqual(S.norm_cdf(S.norm_ppf(0.3)), 0.3, places=10)

    def test_binomial(self):
        self.assertAlmostEqual(S.binom_two_sided(0, 5), 0.0625)
        self.assertAlmostEqual(S.binom_two_sided(1, 10), 22 / 1024)
        self.assertEqual(S.binom_two_sided(0, 0), 1.0)


class DecisionRule(unittest.TestCase):
    M = 0.02

    def v(self, est, lo, hi, hb=True):
        return S.verdict(math.log1p(est), math.log1p(lo), math.log1p(hi), self.M, hb)

    def test_truth_table(self):
        self.assertEqual(self.v(0.05, 0.03, 0.07), "better")
        self.assertEqual(self.v(0.01, 0.002, 0.018), "small")       # real but under the margin
        self.assertEqual(self.v(-0.05, -0.07, -0.03), "worse")
        self.assertEqual(self.v(0.0, -0.01, 0.01), "same")          # inside +-2 %
        self.assertEqual(self.v(0.01, -0.03, 0.05), "undecided")

    def test_lower_is_better_for_times(self):
        self.assertEqual(self.v(-0.05, -0.07, -0.03, hb=False), "better")
        self.assertEqual(self.v(0.05, 0.03, 0.07, hb=False), "worse")

    def test_paired_interval(self):
        a = [100.0] * 6
        b = [103.0, 102.5, 103.5, 102.8, 103.2, 103.0]
        r = S.paired(a, b)
        self.assertEqual(r["n"], 6)
        self.assertAlmostEqual(r["pct"], 3.0, delta=0.1)
        self.assertLess(r["pct_lo"], 3.0)
        self.assertGreater(r["pct_hi"], 3.0)
        self.assertGreater(r["pct_lo"], 2.0)

    def test_sequential_only_stops_at_looks(self):
        a, b = [100.0] * 4, [110.0, 110.1, 109.9, 110.0]
        self.assertEqual(S.sequential(a, b)["state"], "continue")      # 4 is not a look
        r = S.sequential(a + [100.0], b + [110.0])
        self.assertEqual((r["state"], r["verdict"]), ("stop", "better"))

    def test_undecided_at_last_look_reports_precision(self):
        rng = random.Random(3)
        a = [100.0] * 12
        b = [100 * math.exp(rng.gauss(0, 0.05)) for _ in range(12)]
        r = S.sequential(a, b, margin=0.001)
        self.assertEqual(r["state"], "stop")
        if r["verdict"] == "undecided":
            self.assertIn("precision_pct", r)


class SelfTest(unittest.TestCase):
    """The promise: identical configurations are called different at most alpha of the
    time, and a real effect well above the noise is found."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(7)
        cls.diffs = [rng.gauss(0, 0.01) for _ in range(200)]        # 1 % A/A noise

    def test_false_win_rate_within_alpha(self):
        r = S.simulate(self.diffs, 0.0, trials=3000, seed=11)
        self.assertLessEqual(r["better"] + r["worse"], S.ALPHA)

    def test_detects_real_effect(self):
        r = S.simulate(self.diffs, 0.05, trials=1000, seed=12)
        self.assertGreaterEqual(r["better"], 0.95)
        self.assertLess(r["mean_pairs"], 8)                          # stops early when clear

    def test_selftest_summary(self):
        t = S.selftest(self.diffs, trials=400)
        self.assertLessEqual(t["false_win"], 0.06)
        self.assertIsNotNone(t["smallest_reliable"])
        self.assertLessEqual(t["smallest_reliable"], 0.05)

    def test_mde_and_pairs_agree(self):
        n = S.pairs_needed(0.01, 0.03)
        self.assertLessEqual(S.mde(0.01, n), 0.03 + 1e-9)
        self.assertGreater(S.mde(0.01, n - 1), 0.03)


class Tasks(unittest.TestCase):
    def test_mcnemar(self):
        a = [True] * 20 + [False] * 6
        b = [True] * 20 + [True] * 6                 # B fixed 6 tasks A failed, broke none
        r = S.mcnemar(a, b)
        self.assertEqual((r["a_only"], r["b_only"]), (0, 6))
        self.assertAlmostEqual(r["p"], 2 / 64)
        self.assertTrue(r["significant"])
        self.assertGreater(r["lo"], 0)

    def test_mcnemar_no_discordance(self):
        r = S.mcnemar([True, False] * 5, [True, False] * 5)
        self.assertEqual(r["p"], 1.0)
        self.assertFalse(r["significant"])


class Traffic(unittest.TestCase):
    EDGES = [0, 10000, 50000, 120000, 262144]

    def test_ols_recovers_coefficients(self):
        X = [[1.0, x, x * x] for x in range(10)]
        y = [2 + 3 * x - 0.5 * x * x for x in range(10)]
        f = S.ols(X, y)
        for got, want in zip(f["beta"], (2, 3, -0.5)):
            self.assertAlmostEqual(got, want, places=8)

    def _rows(self, rng, n, speed, acc_mean):
        rows = []
        for _ in range(n):
            depth = rng.choice([5000, 30000, 90000, 150000])
            acc = min(0.98, max(0.3, rng.gauss(acc_mean, 0.08)))
            base = 60 * (1 - depth / 400000)
            tps = base * speed * math.exp(0.8 * (acc - 0.8) + rng.gauss(0, 0.02))
            rows.append(dict(depth=depth, accept=acc, decode_tps=tps))
        return rows

    def test_acceptance_confound_removed(self):
        rng = random.Random(5)
        before = self._rows(rng, 150, 1.00, 0.85)
        after = self._rows(rng, 150, 1.00, 0.75)     # same engine, less predictable content
        raw = S.traffic_effect(before, after, self.EDGES, adjust=False)
        adj = S.traffic_effect(before, after, self.EDGES, adjust=True)
        self.assertLess(raw["pct_hi"], 0)            # unadjusted: a false "slower"
        self.assertLess(adj["pct_lo"], 0)            # adjusted: no change ...
        self.assertGreater(adj["pct_hi"], 0)
        self.assertLess(adj["resid_sd"], raw["resid_sd"])

    def test_detects_real_change(self):
        rng = random.Random(6)
        adj = S.traffic_effect(self._rows(rng, 120, 1.0, 0.8), self._rows(rng, 120, 1.05, 0.8),
                               self.EDGES)
        self.assertGreater(adj["pct_lo"], 2.5)
        self.assertLess(adj["pct_hi"], 7.5)

    def test_noise_split(self):
        rng = random.Random(8)
        n = S.traffic_noise(self._rows(rng, 300, 1.0, 0.8), self.EDGES)
        self.assertLess(n["adjusted_sd"], n["raw_sd"])
        self.assertGreater(n["explained"], 0.5)

    def test_requests_needed(self):
        self.assertGreater(S.requests_needed(0.084, 0.03), S.requests_needed(0.036, 0.03))


class RequestTime(unittest.TestCase):
    def test_harmonic_over_depths(self):
        s = S.request_seconds({8192: 60.0, 131072: 30.0}, {8192: 1, 131072: 1}, output_tokens=120)
        self.assertAlmostEqual(s, 120 / 40.0)        # harmonic mean of 60 and 30 is 40
        self.assertIsNone(S.request_seconds({8192: 60.0}, {8192: 1, 131072: 1}, 100))


if __name__ == "__main__":
    unittest.main()
