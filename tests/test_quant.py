import csv
import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout

from quant import indicators as ind
from quant.data import Bars, load_csv
from quant.plan import RULES, analyze, classify_macro

import analyze as cli


def synthetic_bars(n=300, drift=0.001, vol=0.01, seed=7, start=100.0, volume=2_000_000):
    """Deterministic geometric random walk. Test fixture only, not market data."""
    rng = random.Random(seed)
    dates, o, h, l, c, v = [], [], [], [], [], []
    price = start
    for i in range(n):
        open_ = price
        price = price * (1 + drift + rng.gauss(0, vol))
        hi = max(open_, price) * (1 + abs(rng.gauss(0, vol / 2)))
        lo = min(open_, price) * (1 - abs(rng.gauss(0, vol / 2)))
        dates.append(f"2025-{1 + i // 28 % 12:02d}-{1 + i % 28:02d}-{i:04d}")
        o.append(open_); h.append(hi); l.append(lo); c.append(price)
        v.append(volume * (1 + rng.random()))
    return Bars(dates, o, h, l, c, v)


class IndicatorTests(unittest.TestCase):
    def test_sma(self):
        self.assertEqual(ind.sma([1, 2, 3, 4, 5], 3), [None, None, 2.0, 3.0, 4.0])

    def test_ema_seed_and_step(self):
        out = ind.ema([1, 2, 3, 4], 3)
        self.assertEqual(out[:3], [None, None, 2.0])
        self.assertAlmostEqual(out[3], 4 * 0.5 + 2.0 * 0.5)

    def test_rsi_extremes(self):
        self.assertEqual(ind.rsi(list(range(1, 30)))[-1], 100.0)
        self.assertEqual(ind.rsi(list(range(30, 1, -1)))[-1], 0.0)

    def test_rsi_reference_value(self):
        # Classic Wilder example series (StockCharts RSI walkthrough): first RSI 70.53 there,
        # 70.46 with unrounded intermediates.
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                  45.89, 46.03, 45.61, 46.28, 46.28]
        self.assertAlmostEqual(ind.rsi(closes)[14], 70.53, delta=0.1)

    def test_atr_constant_range(self):
        n = 30
        highs, lows, closes = [11.0] * n, [9.0] * n, [10.0] * n
        self.assertAlmostEqual(ind.atr(highs, lows, closes)[-1], 2.0)

    def test_adx_strong_trend(self):
        closes = [100 + i for i in range(60)]
        adx, pdi, mdi = ind.adx([c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
        self.assertGreater(adx[-1], 50)
        self.assertGreater(pdi[-1], mdi[-1])

    def test_bollinger_flat(self):
        mid, up, lo = ind.bollinger([5.0] * 25)
        self.assertEqual((mid[-1], up[-1], lo[-1]), (5.0, 5.0, 5.0))

    def test_obv_and_slope(self):
        self.assertEqual(ind.obv([1, 2, 2, 1], [10, 20, 30, 40]), [0.0, 20.0, 20.0, -20.0])
        self.assertAlmostEqual(ind.linear_slope([1, 3, 5, 7]), 2.0)


class PlanTests(unittest.TestCase):
    REQUIRED = ("CONTEXT", "DECISION", "ENTRY", "STOP_LOSS", "TARGET", "CONFIDENCE")

    def assert_valid(self, plan):
        for key in self.REQUIRED:
            self.assertIn(key, plan)
        self.assertIn(plan["DECISION"], ("BUY", "SELL", "WAIT", "STAY OUT"))
        self.assertTrue(1 <= plan["CONFIDENCE"]["score"] <= 10)
        json.dumps(plan)  # must be serialisable

    def test_insufficient_history_stays_out(self):
        plan = analyze(synthetic_bars(n=120), "TEST")
        self.assert_valid(plan)
        self.assertEqual(plan["DECISION"], "STAY OUT")

    def test_extreme_volatility_stays_out(self):
        plan = analyze(synthetic_bars(vol=0.08), "TEST")
        self.assertEqual(plan["DECISION"], "STAY OUT")

    def test_illiquid_stays_out(self):
        plan = analyze(synthetic_bars(volume=100), "TEST")
        self.assertEqual(plan["DECISION"], "STAY OUT")

    def test_risk_cap_and_rr_hold_for_every_trade(self):
        # Walk-forward: evaluate each series as of many past dates, like a live signal scan.
        trades = 0
        for seed in range(6):
            for drift in (0.0015, -0.0015):
                b = synthetic_bars(n=500, seed=seed, drift=drift, vol=0.012)
                for end in range(220, 500, 4):
                    window = Bars(*(col[:end] for col in (b.dates, b.opens, b.highs, b.lows, b.closes, b.volumes)))
                    plan = analyze(window, "TEST")
                    self.assert_valid(plan)
                    if plan["DECISION"] in ("BUY", "SELL"):
                        trades += 1
                        self.assert_trade_invariants(plan)
                    elif plan["DECISION"] == "WAIT":
                        self.assertTrue(plan["blockers"])
        self.assertGreater(trades, 0, "fixture set should produce at least one actionable plan")

    def assert_trade_invariants(self, plan):
        self.assertLessEqual(plan["STOP_LOSS"]["risk_pct_from_worst_fill"], RULES["max_risk_pct"])
        self.assertGreaterEqual(plan["TARGET"]["reward_risk"], RULES["min_reward_risk"])
        self.assertGreaterEqual(plan["CONFIDENCE"]["score"], RULES["min_confidence_to_trade"])
        self.assertEqual(plan["blockers"], [])
        lo, hi = plan["ENTRY"]["zone"]
        if plan["DECISION"] == "BUY":
            self.assertLess(plan["STOP_LOSS"]["price"], lo)
            self.assertGreater(plan["TARGET"]["price"], hi)
        else:
            self.assertGreater(plan["STOP_LOSS"]["price"], hi)
            self.assertLess(plan["TARGET"]["price"], lo)

    def test_deterministic(self):
        bars = synthetic_bars(seed=3, drift=0.002)
        self.assertEqual(json.dumps(analyze(bars, "T")), json.dumps(analyze(bars, "T")))

    def test_macro_classification(self):
        self.assertEqual(classify_macro(None)[0], "UNKNOWN")
        self.assertEqual(classify_macro({"vix": 14, "spx_above_200dma": True})[0], "RISK_ON")
        self.assertEqual(classify_macro({"vix": 35, "credit_spreads": "widening"})[0], "RISK_OFF")


class IOTests(unittest.TestCase):
    def test_csv_roundtrip_and_cli(self):
        bars = synthetic_bars()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "abc.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
                # Write in reverse to confirm the loader sorts chronologically.
                for row in reversed(list(zip(bars.dates, bars.opens, bars.highs, bars.lows, bars.closes, bars.volumes))):
                    w.writerow(row)
            loaded = load_csv(path)
            self.assertEqual(loaded.dates, bars.dates)
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main([path])
            out = json.loads(buf.getvalue())
            self.assertEqual(out["ticker"], "ABC")

    def test_bad_ohlc_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w") as fh:
                fh.write("date,open,high,low,close,volume\n2025-01-01,10,9,8,9.5,100\n")
            with self.assertRaises(ValueError):
                load_csv(path)


if __name__ == "__main__":
    unittest.main()
