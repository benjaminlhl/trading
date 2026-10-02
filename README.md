# trading — deterministic trade-plan engine

A rules-only analyst. All indicators are computed by scripts in `quant/`, using only the Python
standard library. Each decision comes from fixed thresholds in `quant/plan.RULES`, and every
confidence point is itemised in the output, so nothing is estimated or made up.

## Usage

```bash
python analyze.py data/AAPL.csv --ticker AAPL --macro macro.example.json
python analyze.py data/*.csv                      # batch; ticker = filename
python -m unittest discover -s tests -t .         # test suite
```

Input: daily OHLCV CSV with `date,open,high,low,close,volume` columns (common spellings are
accepted, rows are sorted by date, and bad OHLC rows are rejected). You need at least 200 bars.
Macro input is optional. When it is missing, macro is reported as `UNKNOWN` and earns no points.

## Output (JSON)

| Field | Meaning |
|---|---|
| `CONTEXT` | Trend (close vs SMA50/200 + SMA50 slope, ADX), volume (RVOL vs prior 20 bars, OBV slope), volatility (ATR % of price), macro regime |
| `DECISION` | `BUY` / `SELL` / `WAIT` / `STAY OUT` |
| `ENTRY` | Limit zone; for `WAIT`, a conditional zone + the trigger that would activate it |
| `STOP_LOSS` | Structural stop (10-bar swing ± 0.25 ATR, never tighter than 1 ATR); risk % measured from the **worst** fill in the zone |
| `TARGET` | 2.5R from worst fill, plus nearest support/resistance in the path |
| `CONFIDENCE` | 1–10 = `round(10 * points / 9)`, with the full point ledger |
| `blockers` | Every reason a trade was downgraded |

## Decision rules

**STAY OUT** (hard gates): < 200 bars · ATR ≥ 8% of price (any stop within the cap would be noise) ·
20-day average dollar volume < $1M.

**BUY / SELL** only if *all* hold, otherwise **WAIT**:
- primary trend aligned with the side
- close within 1.5 ATR of EMA20 (no chasing)
- structural stop ≤ **8%** from worst fill
- nearest resistance/support ≥ 2R away
- confidence ≥ 7

**Points (max 9) per side:** trend aligned +2 · ADX ≥ 20 with DI agreement +1 · MACD vs signal +1 ·
MACD histogram direction +1 · RSI in momentum zone (50–70 long / 30–50 short) +1 · last bar in
direction on RVOL ≥ 1 +1 · OBV slope +1 · supportive macro +1.
Penalties: RSI exhaustion (> 75 / < 25) −1 · HIGH volatility −1 · opposing macro −2.

In a walk-forward scan of synthetic data, about 1% of snapshots produce a trade. WAIT is meant to
be the default.

This is a decision-support tool, not investment advice. It places no orders.
