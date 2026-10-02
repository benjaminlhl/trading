"""Rule-based regime classification and trade-plan generation.

All thresholds live in ``RULES`` so every decision is reproducible and auditable.
Decisions: BUY, SELL (short / exit long), WAIT (setup forming or unclear),
STAY OUT (conditions make any trade invalid).
"""

from . import indicators as ind

RULES = {
    "min_bars": 200,
    "max_risk_pct": 8.0,          # hard cap: stop distance from worst-case entry
    "min_reward_risk": 2.0,
    "target_r_multiple": 2.5,
    "min_confidence_to_trade": 7,
    "adx_trending": 20.0,
    "extended_atr": 1.5,          # close farther than this many ATRs from EMA20 = chasing
    "swing_lookback": 10,
    "structure_lookback": 60,
    "rel_volume_lookback": 20,
    "max_atr_pct": 8.0,           # if one ATR >= risk cap, every valid stop sits inside noise
    "min_avg_dollar_volume": 1_000_000.0,
    "max_points": 9,
}


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def compute_indicators(bars):
    c, h, l, v = bars.closes, bars.highs, bars.lows, bars.volumes
    macd_line, macd_sig, macd_hist = ind.macd(c)
    adx_v, pdi, mdi = ind.adx(h, l, c)
    bb_mid, bb_up, bb_lo = ind.bollinger(c)
    obv_v = ind.obv(c, v)
    n_rv = RULES["rel_volume_lookback"]
    prior_vol = v[-n_rv - 1:-1]
    avg_vol = sum(prior_vol) / len(prior_vol) if prior_vol else 0.0
    sma50 = ind.sma(c, 50)
    swing, struct = RULES["swing_lookback"], RULES["structure_lookback"]
    return {
        "close": c[-1],
        "prev_close": c[-2],
        "ema20": ind.ema(c, 20)[-1],
        "sma50": sma50[-1],
        "sma50_10d_ago": sma50[-11],
        "sma200": ind.sma(c, 200)[-1],
        "rsi14": ind.rsi(c)[-1],
        "atr14": ind.atr(h, l, c)[-1],
        "macd": macd_line[-1],
        "macd_signal": macd_sig[-1],
        "macd_hist": macd_hist[-1],
        "macd_hist_prev": macd_hist[-2],
        "adx14": adx_v[-1],
        "plus_di": pdi[-1],
        "minus_di": mdi[-1],
        "bb_upper": bb_up[-1],
        "bb_lower": bb_lo[-1],
        "bb_width_pct": (bb_up[-1] - bb_lo[-1]) / bb_mid[-1] * 100.0,
        "rel_volume": v[-1] / avg_vol if avg_vol else None,
        "avg_dollar_volume_20": avg_vol * c[-1],
        "obv_slope_20": ind.linear_slope(obv_v[-20:]),
        "swing_low": min(l[-swing:]),
        "swing_high": max(h[-swing:]),
        "resistance": max(h[-struct - 1:-1]),
        "support": min(l[-struct - 1:-1]),
        "high_20": max(h[-21:-1]),
        "low_20": min(l[-21:-1]),
    }


def classify_regime(x, macro=None):
    close, atr = x["close"], x["atr14"]
    sma50_slope = x["sma50"] - x["sma50_10d_ago"]
    if close > x["sma50"] > x["sma200"] and sma50_slope > 0:
        trend = "UPTREND"
    elif close < x["sma50"] < x["sma200"] and sma50_slope < 0:
        trend = "DOWNTREND"
    else:
        trend = "RANGE"
    strength = "TRENDING" if x["adx14"] >= RULES["adx_trending"] else "WEAK/CHOPPY"

    rv = x["rel_volume"]
    if rv is None:
        volume = "UNKNOWN"
    elif rv >= 1.5:
        volume = "EXPANDING"
    elif rv < 0.7:
        volume = "CONTRACTING"
    else:
        volume = "NORMAL"
    flow = "ACCUMULATION" if x["obv_slope_20"] > 0 else "DISTRIBUTION" if x["obv_slope_20"] < 0 else "FLAT"

    atr_pct = atr / close * 100.0
    if atr_pct < 1.5:
        vol_regime = "LOW"
    elif atr_pct < 4.0:
        vol_regime = "NORMAL"
    elif atr_pct < RULES["max_atr_pct"]:
        vol_regime = "HIGH"
    else:
        vol_regime = "EXTREME"

    macro_regime, macro_notes = classify_macro(macro)
    summary = (
        f"{trend} ({strength}, ADX {x['adx14']:.1f}); volume {volume} "
        f"(RVOL {rv:.2f}x, OBV {flow}); volatility {vol_regime} (ATR {atr_pct:.2f}% of price); "
        f"macro {macro_regime}."
    ) if rv is not None else f"{trend} ({strength}); volume UNKNOWN; volatility {vol_regime}; macro {macro_regime}."
    return {
        "trend": trend,
        "trend_strength": strength,
        "sma50_slope_10d": _r(sma50_slope, 4),
        "volume": volume,
        "money_flow": flow,
        "volatility": vol_regime,
        "atr_pct": _r(atr_pct),
        "macro": macro_regime,
        "macro_notes": macro_notes,
        "summary": summary,
    }


def classify_macro(macro):
    """Macro inputs are user-supplied only; nothing is assumed when absent.

    Recognised keys: vix (float), spx_above_200dma (bool), credit_spreads ("widening"|"tightening"),
    yield_curve_inverted (bool). Unknown keys are ignored.
    """
    if not macro:
        return "UNKNOWN", ["No macro data supplied; macro point withheld."]
    score, notes = 0, []
    vix = macro.get("vix")
    if vix is not None:
        if vix >= 30:
            score -= 2
            notes.append(f"VIX {vix} >= 30: stress (-2)")
        elif vix >= 20:
            score -= 1
            notes.append(f"VIX {vix} in 20-30: elevated (-1)")
        else:
            score += 1
            notes.append(f"VIX {vix} < 20: calm (+1)")
    if "spx_above_200dma" in macro:
        up = bool(macro["spx_above_200dma"])
        score += 1 if up else -1
        notes.append(f"S&P 500 {'above' if up else 'below'} 200DMA ({'+1' if up else '-1'})")
    spreads = macro.get("credit_spreads")
    if spreads in ("widening", "tightening"):
        score += -1 if spreads == "widening" else 1
        notes.append(f"Credit spreads {spreads} ({'-1' if spreads == 'widening' else '+1'})")
    if macro.get("yield_curve_inverted") is True:
        score -= 1
        notes.append("Yield curve inverted (-1)")
    if not notes:
        return "UNKNOWN", ["Macro data supplied but no recognised keys."]
    regime = "RISK_ON" if score >= 2 else "RISK_OFF" if score <= -2 else "NEUTRAL"
    notes.append(f"Macro score {score:+d} -> {regime}")
    return regime, notes


def score_side(side, x, regime):
    """Explicit point ledger for one direction. Returns (points, ledger)."""
    bull = side == "long"
    ledger = []

    def add(pts, reason):
        ledger.append({"points": pts, "reason": reason})

    trend_ok = regime["trend"] == ("UPTREND" if bull else "DOWNTREND")
    add(2 if trend_ok else 0, f"Trend {regime['trend']} {'aligned' if trend_ok else 'not aligned'} with {side}")

    di_ok = (x["plus_di"] > x["minus_di"]) if bull else (x["minus_di"] > x["plus_di"])
    adx_ok = x["adx14"] >= RULES["adx_trending"] and di_ok
    add(1 if adx_ok else 0, f"ADX {x['adx14']:.1f} (+DI {x['plus_di']:.1f} / -DI {x['minus_di']:.1f})")

    macd_ok = (x["macd"] > x["macd_signal"]) if bull else (x["macd"] < x["macd_signal"])
    add(1 if macd_ok else 0, f"MACD {x['macd']:.3f} vs signal {x['macd_signal']:.3f}")

    hist_ok = (x["macd_hist"] > x["macd_hist_prev"]) if bull else (x["macd_hist"] < x["macd_hist_prev"])
    add(1 if hist_ok else 0, f"MACD histogram {'rising' if x['macd_hist'] > x['macd_hist_prev'] else 'falling'}")

    rsi_v = x["rsi14"]
    rsi_ok = 50 <= rsi_v <= 70 if bull else 30 <= rsi_v <= 50
    add(1 if rsi_ok else 0, f"RSI {rsi_v:.1f} {'in' if rsi_ok else 'outside'} {'50-70' if bull else '30-50'} momentum zone")
    if (bull and rsi_v > 75) or (not bull and rsi_v < 25):
        add(-1, f"RSI {rsi_v:.1f} at exhaustion extreme")

    day_dir = x["close"] > x["prev_close"] if bull else x["close"] < x["prev_close"]
    rv = x["rel_volume"] or 0.0
    add(1 if day_dir and rv >= 1.0 else 0, f"Last bar {'with' if day_dir else 'against'} {side} on RVOL {rv:.2f}x")

    flow_ok = x["obv_slope_20"] > 0 if bull else x["obv_slope_20"] < 0
    add(1 if flow_ok else 0, f"OBV 20-bar flow {regime['money_flow']}")

    macro = regime["macro"]
    if macro == ("RISK_ON" if bull else "RISK_OFF"):
        add(1, f"Macro {macro} supports {side}")
    elif macro == ("RISK_OFF" if bull else "RISK_ON"):
        add(-2, f"Macro {macro} opposes {side}")
    else:
        add(0, f"Macro {macro}: no contribution")

    if regime["volatility"] == "HIGH":
        add(-1, f"Volatility HIGH (ATR {regime['atr_pct']}%)")

    return sum(item["points"] for item in ledger), ledger


def build_levels(side, zone_low, zone_high, x):
    """Structural stop + R-multiple target. Risk is measured from the worst fill in the zone."""
    atr = x["atr14"]
    if side == "long":
        worst = zone_high
        structural = x["swing_low"] - 0.25 * atr
        stop = min(structural, worst - atr)  # never tighter than 1 ATR (noise floor)
        risk = worst - stop
        target = worst + RULES["target_r_multiple"] * risk
        obstacle = x["resistance"] if worst < x["resistance"] < target else None
        room = (x["resistance"] - worst) / risk if obstacle else None
    else:
        worst = zone_low
        structural = x["swing_high"] + 0.25 * atr
        stop = max(structural, worst + atr)
        risk = stop - worst
        target = worst - RULES["target_r_multiple"] * risk
        obstacle = x["support"] if target < x["support"] < worst else None
        room = (worst - x["support"]) / risk if obstacle else None
    risk_pct = risk / worst * 100.0
    return {
        "entry_zone": [_r(zone_low), _r(zone_high)],
        "worst_fill": _r(worst),
        "stop": _r(stop),
        "risk_pct": _r(risk_pct),
        "target": _r(target) if target > 0 else None,
        "reward_risk": RULES["target_r_multiple"] if target > 0 else None,
        "obstacle": _r(obstacle),
        "r_to_obstacle": _r(room),
        "within_risk_cap": risk_pct <= RULES["max_risk_pct"],
    }


def _confidence(points):
    return max(1, min(10, round(10 * max(points, 0) / RULES["max_points"])))


def analyze(bars, ticker, macro=None):
    if len(bars) < RULES["min_bars"]:
        return _stay_out(ticker, bars, f"Insufficient history: {len(bars)} bars < {RULES['min_bars']} required.")

    x = compute_indicators(bars)
    regime = classify_regime(x, macro)

    if regime["volatility"] == "EXTREME":
        return _stay_out(ticker, bars, f"ATR is {regime['atr_pct']}% of price; a stop within the "
                         f"{RULES['max_risk_pct']}% cap would sit inside one ATR of noise.", x, regime)
    if x["avg_dollar_volume_20"] < RULES["min_avg_dollar_volume"]:
        return _stay_out(ticker, bars, f"Illiquid: 20-day avg dollar volume ${x['avg_dollar_volume_20']:,.0f} "
                         f"< ${RULES['min_avg_dollar_volume']:,.0f}.", x, regime)

    long_pts, long_ledger = score_side("long", x, regime)
    short_pts, short_ledger = score_side("short", x, regime)
    side, pts, ledger = ("long", long_pts, long_ledger) if long_pts >= short_pts else ("short", short_pts, short_ledger)
    conf = _confidence(pts)
    close, atr, ema20 = x["close"], x["atr14"], x["ema20"]
    blockers = []

    extension = (close - ema20) / atr
    extended = extension > RULES["extended_atr"] if side == "long" else extension < -RULES["extended_atr"]
    aligned = regime["trend"] == ("UPTREND" if side == "long" else "DOWNTREND")

    if side == "long":
        levels = build_levels(side, close - 0.5 * atr, close, x)
    else:
        levels = build_levels(side, close, close + 0.5 * atr, x)

    if not aligned:
        blockers.append(f"Primary trend is {regime['trend']}, not aligned with {side}.")
    if extended:
        blockers.append(f"Price is {abs(extension):.2f} ATR from EMA20 (> {RULES['extended_atr']}): chasing.")
    if not levels["within_risk_cap"]:
        blockers.append(f"Structural stop implies {levels['risk_pct']}% risk > {RULES['max_risk_pct']}% cap.")
    if levels["target"] is None:
        blockers.append("Target would be non-positive.")
    if levels["r_to_obstacle"] is not None and levels["r_to_obstacle"] < RULES["min_reward_risk"]:
        blockers.append(f"{'Resistance' if side == 'long' else 'Support'} at {levels['obstacle']} is only "
                        f"{levels['r_to_obstacle']}R away (< {RULES['min_reward_risk']}R).")
    if conf < RULES["min_confidence_to_trade"]:
        blockers.append(f"Confidence {conf} < {RULES['min_confidence_to_trade']} minimum.")

    if blockers:
        decision = "WAIT"
        levels = _conditional_levels(side, x, extended, aligned)
    else:
        decision = "BUY" if side == "long" else "SELL"

    return {
        "ticker": ticker,
        "as_of": bars.dates[-1],
        "bars": len(bars),
        "CONTEXT": regime,
        "DECISION": decision,
        "bias": side,
        "ENTRY": {
            "zone": levels["entry_zone"],
            "type": "limit within zone" if decision in ("BUY", "SELL") else "conditional trigger",
            "conditional": decision == "WAIT",
            "trigger": levels.get("trigger"),
        },
        "STOP_LOSS": {"price": levels["stop"], "risk_pct_from_worst_fill": levels["risk_pct"],
                      "within_8pct_cap": levels["within_risk_cap"]},
        "TARGET": {"price": levels["target"], "reward_risk": levels["reward_risk"],
                   "nearest_obstacle": levels["obstacle"], "r_to_obstacle": levels["r_to_obstacle"]},
        "CONFIDENCE": {
            "score": conf,
            "formula": f"round(10 * max(points,0) / {RULES['max_points']}), clamped 1-10",
            "points": pts,
            "max_points": RULES["max_points"],
            "ledger": ledger,
            "opposing_side_points": short_pts if side == "long" else long_pts,
        },
        "blockers": blockers,
        "indicators": {k: _r(v, 4) for k, v in x.items()},
        "rules": RULES,
    }


def _conditional_levels(side, x, extended, aligned):
    """What would have to happen before this becomes actionable."""
    atr, ema20 = x["atr14"], x["ema20"]
    if aligned and extended:
        lo, hi = ema20 - 0.25 * atr, ema20 + 0.25 * atr
        trigger = f"Pullback into EMA20 zone {lo:.2f}-{hi:.2f} with confirmation, then re-run."
    elif side == "long":
        lo, hi = x["high_20"], x["high_20"] + 0.25 * atr
        trigger = f"Daily close above 20-day high {lo:.2f} on RVOL >= 1.0, then re-run."
    else:
        lo, hi = x["low_20"] - 0.25 * atr, x["low_20"]
        trigger = f"Daily close below 20-day low {hi:.2f} on RVOL >= 1.0, then re-run."
    levels = build_levels(side, lo, hi, x)
    levels["trigger"] = trigger
    return levels


def _stay_out(ticker, bars, reason, x=None, regime=None):
    return {
        "ticker": ticker,
        "as_of": bars.dates[-1] if len(bars) else None,
        "bars": len(bars),
        "CONTEXT": regime or {"summary": "Not evaluated."},
        "DECISION": "STAY OUT",
        "ENTRY": {"zone": None, "type": None, "conditional": False, "trigger": None},
        "STOP_LOSS": {"price": None, "risk_pct_from_worst_fill": None, "within_8pct_cap": None},
        "TARGET": {"price": None, "reward_risk": None},
        "CONFIDENCE": {"score": 1, "formula": "Hard gate failed; no score computed.", "points": None},
        "blockers": [reason],
        "indicators": {k: _r(v, 4) for k, v in x.items()} if x else None,
        "rules": RULES,
    }
