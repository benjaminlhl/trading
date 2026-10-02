"""Technical indicators implemented with the standard library only.

Every function takes plain Python lists and returns a list of the same length,
with ``None`` during the warm-up period. No value is ever estimated or filled.
"""

from math import sqrt


def sma(values, period):
    out = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, len(values)):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out


def ema(values, period):
    """EMA seeded with the SMA of the first ``period`` non-None values."""
    out = [None] * len(values)
    start = next((i for i, v in enumerate(values) if v is not None), None)
    if start is None or len(values) - start < period:
        return out
    k = 2.0 / (period + 1)
    seed_end = start + period - 1
    prev = sum(values[start:seed_end + 1]) / period
    out[seed_end] = prev
    for i in range(seed_end + 1, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(closes, period=14):
    """Wilder's RSI."""
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain, avg_loss):
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def true_range(highs, lows, closes):
    out = [highs[0] - lows[0]] if highs else []
    for i in range(1, len(highs)):
        prev_close = closes[i - 1]
        out.append(max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close)))
    return out


def atr(highs, lows, closes, period=14):
    """Wilder's ATR; first value is the mean TR of bars 1..period."""
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    tr = true_range(highs, lows, closes)
    prev = sum(tr[1:period + 1]) / period
    out[period] = prev
    for i in range(period + 1, len(closes)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def macd(closes, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema, slow_ema = ema(closes, fast), ema(closes, slow)
    line = [f - s if f is not None and s is not None else None for f, s in zip(fast_ema, slow_ema)]
    sig = ema(line, signal)
    hist = [m - s if m is not None and s is not None else None for m, s in zip(line, sig)]
    return line, sig, hist


def bollinger(closes, period=20, k=2.0):
    """Returns (middle, upper, lower) using population standard deviation."""
    mid = sma(closes, period)
    upper, lower = [None] * len(closes), [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mean = mid[i]
        sd = sqrt(sum((x - mean) ** 2 for x in window) / period)
        upper[i], lower[i] = mean + k * sd, mean - k * sd
    return mid, upper, lower


def adx(highs, lows, closes, period=14):
    """Wilder's ADX. Returns (adx, plus_di, minus_di)."""
    n = len(closes)
    adx_out, pdi_out, mdi_out = [None] * n, [None] * n, [None] * n
    if n <= 2 * period:
        return adx_out, pdi_out, mdi_out
    tr = true_range(highs, lows, closes)
    pdm, mdm = [0.0] * n, [0.0] * n
    for i in range(1, n):
        up, down = highs[i] - highs[i - 1], lows[i - 1] - lows[i]
        pdm[i] = up if up > down and up > 0 else 0.0
        mdm[i] = down if down > up and down > 0 else 0.0
    s_tr, s_pdm, s_mdm = sum(tr[1:period + 1]), sum(pdm[1:period + 1]), sum(mdm[1:period + 1])
    dx = [None] * n
    for i in range(period, n):
        if i > period:
            s_tr = s_tr - s_tr / period + tr[i]
            s_pdm = s_pdm - s_pdm / period + pdm[i]
            s_mdm = s_mdm - s_mdm / period + mdm[i]
        pdi = 100.0 * s_pdm / s_tr if s_tr else 0.0
        mdi = 100.0 * s_mdm / s_tr if s_tr else 0.0
        pdi_out[i], mdi_out[i] = pdi, mdi
        dx[i] = 100.0 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
    first = 2 * period - 1
    prev = sum(dx[period:first + 1]) / period
    adx_out[first] = prev
    for i in range(first + 1, n):
        prev = (prev * (period - 1) + dx[i]) / period
        adx_out[i] = prev
    return adx_out, pdi_out, mdi_out


def obv(closes, volumes):
    out = [0.0] * len(closes)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - volumes[i]
        else:
            out[i] = out[i - 1]
    return out


def linear_slope(values):
    """Least-squares slope of ``values`` against their index."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean, y_mean = (n - 1) / 2.0, sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den
