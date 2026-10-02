"""OHLCV CSV loading. Accepts common column spellings (Date/Open/High/Low/Close/Volume)."""

import csv
from dataclasses import dataclass

_ALIASES = {
    "date": ("date", "datetime", "timestamp", "time"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "adj close", "adj_close", "adjclose"),
    "volume": ("volume", "vol", "v"),
}


@dataclass
class Bars:
    dates: list
    opens: list
    highs: list
    lows: list
    closes: list
    volumes: list

    def __len__(self):
        return len(self.closes)


def _resolve_columns(fieldnames):
    lowered = {name.strip().lower(): name for name in fieldnames}
    mapping = {}
    for key, aliases in _ALIASES.items():
        # Prefer the raw "close" over "adj close" when both exist.
        found = next((lowered[a] for a in aliases if a in lowered), None)
        if found is None:
            raise ValueError(f"CSV is missing a '{key}' column (have: {list(fieldnames)})")
        mapping[key] = found
    return mapping


def load_csv(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        cols = _resolve_columns(reader.fieldnames)
        rows = []
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append((
                    row[cols["date"]].strip(),
                    float(row[cols["open"]]),
                    float(row[cols["high"]]),
                    float(row[cols["low"]]),
                    float(row[cols["close"]]),
                    float(row[cols["volume"]] or 0),
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: unparseable row {row}") from exc
    rows.sort(key=lambda r: r[0])
    for date, o, h, l, c, _ in rows:
        if not (l <= min(o, c) and h >= max(o, c)) or c <= 0:
            raise ValueError(f"{path}: inconsistent OHLC on {date}: O={o} H={h} L={l} C={c}")
    return Bars(*(list(col) for col in zip(*rows))) if rows else Bars([], [], [], [], [], [])
