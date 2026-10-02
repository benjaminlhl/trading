#!/usr/bin/env python3
"""Generate a deterministic JSON trade plan from an OHLCV CSV.

Usage:
    python analyze.py data/AAPL.csv --ticker AAPL [--macro macro.json]
"""

import argparse
import json
import os
import sys

from quant.data import load_csv
from quant.plan import analyze


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="OHLCV CSV file(s): date,open,high,low,close,volume (daily bars)")
    p.add_argument("--ticker", help="Ticker label (defaults to CSV filename; single file only)")
    p.add_argument("--macro", help="JSON file with macro inputs (vix, spx_above_200dma, credit_spreads, yield_curve_inverted)")
    args = p.parse_args(argv)
    if args.ticker and len(args.csv) > 1:
        p.error("--ticker can only be used with a single CSV")

    macro = None
    if args.macro:
        with open(args.macro) as fh:
            macro = json.load(fh)

    plans = []
    for path in args.csv:
        ticker = args.ticker or os.path.splitext(os.path.basename(path))[0].upper()
        plans.append(analyze(load_csv(path), ticker, macro))
    json.dump(plans[0] if len(plans) == 1 else plans, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
