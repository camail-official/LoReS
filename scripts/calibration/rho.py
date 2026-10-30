"""Aggregate state-range diagnostics into the value/key grouping ratio ρ."""

import argparse
import json
import math
from pathlib import Path


def rho(ranges):
    values, keys, counts = [], [], set()
    for layer in ranges.values():
        if not layer["P_val"] or len(layer["P_val"]) != len(layer["P_key"]):
            raise ValueError("Expected matching, nonempty per-head P_val/P_key arrays")
        values.extend(layer["P_val"])
        keys.extend(layer["P_key"])
        counts.add(layer["n_snapshots"])
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("Layers must use the same positive snapshot count")
    if any(not math.isfinite(x) or x < 0 for x in values + keys) or not math.fsum(keys):
        raise ValueError("Range statistics must be finite, nonnegative, with positive total P_key")
    # P_val/P_key already average over row/column counts; sum heads before dividing.
    return math.fsum(values) / math.fsum(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, nargs="+", required=True, help="explicit *_rangediag.json files")
    args = parser.parse_args()
    results = {}
    for path in args.ranges:
        model = path.stem.removesuffix("_rangediag")
        if model in results:
            parser.error(f"Duplicate model: {model}")
        results[model] = {"rho": rho(json.loads(path.read_text())), "source": str(path.resolve())}
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
