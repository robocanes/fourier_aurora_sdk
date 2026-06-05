#!/usr/bin/env python3

import argparse
import csv
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a GR2 dynamic-walk telemetry CSV.")
    parser.add_argument("log_path")
    return parser.parse_args()


def percentile(values, pct):
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def column_values(rows, name):
    return [float(row[name]) for row in rows if row.get(name) not in (None, "")]


def prefixed_values(rows, prefix):
    values = []
    for row in rows:
        for name, value in row.items():
            if name.startswith(prefix) and value:
                values.append(float(value))
    return values


def print_range(label, values):
    if not values:
        print(f"{label}: no data")
        return
    mean = sum(values) / len(values)
    print(
        f"{label}: min={min(values):+.4f} mean={mean:+.4f} "
        f"p95={percentile(values, 95):+.4f} max={max(values):+.4f}"
    )


def main():
    args = parse_args()
    with open(args.log_path, newline="") as log_file:
        rows = list(csv.DictReader(log_file))

    if not rows:
        raise RuntimeError(f"No rows found in {args.log_path}")

    monotonic = column_values(rows, "monotonic_s")
    duration = max(monotonic) - min(monotonic) if len(monotonic) > 1 else 0.0
    rate = (len(rows) - 1) / duration if duration > 0.0 else math.nan

    print(f"rows: {len(rows)}")
    print(f"duration_s: {duration:.3f}")
    print(f"estimated_rate_hz: {rate:.2f}")
    print_range("cmd_x", column_values(rows, "cmd_x"))
    print_range("cmd_y", column_values(rows, "cmd_y"))
    print_range("cmd_yaw", column_values(rows, "cmd_yaw"))
    print_range("action_abs_max", column_values(rows, "action_abs_max"))
    print_range("step_elapsed_s", column_values(rows, "step_elapsed_s"))
    print_range("loop_overrun_s", column_values(rows, "loop_overrun_s"))
    print_range("all_q", prefixed_values(rows, "q_"))
    print_range("all_qd", prefixed_values(rows, "qd_"))


if __name__ == "__main__":
    main()
