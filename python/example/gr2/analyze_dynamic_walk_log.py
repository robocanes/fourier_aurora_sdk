#!/usr/bin/env python3

import argparse
import csv
import math


ACTION_TO_WHOLE_BODY_INDEX = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18,
    22, 23, 24, 25,
]

ACTION_LABELS = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]


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


def mean(values):
    return sum(values) / len(values) if values else math.nan


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else math.nan


def print_tilt_summary(rows):
    values = []
    for row in rows:
        if not all(row.get(name) for name in ("projected_gravity_x", "projected_gravity_y", "projected_gravity_z")):
            continue
        gx = float(row["projected_gravity_x"])
        gy = float(row["projected_gravity_y"])
        gz = float(row["projected_gravity_z"])
        values.append(math.degrees(math.atan2(math.hypot(gx, gy), abs(gz))))
    print_range("tilt_deg", values)


def print_action_delta_summary(rows):
    raw_names = [f"raw_action_{index:02d}" for index in range(len(ACTION_LABELS))]
    if not rows or any(name not in rows[0] for name in raw_names):
        return

    deltas = []
    previous = None
    for row in rows:
        values = [float(row[name]) for name in raw_names]
        if previous is not None:
            deltas.extend(values[index] - previous[index] for index in range(len(values)))
        previous = values
    print_range("raw_action_delta", deltas)
    print(f"raw_action_delta_rms: {rms(deltas):.4f}")


def print_joint_tracking_summary(rows):
    if not rows:
        return

    tracking = []
    target_means = {}
    q_means = {}
    for action_index, whole_body_index in enumerate(ACTION_TO_WHOLE_BODY_INDEX):
        target_name = f"target_action_joint_{action_index:02d}"
        q_name = f"q_{whole_body_index:02d}"
        if target_name not in rows[0] or q_name not in rows[0]:
            continue

        targets = column_values(rows, target_name)
        positions = column_values(rows, q_name)
        if len(targets) != len(positions) or not targets:
            continue

        label = ACTION_LABELS[action_index]
        errors = [targets[index] - positions[index] for index in range(len(targets))]
        tracking.append((rms(errors), label))
        target_means[label] = mean(targets)
        q_means[label] = mean(positions)

    if tracking:
        print("worst_tracking_rms:")
        for value, label in sorted(tracking, reverse=True)[:8]:
            print(f"  {label}: {value:.4f}")

    arm_labels = [
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]
    if all(label in target_means for label in arm_labels):
        print("arm_target_mean:")
        for label in arm_labels:
            print(f"  {label}: {target_means[label]:+.4f}")
        print("arm_q_mean:")
        for label in arm_labels:
            print(f"  {label}: {q_means[label]:+.4f}")


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
    print_tilt_summary(rows)
    print_action_delta_summary(rows)
    print_joint_tracking_summary(rows)
    print_range("all_q", prefixed_values(rows, "q_"))
    print_range("all_qd", prefixed_values(rows, "qd_"))


if __name__ == "__main__":
    main()
