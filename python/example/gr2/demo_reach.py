#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time

import numpy
import torch

aurora_client_path = "/opt/venv/lib/python3.10/site-packages"
if os.path.isdir(aurora_client_path) and aurora_client_path not in sys.path:
    sys.path.append(aurora_client_path)

from fourier_aurora_client import AuroraClient


ROBOT_NUM_JOINTS = 29
POLICY_NUM_ACTIONS = 29
OBS_LEN = 101
STACK_SIZE = 5
DEFAULT_POLICY_FILE = "policy_gr2_reach_latest_jit.pt"
BALANCE_POLICY_NUM_ACTIONS = 21
BALANCE_OBS_LEN = 88
DEFAULT_BALANCE_POLICY_FILE = "policy_gr2_upper_body_jit.pt"

TARGET_X_RANGE = (0.18, 0.38)
TARGET_Y_ABS_RANGE = (0.10, 0.25)
TARGET_Z_RANGE = (-0.15, 0.375)
TARGET_ROLL_RANGE = (-0.20, 0.20)
TARGET_PITCH_RANGE = (-0.20, 0.20)
TARGET_YAW_RANGE = (-0.35, 0.35)

GROUP_NAMES = [
    "left_leg",
    "right_leg",
    "waist",
    "head",
    "left_manipulator",
    "right_manipulator",
]

DEFAULT_WHOLE_BODY_POSITION = numpy.array([
    -0.1309, 0.0, 0.0, 0.2618, -0.1309, 0.0,
    -0.1309, 0.0, 0.0, 0.2618, -0.1309, 0.0,
    0.0,
    0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=numpy.float32)

# Reach policy action order matches Aurora whole_body order:
# left leg, right leg, waist, head, left arm, right arm.
ACTION_TO_WHOLE_BODY_INDEX = numpy.arange(ROBOT_NUM_JOINTS, dtype=numpy.int64)

MAIN_BODY_ACTION_INDICES = numpy.arange(0, 13, dtype=numpy.int64)
LEG_ACTION_INDICES = numpy.arange(0, 12, dtype=numpy.int64)
WAIST_ACTION_INDICES = numpy.array([12], dtype=numpy.int64)
HEAD_ACTION_INDICES = numpy.array([13, 14], dtype=numpy.int64)
ARM_ACTION_INDICES = numpy.arange(15, 29, dtype=numpy.int64)
LEFT_ARM_ACTION_INDICES = numpy.arange(15, 22, dtype=numpy.int64)
RIGHT_ARM_ACTION_INDICES = numpy.arange(22, 29, dtype=numpy.int64)

MAIN_BODY_WHOLE_BODY_INDICES = numpy.array([
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
], dtype=numpy.int64)
LEG_WHOLE_BODY_INDICES = numpy.arange(0, 12, dtype=numpy.int64)
HEAD_WHOLE_BODY_INDICES = numpy.array([13, 14], dtype=numpy.int64)
LEFT_ARM_WHOLE_BODY_INDICES = numpy.arange(15, 22, dtype=numpy.int64)
RIGHT_ARM_WHOLE_BODY_INDICES = numpy.arange(22, 29, dtype=numpy.int64)
ARM_WHOLE_BODY_INDICES = numpy.arange(15, 29, dtype=numpy.int64)

DEFAULT_ACTION_POSITION = DEFAULT_WHOLE_BODY_POSITION[ACTION_TO_WHOLE_BODY_INDEX]

BALANCE_ACTION_TO_WHOLE_BODY_INDEX = numpy.array([
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18,
    22, 23, 24, 25,
], dtype=numpy.int64)

BALANCE_ACTION_SCALE = numpy.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.75,
    0.20, 0.06, 0.06, 0.10,
    0.20, 0.06, 0.06, 0.10,
], dtype=numpy.float32)

BALANCE_ACTION_MIN = numpy.array([
    -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
    -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
    -2.6180,
    -2.9671, -0.5236, -1.8326, -1.5272,
    -2.9671, -2.7925, -1.8326, -1.5272,
], dtype=numpy.float32)

BALANCE_ACTION_MAX = numpy.array([
    2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
    2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
    2.6180,
    2.9671, 2.7925, 1.8326, 0.4800,
    2.9671, 0.5236, 1.8326, 0.4800,
], dtype=numpy.float32)

BALANCE_RAW_ACTION_CLIP_MIN = BALANCE_ACTION_MIN - 1.0
BALANCE_RAW_ACTION_CLIP_MAX = BALANCE_ACTION_MAX + 1.0

ACTION_SCALE = numpy.array([
    0.12, 0.08, 0.08, 0.12, 0.10, 0.08,
    0.12, 0.08, 0.08, 0.12, 0.10, 0.08,
    0.25,
    0.35, 0.22,
    0.35, 0.22, 0.28, 0.28, 0.18, 0.18, 0.18,
    0.35, 0.22, 0.28, 0.28, 0.18, 0.18, 0.18,
], dtype=numpy.float32)

DOF_POS_OBS_SCALE = numpy.array([
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    0.75,
    1.0, 1.0,
    0.30, 0.30, 0.30, 0.30, 0.50, 0.50, 0.50,
    0.30, 0.30, 0.30, 0.30, 0.50, 0.50, 0.50,
], dtype=numpy.float32)

DOF_VEL_OBS_SCALE = numpy.array([
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    0.75,
    1.0, 1.0,
    0.40, 0.40, 0.40, 0.40, 0.60, 0.60, 0.60,
    0.40, 0.40, 0.40, 0.40, 0.60, 0.60, 0.60,
], dtype=numpy.float32)

OBS_DOF_POS_SCALE = 1.0
OBS_DOF_VEL_SCALE = 0.1

ACTION_MIN = numpy.array([
    -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
    -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
    -2.6180,
    -1.3963, -0.5236,
    -2.9671, -0.5236, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
    -2.9671, -2.7925, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
], dtype=numpy.float32)

ACTION_MAX = numpy.array([
    2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
    2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
    2.6180,
    1.3963, 0.5236,
    2.9671, 2.7925, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
    2.9671, 0.5236, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
], dtype=numpy.float32)

ACTION_SCALE = ACTION_SCALE[ACTION_TO_WHOLE_BODY_INDEX]
DOF_POS_OBS_SCALE = DOF_POS_OBS_SCALE[ACTION_TO_WHOLE_BODY_INDEX]
DOF_VEL_OBS_SCALE = DOF_VEL_OBS_SCALE[ACTION_TO_WHOLE_BODY_INDEX]
ACTION_MIN = ACTION_MIN[ACTION_TO_WHOLE_BODY_INDEX]
ACTION_MAX = ACTION_MAX[ACTION_TO_WHOLE_BODY_INDEX]

RAW_ACTION_CLIP_MIN = ACTION_MIN - 1.0
RAW_ACTION_CLIP_MAX = ACTION_MAX + 1.0
# Gym clips raw actions at 100, but this bridge sends position targets directly.
# Keep the default conservative while allowing --raw-action-clip-abs 100 for exact Gym-style tests.
RAW_ACTION_CLIP_ABS = 4.0

LEGACY_POLICY_NUM_ACTIONS = 27
LEGACY_OBS_LEN = 99
LEGACY_ACTION_TO_WHOLE_BODY_INDEX = numpy.array([
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
], dtype=numpy.int64)
LEGACY_MAIN_BODY_ACTION_INDICES = numpy.arange(0, 13, dtype=numpy.int64)
LEGACY_LEG_ACTION_INDICES = numpy.arange(0, 12, dtype=numpy.int64)
LEGACY_WAIST_ACTION_INDICES = numpy.array([12], dtype=numpy.int64)
LEGACY_ARM_ACTION_INDICES = numpy.arange(13, 27, dtype=numpy.int64)
LEGACY_LEFT_ARM_ACTION_INDICES = numpy.arange(13, 20, dtype=numpy.int64)
LEGACY_RIGHT_ARM_ACTION_INDICES = numpy.arange(20, 27, dtype=numpy.int64)
LEGACY_DEFAULT_ACTION_POSITION = DEFAULT_WHOLE_BODY_POSITION[LEGACY_ACTION_TO_WHOLE_BODY_INDEX]
LEGACY_ACTION_SCALE = numpy.array([
    0.12, 0.08, 0.08, 0.12, 0.10, 0.08,
    0.12, 0.08, 0.08, 0.12, 0.10, 0.08,
    0.25,
    0.35, 0.22, 0.28, 0.28, 0.18, 0.18, 0.18,
    0.35, 0.22, 0.28, 0.28, 0.18, 0.18, 0.18,
], dtype=numpy.float32)
LEGACY_ACTION_MIN = numpy.array([
    -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
    -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
    -2.6180,
    -2.9671, -0.5236, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
    -2.9671, -2.7925, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
], dtype=numpy.float32)
LEGACY_ACTION_MAX = numpy.array([
    2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
    2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
    2.6180,
    2.9671, 2.7925, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
    2.9671, 0.5236, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
], dtype=numpy.float32)


class ReachPolicyRunner:
    def __init__(self, args):
        self.args = args
        self.client = AuroraClient.get_instance(
            domain_id=args.domain_id,
            robot_name=args.robot_name,
        )
        time.sleep(1.0)

        self.policy_action = numpy.zeros(POLICY_NUM_ACTIONS, dtype=numpy.float32)
        self.commanded_action = numpy.zeros(POLICY_NUM_ACTIONS, dtype=numpy.float32)
        self.balance_policy_action = numpy.zeros(BALANCE_POLICY_NUM_ACTIONS, dtype=numpy.float32)
        self.balance_commanded_action = numpy.zeros(BALANCE_POLICY_NUM_ACTIONS, dtype=numpy.float32)
        self.obs_stack = None
        self.balance_obs_stack = None
        self.hold_whole_body_position = None
        self.startup_ramp_start_pose = None
        self.startup_ramp_goal_pose = None
        self.startup_ramp_goal_action = None
        self.startup_ramp_done = False
        self.control_start_time = None
        self.sampled_target_pos_base = numpy.array([args.target_x, args.target_y, args.target_z], dtype=numpy.float32)
        self.target_pos = self.sampled_target_pos_base.copy()
        self.target_world_locked = None
        self.target_rpy = numpy.array([args.target_roll, args.target_pitch, args.target_yaw], dtype=numpy.float32)
        self.active_arm = None
        self.arm_selector = numpy.zeros(2, dtype=numpy.float32)
        self.set_active_arm(args.arm)
        self.rng = numpy.random.default_rng(args.random_target_seed)
        self.last_target_sample_time = None
        self.target_log_file = None
        self.target_log_writer = None

        if args.target_log:
            target_log_path = os.path.abspath(os.path.expanduser(args.target_log))
            target_log_dir = os.path.dirname(target_log_path)
            if target_log_dir:
                os.makedirs(target_log_dir, exist_ok=True)
            self.target_log_file = open(target_log_path, "w", newline="")
            self.target_log_writer = csv.writer(self.target_log_file)
            self.target_log_writer.writerow([
                "time_s",
                "target_frame",
                "target_x",
                "target_y",
                "target_z",
                "ee_x",
                "ee_y",
                "ee_z",
                "error_m",
                "base_world_x",
                "base_world_y",
                "base_world_z",
                "base_quat_x",
                "base_quat_y",
                "base_quat_z",
                "base_quat_w",
                "target_world_x",
                "target_world_y",
                "target_world_z",
            ])

        policy_path = args.policy
        if policy_path is None:
            policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                DEFAULT_POLICY_FILE,
            )
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        self.policy_model = torch.jit.load(policy_path, map_location=torch.device("cpu"))
        self.policy_model.eval()
        self.configure_policy_contract()
        print(f"Loaded policy: {policy_path}")
        print(
            "Runtime settings: "
            f"target_frame_mode={args.target_frame_mode} "
            f"obs_reference={args.obs_reference} "
            f"action_reference={args.action_reference} "
            f"main_body_gain={args.main_body_gain:.3f} "
            f"head_gain={args.head_gain:.3f} "
            f"arm_gain={args.arm_gain:.3f} "
            f"startup_ramp_time={args.startup_ramp_time:.3f} "
            f"hold_first_action={args.startup_ramp_hold_first_action} "
            f"raw_action_clip_abs={args.raw_action_clip_abs:.1f}"
        )

        self.balance_policy_model = None
        if args.main_body_source == "balance":
            balance_policy_path = args.balance_policy
            if balance_policy_path is None:
                balance_policy_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    DEFAULT_BALANCE_POLICY_FILE,
                )
            if not os.path.exists(balance_policy_path):
                raise FileNotFoundError(f"Balance policy file not found: {balance_policy_path}")
            self.balance_policy_model = torch.jit.load(balance_policy_path, map_location=torch.device("cpu"))
            self.balance_policy_model.eval()
            print(f"Loaded balance policy: {balance_policy_path}")

    def configure_policy_contract(self):
        first_linear_in = None
        last_linear_out = None
        for _, param in self.policy_model.named_parameters():
            if param.ndim == 2:
                if first_linear_in is None:
                    first_linear_in = int(param.shape[1])
                last_linear_out = int(param.shape[0])

        if first_linear_in == LEGACY_OBS_LEN * STACK_SIZE and last_linear_out == LEGACY_POLICY_NUM_ACTIONS:
            self.policy_num_actions = LEGACY_POLICY_NUM_ACTIONS
            self.obs_len = LEGACY_OBS_LEN
            self.legacy_policy_contract = True
            self.action_to_whole_body_index = LEGACY_ACTION_TO_WHOLE_BODY_INDEX
            self.default_action_position = LEGACY_DEFAULT_ACTION_POSITION
            self.action_scale = LEGACY_ACTION_SCALE
            self.action_min = LEGACY_ACTION_MIN
            self.action_max = LEGACY_ACTION_MAX
            self.main_body_action_indices = LEGACY_MAIN_BODY_ACTION_INDICES
            self.leg_action_indices = LEGACY_LEG_ACTION_INDICES
            self.waist_action_indices = LEGACY_WAIST_ACTION_INDICES
            self.head_action_indices = numpy.array([], dtype=numpy.int64)
            self.arm_action_indices = LEGACY_ARM_ACTION_INDICES
            self.left_arm_action_indices = LEGACY_LEFT_ARM_ACTION_INDICES
            self.right_arm_action_indices = LEGACY_RIGHT_ARM_ACTION_INDICES
            print("Detected legacy reach contract: 99 obs x 5, 27 actions (waist + arms, no head actions).")
        elif first_linear_in == OBS_LEN * STACK_SIZE and last_linear_out == POLICY_NUM_ACTIONS:
            self.policy_num_actions = POLICY_NUM_ACTIONS
            self.obs_len = OBS_LEN
            self.legacy_policy_contract = False
            self.action_to_whole_body_index = ACTION_TO_WHOLE_BODY_INDEX
            self.default_action_position = DEFAULT_ACTION_POSITION
            self.action_scale = ACTION_SCALE
            self.action_min = ACTION_MIN
            self.action_max = ACTION_MAX
            self.main_body_action_indices = MAIN_BODY_ACTION_INDICES
            self.leg_action_indices = LEG_ACTION_INDICES
            self.waist_action_indices = WAIST_ACTION_INDICES
            self.head_action_indices = HEAD_ACTION_INDICES
            self.arm_action_indices = ARM_ACTION_INDICES
            self.left_arm_action_indices = LEFT_ARM_ACTION_INDICES
            self.right_arm_action_indices = RIGHT_ARM_ACTION_INDICES
        else:
            raise RuntimeError(
                "Unsupported reach policy contract: "
                f"first_linear_in={first_linear_in}, last_linear_out={last_linear_out}."
            )

        self.policy_action = numpy.zeros(self.policy_num_actions, dtype=numpy.float32)
        self.commanded_action = numpy.zeros(self.policy_num_actions, dtype=numpy.float32)

    def set_active_arm(self, arm):
        self.active_arm = arm
        self.arm_selector = numpy.array([1.0, 0.0] if arm == "left" else [0.0, 1.0], dtype=numpy.float32)

    def sample_reach_target(self):
        x = self.rng.uniform(*TARGET_X_RANGE)
        y_abs = self.rng.uniform(*TARGET_Y_ABS_RANGE)
        side = self.args.random_target_side
        if side == "both":
            arm = "left" if self.rng.random() < 0.5 else "right"
        elif side == "active":
            arm = self.args.arm
        else:
            arm = side

        y = y_abs if arm == "left" else -y_abs
        z = self.rng.uniform(*TARGET_Z_RANGE)
        self.sampled_target_pos_base = numpy.array([x, y, z], dtype=numpy.float32)
        self.target_pos = self.sampled_target_pos_base.copy()
        self.target_world_locked = None
        self.set_active_arm(arm)

        if self.args.fixed_target_rpy:
            self.target_rpy = numpy.array(
                [self.args.target_roll, self.args.target_pitch, self.args.target_yaw],
                dtype=numpy.float32,
            )
        else:
            self.target_rpy = numpy.array([
                self.rng.uniform(*TARGET_ROLL_RANGE),
                self.rng.uniform(*TARGET_PITCH_RANGE),
                self.rng.uniform(*TARGET_YAW_RANGE),
            ], dtype=numpy.float32)

        print(
            "Sampled reach target: "
            f"pos={self.sampled_target_pos_base[0]:+.3f},{self.sampled_target_pos_base[1]:+.3f},{self.sampled_target_pos_base[2]:+.3f} "
            f"rpy={self.target_rpy[0]:+.3f},{self.target_rpy[1]:+.3f},{self.target_rpy[2]:+.3f} "
            f"arm={self.active_arm}"
        )
        if self.args.target_frame_mode == "world":
            self.lock_target_world()

    def maybe_resample_target(self, now, force=False):
        if not self.args.random_targets:
            return

        if force or self.last_target_sample_time is None:
            self.sample_reach_target()
            self.last_target_sample_time = now
            return

        if self.args.target_resample_interval <= 0.0:
            return

        if now - self.last_target_sample_time >= self.args.target_resample_interval:
            self.sample_reach_target()
            self.last_target_sample_time = now

    def startup_ramp_alpha(self, now):
        if (
            self.args.startup_ramp_time <= 0.0
            or self.startup_ramp_start_pose is None
            or self.control_start_time is None
        ):
            return 1.0

        alpha = numpy.clip(
            (now - self.control_start_time) / self.args.startup_ramp_time,
            0.0,
            1.0,
        )
        if self.args.startup_ramp_smoothstep:
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return float(alpha)

    def action_from_whole_body_target(self, whole_body_target):
        action_target = whole_body_target[self.action_to_whole_body_index]
        effective_action = (action_target - self.default_action_position) / self.action_scale
        return numpy.clip(effective_action, -self.args.raw_action_clip_abs, self.args.raw_action_clip_abs).astype(numpy.float32)

    def set_pd(self):
        kp_config = {
            "left_leg": [180, 180, 120, 180, 40, 20],
            "right_leg": [180, 180, 120, 180, 40, 20],
            "waist": [40],
            "head": [20, 20],
            "left_manipulator": [90, 90, 60, 60, 40, 20, 20],
            "right_manipulator": [90, 90, 60, 60, 40, 20, 20],
        }
        kd_config = {
            "left_leg": [21, 10, 8, 21, 5, 2],
            "right_leg": [21, 10, 8, 21, 5, 2],
            "waist": [5],
            "head": [2.5, 2.5],
            "left_manipulator": [10, 10, 5, 5, 2.5, 2.5, 2.5],
            "right_manipulator": [10, 10, 5, 5, 2.5, 2.5, 2.5],
        }
        self.client.set_motor_cfg_pd(kp_config, kd_config)

    def read_joint_state(self):
        positions = []
        velocities = []
        for group_name in GROUP_NAMES:
            positions.append(self.client.get_group_state(group_name, "position"))
            velocities.append(self.client.get_group_state(group_name, "velocity"))

        q = numpy.concatenate(positions).astype(numpy.float32)
        qd = numpy.concatenate(velocities).astype(numpy.float32)
        if q.shape[0] != ROBOT_NUM_JOINTS or qd.shape[0] != ROBOT_NUM_JOINTS:
            raise RuntimeError(f"Expected 29 joints, got q={q.shape[0]}, qd={qd.shape[0]}")
        return q, qd

    def ensure_hold_pose(self):
        if self.hold_whole_body_position is None:
            q, _ = self.read_joint_state()
            self.hold_whole_body_position = q.copy()
            if self.legacy_policy_contract:
                offset = (self.hold_whole_body_position - DEFAULT_WHOLE_BODY_POSITION) * DOF_POS_OBS_SCALE
            else:
                offset = (
                    self.hold_whole_body_position[self.action_to_whole_body_index]
                    - self.default_action_position
                ) * OBS_DOF_POS_SCALE
            print(
                "Captured hold pose: "
                f"obs_offset_abs_max={numpy.max(numpy.abs(offset)):.3f} "
                f"q_abs_max={numpy.max(numpy.abs(self.hold_whole_body_position)):.3f}"
            )

    def capture_startup_ramp_pose(self):
        if self.args.startup_ramp_time <= 0.0:
            return

        q, _ = self.read_joint_state()
        self.startup_ramp_start_pose = q.copy()
        self.startup_ramp_goal_pose = None
        self.startup_ramp_goal_action = None
        self.startup_ramp_done = False
        print(
            "Captured policy startup pose: "
            f"q_abs_max={numpy.max(numpy.abs(self.startup_ramp_start_pose)):.3f}"
        )

    def reset_policy_history(self):
        self.policy_action.fill(0.0)
        self.commanded_action.fill(0.0)
        self.balance_policy_action.fill(0.0)
        self.balance_commanded_action.fill(0.0)
        self.obs_stack = None
        self.balance_obs_stack = None

    def prepare_default_pose(self):
        if self.args.default_pose_ramp_time <= 0.0 and self.args.default_pose_hold_time <= 0.0:
            return

        start_pose, _ = self.read_joint_state()
        target_pose = DEFAULT_WHOLE_BODY_POSITION.copy()
        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        start_time = next_tick
        last_print = start_time - self.args.print_period
        ramp_time = max(self.args.default_pose_ramp_time, 0.0)

        print(
            "Preparing Gym default pose: "
            f"ramp={self.args.default_pose_ramp_time:.2f}s "
            f"hold={self.args.default_pose_hold_time:.2f}s "
            f"initial_error_abs_max={numpy.max(numpy.abs(start_pose - target_pose)):.3f}"
        )

        if ramp_time > 0.0:
            while True:
                now = time.monotonic()
                alpha = numpy.clip((now - start_time) / ramp_time, 0.0, 1.0)
                if self.args.startup_ramp_smoothstep:
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                command_pose = (
                    start_pose * (1.0 - alpha)
                    + target_pose * alpha
                ).astype(numpy.float32)
                self.client.set_joint_positions({"whole_body": command_pose.astype(numpy.float64)})

                if now - last_print >= self.args.print_period:
                    last_print = now
                    q, _ = self.read_joint_state()
                    print(
                        "Preparing Gym default pose: "
                        f"alpha={alpha:.2f} "
                        f"error_abs_max={numpy.max(numpy.abs(q - target_pose)):.3f}"
                    )

                if alpha >= 1.0:
                    break

                next_tick += period
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()

        hold_end = time.monotonic() + max(self.args.default_pose_hold_time, 0.0)
        while time.monotonic() < hold_end:
            now = time.monotonic()
            self.client.set_joint_positions({"whole_body": target_pose.astype(numpy.float64)})
            if now - last_print >= self.args.print_period:
                last_print = now
                q, _ = self.read_joint_state()
                print(
                    "Holding Gym default pose: "
                    f"error_abs_max={numpy.max(numpy.abs(q - target_pose)):.3f}"
                )
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

        self.hold_whole_body_position = None
        self.reset_policy_history()
        print("Gym default pose preparation complete; reset policy history.")

    def hold_pose(self, pose, duration, label):
        if duration <= 0.0:
            return

        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        end_time = next_tick + duration
        last_print = next_tick - self.args.print_period
        pose = pose.astype(numpy.float32)

        print(f"{label}: streaming current pose for {duration:.2f}s")
        while time.monotonic() < end_time:
            now = time.monotonic()
            self.client.set_joint_positions({"whole_body": pose.astype(numpy.float64)})

            if now - last_print >= self.args.print_period:
                last_print = now
                q, _ = self.read_joint_state()
                print(
                    f"{label}: "
                    f"tracking_error_abs_max={numpy.max(numpy.abs(q - pose)):.3f}"
                )

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def observation_default_position(self):
        if self.args.hold_only and self.hold_whole_body_position is not None:
            return self.hold_whole_body_position[self.action_to_whole_body_index]
        if self.args.obs_reference == "hold" and self.hold_whole_body_position is not None:
            return self.hold_whole_body_position[self.action_to_whole_body_index]
        return self.default_action_position

    def action_default_position(self):
        if self.args.hold_only and self.hold_whole_body_position is not None:
            return self.hold_whole_body_position[self.action_to_whole_body_index]
        if self.args.action_reference == "hold" and self.hold_whole_body_position is not None:
            return self.hold_whole_body_position[self.action_to_whole_body_index]
        return self.default_action_position

    def active_manipulator_group(self):
        return "left_manipulator" if self.active_arm == "left" else "right_manipulator"

    def base_world_pose(self):
        base_pos = numpy.asarray(self.client.get_base_data("pos_W"), dtype=numpy.float32)
        base_quat = numpy.asarray(self.client.get_base_data("quat_xyzw"), dtype=numpy.float32)
        return base_pos, base_quat

    def lock_target_world(self):
        base_pos, base_quat = self.base_world_pose()
        torch_quat = torch.from_numpy(base_quat).float().unsqueeze(0)
        torch_target = torch.from_numpy(self.sampled_target_pos_base).float().unsqueeze(0)
        self.target_world_locked = (
            base_pos + torch_quat_rotate(torch_quat, torch_target).squeeze(0).numpy()
        ).astype(numpy.float32)
        self.target_pos = self.sampled_target_pos_base.copy()

    def update_target_base_frame(self):
        if self.args.target_frame_mode == "base":
            self.target_pos = self.sampled_target_pos_base.copy()
            return

        if self.target_world_locked is None:
            self.lock_target_world()

        base_pos, base_quat = self.base_world_pose()
        torch_quat = torch.from_numpy(base_quat).float().unsqueeze(0)
        torch_delta = torch.from_numpy(self.target_world_locked - base_pos).float().unsqueeze(0)
        self.target_pos = torch_quat_rotate_inverse(torch_quat, torch_delta).squeeze(0).numpy().astype(numpy.float32)

    def target_world_position(self):
        if self.args.target_frame_mode == "base":
            base_pos, base_quat = self.base_world_pose()
            torch_quat = torch.from_numpy(base_quat).float().unsqueeze(0)
            torch_target = torch.from_numpy(self.sampled_target_pos_base).float().unsqueeze(0)
            return (
                base_pos + torch_quat_rotate(torch_quat, torch_target).squeeze(0).numpy()
            ).astype(numpy.float32)

        if self.target_world_locked is None:
            self.lock_target_world()
        return self.target_world_locked.copy()

    def target_tracking_debug(self):
        try:
            self.update_target_base_frame()
            ee_pose = numpy.asarray(
                self.client.get_cartesian_state(self.active_manipulator_group(), "pose"),
                dtype=numpy.float32,
            )
            if ee_pose.shape[0] < 3:
                return None

            ee_pos = ee_pose[:3]
            target_pos = self.target_pos
            target_frame = "base"
            base_pos, base_quat = self.base_world_pose()
            target_world_pos = self.target_world_position()

            if self.args.ee_pose_frame == "world":
                target_pos = target_world_pos
                target_frame = "world"

            error_m = float(numpy.linalg.norm(ee_pos - target_pos))
            return {
                "target_frame": target_frame,
                "target": target_pos,
                "ee": ee_pos,
                "error_m": error_m,
                "base_pos": base_pos,
                "base_quat": base_quat,
                "target_world": target_world_pos,
            }
        except Exception as exc:
            if not hasattr(self, "_target_debug_warned"):
                self._target_debug_warned = True
                print(f"Target tracking debug unavailable: {exc}")
            return None

    def write_target_log(self, now, target_debug):
        if self.target_log_writer is None or target_debug is None:
            return

        target = target_debug["target"]
        ee = target_debug["ee"]
        base_pos = target_debug["base_pos"]
        base_quat = target_debug["base_quat"]
        target_world = target_debug["target_world"]
        self.target_log_writer.writerow([
            f"{now:.6f}",
            target_debug["target_frame"],
            f"{target[0]:.6f}",
            f"{target[1]:.6f}",
            f"{target[2]:.6f}",
            f"{ee[0]:.6f}",
            f"{ee[1]:.6f}",
            f"{ee[2]:.6f}",
            f"{target_debug['error_m']:.6f}",
            f"{base_pos[0]:.6f}",
            f"{base_pos[1]:.6f}",
            f"{base_pos[2]:.6f}",
            f"{base_quat[0]:.6f}",
            f"{base_quat[1]:.6f}",
            f"{base_quat[2]:.6f}",
            f"{base_quat[3]:.6f}",
            f"{target_world[0]:.6f}",
            f"{target_world[1]:.6f}",
            f"{target_world[2]:.6f}",
        ])
        self.target_log_file.flush()

    def target_debug_text(self, target_debug):
        if target_debug is None:
            return ""

        ee = target_debug["ee"]
        target_world = target_debug["target_world"]
        return (
            f"ee_{target_debug['target_frame']}="
            f"{ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f} "
            f"target_world={target_world[0]:+.3f},{target_world[1]:+.3f},{target_world[2]:+.3f} "
            f"target_err={target_debug['error_m']:.3f} "
        )

    def make_observation(self):
        self.ensure_hold_pose()
        self.update_target_base_frame()
        imu_quat = numpy.asarray(self.client.get_base_data("quat_xyzw"), dtype=numpy.float32)
        imu_angular_velocity = numpy.asarray(self.client.get_base_data("omega_B"), dtype=numpy.float32)
        q, qd = self.read_joint_state()

        torch_quat = torch.from_numpy(imu_quat).float().unsqueeze(0)
        torch_gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        projected_gravity = torch_quat_rotate_inverse(torch_quat, torch_gravity).squeeze(0).numpy()
        if self.args.zero_base_ang_vel:
            imu_angular_velocity = numpy.zeros_like(imu_angular_velocity)
        if self.args.zero_dof_vel:
            qd = numpy.zeros_like(qd)
        if self.legacy_policy_contract:
            obs_default = DEFAULT_WHOLE_BODY_POSITION
            if self.args.obs_reference == "hold" and self.hold_whole_body_position is not None:
                obs_default = self.hold_whole_body_position
            q_offset_obs = (q - obs_default) * OBS_DOF_POS_SCALE
            qd_obs = qd * OBS_DOF_VEL_SCALE
        else:
            q_policy_order = q[self.action_to_whole_body_index]
            qd_policy_order = qd[self.action_to_whole_body_index]
            q_offset_obs = (q_policy_order - self.observation_default_position()) * OBS_DOF_POS_SCALE
            qd_obs = qd_policy_order * OBS_DOF_VEL_SCALE

        obs = numpy.concatenate([
            self.target_pos,
            self.target_rpy,
            self.arm_selector,
            imu_angular_velocity,
            projected_gravity,
            q_offset_obs,
            qd_obs,
            self.policy_action,
        ]).astype(numpy.float32)

        if obs.shape[0] != self.obs_len:
            raise RuntimeError(f"Expected obs len {self.obs_len}, got {obs.shape[0]}")

        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        if self.obs_stack is None:
            self.obs_stack = torch.zeros((1, self.obs_len * STACK_SIZE), dtype=torch.float32)
            self.obs_stack[:, -self.obs_len:] = obs_t
        else:
            self.obs_stack = torch.cat([self.obs_stack[:, self.obs_len:], obs_t], dim=1).float()

        self.last_obs_debug = {
            "target": float(numpy.max(numpy.abs(self.target_pos))),
            "base_ang": float(numpy.max(numpy.abs(imu_angular_velocity))),
            "gravity": float(numpy.max(numpy.abs(projected_gravity))),
            "q_offset": float(numpy.max(numpy.abs(q_offset_obs))),
            "qd": float(numpy.max(numpy.abs(qd_obs))),
            "action_hist": float(numpy.max(numpy.abs(self.policy_action))),
            "obs": float(numpy.max(numpy.abs(obs))),
        }
        return self.obs_stack

    def make_balance_observation(self):
        imu_quat = numpy.asarray(self.client.get_base_data("quat_xyzw"), dtype=numpy.float32)
        imu_angular_velocity = numpy.asarray(self.client.get_base_data("omega_B"), dtype=numpy.float32)
        q, qd = self.read_joint_state()

        torch_quat = torch.from_numpy(imu_quat).float().unsqueeze(0)
        torch_gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        projected_gravity = torch_quat_rotate_inverse(torch_quat, torch_gravity).squeeze(0).numpy()

        obs = numpy.concatenate([
            numpy.zeros(3, dtype=numpy.float32),
            imu_angular_velocity,
            projected_gravity,
            q - DEFAULT_WHOLE_BODY_POSITION,
            qd * 0.1,
            self.balance_policy_action,
        ]).astype(numpy.float32)

        if obs.shape[0] != BALANCE_OBS_LEN:
            raise RuntimeError(f"Expected balance obs len {BALANCE_OBS_LEN}, got {obs.shape[0]}")

        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        if self.balance_obs_stack is None:
            self.balance_obs_stack = torch.cat([obs_t] * STACK_SIZE, dim=1).float()
        else:
            self.balance_obs_stack = torch.cat(
                [self.balance_obs_stack[:, BALANCE_OBS_LEN:], obs_t],
                dim=1,
            ).float()
        return self.balance_obs_stack

    def step_balance_policy(self):
        if self.balance_policy_model is None:
            return None

        obs_stack = self.make_balance_observation()
        with torch.no_grad():
            raw_action = self.balance_policy_model(obs_stack).detach().cpu().float().numpy().squeeze(0)

        if raw_action.shape[0] != BALANCE_POLICY_NUM_ACTIONS:
            raise RuntimeError(f"Expected {BALANCE_POLICY_NUM_ACTIONS} balance actions, got {raw_action.shape[0]}")

        raw_action = numpy.clip(raw_action, BALANCE_RAW_ACTION_CLIP_MIN, BALANCE_RAW_ACTION_CLIP_MAX)
        if self.args.balance_action_abs_limit > 0.0:
            raw_action = numpy.clip(
                raw_action,
                -self.args.balance_action_abs_limit,
                self.args.balance_action_abs_limit,
            )

        if self.args.balance_max_action_delta > 0.0:
            delta = numpy.clip(
                raw_action - self.balance_commanded_action,
                -self.args.balance_max_action_delta,
                self.args.balance_max_action_delta,
            )
            self.balance_commanded_action = (self.balance_commanded_action + delta).astype(numpy.float32)
        else:
            self.balance_commanded_action = raw_action.astype(numpy.float32)

        self.balance_policy_action = self.balance_commanded_action.copy()

        balance_target = DEFAULT_WHOLE_BODY_POSITION.copy()
        balance_action_position = DEFAULT_WHOLE_BODY_POSITION[BALANCE_ACTION_TO_WHOLE_BODY_INDEX] \
            + self.balance_policy_action * BALANCE_ACTION_SCALE
        balance_action_position = numpy.clip(
            balance_action_position,
            BALANCE_ACTION_MIN,
            BALANCE_ACTION_MAX,
        )
        balance_target[BALANCE_ACTION_TO_WHOLE_BODY_INDEX] = balance_action_position
        return balance_target

    def step_policy(self, send_command=True):
        now = time.monotonic()
        self.maybe_resample_target(now)
        obs_stack = self.make_observation()

        if self.args.hold_only:
            policy_raw_action = numpy.zeros(self.policy_num_actions, dtype=numpy.float32)
        else:
            with torch.no_grad():
                policy_raw_action = self.policy_model(obs_stack).detach().cpu().float().numpy().squeeze(0)

        if policy_raw_action.shape[0] != self.policy_num_actions:
            raise RuntimeError(f"Expected {self.policy_num_actions} actions, got {policy_raw_action.shape[0]}")

        policy_raw_abs_max = numpy.max(numpy.abs(policy_raw_action))
        raw_action = numpy.clip(
            policy_raw_action,
            -self.args.raw_action_clip_abs,
            self.args.raw_action_clip_abs,
        ).astype(numpy.float32)
        raw_action[self.main_body_action_indices] *= self.args.main_body_gain
        raw_action[self.head_action_indices] *= self.args.head_gain
        raw_action[self.arm_action_indices] *= self.args.arm_gain

        if self.args.freeze_main_body or not self.args.command_main_body:
            raw_action[self.main_body_action_indices] = 0.0
        if self.args.freeze_legs:
            raw_action[self.leg_action_indices] = 0.0
        if self.args.freeze_head:
            raw_action[self.head_action_indices] = 0.0
        if self.args.freeze_arms:
            raw_action[self.arm_action_indices] = 0.0
        if self.args.active_arm_only:
            if self.active_arm == "left":
                raw_action[self.right_arm_action_indices] = 0.0
            else:
                raw_action[self.left_arm_action_indices] = 0.0
        if self.args.action_abs_limit > 0.0:
            raw_action = numpy.clip(raw_action, -self.args.action_abs_limit, self.args.action_abs_limit)

        if self.args.max_action_delta > 0.0:
            delta = numpy.clip(
                raw_action - self.commanded_action,
                -self.args.max_action_delta,
                self.args.max_action_delta,
            )
            self.commanded_action = (self.commanded_action + delta).astype(numpy.float32)
        else:
            self.commanded_action = raw_action

        action_target = self.action_default_position() + self.commanded_action * self.action_scale
        action_target = numpy.clip(action_target, self.action_min, self.action_max)

        if self.args.action_reference == "hold":
            whole_body_target = self.hold_whole_body_position.copy()
        else:
            whole_body_target = DEFAULT_WHOLE_BODY_POSITION.copy()
        whole_body_target[self.action_to_whole_body_index] = action_target

        balance_target = None
        if self.args.main_body_source == "balance":
            balance_target = self.step_balance_policy()
            balance_blend = 1.0
            if self.args.balance_ramp_time > 0.0 and self.control_start_time is not None:
                balance_blend = numpy.clip(
                    (time.monotonic() - self.control_start_time) / self.args.balance_ramp_time,
                    0.0,
                    1.0,
                )
            if self.args.balance_scope == "full":
                balance_indices = BALANCE_ACTION_TO_WHOLE_BODY_INDEX
                whole_body_target[balance_indices] = \
                    whole_body_target[balance_indices] * (1.0 - balance_blend) \
                    + balance_target[balance_indices] * balance_blend
            elif self.args.balance_scope == "inactive_arm":
                inactive_arm_whole_body_indices = (
                    RIGHT_ARM_WHOLE_BODY_INDICES if self.args.arm == "left" else LEFT_ARM_WHOLE_BODY_INDICES
                )
                inactive_arm_indices = numpy.intersect1d(
                    BALANCE_ACTION_TO_WHOLE_BODY_INDEX,
                    inactive_arm_whole_body_indices,
                )
                balance_indices = numpy.concatenate((
                    numpy.arange(0, 13, dtype=numpy.int64),
                    inactive_arm_indices,
                ))
                whole_body_target[balance_indices] = \
                    whole_body_target[balance_indices] * (1.0 - balance_blend) \
                    + balance_target[balance_indices] * balance_blend
            else:
                whole_body_target[0:13] = \
                    whole_body_target[0:13] * (1.0 - balance_blend) + balance_target[0:13] * balance_blend

        if not self.args.command_main_body and self.args.main_body_source != "balance":
            whole_body_target[MAIN_BODY_WHOLE_BODY_INDICES] = \
                self.hold_whole_body_position[MAIN_BODY_WHOLE_BODY_INDICES]
        if self.args.freeze_main_body:
            whole_body_target[MAIN_BODY_WHOLE_BODY_INDICES] = \
                self.hold_whole_body_position[MAIN_BODY_WHOLE_BODY_INDICES]
        if self.args.freeze_legs:
            whole_body_target[LEG_WHOLE_BODY_INDICES] = \
                self.hold_whole_body_position[LEG_WHOLE_BODY_INDICES]
        if self.args.freeze_head:
            whole_body_target[HEAD_WHOLE_BODY_INDICES] = \
                self.hold_whole_body_position[HEAD_WHOLE_BODY_INDICES]
        if self.args.freeze_arms:
            whole_body_target[ARM_WHOLE_BODY_INDICES] = \
                self.hold_whole_body_position[ARM_WHOLE_BODY_INDICES]
        elif self.args.active_arm_only:
            inactive_arm_whole_body_indices = (
                RIGHT_ARM_WHOLE_BODY_INDICES if self.active_arm == "left" else LEFT_ARM_WHOLE_BODY_INDICES
            )
            whole_body_target[inactive_arm_whole_body_indices] = \
                self.hold_whole_body_position[inactive_arm_whole_body_indices]

        startup_ramp_alpha = self.startup_ramp_alpha(now)
        if startup_ramp_alpha < 1.0 and self.startup_ramp_start_pose is not None:
            if self.startup_ramp_goal_pose is None:
                self.startup_ramp_goal_pose = whole_body_target.copy()
                self.startup_ramp_goal_action = self.commanded_action.copy()
                print(
                    "Startup ramp target captured: "
                    f"duration={self.args.startup_ramp_time:.2f}s "
                    f"goal_cmd_abs_max={numpy.max(numpy.abs(self.startup_ramp_goal_action)):.3f}"
                )

            if self.args.startup_ramp_hold_first_action:
                whole_body_target = self.startup_ramp_goal_pose.copy()
                self.commanded_action = self.startup_ramp_goal_action.copy()
                ramp_action = self.startup_ramp_goal_action
            else:
                ramp_action = self.commanded_action

            whole_body_target = (
                self.startup_ramp_start_pose * (1.0 - startup_ramp_alpha)
                + whole_body_target * startup_ramp_alpha
            ).astype(numpy.float32)
            self.policy_action = (
                ramp_action * startup_ramp_alpha
            ).astype(numpy.float32)
        else:
            if (
                self.args.startup_ramp_time > 0.0
                and self.startup_ramp_start_pose is not None
                and not self.startup_ramp_done
            ):
                self.startup_ramp_done = True
                print("Startup ramp complete; releasing normal reach policy control.")
            self.policy_action = self.commanded_action.copy()

        if send_command:
            self.client.set_joint_positions({"whole_body": whole_body_target.astype(numpy.float64)})

        if not hasattr(self, "_last_print") or now - self._last_print > self.args.print_period:
            self._last_print = now
            target_debug = self.target_tracking_debug() if self.args.print_target_error else None
            self.write_target_log(now, target_debug)
            print(
                f"target={self.target_pos[0]:+.3f},{self.target_pos[1]:+.3f},{self.target_pos[2]:+.3f} "
                f"rpy={self.target_rpy[0]:+.3f},{self.target_rpy[1]:+.3f},{self.target_rpy[2]:+.3f} "
                f"arm={self.active_arm} "
                f"{self.target_debug_text(target_debug)}"
                f"policy_raw_abs_max={policy_raw_abs_max:.3f} "
                f"safe_raw_abs_max={numpy.max(numpy.abs(raw_action)):.3f} "
                f"cmd_abs_max={numpy.max(numpy.abs(self.commanded_action)):.3f} "
                f"startup_ramp={startup_ramp_alpha:.2f} "
                f"main_abs_max={numpy.max(numpy.abs(self.commanded_action[self.main_body_action_indices])):.3f} "
                f"waist_abs_max={numpy.max(numpy.abs(self.commanded_action[self.waist_action_indices])):.3f} "
                f"balance_abs_max={numpy.max(numpy.abs(self.balance_policy_action)):.3f} "
                f"head_abs_max={numpy.max(numpy.abs(self.commanded_action[self.head_action_indices])) if self.head_action_indices.size else 0.0:.3f} "
                f"arm_abs_max={numpy.max(numpy.abs(self.commanded_action[self.arm_action_indices])):.3f} "
                f"obs={self.last_obs_debug}"
            )
        return whole_body_target

    def visualize_only_loop(self):
        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        print(
            f"Visualizing reach target at {self.args.rate:.1f} Hz "
            f"in FSM state {self.args.stand_fsm_state}. Press Ctrl-C to stop."
        )

        while True:
            now = time.monotonic()
            self.maybe_resample_target(now)
            self.update_target_base_frame()
            target_debug = self.target_tracking_debug() if self.args.print_target_error else None
            self.write_target_log(now, target_debug)
            if not hasattr(self, "_last_print") or now - self._last_print > self.args.print_period:
                self._last_print = now
                print(
                    f"target={self.target_pos[0]:+.3f},{self.target_pos[1]:+.3f},{self.target_pos[2]:+.3f} "
                    f"rpy={self.target_rpy[0]:+.3f},{self.target_rpy[1]:+.3f},{self.target_rpy[2]:+.3f} "
                    f"arm={self.active_arm} "
                    f"{self.target_debug_text(target_debug)}"
                )

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def run(self):
        input(f"Press Enter to switch to FSM state {self.args.stand_fsm_state} (stand)")
        self.client.set_fsm_state(self.args.stand_fsm_state)
        time.sleep(1.0)
        self.maybe_resample_target(time.monotonic(), force=True)
        if self.args.target_frame_mode == "world":
            self.lock_target_world()

        if self.args.visualize_only:
            self.visualize_only_loop()
            return

        input("When the robot is stable, press Enter to switch to FSM state 10 (UserCmd)")
        handoff_pose, _ = self.read_joint_state()
        print(
            "Captured pre-UserCmd handoff pose: "
            f"q_abs_max={numpy.max(numpy.abs(handoff_pose)):.3f} "
            f"default_error_abs_max={numpy.max(numpy.abs(handoff_pose - DEFAULT_WHOLE_BODY_POSITION)):.3f}"
        )
        self.client.set_fsm_state(10)
        self.set_pd()
        self.hold_pose(handoff_pose, self.args.handoff_hold_time, "UserCmd handoff")
        self.prepare_default_pose()
        self.capture_startup_ramp_pose()

        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        self.control_start_time = next_tick
        if self.args.random_targets:
            self.last_target_sample_time = next_tick
        print(f"Running reach policy at {self.args.rate:.1f} Hz. Press Ctrl-C to stop.")

        while True:
            self.step_policy()
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def close(self):
        try:
            self.client.set_fsm_state(2)
            time.sleep(0.1)
        except Exception:
            pass
        try:
            if self.target_log_file is not None:
                self.target_log_file.close()
        except Exception:
            pass
        self.client.close()


def torch_quat_rotate(q, v):
    q_w = q[:, -1:]
    q_vec = q[:, :3]
    q_vec_dot_v = torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1)
    q_vec_cross_v = torch.cross(q_vec, v, dim=-1)
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = q_vec_cross_v * q_w * 2.0
    c = q_vec * q_vec_dot_v * 2.0
    return a + b + c


def torch_quat_rotate_inverse(q, v):
    q_w = q[:, -1:]
    q_vec = q[:, :3]
    q_vec_dot_v = torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1)
    q_vec_cross_v = torch.cross(q_vec, v, dim=-1)
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = q_vec_cross_v * q_w * 2.0
    c = q_vec * q_vec_dot_v * 2.0
    return a - b + c


def parse_args():
    parser = argparse.ArgumentParser(description="Run the latest GR2 reach policy through Aurora UserCmd.")
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", default="gr2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--balance-policy", default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--target-x", type=float, default=0.28)
    parser.add_argument("--target-y", type=float, default=-0.18)
    parser.add_argument("--target-z", type=float, default=0.15)
    parser.add_argument("--target-roll", type=float, default=0.0)
    parser.add_argument("--target-pitch", type=float, default=0.0)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--main-body-gain", type=float, default=1.0)
    parser.add_argument("--head-gain", type=float, default=1.0)
    parser.add_argument("--arm-gain", type=float, default=1.0)
    parser.add_argument("--raw-action-clip-abs", type=float, default=RAW_ACTION_CLIP_ABS)
    parser.add_argument("--action-abs-limit", type=float, default=0.0)
    parser.add_argument("--max-action-delta", type=float, default=0.03)
    parser.add_argument("--hold-only", action="store_true")
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--random-targets", action="store_true")
    parser.add_argument("--random-target-side", choices=["active", "left", "right", "both"], default="active")
    parser.add_argument("--target-resample-interval", type=float, default=1.5)
    parser.add_argument("--random-target-seed", type=int, default=None)
    parser.add_argument("--fixed-target-rpy", action="store_true")
    parser.add_argument("--target-frame-mode", choices=["base", "world"], default="base")
    parser.add_argument("--stand-fsm-state", type=int, default=2)
    parser.add_argument("--handoff-hold-time", type=float, default=0.5)
    parser.add_argument("--default-pose-ramp-time", type=float, default=2.0)
    parser.add_argument("--default-pose-hold-time", type=float, default=0.25)
    parser.add_argument("--startup-ramp-time", type=float, default=2.0)
    parser.add_argument("--startup-ramp-hold-first-action", action="store_true", default=True)
    parser.add_argument("--no-startup-ramp-hold-first-action", dest="startup_ramp_hold_first_action", action="store_false")
    parser.add_argument("--startup-ramp-smoothstep", action="store_true", default=True)
    parser.add_argument("--linear-startup-ramp", dest="startup_ramp_smoothstep", action="store_false")
    parser.add_argument("--obs-reference", choices=["default", "hold"], default="default")
    parser.add_argument("--action-reference", choices=["default", "hold"], default="default")
    parser.add_argument("--main-body-source", choices=["reach", "balance"], default="reach")
    parser.add_argument("--balance-scope", choices=["main_body", "inactive_arm", "full"], default="main_body")
    parser.add_argument("--active-arm-only", action="store_true")
    parser.add_argument("--balance-ramp-time", type=float, default=2.0)
    parser.add_argument("--balance-action-abs-limit", type=float, default=0.0)
    parser.add_argument("--balance-max-action-delta", type=float, default=0.03)
    parser.add_argument("--command-main-body", dest="command_main_body", action="store_true", default=True)
    parser.add_argument("--hold-main-body", dest="command_main_body", action="store_false")
    parser.add_argument("--freeze-main-body", action="store_true")
    parser.add_argument("--freeze-legs", action="store_true")
    parser.add_argument("--freeze-head", action="store_true")
    parser.add_argument("--freeze-arms", action="store_true")
    parser.add_argument("--zero-base-ang-vel", action="store_true")
    parser.add_argument("--zero-dof-vel", action="store_true")
    parser.add_argument("--print-target-error", dest="print_target_error", action="store_true", default=True)
    parser.add_argument("--no-print-target-error", dest="print_target_error", action="store_false")
    parser.add_argument("--ee-pose-frame", choices=["base", "world"], default="base")
    parser.add_argument("--target-log", default=None)
    parser.add_argument("--print-period", type=float, default=1.0)
    return parser.parse_args()


def main():
    runner = None
    try:
        runner = ReachPolicyRunner(parse_args())
        runner.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
