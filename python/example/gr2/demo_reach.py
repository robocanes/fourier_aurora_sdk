#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy
import torch
from fourier_aurora_client import AuroraClient


ROBOT_NUM_JOINTS = 29
POLICY_NUM_ACTIONS = 27
OBS_LEN = 99
STACK_SIZE = 5

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

ACTION_TO_WHOLE_BODY_INDEX = numpy.array([
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
], dtype=numpy.int64)

MAIN_BODY_ACTION_SLICE = slice(0, 13)
ARM_ACTION_SLICE = slice(13, 27)

DEFAULT_ACTION_POSITION = DEFAULT_WHOLE_BODY_POSITION[ACTION_TO_WHOLE_BODY_INDEX]

ACTION_SCALE = numpy.array([
    0.12, 0.08, 0.08, 0.12, 0.10, 0.08,
    0.12, 0.08, 0.08, 0.12, 0.10, 0.08,
    0.25,
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

ACTION_MIN = numpy.array([
    -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
    -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
    -2.6180,
    -2.9671, -0.5236, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
    -2.9671, -2.7925, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
], dtype=numpy.float32)

ACTION_MAX = numpy.array([
    2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
    2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
    2.6180,
    2.9671, 2.7925, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
    2.9671, 0.5236, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
], dtype=numpy.float32)

RAW_ACTION_CLIP_MIN = ACTION_MIN - 1.0
RAW_ACTION_CLIP_MAX = ACTION_MAX + 1.0


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
        self.obs_stack = None
        self.hold_whole_body_position = None
        self.target_pos = numpy.array([args.target_x, args.target_y, args.target_z], dtype=numpy.float32)
        self.target_rpy = numpy.array([args.target_roll, args.target_pitch, args.target_yaw], dtype=numpy.float32)
        self.arm_selector = numpy.array([1.0, 0.0] if args.arm == "left" else [0.0, 1.0], dtype=numpy.float32)

        policy_path = args.policy
        if policy_path is None:
            policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "policy_gr2_reach_model7498_jit.pt",
            )
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        self.policy_model = torch.jit.load(policy_path, map_location=torch.device("cpu"))
        self.policy_model.eval()
        print(f"Loaded policy: {policy_path}")

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
            offset = (self.hold_whole_body_position - DEFAULT_WHOLE_BODY_POSITION) * DOF_POS_OBS_SCALE
            print(
                "Captured hold pose: "
                f"obs_offset_abs_max={numpy.max(numpy.abs(offset)):.3f} "
                f"q_abs_max={numpy.max(numpy.abs(self.hold_whole_body_position)):.3f}"
            )

    def observation_default_position(self):
        if self.hold_whole_body_position is not None and not self.args.command_main_body:
            return self.hold_whole_body_position
        return DEFAULT_WHOLE_BODY_POSITION

    def action_default_position(self):
        if self.hold_whole_body_position is not None and not self.args.command_main_body:
            return self.hold_whole_body_position[ACTION_TO_WHOLE_BODY_INDEX]
        return DEFAULT_ACTION_POSITION

    def make_observation(self):
        self.ensure_hold_pose()
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
        q_offset_obs = (q - self.observation_default_position()) * DOF_POS_OBS_SCALE
        qd_obs = qd * DOF_VEL_OBS_SCALE

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

        if obs.shape[0] != OBS_LEN:
            raise RuntimeError(f"Expected obs len {OBS_LEN}, got {obs.shape[0]}")

        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        if self.obs_stack is None:
            self.obs_stack = torch.cat([obs_t] * STACK_SIZE, dim=1).float()
        else:
            self.obs_stack = torch.cat([self.obs_stack[:, OBS_LEN:], obs_t], dim=1).float()

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

    def step_policy(self):
        obs_stack = self.make_observation()

        with torch.no_grad():
            policy_raw_action = self.policy_model(obs_stack).detach().cpu().float().numpy().squeeze(0)

        if policy_raw_action.shape[0] != POLICY_NUM_ACTIONS:
            raise RuntimeError(f"Expected {POLICY_NUM_ACTIONS} actions, got {policy_raw_action.shape[0]}")

        policy_raw_abs_max = numpy.max(numpy.abs(policy_raw_action))
        raw_action = numpy.clip(policy_raw_action, RAW_ACTION_CLIP_MIN, RAW_ACTION_CLIP_MAX).astype(numpy.float32)
        raw_action[MAIN_BODY_ACTION_SLICE] *= self.args.main_body_gain
        raw_action[ARM_ACTION_SLICE] *= self.args.arm_gain

        if self.args.freeze_main_body or not self.args.command_main_body:
            raw_action[MAIN_BODY_ACTION_SLICE] = 0.0
        if self.args.freeze_arms:
            raw_action[ARM_ACTION_SLICE] = 0.0
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

        self.policy_action = self.commanded_action.copy()

        action_target = self.action_default_position() + self.commanded_action * ACTION_SCALE
        action_target = numpy.clip(action_target, ACTION_MIN, ACTION_MAX)

        whole_body_target = self.hold_whole_body_position.copy()
        if self.args.command_main_body:
            whole_body_target[:] = DEFAULT_WHOLE_BODY_POSITION
        whole_body_target[ACTION_TO_WHOLE_BODY_INDEX] = action_target
        if not self.args.command_main_body:
            whole_body_target[ACTION_TO_WHOLE_BODY_INDEX[MAIN_BODY_ACTION_SLICE]] = \
                self.hold_whole_body_position[ACTION_TO_WHOLE_BODY_INDEX[MAIN_BODY_ACTION_SLICE]]
        self.client.set_joint_positions({"whole_body": whole_body_target.astype(numpy.float64)})

        now = time.monotonic()
        if not hasattr(self, "_last_print") or now - self._last_print > self.args.print_period:
            self._last_print = now
            print(
                f"target={self.target_pos[0]:+.3f},{self.target_pos[1]:+.3f},{self.target_pos[2]:+.3f} "
                f"arm={self.args.arm} "
                f"policy_raw_abs_max={policy_raw_abs_max:.3f} "
                f"safe_raw_abs_max={numpy.max(numpy.abs(raw_action)):.3f} "
                f"cmd_abs_max={numpy.max(numpy.abs(self.commanded_action)):.3f} "
                f"main_abs_max={numpy.max(numpy.abs(self.commanded_action[MAIN_BODY_ACTION_SLICE])):.3f} "
                f"arm_abs_max={numpy.max(numpy.abs(self.commanded_action[ARM_ACTION_SLICE])):.3f} "
                f"obs={self.last_obs_debug}"
            )

    def run(self):
        input("Press Enter to switch to FSM state 2 (PD stand)")
        self.client.set_fsm_state(2)
        time.sleep(1.0)

        input("When the robot is stable, press Enter to switch to FSM state 10 (UserCmd)")
        self.client.set_fsm_state(10)
        time.sleep(0.5)
        self.set_pd()

        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
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
            self.client.set_joint_positions({"whole_body": DEFAULT_WHOLE_BODY_POSITION.astype(numpy.float64)})
            time.sleep(0.1)
            self.client.set_fsm_state(2)
        except Exception:
            pass
        self.client.close()


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
    parser = argparse.ArgumentParser(description="Run the GR2 model_7498 reach policy through Aurora UserCmd.")
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", default="gr2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--target-x", type=float, default=0.28)
    parser.add_argument("--target-y", type=float, default=-0.18)
    parser.add_argument("--target-z", type=float, default=0.15)
    parser.add_argument("--target-roll", type=float, default=0.0)
    parser.add_argument("--target-pitch", type=float, default=0.0)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--main-body-gain", type=float, default=0.35)
    parser.add_argument("--arm-gain", type=float, default=0.50)
    parser.add_argument("--action-abs-limit", type=float, default=0.60)
    parser.add_argument("--max-action-delta", type=float, default=0.03)
    parser.add_argument("--command-main-body", action="store_true")
    parser.add_argument("--freeze-main-body", action="store_true")
    parser.add_argument("--freeze-arms", action="store_true")
    parser.add_argument("--zero-base-ang-vel", action="store_true")
    parser.add_argument("--zero-dof-vel", action="store_true")
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
