#!/usr/bin/env python3

import argparse
import os
import struct
import sys
import threading
import time

import numpy
import torch
from fourier_aurora_client import AuroraClient


ROBOT_NUM_JOINTS = 29
POLICY_NUM_ACTIONS = 21
OBS_LEN = 88
STACK_SIZE = 5

GROUP_NAMES = [
    "left_leg",
    "right_leg",
    "waist",
    "head",
    "left_manipulator",
    "right_manipulator",
]

ACTION_TO_WHOLE_BODY_INDEX = numpy.array([
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18,
    22, 23, 24, 25,
], dtype=numpy.int64)

DEFAULT_WHOLE_BODY_POSITION = numpy.array([
    -0.1309, 0.0, 0.0, 0.2618, -0.1309, 0.0,
    -0.1309, 0.0, 0.0, 0.2618, -0.1309, 0.0,
    0.0,
    0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=numpy.float32)

DEFAULT_ACTION_POSITION = DEFAULT_WHOLE_BODY_POSITION[ACTION_TO_WHOLE_BODY_INDEX]

ACTION_SCALE = numpy.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.75,
    0.20, 0.06, 0.06, 0.10,
    0.20, 0.06, 0.06, 0.10,
], dtype=numpy.float32)

ACTION_MIN = numpy.array([
    -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
    -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
    -2.6180,
    -2.9671, -0.5236, -1.8326, -1.5272,
    -2.9671, -2.7925, -1.8326, -1.5272,
], dtype=numpy.float32)

ACTION_MAX = numpy.array([
    2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
    2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
    2.6180,
    2.9671, 2.7925, 1.8326, 0.4800,
    2.9671, 0.5236, 1.8326, 0.4800,
], dtype=numpy.float32)

RAW_ACTION_CLIP_MIN = ACTION_MIN - 1.0
RAW_ACTION_CLIP_MAX = ACTION_MAX + 1.0


class UpperBodyPolicyRunner:
    def __init__(self, args):
        self.args = args
        self.client = AuroraClient.get_instance(
            domain_id=args.domain_id,
            robot_name=args.robot_name,
        )
        time.sleep(1.0)

        self.policy_action = numpy.zeros(POLICY_NUM_ACTIONS, dtype=numpy.float32)
        self.obs_stack = None
        self.stop_event = threading.Event()
        self.axis_left = (0.0, 0.0)
        self.axis_right = (0.0, 0.0)
        self.commands_filtered = numpy.array([args.vx, args.vy, args.yaw], dtype=numpy.float32)
        self.pygame = None
        self.joystick = None
        self.joystick_device = None
        self.joystick_thread = None

        policy_path = args.policy
        if policy_path is None:
            policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "policy_gr2_upper_body_jit.pt",
            )
        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"Policy file not found: {policy_path}\n"
                "Copy policy_jit.pt to policy_gr2_upper_body_jit.pt first."
            )
        self.policy_model = torch.jit.load(policy_path, map_location=torch.device("cpu"))
        self.policy_model.eval()
        print(f"Loaded policy: {policy_path}")

    def setup_joystick(self):
        if not self.args.joystick:
            return

        try:
            import pygame

            self.pygame = pygame
            self.pygame.init()
            self.pygame.joystick.init()

            joystick_count = self.pygame.joystick.get_count()
            if joystick_count > 0:
                print(f"Detected {joystick_count} joystick(s) through pygame")
                self.joystick = self.pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_thread = threading.Thread(target=self.pygame_joystick_listener, daemon=True)
                self.joystick_thread.start()
                return
        except Exception as exc:
            print(f"pygame joystick unavailable, trying Linux joystick device: {exc}")

        if not os.path.exists(self.args.joystick_device):
            raise RuntimeError(
                f"No joystick detected. Install pygame or pass a valid --joystick-device. "
                f"Missing: {self.args.joystick_device}"
            )

        print(f"Reading joystick directly from {self.args.joystick_device}")
        self.joystick_device = open(self.args.joystick_device, "rb", buffering=0)
        self.joystick_thread = threading.Thread(target=self.linux_joystick_listener, daemon=True)
        self.joystick_thread.start()

    def pygame_joystick_listener(self):
        while not self.stop_event.is_set():
            self.pygame.event.get()
            self.axis_left = self.joystick.get_axis(0), self.joystick.get_axis(1)
            self.axis_right = self.joystick.get_axis(3), 0.0
            time.sleep(0.02)

    def linux_joystick_listener(self):
        axes = {}
        while not self.stop_event.is_set():
            event = self.joystick_device.read(8)
            if len(event) != 8:
                time.sleep(0.01)
                continue

            _, value, event_type, number = struct.unpack("IhBB", event)
            if event_type & 0x02:
                axes[number] = max(-1.0, min(1.0, value / 32767.0))
                self.axis_left = axes.get(0, 0.0), axes.get(1, 0.0)
                self.axis_right = axes.get(3, 0.0), 0.0

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

    def update_command(self):
        if self.args.joystick:
            commands_norm = numpy.array([
                -self.axis_left[1],
                -self.axis_left[0],
                -self.axis_right[0],
            ], dtype=numpy.float32)
            commands = numpy.array([
                commands_norm[0] * self.args.max_vx,
                commands_norm[1] * self.args.max_vy,
                commands_norm[2] * self.args.max_yaw,
            ], dtype=numpy.float32)
        else:
            commands = numpy.array([self.args.vx, self.args.vy, self.args.yaw], dtype=numpy.float32)

        alpha = numpy.array([self.args.filter_x, self.args.filter_y, self.args.filter_yaw], dtype=numpy.float32)
        self.commands_filtered = self.commands_filtered * alpha + commands * (1.0 - alpha)
        self.commands_filtered[0] = numpy.clip(self.commands_filtered[0], -self.args.max_vx, self.args.max_vx)
        self.commands_filtered[1] = numpy.clip(self.commands_filtered[1], -self.args.max_vy, self.args.max_vy)
        self.commands_filtered[2] = numpy.clip(self.commands_filtered[2], -self.args.max_yaw, self.args.max_yaw)

    def make_observation(self):
        imu_quat = numpy.asarray(self.client.get_base_data("quat_xyzw"), dtype=numpy.float32)
        imu_angular_velocity = numpy.asarray(self.client.get_base_data("omega_B"), dtype=numpy.float32)
        q, qd = self.read_joint_state()

        torch_quat = torch.from_numpy(imu_quat).float().unsqueeze(0)
        torch_gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        projected_gravity = torch_quat_rotate_inverse(torch_quat, torch_gravity).squeeze(0).numpy()

        obs = numpy.concatenate([
            self.commands_filtered,
            imu_angular_velocity,
            projected_gravity,
            q - DEFAULT_WHOLE_BODY_POSITION,
            qd * 0.1,
            self.policy_action,
        ]).astype(numpy.float32)

        if obs.shape[0] != OBS_LEN:
            raise RuntimeError(f"Expected obs len {OBS_LEN}, got {obs.shape[0]}")

        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        if self.obs_stack is None:
            self.obs_stack = torch.cat([obs_t] * STACK_SIZE, dim=1).float()
        else:
            self.obs_stack = torch.cat([self.obs_stack[:, OBS_LEN:], obs_t], dim=1).float()
        return self.obs_stack

    def step_policy(self):
        self.update_command()
        obs_stack = self.make_observation()

        with torch.no_grad():
            raw_action = self.policy_model(obs_stack).detach().cpu().float().numpy().squeeze(0)

        if raw_action.shape[0] != POLICY_NUM_ACTIONS:
            raise RuntimeError(f"Expected {POLICY_NUM_ACTIONS} actions, got {raw_action.shape[0]}")

        raw_action = numpy.clip(raw_action, RAW_ACTION_CLIP_MIN, RAW_ACTION_CLIP_MAX)
        self.policy_action = raw_action.astype(numpy.float32)

        action_target = DEFAULT_ACTION_POSITION + self.policy_action * ACTION_SCALE
        action_target = numpy.clip(action_target, ACTION_MIN, ACTION_MAX)

        whole_body_target = DEFAULT_WHOLE_BODY_POSITION.copy()
        whole_body_target[ACTION_TO_WHOLE_BODY_INDEX] = action_target

        self.client.set_joint_positions({"whole_body": whole_body_target.astype(numpy.float64)})

        now = time.monotonic()
        if not hasattr(self, "_last_print") or now - self._last_print > self.args.print_period:
            self._last_print = now
            print(
                "cmd="
                f"{self.commands_filtered[0]:+.3f},"
                f"{self.commands_filtered[1]:+.3f},"
                f"{self.commands_filtered[2]:+.3f} "
                f"action_abs_max={numpy.max(numpy.abs(self.policy_action)):.3f}"
            )

    def run(self):
        self.setup_joystick()

        input("Press Enter to switch to FSM state 2 (PD stand)")
        self.client.set_fsm_state(2)
        time.sleep(1.0)

        input("When the robot is stable, press Enter to switch to FSM state 10 (UserCmd)")
        self.client.set_fsm_state(10)
        time.sleep(0.5)
        self.set_pd()

        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        print(f"Running upper-body policy at {self.args.rate:.1f} Hz. Press Ctrl-C to stop.")

        while not self.stop_event.is_set():
            self.step_policy()
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def close(self):
        self.stop_event.set()
        if self.joystick_thread is not None:
            self.joystick_thread.join(timeout=1.0)
        try:
            self.client.set_joint_positions({"whole_body": DEFAULT_WHOLE_BODY_POSITION.astype(numpy.float64)})
            time.sleep(0.1)
            self.client.set_fsm_state(2)
        except Exception:
            pass
        try:
            if self.pygame is not None:
                self.pygame.quit()
        except Exception:
            pass
        try:
            if self.joystick_device is not None:
                self.joystick_device.close()
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
    parser = argparse.ArgumentParser(description="Run the GR2 21-action upper-body walking policy through Aurora UserCmd.")
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", default="gr2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--joystick", action="store_true")
    parser.add_argument("--joystick-device", default="/dev/input/js0")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--max-vx", type=float, default=0.35)
    parser.add_argument("--max-vy", type=float, default=0.08)
    parser.add_argument("--max-yaw", type=float, default=0.70)
    parser.add_argument("--filter-x", type=float, default=0.92)
    parser.add_argument("--filter-y", type=float, default=0.40)
    parser.add_argument("--filter-yaw", type=float, default=0.00)
    parser.add_argument("--print-period", type=float, default=1.0)
    return parser.parse_args()


def main():
    runner = None
    try:
        runner = UpperBodyPolicyRunner(parse_args())
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