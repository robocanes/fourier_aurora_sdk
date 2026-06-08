#!/usr/bin/env python3

import argparse
import csv
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
ARM_ACTION_INDICES = numpy.array([13, 14, 15, 16, 17, 18, 19, 20], dtype=numpy.int64)

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
        self.log_file = None
        self.log_writer = None
        self.log_rows_since_flush = 0
        self.latest_q = None
        self.latest_qd = None
        self.latest_imu_quat = None
        self.latest_imu_angular_velocity = None
        self.latest_projected_gravity = None
        self.entry_start_time = None
        self.entry_action_start = DEFAULT_ACTION_POSITION.copy()
        self.sequence_start_time = None
        self.arm_swing_phase = 0.0
        self.last_policy_time = None
        self.action_scale = ACTION_SCALE.copy()
        self.action_scale[[4, 10]] = args.ankle_pitch_scale
        self.action_scale[[5, 11]] = args.ankle_roll_scale

        policy_path = args.policy
        if policy_path is None:
            policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "policy_gr2_dynamic_walk_model13999_jit.pt",
            )
        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"Policy file not found: {policy_path}\n"
                "Copy/export the proven model_13999 JIT policy to "
                "policy_gr2_dynamic_walk_model13999_jit.pt first."
            )
        self.policy_model = torch.jit.load(policy_path, map_location=torch.device("cpu"))
        self.policy_model.eval()
        print(f"Loaded policy: {policy_path}")
        self.setup_logging(policy_path)

    def setup_logging(self, policy_path):
        if self.args.no_log:
            return

        log_path = self.args.log_path
        if log_path is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            if self.args.log_folder:
                log_dir = os.path.join(log_dir, self.args.log_folder, "logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            policy_name = os.path.splitext(os.path.basename(policy_path))[0]
            log_path = os.path.join(log_dir, f"dynamic_walk_{policy_name}_{timestamp}.csv")
        else:
            log_dir = os.path.dirname(os.path.abspath(log_path))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        self.log_file = open(log_path, "w", newline="")
        self.log_writer = csv.writer(self.log_file)

        header = [
            "wall_time_s",
            "monotonic_s",
            "step_elapsed_s",
            "loop_overrun_s",
            "policy_path",
            "rate_hz",
            "max_vx",
            "max_vy",
            "max_yaw",
            "ankle_pitch_scale",
            "ankle_roll_scale",
            "filter_x",
            "filter_y",
            "filter_yaw",
            "entry_ramp_s",
            "entry_hold_s",
            "free_arms",
            "arm_shoulder_pitch",
            "arm_shoulder_roll",
            "arm_shoulder_yaw",
            "arm_elbow_pitch",
            "arm_swing_amplitude",
            "arm_swing_elbow_amplitude",
            "arm_swing_frequency",
            "arm_swing_vx_scale",
            "joystick_axis_left_x",
            "joystick_axis_left_y",
            "joystick_axis_right_x",
            "joystick_axis_right_y",
            "cmd_x",
            "cmd_y",
            "cmd_yaw",
            "action_abs_max",
        ]
        header += [f"imu_quat_{axis}" for axis in ("x", "y", "z", "w")]
        header += [f"imu_omega_{axis}" for axis in ("x", "y", "z")]
        header += [f"projected_gravity_{axis}" for axis in ("x", "y", "z")]
        header += [f"raw_action_{i:02d}" for i in range(POLICY_NUM_ACTIONS)]
        header += [f"target_action_joint_{i:02d}" for i in range(POLICY_NUM_ACTIONS)]
        header += [f"q_{i:02d}" for i in range(ROBOT_NUM_JOINTS)]
        header += [f"qd_{i:02d}" for i in range(ROBOT_NUM_JOINTS)]
        self.log_writer.writerow(header)
        self.log_file.flush()
        self.policy_path = policy_path
        print(f"Logging telemetry to: {log_path}")

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
        elif self.sequence_start_time is not None:
            commands = self.sequence_command()
        else:
            commands = numpy.array([self.args.vx, self.args.vy, self.args.yaw], dtype=numpy.float32)

        alpha = numpy.array([self.args.filter_x, self.args.filter_y, self.args.filter_yaw], dtype=numpy.float32)
        self.commands_filtered = self.commands_filtered * alpha + commands * (1.0 - alpha)
        self.commands_filtered[0] = numpy.clip(self.commands_filtered[0], -self.args.max_vx, self.args.max_vx)
        self.commands_filtered[1] = numpy.clip(self.commands_filtered[1], -self.args.max_vy, self.args.max_vy)
        self.commands_filtered[2] = numpy.clip(self.commands_filtered[2], -self.args.max_yaw, self.args.max_yaw)

    def sequence_command(self):
        distance = max(0.0, self.args.out_back_distance)
        forward_vx = max(1.0e-6, abs(self.args.out_vx))
        backward_vx = -max(1.0e-6, abs(self.args.back_vx))
        forward_duration = distance / forward_vx
        stop_duration = max(0.0, self.args.sequence_stop_s)
        backward_duration = distance / abs(backward_vx)

        elapsed = time.monotonic() - self.sequence_start_time
        if elapsed < forward_duration:
            return numpy.array([forward_vx, 0.0, 0.0], dtype=numpy.float32)
        if elapsed < forward_duration + stop_duration:
            return numpy.zeros(3, dtype=numpy.float32)
        if elapsed < forward_duration + stop_duration + backward_duration:
            return numpy.array([backward_vx, 0.0, 0.0], dtype=numpy.float32)

        if self.args.sequence_exit:
            self.stop_event.set()
        return numpy.zeros(3, dtype=numpy.float32)

    def entry_ramp(self):
        if self.entry_start_time is None:
            return 1.0
        duration = max(0.0, self.args.entry_ramp_s)
        if duration <= 0.0:
            return 1.0

        phase = numpy.clip((time.monotonic() - self.entry_start_time) / duration, 0.0, 1.0)
        return float(phase * phase * (3.0 - 2.0 * phase))

    def reset_policy_state_from_robot(self):
        q, _ = self.read_joint_state()
        action_q = numpy.clip(q[ACTION_TO_WHOLE_BODY_INDEX], ACTION_MIN, ACTION_MAX)
        self.entry_action_start = action_q.astype(numpy.float32)
        self.policy_action = numpy.clip(
            (self.entry_action_start - DEFAULT_ACTION_POSITION) / self.action_scale,
            RAW_ACTION_CLIP_MIN,
            RAW_ACTION_CLIP_MAX,
        ).astype(numpy.float32)
        self.commands_filtered[:] = 0.0
        self.obs_stack = None
        self.entry_start_time = time.monotonic()
        self.arm_swing_phase = 0.0
        self.last_policy_time = None

    def make_observation(self):
        imu_quat = numpy.asarray(self.client.get_base_data("quat_xyzw"), dtype=numpy.float32)
        imu_angular_velocity = numpy.asarray(self.client.get_base_data("omega_B"), dtype=numpy.float32)
        q, qd = self.read_joint_state()

        torch_quat = torch.from_numpy(imu_quat).float().unsqueeze(0)
        torch_gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        projected_gravity = torch_quat_rotate_inverse(torch_quat, torch_gravity).squeeze(0).numpy()
        self.latest_q = q
        self.latest_qd = qd
        self.latest_imu_quat = imu_quat
        self.latest_imu_angular_velocity = imu_angular_velocity
        self.latest_projected_gravity = projected_gravity

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

    def write_telemetry(self, step_elapsed_s, loop_overrun_s, action_target):
        if self.log_writer is None:
            return

        row = [
            time.time(),
            time.monotonic(),
            step_elapsed_s,
            loop_overrun_s,
            self.policy_path,
            self.args.rate,
            self.args.max_vx,
            self.args.max_vy,
            self.args.max_yaw,
            self.args.ankle_pitch_scale,
            self.args.ankle_roll_scale,
            self.args.filter_x,
            self.args.filter_y,
            self.args.filter_yaw,
            self.args.entry_ramp_s,
            self.args.entry_hold_s,
            int(self.args.free_arms),
            self.args.arm_shoulder_pitch,
            self.args.arm_shoulder_roll,
            self.args.arm_shoulder_yaw,
            self.args.arm_elbow_pitch,
            self.args.arm_swing_amplitude,
            self.args.arm_swing_elbow_amplitude,
            self.args.arm_swing_frequency,
            self.args.arm_swing_vx_scale,
            self.axis_left[0],
            self.axis_left[1],
            self.axis_right[0],
            self.axis_right[1],
            *self.commands_filtered.tolist(),
            float(numpy.max(numpy.abs(self.policy_action))),
            *self.latest_imu_quat.tolist(),
            *self.latest_imu_angular_velocity.tolist(),
            *self.latest_projected_gravity.tolist(),
            *self.policy_action.tolist(),
            *action_target.tolist(),
            *self.latest_q.tolist(),
            *self.latest_qd.tolist(),
        ]
        self.log_writer.writerow(row)
        self.log_rows_since_flush += 1
        if self.log_rows_since_flush >= self.args.log_flush_rows:
            self.log_file.flush()
            self.log_rows_since_flush = 0

    def step_policy(self, loop_overrun_s=0.0):
        step_start = time.monotonic()
        self.update_command()
        self.update_arm_swing_phase(step_start)
        ramp = self.entry_ramp()
        if ramp < 1.0:
            self.commands_filtered *= ramp
        obs_stack = self.make_observation()

        with torch.no_grad():
            raw_action = self.policy_model(obs_stack).detach().cpu().float().numpy().squeeze(0)

        if raw_action.shape[0] != POLICY_NUM_ACTIONS:
            raise RuntimeError(f"Expected {POLICY_NUM_ACTIONS} actions, got {raw_action.shape[0]}")

        raw_action = numpy.clip(raw_action, RAW_ACTION_CLIP_MIN, RAW_ACTION_CLIP_MAX)
        policy_action_target = DEFAULT_ACTION_POSITION + raw_action.astype(numpy.float32) * self.action_scale
        policy_action_target = numpy.clip(policy_action_target, ACTION_MIN, ACTION_MAX)
        action_target = self.entry_action_start * (1.0 - ramp) + policy_action_target * ramp
        if not self.args.free_arms:
            arm_hang_target = self.arm_hang_action_position()
            action_target[ARM_ACTION_INDICES] = (
                self.entry_action_start[ARM_ACTION_INDICES] * (1.0 - ramp)
                + arm_hang_target[ARM_ACTION_INDICES] * ramp
            )
        action_target = numpy.clip(action_target, ACTION_MIN, ACTION_MAX).astype(numpy.float32)
        self.policy_action = numpy.clip(
            (action_target - DEFAULT_ACTION_POSITION) / self.action_scale,
            RAW_ACTION_CLIP_MIN,
            RAW_ACTION_CLIP_MAX,
        ).astype(numpy.float32)

        whole_body_target = DEFAULT_WHOLE_BODY_POSITION.copy()
        whole_body_target[ACTION_TO_WHOLE_BODY_INDEX] = action_target

        self.client.set_joint_positions({"whole_body": whole_body_target.astype(numpy.float64)})
        self.write_telemetry(time.monotonic() - step_start, loop_overrun_s, action_target)

        now = time.monotonic()
        if not hasattr(self, "_last_print") or now - self._last_print > self.args.print_period:
            self._last_print = now
            print(
                "cmd="
                f"{self.commands_filtered[0]:+.3f},"
                f"{self.commands_filtered[1]:+.3f},"
                f"{self.commands_filtered[2]:+.3f} "
                f"ramp={ramp:.2f} "
                f"action_abs_max={numpy.max(numpy.abs(self.policy_action)):.3f}"
            )

    def arm_hang_action_position(self):
        action_position = DEFAULT_ACTION_POSITION.copy()
        action_position[13] = self.args.arm_shoulder_pitch
        action_position[14] = self.args.arm_shoulder_roll
        action_position[15] = self.args.arm_shoulder_yaw
        action_position[16] = self.args.arm_elbow_pitch
        action_position[17] = self.args.arm_shoulder_pitch
        action_position[18] = -self.args.arm_shoulder_roll
        action_position[19] = -self.args.arm_shoulder_yaw
        action_position[20] = self.args.arm_elbow_pitch
        swing_scale = self.arm_swing_command_scale()
        if self.args.arm_swing_amplitude > 0.0 and swing_scale > 0.0:
            swing = self.args.arm_swing_amplitude * swing_scale * numpy.sin(self.arm_swing_phase)
            elbow_swing = self.args.arm_swing_elbow_amplitude * swing_scale * (
                0.5 + 0.5 * numpy.sin(self.arm_swing_phase + numpy.pi)
            )
            if self.commands_filtered[0] < 0.0:
                swing = -swing
            action_position[13] += swing
            action_position[17] -= swing
            action_position[16] += elbow_swing
            action_position[20] += elbow_swing
        return action_position.astype(numpy.float32)

    def arm_swing_command_scale(self):
        vx_scale = max(1.0e-6, self.args.arm_swing_vx_scale)
        forward_scale = abs(float(self.commands_filtered[0])) / vx_scale
        yaw_scale = 0.0
        if abs(self.args.max_yaw) > 1.0e-6:
            yaw_scale = 0.35 * abs(float(self.commands_filtered[2])) / abs(self.args.max_yaw)
        return float(numpy.clip(max(forward_scale, yaw_scale), 0.0, 1.0))

    def update_arm_swing_phase(self, now):
        if self.last_policy_time is None:
            self.last_policy_time = now
            return
        dt = max(0.0, min(0.1, now - self.last_policy_time))
        self.last_policy_time = now
        swing_scale = self.arm_swing_command_scale()
        if swing_scale <= 1.0e-4:
            return
        self.arm_swing_phase = (
            self.arm_swing_phase
            + 2.0 * numpy.pi * self.args.arm_swing_frequency * swing_scale * dt
        ) % (2.0 * numpy.pi)

    def run(self):
        self.setup_joystick()

        input("Press Enter to switch to FSM state 2 (PD stand)")
        self.client.set_fsm_state(2)
        time.sleep(1.0)

        input("When the robot is stable, press Enter to switch to FSM state 10 (UserCmd)")
        self.reset_policy_state_from_robot()
        self.client.set_fsm_state(10)
        time.sleep(self.args.entry_hold_s)
        self.set_pd()
        if self.args.out_back_distance > 0.0:
            self.sequence_start_time = time.monotonic()
            print(
                "Running out/back sequence: "
                f"+{self.args.out_back_distance:.2f}m at {abs(self.args.out_vx):.2f}m/s, "
                f"stop {self.args.sequence_stop_s:.1f}s, "
                f"-{self.args.out_back_distance:.2f}m at {abs(self.args.back_vx):.2f}m/s"
            )

        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        print(f"Running dynamic-walk policy at {self.args.rate:.1f} Hz. Press Ctrl-C to stop.")

        while not self.stop_event.is_set():
            loop_overrun_s = max(0.0, time.monotonic() - next_tick)
            self.step_policy(loop_overrun_s)
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
        try:
            if self.log_file is not None:
                self.log_file.flush()
                self.log_file.close()
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
    parser = argparse.ArgumentParser(description="Run the GR2 21-action dynamic-walk policy through Aurora UserCmd.")
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", default="gr2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--joystick", action="store_true")
    parser.add_argument("--joystick-device", default="/dev/input/js0")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--out-back-distance", type=float, default=0.0)
    parser.add_argument("--out-vx", type=float, default=0.25)
    parser.add_argument("--back-vx", type=float, default=-0.15)
    parser.add_argument("--sequence-stop-s", type=float, default=1.5)
    parser.add_argument("--sequence-exit", action="store_true")
    parser.add_argument("--max-vx", type=float, default=0.35)
    parser.add_argument("--max-vy", type=float, default=0.08)
    parser.add_argument("--max-yaw", type=float, default=0.35)
    parser.add_argument("--ankle-pitch-scale", type=float, default=1.0)
    parser.add_argument("--ankle-roll-scale", type=float, default=1.0)
    parser.add_argument("--filter-x", type=float, default=0.92)
    parser.add_argument("--filter-y", type=float, default=0.40)
    parser.add_argument("--filter-yaw", type=float, default=0.00)
    parser.add_argument("--print-period", type=float, default=1.0)
    parser.add_argument("--entry-ramp-s", type=float, default=1.0)
    parser.add_argument("--entry-hold-s", type=float, default=0.1)
    parser.add_argument("--free-arms", action="store_true", help="Use policy arm actions instead of holding a neutral hanging arm pose.")
    parser.add_argument("--arm-shoulder-pitch", type=float, default=0.0)
    parser.add_argument("--arm-shoulder-roll", type=float, default=0.0)
    parser.add_argument("--arm-shoulder-yaw", type=float, default=0.0)
    parser.add_argument("--arm-elbow-pitch", type=float, default=0.0)
    parser.add_argument("--arm-swing-amplitude", type=float, default=0.0)
    parser.add_argument("--arm-swing-elbow-amplitude", type=float, default=0.0)
    parser.add_argument("--arm-swing-frequency", type=float, default=1.35)
    parser.add_argument("--arm-swing-vx-scale", type=float, default=0.35)
    parser.add_argument("--log-path", default=None)
    parser.add_argument(
        "--log-folder",
        default=None,
        help="Optional subfolder under logs/<name>/logs for separating physical-gr2 and simulation-gr2 runs.",
    )
    parser.add_argument("--log-flush-rows", type=int, default=50)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    for name in ("filter_x", "filter_y", "filter_yaw"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0.0, 1.0); got {value}")

    for name in ("ankle_pitch_scale", "ankle_roll_scale"):
        value = getattr(args, name)
        if value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive; got {value}")

    for name in ("arm_swing_amplitude", "arm_swing_elbow_amplitude"):
        value = getattr(args, name)
        if value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative; got {value}")

    for name in ("arm_swing_frequency", "arm_swing_vx_scale"):
        value = getattr(args, name)
        if value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive; got {value}")

    if args.out_back_distance < 0.0:
        parser.error(f"--out-back-distance must be non-negative; got {args.out_back_distance}")
    if args.out_back_distance > 0.0:
        if abs(args.out_vx) <= 1.0e-6:
            parser.error("--out-vx must be non-zero when --out-back-distance is used")
        if abs(args.back_vx) <= 1.0e-6:
            parser.error("--back-vx must be non-zero when --out-back-distance is used")

    return args


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
