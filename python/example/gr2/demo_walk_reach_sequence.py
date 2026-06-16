#!/usr/bin/env python3

import argparse
import os
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy

import demo_reach
import demo_walk_dynamic


def smoothstep(alpha):
    alpha = float(numpy.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def default_walk_args(args):
    return SimpleNamespace(
        domain_id=args.domain_id,
        robot_name=args.robot_name,
        policy=args.walk_policy,
        rate=args.rate,
        vx=0.0,
        vy=0.0,
        yaw=0.0,
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_yaw=args.max_yaw,
        ankle_pitch_scale=args.ankle_pitch_scale,
        ankle_roll_scale=args.ankle_roll_scale,
        filter_x=args.filter_x,
        filter_y=args.filter_y,
        filter_yaw=args.filter_yaw,
        entry_ramp_s=args.walk_entry_ramp_s,
        entry_hold_s=args.entry_hold_s,
        stand_fsm_state=args.stand_fsm_state,
        usercmd_fsm_state=args.usercmd_fsm_state,
        exit_fsm_state=args.exit_fsm_state,
        start_paused=False,
        free_arms=args.walk_free_arms,
        arm_shoulder_pitch=args.arm_shoulder_pitch,
        arm_shoulder_roll=args.arm_shoulder_roll,
        arm_shoulder_yaw=args.arm_shoulder_yaw,
        arm_elbow_pitch=args.arm_elbow_pitch,
        arm_swing_amplitude=args.arm_swing_amplitude,
        arm_swing_elbow_amplitude=args.arm_swing_elbow_amplitude,
        arm_swing_frequency=args.arm_swing_frequency,
        arm_swing_vx_scale=args.arm_swing_vx_scale,
        joystick=args.joystick,
        joystick_device=args.joystick_device,
        joystick_deadzone=args.joystick_deadzone,
        joystick_forward_axis=args.joystick_forward_axis,
        joystick_lateral_axis=args.joystick_lateral_axis,
        joystick_yaw_axis=args.joystick_yaw_axis,
        log_path=None,
        log_folder=None,
        no_log=True,
        log_flush_rows=50,
        print_period=args.print_period,
        out_back_distance=0.0,
        out_vx=0.18,
        back_vx=0.12,
        sequence_stop_s=1.0,
        sequence_exit=False,
        straight_turn_distance=0.0,
        straight_turn_vx=0.18,
        straight_turn_yaw=0.35,
        straight_turn_angle_deg=90.0,
        skip_start_check=args.skip_start_check,
        min_start_gravity_z=args.min_start_gravity_z,
        max_start_tilt_xy=args.max_start_tilt_xy,
        max_start_qd=args.max_start_qd,
    )


def default_reach_args(args):
    return SimpleNamespace(
        domain_id=args.domain_id,
        robot_name=args.robot_name,
        policy=args.reach_policy,
        balance_policy=args.balance_policy,
        rate=args.rate,
        arm=args.arm,
        target_x=args.target_x,
        target_y=args.target_y,
        target_z=args.target_z,
        target_roll=args.target_roll,
        target_pitch=args.target_pitch,
        target_yaw=args.target_yaw,
        main_body_gain=args.reach_main_body_gain,
        head_gain=args.reach_head_gain,
        arm_gain=args.reach_arm_gain,
        raw_action_clip_abs=args.reach_raw_action_clip_abs,
        action_abs_limit=args.reach_action_abs_limit,
        max_action_delta=args.reach_max_action_delta,
        hold_only=False,
        visualize_only=False,
        random_targets=args.random_targets,
        random_target_side=args.random_target_side,
        target_resample_interval=args.target_resample_interval,
        random_target_seed=args.random_target_seed,
        fixed_target_rpy=args.fixed_target_rpy,
        target_frame_mode=args.target_frame_mode,
        stand_fsm_state=args.stand_fsm_state,
        handoff_hold_time=args.handoff_hold_time,
        default_pose_ramp_time=0.0,
        default_pose_hold_time=0.0,
        startup_ramp_time=args.reach_startup_ramp_s,
        startup_ramp_hold_first_action=True,
        startup_ramp_smoothstep=True,
        obs_reference=args.reach_obs_reference,
        action_reference=args.reach_action_reference,
        main_body_source=args.reach_main_body_source,
        balance_scope=args.reach_balance_scope,
        active_arm_only=args.active_arm_only,
        balance_ramp_time=args.balance_ramp_time,
        balance_action_abs_limit=args.balance_action_abs_limit,
        balance_max_action_delta=args.balance_max_action_delta,
        command_main_body=args.reach_command_main_body,
        freeze_main_body=False,
        freeze_legs=args.reach_freeze_legs,
        freeze_head=args.reach_freeze_head,
        freeze_arms=False,
        zero_base_ang_vel=False,
        zero_dof_vel=False,
        print_target_error=args.print_target_error,
        ee_pose_frame=args.ee_pose_frame,
        target_log=args.target_log,
        dry_run=args.reach_dry_run,
        print_period=args.print_period,
    )


class WalkReachSequence:
    def __init__(self, args):
        self.args = args
        self.walk = demo_walk_dynamic.UpperBodyPolicyRunner(default_walk_args(args))
        self.reach = None
        self.client = self.walk.client
        self.mode = args.initial_mode
        self.hold_pose_target = None
        self.last_command_print = 0.0
        self.last_overlay_print = 0.0
        self.last_overlay_arm_target = None
        self.last_overlay_counter_arm_target = None
        self.last_overlay_head_target = None
        self.last_overlay_waist_target = None
        self.last_reach_walk_target = None
        self.next_reach_arm = args.arm
        self.next_reach_target_profile = 0
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.command_thread = None

    def read_commands(self):
        while not self.stop_event.is_set():
            line = sys.stdin.readline()
            if line == "":
                time.sleep(0.05)
                continue
            self.command_queue.put(line.strip().lower())

    def read_joint_state(self):
        return self.walk.read_joint_state()

    def ensure_reach_runner(self):
        if self.reach is None:
            print("Loading reach runner...")
            self.reach = demo_reach.ReachPolicyRunner(default_reach_args(self.args))
            self.client = self.walk.client
        return self.reach

    def reset_overlay_targets_from_pose(self, q):
        reach = self.ensure_reach_runner()
        active_arm_indices = (
            demo_reach.LEFT_ARM_WHOLE_BODY_INDICES
            if reach.active_arm == "left"
            else demo_reach.RIGHT_ARM_WHOLE_BODY_INDICES
        )
        overlay_arm_indices = (
            demo_reach.ARM_WHOLE_BODY_INDICES
            if self.args.reach_overlay_arms == "both"
            else active_arm_indices
        )
        self.last_overlay_arm_target = q[overlay_arm_indices].copy()
        inactive_arm_indices = (
            demo_reach.RIGHT_ARM_WHOLE_BODY_INDICES
            if reach.active_arm == "left"
            else demo_reach.LEFT_ARM_WHOLE_BODY_INDICES
        )
        self.last_overlay_counter_arm_target = q[inactive_arm_indices].copy()
        self.last_overlay_head_target = q[demo_reach.HEAD_WHOLE_BODY_INDICES].copy()
        self.last_overlay_waist_target = q[demo_reach.WAIST_ACTION_INDICES].copy()

    def hold_pose(self, pose, duration, label):
        if duration <= 0.0:
            return
        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        end_time = next_tick + duration
        print(f"{label}: holding measured pose for {duration:.2f}s")
        while time.monotonic() < end_time:
            self.client.set_joint_positions({"whole_body": pose.astype(numpy.float64)})
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def reset_walk_from_robot(self):
        self.walk.reset_policy_state_from_robot()
        self.walk.commands_filtered[:] = 0.0
        self.walk.axis_left = (0.0, 0.0)
        self.walk.axis_right = (0.0, 0.0)

    def active_walk_hold(self, duration, label):
        if duration <= 0.0:
            return
        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        end_time = next_tick + duration
        joystick_enabled = self.walk.args.joystick
        vx, vy, yaw = self.walk.args.vx, self.walk.args.vy, self.walk.args.yaw
        print(f"{label}: active zero-command walk hold for {duration:.2f}s")
        try:
            self.walk.args.joystick = False
            self.walk.args.vx = 0.0
            self.walk.args.vy = 0.0
            self.walk.args.yaw = 0.0
            while time.monotonic() < end_time:
                self.walk.step_policy(0.0)
                next_tick += period
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()
        finally:
            self.walk.args.joystick = joystick_enabled
            self.walk.args.vx = vx
            self.walk.args.vy = vy
            self.walk.args.yaw = yaw

    def reset_reach_from_robot(self):
        reach = self.ensure_reach_runner()
        reach.reset_policy_history()
        q, _ = self.read_joint_state()
        self.last_overlay_arm_target = None
        self.last_overlay_counter_arm_target = None
        self.last_overlay_head_target = None
        self.last_overlay_waist_target = None
        self.last_reach_walk_target = None
        reach.hold_whole_body_position = q.copy()
        reach.startup_ramp_start_pose = q.copy()
        reach.startup_ramp_goal_pose = None
        reach.startup_ramp_goal_action = None
        reach.startup_ramp_done = False
        reach.control_start_time = time.monotonic()
        reach.maybe_resample_target(reach.control_start_time, force=True)
        if self.args.alternate_reach_side:
            self.set_reach_target_side(self.next_reach_arm, "selected reach target")
            self.next_reach_arm = "left" if self.next_reach_arm == "right" else "right"
        self.reset_overlay_targets_from_pose(q)
        if reach.args.target_frame_mode == "world":
            reach.lock_target_world()
        print(
            "Reach handoff captured: "
            f"hold_q_abs_max={numpy.max(numpy.abs(q)):.3f} "
            f"target={reach.target_pos[0]:+.3f},"
            f"{reach.target_pos[1]:+.3f},"
            f"{reach.target_pos[2]:+.3f}"
        )

    def reset_reach_ramp_from_robot(self, label):
        reach = self.ensure_reach_runner()
        reach.reset_policy_history()
        q, _ = self.read_joint_state()
        self.last_overlay_arm_target = None
        self.last_overlay_counter_arm_target = None
        self.last_overlay_head_target = None
        self.last_overlay_waist_target = None
        self.last_reach_walk_target = None
        reach.hold_whole_body_position = q.copy()
        reach.startup_ramp_start_pose = q.copy()
        reach.startup_ramp_goal_pose = None
        reach.startup_ramp_goal_action = None
        reach.startup_ramp_done = False
        reach.control_start_time = time.monotonic()
        self.reset_overlay_targets_from_pose(q)
        print(f"{label}: reset reach history and startup ramp")

    def apply_reach_target_profile(self, target):
        if not self.args.cycle_reach_targets:
            return target
        if self.args.reach_target_mode == "high_sweep":
            profiles = (
                (self.args.reach_target_near_x, self.args.reach_target_high_z),
                (self.args.reach_target_high_near_x, self.args.reach_target_higher_z),
                (self.args.reach_target_highest_near_x, self.args.reach_target_highest_z),
                (self.args.reach_target_extreme_near_x, self.args.reach_target_extreme_z),
            )
        else:
            profiles = (
                (self.args.target_x, self.args.target_z),
                (self.args.reach_target_near_x, self.args.reach_target_high_z),
                (self.args.reach_target_near_x, self.args.target_z),
                (self.args.reach_target_far_x, self.args.reach_target_low_z),
            )
        x, z = profiles[self.next_reach_target_profile % len(profiles)]
        self.next_reach_target_profile += 1
        target = target.copy()
        target[0] = x
        target[2] = z
        return target

    def set_reach_target_side(self, arm, label, advance_profile=True):
        reach = self.ensure_reach_runner()
        y_abs = abs(float(reach.sampled_target_pos_base[1]))
        if y_abs < 1.0e-6:
            y_abs = abs(float(self.args.target_y))
        y_abs = max(y_abs, 1.0e-6)
        target = reach.sampled_target_pos_base.copy()
        if advance_profile:
            target = self.apply_reach_target_profile(target)
        target[1] = y_abs if arm == "left" else -y_abs
        reach.sampled_target_pos_base = target.astype(numpy.float32)
        reach.target_pos = reach.sampled_target_pos_base.copy()
        reach.target_world_locked = None
        reach.set_active_arm(arm)
        print(
            f"{label}: arm={arm} "
            f"target={reach.target_pos[0]:+.3f},"
            f"{reach.target_pos[1]:+.3f},"
            f"{reach.target_pos[2]:+.3f}"
        )

    def flip_reach_target_side(self):
        reach = self.ensure_reach_runner()
        next_arm = "left" if reach.active_arm == "right" else "right"
        if self.args.reach_flip_settle_s > 0.0:
            q, _ = self.read_joint_state()
            self.hold_pose(q, self.args.reach_flip_settle_s, "target flip settle")
        self.set_reach_target_side(next_arm, "flipped reach target")
        self.next_reach_arm = "left" if next_arm == "right" else "right"
        self.reset_reach_ramp_from_robot("flipped target")

    def switch_mode(self, mode):
        if mode == self.mode:
            return
        self.last_overlay_arm_target = None
        self.last_overlay_counter_arm_target = None
        self.last_overlay_head_target = None
        self.last_overlay_waist_target = None
        self.last_reach_walk_target = None
        old_mode = self.mode
        q, _ = self.read_joint_state()
        self.hold_pose_target = q.copy()
        if mode == "walk" and old_mode == "reach":
            print(f"{old_mode} -> {mode}: continuing active walk policy")
        else:
            self.active_walk_hold(self.args.handoff_hold_time, f"{old_mode} -> {mode}")
        if mode == "walk":
            if old_mode != "reach":
                self.reset_walk_from_robot()
        elif mode == "reach":
            self.reset_reach_from_robot()
        elif mode != "hold":
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        print(f"Active mode: {self.mode}")

    def handle_keyboard(self):
        try:
            command = self.command_queue.get_nowait()
        except queue.Empty:
            return True
        if command in ("q", "quit", "exit"):
            print("Command received: quit")
            return False
        if command in ("w", "walk"):
            print("Command received: walk")
            self.switch_mode("walk")
        elif command in ("r", "reach"):
            print("Command received: reach")
            if self.mode == "reach":
                self.flip_reach_target_side()
            else:
                self.switch_mode("reach")
        elif command in ("h", "hold"):
            print("Command received: hold")
            self.switch_mode("hold")
        elif command in ("t", "target"):
            print("Command received: new target")
            if self.mode == "reach":
                self.ensure_reach_runner().sample_reach_target()
                self.reset_reach_ramp_from_robot("new target")
            else:
                print("new target ignored until reach mode is active")
        elif command:
            print("Commands: w=walk, r=reach, h=hold, t=new target, q=quit")
        return True

    def step_hold(self):
        self.step_zero_command_walk(send_command=True)
        now = time.monotonic()
        if now - self.last_command_print >= self.args.print_period:
            self.last_command_print = now
            print("hold: streaming zero-command walk policy")

    def step_zero_command_walk(self, loop_overrun_s=0.0, send_command=True):
        joystick_enabled = self.walk.args.joystick
        vx, vy, yaw = self.walk.args.vx, self.walk.args.vy, self.walk.args.yaw
        try:
            self.walk.args.joystick = False
            self.walk.args.vx = 0.0
            self.walk.args.vy = 0.0
            self.walk.args.yaw = 0.0
            return self.walk.step_policy(loop_overrun_s, send_command=send_command)
        finally:
            self.walk.args.joystick = joystick_enabled
            self.walk.args.vx = vx
            self.walk.args.vy = vy
            self.walk.args.yaw = yaw

    def step_reach_overlay(self, loop_overrun_s):
        reach = self.ensure_reach_runner()
        if self.args.reach_body_source == "hold":
            raw_walk_target = reach.hold_whole_body_position
            if raw_walk_target is None:
                q, _ = self.read_joint_state()
                raw_walk_target = q.copy()
                reach.hold_whole_body_position = raw_walk_target.copy()
        else:
            raw_walk_target = self.step_zero_command_walk(loop_overrun_s, send_command=False)
        reach_target = reach.step_policy(send_command=False)
        if raw_walk_target is None or reach_target is None:
            return

        walk_filter = float(numpy.clip(self.args.reach_walk_target_filter, 0.0, 0.98))
        if walk_filter > 0.0:
            if self.last_reach_walk_target is None:
                self.last_reach_walk_target = raw_walk_target.copy()
            walk_target = (
                self.last_reach_walk_target * walk_filter
                + raw_walk_target * (1.0 - walk_filter)
            )
            self.last_reach_walk_target = walk_target.copy()
        else:
            walk_target = raw_walk_target
            self.last_reach_walk_target = None

        whole_body_target = walk_target.copy()
        active_arm_indices = (
            demo_reach.LEFT_ARM_WHOLE_BODY_INDICES
            if reach.active_arm == "left"
            else demo_reach.RIGHT_ARM_WHOLE_BODY_INDICES
        )
        overlay_arm_indices = (
            demo_reach.ARM_WHOLE_BODY_INDICES
            if self.args.reach_overlay_arms == "both"
            else active_arm_indices
        )
        overlay_arm_label = (
            "both" if self.args.reach_overlay_arms == "both" else reach.active_arm
        )
        overlay_alpha = float(numpy.clip(self.args.reach_overlay_scale, 0.0, 1.0))
        tilt_xy = 0.0
        if self.walk.latest_projected_gravity is not None:
            tilt_xy = float(numpy.max(numpy.abs(self.walk.latest_projected_gravity[:2])))
        max_qd = 0.0
        if self.walk.latest_qd is not None:
            max_qd = float(numpy.max(numpy.abs(self.walk.latest_qd)))
        ankle_pitch_effort = float(numpy.max(numpy.abs(self.walk.policy_action[[4, 10]])))
        stability_scale = 1.0
        min_stability_scale = float(numpy.clip(self.args.reach_stability_min_scale, 0.0, 1.0))
        if self.args.reach_stability_tilt_limit > 0.0 and tilt_xy > self.args.reach_stability_tilt_limit:
            stability_scale = min(
                stability_scale,
                self.args.reach_stability_tilt_limit / max(tilt_xy, 1.0e-6),
            )
        if self.args.reach_stability_qd_limit > 0.0 and max_qd > self.args.reach_stability_qd_limit:
            stability_scale = min(
                stability_scale,
                self.args.reach_stability_qd_limit / max(max_qd, 1.0e-6),
            )
        if (
            self.args.reach_stability_ankle_pitch_limit > 0.0
            and ankle_pitch_effort > self.args.reach_stability_ankle_pitch_limit
        ):
            stability_scale = min(
                stability_scale,
                self.args.reach_stability_ankle_pitch_limit / max(ankle_pitch_effort, 1.0e-6),
            )
        stability_scale = float(numpy.clip(stability_scale, min_stability_scale, 1.0))
        desired_arm_target = (
            walk_target[overlay_arm_indices] * (1.0 - overlay_alpha)
            + reach_target[overlay_arm_indices] * overlay_alpha
        )
        if self.args.reach_overlay_arms == "active":
            active_desired_delta = desired_arm_target - walk_target[active_arm_indices]
            shoulder_pitch_limit = float(self.args.reach_active_shoulder_pitch_delta_limit)
            elbow_limit = float(self.args.reach_active_elbow_delta_limit)
            if shoulder_pitch_limit > 0.0:
                active_desired_delta[0] = numpy.clip(
                    active_desired_delta[0],
                    -shoulder_pitch_limit,
                    shoulder_pitch_limit,
                )
            if elbow_limit > 0.0:
                active_desired_delta[3] = numpy.clip(
                    active_desired_delta[3],
                    -elbow_limit,
                    elbow_limit,
                )
            desired_arm_target = walk_target[active_arm_indices] + active_desired_delta
        if self.last_overlay_arm_target is None:
            self.last_overlay_arm_target = walk_target[overlay_arm_indices].copy()
        elif self.last_overlay_arm_target.shape != desired_arm_target.shape:
            self.last_overlay_arm_target = walk_target[overlay_arm_indices].copy()
        max_overlay_delta = float(self.args.reach_overlay_max_delta) * stability_scale
        if max_overlay_delta > 0.0:
            arm_step = numpy.clip(
                desired_arm_target - self.last_overlay_arm_target,
                -max_overlay_delta,
                max_overlay_delta,
            )
            sent_arm_target = self.last_overlay_arm_target + arm_step
        else:
            sent_arm_target = desired_arm_target
        whole_body_target[overlay_arm_indices] = sent_arm_target

        counter_delta_abs_max = 0.0
        if (
            self.args.reach_counterbalance_arm
            and self.args.reach_overlay_arms == "active"
        ):
            inactive_arm_indices = (
                demo_reach.RIGHT_ARM_WHOLE_BODY_INDICES
                if reach.active_arm == "left"
                else demo_reach.LEFT_ARM_WHOLE_BODY_INDICES
            )
            active_delta = sent_arm_target - walk_target[active_arm_indices]
            counter_weights = numpy.array([
                -1.0, 0.0, 0.0, -0.35, 0.0, 0.0, 0.0,
            ], dtype=numpy.float32)
            desired_counter_target = (
                walk_target[inactive_arm_indices]
                + active_delta * float(self.args.reach_counterbalance_scale) * counter_weights
            )
            if self.last_overlay_counter_arm_target is None:
                self.last_overlay_counter_arm_target = walk_target[inactive_arm_indices].copy()
            counter_error = desired_counter_target - self.last_overlay_counter_arm_target
            counter_offset = float(numpy.max(
                numpy.abs(self.last_overlay_counter_arm_target - walk_target[inactive_arm_indices])
            ))
            max_counter_delta = float(self.args.reach_counterbalance_max_delta)
            if counter_offset > self.args.reach_counterbalance_return_threshold:
                max_counter_delta = max(
                    max_counter_delta,
                    float(self.args.reach_counterbalance_return_max_delta),
                )
            max_counter_delta *= stability_scale
            if max_counter_delta > 0.0:
                counter_step = numpy.clip(
                    counter_error,
                    -max_counter_delta,
                    max_counter_delta,
                )
                sent_counter_target = self.last_overlay_counter_arm_target + counter_step
            else:
                sent_counter_target = desired_counter_target
            whole_body_target[inactive_arm_indices] = sent_counter_target
            counter_delta_abs_max = float(
                numpy.max(numpy.abs(sent_counter_target - walk_target[inactive_arm_indices]))
            )

        waist_delta_abs_max = 0.0
        if self.args.reach_overlay_waist_scale > 0.0:
            waist_indices = demo_reach.WAIST_ACTION_INDICES
            waist_alpha = float(numpy.clip(self.args.reach_overlay_waist_scale, 0.0, 1.0))
            desired_waist_target = (
                walk_target[waist_indices] * (1.0 - waist_alpha)
                + reach_target[waist_indices] * waist_alpha
            )
            if self.last_overlay_waist_target is None:
                self.last_overlay_waist_target = walk_target[waist_indices].copy()
            max_waist_delta = float(self.args.reach_overlay_waist_max_delta) * stability_scale
            if max_waist_delta > 0.0:
                waist_step = numpy.clip(
                    desired_waist_target - self.last_overlay_waist_target,
                    -max_waist_delta,
                    max_waist_delta,
                )
                sent_waist_target = self.last_overlay_waist_target + waist_step
            else:
                sent_waist_target = desired_waist_target
            whole_body_target[waist_indices] = sent_waist_target
            waist_delta_abs_max = float(
                numpy.max(numpy.abs(sent_waist_target - walk_target[waist_indices]))
            )

        head_delta_abs_max = 0.0
        if not self.args.reach_freeze_head:
            head_indices = demo_reach.HEAD_WHOLE_BODY_INDICES
            if self.args.reach_head_mode == "look":
                target = reach.target_pos
                target_yaw = numpy.arctan2(float(target[1]), max(0.05, float(target[0])))
                horizontal_dist = max(0.05, float(numpy.linalg.norm(target[:2])))
                head_height = float(self.args.reach_head_look_height)
                target_pitch = -numpy.arctan2(float(target[2]) - head_height, horizontal_dist)
                desired_head_target = numpy.array([
                    numpy.clip(
                        target_yaw * self.args.reach_head_look_yaw_gain,
                        -self.args.reach_head_look_yaw_limit,
                        self.args.reach_head_look_yaw_limit,
                    ),
                    numpy.clip(
                        target_pitch * self.args.reach_head_look_pitch_gain,
                        -self.args.reach_head_look_pitch_limit,
                        self.args.reach_head_look_pitch_limit,
                    ),
                ], dtype=numpy.float32)
            else:
                head_alpha = float(numpy.clip(self.args.reach_overlay_head_scale, 0.0, 1.0))
                desired_head_target = (
                    walk_target[head_indices] * (1.0 - head_alpha)
                    + reach_target[head_indices] * head_alpha
                )
            if self.last_overlay_head_target is None:
                self.last_overlay_head_target = walk_target[head_indices].copy()
            max_head_delta = float(self.args.reach_overlay_head_max_delta)
            if max_head_delta > 0.0:
                head_step = numpy.clip(
                    desired_head_target - self.last_overlay_head_target,
                    -max_head_delta,
                    max_head_delta,
                )
                sent_head_target = self.last_overlay_head_target + head_step
            else:
                sent_head_target = desired_head_target
            whole_body_target[head_indices] = sent_head_target
            head_delta_abs_max = float(
                numpy.max(numpy.abs(sent_head_target - walk_target[head_indices]))
            )

        if self.args.reach_dry_run:
            whole_body_target = walk_target
        else:
            self.last_overlay_arm_target = sent_arm_target.copy()
            if (
                self.args.reach_counterbalance_arm
                and self.args.reach_overlay_arms == "active"
            ):
                inactive_arm_indices = (
                    demo_reach.RIGHT_ARM_WHOLE_BODY_INDICES
                    if reach.active_arm == "left"
                    else demo_reach.LEFT_ARM_WHOLE_BODY_INDICES
                )
                self.last_overlay_counter_arm_target = whole_body_target[
                    inactive_arm_indices
                ].copy()
            if self.args.reach_overlay_waist_scale > 0.0:
                self.last_overlay_waist_target = whole_body_target[
                    demo_reach.WAIST_ACTION_INDICES
                ].copy()
            if not self.args.reach_freeze_head:
                self.last_overlay_head_target = whole_body_target[
                    demo_reach.HEAD_WHOLE_BODY_INDICES
                ].copy()
        self.client.set_joint_positions({"whole_body": whole_body_target.astype(numpy.float64)})

        now = time.monotonic()
        if now - self.last_overlay_print >= self.args.print_period:
            self.last_overlay_print = now
            desired_delta = desired_arm_target - walk_target[overlay_arm_indices]
            sent_delta = whole_body_target[overlay_arm_indices] - walk_target[overlay_arm_indices]
            active_sent_delta = whole_body_target[active_arm_indices] - walk_target[active_arm_indices]
            walk_filter_delta = walk_target - raw_walk_target
            print(
                f"overlay_arm={overlay_arm_label} "
                f"body_source={self.args.reach_body_source} "
                f"scale={overlay_alpha:.2f} "
                f"walk_filter={walk_filter:.2f} "
                f"stability_scale={stability_scale:.2f} "
                f"tilt_xy={tilt_xy:.3f} "
                f"max_qd={max_qd:.3f} "
                f"ankle_pitch_effort={ankle_pitch_effort:.3f} "
                f"walk_filter_delta_abs_max={numpy.max(numpy.abs(walk_filter_delta)):.3f} "
                f"desired_delta_abs_max={numpy.max(numpy.abs(desired_delta)):.3f} "
                f"sent_delta_abs_max={numpy.max(numpy.abs(sent_delta)):.3f} "
                f"active_shoulder_pitch_delta={active_sent_delta[0]:+.3f} "
                f"active_elbow_delta={active_sent_delta[3]:+.3f} "
                f"counter_delta_abs_max={counter_delta_abs_max:.3f} "
                f"waist_delta_abs_max={waist_delta_abs_max:.3f} "
                f"head_delta_abs_max={head_delta_abs_max:.3f}"
            )

    def run(self):
        if self.args.joystick:
            self.walk.setup_joystick()

        input(f"Press Enter to switch to FSM state {self.args.stand_fsm_state} (stand)")
        self.client.set_fsm_state(self.args.stand_fsm_state)
        time.sleep(1.0)

        input(
            "When stable and DDS has one correct publisher, press Enter to switch "
            f"to FSM state {self.args.usercmd_fsm_state} (UserCmd)"
        )
        self.walk.check_start_state()
        handoff_pose, _ = self.read_joint_state()
        self.hold_pose_target = handoff_pose.copy()
        self.reset_walk_from_robot()
        self.client.set_fsm_state(self.args.usercmd_fsm_state)
        time.sleep(self.args.entry_hold_s)
        self.walk.set_pd()

        self.mode = self.args.initial_mode
        if self.mode == "reach":
            self.reset_reach_from_robot()

        self.command_thread = threading.Thread(target=self.read_commands, daemon=True)
        self.command_thread.start()

        print(
            f"Running walk/reach supervisor at {self.args.rate:.1f} Hz. "
            "Type w/r/h/t/q then Enter."
        )
        period = 1.0 / self.args.rate
        next_tick = time.monotonic()
        while True:
            if not self.handle_keyboard():
                break
            loop_overrun_s = max(0.0, time.monotonic() - next_tick)
            if self.mode == "walk":
                self.walk.step_policy(loop_overrun_s)
            elif self.mode == "reach":
                self.step_reach_overlay(loop_overrun_s)
            else:
                self.step_hold()

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def close(self):
        self.stop_event.set()
        try:
            self.client.set_fsm_state(self.args.exit_fsm_state)
        except Exception:
            pass
        try:
            self.walk.close()
        except Exception:
            pass
        try:
            if self.reach is not None and self.reach.target_log_file is not None:
                self.reach.target_log_file.close()
        except Exception:
            pass


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Switch between GR2 dynamic walk and reach policies in one UserCmd session."
    )
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", default="gr2")
    parser.add_argument("--walk-policy", default=os.path.join(here, "policy_gr2_dynamic_walk_model15199_yaw_ankle_smooth_jit.pt"))
    parser.add_argument("--reach-policy", default=os.path.join(here, "policy_gr2_reach_headlook_model4999_jit.pt"))
    parser.add_argument("--balance-policy", default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--initial-mode", choices=["hold", "walk", "reach"], default="walk")
    parser.add_argument("--stand-fsm-state", type=int, default=2)
    parser.add_argument("--usercmd-fsm-state", type=int, default=10)
    parser.add_argument("--exit-fsm-state", type=int, default=2)
    parser.add_argument("--entry-hold-s", type=float, default=0.1)
    parser.add_argument("--handoff-hold-time", type=float, default=0.75)
    parser.add_argument("--print-period", type=float, default=1.0)

    parser.add_argument("--joystick", action="store_true")
    parser.add_argument("--joystick-device", default="/dev/input/js0")
    parser.add_argument("--joystick-deadzone", type=float, default=0.08)
    parser.add_argument("--joystick-forward-axis", type=int, default=1)
    parser.add_argument("--joystick-lateral-axis", type=int, default=0)
    parser.add_argument("--joystick-yaw-axis", type=int, default=3)
    parser.add_argument("--max-vx", type=float, default=0.30)
    parser.add_argument("--max-vy", type=float, default=0.0)
    parser.add_argument("--max-yaw", type=float, default=0.60)
    parser.add_argument("--filter-x", type=float, default=0.92)
    parser.add_argument("--filter-y", type=float, default=0.4)
    parser.add_argument("--filter-yaw", type=float, default=0.45)
    parser.add_argument("--ankle-pitch-scale", type=float, default=1.0)
    parser.add_argument("--ankle-roll-scale", type=float, default=1.0)
    parser.add_argument("--walk-entry-ramp-s", type=float, default=1.0)
    parser.add_argument("--walk-free-arms", action="store_true")
    parser.add_argument("--arm-shoulder-pitch", type=float, default=0.0)
    parser.add_argument("--arm-shoulder-roll", type=float, default=0.0)
    parser.add_argument("--arm-shoulder-yaw", type=float, default=0.0)
    parser.add_argument("--arm-elbow-pitch", type=float, default=0.0)
    parser.add_argument("--arm-swing-amplitude", type=float, default=0.08)
    parser.add_argument("--arm-swing-elbow-amplitude", type=float, default=0.03)
    parser.add_argument("--arm-swing-frequency", type=float, default=1.25)
    parser.add_argument("--arm-swing-vx-scale", type=float, default=0.35)

    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--target-x", type=float, default=0.24)
    parser.add_argument("--target-y", type=float, default=-0.18)
    parser.add_argument("--target-z", type=float, default=0.15)
    parser.add_argument("--target-roll", type=float, default=0.0)
    parser.add_argument("--target-pitch", type=float, default=0.0)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--target-frame-mode", choices=["base", "world"], default="base")
    parser.add_argument("--reach-body-source", choices=["walk", "hold"], default="walk")
    parser.add_argument("--reach-overlay-arms", choices=["active", "both"], default="active")
    parser.add_argument("--reach-overlay-scale", type=float, default=0.82)
    parser.add_argument("--reach-overlay-max-delta", type=float, default=0.012)
    parser.add_argument("--reach-active-shoulder-pitch-delta-limit", type=float, default=0.10)
    parser.add_argument("--reach-active-elbow-delta-limit", type=float, default=0.16)
    parser.add_argument("--reach-overlay-head-scale", type=float, default=1.0)
    parser.add_argument("--reach-overlay-head-max-delta", type=float, default=0.040)
    parser.add_argument("--reach-overlay-waist-scale", type=float, default=0.0)
    parser.add_argument("--reach-overlay-waist-max-delta", type=float, default=0.003)
    parser.add_argument("--reach-counterbalance-arm", dest="reach_counterbalance_arm", action="store_true", default=True)
    parser.add_argument("--no-reach-counterbalance-arm", dest="reach_counterbalance_arm", action="store_false")
    parser.add_argument("--reach-counterbalance-scale", type=float, default=0.85)
    parser.add_argument("--reach-counterbalance-max-delta", type=float, default=0.020)
    parser.add_argument("--reach-counterbalance-return-max-delta", type=float, default=0.030)
    parser.add_argument("--reach-counterbalance-return-threshold", type=float, default=0.080)
    parser.add_argument("--reach-stability-tilt-limit", type=float, default=0.028)
    parser.add_argument("--reach-stability-qd-limit", type=float, default=0.80)
    parser.add_argument("--reach-stability-ankle-pitch-limit", type=float, default=0.56)
    parser.add_argument("--reach-stability-min-scale", type=float, default=0.25)
    parser.add_argument("--reach-head-mode", choices=["policy", "look"], default="policy")
    parser.add_argument("--reach-head-look-height", type=float, default=0.75)
    parser.add_argument("--reach-head-look-yaw-gain", type=float, default=1.0)
    parser.add_argument("--reach-head-look-pitch-gain", type=float, default=0.9)
    parser.add_argument("--reach-head-look-yaw-limit", type=float, default=0.45)
    parser.add_argument("--reach-head-look-pitch-limit", type=float, default=0.30)
    parser.add_argument("--reach-walk-target-filter", type=float, default=0.0)
    parser.add_argument("--random-targets", action="store_true")
    parser.add_argument("--random-target-side", choices=["active", "left", "right", "both"], default="active")
    parser.add_argument("--alternate-reach-side", dest="alternate_reach_side", action="store_true", default=True)
    parser.add_argument("--fixed-reach-side", dest="alternate_reach_side", action="store_false")
    parser.add_argument("--cycle-reach-targets", dest="cycle_reach_targets", action="store_true", default=True)
    parser.add_argument("--fixed-reach-target", dest="cycle_reach_targets", action="store_false")
    parser.add_argument("--reach-target-mode", choices=["mixed", "high_sweep"], default="mixed")
    parser.add_argument("--reach-target-high-z", type=float, default=0.20)
    parser.add_argument("--reach-target-higher-z", type=float, default=0.25)
    parser.add_argument("--reach-target-highest-z", type=float, default=0.30)
    parser.add_argument("--reach-target-extreme-z", type=float, default=0.35)
    parser.add_argument("--reach-target-low-z", type=float, default=0.12)
    parser.add_argument("--reach-target-near-x", type=float, default=0.24)
    parser.add_argument("--reach-target-high-near-x", type=float, default=0.22)
    parser.add_argument("--reach-target-highest-near-x", type=float, default=0.20)
    parser.add_argument("--reach-target-extreme-near-x", type=float, default=0.18)
    parser.add_argument("--reach-target-far-x", type=float, default=0.26)
    parser.add_argument("--target-resample-interval", type=float, default=1.5)
    parser.add_argument("--random-target-seed", type=int, default=None)
    parser.add_argument("--fixed-target-rpy", action="store_true")
    parser.add_argument("--reach-startup-ramp-s", type=float, default=0.70)
    parser.add_argument("--reach-flip-settle-s", type=float, default=0.0)
    parser.add_argument("--reach-obs-reference", choices=["default", "hold"], default="default")
    parser.add_argument("--reach-action-reference", choices=["default", "hold"], default="default")
    parser.add_argument("--reach-dry-run", action="store_true")
    parser.add_argument("--reach-main-body-source", choices=["reach", "balance"], default="reach")
    parser.add_argument("--reach-balance-scope", choices=["main_body", "inactive_arm", "full"], default="main_body")
    parser.add_argument("--reach-command-main-body", dest="reach_command_main_body", action="store_true", default=True)
    parser.add_argument("--reach-hold-main-body", dest="reach_command_main_body", action="store_false")
    parser.add_argument("--reach-freeze-legs", dest="reach_freeze_legs", action="store_true", default=True)
    parser.add_argument("--reach-command-legs", dest="reach_freeze_legs", action="store_false")
    parser.add_argument("--reach-freeze-head", dest="reach_freeze_head", action="store_true", default=True)
    parser.add_argument("--reach-command-head", dest="reach_freeze_head", action="store_false")
    parser.add_argument("--reach-main-body-gain", type=float, default=0.0)
    parser.add_argument("--reach-head-gain", type=float, default=1.00)
    parser.add_argument("--reach-arm-gain", type=float, default=0.70)
    parser.add_argument("--reach-raw-action-clip-abs", type=float, default=4.0)
    parser.add_argument("--reach-action-abs-limit", type=float, default=0.60)
    parser.add_argument("--reach-max-action-delta", type=float, default=0.012)
    parser.add_argument("--active-arm-only", action="store_true", default=True)
    parser.add_argument("--both-arms", dest="active_arm_only", action="store_false")
    parser.add_argument("--balance-ramp-time", type=float, default=2.0)
    parser.add_argument("--balance-action-abs-limit", type=float, default=0.0)
    parser.add_argument("--balance-max-action-delta", type=float, default=0.03)
    parser.add_argument("--print-target-error", action="store_true", default=True)
    parser.add_argument("--no-print-target-error", dest="print_target_error", action="store_false")
    parser.add_argument("--ee-pose-frame", choices=["base", "world"], default="base")
    parser.add_argument("--target-log", default=None)

    parser.add_argument("--skip-start-check", action="store_true")
    parser.add_argument("--min-start-gravity-z", type=float, default=-0.95)
    parser.add_argument("--max-start-tilt-xy", type=float, default=0.20)
    parser.add_argument("--max-start-qd", type=float, default=1.0)
    return parser.parse_args()


def main():
    runner = None
    try:
        runner = WalkReachSequence(parse_args())
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
