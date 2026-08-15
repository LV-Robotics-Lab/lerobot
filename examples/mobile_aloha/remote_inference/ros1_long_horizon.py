#!/usr/bin/env python3
"""Bounded, continuously replanned Mobile ALOHA remote-policy execution."""

import argparse
import time

import numpy as np
import rospy
from piper_msgs.msg import PiperStatusMsg
from ros1_safe_execute import ACTION_MAX, ACTION_MIN, capture, message, rpc
from sensor_msgs.msg import JointState


def assert_arm_safe(topic):
    status = rospy.wait_for_message(topic, PiperStatusMsg, timeout=0.25)
    if status.err_code != 0 or status.arm_status != 0:
        raise RuntimeError(f"unsafe arm status on {topic}: err={status.err_code} status={status.arm_status}")
    fields = [
        "joint_1_angle_limit",
        "joint_2_angle_limit",
        "joint_3_angle_limit",
        "joint_4_angle_limit",
        "joint_5_angle_limit",
        "joint_6_angle_limit",
        "communication_status_joint_1",
        "communication_status_joint_2",
        "communication_status_joint_3",
        "communication_status_joint_4",
        "communication_status_joint_5",
        "communication_status_joint_6",
    ]
    bad = [name for name in fields if getattr(status, name)]
    if bad:
        raise RuntimeError(f"unsafe arm flags on {topic}: {bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--steps-per-plan", type=int, default=20)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step", type=float, default=0.005)
    parser.add_argument("--max-gripper-step", type=float, default=0.0005)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    if not 1 <= args.duration <= 300 or not 1 <= args.steps_per_plan <= 20:
        raise ValueError("duration must be 1..300s and steps-per-plan 1..20")

    rospy.init_node("pi05_remote_long_horizon", anonymous=True, disable_signals=True)
    if not args.execute:
        print("DRY_RUN_ONLY: add --execute only after status and command topics are verified")
        return
    left_pub = rospy.Publisher("/master/joint_left", JointState, queue_size=1, tcp_nodelay=True)
    right_pub = rospy.Publisher("/master/joint_right", JointState, queue_size=1, tcp_nodelay=True)
    deadline = time.monotonic() + 3
    while (
        left_pub.get_num_connections() < 1 or right_pub.get_num_connections() < 1
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    if left_pub.get_num_connections() != 1 or right_pub.get_num_connections() != 1:
        raise RuntimeError("expected exactly one subscriber per arm command topic")

    max_step = np.array(
        [args.max_joint_step] * 6
        + [args.max_gripper_step]
        + [args.max_joint_step] * 6
        + [args.max_gripper_step],
        dtype=np.float32,
    )
    end_time = time.monotonic() + args.duration
    plan_index = 0
    executed = 0
    try:
        while time.monotonic() < end_time and not rospy.is_shutdown():
            assert_arm_safe("/puppet/arm_status_left")
            assert_arm_safe("/puppet/arm_status_right")
            request = capture(args.task)
            rpc_start = time.perf_counter()
            response = rpc(args.server, args.port, request)
            if "error" in response:
                raise RuntimeError(response["error"])
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid action chunk: {actions.shape}")
            plan_index += 1
            print(
                f"PLAN {plan_index} server_s={response['inference_s']:.3f} "
                f"roundtrip_s={time.perf_counter() - rpc_start:.3f}",
                flush=True,
            )
            rate = rospy.Rate(args.hz)
            for action in actions[: args.steps_per_plan]:
                if time.monotonic() >= end_time:
                    break
                assert_arm_safe("/puppet/arm_status_left")
                assert_arm_safe("/puppet/arm_status_right")
                left = rospy.wait_for_message("/puppet/joint_left", JointState, timeout=0.25)
                right = rospy.wait_for_message("/puppet/joint_right", JointState, timeout=0.25)
                current = np.asarray(left.position + right.position, dtype=np.float32)
                desired = np.clip(action, ACTION_MIN, ACTION_MAX)
                safe = current + np.clip(desired - current, -max_step, max_step)
                left_pub.publish(message(safe[:7]))
                right_pub.publish(message(safe[7:]))
                executed += 1
                rate.sleep()
    finally:
        left_pub.unregister()
        right_pub.unregister()
        print(f"LONG_HORIZON_STOP plans={plan_index} executed_steps={executed}", flush=True)


if __name__ == "__main__":
    main()
