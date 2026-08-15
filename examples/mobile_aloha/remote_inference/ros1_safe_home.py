#!/usr/bin/env python3
"""Return Mobile ALOHA to an explicitly supplied, operator-recorded pose."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rospy
from ros1_long_horizon import assert_arm_safe
from ros1_safe_execute import ACTION_MAX, ACTION_MIN, message
from sensor_msgs.msg import JointState


def load_target(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    target = np.asarray(payload["joint_positions"], dtype=np.float32)
    if target.shape != (14,) or not np.isfinite(target).all():
        raise ValueError("home pose must contain 14 finite joint positions")
    if np.any(target < ACTION_MIN) or np.any(target > ACTION_MAX):
        raise ValueError("home pose exceeds the audited Mobile ALOHA action bounds")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    target = load_target(args.target.resolve())
    if not args.execute:
        print(f"DRY_RUN_ONLY target={args.target.resolve()} joint_positions={target.tolist()}")
        return

    rospy.init_node("mobile_aloha_safe_home", anonymous=True, disable_signals=True)
    left_pub = rospy.Publisher("/master/joint_left", JointState, queue_size=1, tcp_nodelay=True)
    right_pub = rospy.Publisher("/master/joint_right", JointState, queue_size=1, tcp_nodelay=True)
    deadline = time.monotonic() + 3
    while (
        left_pub.get_num_connections() < 1 or right_pub.get_num_connections() < 1
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    if left_pub.get_num_connections() != 1 or right_pub.get_num_connections() != 1:
        raise RuntimeError("expected exactly one subscriber per arm command topic")
    max_step = np.array([0.005] * 6 + [0.0005] + [0.005] * 6 + [0.0005], dtype=np.float32)
    end = time.monotonic() + 60
    steps = 0
    try:
        while time.monotonic() < end:
            assert_arm_safe("/puppet/arm_status_left")
            assert_arm_safe("/puppet/arm_status_right")
            left = rospy.wait_for_message("/puppet/joint_left", JointState, timeout=0.25)
            right = rospy.wait_for_message("/puppet/joint_right", JointState, timeout=0.25)
            current = np.asarray(left.position + right.position, dtype=np.float32)
            error = target - current
            if np.max(np.abs(error[:6].tolist() + error[7:13].tolist())) < 0.008:
                print(f"HOME_COMPLETE steps={steps} max_joint_error={np.max(np.abs(error)):.5f}")
                return
            safe = current + np.clip(error, -max_step, max_step)
            left_pub.publish(message(safe[:7]))
            right_pub.publish(message(safe[7:]))
            steps += 1
            if steps % 10 == 0:
                print(f"HOME_PROGRESS steps={steps} max_error={np.max(np.abs(error)):.4f}", flush=True)
            rospy.sleep(0.1)
        raise RuntimeError("home motion timed out")
    finally:
        left_pub.unregister()
        right_pub.unregister()


if __name__ == "__main__":
    main()
