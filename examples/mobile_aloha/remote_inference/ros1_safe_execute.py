#!/usr/bin/env python3
"""Short, rate-limited Mobile ALOHA execution of one remotely inferred action chunk."""

import argparse
import pickle  # nosec B403: compatibility-bound protocol for an isolated lab RPC
import socket
import struct
import time

import numpy as np
import rospy
from sensor_msgs.msg import Image, JointState

HEADER = struct.Struct("!Q")
ACTION_MIN = np.array(
    [
        -1.5176,
        0.0,
        -2.9655,
        -0.9553,
        -1.2211,
        -1.5961,
        0.0,
        -0.3750,
        0.0,
        -2.1464,
        -1.2188,
        -0.9200,
        -0.5045,
        0.0,
    ],
    dtype=np.float32,
)
ACTION_MAX = np.array(
    [
        0.2133,
        2.3618,
        0.0,
        1.5719,
        1.2211,
        0.4225,
        0.0961,
        0.7780,
        2.1146,
        0.0,
        1.5377,
        0.7834,
        1.2974,
        0.0831,
    ],
    dtype=np.float32,
)


def recv_exact(conn, size):
    chunks = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise ConnectionError("server disconnected")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def rpc(server, port, request):
    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    with socket.create_connection((server, port), timeout=10) as conn:
        conn.settimeout(300)
        conn.sendall(HEADER.pack(len(payload)) + payload)
        size = HEADER.unpack(recv_exact(conn, HEADER.size))[0]
        return pickle.loads(recv_exact(conn, size))  # nosec: trusted private LAN


def image_to_numpy(msg):
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == "bgr8":
        image = image[:, :, ::-1]
    elif msg.encoding.lower() not in ("rgb8", "8uc3"):
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    return np.ascontiguousarray(image)


def capture(task):
    left = rospy.wait_for_message("/puppet/joint_left", JointState, timeout=2)
    right = rospy.wait_for_message("/puppet/joint_right", JointState, timeout=2)
    if min(len(left.position), len(right.position)) != 7:
        raise RuntimeError("expected seven joints per arm")
    request = {
        "observation.state": np.asarray(left.position + right.position, dtype=np.float32),
        "observation.velocity": np.asarray(left.velocity + right.velocity, dtype=np.float32),
        "observation.effort": np.asarray(left.effort + right.effort, dtype=np.float32),
        "observation.images.cam_high": image_to_numpy(
            rospy.wait_for_message("/camera_f/color/image_raw", Image, timeout=2)
        ),
        "observation.images.cam_left_wrist": image_to_numpy(
            rospy.wait_for_message("/camera_l/color/image_raw", Image, timeout=2)
        ),
        "observation.images.cam_right_wrist": image_to_numpy(
            rospy.wait_for_message("/camera_r/color/image_raw", Image, timeout=2)
        ),
        "task": task,
    }
    return request


def message(values):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = [f"joint{i}" for i in range(7)]
    msg.position = values.tolist()
    msg.velocity = [0.0] * 7
    msg.effort = [0.0] * 7
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--task", default="")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-joint-step", type=float, default=0.01)
    parser.add_argument("--max-gripper-step", type=float, default=0.001)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    if not 1 <= args.steps <= 20 or not 1 <= args.hz <= 20:
        raise ValueError("safety limits require steps=1..20 and hz=1..20")

    rospy.init_node("pi05_remote_safe_execute", anonymous=True, disable_signals=True)
    request = capture(args.task)
    start = time.perf_counter()
    response = rpc(args.server, args.port, request)
    if "error" in response:
        raise RuntimeError(response["error"])
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (50, 14) or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid action chunk: {actions.shape}")
    print(
        f"INFERENCE_OK server_s={response['inference_s']:.3f} roundtrip_s={time.perf_counter() - start:.3f}"
    )
    if not args.execute:
        print("DRY_RUN_OK: add --execute only after control subscribers are ready")
        return

    left_pub = rospy.Publisher("/master/joint_left", JointState, queue_size=1, tcp_nodelay=True)
    right_pub = rospy.Publisher("/master/joint_right", JointState, queue_size=1, tcp_nodelay=True)
    deadline = time.monotonic() + 3
    while (
        left_pub.get_num_connections() < 1 or right_pub.get_num_connections() < 1
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    if left_pub.get_num_connections() < 1 or right_pub.get_num_connections() < 1:
        raise RuntimeError("policy command topics have no arm subscribers; refusing to execute")

    current = request["observation.state"].copy()
    max_step = np.array(
        [args.max_joint_step] * 6
        + [args.max_gripper_step]
        + [args.max_joint_step] * 6
        + [args.max_gripper_step],
        dtype=np.float32,
    )
    rate = rospy.Rate(args.hz)
    for index in range(args.steps):
        left = rospy.wait_for_message("/puppet/joint_left", JointState, timeout=0.25)
        right = rospy.wait_for_message("/puppet/joint_right", JointState, timeout=0.25)
        current = np.asarray(left.position + right.position, dtype=np.float32)
        desired = np.clip(actions[index], ACTION_MIN, ACTION_MAX)
        safe = current + np.clip(desired - current, -max_step, max_step)
        left_pub.publish(message(safe[:7]))
        right_pub.publish(message(safe[7:]))
        print(f"EXEC step={index + 1}/{args.steps} max_delta={np.max(np.abs(safe - current)):.5f}")
        rate.sleep()
    print("SAFE_EXECUTION_COMPLETE: command publishing stopped")


if __name__ == "__main__":
    main()
