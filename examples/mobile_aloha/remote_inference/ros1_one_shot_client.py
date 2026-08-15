#!/usr/bin/env python3
"""Capture one live ROS1 observation and request remote PI0.5 actions; never commands motors."""

import argparse
import pickle
import socket
import struct
import time

import numpy as np
import rospy
from sensor_msgs.msg import Image, JointState

HEADER = struct.Struct("!Q")


def image_to_numpy(msg):
    channels = 3
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding.lower() == "bgr8":
        image = image[:, :, ::-1]
    elif msg.encoding.lower() not in ("rgb8", "8uc3"):
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    return np.ascontiguousarray(image)


def recv_exact(conn, size):
    chunks = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise ConnectionError("server disconnected")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def send_packet(conn, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER.pack(len(payload)) + payload)


def recv_packet(conn):
    size = HEADER.unpack(recv_exact(conn, HEADER.size))[0]
    return pickle.loads(recv_exact(conn, size))  # nosec: trusted private LAN only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--task", default="")
    parser.add_argument("--output", default="/tmp/pi05_actions.npy")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("pi05_remote_dry_run", anonymous=True, disable_signals=True)

    left = rospy.wait_for_message("/puppet/joint_left", JointState, timeout=5)
    right = rospy.wait_for_message("/puppet/joint_right", JointState, timeout=5)
    front = rospy.wait_for_message("/camera_f/color/image_raw", Image, timeout=5)
    left_cam = rospy.wait_for_message("/camera_l/color/image_raw", Image, timeout=5)
    right_cam = rospy.wait_for_message("/camera_r/color/image_raw", Image, timeout=5)
    request = {
        "observation.state": np.asarray(left.position + right.position, dtype=np.float32),
        "observation.velocity": np.asarray(left.velocity + right.velocity, dtype=np.float32),
        "observation.effort": np.asarray(left.effort + right.effort, dtype=np.float32),
        "observation.images.cam_high": image_to_numpy(front),
        "observation.images.cam_left_wrist": image_to_numpy(left_cam),
        "observation.images.cam_right_wrist": image_to_numpy(right_cam),
        "task": args.task,
    }
    start = time.perf_counter()
    with socket.create_connection((args.server, args.port), timeout=300) as conn:
        send_packet(conn, request)
        response = recv_packet(conn)
    if "error" in response:
        raise RuntimeError(response["error"])
    actions = np.asarray(response["actions"], dtype=np.float32)
    np.save(args.output, actions)
    print(
        f"REMOTE_INFERENCE_OK shape={actions.shape} server_s={response['inference_s']:.3f} roundtrip_s={time.perf_counter() - start:.3f}"
    )
    print(f"first_action={actions[0].tolist()}")
    print(f"saved={args.output}")
    print("DRY_RUN_ONLY: no ROS action topic was published")


if __name__ == "__main__":
    main()
