#!/usr/bin/env python3
"""Minimal PI0.5 inference server for the Mobile ALOHA ROS bridge.

The wire format uses pickle and must only be exposed on a trusted, isolated lab
network. Non-loopback binds require an explicit acknowledgement flag.
"""

import argparse
import pickle
import socket
import struct
import time
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

HEADER = struct.Struct("!Q")


def recv_packet(conn):
    header = recv_exact(conn, HEADER.size)
    return pickle.loads(recv_exact(conn, HEADER.unpack(header)[0]))  # nosec: trusted private LAN only


def recv_exact(conn, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("peer disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(conn, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER.pack(len(payload)) + payload)


def load_policy(adapter_path: Path, device: torch.device):
    config = PreTrainedConfig.from_pretrained(adapter_path, local_files_only=True)
    config.device = str(device)
    config.pretrained_path = str(adapter_path)
    peft_config = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
    policy_cls = get_policy_class(config.type)
    base = policy_cls.from_pretrained(
        peft_config.base_model_name_or_path,
        config=config,
        revision=peft_config.revision,
        local_files_only=True,
    )
    policy = PeftModel.from_pretrained(base, adapter_path, config=peft_config, local_files_only=True).eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(adapter_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post


def infer(policy, pre, post, request, device):
    task = request.pop("task", "")
    obs = prepare_observation_for_inference(request, device, task, "mobile_aloha")
    with torch.inference_mode():
        obs = pre(obs)
        chunk = policy.predict_action_chunk(obs)
        processed = [post(chunk[:, i, :]).squeeze(0).cpu() for i in range(chunk.shape[1])]
    return torch.stack(processed).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="required when binding the pickle RPC server to a non-loopback address",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_unsafe_pickle:
        parser.error("non-loopback binds require --allow-unsafe-pickle on a trusted lab network")
    device = torch.device("cuda")
    policy, pre, post = load_policy(args.adapter.resolve(), device)
    print(f"SERVER_READY {args.host}:{args.port}", flush=True)
    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        while True:
            conn, address = server.accept()
            with conn:
                try:
                    request = recv_packet(conn)
                    start = time.perf_counter()
                    actions = infer(policy, pre, post, request, device)
                    send_packet(
                        conn,
                        {"actions": actions.tolist(), "inference_s": time.perf_counter() - start},
                    )
                    print(f"INFERENCE_OK peer={address[0]} shape={actions.shape}", flush=True)
                except Exception as exc:
                    send_packet(conn, {"error": repr(exc)})
                    print(f"INFERENCE_ERROR {exc!r}", flush=True)


if __name__ == "__main__":
    main()
