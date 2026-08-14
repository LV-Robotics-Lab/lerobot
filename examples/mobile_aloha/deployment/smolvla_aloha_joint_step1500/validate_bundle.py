"""Offline smoke test for the packaged SmolVLA policy."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from load_policy import format_observation, load_bundle, select_action


def main() -> None:
    bundle_dir = Path(__file__).resolve().parent
    policy, preprocessor, postprocessor, cfg = load_bundle(bundle_dir)
    policy.reset()

    black_image = torch.zeros((480, 640, 3), dtype=torch.uint8)
    observation = format_observation(
        cam_high=black_image,
        cam_left_wrist=black_image,
        cam_right_wrist=black_image,
        state=torch.zeros(14),
        velocity=torch.zeros(14),
        effort=torch.zeros(14),
    )
    action = select_action(policy, preprocessor, postprocessor, observation)
    print(
        json.dumps(
            {
                "checkpoint_load": "ok",
                "offline_mode": True,
                "action_shape": list(action.shape),
                "action_finite": bool(torch.isfinite(action).all()),
                "n_action_steps": cfg.n_action_steps,
                "chunk_size": cfg.chunk_size,
                "load_vlm_weights": cfg.load_vlm_weights,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
