"""Portable, offline loader for the selected ALOHA SmolVLA checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def load_bundle(bundle_dir: str | Path, device: str = "cuda"):
    """Load the full policy and processors without contacting Hugging Face Hub."""
    bundle_dir = Path(bundle_dir).resolve()
    model_dir = bundle_dir / "pretrained_model"
    vlm_assets = bundle_dir / "vlm_assets"

    cfg = SmolVLAConfig.from_pretrained(model_dir)
    cfg.pretrained_path = str(model_dir)
    cfg.vlm_model_name = str(vlm_assets)
    cfg.load_vlm_weights = False
    cfg.device = device
    cfg.use_amp = False

    policy = SmolVLAPolicy.from_pretrained(model_dir, config=cfg)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=model_dir,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": str(vlm_assets)},
        },
    )
    return policy, preprocessor, postprocessor, cfg


def _image_batch(value: Any) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected a 3D/4D image tensor, got shape {tuple(image.shape)}")
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    if image.shape[1] != 3:
        raise ValueError(f"Expected RGB channels, got shape {tuple(image.shape)}")
    image = image.to(torch.float32).div(255.0) if image.dtype == torch.uint8 else image.to(torch.float32)
    return image.contiguous()


def _vector_batch(value: Any, name: str) -> torch.Tensor:
    vector = torch.as_tensor(value, dtype=torch.float32)
    if vector.ndim == 1:
        vector = vector.unsqueeze(0)
    if vector.shape != (1, 14):
        raise ValueError(f"{name} must have shape (14,) or (1, 14), got {tuple(vector.shape)}")
    return vector


def format_observation(
    *,
    cam_high: Any,
    cam_left_wrist: Any,
    cam_right_wrist: Any,
    state: Any,
    velocity: Any,
    effort: Any,
    task: str = "sweep the small objects into the dustpan",
) -> dict[str, Any]:
    """Build one policy observation. Camera arrays must be RGB, not OpenCV BGR."""
    return {
        CAMERA_KEYS[0]: _image_batch(cam_high),
        CAMERA_KEYS[1]: _image_batch(cam_left_wrist),
        CAMERA_KEYS[2]: _image_batch(cam_right_wrist),
        "observation.state": _vector_batch(state, "state"),
        "observation.velocity": _vector_batch(velocity, "velocity"),
        "observation.effort": _vector_batch(effort, "effort"),
        "task": [task],
    }


def select_action(policy, preprocessor, postprocessor, observation: dict[str, Any]) -> torch.Tensor:
    """Return one denormalized 14-DoF action on CPU."""
    processed = preprocessor(observation)
    with torch.inference_mode():
        action = postprocessor(policy.select_action(processed))
    return action.detach().cpu()
