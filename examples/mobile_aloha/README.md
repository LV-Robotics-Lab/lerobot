# Mobile ALOHA training and remote inference

This directory contains the reproducible source recovered from the yuhang
Mobile ALOHA workstation. The dataset, trained weights, optimizer states, and
virtual environments do not belong in Git.

## Dataset contract

The validated LeRobot v3 dataset contains 102 episodes, 123,100 frames at
50 Hz, three RGB cameras, 14 arm/gripper state and action dimensions, and one
task: `sweep the small objects into the dustpan`. The recorded `base_action` is
always zero, so the published policies control the two arms and grippers only.

The default Hub dataset is
`robotics-lv/aloha-mobile-dummy-lerobot-v3`. Set `DATASET_ROOT` to use an
already downloaded local copy, or override `DATASET_REPO_ID` for another
dataset with the same feature contract.

## Training

Run from this LeRobot checkout:

```bash
examples/mobile_aloha/train_pi05_lora.sh

STEPS=20000 BATCH_SIZE=32 SAVE_FREQ=5000 \
  examples/mobile_aloha/train_diffusion_policy.sh
```

The migrated workstation evidence supports these selections:

| Policy | Selected artifact | Evidence |
| --- | --- | --- |
| PI0.5 LoRA | step 20,000 adapter | completed 20,000 steps; final train loss 0.065 |
| SmolVLA joint | step 1,500 deployment bundle | this was the explicitly exported and checksum-manifested workstation bundle |
| Diffusion Policy | step 15,000 | lowest recorded eval loss, 0.0107; step 20,000 was 0.0111 |

The SmolVLA expert run and the 216 GB legacy PI0 checkpoint collection are not
selected for publication without independent evaluation or a documented
operator choice.

## Data-quality audit

Install the notebook dependencies, then generate and run the notebook against
an exact local dataset root:

```bash
LEROBOT_DATASET_ROOT=/absolute/path/to/dataset \
  uv run python examples/mobile_aloha/audit/build_data_quality_notebook.py
```

The recovered audit ran 24 structural checks with no failures. It also found
14 constant observation dimensions, a chronological split drift up to 0.606
standardized mean difference, and a best action/state offset of three frames
(60 ms). Those are diagnostics, not proof that a policy is safe for hardware.

## Remote PI0.5 inference

The server protocol uses Python pickle. It is suitable only for a trusted,
isolated lab network and provides neither authentication nor encryption. The
server binds to loopback by default; exposing it requires both an explicit host
and `--allow-unsafe-pickle`.

```bash
uv run python examples/mobile_aloha/remote_inference/pi05_server.py \
  --adapter /absolute/path/to/pi05_adapter \
  --host 0.0.0.0 \
  --allow-unsafe-pickle

python3 examples/mobile_aloha/remote_inference/ros1_one_shot_client.py \
  --server GPU_HOST \
  --task "sweep the small objects into the dustpan"
```

`ros1_one_shot_client.py` never publishes motor commands.
`ros1_safe_execute.py`, `ros1_long_horizon.py`, and `ros1_safe_home.py` require
an explicit `--execute` flag before motion. The home-pose script also requires
an explicit JSON pose file. The included yuhang pose is historical calibration
evidence; verify joint order, units, wiring, ROS topics, status messages, and
current calibration before using it.

## Source ownership

- LeRobot-specific training, dataset audit, and policy-serving examples live in
  this fork.
- OpenPI-specific training configuration lives in the
  `LV-Robotics-Lab/openpi` fork.
- Robot hardware wrappers remain separate repositories/submodules rather than
  being copied into this tree.

See [PROVENANCE.md](PROVENANCE.md) for the source-path and hash record used
before removing the non-Git workstation directory.
