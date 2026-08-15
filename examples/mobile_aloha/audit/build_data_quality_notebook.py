from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = AUDIT_DIR / "smolvla_data_quality_audit.ipynb"
RESULTS_PATH = AUDIT_DIR / "audit_results.json"


def verified_tldr() -> str:
    if not RESULTS_PATH.exists():
        return (
            "## tl;dr\n\n"
            "This summary will be populated after the first successful top-to-bottom execution. "
            "The executable checks below are the source of truth."
        )

    results = json.loads(RESULTS_PATH.read_text())
    integrity = results["integrity"]
    split = results["split_drift"]
    constants = results["constant_observation_dimensions"]
    video = results["video_summary"]
    alignment = results["action_state_alignment"]
    duplicates = results["episode_duplication"]

    structural = "passed" if integrity["failed_checks"] == 0 else "failed"
    max_smd = split["max_abs_episode_mean_smd"]
    bad_video = sum(
        row["near_black_share"] > 0.01
        or row["overexposed_share"] > 0.01
        or row["near_duplicate_1s_share"] > 0.01
        for row in video
    )
    return f"""## tl;dr

- Structural integrity **{structural}**: {integrity["failed_checks"]} failed contract checks across {results["dataset"]["frames"]:,} frames and {results["dataset"]["episodes"]} episodes.
- The model receives **{constants["count"]} zero-variance observation dimensions** out of 42 scalar state inputs; any non-zero runtime value in these dimensions is amplified by MEAN_STD normalization with `eps=1e-8`.
- The chronological train/eval split has a maximum absolute episode-mean standardized difference of **{max_smd:.2f}**, so offline evaluation is not an independent, identically distributed test.
- Action/state alignment is best at **+{alignment["best_future_offset_frames"]} frames** ({alignment["best_future_offset_ms"]:.0f} ms) under the normalized RMSE diagnostic.
- The closest pair of resampled action trajectories has normalized RMSE **{duplicates["closest_pair_normalized_rmse"]:.4f}** (episodes {duplicates["closest_pair"][0]} and {duplicates["closest_pair"][1]}).
- At 1 Hz video sampling, **{bad_video} of 3 cameras** exceed the 1% heuristic threshold for black, overexposed, or near-duplicate frames. See the video section and contact sheet before interpreting this as a camera defect.
"""


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "LeRobot Python 3.12",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

cells: list = []
cells.append(nbf.v4.new_markdown_cell("# SmolVLA training-data quality audit"))
cells.append(nbf.v4.new_markdown_cell(verified_tldr()))
cells.append(
    nbf.v4.new_markdown_cell(
        """## Context & Methods

This notebook audits the local LeRobot v3 dataset used by the deployed SmolVLA checkpoint. The intended grain is one synchronized frame at 50 Hz within one demonstration episode. The downstream use is real-robot imitation learning for the single task `sweep the small objects into the dustpan`.

### Key Assumptions

- `observation.state` and `action` use the same 14-joint order and compatible position units.
- `observation.velocity` is intended to represent the derivative of joint position closely enough for correlation checks, although filtering and sensor latency can reduce agreement.
- Video-container frame order is aligned to the frame-level parquet through episode video timestamps.
- The final 11 episodes are evaluation data because the local LeRobot implementation holds out the last `ceil(102 × 0.1)` episodes per task.
- Video-quality thresholds are diagnostics, not universal pass/fail limits; visual inspection and real-robot observation logs remain necessary.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from __future__ import annotations

import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get(\"LEROBOT_DATASET_ROOT\", Path.cwd())).resolve()
AUDIT_DIR = ROOT / \"audit\"
AUDIT_DIR.mkdir(exist_ok=True)
FPS = 50.0
VECTOR_COLUMNS = [
    \"observation.state\",
    \"observation.velocity\",
    \"observation.effort\",
    \"action\",
]
JOINT_NAMES = [
    \"left_waist\", \"left_shoulder\", \"left_elbow\", \"left_forearm_roll\",
    \"left_wrist_angle\", \"left_wrist_rotate\", \"left_gripper\",
    \"right_waist\", \"right_shoulder\", \"right_elbow\", \"right_forearm_roll\",
    \"right_wrist_angle\", \"right_wrist_rotate\", \"right_gripper\",
]
CAMERA_KEYS = [\"cam_high\", \"cam_left_wrist\", \"cam_right_wrist\"]
plt.style.use(\"seaborn-v0_8-whitegrid\")
pd.set_option(\"display.max_columns\", 60)
pd.set_option(\"display.width\", 180)
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Data\n\n### 1. Load the exact training inputs"))
cells.append(
    nbf.v4.new_code_cell(
        """info = json.loads((ROOT / \"meta/info.json\").read_text())
stats = json.loads((ROOT / \"meta/stats.json\").read_text())
tasks = pd.read_parquet(ROOT / \"meta/tasks.parquet\")
episodes = pd.read_parquet(ROOT / \"meta/episodes/chunk-000/file-000.parquet\")
frames = pd.read_parquet(ROOT / \"data/chunk-000/file-000.parquet\")

arrays = {column: np.stack(frames[column].to_numpy()).astype(np.float64) for column in VECTOR_COLUMNS}
arrays[\"base_action\"] = np.stack(frames[\"base_action\"].to_numpy()).astype(np.float64)

state = arrays[\"observation.state\"]
velocity = arrays[\"observation.velocity\"]
effort = arrays[\"observation.effort\"]
action = arrays[\"action\"]
base_action = arrays[\"base_action\"]

n_eval = math.ceil(info[\"total_episodes\"] * 0.1)
eval_episode_start = info[\"total_episodes\"] - n_eval
is_eval_frame = frames[\"episode_index\"].to_numpy() >= eval_episode_start
is_train_frame = ~is_eval_frame

dataset_overview = pd.DataFrame([
    {
        \"codebase_version\": info[\"codebase_version\"],
        \"robot_type\": info[\"robot_type\"],
        \"episodes\": len(episodes),
        \"frames\": len(frames),
        \"hours\": len(frames) / FPS / 3600,
        \"tasks\": len(tasks),
        \"task_text\": tasks.index[0],
        \"train_episodes\": eval_episode_start,
        \"eval_episodes\": n_eval,
    }
])
display(dataset_overview)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## Results\n\n### 2. Structural contracts and synchronization are checked first"
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """integrity_rows = []

def add_check(name: str, passed: bool, evidence: str) -> None:
    integrity_rows.append({\"check\": name, \"passed\": bool(passed), \"evidence\": evidence})

add_check(\"metadata frame count\", len(frames) == info[\"total_frames\"], f\"parquet={len(frames):,}, metadata={info['total_frames']:,}\")
add_check(\"metadata episode count\", len(episodes) == info[\"total_episodes\"], f\"parquet={len(episodes)}, metadata={info['total_episodes']}\")
add_check(\"global index contiguous\", np.array_equal(frames[\"index\"].to_numpy(), np.arange(len(frames))), f\"unique={frames['index'].nunique():,}\")
add_check(\"episode ids contiguous\", np.array_equal(np.sort(frames[\"episode_index\"].unique()), np.arange(len(episodes))), f\"range={frames.episode_index.min()}..{frames.episode_index.max()}\")
add_check(\"task ids valid\", set(frames[\"task_index\"].unique()) == set(tasks[\"task_index\"].to_numpy()), f\"observed={sorted(frames.task_index.unique())}\")

episode_length_matches = frames.groupby(\"episode_index\").size().reindex(episodes[\"episode_index\"]).to_numpy() == episodes[\"length\"].to_numpy()
add_check(\"episode lengths match\", episode_length_matches.all(), f\"mismatches={int((~episode_length_matches).sum())}\")

frame_contract_failures = 0
timestamp_max_error = 0.0
for episode_index, group in frames.groupby(\"episode_index\", sort=True):
    expected_frame_index = np.arange(len(group))
    if not np.array_equal(group[\"frame_index\"].to_numpy(), expected_frame_index):
        frame_contract_failures += 1
    expected_timestamp = expected_frame_index / FPS
    timestamp_max_error = max(timestamp_max_error, float(np.max(np.abs(group[\"timestamp\"].to_numpy() - expected_timestamp))))
add_check(\"frame index resets per episode\", frame_contract_failures == 0, f\"bad episodes={frame_contract_failures}\")
add_check(\"timestamps follow 50 Hz\", timestamp_max_error < 5e-6, f\"max absolute error={timestamp_max_error:.3g} s\")

for column, width in [(name, 14) for name in VECTOR_COLUMNS] + [(\"base_action\", 2)]:
    observed_lengths = frames[column].map(len)
    add_check(f\"{column} shape\", bool((observed_lengths == width).all()), f\"bad rows={int((observed_lengths != width).sum())}\")
    add_check(f\"{column} finite\", bool(np.isfinite(arrays[column]).all()), f\"nonfinite={int((~np.isfinite(arrays[column])).sum())}\")

video_contract_rows = []
for camera in CAMERA_KEYS:
    video_dir = ROOT / f\"videos/observation.images.{camera}/chunk-000\"
    files_for_camera = sorted(video_dir.glob(\"*.mp4\"))
    declared_frames = 0
    for video_path in files_for_camera:
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        declared_frames += frame_count
        video_contract_rows.append({
            \"camera\": camera,
            \"file\": video_path.name,
            \"frames\": frame_count,
            \"fps\": fps,
            \"width\": width,
            \"height\": height,
        })
    add_check(f\"{camera} total video frames\", declared_frames == len(frames), f\"video={declared_frames:,}, parquet={len(frames):,}\")

    duration_errors = []
    from_column = f\"videos/observation.images.{camera}/from_timestamp\"
    to_column = f\"videos/observation.images.{camera}/to_timestamp\"
    for _, row in episodes.iterrows():
        duration_errors.append(abs((row[to_column] - row[from_column]) - row[\"length\"] / FPS))
    add_check(f\"{camera} episode timestamp spans\", max(duration_errors) < 5e-6, f\"max duration error={max(duration_errors):.3g} s\")

integrity = pd.DataFrame(integrity_rows)
display(integrity)
display(pd.DataFrame(video_contract_rows))
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 3. Episode design reveals chronological collection batches"))
cells.append(
    nbf.v4.new_code_cell(
        """episode_summary = episodes[[\"episode_index\", \"length\"]].copy()
episode_summary[\"split\"] = np.where(episode_summary[\"episode_index\"] >= eval_episode_start, \"eval\", \"train\")
episode_summary[\"duration_s\"] = episode_summary[\"length\"] / FPS

for episode_index, group in frames.groupby(\"episode_index\", sort=True):
    idx = group.index.to_numpy()
    local_action = action[idx]
    local_state = state[idx]
    local_action_diff = np.diff(local_action, axis=0)
    target_row = episode_summary.index[episode_summary[\"episode_index\"] == episode_index][0]
    episode_summary.loc[target_row, \"action_idle_share_1e-4\"] = float(np.mean(np.max(np.abs(local_action_diff), axis=1) < 1e-4))
    episode_summary.loc[target_row, \"action_motion_mean\"] = float(np.mean(np.abs(local_action_diff)))
    episode_summary.loc[target_row, \"left_gripper_range\"] = float(np.ptp(local_action[:, 6]))
    episode_summary.loc[target_row, \"right_gripper_range\"] = float(np.ptp(local_action[:, 13]))
    for j, name in enumerate(JOINT_NAMES):
        episode_summary.loc[target_row, f\"action_mean__{name}\"] = float(local_action[:, j].mean())
        episode_summary.loc[target_row, f\"state_mean__{name}\"] = float(local_state[:, j].mean())

display(episode_summary.groupby([\"split\", \"length\"]).size().rename(\"episodes\").reset_index())

fig, ax = plt.subplots(figsize=(11, 3.8))
colors = np.where(episode_summary[\"split\"].eq(\"eval\"), \"#d28b26\", \"#3f6f9f\")
ax.bar(episode_summary[\"episode_index\"], episode_summary[\"duration_s\"], color=colors, width=0.9)
ax.axvline(eval_episode_start - 0.5, color=\"#31363b\", linestyle=\"--\", linewidth=1.3)
ax.set(title=\"Episode duration and chronological evaluation split\", xlabel=\"Episode index\", ylabel=\"Duration (s)\")
fig.tight_layout()
fig.savefig(AUDIT_DIR / \"episode_duration_split.png\", dpi=160)
plt.show()
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 4. Scalar-input validity and normalization hazards"))
cells.append(
    nbf.v4.new_code_cell(
        """scalar_rows = []
for feature_name, values in [(\"state\", state), (\"velocity\", velocity), (\"effort\", effort), (\"action\", action)]:
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        column_values = values[:, joint_index]
        scalar_rows.append({
            \"feature\": feature_name,
            \"joint\": joint_name,
            \"min\": float(column_values.min()),
            \"max\": float(column_values.max()),
            \"mean\": float(column_values.mean()),
            \"std\": float(column_values.std()),
            \"zero_share\": float(np.mean(np.abs(column_values) <= 1e-12)),
            \"unique_values\": int(np.unique(column_values).size),
        })
scalar_profile = pd.DataFrame(scalar_rows)
constant_observation = scalar_profile.query("feature in ['state', 'velocity', 'effort'] and std <= 1e-8").copy()
display(constant_observation)

within_episode = frames[\"episode_index\"].to_numpy()[1:] == frames[\"episode_index\"].to_numpy()[:-1]
action_delta = np.diff(action, axis=0)[within_episode]
state_delta = np.diff(state, axis=0)[within_episode]
motion_summary = pd.DataFrame([
    {
        \"metric\": \"consecutive action rows unchanged\",
        \"share\": float(np.mean(np.max(np.abs(action_delta), axis=1) <= 1e-12)),
    },
    {
        \"metric\": \"consecutive action rows nearly unchanged (<1e-4)\",
        \"share\": float(np.mean(np.max(np.abs(action_delta), axis=1) < 1e-4)),
    },
    {
        \"metric\": \"consecutive state rows unchanged\",
        \"share\": float(np.mean(np.max(np.abs(state_delta), axis=1) <= 1e-12)),
    },
    {
        \"metric\": \"base_action rows exactly zero\",
        \"share\": float(np.mean(np.max(np.abs(base_action), axis=1) == 0)),
    },
])
display(motion_summary)
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 5. Velocity is compared with finite-difference position"))
cells.append(
    nbf.v4.new_code_cell(
        """finite_difference_velocity = np.diff(state, axis=0)[within_episode] * FPS
reported_velocity = velocity[1:][within_episode]
velocity_consistency_rows = []
for joint_index, joint_name in enumerate(JOINT_NAMES):
    fd = finite_difference_velocity[:, joint_index]
    rv = reported_velocity[:, joint_index]
    if fd.std() > 1e-12 and rv.std() > 1e-12:
        correlation = float(np.corrcoef(fd, rv)[0, 1])
    else:
        correlation = np.nan
    velocity_consistency_rows.append({
        \"joint\": joint_name,
        \"finite_difference_std\": float(fd.std()),
        \"reported_velocity_std\": float(rv.std()),
        \"correlation\": correlation,
        \"rmse\": float(np.sqrt(np.mean((fd - rv) ** 2))),
    })
velocity_consistency = pd.DataFrame(velocity_consistency_rows)
display(velocity_consistency)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell("### 6. Action targets are tested against current and future measured states")
)
cells.append(
    nbf.v4.new_code_cell(
        """offset_rows = []
joint_scale = np.maximum(action.std(axis=0), 1e-8)
for offset in range(0, 11):
    action_parts = []
    future_state_parts = []
    for episode_index, group in frames.groupby(\"episode_index\", sort=True):
        idx = group.index.to_numpy()
        if offset == 0:
            action_parts.append(action[idx])
            future_state_parts.append(state[idx])
        else:
            action_parts.append(action[idx[:-offset]])
            future_state_parts.append(state[idx[offset:]])
    aligned_action = np.concatenate(action_parts)
    aligned_state = np.concatenate(future_state_parts)
    normalized_error = (aligned_action - aligned_state) / joint_scale
    offset_rows.append({
        \"future_offset_frames\": offset,
        \"future_offset_ms\": offset / FPS * 1000,
        \"normalized_rmse\": float(np.sqrt(np.mean(normalized_error ** 2))),
        \"raw_mae\": float(np.mean(np.abs(aligned_action - aligned_state))),
    })
offset_alignment = pd.DataFrame(offset_rows)
display(offset_alignment)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(offset_alignment[\"future_offset_ms\"], offset_alignment[\"normalized_rmse\"], marker=\"o\", color=\"#3f6f9f\")
ax.set(title=\"Action target versus future measured state\", xlabel=\"Future state offset (ms)\", ylabel=\"Normalized RMSE (lower is better)\")
fig.tight_layout()
fig.savefig(AUDIT_DIR / \"action_state_alignment.png\", dpi=160)
plt.show()
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 7. Chronological train/eval drift is measured at episode grain"))
cells.append(
    nbf.v4.new_code_cell(
        """episode_feature_columns = [column for column in episode_summary.columns if column.startswith((\"action_mean__\", \"state_mean__\"))]
train_episode_features = episode_summary.loc[episode_summary[\"split\"] == \"train\", episode_feature_columns]
eval_episode_features = episode_summary.loc[episode_summary[\"split\"] == \"eval\", episode_feature_columns]

split_drift_rows = []
for column in episode_feature_columns:
    train_values = train_episode_features[column].to_numpy(dtype=float)
    eval_values = eval_episode_features[column].to_numpy(dtype=float)
    pooled_variance = ((len(train_values) - 1) * train_values.var(ddof=1) + (len(eval_values) - 1) * eval_values.var(ddof=1)) / (len(train_values) + len(eval_values) - 2)
    pooled_std = math.sqrt(max(pooled_variance, 0.0))
    smd = (eval_values.mean() - train_values.mean()) / pooled_std if pooled_std > 1e-12 else np.nan
    feature, joint = column.split(\"__\", 1)
    split_drift_rows.append({
        \"feature\": feature.replace(\"_mean\", \"\"),
        \"joint\": joint,
        \"train_episode_mean\": float(train_values.mean()),
        \"eval_episode_mean\": float(eval_values.mean()),
        \"standardized_mean_difference\": float(smd),
        \"abs_smd\": float(abs(smd)),
    })
split_drift = pd.DataFrame(split_drift_rows).sort_values(\"abs_smd\", ascending=False)
display(split_drift.head(16))

plot_drift = split_drift.head(16).sort_values(\"abs_smd\")
fig, ax = plt.subplots(figsize=(8.5, 6.0))
labels = plot_drift[\"feature\"] + \" / \" + plot_drift[\"joint\"]
ax.barh(labels, plot_drift[\"abs_smd\"], color=\"#d28b26\")
ax.axvline(0.5, color=\"#31363b\", linestyle=\"--\", linewidth=1.1)
ax.set(title=\"Largest train/eval shifts in episode-mean joint positions\", xlabel=\"Absolute standardized mean difference\", ylabel=\"\")
fig.tight_layout()
fig.savefig(AUDIT_DIR / \"train_eval_split_drift.png\", dpi=160)
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell("### 8. Demonstration diversity is checked using resampled action trajectories")
)
cells.append(
    nbf.v4.new_code_cell(
        """trajectory_points = 100
normalized_trajectories = []
for episode_index, group in frames.groupby(\"episode_index\", sort=True):
    idx = group.index.to_numpy()
    local_action = action[idx]
    source_axis = np.linspace(0.0, 1.0, len(local_action))
    target_axis = np.linspace(0.0, 1.0, trajectory_points)
    resampled = np.column_stack([
        np.interp(target_axis, source_axis, local_action[:, joint_index])
        for joint_index in range(action.shape[1])
    ])
    normalized_trajectories.append((resampled - action.mean(axis=0)) / joint_scale)
normalized_trajectories = np.stack(normalized_trajectories)

pairwise_rmse = np.full((len(episodes), len(episodes)), np.inf)
for left in range(len(episodes)):
    errors = normalized_trajectories[left + 1:] - normalized_trajectories[left]
    if len(errors):
        distances = np.sqrt(np.mean(errors ** 2, axis=(1, 2)))
        pairwise_rmse[left, left + 1:] = distances
closest_flat = np.argmin(pairwise_rmse)
closest_pair = np.unravel_index(closest_flat, pairwise_rmse.shape)

nearest_distance = np.minimum(pairwise_rmse, pairwise_rmse.T).min(axis=1)
episode_summary[\"nearest_trajectory_rmse\"] = nearest_distance
duplicate_summary = pd.DataFrame([
    {
        \"closest_episode_a\": int(closest_pair[0]),
        \"closest_episode_b\": int(closest_pair[1]),
        \"closest_normalized_rmse\": float(pairwise_rmse[closest_pair]),
        \"median_nearest_normalized_rmse\": float(np.median(nearest_distance)),
        \"episodes_with_nearest_rmse_below_0.05\": int(np.sum(nearest_distance < 0.05)),
    }
])
display(duplicate_summary)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "### 9. Full-duration video sampling checks exposure, sharpness, and near-duplicate frames"
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """SAMPLE_WIDTH = 160
SAMPLE_HEIGHT = 120
SAMPLE_SECONDS = set(np.linspace(0, int(len(frames) / FPS) - 1, 12, dtype=int).tolist())

def scan_camera(camera: str):
    rows = []
    selected_frames = {}
    global_second = 0
    video_paths = sorted((ROOT / f\"videos/observation.images.{camera}/chunk-000\").glob(\"*.mp4\"))
    for video_path in video_paths:
        command = [
            \"ffmpeg\", \"-v\", \"error\", \"-i\", str(video_path),
            \"-vf\", f\"fps=1,scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT}\",
            \"-f\", \"rawvideo\", \"-pix_fmt\", \"rgb24\", \"pipe:1\",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = SAMPLE_WIDTH * SAMPLE_HEIGHT * 3
        previous_gray = None
        local_second = 0
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(SAMPLE_HEIGHT, SAMPLE_WIDTH, 3).copy()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            temporal_mad = np.nan if previous_gray is None else float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))))
            rows.append({
                \"camera\": camera,
                \"video_file\": video_path.name,
                \"global_second\": global_second,
                \"local_second\": local_second,
                \"brightness\": float(gray.mean()),
                \"contrast\": float(gray.std()),
                \"saturation\": float(hsv[:, :, 1].mean()),
                \"laplacian_variance\": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                \"dark_pixel_share\": float(np.mean(gray < 10)),
                \"bright_pixel_share\": float(np.mean(gray > 245)),
                \"temporal_mad_1s\": temporal_mad,
            })
            if global_second in SAMPLE_SECONDS:
                selected_frames[global_second] = rgb
            previous_gray = gray
            global_second += 1
            local_second += 1
        stderr = process.stderr.read().decode(\"utf-8\", errors=\"replace\")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f\"ffmpeg failed for {video_path}: {stderr}\")
    return pd.DataFrame(rows), selected_frames

with ThreadPoolExecutor(max_workers=3) as executor:
    scan_results = list(executor.map(scan_camera, CAMERA_KEYS))

video_metrics = pd.concat([result[0] for result in scan_results], ignore_index=True)
contact_frames = {camera: result[1] for camera, result in zip(CAMERA_KEYS, scan_results)}

video_summary_rows = []
for camera, group in video_metrics.groupby(\"camera\"):
    valid_temporal = group[\"temporal_mad_1s\"].dropna()
    video_summary_rows.append({
        \"camera\": camera,
        \"sampled_frames\": len(group),
        \"brightness_p01\": float(group[\"brightness\"].quantile(0.01)),
        \"brightness_median\": float(group[\"brightness\"].median()),
        \"brightness_p99\": float(group[\"brightness\"].quantile(0.99)),
        \"contrast_median\": float(group[\"contrast\"].median()),
        \"laplacian_p01\": float(group[\"laplacian_variance\"].quantile(0.01)),
        \"laplacian_median\": float(group[\"laplacian_variance\"].median()),
        \"near_black_share\": float(np.mean((group[\"brightness\"] < 20) | (group[\"dark_pixel_share\"] > 0.90))),
        \"overexposed_share\": float(np.mean((group[\"brightness\"] > 235) | (group[\"bright_pixel_share\"] > 0.90))),
        \"near_duplicate_1s_share\": float(np.mean(valid_temporal < 0.5)),
        \"low_motion_1s_share\": float(np.mean(valid_temporal < 2.0)),
    })
video_summary = pd.DataFrame(video_summary_rows).sort_values(\"camera\")
display(video_summary)

thumb_width, thumb_height = SAMPLE_WIDTH, SAMPLE_HEIGHT
label_height = 22
ordered_seconds = sorted(SAMPLE_SECONDS)
sheet = Image.new(\"RGB\", (len(ordered_seconds) * thumb_width, len(CAMERA_KEYS) * (thumb_height + label_height)), \"white\")
draw = ImageDraw.Draw(sheet)
for row_index, camera in enumerate(CAMERA_KEYS):
    for column_index, second in enumerate(ordered_seconds):
        frame = contact_frames[camera].get(second)
        x = column_index * thumb_width
        y = row_index * (thumb_height + label_height)
        if frame is not None:
            sheet.paste(Image.fromarray(frame), (x, y))
        draw.text((x + 3, y + thumb_height + 3), f\"{camera} t={second}s\", fill=\"black\")
sheet.save(AUDIT_DIR / \"video_contact_sheet.jpg\", quality=92)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
for ax, metric, title in zip(
    axes,
    [\"brightness\", \"laplacian_variance\", \"temporal_mad_1s\"],
    [\"Brightness\", \"Sharpness proxy\", \"One-second frame change\"],
):
    grouped = [video_metrics.loc[video_metrics.camera == camera, metric].dropna().to_numpy() for camera in CAMERA_KEYS]
    ax.boxplot(grouped, tick_labels=CAMERA_KEYS, showfliers=False)
    ax.set_title(title)
    ax.tick_params(axis=\"x\", rotation=25)
fig.tight_layout()
fig.savefig(AUDIT_DIR / \"video_quality_distributions.png\", dpi=160)
plt.show()
"""
    )
)

cells.append(nbf.v4.new_markdown_cell("### 10. Save compact, inspectable evidence for the technical report"))
cells.append(
    nbf.v4.new_code_cell(
        """best_offset_row = offset_alignment.loc[offset_alignment[\"normalized_rmse\"].idxmin()]
finite_velocity_corr = velocity_consistency[\"correlation\"].dropna()

audit_results = {
    \"dataset\": {
        \"episodes\": int(len(episodes)),
        \"frames\": int(len(frames)),
        \"hours\": float(len(frames) / FPS / 3600),
        \"fps\": FPS,
        \"tasks\": int(len(tasks)),
        \"task_text\": str(tasks.index[0]),
        \"train_episodes\": int(eval_episode_start),
        \"eval_episodes\": int(n_eval),
        \"train_frames\": int(is_train_frame.sum()),
        \"eval_frames\": int(is_eval_frame.sum()),
    },
    \"integrity\": {
        \"checks\": int(len(integrity)),
        \"failed_checks\": int((~integrity[\"passed\"]).sum()),
        \"timestamp_max_error_s\": float(timestamp_max_error),
    },
    \"constant_observation_dimensions\": {
        \"count\": int(len(constant_observation)),
        \"dimensions\": (constant_observation[\"feature\"] + \"/\" + constant_observation[\"joint\"]).tolist(),
        \"total_scalar_observation_dimensions\": 42,
        \"normalizer_epsilon\": 1e-8,
    },
    \"motion\": {row[\"metric\"]: float(row[\"share\"]) for row in motion_summary.to_dict(\"records\")},
    \"velocity_consistency\": {
        \"median_finite_correlation\": float(finite_velocity_corr.median()),
        \"minimum_finite_correlation\": float(finite_velocity_corr.min()),
        \"zero_variance_joints\": velocity_consistency.loc[velocity_consistency[\"correlation\"].isna(), \"joint\"].tolist(),
    },
    \"action_state_alignment\": {
        \"best_future_offset_frames\": int(best_offset_row[\"future_offset_frames\"]),
        \"best_future_offset_ms\": float(best_offset_row[\"future_offset_ms\"]),
        \"best_normalized_rmse\": float(best_offset_row[\"normalized_rmse\"]),
        \"current_state_normalized_rmse\": float(offset_alignment.iloc[0][\"normalized_rmse\"]),
    },
    \"split_drift\": {
        \"max_abs_episode_mean_smd\": float(split_drift[\"abs_smd\"].max()),
        \"dimensions_abs_smd_gt_0_5\": int((split_drift[\"abs_smd\"] > 0.5).sum()),
        \"dimensions_abs_smd_gt_1\": int((split_drift[\"abs_smd\"] > 1.0).sum()),
        \"largest_dimensions\": split_drift.head(8).to_dict(\"records\"),
    },
    \"episode_duplication\": {
        \"closest_pair\": [int(closest_pair[0]), int(closest_pair[1])],
        \"closest_pair_normalized_rmse\": float(pairwise_rmse[closest_pair]),
        \"median_nearest_normalized_rmse\": float(np.median(nearest_distance)),
        \"episodes_with_nearest_rmse_below_0_05\": int(np.sum(nearest_distance < 0.05)),
    },
    \"video_summary\": video_summary.to_dict(\"records\"),
}

(AUDIT_DIR / \"audit_results.json\").write_text(json.dumps(audit_results, indent=2, ensure_ascii=False))
integrity.to_csv(AUDIT_DIR / \"integrity_checks.csv\", index=False)
episode_summary.to_csv(AUDIT_DIR / \"episode_summary.csv\", index=False)
scalar_profile.to_csv(AUDIT_DIR / \"scalar_profile.csv\", index=False)
velocity_consistency.to_csv(AUDIT_DIR / \"velocity_consistency.csv\", index=False)
offset_alignment.to_csv(AUDIT_DIR / \"action_state_alignment.csv\", index=False)
split_drift.to_csv(AUDIT_DIR / \"train_eval_split_drift.csv\", index=False)
video_metrics.to_csv(AUDIT_DIR / \"video_sample_metrics.csv\", index=False)
video_summary.to_csv(AUDIT_DIR / \"video_summary.csv\", index=False)

display(pd.DataFrame([audit_results[\"dataset\"]]))
print(f\"Saved audit evidence to {AUDIT_DIR}\")
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Takeaways

After execution, use `audit_results.json` and the bounded CSV outputs to write the reader-facing conclusions. Structural validity alone does not establish deployment fitness. The final decision must also consider zero-variance normalized inputs, chronological split drift, visual-domain coverage, trajectory diversity, and whether real-robot observations obey the same units and zero conventions.
"""
    )
)

nb["cells"] = cells
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
