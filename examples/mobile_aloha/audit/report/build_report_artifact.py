from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = REPORT_DIR.parent


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


results = read_json(AUDIT_DIR / "audit_results.json")
episodes = read_csv(AUDIT_DIR / "episode_summary.csv")
alignment = read_csv(AUDIT_DIR / "action_state_alignment.csv")
drift = read_csv(AUDIT_DIR / "train_eval_split_drift.csv")
video = read_csv(AUDIT_DIR / "video_summary.csv")
velocity = read_csv(AUDIT_DIR / "velocity_consistency.csv")

duration_rows = [
    {
        "episode": int(row["episode_index"]),
        "duration_s": round(float(row["length"]) / results["dataset"]["fps"], 1),
        "split": "train" if int(row["episode_index"]) < 91 else "eval",
    }
    for row in episodes
]

alignment_rows = [
    {
        "offset_frames": int(row["future_offset_frames"]),
        "offset_ms": int(float(row["future_offset_ms"])),
        "normalized_rmse": float(row["normalized_rmse"]),
    }
    for row in alignment
]

drift_rows = sorted(
    [
        {
            "dimension": f"{row['feature']}/{row['joint']}",
            "abs_smd": abs(float(row["standardized_mean_difference"])),
            "signed_smd": float(row["standardized_mean_difference"]),
        }
        for row in drift
    ],
    key=lambda item: item["abs_smd"],
    reverse=True,
)[:12]

constant_rows = [
    {
        "dimension": name,
        "risk": "runtime non-zero value is amplified by MEAN_STD normalization",
        "severity": "High",
    }
    for name in results["constant_observation_dimensions"]["dimensions"]
]

video_rows = [
    {
        "camera": row["camera"],
        "sampled_frames": int(row["sampled_frames"]),
        "brightness_median": round(float(row["brightness_median"]), 2),
        "sharpness_p01": round(float(row["laplacian_p01"]), 2),
        "low_motion_share": float(row["low_motion_1s_share"]),
        "near_black_share": float(row["near_black_share"]),
        "overexposed_share": float(row["overexposed_share"]),
        "near_duplicate_share": float(row["near_duplicate_1s_share"]),
    }
    for row in video
]

velocity_scale = []
for row in velocity:
    reported = float(row["reported_velocity_std"])
    finite = float(row["finite_difference_std"])
    if finite > 0 and reported > 0:
        velocity_scale.append(reported / finite)
median_velocity_scale = sorted(velocity_scale)[len(velocity_scale) // 2]

finding_rows = [
    {
        "priority": "P0",
        "severity": "High",
        "finding": "14/42 scalar observation dimensions have zero training variance",
        "evidence": "Two gripper velocities and twelve effort channels are constant; MEAN_STD uses epsilon 1e-8.",
        "action": "Remove unsupported fields and retrain, or force exactly zero temporarily with runtime assertions.",
    },
    {
        "priority": "P0",
        "severity": "High",
        "finding": "No untouched test set or real task-success metric",
        "evidence": "The last 11 episodes are validation data and were reused for checkpoint selection.",
        "action": "Split by collection session into train/validation/test and keep test sealed until final evaluation.",
    },
    {
        "priority": "P0",
        "severity": "High",
        "finding": "Visual and task coverage is narrow",
        "evidence": "0.684 h, one task, one fixed setup; image transforms disabled in both training stages.",
        "action": "Collect variation in lighting, camera pose, objects, dustpan pose, operator, session, and speed.",
    },
    {
        "priority": "P1",
        "severity": "Medium",
        "finding": "Success/completion/occlusion labels are absent",
        "evidence": "24 sampled start/end pairs look successful, but two terminal eval views are materially occluded.",
        "action": "Annotate success, completion frame, failure reason, and occlusion; trim post-completion tails.",
    },
    {
        "priority": "P1",
        "severity": "Medium",
        "finding": "Action stream contains long low-motion regions",
        "evidence": "21.10% of consecutive actions change by less than 1e-4; 84/102 episodes are exactly 22 s.",
        "action": "Audit padding and post-success idle frames, then trim or mask them consistently.",
    },
    {
        "priority": "P1",
        "severity": "Medium",
        "finding": "Velocity semantics must match deployment",
        "evidence": f"Median reported/finite-difference velocity scale is about {median_velocity_scale:.2f}; gripper velocities are always zero.",
        "action": "Match source, units, filtering, and timestamps on robot; log and compare before enabling motors.",
    },
    {
        "priority": "P1",
        "severity": "Medium",
        "finding": "Learned action target leads observed state by about 60 ms",
        "evidence": "Best normalized action/state RMSE occurs at +3 frames at 50 Hz.",
        "action": "Match the 50 Hz control loop, queue behavior, sensor freshness, and latency on robot.",
    },
]

training_rows = [
    {
        "stage": "Expert-only",
        "selected_checkpoint": "2,500",
        "configured_steps": "10,000",
        "batch_size": 64,
        "learning_rate": 0.0001,
        "vision_encoder": "frozen",
        "image_transforms": "disabled",
    },
    {
        "stage": "Joint",
        "selected_checkpoint": "1,500",
        "configured_steps": "3,000",
        "batch_size": 16,
        "learning_rate": 0.00002,
        "vision_encoder": "trainable",
        "image_transforms": "disabled",
    },
]

overview = [
    {
        "hours": results["dataset"]["hours"],
        "structural_pass_rate": 1.0,
        "zero_variance_share": results["constant_observation_dimensions"]["count"]
        / results["constant_observation_dimensions"]["total_scalar_observation_dimensions"],
        "eval_episode_share": results["dataset"]["eval_episodes"] / results["dataset"]["episodes"],
    }
]

generated_at = datetime.now(UTC).isoformat()
sources = [
    {
        "id": "audit_synthesis",
        "label": "Reproduction SQL over packaged SmolVLA audit evidence",
        "path": "evidence/audit_sources.sql",
        "query": {
            "engine": "python",
            "language": "python",
            "description": "Reproducible structural, scalar, trajectory, split, and video audit.",
            "tables_used": [
                "evidence/integrity_checks.csv",
                "evidence/episode_summary.csv",
                "evidence/scalar_profile.csv",
                "evidence/velocity_consistency.csv",
                "evidence/action_state_alignment.csv",
                "evidence/train_eval_split_drift.csv",
                "evidence/video_summary.csv",
            ],
        },
    },
    {
        "id": "training_configs",
        "label": "Expert and joint training configurations",
        "path": "evidence/audit_sources.sql",
        "query": {
            "tables_used": [
                "evidence/expert_stage_train_config.json",
                "evidence/joint_stage_train_config.json",
            ]
        },
    },
    {
        "id": "deployment_manifest",
        "label": "Deployment bundle manifest",
        "path": "evidence/deployment_manifest.json",
    },
    {
        "id": "endpoint_review",
        "label": "Representative episode start/end review sheet",
        "path": "evidence/episode_start_end_contact_sheet.jpg",
    },
]

for source in sources:
    source.pop("query", None)

cards = [
    {
        "id": "hours",
        "dataset": "overview",
        "sourceId": "audit_synthesis",
        "metrics": [{"label": "Recorded hours", "field": "hours", "format": "number"}],
    },
    {
        "id": "integrity",
        "dataset": "overview",
        "sourceId": "audit_synthesis",
        "metrics": [{"label": "Structural pass rate", "field": "structural_pass_rate", "format": "percent"}],
    },
    {
        "id": "zero_variance",
        "dataset": "overview",
        "sourceId": "audit_synthesis",
        "metrics": [
            {"label": "Zero-variance scalar inputs", "field": "zero_variance_share", "format": "percent"}
        ],
    },
    {
        "id": "eval_share",
        "dataset": "overview",
        "sourceId": "audit_synthesis",
        "metrics": [
            {"label": "Validation episode share", "field": "eval_episode_share", "format": "percent"}
        ],
    },
]

charts = [
    {
        "id": "episode_duration",
        "title": "Episode duration by episode",
        "subtitle": "Durations are concentrated at three exact horizons; validation is the final 11 episodes.",
        "intent": "distribution",
        "question": "Are episode lengths naturally varied or dominated by fixed cutoffs?",
        "rationale": "A per-episode bar chart exposes repeated capture horizons and the temporal split boundary.",
        "type": "bar",
        "dataset": "episode_duration",
        "sourceId": "audit_synthesis",
        "encodings": {
            "x": {"field": "episode", "type": "ordinal", "label": "Episode"},
            "y": {"field": "duration_s", "type": "quantitative", "label": "Duration", "unit": "s"},
            "color": {"field": "split", "type": "nominal", "label": "Split"},
        },
        "xAxisTitle": "Episode index",
        "yAxisTitle": "Duration (s)",
        "layout": "full",
        "surface": {"viewMode": "visualization", "showControls": True},
    },
    {
        "id": "split_drift",
        "title": "Train/validation feature drift",
        "subtitle": "Seven scalar dimensions exceed an absolute episode-mean SMD of 0.5.",
        "intent": "comparison",
        "question": "Does the chronological validation tail match the training distribution?",
        "rationale": "Absolute SMD is scale-free and makes the largest distribution shifts comparable across joints.",
        "type": "horizontalBar",
        "dataset": "split_drift",
        "sourceId": "audit_synthesis",
        "encodings": {
            "x": {"field": "dimension", "type": "nominal", "label": "Dimension"},
            "y": {"field": "abs_smd", "type": "quantitative", "label": "Absolute SMD"},
        },
        "referenceLines": [
            {"axis": "y", "value": 0.5, "label": "moderate drift", "color": "orange", "lineStyle": "dashed"}
        ],
        "layout": "full",
        "surface": {"viewMode": "visualization", "showControls": True},
    },
    {
        "id": "action_alignment",
        "title": "Action–state alignment by future offset",
        "subtitle": "Normalized RMSE reaches its minimum at +60 ms.",
        "intent": "relationship",
        "question": "What sensor-to-action timing is encoded in the demonstrations?",
        "rationale": "The line shape directly shows the empirical error minimum without implying causal latency.",
        "type": "line",
        "dataset": "action_alignment",
        "sourceId": "audit_synthesis",
        "encodings": {
            "x": {"field": "offset_ms", "type": "quantitative", "label": "Future offset", "unit": "ms"},
            "y": {"field": "normalized_rmse", "type": "quantitative", "label": "Normalized RMSE"},
        },
        "xAxisTitle": "Future state offset (ms)",
        "yAxisTitle": "Normalized RMSE (lower is better)",
        "referenceLines": [
            {"axis": "x", "value": 60, "label": "minimum", "color": "green", "lineStyle": "dashed"}
        ],
        "layout": "full",
        "surface": {"viewMode": "visualization", "showControls": True},
    },
]

tables = [
    {
        "id": "findings",
        "title": "Risk register and required remediation",
        "dataset": "findings",
        "sourceId": "audit_synthesis",
        "defaultSort": {"field": "priority", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "priority", "label": "Priority", "type": "text"},
            {"field": "severity", "label": "Severity", "type": "text"},
            {"field": "finding", "label": "Finding", "type": "text"},
            {"field": "evidence", "label": "Evidence", "type": "text"},
            {"field": "action", "label": "Required action", "type": "text"},
        ],
    },
    {
        "id": "constant_dimensions",
        "title": "Constant scalar inputs present in the policy observation",
        "dataset": "constant_dimensions",
        "sourceId": "audit_synthesis",
        "defaultSort": {"field": "dimension", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "dimension", "label": "Dimension", "type": "text"},
            {"field": "severity", "label": "Severity", "type": "text"},
            {"field": "risk", "label": "Deployment risk", "type": "text"},
        ],
    },
    {
        "id": "video_quality",
        "title": "Video integrity at one-second sampling",
        "dataset": "video_quality",
        "sourceId": "audit_synthesis",
        "defaultSort": {"field": "camera", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "camera", "label": "Camera", "type": "text"},
            {"field": "sampled_frames", "label": "Samples", "format": "number"},
            {"field": "brightness_median", "label": "Median brightness", "format": "number"},
            {"field": "sharpness_p01", "label": "Sharpness p01", "format": "number"},
            {"field": "low_motion_share", "label": "Low-motion share", "format": "percent"},
            {"field": "near_black_share", "label": "Near-black", "format": "percent"},
            {"field": "overexposed_share", "label": "Overexposed", "format": "percent"},
            {"field": "near_duplicate_share", "label": "Near-duplicate", "format": "percent"},
        ],
    },
    {
        "id": "training_flow",
        "title": "Training stages and selected checkpoints",
        "dataset": "training_flow",
        "sourceId": "training_configs",
        "defaultSort": {"field": "stage", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "stage", "label": "Stage", "type": "text"},
            {"field": "selected_checkpoint", "label": "Selected checkpoint", "type": "text"},
            {"field": "configured_steps", "label": "Configured steps", "type": "text"},
            {"field": "batch_size", "label": "Batch size", "format": "number"},
            {"field": "learning_rate", "label": "Learning rate", "format": "number"},
            {"field": "vision_encoder", "label": "Vision encoder", "type": "text"},
            {"field": "image_transforms", "label": "Image transforms", "type": "text"},
        ],
    },
]

blocks = [
    {
        "id": "title",
        "type": "markdown",
        "body": "# SmolVLA Training-Data Quality Audit\n\nTechnical audit for real-robot deployment readiness. Generated from the executed local dataset and training artifacts.",
    },
    {
        "id": "summary",
        "type": "markdown",
        "body": "## Technical summary\n\n**Overall assessment: Needs revision.** The dataset is structurally valid and the sampled demonstrations appear task-correct, but it is not qualified as evidence of reliable real-robot performance. The leading deployment risk is a scalar-input contract mismatch: 14 of 42 observation dimensions have zero variance yet remain in MEAN_STD normalization. The evaluation design also lacks an untouched test set, and the visual/task domain is too narrow to support robust deployment claims.",
    },
    {
        "id": "metrics",
        "type": "metric-strip",
        "cardIds": ["hours", "integrity", "zero_variance", "eval_share"],
        "sourceId": "audit_synthesis",
    },
    {
        "id": "decision",
        "type": "markdown",
        "body": "## Deployment decision\n\nDo not use the current validation loss or episode-91 MAE as a production go/no-go signal. Before motor-enabled testing, run a shadow-mode input-contract check on the robot and either remove the constant fields and retrain or guarantee their exact training-time values with hard assertions. Treat the current model as an experiment checkpoint, not a deployment-qualified policy.",
    },
    {"id": "risk_table", "type": "table", "tableId": "findings", "sourceId": "audit_synthesis"},
    {
        "id": "zero_var_note",
        "type": "markdown",
        "body": "## Finding 1 — normalization contract can fail catastrophically\n\nTwo gripper-velocity fields and twelve effort fields are exactly constant. The local normalizer computes (value − mean) / (std + 1e-8); therefore a real-robot deviation of only 0.1 in a constant field can become roughly 10,000,000 normalized units. This is a plausible direct cause of poor real-robot behavior, but it remains conditional until a real input log is captured.",
    },
    {
        "id": "zero_var_table",
        "type": "table",
        "tableId": "constant_dimensions",
        "sourceId": "audit_synthesis",
    },
    {
        "id": "split_note",
        "type": "markdown",
        "body": "## Finding 2 — evaluation is validation, not independent testing\n\nThe loader holds out the final ceil(102 × 0.1) = 11 episodes. Those episodes were repeatedly evaluated while selecting checkpoints, so reported eval loss 0.1123 and episode-91 MAE 0.0283 are validation metrics. They estimate fit to this collection tail, not real-world task success. Seven dimensions show absolute episode-mean SMD above 0.5 between training and validation.",
    },
    {"id": "split_chart", "type": "chart", "chartId": "split_drift", "sourceId": "audit_synthesis"},
    {
        "id": "coverage_note",
        "type": "markdown",
        "body": "## Finding 3 — coverage is too narrow for robust deployment\n\nThe dataset contains 0.684 recorded hours, one task, one physical setup, and tightly clustered camera brightness. Image transforms were disabled in both training stages, while the vision encoder was unfrozen during joint fine-tuning. This combination can specialize the model to the recording domain. The endpoint review of 24 episodes was visually successful, but two evaluation endpoints were materially occluded and the dataset contains no explicit success labels.",
    },
    {"id": "video_table", "type": "table", "tableId": "video_quality", "sourceId": "audit_synthesis"},
    {
        "id": "duration_note",
        "type": "markdown",
        "body": "## Finding 4 — fixed horizons and idle regions need trimming review\n\nEpisode durations occur at only 36 s, 34 s, and 22 s; 84 of 102 episodes are exactly 22 s. In addition, 21.10% of consecutive action rows are nearly unchanged. These patterns are not proof of bad demonstrations, but they are consistent with fixed capture windows, padding, or post-completion idle segments that can dilute behavior learning.",
    },
    {"id": "duration_chart", "type": "chart", "chartId": "episode_duration", "sourceId": "audit_synthesis"},
    {
        "id": "timing_note",
        "type": "markdown",
        "body": "## Finding 5 — deployment timing must reproduce demonstration timing\n\nAction vectors match observed state best three frames later: 60 ms at 50 Hz. This supports a coherent action/state relationship, but it also makes loop rate, sensor freshness, queue length, and reset behavior part of the learned interface. The evidence is correlational and does not by itself identify the physical control latency.",
    },
    {"id": "alignment_chart", "type": "chart", "chartId": "action_alignment", "sourceId": "audit_synthesis"},
    {
        "id": "scope",
        "type": "markdown",
        "body": "## Scope, data, and metric definitions\n\nAudited scope: LeRobot v3 dataset with 102 episodes, 123,100 frames, 50 Hz, three 640×480 RGB video streams, 14-D state/action, 14-D velocity, 14-D effort, and 2-D base action. Task text: sweep the small objects into the dustpan. SMD is standardized mean difference of episode-level feature means. Normalized RMSE scales each joint by its dataset standard deviation. Video diagnostics sample one frame per second per camera.",
    },
    {
        "id": "method",
        "type": "markdown",
        "body": "## Methodology\n\nThe executed notebook checks parquet schemas, episode/frame continuity, timestamps, task mapping, vector dimensions and finiteness, video frame counts and decoding, scalar variance, action/state alignment, velocity consistency, chronological split drift, trajectory similarity, and sampled video integrity. Independent spot checks also used the real LeRobotDataset decode path at the first frame, validation boundary, and final frame. A 24-episode contact sheet covers 13 representative training episodes and all 11 validation episodes.",
    },
    {"id": "training", "type": "table", "tableId": "training_flow", "sourceId": "training_configs"},
    {
        "id": "limitations",
        "type": "markdown",
        "body": "## Limitations, uncertainty, and robustness\n\nNo real-robot telemetry was supplied, so camera ordering, scalar units, freshness, and runtime normalization cannot yet be compared directly. Video quality was exhaustively checked at one-second sampling rather than every frame. Semantic success was manually reviewed on 24 endpoint pairs, not all 102 complete trajectories. The dataset contains no session, operator, success, failure, completion-frame, or occlusion labels, preventing session-grouped leakage checks and automatic outcome certification.",
    },
    {
        "id": "next",
        "type": "markdown",
        "body": "## Recommended next steps\n\n1. **P0 — shadow log:** record 2–5 minutes on the robot with motors disabled; run the exact preprocessor and assert camera order, RGB range, 50 Hz freshness, units, and normalized scalar magnitudes (target mostly within ±5).\n2. **P0 — observation contract:** remove unsupported constant dimensions and retrain. A temporary bridge may force only those 14 fields to exact zero, with a hard runtime assertion.\n3. **P0 — evaluation redesign:** split by collection session into train/validation/untouched test and add per-episode real success.\n4. **P1 — broaden coverage:** vary lighting, camera pose, object layout, dustpan pose, operator, speed, and collection day; add conservative train-only image augmentation.\n5. **P1 — clean temporal labels:** annotate success/completion/occlusion and trim or mask padding and post-success idle.\n6. **P1 — match control semantics:** verify velocity source/filtering, 50 Hz loop, roughly 60 ms action/state relationship, action queue, and reset behavior.",
    },
    {
        "id": "questions",
        "type": "markdown",
        "body": "## Further questions\n\n- Do real-robot effort and gripper-velocity channels remain exactly zero after preprocessing?\n- Are the deployed camera keys, ordering, crop, color space, and timestamps identical to training?\n- Were episodes collected across multiple sessions, and can those groups be recovered for a leakage-safe split?\n- What fraction of episodes remains successful under altered lighting, object placement, and dustpan pose?\n- Does action-queue state reset cleanly at episode start and after intervention?",
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "SmolVLA Training-Data Quality Audit",
        "description": "Technical audit of training-data and evaluation readiness for real-robot deployment.",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "overview": overview,
            "episode_duration": duration_rows,
            "split_drift": drift_rows,
            "action_alignment": alignment_rows,
            "constant_dimensions": constant_rows,
            "video_quality": video_rows,
            "findings": finding_rows,
            "training_flow": training_rows,
        },
    },
    "sources": sources,
}

(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
)
(REPORT_DIR / "evidence" / "chart_map.json").write_text(
    json.dumps(
        {
            chart["id"]: {
                "question": chart["question"],
                "intent": chart["intent"],
                "rationale": chart["rationale"],
                "dataset": chart["dataset"],
            }
            for chart in charts
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(REPORT_DIR / "artifact.json")
