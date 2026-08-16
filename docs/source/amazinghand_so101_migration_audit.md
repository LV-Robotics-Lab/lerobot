# AmazingHand + SO-101 migration audit

Audit date: 2026-08-13

## Source provenance

The implementation was recovered from both of the following remote snapshots:

- `LV-Robotics-Lab/lerobot`, branch `hardware/dm-robotics`, commit
  `1a0f42d7799d6dbdfb6c5df2ee1c2a53a8a5552e`.
- `LV-Robotics-Lab/dm-lerobot-standalone-backup`, commit
  `fd0198a9b013b5f2b67e341e12f4ae5ff885c3f2`.

Their Git tree IDs are byte-identical:
`0cd8f74ea541e6530be1d0f770ac6b7899b7cbbd`.

The original snapshot recorded its pre-import AmazingHand commit as
`72d8d0370b395e37bec8d0476372a52ce8ff9120`.

## Migration mapping

| Previous path                      | New owner                                                              |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `bi_so_follower/amazing_hand.py`   | `submodules/amazinghand_wrapper/src/amazinghand_wrapper/`              |
| `AmazingHandConfig`                | wrapper `AmazingHandConfig` plus LeRobot `AmazingHandAttachmentConfig` |
| scalar-to-eight-motor synergy      | wrapper `GripperSynergyMapper`                                         |
| open/closed JSON calibration       | wrapper `HandCalibration` with atomic writes and schema ID             |
| bimanual routing in `BiSOFollower` | explicit `BiSO101AmazingFollower`                                      |
| missing single-arm composition     | new `SO101AmazingFollower`                                             |
| SO follower `use_gripper` patch    | current `SOFollowerConfig.use_gripper`                                 |
| five mock tests                    | wrapper safety tests plus LeRobot single/bimanual contract tests       |

The original behavior is retained, while baud probing, accepted model-number
configuration, stale-command rejection, time-based velocity limiting,
temperature/load checks, atomic calibration, single-arm support, and explicit
fault state are added.

## Host audit

Host `jingxiang@100.64.0.6` was inspected before migration. No AmazingHand
source, documentation, archive, or calibration was found under the user's home,
workspace, Desktop, Documents, or Downloads. The only SO-101 repository was
`workspace/agenticsim-runtime/candidates/lerobot_so101_teleop`, an Isaac/AgenticSim
candidate unrelated to physical AmazingHand control, so it was left untouched.

Desktop was empty. Documents contained only empty application directories and a
workspace symlink. Downloads contained simulation reports, model-comparison
archives, and application installers unrelated to this integration. No item was
deleted or imported because none was a LeRobot/AmazingHand source asset or a
verified duplicate of this repository.

## Evidence boundary

Software tests use fake/mocked buses only. This audit does not claim a successful
real serial probe, calibration, motion command, dataset recording, or soak test.
Physical validation must follow the fail-closed sequence in
`amazinghand_so101.mdx`.
