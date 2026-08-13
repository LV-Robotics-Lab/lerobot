# Local robot data

This directory is the workstation-local boundary for calibration, captured
datasets, camera checks, and operator notes. Its contents are intentionally
ignored by Git because they may be large or tied to one physical rig.

The Jingxiang SO-101/XLeRobot workstation uses this layout:

```text
local_data/so101/
├── calibration/
├── datasets/
├── outputs/
├── manifests/
└── port-map.txt
```

Reusable source, tests, and documentation belong elsewhere in the LeRobot
repository. Model or dataset publication should use Hugging Face or object
storage with a manifest and hashes rather than committing payloads here.
