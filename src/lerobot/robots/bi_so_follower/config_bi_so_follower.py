#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..so_follower import SOFollowerConfig


@dataclass
class AmazingHandConfig:
    """Configuration for one Pollen Robotics AmazingHand serial bus."""

    # An empty port disables the hand on that side.
    port: str = ""
    baudrate: int = 250_000
    leader_open_value: float = 100.0
    leader_closed_value: float = 0.0
    max_raw_step: int = 12
    disable_torque_on_disconnect: bool = True
    calibration_file: str = ""
    open_positions: list[int] = field(default_factory=list)
    closed_positions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if self.leader_open_value == self.leader_closed_value:
            raise ValueError("leader_open_value and leader_closed_value must differ")
        if self.max_raw_step <= 0:
            raise ValueError("max_raw_step must be positive")
        if bool(self.open_positions) != bool(self.closed_positions):
            raise ValueError("open_positions and closed_positions must be provided together")
        for name, positions in (
            ("open_positions", self.open_positions),
            ("closed_positions", self.closed_positions),
        ):
            if positions and len(positions) != 8:
                raise ValueError(f"{name} must contain exactly 8 raw servo positions")
            if positions and any(not 0 <= value <= 1023 for value in positions):
                raise ValueError(f"{name} values must be within the SCS0009 range [0, 1023]")


@RobotConfig.register_subclass("bi_so_follower")
@dataclass
class BiSOFollowerConfig(RobotConfig):
    """Configuration class for Bi SO Follower robots."""

    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig
    left_hand_config: AmazingHandConfig = field(default_factory=AmazingHandConfig)
    right_hand_config: AmazingHandConfig = field(default_factory=AmazingHandConfig)

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
