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

import logging
from functools import cached_property

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..so_follower import SOFollower, SOFollowerRobotConfig
from .amazing_hand import AmazingHand
from .config_bi_so_follower import BiSOFollowerConfig

logger = logging.getLogger(__name__)


class BiSOFollower(BimanualMixin, Robot):
    """
    [Bimanual SO Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BiSOFollowerConfig
    name = "bi_so_follower"

    def __init__(self, config: BiSOFollowerConfig):
        super().__init__(config)
        self.config = config

        # Top-level cameras are opened by `left_arm` for convenience, but their
        # keys stay unprefixed in observations (tracked via `_top_level_cam_keys`).
        self._top_level_cam_keys = set(config.cameras)
        _collisions = self._top_level_cam_keys & set(
            config.left_arm_config.cameras
        ) | self._top_level_cam_keys & set(config.right_arm_config.cameras)
        if _collisions:
            raise ValueError(
                f"Top-level camera names collide with per-arm camera names: {sorted(_collisions)}"
            )
        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}

        left_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.left_arm_config.max_relative_target,
            use_degrees=config.left_arm_config.use_degrees,
            use_gripper=not config.left_hand_config.port and config.left_arm_config.use_gripper,
            cameras=left_arm_cameras,
        )

        right_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.right_arm_config.max_relative_target,
            use_degrees=config.right_arm_config.use_degrees,
            use_gripper=not config.right_hand_config.port and config.right_arm_config.use_gripper,
            cameras=config.right_arm_config.cameras,
        )

        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)
        calibration_stem = config.id or "bi_so_follower"
        self.left_hand = (
            AmazingHand(
                config.left_hand_config,
                self.calibration_dir / f"{calibration_stem}_left_amazing_hand.json",
            )
            if config.left_hand_config.port
            else None
        )
        self.right_hand = (
            AmazingHand(
                config.right_hand_config,
                self.calibration_dir / f"{calibration_stem}_right_amazing_hand.json",
            )
            if config.right_hand_config.port
            else None
        )

        # Only for compatibility with other parts of the codebase that expect a `robot.cameras` attribute
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = dict(self.left_arm._motors_ft)
        right_arm_motors_ft = dict(self.right_arm._motors_ft)

        if self.left_hand is not None:
            left_arm_motors_ft["gripper.pos"] = float
            left_arm_motors_ft.update(self.left_hand.observation_features)
        if self.right_hand is not None:
            right_arm_motors_ft["gripper.pos"] = float
            right_arm_motors_ft.update(self.right_hand.observation_features)

        return {
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for k, v in self.left_arm._cameras_ft.items():
            out[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm._cameras_ft.items():
            out[f"right_{k}"] = v
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        left = dict(self.left_arm._motors_ft)
        right = dict(self.right_arm._motors_ft)
        if self.left_hand is not None:
            left["gripper.pos"] = float
        if self.right_hand is not None:
            right["gripper.pos"] = float
        return {
            **{f"left_{key}": value for key, value in left.items()},
            **{f"right_{key}": value for key, value in right.items()},
        }

    @property
    def is_connected(self) -> bool:
        hands_connected = (self.left_hand is None or self.left_hand.is_connected) and (
            self.right_hand is None or self.right_hand.is_connected
        )
        return self.left_arm.is_connected and self.right_arm.is_connected and hands_connected

    @property
    def is_calibrated(self) -> bool:
        hands_calibrated = (self.left_hand is None or self.left_hand.is_calibrated) and (
            self.right_hand is None or self.right_hand.is_calibrated
        )
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated and hands_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected = []
        try:
            for device in (self.left_arm, self.right_arm, self.left_hand, self.right_hand):
                if device is not None:
                    device.connect(calibrate)
                    connected.append(device)
        except Exception:
            for device in reversed(connected):
                device.disconnect()
            raise

    def calibrate(self) -> None:
        for device in (self.left_arm, self.right_arm, self.left_hand, self.right_hand):
            if device is not None:
                device.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}

        # Add "left_" prefix to per-arm keys; keep top-level camera keys unprefixed.
        for key, value in self.left_arm.get_observation().items():
            obs_dict[key if key in self._top_level_cam_keys else f"left_{key}"] = value
        if self.left_hand is not None:
            hand_obs = self.left_hand.get_observation()
            obs_dict.update({f"left_{key}": value for key, value in hand_obs.items()})
            obs_dict["left_gripper.pos"] = sum(hand_obs.values()) / len(hand_obs)

        # Add "right_" prefix
        for key, value in self.right_arm.get_observation().items():
            obs_dict[f"right_{key}"] = value
        if self.right_hand is not None:
            hand_obs = self.right_hand.get_observation()
            obs_dict.update({f"right_{key}": value for key, value in hand_obs.items()})
            obs_dict["right_gripper.pos"] = sum(hand_obs.values()) / len(hand_obs)

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        left_gripper = left_action.pop("gripper.pos", None) if self.left_hand is not None else None
        right_gripper = right_action.pop("gripper.pos", None) if self.right_hand is not None else None

        sent_action_left = dict(self.left_arm.send_action(left_action))
        sent_action_right = dict(self.right_arm.send_action(right_action))

        if self.left_hand is not None:
            if left_gripper is None:
                raise ValueError("Missing left_gripper.pos for AmazingHand control")
            self.left_hand.send_gripper(float(left_gripper))
            sent_action_left["gripper.pos"] = float(left_gripper)
        if self.right_hand is not None:
            if right_gripper is None:
                raise ValueError("Missing right_gripper.pos for AmazingHand control")
            self.right_hand.send_gripper(float(right_gripper))
            sent_action_right["gripper.pos"] = float(right_gripper)

        # Add prefixes back
        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}

        return {**prefixed_sent_action_left, **prefixed_sent_action_right}

    @check_if_not_connected
    def disconnect(self) -> None:
        for device in (self.right_hand, self.left_hand, self.right_arm, self.left_arm):
            if device is not None and device.is_connected:
                device.disconnect()
