#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_so_follower import AmazingHandConfig, BiSOFollower, BiSOFollowerConfig
from lerobot.robots.bi_so_follower.amazing_hand import AMAZING_HAND_MOTORS, GripperSynergyMapper
from lerobot.robots.so_follower import SOFollowerConfig


def _calibration() -> tuple[dict[str, int], dict[str, int]]:
    open_raw = {motor: 100 + index * 20 for index, motor in enumerate(AMAZING_HAND_MOTORS)}
    closed_raw = {
        motor: open_raw[motor] + (200 if index % 2 == 0 else -80)
        for index, motor in enumerate(AMAZING_HAND_MOTORS)
    }
    return open_raw, closed_raw


def test_gripper_synergy_maps_endpoints_and_midpoint():
    open_raw, closed_raw = _calibration()
    mapper = GripperSynergyMapper(open_raw, closed_raw, max_raw_step=500)

    assert mapper.targets(100.0) == open_raw
    assert mapper.targets(0.0) == closed_raw
    assert mapper.targets(50.0) == {
        motor: round((open_raw[motor] + closed_raw[motor]) / 2) for motor in AMAZING_HAND_MOTORS
    }


def test_gripper_synergy_clips_input_and_rate_limits_targets():
    open_raw, closed_raw = _calibration()
    mapper = GripperSynergyMapper(open_raw, closed_raw, max_raw_step=12)

    assert mapper.targets(150.0) == open_raw
    assert mapper.targets(-50.0) == closed_raw

    limited = mapper.targets(0.0, previous_raw=open_raw)
    for motor in AMAZING_HAND_MOTORS:
        expected_delta = 12 if closed_raw[motor] > open_raw[motor] else -12
        assert limited[motor] == open_raw[motor] + expected_delta


def test_gripper_synergy_reports_normalized_per_motor_closure():
    open_raw, closed_raw = _calibration()
    mapper = GripperSynergyMapper(open_raw, closed_raw)

    for motor in AMAZING_HAND_MOTORS:
        midpoint = round((open_raw[motor] + closed_raw[motor]) / 2)
        assert mapper.motor_closure(motor, open_raw[motor]) == 0.0
        assert mapper.motor_closure(motor, closed_raw[motor]) == 100.0
        assert mapper.motor_closure(motor, midpoint) == pytest.approx(50.0, abs=0.7)


def test_amazing_hand_config_rejects_incomplete_calibration():
    with pytest.raises(ValueError, match="provided together"):
        AmazingHandConfig(port="/dev/null", open_positions=[0] * 8)
    with pytest.raises(ValueError, match="exactly 8"):
        AmazingHandConfig(port="/dev/null", open_positions=[0] * 7, closed_positions=[1] * 7)


def _arm_mock(side: str) -> MagicMock:
    arm = MagicMock(name=f"{side}_arm")
    arm._motors_ft = {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
    }
    arm.cameras = {}
    arm.is_connected = True
    arm.is_calibrated = True
    arm.get_observation.return_value = dict.fromkeys(arm._motors_ft, 0.0)
    arm.send_action.side_effect = lambda action: action
    return arm


def _hand_mock(side: str) -> MagicMock:
    hand = MagicMock(name=f"{side}_hand")
    hand.observation_features = {f"hand_{motor}.pos": float for motor in AMAZING_HAND_MOTORS}
    hand.is_connected = True
    hand.is_calibrated = True
    hand.get_observation.return_value = {f"hand_{motor}.pos": 25.0 for motor in AMAZING_HAND_MOTORS}
    return hand


def test_bimanual_robot_routes_scalar_grippers_to_independent_hands(tmp_path):
    arms = [_arm_mock("left"), _arm_mock("right")]
    hands = [_hand_mock("left"), _hand_mock("right")]
    config = BiSOFollowerConfig(
        id="amazing",
        calibration_dir=tmp_path,
        left_arm_config=SOFollowerConfig(port="/dev/left-arm"),
        right_arm_config=SOFollowerConfig(port="/dev/right-arm"),
        left_hand_config=AmazingHandConfig(port="/dev/left-hand"),
        right_hand_config=AmazingHandConfig(port="/dev/right-hand"),
    )

    with (
        patch(
            "lerobot.robots.bi_so_follower.bi_so_follower.SOFollower",
            side_effect=arms,
        ),
        patch(
            "lerobot.robots.bi_so_follower.bi_so_follower.AmazingHand",
            side_effect=hands,
        ),
    ):
        robot = BiSOFollower(config)

    action = {
        **{f"left_{key}": 1.0 for key in arms[0]._motors_ft},
        **{f"right_{key}": 2.0 for key in arms[1]._motors_ft},
        "left_gripper.pos": 20.0,
        "right_gripper.pos": 80.0,
    }
    returned = robot.send_action(action)

    hands[0].send_gripper.assert_called_once_with(20.0)
    hands[1].send_gripper.assert_called_once_with(80.0)
    assert "gripper.pos" not in arms[0].send_action.call_args.args[0]
    assert "gripper.pos" not in arms[1].send_action.call_args.args[0]
    assert returned == action

    observation = robot.get_observation()
    assert observation["left_gripper.pos"] == 25.0
    assert observation["right_gripper.pos"] == 25.0
    assert len(robot.action_features) == 12
    assert len(robot.observation_features) == 28
