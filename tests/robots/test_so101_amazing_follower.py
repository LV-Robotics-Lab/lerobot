from unittest.mock import MagicMock, patch

from lerobot.robots.so101_amazing_follower import (
    AmazingHandAttachmentConfig,
    BiSO101AmazingFollower,
    BiSO101AmazingFollowerConfig,
    SO101AmazingFollower,
    SO101AmazingFollowerConfig,
)
from lerobot.robots.so_follower import SOFollowerConfig


def _arm_mock() -> MagicMock:
    arm = MagicMock()
    arm.action_features = {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
    }
    arm.observation_features = dict(arm.action_features)
    arm.cameras = {}
    arm.is_connected = True
    arm.is_calibrated = True
    arm.get_observation.return_value = dict.fromkeys(arm.observation_features, 0.0)
    arm.send_action.side_effect = lambda action: action
    return arm


def _hand_mock() -> MagicMock:
    hand = MagicMock()
    hand.is_connected = True
    hand.is_calibrated = True
    hand.is_active = True
    hand.observe.return_value.grasp_closure = 35.0
    hand.observe.return_value.motor_closure = {f"motor_{index}": float(index) for index in range(1, 9)}
    return hand


def _single_config(tmp_path, robot_id="single") -> SO101AmazingFollowerConfig:
    return SO101AmazingFollowerConfig(
        id=robot_id,
        calibration_dir=tmp_path,
        arm_config=SOFollowerConfig(port="/dev/arm"),
        hand_config=AmazingHandAttachmentConfig(port="/dev/hand"),
    )


def test_single_robot_preserves_six_dimensional_so101_contract(tmp_path):
    arm = _arm_mock()
    hand = _hand_mock()
    with (
        patch(
            "lerobot.robots.so101_amazing_follower.so101_amazing_follower.SOFollower",
            return_value=arm,
        ),
        patch(
            "lerobot.robots.so101_amazing_follower.so101_amazing_follower._make_hand",
            return_value=hand,
        ),
    ):
        robot = SO101AmazingFollower(_single_config(tmp_path))

    assert len(robot.action_features) == 6
    assert len(robot.observation_features) == 6
    action = {**dict.fromkeys(arm.action_features, 1.0), "gripper.pos": 60.0}
    assert robot.send_action(action) == action
    hand.command_grasp.assert_called_once_with(60.0)
    assert robot.get_observation()["gripper.pos"] == 35.0


def test_single_robot_can_expose_motor_diagnostics_without_changing_actions(tmp_path):
    arm = _arm_mock()
    hand = _hand_mock()
    config = _single_config(tmp_path)
    config.hand_config.include_motor_observations = True
    with (
        patch(
            "lerobot.robots.so101_amazing_follower.so101_amazing_follower.SOFollower",
            return_value=arm,
        ),
        patch(
            "lerobot.robots.so101_amazing_follower.so101_amazing_follower._make_hand",
            return_value=hand,
        ),
    ):
        robot = SO101AmazingFollower(config)

    assert len(robot.action_features) == 6
    assert len(robot.observation_features) == 14
    assert len(robot.get_observation()) == 14


def test_bimanual_robot_routes_two_independent_grippers(tmp_path):
    left = MagicMock()
    right = MagicMock()
    features = {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
        "gripper.pos": float,
    }
    for robot in (left, right):
        robot.action_features = dict(features)
        robot.observation_features = dict(features)
        robot.cameras = {}
        robot.is_connected = True
        robot.is_calibrated = True
        robot.send_action.side_effect = lambda action: action
        robot.get_observation.return_value = dict.fromkeys(features, 0.0)

    config = BiSO101AmazingFollowerConfig(
        id="dual",
        calibration_dir=tmp_path,
        left_arm_config=SOFollowerConfig(port="/dev/left-arm"),
        right_arm_config=SOFollowerConfig(port="/dev/right-arm"),
        left_hand_config=AmazingHandAttachmentConfig(port="/dev/left-hand"),
        right_hand_config=AmazingHandAttachmentConfig(port="/dev/right-hand"),
    )
    with patch(
        "lerobot.robots.so101_amazing_follower.so101_amazing_follower.SO101AmazingFollower",
        side_effect=[left, right],
    ):
        robot = BiSO101AmazingFollower(config)

    action = {
        **{f"left_{key}": 1.0 for key in features},
        **{f"right_{key}": 2.0 for key in features},
    }
    assert robot.send_action(action) == action
    assert len(robot.action_features) == 12
    assert len(robot.observation_features) == 12
