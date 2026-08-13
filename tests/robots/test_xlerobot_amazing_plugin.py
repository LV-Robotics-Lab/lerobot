from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lerobot_robot_xlerobot_amazing import (
    XLerobotAmazingFollower,
    XLerobotAmazingFollowerConfig,
    XLerobotBiSOLeader,
    XLerobotBiSOLeaderConfig,
)
from lerobot_robot_xlerobot_amazing._upstream import upstream, xlerobot_root

from lerobot.robots.so101_amazing_follower import AmazingHandAttachmentConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig


def _hand() -> MagicMock:
    hand = MagicMock()
    hand.is_connected = True
    hand.is_calibrated = True
    hand.is_active = True
    hand.observe.return_value.grasp_closure = 42.0
    hand.observe.return_value.motor_closure = {
        f"motor_{index}": float(index) for index in range(1, 9)
    }
    return hand


def _robot_config(tmp_path: Path) -> XLerobotAmazingFollowerConfig:
    return XLerobotAmazingFollowerConfig(
        id="xle",
        calibration_dir=tmp_path,
        port1="/dev/left-head",
        port2="/dev/right-base",
        left_hand_config=AmazingHandAttachmentConfig(port="/dev/left-hand"),
        right_hand_config=AmazingHandAttachmentConfig(port="/dev/right-hand"),
    )


def _fake_upstream_init(robot, config) -> None:
    robot.config = config
    robot.id = config.id
    robot.calibration_dir = Path(config.calibration_dir)
    robot.bus1 = MagicMock(
        motors={
            "left_arm_shoulder_pan": object(),
            "left_arm_gripper": object(),
            "head_motor_1": object(),
        },
        calibration={"left_arm_gripper": object()},
        is_connected=True,
    )
    robot.bus2 = MagicMock(
        motors={
            "right_arm_shoulder_pan": object(),
            "right_arm_gripper": object(),
            "base_left_wheel": object(),
        },
        calibration={"right_arm_gripper": object()},
        is_connected=True,
    )
    robot.left_arm_motors = ["left_arm_shoulder_pan", "left_arm_gripper"]
    robot.right_arm_motors = ["right_arm_shoulder_pan", "right_arm_gripper"]
    robot.head_motors = ["head_motor_1"]
    robot.base_motors = ["base_left_wheel"]
    robot.cameras = {}


def _robot(tmp_path: Path) -> XLerobotAmazingFollower:
    left_hand = _hand()
    right_hand = _hand()
    with (
        patch(
            "lerobot_robot_xlerobot_amazing.robot.XLerobot.__init__",
            new=_fake_upstream_init,
        ),
        patch(
            "lerobot_robot_xlerobot_amazing.robot._make_hand",
            side_effect=[left_hand, right_hand],
        ),
    ):
        return XLerobotAmazingFollower(_robot_config(tmp_path))


def test_xlerobot_is_loaded_from_the_pinned_submodule():
    assert xlerobot_root().name == "xlerobot"
    assert (xlerobot_root() / "software" / "src" / "robots" / "xlerobot").is_dir()
    assert upstream.__name__ == "lerobot.robots.xlerobot"


def test_robot_replaces_only_stock_grippers(tmp_path):
    robot = _robot(tmp_path)

    assert "left_arm_gripper" not in robot.bus1.motors
    assert "right_arm_gripper" not in robot.bus2.motors
    assert robot.left_arm_motors == ["left_arm_shoulder_pan"]
    assert robot.right_arm_motors == ["right_arm_shoulder_pan"]
    assert "head_motor_1" in robot.bus1.motors
    assert "base_left_wheel" in robot.bus2.motors


def test_robot_routes_grippers_to_hands_and_preserves_xlerobot_schema(tmp_path):
    robot = _robot(tmp_path)
    action = {
        "left_arm_shoulder_pan.pos": 1.0,
        "left_arm_gripper.pos": 60.0,
        "right_arm_shoulder_pan.pos": 2.0,
        "right_arm_gripper.pos": 70.0,
        "x.vel": 0.1,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }

    with patch.object(upstream.XLerobot, "send_action", return_value={"x.vel": 0.1}) as send:
        sent = robot.send_action(action)

    body_action = send.call_args.args[1]
    assert "left_arm_gripper.pos" not in body_action
    assert "right_arm_gripper.pos" not in body_action
    robot.left_hand.command_grasp.assert_called_once_with(60.0)
    robot.right_hand.command_grasp.assert_called_once_with(70.0)
    assert sent["left_arm_gripper.pos"] == 60.0
    assert sent["right_arm_gripper.pos"] == 70.0

    with patch.object(upstream.XLerobot, "get_observation", return_value={"x.vel": 0.0}):
        observation = robot.get_observation()
    assert observation["left_arm_gripper.pos"] == 42.0
    assert observation["right_arm_gripper.pos"] == 42.0


def test_calibrate_cli_can_connect_with_hands_inactive(tmp_path):
    robot = _robot(tmp_path)
    robot.bus1.is_connected = False
    robot.bus2.is_connected = False
    for hand in (robot.left_hand, robot.right_hand):
        hand.is_connected = False
        hand.is_calibrated = False
        hand.connect.side_effect = lambda current=hand: setattr(current, "is_connected", True)

    def connect_body(current, calibrate):
        assert calibrate is False
        current.bus1.is_connected = True
        current.bus2.is_connected = True

    with patch.object(upstream.XLerobot, "connect", new=connect_body):
        robot.connect(calibrate=False)

    assert robot.is_connected
    assert not robot.is_calibrated
    robot.left_hand.activate.assert_not_called()
    robot.right_hand.activate.assert_not_called()


def test_optional_motor_diagnostics_are_declared_and_observed(tmp_path):
    config = _robot_config(tmp_path)
    config.left_hand_config.include_motor_observations = True
    left_hand = _hand()
    right_hand = _hand()
    with (
        patch(
            "lerobot_robot_xlerobot_amazing.robot.XLerobot.__init__",
            new=_fake_upstream_init,
        ),
        patch(
            "lerobot_robot_xlerobot_amazing.robot._make_hand",
            side_effect=[left_hand, right_hand],
        ),
    ):
        robot = XLerobotAmazingFollower(config)

    assert "left_hand_8.pos" in robot.observation_features
    assert "right_hand_8.pos" not in robot.observation_features
    with patch.object(upstream.XLerobot, "get_observation", return_value={}):
        observation = robot.get_observation()
    assert observation["left_hand_1.pos"] == 1.0
    assert observation["left_hand_8.pos"] == 8.0


def test_command_failure_stops_base_and_faults_both_hands(tmp_path):
    robot = _robot(tmp_path)
    robot.right_hand.command_grasp.side_effect = RuntimeError("write failed")
    action = {
        "left_arm_gripper.pos": 50.0,
        "right_arm_gripper.pos": 50.0,
    }

    with (
        patch.object(upstream.XLerobot, "send_action", return_value={}),
        patch.object(robot, "stop_base") as stop_base,
        pytest.raises(RuntimeError, match="write failed"),
    ):
        robot.send_action(action)

    stop_base.assert_called_once()
    robot.left_hand.emergency_stop.assert_called_once()
    robot.right_hand.emergency_stop.assert_called_once()


def _teleop_config(*, control_base: bool) -> XLerobotBiSOLeaderConfig:
    return XLerobotBiSOLeaderConfig(
        id="leaders",
        left_arm_config=SOLeaderConfig(port="/dev/left-leader"),
        right_arm_config=SOLeaderConfig(port="/dev/right-leader"),
        control_base=control_base,
    )


def test_composite_teleoperator_maps_bimanual_arm_names():
    teleop = object.__new__(XLerobotBiSOLeader)
    teleop.config = _teleop_config(control_base=False)
    teleop.arms = MagicMock(is_connected=True, is_calibrated=True)
    teleop.arms.action_features = {
        "left_shoulder_pan.pos": float,
        "left_gripper.pos": float,
        "right_shoulder_pan.pos": float,
        "right_gripper.pos": float,
    }
    teleop.arms.get_action.return_value = dict.fromkeys(teleop.arms.action_features, 25.0)
    teleop.keyboard = None
    teleop._speed_index = 0
    teleop._last_pressed = set()

    action = teleop.get_action()

    assert set(action) == {
        "left_arm_shoulder_pan.pos",
        "left_arm_gripper.pos",
        "right_arm_shoulder_pan.pos",
        "right_arm_gripper.pos",
    }


def test_composite_teleoperator_can_add_fail_stop_base_velocity():
    teleop = object.__new__(XLerobotBiSOLeader)
    teleop.config = _teleop_config(control_base=True)
    teleop.arms = MagicMock(is_connected=True, is_calibrated=True)
    teleop.arms.get_action.return_value = {
        "left_gripper.pos": 10.0,
        "right_gripper.pos": 20.0,
    }
    teleop.keyboard = MagicMock(is_connected=True)
    teleop.keyboard.get_action.side_effect = [{"i": None, "n": None}, {}]
    teleop._speed_index = 0
    teleop._last_pressed = set()

    moving = teleop.get_action()
    stopped = teleop.get_action()

    assert moving["x.vel"] == 0.2
    assert moving["y.vel"] == 0.0
    assert stopped["x.vel"] == 0.0
    assert stopped["theta.vel"] == 0.0
