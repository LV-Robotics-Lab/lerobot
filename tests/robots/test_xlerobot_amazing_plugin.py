from pathlib import Path
from types import SimpleNamespace
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


def _hand(calibration_file: Path | None = None) -> MagicMock:
    hand = MagicMock()
    hand.config = SimpleNamespace(
        calibration_file=calibration_file or Path("unused-amazinghand-calibration.json"),
        motor_ids=tuple(range(1, 9)),
    )
    hand.backend = MagicMock()
    hand.calibration = SimpleNamespace(schema="lv_robotics.amazinghand_calibration.v1")
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
    robot.calibration_fpath = robot.calibration_dir / f"{config.id}.json"
    robot.calibration = {
        "left_arm_shoulder_pan": SimpleNamespace(id=1),
        "left_arm_gripper": SimpleNamespace(id=6),
        "head_motor_1": SimpleNamespace(id=7),
        "right_arm_shoulder_pan": SimpleNamespace(id=1),
        "right_arm_gripper": SimpleNamespace(id=6),
        "base_left_wheel": SimpleNamespace(id=7),
    }
    robot.bus1 = MagicMock(
        motors={
            "left_arm_shoulder_pan": SimpleNamespace(id=1),
            "left_arm_gripper": SimpleNamespace(id=6),
            "head_motor_1": SimpleNamespace(id=7),
        },
        calibration={"left_arm_gripper": robot.calibration["left_arm_gripper"]},
        is_connected=True,
        is_calibrated=True,
    )
    robot.bus2 = MagicMock(
        motors={
            "right_arm_shoulder_pan": SimpleNamespace(id=1),
            "right_arm_gripper": SimpleNamespace(id=6),
            "base_left_wheel": SimpleNamespace(id=7),
        },
        calibration={"right_arm_gripper": robot.calibration["right_arm_gripper"]},
        is_connected=True,
        is_calibrated=True,
    )
    robot.left_arm_motors = ["left_arm_shoulder_pan", "left_arm_gripper"]
    robot.right_arm_motors = ["right_arm_shoulder_pan", "right_arm_gripper"]
    robot.head_motors = ["head_motor_1"]
    robot.base_motors = ["base_left_wheel"]
    robot.cameras = {}


def _robot(tmp_path: Path) -> XLerobotAmazingFollower:
    left_hand = _hand(tmp_path / "xle_left_amazing_hand.json")
    right_hand = _hand(tmp_path / "xle_right_amazing_hand.json")
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


def _prepare_connect(robot: XLerobotAmazingFollower, tmp_path: Path) -> None:
    (tmp_path / "xle.json").write_text("{}")
    (tmp_path / "xle_left_amazing_hand.json").write_text("{}")
    (tmp_path / "xle_right_amazing_hand.json").write_text("{}")
    robot.config.calibration_provenance_verified = True

    for bus in (robot.bus1, robot.bus2):
        bus.is_connected = False
        bus.connect.side_effect = lambda current=bus: setattr(current, "is_connected", True)
        bus.disconnect.side_effect = lambda _disable_torque, current=bus: setattr(
            current, "is_connected", False
        )
    for hand in (robot.left_hand, robot.right_hand):
        hand.is_connected = False
        hand.is_active = False
        hand.connect.side_effect = lambda current=hand: setattr(current, "is_connected", True)
        hand.activate.side_effect = lambda current=hand: setattr(current, "is_active", True)
        hand.disconnect.side_effect = lambda current=hand: setattr(current, "is_connected", False)
        hand.backend.read_positions.return_value = dict.fromkeys(hand.config.motor_ids, 512)


@pytest.mark.parametrize("calibrate", [False, True])
def test_noninteractive_connect_fails_closed_before_io_when_calibration_is_missing(tmp_path, calibrate):
    robot = _robot(tmp_path)
    robot.bus1.is_connected = False
    robot.bus2.is_connected = False
    for hand in (robot.left_hand, robot.right_hand):
        hand.is_connected = False

    with (
        patch("builtins.input", side_effect=AssertionError("connect must not read stdin")),
        pytest.raises(RuntimeError, match="body calibration file.*xle.json"),
    ):
        robot.connect(calibrate=calibrate)

    robot.bus1.connect.assert_not_called()
    robot.bus2.connect.assert_not_called()
    robot.left_hand.connect.assert_not_called()
    robot.right_hand.connect.assert_not_called()


def test_explicit_interactive_calibration_mode_is_torque_off_even_with_complete_calibration(tmp_path):
    robot = _robot(tmp_path)
    _prepare_connect(robot, tmp_path)
    robot.config.allow_interactive_calibration = True

    with patch("builtins.input", side_effect=AssertionError("connect must not read stdin")):
        robot.connect(calibrate=False)

    assert robot.is_connected
    robot.bus1.disable_torque.assert_called_once()
    robot.bus2.disable_torque.assert_called_once()
    robot.bus1.write_calibration.assert_not_called()
    robot.bus2.write_calibration.assert_not_called()
    robot.bus1.configure_motors.assert_not_called()
    robot.bus2.configure_motors.assert_not_called()
    robot.bus1.enable_torque.assert_not_called()
    robot.bus2.enable_torque.assert_not_called()
    robot.left_hand.activate.assert_not_called()
    robot.right_hand.activate.assert_not_called()


def test_runtime_requires_external_calibration_provenance_before_io(tmp_path):
    robot = _robot(tmp_path)
    _prepare_connect(robot, tmp_path)
    robot.config.calibration_provenance_verified = False

    with pytest.raises(RuntimeError, match="external calibration provenance verification"):
        robot.connect(calibrate=False)

    robot.bus1.connect.assert_not_called()
    robot.bus2.connect.assert_not_called()


def test_runtime_rejects_unsupported_hand_calibration_schema_before_io(tmp_path):
    robot = _robot(tmp_path)
    _prepare_connect(robot, tmp_path)
    robot.left_hand.calibration.schema = "unknown.v0"

    with pytest.raises(RuntimeError, match="unsupported schema 'unknown.v0'"):
        robot.connect(calibrate=False)

    robot.bus1.connect.assert_not_called()
    robot.left_hand.connect.assert_not_called()


def test_noninteractive_connect_rejects_body_calibration_for_other_motor_ids(tmp_path):
    robot = _robot(tmp_path)
    _prepare_connect(robot, tmp_path)
    robot.calibration["head_motor_1"] = SimpleNamespace(id=8)

    with (
        patch("builtins.input", side_effect=AssertionError("connect must not read stdin")),
        pytest.raises(RuntimeError, match="motor IDs do not match robot id 'xle'"),
    ):
        robot.connect(calibrate=False)

    robot.bus1.connect.assert_not_called()
    robot.left_hand.connect.assert_not_called()


def test_noninteractive_connect_latches_all_goals_before_any_torque_enable(tmp_path):
    robot = _robot(tmp_path)
    _prepare_connect(robot, tmp_path)
    events: list[tuple[str, object]] = []

    for label, bus in (("bus1", robot.bus1), ("bus2", robot.bus2)):
        bus.connect.side_effect = lambda current=bus, current_label=label: (
            events.append((f"{current_label}_connect", None)),
            setattr(current, "is_connected", True),
        )
        bus.disable_torque.side_effect = lambda current_label=label: events.append(
            (f"{current_label}_disable", None)
        )

    robot.bus1.sync_read.side_effect = lambda register, motors: (
        events.append(("bus1_read", tuple(motors)))
        or {name: float(index) for index, name in enumerate(motors)}
    )
    robot.bus2.sync_read.side_effect = lambda register, motors: (
        events.append(("bus2_read", tuple(motors)))
        or {name: float(index) for index, name in enumerate(motors)}
    )
    robot.bus1.sync_write.side_effect = lambda register, values, **_kwargs: events.append(
        ("bus1_goal", (register, dict(values)))
    )
    robot.bus2.sync_write.side_effect = lambda register, values, **_kwargs: events.append(
        ("bus2_goal", (register, dict(values)))
    )
    robot.bus1.enable_torque.side_effect = lambda: events.append(("bus1_enable", None))
    robot.bus2.enable_torque.side_effect = lambda: events.append(("bus2_enable", None))

    for side, hand in (("left", robot.left_hand), ("right", robot.right_hand)):
        hand.backend.write_positions.side_effect = lambda values, current_side=side: events.append(
            (f"{current_side}_hand_goal", dict(values))
        )
        hand.activate.side_effect = lambda current=hand, current_side=side: (
            events.append((f"{current_side}_hand_enable", None)),
            setattr(current, "is_active", True),
        )

    with patch("builtins.input", side_effect=AssertionError("connect must not read stdin")):
        robot.connect(calibrate=False)

    labels = [label for label, _payload in events]
    assert labels[:4] == ["bus1_connect", "bus1_disable", "bus2_connect", "bus2_disable"]
    first_enable = min(index for index, label in enumerate(labels) if label.endswith("enable"))
    for required in ("bus1_goal", "bus2_goal", "left_hand_goal", "right_hand_goal"):
        assert labels.index(required) < first_enable

    base_stop_index = next(
        index
        for index, event in enumerate(events)
        if event
        == (
            "bus2_goal",
            ("Goal_Velocity", {"base_left_wheel": 0}),
        )
    )
    assert base_stop_index < first_enable
    assert events[labels.index("bus1_goal")][1] == (
        "Goal_Position",
        {"left_arm_shoulder_pan": 0.0, "head_motor_1": 1.0},
    )
    assert events[labels.index("bus2_goal")][1] == (
        "Goal_Position",
        {"right_arm_shoulder_pan": 0.0},
    )
    assert robot.is_connected
    assert robot.left_hand.is_active
    assert robot.right_hand.is_active


def test_configure_failure_forces_torque_off_before_disconnect_even_when_disabled_by_config(
    tmp_path,
):
    robot = _robot(tmp_path)
    _prepare_connect(robot, tmp_path)
    robot.config.disable_torque_on_disconnect = False
    events: list[str] = []

    robot.bus1.sync_read.return_value = {
        "left_arm_shoulder_pan": 0.0,
        "head_motor_1": 0.0,
    }
    robot.bus2.sync_read.return_value = {"right_arm_shoulder_pan": 0.0}
    for label, bus in (("bus1", robot.bus1), ("bus2", robot.bus2)):
        bus.disable_torque.side_effect = lambda current_label=label: events.append(
            f"{current_label}_disable"
        )
        bus.disconnect.side_effect = lambda disable, current=bus, current_label=label: (
            events.append(f"{current_label}_disconnect_{disable}"),
            setattr(current, "is_connected", False),
        )
    robot.bus1.enable_torque.side_effect = lambda: events.append("bus1_enable")

    def fail_bus2_enable() -> None:
        events.append("bus2_enable_attempt")
        raise RuntimeError("enable failed")

    robot.bus2.enable_torque.side_effect = fail_bus2_enable

    with pytest.raises(RuntimeError, match="enable failed"):
        robot.connect(calibrate=False)

    assert events.index("bus1_enable") < max(
        index for index, event in enumerate(events) if event == "bus1_disable"
    )
    assert events.index("bus2_enable_attempt") < max(
        index for index, event in enumerate(events) if event == "bus2_disable"
    )
    robot.bus1.disconnect.assert_called_once_with(True)
    robot.bus2.disconnect.assert_called_once_with(True)


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
