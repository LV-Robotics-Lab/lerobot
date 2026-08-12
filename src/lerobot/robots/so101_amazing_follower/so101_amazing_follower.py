from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..so_follower import SOFollower, SOFollowerConfig, SOFollowerRobotConfig
from .config_so101_amazing_follower import (
    AmazingHandAttachmentConfig,
    BiSO101AmazingFollowerConfig,
    SO101AmazingFollowerConfig,
)


def _arm_config(
    config: SOFollowerConfig,
    *,
    robot_id: str | None,
    calibration_dir: Path,
    cameras: dict[str, Any] | None = None,
) -> SOFollowerRobotConfig:
    return SOFollowerRobotConfig(
        id=robot_id,
        calibration_dir=calibration_dir,
        port=config.port,
        disable_torque_on_disconnect=config.disable_torque_on_disconnect,
        max_relative_target=config.max_relative_target,
        cameras=config.cameras if cameras is None else cameras,
        use_degrees=config.use_degrees,
        use_gripper=False,
        position_p_coefficient=config.position_p_coefficient,
        position_i_coefficient=config.position_i_coefficient,
        position_d_coefficient=config.position_d_coefficient,
        num_read_retries=config.num_read_retries,
    )


def _make_hand(
    config: AmazingHandAttachmentConfig,
    *,
    default_calibration_file: Path,
) -> Any:
    try:
        from amazinghand_wrapper import AmazingHandConfig, AmazingHandController, LeRobotFeetechBackend
    except ImportError as error:
        raise ImportError(
            "AmazingHand support requires the pinned submodule: "
            "python -m pip install -e submodules/amazinghand_wrapper"
        ) from error

    calibration_file = Path(config.calibration_file) if config.calibration_file else default_calibration_file
    motor_ids = tuple(config.motor_ids)
    wrapper_config = AmazingHandConfig(
        port=config.port,
        calibration_file=calibration_file,
        baudrates=tuple(config.baudrates),
        expected_model_numbers=tuple(config.expected_model_numbers),
        motor_ids=motor_ids,
        leader_open_value=config.leader_open_value,
        leader_closed_value=config.leader_closed_value,
        max_raw_velocity=config.max_raw_velocity,
        command_timeout_s=config.command_timeout_s,
        max_temperature_c=config.max_temperature_c,
        max_abs_load=config.max_abs_load,
    )
    return AmazingHandController(wrapper_config, LeRobotFeetechBackend(motor_ids))


class SO101AmazingFollower(Robot):
    config_class = SO101AmazingFollowerConfig
    name = "so101_amazing_follower"

    def __init__(self, config: SO101AmazingFollowerConfig):
        super().__init__(config)
        self.config = config
        self.arm = SOFollower(
            _arm_config(
                config.arm_config,
                robot_id=config.id,
                calibration_dir=self.calibration_dir,
            )
        )
        calibration_stem = config.id or self.name
        self.hand = _make_hand(
            config.hand_config,
            default_calibration_file=self.calibration_dir / f"{calibration_stem}_amazing_hand.json",
        )
        self.cameras = self.arm.cameras

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {**self.arm.observation_features, "gripper.pos": float}
        if self.config.hand_config.include_motor_observations:
            features.update({f"hand_{index}.pos": float for index in range(1, 9)})
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self.arm.action_features, "gripper.pos": float}

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self.hand.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated and self.hand.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected_arm = False
        connected_hand = False
        try:
            self.arm.connect(calibrate)
            connected_arm = True
            self.hand.connect()
            connected_hand = True
            if not self.hand.is_calibrated:
                if not calibrate:
                    return
                self.hand.calibrate_interactive()
            self.hand.activate()
        except Exception:
            if connected_hand:
                self.hand.disconnect()
            if connected_arm:
                self.arm.disconnect()
            raise

    def calibrate(self) -> None:
        self.arm.calibrate()
        if not self.hand.is_connected:
            raise RuntimeError("connect the AmazingHand before calibration")
        self.hand.calibrate_interactive()

    def configure(self) -> None:
        self.arm.configure()

    def setup_motors(self) -> None:
        self.arm.setup_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation = dict(self.arm.get_observation())
        hand_observation = self.hand.observe()
        observation["gripper.pos"] = hand_observation.grasp_closure
        if self.config.hand_config.include_motor_observations:
            observation.update(
                {
                    f"hand_{index}.pos": hand_observation.motor_closure[name]
                    for index, name in enumerate(hand_observation.motor_closure, start=1)
                }
            )
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.hand.is_active:
            raise RuntimeError("AmazingHand is connected but not active")
        arm_action = dict(action)
        gripper = arm_action.pop("gripper.pos", None)
        if gripper is None:
            raise ValueError("Missing gripper.pos for AmazingHand control")
        sent = dict(self.arm.send_action(arm_action))
        self.hand.command_grasp(float(gripper))
        sent["gripper.pos"] = float(gripper)
        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.hand.disconnect()
        finally:
            self.arm.disconnect()


class BiSO101AmazingFollower(BimanualMixin, Robot):
    config_class = BiSO101AmazingFollowerConfig
    name = "bi_so101_amazing_follower"

    def __init__(self, config: BiSO101AmazingFollowerConfig):
        super().__init__(config)
        self.config = config
        self._top_level_cam_keys = set(config.cameras)
        collisions = self._top_level_cam_keys & (
            set(config.left_arm_config.cameras) | set(config.right_arm_config.cameras)
        )
        if collisions:
            raise ValueError(f"Top-level camera names collide with per-arm cameras: {sorted(collisions)}")
        left_arm_config = SOFollowerConfig(
            **{
                **config.left_arm_config.__dict__,
                "cameras": {**config.left_arm_config.cameras, **config.cameras},
            }
        )
        self.left_arm = SO101AmazingFollower(
            SO101AmazingFollowerConfig(
                id=f"{config.id}_left" if config.id else None,
                calibration_dir=config.calibration_dir,
                arm_config=left_arm_config,
                hand_config=config.left_hand_config,
            )
        )
        self.right_arm = SO101AmazingFollower(
            SO101AmazingFollowerConfig(
                id=f"{config.id}_right" if config.id else None,
                calibration_dir=config.calibration_dir,
                arm_config=config.right_arm_config,
                hand_config=config.right_hand_config,
            )
        )
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected: list[SO101AmazingFollower] = []
        try:
            for robot in (self.left_arm, self.right_arm):
                robot.connect(calibrate)
                connected.append(robot)
        except Exception:
            for robot in reversed(connected):
                robot.disconnect()
            raise

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        output: dict[str, type | tuple] = {}
        for key, value in self.left_arm.observation_features.items():
            output[key if key in self._top_level_cam_keys else f"left_{key}"] = value
        output.update({f"right_{key}": value for key, value in self.right_arm.observation_features.items()})
        return output

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = {}
        for key, value in self.left_arm.get_observation().items():
            observation[key if key in self._top_level_cam_keys else f"left_{key}"] = value
        observation.update(
            {f"right_{key}": value for key, value in self.right_arm.get_observation().items()}
        )
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        sent_left = self.left_arm.send_action(left)
        sent_right = self.right_arm.send_action(right)
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def disconnect(self) -> None:
        errors: list[Exception] = []
        for robot in (self.right_arm, self.left_arm):
            try:
                robot.disconnect()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"bimanual disconnect failed: {errors}")
