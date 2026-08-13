from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.bi_so_leader import BiSOLeader
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import XLerobotBiSOLeaderConfig


def _to_xlerobot_arm_key(key: str) -> str:
    if key.startswith("left_"):
        return f"left_arm_{key.removeprefix('left_')}"
    if key.startswith("right_"):
        return f"right_arm_{key.removeprefix('right_')}"
    raise ValueError(f"Unexpected bimanual SO leader key: {key}")


def _to_bimanual_leader_key(key: str) -> str | None:
    if key.startswith("left_arm_"):
        return f"left_{key.removeprefix('left_arm_')}"
    if key.startswith("right_arm_"):
        return f"right_{key.removeprefix('right_arm_')}"
    return None


class XLerobotBiSOLeader(Teleoperator):
    """Two SO-101 leaders plus optional keyboard control for the XLeRobot base."""

    config_class = XLerobotBiSOLeaderConfig
    name = "xlerobot_bi_so_leader"

    def __init__(self, config: XLerobotBiSOLeaderConfig):
        super().__init__(config)
        self.config = config
        self.arms = BiSOLeader(config.arms_config())
        self.keyboard = (
            KeyboardTeleop(
                KeyboardTeleopConfig(
                    id=f"{config.id}_base" if config.id else None,
                    calibration_dir=config.calibration_dir,
                )
            )
            if config.control_base
            else None
        )
        self._speed_index = 0
        self._last_pressed: set[str] = set()

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {_to_xlerobot_arm_key(key): value for key, value in self.arms.action_features.items()}
        if self.config.control_base:
            features.update({"x.vel": float, "y.vel": float, "theta.vel": float})
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {_to_xlerobot_arm_key(key): value for key, value in self.arms.feedback_features.items()}

    @property
    def is_connected(self) -> bool:
        keyboard_connected = self.keyboard is None or self.keyboard.is_connected
        return self.arms.is_connected and keyboard_connected

    @property
    def is_calibrated(self) -> bool:
        return self.arms.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self.arms.connect(calibrate)
            if self.keyboard is not None:
                self.keyboard.connect()
                if not self.keyboard.is_connected:
                    raise RuntimeError(
                        "XLeRobot base keyboard control needs an interactive X11/macOS/Windows session"
                    )
        except Exception:
            self._disconnect_best_effort()
            raise

    def calibrate(self) -> None:
        self.arms.calibrate()

    def configure(self) -> None:
        self.arms.configure()

    def _base_action(self, pressed: set[str]) -> RobotAction:
        rising = pressed - self._last_pressed
        self._last_pressed = pressed
        if self.config.speed_up_key in rising:
            self._speed_index = min(self._speed_index + 1, len(self.config.base_speed_levels_xy) - 1)
        if self.config.speed_down_key in rising:
            self._speed_index = max(self._speed_index - 1, 0)

        xy_speed = self.config.base_speed_levels_xy[self._speed_index]
        theta_speed = self.config.base_speed_levels_theta[self._speed_index]
        return {
            "x.vel": xy_speed
            * (int(self.config.forward_key in pressed) - int(self.config.backward_key in pressed)),
            "y.vel": xy_speed
            * (int(self.config.left_key in pressed) - int(self.config.right_key in pressed)),
            "theta.vel": theta_speed
            * (int(self.config.rotate_left_key in pressed) - int(self.config.rotate_right_key in pressed)),
        }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action = {_to_xlerobot_arm_key(key): value for key, value in self.arms.get_action().items()}
        if self.keyboard is not None:
            action.update(self._base_action(set(self.keyboard.get_action())))
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        arm_feedback = {}
        for key, value in feedback.items():
            mapped = _to_bimanual_leader_key(key)
            if mapped is not None:
                arm_feedback[mapped] = value
        if arm_feedback:
            self.arms.send_feedback(arm_feedback)

    def _disconnect_best_effort(self) -> list[Exception]:
        errors: list[Exception] = []
        if self.keyboard is not None and self.keyboard.is_connected:
            try:
                self.keyboard.disconnect()
            except Exception as error:
                errors.append(error)
        for arm in (self.arms.right_arm, self.arms.left_arm):
            if arm.is_connected:
                try:
                    arm.disconnect()
                except Exception as error:
                    errors.append(error)
        return errors

    def disconnect(self) -> None:
        errors = self._disconnect_best_effort()
        if errors:
            raise RuntimeError(f"XLeRobot teleoperator disconnect failed: {errors}")
