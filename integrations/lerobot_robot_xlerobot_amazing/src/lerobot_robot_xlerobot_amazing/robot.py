import logging
from functools import cached_property

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.so101_amazing_follower.so101_amazing_follower import _make_hand
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ._upstream import XLerobot
from .config import XLerobotAmazingFollowerConfig

logger = logging.getLogger(__name__)


class XLerobotAmazingFollower(XLerobot):
    """XLeRobot's dual-arm mobile body with stock ID-6 grippers replaced by AmazingHands."""

    config_class = XLerobotAmazingFollowerConfig
    name = "xlerobot_amazing_follower"

    def __init__(self, config: XLerobotAmazingFollowerConfig):
        super().__init__(config)
        self.config = config

        # XLeRobot keeps each arm on the same bus as its head/base auxiliaries. Remove only
        # the absent stock ID-6 grippers; all arm, head, and wheel motors remain upstream-owned.
        self.bus1.motors.pop("left_arm_gripper", None)
        self.bus2.motors.pop("right_arm_gripper", None)
        self.bus1.calibration.pop("left_arm_gripper", None)
        self.bus2.calibration.pop("right_arm_gripper", None)
        self.left_arm_motors = [name for name in self.left_arm_motors if name != "left_arm_gripper"]
        self.right_arm_motors = [name for name in self.right_arm_motors if name != "right_arm_gripper"]

        calibration_stem = config.id or self.name
        self.left_hand = _make_hand(
            config.left_hand_config,
            default_calibration_file=self.calibration_dir / f"{calibration_stem}_left_amazing_hand.json",
        )
        self.right_hand = _make_hand(
            config.right_hand_config,
            default_calibration_file=self.calibration_dir / f"{calibration_stem}_right_amazing_hand.json",
        )

    @property
    def is_connected(self) -> bool:
        body_connected = bool(XLerobot.is_connected.fget(self))
        return body_connected and self.left_hand.is_connected and self.right_hand.is_connected

    @property
    def is_calibrated(self) -> bool:
        body_calibrated = bool(XLerobot.is_calibrated.fget(self))
        return body_calibrated and self.left_hand.is_calibrated and self.right_hand.is_calibrated

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = dict(XLerobot.observation_features.__get__(self, type(self)))
        for side, hand_config in (
            ("left", self.config.left_hand_config),
            ("right", self.config.right_hand_config),
        ):
            if hand_config.include_motor_observations:
                features.update({f"{side}_hand_{index}.pos": float for index in range(1, 9)})
        return features

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # Hand connect is probe-only and leaves torque disabled. Connecting it first lets a
            # body calibration invoked by upstream also calibrate both hands in one supervised flow.
            self.left_hand.connect()
            self.right_hand.connect()
            XLerobot.connect(self, calibrate=calibrate)

            missing_calibration = [
                hand for hand in (self.left_hand, self.right_hand) if not hand.is_calibrated
            ]
            # ``lerobot-calibrate`` intentionally connects with calibrate=False before it
            # invokes ``calibrate()``. Leave both hands torque-free for that workflow.
            if missing_calibration and not calibrate:
                return
            for hand in missing_calibration:
                hand.calibrate_interactive()
            for hand in (self.left_hand, self.right_hand):
                hand.activate()
        except Exception:
            self._shutdown_best_effort()
            raise

    def calibrate(self) -> None:
        XLerobot.calibrate(self)
        for hand in (self.left_hand, self.right_hand):
            if not hand.is_connected:
                raise RuntimeError("connect both AmazingHands before calibration")
            hand.calibrate_interactive()

    def _fail_closed(self, reason: str) -> None:
        if getattr(self, "bus2", None) is not None and self.bus2.is_connected:
            try:
                self.stop_base()
            except Exception:
                logger.exception("Failed to stop XLeRobot base while handling %s", reason)
        for hand in (self.left_hand, self.right_hand):
            if hand.is_connected or hand.is_active:
                try:
                    hand.emergency_stop(reason)
                except Exception:
                    logger.exception("Failed to stop an AmazingHand while handling %s", reason)

    def get_observation(self) -> RobotObservation:
        try:
            observation = dict(XLerobot.get_observation(self))
            for side, hand, hand_config in (
                ("left", self.left_hand, self.config.left_hand_config),
                ("right", self.right_hand, self.config.right_hand_config),
            ):
                hand_observation = hand.observe()
                observation[f"{side}_arm_gripper.pos"] = hand_observation.grasp_closure
                if hand_config.include_motor_observations:
                    observation.update(
                        {
                            f"{side}_hand_{index}.pos": value
                            for index, value in enumerate(
                                hand_observation.motor_closure.values(), start=1
                            )
                        }
                    )
            return observation
        except Exception as error:
            self._fail_closed(f"observation failure: {error}")
            raise

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        if not self.left_hand.is_active or not self.right_hand.is_active:
            raise RuntimeError("both AmazingHands must be active before commanding XLeRobot")

        left_gripper = action.get("left_arm_gripper.pos")
        right_gripper = action.get("right_arm_gripper.pos")
        if left_gripper is None or right_gripper is None:
            raise ValueError("XLeRobot AmazingHand action requires both arm gripper positions")

        body_action = dict(action)
        body_action.pop("left_arm_gripper.pos")
        body_action.pop("right_arm_gripper.pos")
        try:
            sent = dict(XLerobot.send_action(self, body_action))
            self.left_hand.command_grasp(float(left_gripper))
            self.right_hand.command_grasp(float(right_gripper))
            sent["left_arm_gripper.pos"] = float(left_gripper)
            sent["right_arm_gripper.pos"] = float(right_gripper)
            return sent
        except Exception as error:
            self._fail_closed(f"command failure: {error}")
            raise

    def _shutdown_best_effort(self) -> list[Exception]:
        errors: list[Exception] = []
        if getattr(self, "bus2", None) is not None and self.bus2.is_connected:
            try:
                self.stop_base()
            except Exception as error:
                errors.append(error)

        for hand in (getattr(self, "right_hand", None), getattr(self, "left_hand", None)):
            if hand is not None:
                try:
                    hand.disconnect()
                except Exception as error:
                    errors.append(error)

        for camera in getattr(self, "cameras", {}).values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception as error:
                    errors.append(error)

        for bus in (getattr(self, "bus2", None), getattr(self, "bus1", None)):
            if bus is not None and bus.is_connected:
                try:
                    bus.disconnect(self.config.disable_torque_on_disconnect)
                except Exception as error:
                    errors.append(error)
        return errors

    def disconnect(self) -> None:
        errors = self._shutdown_best_effort()
        if errors:
            raise RuntimeError(f"XLeRobot AmazingHand disconnect failed: {errors}")
