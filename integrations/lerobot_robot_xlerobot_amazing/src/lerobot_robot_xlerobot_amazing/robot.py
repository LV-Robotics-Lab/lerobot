import logging
from functools import cached_property
from pathlib import Path

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors.feetech import OperatingMode
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
        """Connect without ever falling back to an interactive calibration prompt.

        XLeRobot's upstream ``connect`` asks on stdin even when ``calibrate=False`` and enables
        torque before latching position goals or stopping the mobile base.  A Prometheus process
        has no safe way to answer that prompt, and stale controller goals can move hardware before
        its first action.  Keep the safer connection contract in this LV-owned integration layer.

        ``calibrate`` is retained for LeRobot API compatibility but never grants permission to
        prompt. All three calibration files must already pass the runtime gates unless the dedicated
        ``allow_interactive_calibration`` config gate is set for a supervised ``lerobot-calibrate``
        invocation. That mode connects with all torque disabled and returns without configuring or
        activating anything; the CLI then calls :meth:`calibrate` explicitly.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            if self.config.allow_interactive_calibration:
                body_calibrations = None
            else:
                body_calibrations = self._require_saved_calibrations()

            self.bus1.connect()
            self.bus1.disable_torque()
            self.bus2.connect()
            self.bus2.disable_torque()

            # AmazingHand.connect() is probe-only and also leaves torque disabled.
            self.left_hand.connect()
            self.right_hand.connect()

            for camera in self.cameras.values():
                camera.connect()

            if body_calibrations is None:
                logger.warning(
                    "%s connected torque-off for explicitly requested interactive calibration",
                    self,
                )
                return

            self._restore_body_calibration(body_calibrations)

            # configure() latches every position motor and writes base velocity zero before it
            # enables either body bus. It also latches both hand goals while hand torque is off.
            self.configure()
            for hand in (self.left_hand, self.right_hand):
                hand.activate()
            if calibrate:
                logger.info(
                    "Automatic calibration is disabled for %s; restored verified saved calibration",
                    self,
                )
            logger.info("%s connected with non-interactive fail-closed calibration restore", self)
        except Exception:
            self._shutdown_best_effort(force_disable_torque=True)
            raise

    def _require_saved_calibrations(self) -> tuple[dict, dict]:
        if not self.id:
            raise RuntimeError("XLeRobot AmazingHand requires an explicit robot id for calibration")

        expected_body_path = self.calibration_dir / f"{self.id}.json"
        if self.calibration_fpath != expected_body_path or not expected_body_path.is_file():
            raise RuntimeError(
                f"body calibration file for robot id {self.id!r} is missing: {expected_body_path}"
            )

        calibrations_by_bus: list[dict] = []
        for label, bus in (("left/head", self.bus1), ("right/base", self.bus2)):
            missing = sorted(set(bus.motors) - set(self.calibration))
            if missing:
                raise RuntimeError(
                    f"body calibration for robot id {self.id!r} is missing {label} motors: {missing}"
                )
            mismatched = {
                name: (self.calibration[name].id, motor.id)
                for name, motor in bus.motors.items()
                if self.calibration[name].id != motor.id
            }
            if mismatched:
                raise RuntimeError(
                    f"body calibration motor IDs do not match robot id {self.id!r}: {mismatched}"
                )
            calibrations_by_bus.append({name: self.calibration[name] for name in bus.motors})

        for side, hand, hand_config in (
            ("left", self.left_hand, self.config.left_hand_config),
            ("right", self.right_hand, self.config.right_hand_config),
        ):
            expected_path = (
                Path(hand_config.calibration_file).expanduser()
                if hand_config.calibration_file
                else self.calibration_dir / f"{self.id}_{side}_amazing_hand.json"
            )
            actual_path = Path(hand.config.calibration_file).expanduser()
            if actual_path != expected_path or not expected_path.is_file() or not hand.is_calibrated:
                raise RuntimeError(
                    f"{side} AmazingHand calibration path is missing or failed to load: {expected_path}"
                )
            schema = getattr(hand.calibration, "schema", None)
            if schema != "lv_robotics.amazinghand_calibration.v1":
                raise RuntimeError(
                    f"{side} AmazingHand calibration has unsupported schema {schema!r}: {expected_path}"
                )

        if not self.config.calibration_provenance_verified:
            raise RuntimeError(
                "body and AmazingHand calibration files do not encode hardware identity; runtime "
                "requires external calibration provenance verification"
            )

        return calibrations_by_bus[0], calibrations_by_bus[1]

    def _restore_body_calibration(self, calibrations: tuple[dict, dict]) -> None:
        for bus, calibration in zip((self.bus1, self.bus2), calibrations, strict=True):
            bus.calibration = calibration
            bus.write_calibration(calibration)
        if not bool(XLerobot.is_calibrated.fget(self)):
            raise RuntimeError(f"body calibration restore did not verify for robot id {self.id!r}")

    def _configure_body_without_torque(self) -> None:
        for bus in (self.bus1, self.bus2):
            bus.disable_torque()
            bus.configure_motors()

        for bus, motors in (
            (self.bus1, self.left_arm_motors),
            (self.bus1, self.head_motors),
            (self.bus2, self.right_arm_motors),
        ):
            for name in motors:
                bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                bus.write("P_Coefficient", name, 16)
                bus.write("I_Coefficient", name, 0)
                bus.write("D_Coefficient", name, 43)
        for name in self.base_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

    def _latch_present_positions_and_stop_base(self) -> None:
        for bus, motors in (
            (self.bus1, self.left_arm_motors + self.head_motors),
            (self.bus2, self.right_arm_motors),
        ):
            present = bus.sync_read("Present_Position", motors)
            if set(present) != set(motors):
                raise RuntimeError(f"incomplete present-position read before torque enable: {present}")
            bus.sync_write("Goal_Position", present)

        # Do not rely on a later partial 12-D action to stop a base that may retain a stale goal.
        self.stop_base()

        # AmazingHand's controller intentionally leaves torque off after connect. Latch its current
        # raw positions as goals before any actuator in the composite robot is torque-enabled.
        for side, hand in (("left", self.left_hand), ("right", self.right_hand)):
            present = hand.backend.read_positions()
            expected_ids = set(hand.config.motor_ids)
            if set(present) != expected_ids:
                raise RuntimeError(f"{side} AmazingHand position read does not match configured motor IDs")
            hand.backend.write_positions(present)

    def configure(self) -> None:
        try:
            self._configure_body_without_torque()
            self._latch_present_positions_and_stop_base()
            self.bus1.enable_torque()
            self.bus2.enable_torque()
        except Exception:
            self._shutdown_best_effort(force_disable_torque=True)
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

    def _shutdown_best_effort(self, *, force_disable_torque: bool = False) -> list[Exception]:
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
                if force_disable_torque:
                    try:
                        bus.disable_torque()
                    except Exception as error:
                        errors.append(error)
                try:
                    bus.disconnect(
                        True if force_disable_torque else self.config.disable_torque_on_disconnect
                    )
                except Exception as error:
                    errors.append(error)
        return errors

    def disconnect(self) -> None:
        errors = self._shutdown_best_effort()
        if errors:
            raise RuntimeError(f"XLeRobot AmazingHand disconnect failed: {errors}")
