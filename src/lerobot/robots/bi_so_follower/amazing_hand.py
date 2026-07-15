#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from dataclasses import dataclass
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .config_bi_so_follower import AmazingHandConfig

AMAZING_HAND_MOTORS = (
    "index_1",
    "index_2",
    "middle_1",
    "middle_2",
    "ring_1",
    "ring_2",
    "thumb_1",
    "thumb_2",
)
SCS0009_MODEL_NUMBER = 1284


@dataclass(frozen=True)
class GripperSynergyMapper:
    """Map one normalized leader-gripper position to eight raw hand targets."""

    open_raw: dict[str, int]
    closed_raw: dict[str, int]
    leader_open_value: float = 100.0
    leader_closed_value: float = 0.0
    max_raw_step: int = 12

    def __post_init__(self) -> None:
        expected = set(AMAZING_HAND_MOTORS)
        if set(self.open_raw) != expected or set(self.closed_raw) != expected:
            raise ValueError(
                f"AmazingHand calibration must define exactly these motors: {AMAZING_HAND_MOTORS}"
            )
        if self.leader_open_value == self.leader_closed_value:
            raise ValueError("leader_open_value and leader_closed_value must differ")
        if self.max_raw_step <= 0:
            raise ValueError("max_raw_step must be positive")
        for motor in AMAZING_HAND_MOTORS:
            if self.open_raw[motor] == self.closed_raw[motor]:
                raise ValueError(f"Open and closed calibration are identical for '{motor}'")

    def closure_from_gripper(self, gripper_position: float) -> float:
        span = self.leader_closed_value - self.leader_open_value
        closure = (float(gripper_position) - self.leader_open_value) / span
        return min(1.0, max(0.0, closure))

    def targets(self, gripper_position: float, previous_raw: dict[str, int] | None = None) -> dict[str, int]:
        closure = self.closure_from_gripper(gripper_position)
        targets = {
            motor: round(self.open_raw[motor] + closure * (self.closed_raw[motor] - self.open_raw[motor]))
            for motor in AMAZING_HAND_MOTORS
        }
        if previous_raw is None:
            return targets

        limited = {}
        for motor, target in targets.items():
            previous = previous_raw[motor]
            limited[motor] = min(previous + self.max_raw_step, max(previous - self.max_raw_step, target))
        return limited

    def motor_closure(self, motor: str, raw_position: int) -> float:
        span = self.closed_raw[motor] - self.open_raw[motor]
        closure = (raw_position - self.open_raw[motor]) / span
        return min(100.0, max(0.0, closure * 100.0))


class AmazingHand:
    """Eight-SCS0009 AmazingHand controlled by one scalar grasp synergy."""

    def __init__(self, config: AmazingHandConfig, default_calibration_file: Path):
        self.config = config
        self.calibration_file = (
            Path(config.calibration_file) if config.calibration_file else default_calibration_file
        )
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                motor: Motor(index, "scs0009", MotorNormMode.RANGE_0_100)
                for index, motor in enumerate(AMAZING_HAND_MOTORS, start=1)
            },
            protocol_version=1,
        )
        self.mapper: GripperSynergyMapper | None = self._load_mapper()
        self._last_raw: dict[str, int] | None = None

    def _load_mapper(self) -> GripperSynergyMapper | None:
        if self.config.open_positions and self.config.closed_positions:
            open_raw = dict(zip(AMAZING_HAND_MOTORS, self.config.open_positions, strict=True))
            closed_raw = dict(zip(AMAZING_HAND_MOTORS, self.config.closed_positions, strict=True))
        elif self.calibration_file.is_file():
            data = json.loads(self.calibration_file.read_text())
            open_raw = data["open_raw"]
            closed_raw = data["closed_raw"]
        else:
            return None
        return GripperSynergyMapper(
            open_raw=open_raw,
            closed_raw=closed_raw,
            leader_open_value=self.config.leader_open_value,
            leader_closed_value=self.config.leader_closed_value,
            max_raw_step=self.config.max_raw_step,
        )

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.mapper is not None

    @property
    def observation_features(self) -> dict[str, type]:
        return {f"hand_{motor}.pos": float for motor in AMAZING_HAND_MOTORS}

    def connect(self, calibrate: bool = True) -> None:
        # Open without the default 1 Mbit/s handshake, then select the detected hand baud rate.
        self.bus.connect(handshake=False)
        self.bus.set_baudrate(self.config.baudrate)
        missing = [motor for motor in AMAZING_HAND_MOTORS if self.bus.ping(motor) != SCS0009_MODEL_NUMBER]
        if missing:
            self.bus.disconnect(disable_torque=False)
            raise ConnectionError(f"AmazingHand is missing SCS0009 motors: {missing}")

        if not self.is_calibrated:
            if not calibrate:
                self.bus.disconnect(disable_torque=False)
                raise RuntimeError(
                    f"AmazingHand has no open/closed calibration at '{self.calibration_file}'."
                )
            self.calibrate()

        self._last_raw = self.read_raw_positions()
        self.bus.enable_torque()

    def read_raw_positions(self) -> dict[str, int]:
        return {
            motor: int(self.bus.read("Present_Position", motor, normalize=False))
            for motor in AMAZING_HAND_MOTORS
        }

    def calibrate(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Connect the AmazingHand before calibration")
        self.bus.disable_torque()
        input("Place the AmazingHand in its safe fully OPEN pose, then press ENTER...")
        open_raw = self.read_raw_positions()
        input("Place the AmazingHand in its safe fully CLOSED pose, then press ENTER...")
        closed_raw = self.read_raw_positions()
        self.mapper = GripperSynergyMapper(
            open_raw=open_raw,
            closed_raw=closed_raw,
            leader_open_value=self.config.leader_open_value,
            leader_closed_value=self.config.leader_closed_value,
            max_raw_step=self.config.max_raw_step,
        )
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_file.write_text(
            json.dumps({"open_raw": open_raw, "closed_raw": closed_raw}, indent=2) + "\n"
        )
        self._last_raw = closed_raw

    def get_observation(self) -> dict[str, float]:
        if self.mapper is None:
            raise RuntimeError("AmazingHand is not calibrated")
        raw = self.read_raw_positions()
        return {
            f"hand_{motor}.pos": self.mapper.motor_closure(motor, raw[motor]) for motor in AMAZING_HAND_MOTORS
        }

    def send_gripper(self, gripper_position: float) -> dict[str, int]:
        if self.mapper is None or self._last_raw is None:
            raise RuntimeError("AmazingHand is not connected and calibrated")
        targets = self.mapper.targets(gripper_position, previous_raw=self._last_raw)
        self.bus.sync_write("Goal_Position", targets, normalize=False)
        self._last_raw = targets
        return targets

    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
