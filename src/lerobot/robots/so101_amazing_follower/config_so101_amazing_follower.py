from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..so_follower import SOFollowerConfig


@dataclass
class AmazingHandAttachmentConfig:
    port: str
    calibration_file: str = ""
    baudrates: list[int] = field(default_factory=lambda: [1_000_000, 250_000])
    expected_model_numbers: list[int] = field(default_factory=lambda: [1280, 1284])
    motor_ids: list[int] = field(default_factory=lambda: list(range(1, 9)))
    leader_open_value: float = 100.0
    leader_closed_value: float = 0.0
    max_raw_velocity: float = 240.0
    command_timeout_s: float = 0.25
    max_temperature_c: float | None = 55.0
    max_abs_load: float | None = 900.0
    include_motor_observations: bool = False


@RobotConfig.register_subclass("so101_amazing_follower")
@dataclass
class SO101AmazingFollowerConfig(RobotConfig):
    arm_config: SOFollowerConfig
    hand_config: AmazingHandAttachmentConfig


@RobotConfig.register_subclass("bi_so101_amazing_follower")
@dataclass
class BiSO101AmazingFollowerConfig(RobotConfig):
    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig
    left_hand_config: AmazingHandAttachmentConfig
    right_hand_config: AmazingHandAttachmentConfig
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
