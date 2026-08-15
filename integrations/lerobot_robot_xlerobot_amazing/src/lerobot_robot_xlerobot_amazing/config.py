from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.robots.so101_amazing_follower import AmazingHandAttachmentConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

from ._upstream import XLerobotConfig


@RobotConfig.register_subclass("xlerobot_amazing_follower")
@dataclass(kw_only=True)
class XLerobotAmazingFollowerConfig(XLerobotConfig):
    left_hand_config: AmazingHandAttachmentConfig
    right_hand_config: AmazingHandAttachmentConfig
    # Neither the XLeRobot body JSON nor the hand JSONs encode physical-device identity. Runtime
    # use requires an explicit attestation after checking persistent paths and calibration records.
    calibration_provenance_verified: bool = False
    # Deliberate escape hatch for ``lerobot-calibrate`` only. Runtime launchers must leave this
    # false so missing or mismatched calibration fails before any device I/O.
    allow_interactive_calibration: bool = False


@TeleoperatorConfig.register_subclass("xlerobot_bi_so_leader")
@dataclass
class XLerobotBiSOLeaderConfig(TeleoperatorConfig):
    left_arm_config: SOLeaderConfig
    right_arm_config: SOLeaderConfig
    control_base: bool = False
    base_speed_levels_xy: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    base_speed_levels_theta: list[float] = field(default_factory=lambda: [30.0, 60.0, 90.0])
    forward_key: str = "i"
    backward_key: str = "k"
    left_key: str = "j"
    right_key: str = "l"
    rotate_left_key: str = "u"
    rotate_right_key: str = "o"
    speed_up_key: str = "n"
    speed_down_key: str = "m"

    def __post_init__(self) -> None:
        if len(self.base_speed_levels_xy) != len(self.base_speed_levels_theta):
            raise ValueError("XLeRobot base speed level lists must have the same length")
        if not self.base_speed_levels_xy:
            raise ValueError("XLeRobot requires at least one base speed level")
        if any(speed <= 0 for speed in self.base_speed_levels_xy):
            raise ValueError("XLeRobot xy speed levels must be positive")
        if any(speed <= 0 for speed in self.base_speed_levels_theta):
            raise ValueError("XLeRobot theta speed levels must be positive")

    def arms_config(self) -> BiSOLeaderConfig:
        return BiSOLeaderConfig(
            id=self.id,
            calibration_dir=self.calibration_dir,
            left_arm_config=self.left_arm_config,
            right_arm_config=self.right_arm_config,
        )
