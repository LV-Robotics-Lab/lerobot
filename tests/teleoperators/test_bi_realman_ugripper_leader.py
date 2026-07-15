#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.teleoperators.bi_realman_ugripper_leader import (
    BiRealmanUGripperLeader,
    BiRealmanUGripperLeaderConfig,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config


def test_bi_realman_ugripper_leader_factory_and_features():
    cfg = BiRealmanUGripperLeaderConfig(id="test")

    teleop = make_teleoperator_from_config(cfg)

    assert isinstance(teleop, BiRealmanUGripperLeader)
    assert cfg.type == "bi_realman_ugripper_leader"
    assert teleop.action_features == {
        "left_main_joint1": float,
        "left_main_joint2": float,
        "left_main_joint3": float,
        "left_main_joint4": float,
        "left_main_joint5": float,
        "left_main_joint6": float,
        "left_main_joint7": float,
        "left_main_gripper": float,
        "right_main_joint1": float,
        "right_main_joint2": float,
        "right_main_joint3": float,
        "right_main_joint4": float,
        "right_main_joint5": float,
        "right_main_joint6": float,
        "right_main_joint7": float,
        "right_main_gripper": float,
    }
