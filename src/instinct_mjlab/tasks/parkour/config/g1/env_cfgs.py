"""Alias module for parkour environment configurations.

This module provides backward compatibility by re-exporting the main
parkour configuration functions and constants.
"""

from instinct_mjlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
  instinct_g1_parkour_amp_env_cfg,
  instinct_g1_parkour_amp_final_cfg,
)
from instinct_mjlab.tasks.parkour.config.parkour_env_cfg import (
  _BASE_VELOCITY_COMMAND_NAME,
)

__all__ = [
  "instinct_g1_parkour_amp_env_cfg",
  "instinct_g1_parkour_amp_final_cfg",
  "_BASE_VELOCITY_COMMAND_NAME",
]
