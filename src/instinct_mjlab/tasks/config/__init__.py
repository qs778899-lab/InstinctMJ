"""Task config packages."""

from instinct_mjlab.tasks.config.scene_style_cfg import (
  INSTINCT_BRIGHT_SCENE_STYLE_CFG,
  SceneVisualStyleCfg,
  apply_scene_visual_style,
  edit_spec_with_scene_visual_style,
)

__all__ = [
  "SceneVisualStyleCfg",
  "INSTINCT_BRIGHT_SCENE_STYLE_CFG",
  "edit_spec_with_scene_visual_style",
  "apply_scene_visual_style",
]
