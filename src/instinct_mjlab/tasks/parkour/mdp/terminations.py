from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def sub_terrain_out_of_bounds(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distance_buffer: float = 3.0,
) -> torch.Tensor:
  """Terminate when the actor move too close to the edge of the sub terrain.

  If the actor moves too close to the edge of the sub terrain, the termination is activated. The distance
  to the edge of the sub terrain is calculated based on the size of the sub terrain and the distance buffer.
  """
  if env.cfg.scene.terrain.terrain_type == "plane":
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  elif env.cfg.scene.terrain.terrain_type == "generator":
    # obtain the size of the sub-terrains
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    grid_width, grid_length = terrain_gen_cfg.size
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_link_pos_w
    terrain_origins = env.scene.terrain.env_origins

    # check if the agent is out of bounds
    x_out_of_bounds = (
      torch.abs(root_pos_w[:, 0] - terrain_origins[:, 0]) > 0.5 * grid_width - distance_buffer
    )
    y_out_of_bounds = (
      torch.abs(root_pos_w[:, 1] - terrain_origins[:, 1]) > 0.5 * grid_length - distance_buffer
    )
    return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
  else:
    raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def root_height_below_env_origin_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when root height drops below env origin + minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
  root_pos_w = asset.data.root_link_pos_w
  return root_pos_w[:, 2] - terrain_base_height < minimum_height


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force_history = contact_sensor.data.force_history
  if force_history is not None:
    in_contact = torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0] > threshold
  else:
    force = contact_sensor.data.force
    assert force is not None
    in_contact = torch.linalg.vector_norm(force, dim=-1) > threshold
  return torch.any(in_contact, dim=1)
