from __future__ import annotations
from dataclasses import dataclass, field

from typing import Literal

from mjlab.terrains import TerrainImporterCfg as TerrainImporterCfgBase

from .terrain_importer import TerrainImporter
from .virtual_obstacle import VirtualObstacleCfg


@dataclass(kw_only=True)
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    """The inherited class to use for the terrain importer."""

    virtual_obstacles: dict[str, VirtualObstacleCfg] = field(default_factory=dict)
    """The virtual obstacles to use for the terrain importer."""

    collision_debug_vis: bool = False
    """Whether to visualize terrain collision geoms by tinting them in purple."""

    collision_debug_rgba: tuple[float, float, float, float] = (0.62, 0.2, 0.9, 0.35)
    """RGBA tint for terrain collision debug visualization."""

    terrain_type: Literal["generator", "plane", "hacked_generator"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane" and "generator".

    ## NOTE
    The TerrainImporter of this package has some dedicated hack to fit the self-defined tasks.
    We add a "hacked_generator" option to hack and run our own terrain generator implementation.
    """
