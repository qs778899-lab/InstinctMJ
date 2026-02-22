from __future__ import annotations

import copy
import inspect
import mujoco
import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

from mjlab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


def _resolve_patch_radii(
    patch_radius: float | list[float] | tuple[float, ...],
) -> list[float]:
    """Normalize patch radius config into a non-empty list."""
    if isinstance(patch_radius, (list, tuple)):
        patch_radii = [float(radius) for radius in patch_radius]
    else:
        patch_radii = [float(patch_radius)]
    if len(patch_radii) == 0:
        raise ValueError("patch_radius list cannot be empty.")
    if any(radius < 0.0 for radius in patch_radii):
        raise ValueError(f"patch_radius must be non-negative. Got: {patch_radii}")
    return patch_radii


def _make_mujoco_mesh_qhull_safe(
    mesh: trimesh.Trimesh,
    *,
    coplanar_tol: float = 1.0e-12,
    z_perturbation: float = 1.0e-6,
) -> trimesh.Trimesh:
    """Return a mesh that is safe for MuJoCo qhull compilation.

    MuJoCo calls qhull when compiling `mjGEOM_MESH`. Purely coplanar meshes
    (e.g., perfectly flat terrain tiles) can trigger:
      "Initial simplex is flat ... Error: qhull error"

    To keep terrain semantics unchanged while avoiding compile failure, we only
    apply a tiny z perturbation to one vertex when the mesh is detected as
    coplanar.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] < 4:
        return mesh

    z_span = float(np.max(vertices[:, 2]) - np.min(vertices[:, 2]))
    if z_span > coplanar_tol:
        return mesh

    mesh_safe = mesh.copy()
    vertices_safe = np.asarray(mesh_safe.vertices, dtype=np.float64).copy()
    xy = vertices_safe[:, :2]
    center_xy = np.mean(xy, axis=0)
    # Use a boundary vertex (farthest from center) to localize the perturbation.
    perturb_idx = int(np.argmax(np.sum((xy - center_xy) ** 2, axis=1)))
    vertices_safe[perturb_idx, 2] -= z_perturbation
    mesh_safe.vertices = vertices_safe
    return mesh_safe


def _find_flat_patches_on_legacy_mesh(
    mesh: trimesh.Trimesh,
    *,
    device: str,
    num_patches: int,
    patch_radius: float | list[float] | tuple[float, ...],
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    max_height_diff: float,
) -> np.ndarray:
    """Find flat patches on legacy terrains (IsaacLab-compatible behavior).

    This mirrors IsaacLab's mesh-based `find_flat_patches()` flow used by the
    original InstinctLab terrain generator:
    1. Sample XY candidates in configured local ranges around `origin`.
    2. Ray-cast circular footprints onto the mesh.
    3. Reject candidates violating z-range or max height-difference.
    4. Return patch locations in the same frame as InstinctLab
       (mesh frame relative to `origin`).
    """
    from mjlab.utils.warp import convert_to_warp_mesh, raycast_mesh

    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    torch_device = torch.device(device)
    if isinstance(origin, np.ndarray):
        origin_t = torch.from_numpy(origin).to(dtype=torch.float, device=torch_device)
    elif isinstance(origin, torch.Tensor):
        origin_t = origin.to(dtype=torch.float, device=torch_device)
    else:
        origin_t = torch.tensor(origin, dtype=torch.float, device=torch_device)

    patch_radii = _resolve_patch_radii(patch_radius)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    x_lo = max(x_range[0] + origin_t[0].item(), float(vertices[:, 0].min()))
    x_hi = min(x_range[1] + origin_t[0].item(), float(vertices[:, 0].max()))
    y_lo = max(y_range[0] + origin_t[1].item(), float(vertices[:, 1].min()))
    y_hi = min(y_range[1] + origin_t[1].item(), float(vertices[:, 1].max()))
    z_lo = z_range[0] + origin_t[2].item()
    z_hi = z_range[1] + origin_t[2].item()

    if x_lo > x_hi or y_lo > y_hi:
        raise RuntimeError(
            "Failed to find valid patches! Sampling range is outside mesh bounds."
            f"\n\tx_range after clipping: ({x_lo}, {x_hi})"
            f"\n\ty_range after clipping: ({y_lo}, {y_hi})"
        )

    angle = torch.linspace(0.0, 2.0 * np.pi, 10, device=torch_device)
    query_x = []
    query_y = []
    for radius in patch_radii:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)
    query_y = torch.cat(query_y).unsqueeze(1)
    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    points_ids = torch.arange(num_patches, device=torch_device)
    flat_patches = torch.zeros(num_patches, 3, device=torch_device)

    iter_count = 0
    while len(points_ids) > 0 and iter_count < 10000:
        pos_x = torch.empty(len(points_ids), device=torch_device).uniform_(x_lo, x_hi)
        pos_y = torch.empty(len(points_ids), device=torch_device).uniform_(y_lo, y_hi)
        flat_patches[points_ids, :2] = torch.stack([pos_x, pos_y], dim=-1)

        points = flat_patches[points_ids].unsqueeze(1) + query_points
        points[..., 2] = 100.0
        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        heights = ray_hits.view(points.shape)[..., 2]

        flat_patches[points_ids, 2] = heights[..., -1]

        not_valid = torch.any(torch.logical_or(heights < z_lo, heights > z_hi), dim=1)
        not_valid = torch.logical_or(
            not_valid,
            (heights.max(dim=1)[0] - heights.min(dim=1)[0]) > max_height_diff,
        )
        points_ids = points_ids[not_valid]
        iter_count += 1

    if len(points_ids) > 0:
        raise RuntimeError(
            "Failed to find valid patches! Please check the input parameters."
            f"\n\tMaximum number of iterations reached: {iter_count}"
            f"\n\tNumber of invalid patches: {len(points_ids)}"
            f"\n\tMaximum height difference: {max_height_diff}"
        )

    return (flat_patches - origin_t).cpu().numpy()


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):

        # Access the i-th row, j-th column subterrain config by
        # self._subterrain_specific_cfgs[i*num_cols + j]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        self._terrain_meshes: list[trimesh.Trimesh] = []
        self.terrain_mesh: trimesh.Trimesh | None = None
        # Keep original patch radii (including list-style radii) for legacy
        # mesh flat-patch sampling after normalizing runtime cfg for mjlab core.
        self._original_patch_radii_by_cfg_id: dict[
            int, dict[str, float | list[float] | tuple[float, ...]]
        ] = {}
        # Keep InstinctLab semantics: generator-level scales apply to every heightfield-like subterrain.
        runtime_cfg = copy.deepcopy(cfg)
        self._sync_legacy_subterrain_scales(runtime_cfg)
        self._cache_original_patch_radii(runtime_cfg)
        self._normalize_patch_radii_for_mjlab_core(runtime_cfg)
        super().__init__(runtime_cfg, device)

    def _cache_original_patch_radii(self, cfg: FiledTerrainGeneratorCfg) -> None:
        """Cache original patch-radius config per subterrain for legacy sampling."""
        self._original_patch_radii_by_cfg_id.clear()
        for sub_cfg in cfg.sub_terrains.values():
            patch_sampling = getattr(sub_cfg, "flat_patch_sampling", None)
            if patch_sampling is None:
                continue
            cached_patch_radii: dict[str, float | list[float] | tuple[float, ...]] = {}
            for patch_name, patch_cfg in patch_sampling.items():
                cached_patch_radii[patch_name] = copy.deepcopy(patch_cfg.patch_radius)
            self._original_patch_radii_by_cfg_id[id(sub_cfg)] = cached_patch_radii

    @staticmethod
    def _normalize_patch_radii_for_mjlab_core(cfg: FiledTerrainGeneratorCfg) -> None:
        """Normalize list-style patch radii to scalar max for mjlab core init."""
        for sub_cfg in cfg.sub_terrains.values():
            patch_sampling = getattr(sub_cfg, "flat_patch_sampling", None)
            if patch_sampling is None:
                continue
            for patch_cfg in patch_sampling.values():
                patch_cfg.patch_radius = max(_resolve_patch_radii(patch_cfg.patch_radius))

    def _get_original_patch_radius(
        self,
        sub_terrain_cfg: SubTerrainBaseCfg,
        patch_name: str,
        fallback: float | list[float] | tuple[float, ...],
    ) -> float | list[float] | tuple[float, ...]:
        """Get original patch-radius config for this subterrain + patch name."""
        cfg_patch_radii = self._original_patch_radii_by_cfg_id.get(id(sub_terrain_cfg))
        if cfg_patch_radii is None:
            return fallback
        return cfg_patch_radii.get(patch_name, fallback)

    @staticmethod
    def _sync_legacy_subterrain_scales(cfg: FiledTerrainGeneratorCfg) -> None:
        """Apply generator-level scales to all compatible sub-terrains.

        InstinctLab/IsaacLab terrain generation treats ``horizontal_scale``,
        ``vertical_scale`` and ``slope_threshold`` as generator-wide settings.
        Parkour configs rely on this behavior (for example stair step discretization).
        """
        for sub_cfg in cfg.sub_terrains.values():
            if cfg.horizontal_scale is not None and hasattr(sub_cfg, "horizontal_scale"):
                sub_cfg.horizontal_scale = cfg.horizontal_scale
            if cfg.vertical_scale is not None and hasattr(sub_cfg, "vertical_scale"):
                sub_cfg.vertical_scale = cfg.vertical_scale
            if cfg.slope_threshold is not None and hasattr(sub_cfg, "slope_threshold"):
                sub_cfg.slope_threshold = cfg.slope_threshold

    def compile(self, spec: mujoco.MjSpec) -> None:
        self._terrain_meshes = []
        self.terrain_mesh = None
        super().compile(spec)
        if len(self._terrain_meshes) == 1:
            self.terrain_mesh = self._terrain_meshes[0]
        elif len(self._terrain_meshes) > 1:
            self.terrain_mesh = trimesh.util.concatenate(self._terrain_meshes)

    def _get_subterrain_function(self, cfg: SubTerrainBaseCfg):
        terrain_function = inspect.getattr_static(type(cfg), "function")
        if isinstance(terrain_function, (staticmethod, classmethod)):
            terrain_function = terrain_function.__func__
        return terrain_function

    def _create_legacy_terrain_geom(
        self,
        spec: mujoco.MjSpec,
        world_position: np.ndarray,
        meshes: trimesh.Trimesh | list[trimesh.Trimesh] | tuple[trimesh.Trimesh, ...],
        origin: np.ndarray,
        sub_terrain_cfg: SubTerrainBaseCfg,
        sub_row: int,
        sub_col: int,
    ) -> np.ndarray:
        if isinstance(meshes, trimesh.Trimesh):
            meshes_list = [meshes]
        elif isinstance(meshes, (list, tuple)):
            meshes_list = list(meshes)
        else:
            raise TypeError(
                "Legacy terrain function must return a trimesh.Trimesh or a list/tuple of trimesh.Trimesh."
            )

        body = spec.body("terrain")
        for mesh_idx, mesh in enumerate(meshes_list):
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError("Legacy terrain function returned a non-trimesh mesh entry.")
            mesh = _make_mujoco_mesh_qhull_safe(mesh)
            mesh_name = f"terrain_mesh_{sub_row}_{sub_col}_{mesh_idx}"
            spec.add_mesh(
                name=mesh_name,
                uservert=np.asarray(mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
                userface=np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist(),
            )
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=world_position,
            )
            if self.cfg.color_scheme == "random":
                geom.rgba[:3] = self.np_rng.uniform(0.3, 0.8, 3)
                geom.rgba[3] = 1.0
            elif self.cfg.color_scheme == "none":
                geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)

            # Keep a world-frame terrain mesh for virtual obstacle generation.
            world_mesh = mesh.copy()
            world_mesh.apply_translation(world_position)
            self._terrain_meshes.append(world_mesh)

        spawn_origin = np.asarray(origin, dtype=np.float64) + world_position
        for _, arr in self.flat_patches.items():
            # Keep fallback behavior from mjlab TerrainGenerator: every slot has a valid reset location.
            arr[sub_row, sub_col] = spawn_origin

        # Legacy terrain functions use the old `(difficulty, cfg) -> (meshes, origin)` signature and do not
        # return flat patches directly. Sample them from mesh geometry to match InstinctLab behavior.
        if sub_terrain_cfg.flat_patch_sampling is not None:
            sampling_mesh = trimesh.util.concatenate(meshes_list)
            for patch_name, patch_cfg in sub_terrain_cfg.flat_patch_sampling.items():
                if patch_name not in self.flat_patches:
                    self.flat_patches[patch_name] = np.zeros(
                        (
                            self.cfg.num_rows,
                            self.cfg.num_cols,
                            patch_cfg.num_patches,
                            3,
                        ),
                        dtype=np.float64,
                    )
                sampled_patches = _find_flat_patches_on_legacy_mesh(
                    sampling_mesh,
                    device=self.device,
                    num_patches=patch_cfg.num_patches,
                    patch_radius=self._get_original_patch_radius(
                        sub_terrain_cfg,
                        patch_name,
                        patch_cfg.patch_radius,
                    ),
                    origin=origin,
                    x_range=patch_cfg.x_range,
                    y_range=patch_cfg.y_range,
                    z_range=patch_cfg.z_range,
                    max_height_diff=patch_cfg.max_height_diff,
                )
                sampled_patches += spawn_origin
                patch_buffer = self.flat_patches[patch_name]
                num_patches_to_write = min(sampled_patches.shape[0], patch_buffer.shape[2])
                patch_buffer[sub_row, sub_col, :num_patches_to_write] = sampled_patches[:num_patches_to_write]
                if num_patches_to_write < patch_buffer.shape[2]:
                    patch_buffer[sub_row, sub_col, num_patches_to_write:] = spawn_origin
        return spawn_origin

    def _create_terrain_geom(
        self,
        spec: mujoco.MjSpec,
        world_position: np.ndarray,
        difficulty: float,
        cfg: SubTerrainBaseCfg,
        sub_row: int,
        sub_col: int,
    ):
        """This function intercept the terrain mesh generation process and records the specific config
        for each subterrain.
        """
        terrain_function = self._get_subterrain_function(cfg)
        num_args = len(inspect.signature(terrain_function).parameters)
        if num_args == 2:
            meshes, origin = terrain_function(difficulty, cfg)
            spawn_origin = self._create_legacy_terrain_geom(
                spec, world_position, meshes, origin, cfg, sub_row, sub_col
            )
        elif num_args == 4:
            # Record mesh names before calling super so we can identify newly-added mesh geoms.
            mesh_names_before = {m.name for m in spec.meshes}
            spawn_origin = super()._create_terrain_geom(
                spec,
                world_position,
                difficulty,
                cfg,
                sub_row,
                sub_col,
            )
            # Collect world-frame mesh for virtual obstacle generation (mirrors legacy path).
            new_mesh_names = {m.name for m in spec.meshes} - mesh_names_before
            for geom in spec.body("terrain").geoms:
                mesh_name = getattr(geom, "meshname", "")
                if not isinstance(mesh_name, str) or mesh_name not in new_mesh_names:
                    continue
                mjs_mesh = spec.mesh(mesh_name)
                if mjs_mesh is None:
                    continue
                verts = np.array(mjs_mesh.uservert, dtype=np.float32).reshape(-1, 3)
                faces = np.array(mjs_mesh.userface, dtype=np.int32).reshape(-1, 3)
                geom_pos = np.array(geom.pos, dtype=np.float64)
                world_mesh = trimesh.Trimesh(vertices=verts + geom_pos, faces=faces, process=False)
                self._terrain_meshes.append(world_mesh)
        else:
            raise TypeError(
                f"Unsupported terrain function signature for {type(cfg).__name__}: "
                f"expected 2 (legacy) or 4 (mjlab) arguments, got {num_args}."
            )
        # >>> NOTE: This code snippet is copied from the super implementation because they copied the cfg
        # but we need to store the modified cfg for each subterrain.
        cfg = copy.deepcopy(cfg)
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # <<< NOTE
        self._subterrain_specific_cfgs.append(cfg)  # since in super function, cfg is a copy of the original config.

        return spawn_origin

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy().tolist()  # Convert to list if it's a tensor.
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, int):
            return self._subterrain_specific_cfgs[idx] if 0 <= idx < len(self._subterrain_specific_cfgs) else None
