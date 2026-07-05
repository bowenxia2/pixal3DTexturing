"""
Visualize a PBR voxel field as a colored point cloud.

The texturing / image-to-3D pipelines produce a *sparse* PBR voxel field
(`pbr_voxel`, a SparseTensor) where every occupied voxel stores 6 channels:
    0-2 = base_color (RGB)   3 = metallic   4 = roughness   5 = alpha
This is the volumetric representation the baking step samples from. It is a
hollow shell of colored cubes wrapped around the mesh surface.

This script turns that field into a colored point cloud (one point per occupied
voxel, colored by base_color) and writes a .ply you can open in MeshLab,
Blender, or any online viewer.

Two ways to use it:

1) Dump the field during a pipeline run, then view it offline:

    from visualize_voxels import save_voxel_field
    ...
    pbr_voxel = self.decode_tex_slat(tex_slat)
    save_voxel_field(pbr_voxel, "pbr_voxels.pt")   # add this line, run once
    ...
    # then, separately:
    #   python visualize_voxels.py pbr_voxels.pt --output pbr_voxels.ply

2) Convert a SparseTensor directly in-process:

    from visualize_voxels import voxel_field_to_pointcloud
    cloud = voxel_field_to_pointcloud(pbr_voxel)   # -> trimesh.PointCloud
    cloud.export("pbr_voxels.ply")
"""

import argparse

import numpy as np
import torch
import trimesh


# Channel layout of the decoded PBR voxel field.
PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def save_voxel_field(pbr_voxel, path: str) -> None:
    """Dump a SparseTensor PBR voxel field to a .pt file for offline viewing.

    Stores only the plain tensors needed to reconstruct the point cloud, so the
    dump does not depend on the pixal3d SparseTensor class being importable.
    """
    grid = int(pbr_voxel.spatial_shape[0])
    torch.save(
        {
            "coords": pbr_voxel.coords.detach().cpu(),  # [N, 4] = (batch, x, y, z)
            "feats": pbr_voxel.feats.detach().cpu().float(),  # [N, C]
            "grid": grid,
        },
        path,
    )
    print(f"[save_voxel_field] wrote {pbr_voxel.coords.shape[0]} voxels (grid={grid}) -> {path}")


def voxel_field_to_pointcloud(
    pbr_voxel=None,
    *,
    coords: torch.Tensor = None,
    feats: torch.Tensor = None,
    grid: int = None,
    color: str = "base_color",
    batch: int = 0,
) -> trimesh.PointCloud:
    """Build a colored ``trimesh.PointCloud`` from a PBR voxel field.

    Pass either a SparseTensor via ``pbr_voxel``, or raw ``coords`` / ``feats`` /
    ``grid`` (as produced by :func:`save_voxel_field`).

    Args:
        pbr_voxel: SparseTensor with ``.coords`` [N,4], ``.feats`` [N,C], ``.spatial_shape``.
        coords:    [N, 4] (batch, x, y, z) voxel indices. Used if ``pbr_voxel`` is None.
        feats:     [N, C] per-voxel PBR channels. Used if ``pbr_voxel`` is None.
        grid:      Grid resolution (e.g. 1024). Used if ``pbr_voxel`` is None.
        color:     Which channels drive point color: "base_color" (RGB),
                   "metallic", "roughness", or "alpha" (grayscale).
        batch:     Which batch index to extract (sparse coords are [batch, x, y, z]).

    Returns:
        trimesh.PointCloud with points normalized to [-0.5, 0.5].
    """
    if pbr_voxel is not None:
        coords = pbr_voxel.coords
        feats = pbr_voxel.feats
        grid = int(pbr_voxel.spatial_shape[0])
    if coords is None or feats is None or grid is None:
        raise ValueError("Provide either pbr_voxel, or all of coords/feats/grid.")

    coords = coords.detach().cpu()
    feats = feats.detach().cpu().float()

    # Keep only the requested batch, then drop the batch column.
    keep = coords[:, 0] == batch
    xyz = coords[keep, 1:].numpy().astype(np.float64)
    feats = feats[keep]

    # Voxel index -> normalized position in [-0.5, 0.5].
    pts = xyz / float(grid) - 0.5

    sel = PBR_ATTR_LAYOUT[color]
    chan = feats[:, sel].numpy()
    chan = np.clip(chan, 0.0, 1.0)
    if chan.shape[1] == 1:  # grayscale channel -> replicate to RGB
        chan = np.repeat(chan, 3, axis=1)
    colors = (chan * 255).astype(np.uint8)

    return trimesh.PointCloud(vertices=pts, colors=colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a .pt voxel-field dump written by save_voxel_field().")
    parser.add_argument("--output", "-o", default=None, help="Output .ply path (default: input with .ply suffix).")
    parser.add_argument(
        "--color",
        default="base_color",
        choices=list(PBR_ATTR_LAYOUT.keys()),
        help="Which PBR channel(s) to colorize points by (default: base_color).",
    )
    parser.add_argument("--batch", type=int, default=0, help="Batch index to extract (default: 0).")
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu")
    cloud = voxel_field_to_pointcloud(
        coords=data["coords"],
        feats=data["feats"],
        grid=int(data["grid"]),
        color=args.color,
        batch=args.batch,
    )

    out = args.output or (args.input.rsplit(".", 1)[0] + ".ply")
    cloud.export(out)
    print(f"[visualize_voxels] {len(cloud.vertices)} points colored by '{args.color}' -> {out}")


if __name__ == "__main__":
    main()
