"""
Texture an externally-supplied mesh from a single photo using Pixal3D's proj-mode
texture DIT, with automatic pose alignment.

The proj-mode texture DIT back-projects pixel features onto the voxel grid assuming the
photo was shot from the mesh's canonical front.  For an external mesh (e.g. a Hunyuan3D
output) + a natural photo, that assumption is usually wrong.  This script estimates the
photo<->mesh rotation from MoGe-2's visible-surface point map and re-poses the mesh into
the canonical frame before texturing (falling back to the front-view assumption when the
registration is not confident).  See ``pixal3d/utils/pose_utils.py``.

Usage:
    python texture_inference.py --mesh mesh.glb --image photo.png --output out.glb --low_vram
    python texture_inference.py --mesh mesh.glb --image photo.png --output out.glb --no_pose
    python texture_inference.py --mesh mesh.glb --image photo.png --output out.glb --debug_proj
"""

import os
import argparse
import math
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"
)

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

# Reuse the camera helpers + model builders from the image-to-3D entry point.
from inference import (
    IMAGE_COND_CONFIGS,
    MOGE_MODEL_NAME,
    build_image_cond_model,
    distance_from_fov,
)
from pixal3d.pipelines import Pixal3DTexturingPipeline


# MoGe-2 is installed in a different conda env (`unified3d`), not `trellis2`.  Import torch
# first (done above) so MoGe reuses the already-loaded torch, then make `moge` importable.
# Override the location with the MOGE_SITE_PACKAGES env var if it moves.
_MOGE_SITE_PACKAGES = os.environ.get(
    "MOGE_SITE_PACKAGES",
    "/mmfs1/gscratch/krishna/bxia2/conda/envs/unified3d/lib/python3.11/site-packages",
)


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    try:
        from moge.model.v2 import MoGeModel
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, _MOGE_SITE_PACKAGES)
        try:
            from moge.model.v2 import MoGeModel
        finally:
            sys.path.pop(0)
    return MoGeModel.from_pretrained(model_name).to(device).eval()


# ============================================================================
# MoGe: camera params + visible-surface point map (single inference)
# ============================================================================

def get_camera_and_points(
    image: Image.Image,
    moge_model,
    device: str = "cuda",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
):
    """Run MoGe-2 once; return (camera_params, points[H,W,3], valid_mask[H,W]).

    camera_params has keys camera_angle_x, distance, mesh_scale (same as inference.py).
    points are in MoGe's camera frame; valid_mask is MoGe's validity mask.
    """
    pil = image.convert("RGB")
    width, height = pil.size
    image_np = np.asarray(pil).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        out = moge_model.infer(image_tensor)

    intrinsics = out["intrinsics"].squeeze().cpu().numpy()
    fx = intrinsics[0, 0] * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))
    distance = distance_from_fov(
        camera_angle_x, torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution,
    )["distance_from_x"]
    camera_params = {
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": mesh_scale,
    }

    points = out["points"].detach().cpu().numpy().astype(np.float32)  # [H, W, 3]
    if "mask" in out and out["mask"] is not None:
        valid = out["mask"].detach().cpu().numpy().astype(bool)
    else:
        valid = np.isfinite(points).all(axis=-1)
    return camera_params, points, valid


# ============================================================================
# Debug: project the (re-posed) mesh onto the photo to validate alignment
# ============================================================================

def save_proj_overlay(pipeline, orig_mesh, image, camera_params, out_path, image_resolution=1024):
    """Project the working (re-posed) mesh's front surface onto *image* and save an overlay.

    This is the ground-truth check that the estimated rotation is correct: front-facing
    voxels should land on the photographed object, not its back/side.
    """
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        project_points_to_image_batch,
        ProjGrid,
    )

    # Reconstruct the working-frame mesh exactly as run() does.
    mesh = pipeline.preprocess_mesh(orig_mesh.copy())
    R = pipeline._align_R
    if not np.allclose(R, np.eye(3)):
        verts = pipeline._normalize_verts(mesh.vertices @ R.T)
        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

    pts, fidx = trimesh.sample.sample_surface(mesh, 6000)
    pts = np.asarray(pts, dtype=np.float32)
    normals = mesh.face_normals[fidx]
    front = normals[:, 1] < 0  # camera-facing (canonical camera looks +Y from -Y)
    pts = pts[front]

    tm = ProjGrid().front_view_transform_matrix.clone()
    tm[1, 3] = -float(camera_params["distance"])
    pts2d, depth, valid = project_points_to_image_batch(
        torch.from_numpy(pts)[None].float(),
        tm[None].float(),
        torch.tensor([float(camera_params["camera_angle_x"])]),
        image_resolution,
    )
    pts2d = pts2d[0].numpy()
    valid = valid[0].numpy()
    vp = pts2d[valid]

    res = image_resolution
    base = image.convert("RGB").resize((res, res))
    # Measure the object silhouette from the CLEAN image (before drawing dots, which would
    # otherwise inflate the bbox and make the ratio read ~1.0 regardless of pose).
    sil = np.asarray(base).sum(-1) > 24  # non-black object pixels
    ys_s, xs_s = np.where(sil)

    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    for x, y in vp:
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 0, 0))
    canvas.save(out_path)
    print(f"[Debug] projection overlay -> {out_path}")

    # Quantify shift as pose (center offset) vs framing (size ratio).  preprocess_image
    # crops the object with 1.1x bbox padding, so a correct pose still projects ~1.1x larger
    # than the silhouette and ~0 center offset.  A real pose error shows up as center offset.
    if len(vp) and len(xs_s):
        px0, py0, px1, py1 = vp[:, 0].min(), vp[:, 1].min(), vp[:, 0].max(), vp[:, 1].max()
        sx0, sy0, sx1, sy1 = xs_s.min(), ys_s.min(), xs_s.max(), ys_s.max()
        dcx = (px0 + px1) / 2 - (sx0 + sx1) / 2
        dcy = (py0 + py1) / 2 - (sy0 + sy1) / 2
        rx = (px1 - px0) / max(1, sx1 - sx0)
        ry = (py1 - py0) / max(1, sy1 - sy0)
        print(f"[Debug] center offset=({dcx:.0f},{dcy:.0f})px / {res}  size ratio=({rx:.2f},{ry:.2f})"
              f"  (offset~0 => pose correct; ratio~1.1 => expected framing padding)")

        # Padding-compensated view (dots /1.1 about center) — overlays the silhouette if pose is right.
        canvas2 = image.convert("RGB").resize((res, res))
        d2 = ImageDraw.Draw(canvas2)
        c = res / 2.0
        for x, y in vp:
            xc, yc = c + (x - c) / 1.1, c + (y - c) / 1.1
            d2.ellipse([xc - 1, yc - 1, xc + 1, yc + 1], fill=(0, 255, 0))
        comp = out_path.replace(".png", "_compensated.png")
        canvas2.save(comp)
        print(f"[Debug] padding-compensated overlay -> {comp}")


# ============================================================================
# Main
# ============================================================================

def run(
    mesh_path: str,
    image_path: str,
    output_path: str,
    seed: int = 42,
    resolution: int = 1024,
    texture_size: int = 2048,
    low_vram: bool = False,
    manual_fov: float = -1.0,
    use_pose: bool = True,
    debug_proj: bool = False,
    image_resolution: int = 512,
    mesh_scale: float = 1.0,
    bbox_padding: float = 1.1,
):
    device = "cuda"
    print(f"[Texture] Loading pipeline...")
    pipeline = Pixal3DTexturingPipeline.from_pretrained()
    pipeline.low_vram = low_vram
    pipeline._device = torch.device(device)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor (tex_1024)...")
    image_cond_model = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])
    if getattr(image_cond_model, "use_naf_upsample", False):
        image_cond_model._load_naf()
    pipeline.image_cond_model = image_cond_model

    if not low_vram:
        pipeline.cuda()
        pipeline.image_cond_model.cuda()
        pipeline.rembg_model.to(device)

    # --- Load + preprocess inputs ---
    print(f"[Texture] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    print(f"[Texture] Preprocessing image: {image_path}")
    img = Image.open(image_path)
    image_pp, alpha = pipeline.preprocess_image(img, return_mask=True, bbox_padding=bbox_padding)

    # --- Camera + pose conditioning via MoGe-2 ---
    moge_points = moge_mask = None
    if manual_fov > 0:
        camera_angle_x = float(manual_fov)
        distance = distance_from_fov(
            camera_angle_x, torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0, image_resolution - 1]), mesh_scale, image_resolution,
        )["distance_from_x"]
        camera_params = {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}
        print(f"[Texture] Manual FOV {math.degrees(manual_fov):.1f}deg; pose estimation disabled (needs MoGe point map).")
    else:
        print("[MoGe-2] Loading model...")
        moge_model = load_moge_model(device=device, model_name=MOGE_MODEL_NAME)
        camera_params, points, valid = get_camera_and_points(
            image_pp, moge_model, device=device, mesh_scale=mesh_scale, image_resolution=image_resolution,
        )
        moge_model.cpu(); del moge_model; torch.cuda.empty_cache()
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f} distance={camera_params['distance']:.4f}")

        if use_pose:
            # Object foreground mask = rembg alpha AND MoGe validity (resized to point map).
            a = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (points.shape[1], points.shape[0]), Image.NEAREST)).astype(np.float32) / 255.0
            moge_mask = (a > 0.5) & valid
            moge_points = points

    # --- Texture ---
    print("[Texture] Running texture DIT...")
    t0 = time.time()
    out_mesh = pipeline.run(
        mesh,
        image_pp,
        camera_params=camera_params,
        seed=seed,
        preprocess_image=False,
        resolution=resolution,
        texture_size=texture_size,
        moge_points=moge_points,
        moge_mask=moge_mask,
    )
    print(f"[Texture] Done in {time.time() - t0:.1f}s")

    if debug_proj:
        dbg = os.path.splitext(output_path)[0] + "_proj.png"
        save_proj_overlay(pipeline, mesh, image_pp, camera_params, dbg)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_mesh.export(output_path)
    print(f"[Done] Textured GLB saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D texture-only inference with pose alignment")
    parser.add_argument("--mesh", required=True, help="Input mesh (GLB/OBJ/PLY).")
    parser.add_argument("--image", required=True, help="Reference photo.")
    parser.add_argument("--output", default="./textured.glb", help="Output GLB path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024, help="Voxel grid resolution.")
    parser.add_argument("--texture_size", type=int, default=2048, help="Output atlas size.")
    parser.add_argument("--low_vram", action="store_true", help="Load models to GPU on-demand.")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual FOV (radians). Disables MoGe + pose estimation.")
    parser.add_argument("--bbox_padding", type=float, default=1.1,
                        help="Object crop padding (1.1=main-pipeline default; 1.0=object fills frame, "
                             "aligns proj grid with the object).")
    parser.add_argument("--no_pose", action="store_true",
                        help="Disable pose alignment (assume mesh already front-aligned). For A/B testing.")
    parser.add_argument("--debug_proj", action="store_true",
                        help="Save an overlay of the re-posed mesh projected onto the photo.")
    args = parser.parse_args()

    run(
        mesh_path=args.mesh,
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        resolution=args.resolution,
        texture_size=args.texture_size,
        low_vram=args.low_vram,
        manual_fov=args.fov,
        use_pose=not args.no_pose,
        debug_proj=args.debug_proj,
        bbox_padding=args.bbox_padding,
    )
