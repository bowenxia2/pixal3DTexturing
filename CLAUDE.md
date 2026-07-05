# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixal3D is a SIGGRAPH 2026 single-image-to-3D generation system. Its core idea: instead of loosely injecting image features via cross-attention, it back-projects DINOv3 pixel features into 3D space using estimated camera parameters, creating explicit pixel-to-3D correspondences. Built on top of [Trellis.2](https://github.com/microsoft/TRELLIS.2).

Two branches exist: `main` (Trellis.2-based, current) and `paper` (Direct3D-S2-based, original paper results).

## Installation

Requires Trellis.2 environment (`trellis2` conda env) installed first, then:

```bash
pip install -r requirements.txt
NATTEN_CUDA_ARCH="xx" NATTEN_N_WORKERS=xx pip install natten==0.21.0 --no-build-isolation
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

If `flash_attn` is unavailable, set `ATTN_BACKEND=sdpa` before running.

## Common Commands

```bash
# Inference: image → GLB
python inference.py --image assets/images/0_img.png --output ./output.glb
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram  # ~10-12GB VRAM
python inference.py --image assets/images/0_img.png --output ./output.glb --resolution 1024  # override resolution

# Web demo
python app.py
python app.py --low_vram  # or LOW_VRAM=1 python app.py

# Training (all three stages use the same entry point)
python train.py --config <CONFIG_JSON> --output_dir <OUTPUT_DIR> --data_dir '<DATA_DIR_JSON>'

# Distributed training
python train.py --config <config> --output_dir <dir> --data_dir '<json>' \
  --num_nodes 4 --node_rank 0 --master_addr <addr>
```

## Architecture

### Inference Pipeline (`pixal3d/pipelines/pixal3d_image_to_3d.py`)

`Pixal3DImageTo3DPipeline.run()` executes five sequential stages:

1. **Sparse Structure (SS)** — `sparse_structure_flow_model` generates a 32³ binary occupancy grid. Conditioned on `DinoV3ProjFeatureExtractor` (SS config: 512px input, 16³ grid).
2. **Shape LR 512** — `shape_slat_flow_model_512` generates structured latents at 512 resolution on the sparse coordinates. Conditioned on proj features at 32³ grid.
3. **Upsample LR → HR** — `shape_slat_decoder.upsample()` predicts HR voxel positions from LR latents. Token count is capped at `max_num_tokens=49152`; resolution drops from 1536 toward 1024 in 128-step decrements if exceeded.
4. **Shape HR 1024/1536** — `shape_slat_flow_model_1024` refines at HR coordinates. Conditioned on proj features at `resolution//16` grid.
5. **Texture 1024** — `tex_slat_flow_model_1024` generates PBR textures (base_color, metallic, roughness, alpha). Shape latent is concatenated as `concat_cond`.

Two pipeline modes: `1024_cascade` (default in low-VRAM) and `1536_cascade` (default standard).

Models loaded from `TencentARC/Pixal3D` on HuggingFace. Pipeline config stored in `pipeline.json` inside the model directory.

### Pixel-Aligned Projection (`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`)

`DinoV3ProjFeatureExtractor` is the key innovation:
- Runs DINOv3 ViT on the input image to extract patch features
- Optionally upsamples via NAF (Nonlinear Activation Free) network to higher resolution
- Back-projects the 3D grid into image space using the estimated camera (FOV + distance + mesh scale)
- Returns `(z_global, z_proj)` — global CLS-style token and per-voxel projected features

Camera parameters come from **MoGe-2** (`Ruicheng/moge-2-vitl`) monocular geometry estimator. Manual FOV override is available via `--fov` (in radians; try `0.2` for in-the-wild images with distortion).

### Module Layout

```
pixal3d/
  pipelines/         # Inference pipelines (image-to-3D, texturing)
  models/            # Flow models, VAEs
  modules/           # Sparse tensors, image feature extractors, spatial ops
  trainers/          # Training logic
    flow_matching/   # Main flow matching trainers
      mixins/        # Conditioning mixins (image_conditioned_proj.py = key file)
    vae/             # VAE trainers
  datasets/          # Dataset loaders for training
  renderers/         # Mesh/voxel/PBR renderers
  representations/   # Mesh, MeshWithVoxel data structures
  utils/             # Distributed, loss, mesh, render utilities
```

### Three-Stage Training Cascade

| Stage | Config prefix | Resolution steps | Dataset keys needed |
|-------|--------------|------------------|---------------------|
| 1 — Sparse Structure | `ss_flow_img_dit_*_proj_finetune` | 32 → 64 | `base`, `ss_latent`, `render_cond` |
| 2 — Shape | `slat_flow_img2shape_*_proj_finetune` | 256 → 512 → 1024 | `base`, `shape_latent`, `render_cond` |
| 3 — Texture | `slat_flow_imgshape2tex_*_proj_finetune` | 256 → 512 → 1024 | `base`, `shape_latent`, `pbr_latent`, `render_cond` |

Within each stage, train at the lowest resolution first, then progressively fine-tune by setting `finetune_ckpt` in the config JSON to the previous checkpoint.

## Data Preparation (`data_toolkit/`)

Eight-step pipeline, all scripts support `--rank` / `--world_size` for distributed execution:

```bash
# Step 1: Init metadata
python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab

# Step 2: Download assets
python data_toolkit/download.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Step 3: Process mesh + PBR
python data_toolkit/dump_mesh.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/dump_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Step 4: Render multi-view condition images (2 views default, uses Blender)
python data_toolkit/render_cond.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Step 5: Convert to view-aligned O-Voxels (CPU)
python data_toolkit/dual_grid_view.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256 --view_indices 0-1
python data_toolkit/voxelize_pbr_view.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256 --view_indices 0-1

# Step 6: Encode latents (shape, PBR, then SS)
python data_toolkit/encode_shape_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 1024 --view_indices 0-1
python data_toolkit/encode_pbr_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 1024 --view_indices 0-1
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/encode_ss_latent_view.py --root datasets/ObjaverseXL_sketchfab \
  --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024_view --resolution 64 --view_indices 0-1

# Run build_metadata.py after each step that produces new files
```

Supported datasets: `ObjaverseXL` (sketchfab / github sources), `ABO`, `TexVerse` for training; `SketchfabPicked`, `Toys4k` for evaluation.

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `ATTN_BACKEND` | `flash_attn` (default) or `sdpa` (PyTorch built-in, slower) |
| `PYTORCH_CUDA_ALLOC_CONF` | Set to `expandable_segments:True` to reduce fragmentation |
| `LOW_VRAM` | Set to `1` to enable low-VRAM mode in `app.py` |
| `FLEX_GEMM_AUTOTUNE_CACHE_PATH` | Points to `autotune_cache.json` in repo root |

## Training Arguments

Key `train.py` flags beyond config/output/data:

| Flag | Default | Purpose |
|------|---------|---------|
| `--ckpt` | `latest` | Resume from specific step |
| `--auto_retry` | `3` | Retries on OOM or crash |
| `--tryrun` | false | Dry run (no actual training) |
| `--num_nodes` / `--node_rank` | 1 / 0 | Multi-node distributed |
| `--use_wandb` | false | Enable W&B logging |

## GLB Export

After pipeline produces `MeshWithVoxel` objects, `o_voxel.postprocess.to_glb()` converts them to trimesh GLB:
- `decimation_target=1000000`, `texture_size=4096`
- PBR attribute layout: channels 0-2=base_color, 3=metallic, 4=roughness, 5=alpha
- A coordinate rotation (`rot` matrix in `inference.py`) is applied before export to match standard GLB conventions
