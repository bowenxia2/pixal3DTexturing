
<div align="center">

# Pixal3D: Pixel-Aligned 3D Generation from Images

<h3>SIGGRAPH 2026</h3>

<small>[Dong-Yang Li](https://ldyang694.github.io/)¹ · [Wang Zhao](https://thuzhaowang.github.io/)²* · [Yuxin Chen](https://orcid.org/0000-0002-7854-1072)² · [Wenbo Hu](https://wbhu.github.io/)² · [Meng-Hao Guo](https://menghaoguo.github.io/)¹ · [Fang-Lue Zhang](https://fanglue.github.io/)³ · [Ying Shan](https://www.linkedin.com/in/YingShanProfile)² · [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)¹✉</small>

¹Tsinghua University (BNRist) &nbsp;&nbsp; ²Tencent ARC Lab &nbsp;&nbsp; ³Victoria University of Wellington

*Project lead &nbsp;&nbsp; ✉Corresponding author

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href="https://huggingface.co/spaces/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href="https://huggingface.co/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
  <a href="LICENSE"><img src=https://img.shields.io/badge/License-MIT-yellow.svg height=22px></a>
</div>

<div align="center">
    <img src="assets/teaser.png" alt="Teaser image of Pixal3D"/>
</div>

**Pixal3D** generates high-fidelity 3D assets from a single image.
Instead of loosely injecting image features via attention, it explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences.

---

## 🔬 This Fork

This is a fork of the [original Pixal3D repo](https://github.com/TencentARC/Pixal3D) that adds **texture-only inference with automatic pose alignment**: texturing an externally-supplied mesh (e.g. a Hunyuan3D output) from a single photo, even when the photo is not shot from the mesh's canonical front.

**Added files**

| File | Purpose |
|------|---------|
| `texture_inference.py` | CLI entry point: texture an external mesh from a photo. |
| `pixal3d/pipelines/pixal3d_texturing.py` | Texturing pipeline (proj-mode texture DIT on an external mesh). |
| `pixal3d/utils/pose_utils.py` | Confidence-gated photo↔mesh rotation estimation from MoGe-2 point maps. |
| `visualize_voxels.py` | Export a `.pt` voxel-field dump to `.ply` for inspection. |

**Modified files**

| File | Change |
|------|--------|
| `pixal3d/pipelines/__init__.py` | Registers `Pixal3DTexturingPipeline`. |
| `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` | Proj-feature extractor changes supporting the texturing pipeline. |

Everything else is unchanged from upstream.

## 📌 Branches

| Branch | Description |
|--------|-------------|
| `main` | **Latest version** — improved implementation based on the [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone. |
| `paper` | **Paper version** — original [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)-based implementation reproducing the SIGGRAPH 2026 paper results. |

## 🚀 Installation

First set up the base [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) environment, then:

```bash
pip install -r requirements.txt

# Replace xx with your CUDA arch and desired build worker count
NATTEN_CUDA_ARCH="xx" NATTEN_N_WORKERS=xx pip install natten==0.21.0 --no-build-isolation

pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

> If `flash_attn` is unavailable, prefix any command with `ATTN_BACKEND=sdpa`.

## 🖼️ Inference

**Image → 3D** (generate a GLB from a single image):

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram   # ~10-12GB VRAM
python inference.py --image assets/images/0_img.png --output ./output.glb --resolution 1024
```

Default resolution is 1536 (standard) or 1024 (low-VRAM); override with `--resolution`.

**Texture an external mesh** (this fork) — pose-aligned texturing of a supplied mesh from a photo:

```bash
python texture_inference.py --mesh mesh.glb --image photo.png --output ./textured.glb --low_vram
python texture_inference.py --mesh mesh.glb --image photo.png --output ./textured.glb --no_pose      # skip pose alignment
python texture_inference.py --mesh mesh.glb --image photo.png --output ./textured.glb --debug_proj   # dump projection overlays
```

Key options: `--resolution` (voxel grid, default 1024), `--texture_size` (atlas, default 2048), `--fov` (radians, override MoGe estimate), `--bbox_padding` (default 1.1).

**Web demo** (Gradio):

```bash
python app.py
python app.py --low_vram   # or: LOW_VRAM=1 python app.py
```

## 🔧 Training

Pixal3D is trained as a three-stage cascade, each stage progressively increasing resolution.
All stages use pixel-aligned projection conditioning and view-aligned latents (2 views by default).
Within a stage, train the lowest resolution first, then fine-tune upward by setting `finetune_ckpt` in the config to the previous checkpoint.

First prepare data following **[data_toolkit/README.md](data_toolkit/README.md)**, then:

```bash
python train.py --config <CONFIG_JSON> --output_dir <OUTPUT_DIR> --data_dir '<DATA_DIR_JSON>'
```

`--data_dir` is a JSON string describing the dataset layout. Stages, configs, and required keys:

| Stage | Config prefix | Resolutions | Required data keys |
|-------|---------------|-------------|--------------------|
| 1 — Sparse Structure | `ss_flow_img_dit_*_proj_finetune` | 32 → 64 | `base`, `ss_latent`, `render_cond` |
| 2 — Shape | `slat_flow_img2shape_*_proj_finetune` | 256 → 512 → 1024 | `base`, `shape_latent`, `render_cond` |
| 3 — Texture | `slat_flow_imgshape2tex_*_proj_finetune` | 256 → 512 → 1024 | `base`, `shape_latent`, `pbr_latent`, `render_cond` |

<details>
<summary><b>Example: full three-stage sequence (ObjaverseXL)</b></summary>

Set `finetune_ckpt` in each higher-resolution config to the previous checkpoint.

```sh
# --- Stage 1: Sparse Structure (32 → 64) ---
python train.py --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune.json \
  --output_dir results/ss_32 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "ss_latent": "datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

python train.py --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json \
  --output_dir results/ss_ft64 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "ss_latent": "datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# --- Stage 2: Shape (256 → 512 → 1024) ---
python train.py --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune.json \
  --output_dir results/shape_256 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_256_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
# ...then _ft512 (shape_..._512_view) and _ft1024 (shape_..._1024_view) configs.

# --- Stage 3: Texture (256 → 512 → 1024) ---
python train.py --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_256_bf16_proj_finetune.json \
  --output_dir results/tex_256 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_256_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_256_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
# ...then the _512 and _ft1024 texture configs with matching 512/1024 latent dirs.
```
</details>

<details>
<summary><b>Common <code>train.py</code> flags</b></summary>

| Flag | Default | Purpose |
|------|---------|---------|
| `--ckpt` | `latest` | Resume from a specific step. |
| `--load_dir` | `output_dir` | Checkpoint load directory. |
| `--auto_retry` | `3` | Retries on OOM / crash. |
| `--num_nodes` / `--node_rank` | `1` / `0` | Multi-node distributed. |
| `--num_gpus` | all | GPUs per node. |
| `--master_addr` / `--master_port` | `localhost` / `12666` | Distributed rendezvous. |
| `--use_wandb` | `false` | Enable W&B logging (`--wandb_project`, `--wandb_name`, `--wandb_id`). |
| `--tryrun` | `false` | Dry run. |

</details>

## 🤗 Acknowledgements

Built upon [Trellis.2](https://github.com/microsoft/TRELLIS.2), [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), and [Trellis](https://github.com/microsoft/TRELLIS).
Pose alignment uses [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl).
We sincerely thank the authors for their outstanding work.

## 📄 Citation

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```

## 📜 License

Released under the [MIT License](LICENSE).
Third-party components remain under their original terms; see [NOTICE](NOTICE).
