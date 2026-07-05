from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trimesh
import cv2
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import flex_gemm


class Pixal3DTexturingPipeline(Pipeline):
    """
    Texture-only pipeline using Pixal3D's proj-mode texture DIT on an input mesh.

    Combines TRELLIS.2's shape_slat_encoder (to voxelize any input mesh into the
    shared shape-SLat latent space) with Pixal3D's camera-aware proj-mode texture DIT
    and decoder.  The shape geometry is never modified.

    Compared to Trellis2TexturingPipeline, the image conditioning is proj-aligned:
    pixel features are back-projected onto the sparse voxel grid using the camera
    parameters, so the model can match visible surface detail to 3D position.

    Args:
        models: dict with keys 'shape_slat_encoder', 'tex_slat_flow_model_1024',
                'tex_slat_decoder'.
        tex_slat_sampler: sampler for the texture SLat flow model.
        tex_slat_sampler_params: default sampler hyper-parameters.
        shape_slat_normalization: channel-wise mean/std of shape SLat features.
        tex_slat_normalization: channel-wise mean/std of texture SLat features.
        image_cond_model: DinoV3ProjFeatureExtractor for the texture stage.
            Must be set before calling run() — not loaded from the pipeline config.
        rembg_model: background-removal model.
        low_vram: keep models on CPU and move to GPU only when needed.
    """

    model_names_to_load = [
        'shape_slat_encoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        tex_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
    ):
        if models is None:
            return
        super().__init__(models)
        self.tex_slat_sampler = tex_slat_sampler
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self._align_R = np.eye(3, dtype=np.float64)  # mesh->canonical rotation (identity = front-view)
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic':   slice(3, 4),
            'roughness':  slice(4, 5),
            'alpha':      slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(
        cls,
        pixal3d_path: str = "TencentARC/Pixal3D",
        trellis2_path: str = "microsoft/TRELLIS.2-4B",
    ) -> "Pixal3DTexturingPipeline":
        """
        Load the pipeline from two repos:
        - tex_slat_flow_model_1024 and tex_slat_decoder from *pixal3d_path*
        - shape_slat_encoder from *trellis2_path*

        Normalization stats are taken from *pixal3d_path* (they are identical in both
        repos, but Pixal3D's config is the authoritative source for Pixal3D's DIT).
        """
        import os, json
        from .. import models as model_loader

        def load_config(repo_path, filename):
            if os.path.exists(f"{repo_path}/{filename}"):
                with open(f"{repo_path}/{filename}") as f:
                    return json.load(f)
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(repo_path, filename)) as f:
                return json.load(f)

        pixal_cfg  = load_config(pixal3d_path,  "pipeline.json")['args']
        trellis_cfg = load_config(trellis2_path, "texturing_pipeline.json")['args']

        def load_model(repo_path, rel_path):
            try:
                return model_loader.from_pretrained(f"{repo_path}/{rel_path}")
            except Exception:
                return model_loader.from_pretrained(rel_path)

        loaded_models = {
            'tex_slat_flow_model_1024': load_model(
                pixal3d_path, pixal_cfg['models']['tex_slat_flow_model_1024']
            ),
            'tex_slat_decoder': load_model(
                pixal3d_path, pixal_cfg['models']['tex_slat_decoder']
            ),
            'shape_slat_encoder': load_model(
                trellis2_path, trellis_cfg['models']['shape_slat_encoder']
            ),
        }

        pipeline = cls(loaded_models)

        pipeline.tex_slat_sampler = getattr(
            samplers, pixal_cfg['tex_slat_sampler']['name']
        )(**pixal_cfg['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = pixal_cfg['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = pixal_cfg['shape_slat_normalization']
        pipeline.tex_slat_normalization   = pixal_cfg['tex_slat_normalization']

        rembg_cfg = pixal_cfg['rembg_model']
        pipeline.rembg_model = getattr(rembg, rembg_cfg['name'])(**rembg_cfg['args'])

        pipeline.image_cond_model = None  # set externally (DinoV3ProjFeatureExtractor)
        pipeline.low_vram = True
        pipeline._device = 'cpu'
        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Normalise vertices to [-0.5, 0.5] and apply the Y/Z swap expected by o_voxel."""
        verts = mesh.vertices
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        center = (lo + hi) / 2
        scale = 0.99999 / (hi - lo).max()
        verts = (verts - center) * scale
        # swap Y↔Z to match internal convention
        tmp = verts[:, 1].copy()
        verts[:, 1] = -verts[:, 2]
        verts[:, 2] = tmp
        assert np.all(verts >= -0.5) and np.all(verts <= 0.5)
        return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

    @staticmethod
    def _normalize_verts(verts: np.ndarray) -> np.ndarray:
        """Center vertices on their bounding box and scale to fill [-0.5, 0.5]."""
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        center = (lo + hi) / 2
        scale = 0.99999 / (hi - lo).max()
        return (verts - center) * scale

    def align_mesh_to_image(
        self,
        mesh: trimesh.Trimesh,
        moge_points: np.ndarray,
        moge_mask: np.ndarray,
    ) -> trimesh.Trimesh:
        """Re-pose *mesh* (already in canonical V frame) so its photographed front faces the
        front-view camera, using MoGe's visible-surface point map.

        Estimates the rotation R (see ``pixal3d.utils.pose_utils``), applies it, and
        re-normalizes to [-0.5, 0.5].  Stores R in ``self._align_R`` so ``postprocess_mesh``
        can undo it on the output.  Falls back to identity (front-view) when registration
        is not confident — geometry is never harmed.
        """
        from ..utils.pose_utils import estimate_alignment_rotation

        R, info = estimate_alignment_rotation(mesh, moge_points, moge_mask)
        self._align_R = R
        tag = "fallback->front-view" if info["fallback"] else "applied"
        print(f"[Pose] {tag}: inlier_frac={info['inlier_frac']:.2f} "
              f"rmse={info['rmse']:.3f} n_tgt={info['n_tgt']}")
        if info["fallback"]:
            return mesh

        verts = self._normalize_verts(mesh.vertices @ R.T)
        return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

    def preprocess_image(self, input: Image.Image, bg_color: tuple = (0, 0, 0),
                         return_mask: bool = False, bbox_padding: float = 1.1):
        """Remove background, crop to object bounding box, composite onto bg_color.

        If ``return_mask`` is True, also return the cropped foreground alpha as a float
        ``[H, W]`` array in [0, 1] (same size as the returned image).  Used to isolate the
        object from the background for pose estimation.

        ``bbox_padding`` controls how much margin around the object bounding box is kept
        (1.1 = object fills ~91% of the frame, the main-pipeline convention; 1.0 = object
        fills the frame, which aligns the proj grid with the object but removes the margin).
        """
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize(
                (int(input.width * scale), int(input.height * scale)),
                Image.Resampling.LANCZOS,
            )
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        out_np = np.array(output)
        alpha = out_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (np.min(bbox[:, 1]), np.min(bbox[:, 0]),
                np.max(bbox[:, 1]), np.max(bbox[:, 0]))
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * bbox_padding)
        bbox = (center[0] - size // 2, center[1] - size // 2,
                center[0] + size // 2, center[1] + size // 2)
        output = output.crop(bbox)
        out_np = np.array(output).astype(np.float32) / 255
        rgb, a = out_np[:, :, :3], out_np[:, :, 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        composited = rgb * a + bg * (1.0 - a)
        out_img = Image.fromarray((np.clip(composited, 0, 1) * 255).astype(np.uint8))
        if return_mask:
            return out_img, a[..., 0]
        return out_img

    # ------------------------------------------------------------------
    # Shape encoding  (uses TRELLIS.2 shape_slat_encoder)
    # ------------------------------------------------------------------

    def encode_shape_slat(self, mesh: trimesh.Trimesh, resolution: int = 1024) -> SparseTensor:
        """Voxelise *mesh* and encode it into shape SLat using the TRELLIS.2 encoder."""
        vertices = torch.from_numpy(mesh.vertices).float()
        faces    = torch.from_numpy(mesh.faces).long()

        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )

        coords = torch.cat(
            [torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1
        )
        verts_st = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices, coords=coords
        ).to(self.device)
        inter_st = verts_st.replace(intersected).to(self.device)

        if self.low_vram:
            self.models['shape_slat_encoder'].to(self.device)
        shape_slat = self.models['shape_slat_encoder'](verts_st, inter_st)
        if self.low_vram:
            self.models['shape_slat_encoder'].cpu()
        return shape_slat

    # ------------------------------------------------------------------
    # Proj-mode image conditioning  (Pixal3D-style)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_proj_cond(
        self,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float,
        distance: float,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """
        Build camera-aware proj conditioning aligned to *coords*.

        Returns a dict with 'cond' and 'neg_cond', each containing
        {'global': tensor, 'proj': SparseTensor}.
        """
        assert self.image_cond_model is not None, (
            "image_cond_model must be set before calling run(). "
            "Assign a DinoV3ProjFeatureExtractor to pipeline.image_cond_model."
        )
        device = self.device
        model = self.image_cond_model

        if self.low_vram:
            model.to(device)

        orig_grid_res = model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            model.grid_resolution = grid_resolution_override
            model.proj_grid = model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=model.proj_grid.image_resolution,
            ).to(device)

        cam_angle  = torch.tensor([camera_angle_x], device=device)
        dist_t     = torch.tensor([distance],        device=device)
        scale_t    = torch.tensor([mesh_scale],      device=device)
        z_global, z_proj = model(
            image,
            camera_angle_x=cam_angle,
            distance=dist_t,
            mesh_scale=scale_t,
        )

        grid_res = model.grid_resolution
        z_proj_grid = z_proj.reshape(1, grid_res, grid_res, grid_res, -1)
        bi = coords[:, 0].long()
        xi = coords[:, 1].long()
        yi = coords[:, 2].long()
        zi = coords[:, 3].long()
        z_proj_sparse = z_proj_grid[bi, xi, yi, zi]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            model.grid_resolution = orig_grid_res
            model.proj_grid = model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=model.proj_grid.image_resolution,
            ).to(device)

        if self.low_vram:
            model.cpu()

        return {
            'cond':     {'global': z_global,                  'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global),
                         'proj':   SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    # ------------------------------------------------------------------
    # Texture DIT sampling  (Pixal3D-style)
    # ------------------------------------------------------------------

    def sample_tex_slat(
        self,
        cond: dict,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """Run the texture DIT, conditioned on *shape_slat* (geometry) and *cond* (image)."""
        std  = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_norm = (shape_slat - mean) / std

        flow_model = self.models['tex_slat_flow_model_1024']
        in_ch = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat_norm.replace(
            feats=torch.randn(
                shape_slat_norm.coords.shape[0],
                in_ch - shape_slat_norm.feats.shape[1],
            ).to(self.device)
        )

        params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat_norm,
            **cond,
            **params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (Pixal3D proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std  = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        return slat * std + mean

    # ------------------------------------------------------------------
    # Texture decoding
    # ------------------------------------------------------------------

    def decode_tex_slat(self, slat: SparseTensor) -> SparseTensor:
        """Decode texture SLat to PBR voxel grid (base_color, metallic, roughness, alpha)."""
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        # guide_subs=None is valid — the decoder supports it natively
        ret = self.models['tex_slat_decoder'](slat, guide_subs=None) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret

    # ------------------------------------------------------------------
    # Mesh postprocessing  (UV-unwrap + texture bake)
    # ------------------------------------------------------------------

    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 2048,
    ) -> trimesh.Trimesh:
        """UV-unwrap *mesh*, sample PBR values from *pbr_voxel*, return textured trimesh."""
        vertices = mesh.vertices.copy()
        faces    = mesh.faces.copy()
        normals  = mesh.vertex_normals.copy()

        verts_t = torch.from_numpy(vertices).float().cuda()
        faces_t = torch.from_numpy(faces).int().cuda()

        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_t = torch.from_numpy(uvs).float().cuda()
        else:
            _cm = cumesh.CuMesh()
            _cm.init(verts_t, faces_t)
            verts_t, faces_t, uvs_t, vmap = _cm.uv_unwrap(return_vmaps=True)
            verts_t = verts_t.cuda()
            faces_t = faces_t.cuda()
            uvs_t   = uvs_t.cuda()
            vertices = verts_t.cpu().numpy()
            faces    = faces_t.cpu().numpy()
            uvs      = uvs_t.cpu().numpy()
            normals  = normals[vmap.cpu().numpy()]

        ctx = dr.RasterizeCudaContext()
        uvs_clip = torch.cat(
            [uvs_t * 2 - 1, torch.zeros_like(uvs_t[:, :1]), torch.ones_like(uvs_t[:, :1])],
            dim=-1,
        ).unsqueeze(0)
        rast, _ = dr.rasterize(ctx, uvs_clip, faces_t, resolution=[texture_size, texture_size])
        mask = rast[0, ..., 3] > 0
        pos  = dr.interpolate(verts_t.unsqueeze(0), rast, faces_t)[0][0]

        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
            mode='trilinear',
        )

        mask_np = mask.cpu().numpy()
        inpaint_mask = (~mask_np).astype(np.uint8)

        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic   = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy()   * 255, 0, 255).astype(np.uint8)
        roughness  = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy()  * 255, 0, 255).astype(np.uint8)
        alpha      = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy()      * 255, 0, 255).astype(np.uint8)

        base_color = cv2.inpaint(base_color, inpaint_mask, 3, cv2.INPAINT_TELEA)
        metallic   = cv2.inpaint(metallic,   inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
        roughness  = cv2.inpaint(roughness,  inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]
        alpha      = cv2.inpaint(alpha,      inpaint_mask, 1, cv2.INPAINT_TELEA)[..., None]

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(
                np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
            ),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=True,
        )

        # Undo the pose-alignment rotation (R applied as verts @ R.T -> inverse is @ R) so the
        # textured output returns to the input mesh's orientation.  Identity when no alignment.
        R = self._align_R
        if not np.allclose(R, np.eye(3)):
            vertices = vertices @ R
            normals = normals @ R

        # Undo the Y/Z swap applied in preprocess_mesh so the output matches GLB convention
        vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
        normals[:, 1],  normals[:, 2]  = normals[:, 2].copy(),  -normals[:, 1].copy()
        uvs[:, 1] = 1 - uvs[:, 1]

        return trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        camera_params: dict,
        seed: int = 42,
        preprocess_image: bool = True,
        resolution: int = 1024,
        texture_size: int = 2048,
        tex_slat_sampler_params: dict = {},
        moge_points: np.ndarray = None,
        moge_mask: np.ndarray = None,
    ) -> trimesh.Trimesh:
        """
        Apply Pixal3D's proj-mode texture DIT to *mesh* guided by *image*.

        Args:
            mesh: Input geometry (trimesh.Trimesh or trimesh.Scene — Scene is flattened).
            image: Reference texture image.
            camera_params: dict with keys:
                - camera_angle_x (float): horizontal FOV in radians.
                - distance (float): camera distance (used for proj alignment).
                - mesh_scale (float, optional): scale factor, default 1.0.
            seed: Random seed for the texture DIT sampler.
            preprocess_image: Remove background and crop before conditioning.
            resolution: Voxel grid resolution (1024 recommended).
            texture_size: Output texture atlas size in pixels.
            tex_slat_sampler_params: Override default sampler params
                (steps, guidance_strength, guidance_rescale, rescale_t).
            moge_points: Optional [H, W, 3] MoGe-2 camera-frame point map.  When supplied
                (with ``moge_mask``), the mesh is re-posed to align with the photo so the
                front-view projection conditions on the correct pixels.  When None, the mesh
                is assumed already front-aligned (original behaviour).
            moge_mask: Optional [H, W] bool object-foreground mask for ``moge_points``.

        Returns:
            trimesh.Trimesh with PBR texture atlas attached.
        """
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()

        camera_angle_x = camera_params['camera_angle_x']
        distance       = camera_params['distance']
        mesh_scale     = camera_params.get('mesh_scale', 1.0)

        if preprocess_image:
            image = self.preprocess_image(image)

        mesh = self.preprocess_mesh(mesh)

        # Pose alignment: re-pose the mesh into the canonical front frame so the hardcoded
        # front-view projection samples the correct pixels.  No-op / identity if not supplied
        # or registration is low-confidence.
        self._align_R = np.eye(3, dtype=np.float64)
        if moge_points is not None and moge_mask is not None:
            mesh = self.align_mesh_to_image(mesh, moge_points, moge_mask)

        torch.manual_seed(seed)

        # Stage 1: encode mesh geometry → shape SLat
        shape_slat = self.encode_shape_slat(mesh, resolution)

        # Stage 2: proj-mode image conditioning aligned to shape SLat coords
        cond = self.get_proj_cond(
            [image],
            shape_slat.coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=resolution // 16,
        )

        # Stage 3: run texture DIT
        tex_slat = self.sample_tex_slat(cond, shape_slat, tex_slat_sampler_params)

        # Stage 4: decode to PBR voxels
        pbr_voxel = self.decode_tex_slat(tex_slat)

        # Stage 5: UV-unwrap + bake into a textured trimesh
        return self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)
