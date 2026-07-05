"""
Pose estimation for the proj-mode texturing pipeline.

The texture DIT back-projects pixel features onto the voxel grid using a *hardcoded
front-view camera* (see ``ProjGrid.front_view_transform_matrix`` in
``trainers/flow_matching/mixins/image_conditioned_proj.py``).  That assumes the input
image was shot from the mesh's canonical -Y front.  For an externally-supplied mesh
(e.g. a Hunyuan3D output) paired with a natural photo, that assumption is violated and
the projection samples the wrong pixels.

This module recovers the rotation that aligns the mesh with the photo so the mesh can be
re-posed into the canonical front frame *before* voxelization, making the front-view
projection correct.

Method (no training, reuses what MoGe already computes):
  1. MoGe-2 produces a metric point map of the *visible* surface in its camera frame.
  2. Map those points into the canonical world frame the projection uses.
  3. Register that partial cloud to a surface sample of the mesh with multi-init ICP
     (scipy cKDTree nearest-neighbour + numpy Kabsch), picking the lowest-RMSE init.
  4. Return the rotation; fall back to identity if registration is low-confidence.

Coordinate conventions (established from ``project_points_to_image_batch`` +
``ProjGrid.front_view_transform_matrix``):
  - Canonical camera: c2w with camera at (0, -distance, 0), looking +Y, up +Z, right +X
    (Blender style, depth = -z_cam).  This is the frame the *preprocessed* mesh lives in.
  - MoGe is OpenCV style (camera at origin, looking +Z, y-down).
  - MoGe-cam -> canonical-world is therefore the proper rotation (x, y, z) -> (x, z, -y).
"""

from typing import Tuple, Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as _Rotation


# Default multi-init grid.  Azimuth (object facing) is the dominant unknown for a front
# photo; elevation/roll are usually small, so a coarse elevation sweep suffices.
_DEFAULT_AZIMUTHS = tuple(range(0, 360, 30))          # 0,30,...,330 about V-up (+Z)
_DEFAULT_ELEVATIONS = (-20.0, 0.0, 20.0)               # about +X

# Confidence gate.  Geometric partial-to-full registration only *reliably* succeeds when
# it converges tightly; the fraction of MoGe points with a nearby visible-mesh match
# cleanly separates a true fit (~0.9+) from a stuck local minimum (~0.4).  Below this we
# return identity (front-view) rather than apply a wrong rotation — "align when confident".
_MIN_INLIER_FRAC = 0.6
_INLIER_DIST = 0.05                                    # normalised-radius units
# Accept the level (azimuth-only) fit unless the free-3D fit's inlier fraction beats it by
# more than this — i.e. only tilt out of plane when the data clearly earns it.
_LEVEL_MARGIN = 0.05


def _moge_to_canonical(points: np.ndarray) -> np.ndarray:
    """Map MoGe camera-frame points to the canonical world frame: (x, y, z) -> (x, z, -y)."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x, z, -y], axis=-1)


def _normalize(cloud: np.ndarray) -> np.ndarray:
    """Center a point cloud and scale it to unit max-radius (rotation-only registration)."""
    c = cloud - cloud.mean(axis=0, keepdims=True)
    r = np.linalg.norm(c, axis=1).max()
    if r < 1e-8:
        return c
    return c / r


def _umeyama(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Best-fit similarity (scale s, proper rotation R, translation t) with ``s*(A @ R.T)+t ~= B``.

    A, B are [N, 3].  Estimating scale/translation alongside rotation makes the rotation
    robust to the partial-vs-full scale and centroid mismatch (MoGe sees only the front).
    Only R is used downstream; s and t just sharpen the correspondences.
    """
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    Ac = A - mu_A
    Bc = B - mu_B
    H = (Bc.T @ Ac) / len(A)          # column-convention cross-covariance (maps A -> B)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt                    # column convention: R @ a ~= b  (=> row: a @ R.T)
    var_A = (Ac ** 2).sum() / len(A)
    s = float(np.trace(np.diag(D) @ S) / var_A) if var_A > 1e-12 else 1.0
    t = mu_B - s * (R @ mu_A)
    return s, R, t


def _visible(src_normals: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Boolean mask of source points whose (rotated) normal faces the canonical camera.

    The canonical camera sits on -Y looking +Y, so a camera-facing normal points toward
    -Y, i.e. its rotated y-component is negative.  This is the key to partial-to-full
    registration: the MoGe target only contains the front surface, so the mesh's back/side
    faces must never be allowed to match it (otherwise a wrong rotation fits the front
    cloud onto the back and the rotation flips ~180 deg).
    """
    return (src_normals @ R.T)[:, 1] < 0.0


def _icp(
    src: np.ndarray,
    src_normals: np.ndarray,
    tgt: np.ndarray,
    R0: np.ndarray,
    iters: int = 50,
    trim: float = 0.8,
    constrain_up: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Visibility-aware partial->full similarity-ICP.

    Aligns the camera-facing part of the full ``src`` cloud (mesh surface) to the partial
    ``tgt`` cloud (MoGe visible surface).  Returns ``(R, s, t)`` of the similarity
    ``s*(src @ R.T)+t ~= tgt``.  ``tgt`` is the query set since it is the partial view.

    ``constrain_up`` restricts the rotation to azimuth about the up-axis (+Z) each
    iteration — the reliable degree of freedom for a level photo.  It structurally removes
    the spurious pitch/roll that MoGe's unreliable depth otherwise bakes in.
    """
    s, R, t = 1.0, R0.copy(), np.zeros(3)
    if constrain_up:
        R = _project_azimuth(R)
    keep_n = max(4, int(len(tgt) * trim))
    for _ in range(iters):
        vis = _visible(src_normals, R)
        if vis.sum() < 4:
            break
        src_vis = src[vis]
        src_t = s * (src_vis @ R.T) + t
        dist, idx = cKDTree(src_t).query(tgt, k=1)
        # Keep the closest `trim` fraction of correspondences (reject occluded/outlier pts).
        order = np.argsort(dist)[:keep_n]
        A = src_vis[idx[order]]       # matched (visible) source points, untransformed
        B = tgt[order]                # corresponding target points
        s, R_new, t = _umeyama(A, B)
        if constrain_up:
            R_new = _project_azimuth(R_new)
            t = B.mean(0) - s * (A @ R_new.T).mean(0)
        delta = np.linalg.norm(R_new - R)
        R = R_new
        if delta < 1e-7:
            break
    return R, s, t


def _project_azimuth(R: np.ndarray) -> np.ndarray:
    """Nearest rotation about the up-axis (+Z) to ``R`` (removes pitch/roll, keeps azimuth)."""
    theta = np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _chamfer(src: np.ndarray, src_normals: np.ndarray, tgt: np.ndarray,
             s: float, R: np.ndarray, t: np.ndarray) -> float:
    """Symmetric chamfer between the camera-facing ``s*(src @ R.T)+t`` and ``tgt``.

    Only the visible source surface is scored (matching how ``tgt`` was formed).
    Bidirectional so a degenerate scale-shrink (src collapsing into a tgt cluster) is
    penalised by the src->tgt term.
    """
    vis = _visible(src_normals, R)
    if vis.sum() < 4:
        return float("inf")
    src_t = s * (src[vis] @ R.T) + t
    d_ts, _ = cKDTree(src_t).query(tgt, k=1)   # every tgt point covered by visible src?
    d_st, _ = cKDTree(tgt).query(src_t, k=1)   # every visible src point near tgt? (spill)
    return float(np.sqrt((d_ts ** 2).mean()) + np.sqrt((d_st ** 2).mean()))


def estimate_alignment_rotation(
    mesh_v: trimesh.Trimesh,
    moge_points: np.ndarray,
    valid_mask: np.ndarray,
    n_samples: int = 5000,
    azimuths=_DEFAULT_AZIMUTHS,
    elevations=_DEFAULT_ELEVATIONS,
    min_inlier_frac: float = _MIN_INLIER_FRAC,
    n_refine: int = 5,
    level_weight: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """Estimate the rotation that aligns ``mesh_v`` with the photo.

    Args:
        mesh_v: Mesh already in the canonical V frame (i.e. after ``preprocess_mesh``).
        moge_points: [H, W, 3] MoGe camera-frame point map.
        valid_mask: [H, W] bool mask selecting object foreground AND MoGe-valid pixels.
        n_samples: number of mesh-surface / MoGe points used for registration.
        min_inlier_frac: below this confidence the rotation is rejected (identity fallback).
        n_refine: how many of the best coarse inits to refine on the full clouds.
        level_weight: strength of the "photos are shot roughly level" prior.  Penalises
            tilting the object's up-axis (pitch/roll) — the main unknown is azimuth, and
            MoGe's depth on stylised/flat images is unreliable enough to fake an out-of-plane
            tilt.  Azimuth is unaffected.  0 disables the prior.

    Returns:
        R: [3, 3] rotation such that ``verts @ R.T`` re-poses the mesh to the photo's front.
        info: dict with 'rmse', 'inlier_frac', 'fallback' (bool), 'n_tgt', 'tilt_deg'.
    """
    def _score(chamfer, R):
        # Add a soft penalty for tilting the up-axis (+Z): arccos(R[2,2]) is 0 for pure
        # azimuth and grows with pitch/roll.  Keeps azimuth free, discourages spurious tilt.
        tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
        return chamfer + level_weight * tilt
    identity = np.eye(3, dtype=np.float64)
    rng = np.random.default_rng(0)

    pts = moge_points[valid_mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    info = {"rmse": float("inf"), "inlier_frac": 0.0, "fallback": True, "n_tgt": int(len(pts))}
    if len(pts) < 50:
        # Not enough visible surface to register against — keep front-view behaviour.
        return identity, info

    # Target: MoGe visible surface in canonical frame; Source: mesh surface sample (with
    # per-point normals so the ICP can cull back-faces).
    P_tgt = _normalize(_moge_to_canonical(pts.astype(np.float64)))
    if len(P_tgt) > n_samples:
        P_tgt = P_tgt[rng.choice(len(P_tgt), n_samples, replace=False)]
    src_pts, face_idx = trimesh.sample.sample_surface(mesh_v, n_samples)
    P_src = _normalize(np.asarray(src_pts, dtype=np.float64))
    N_src = np.asarray(mesh_v.face_normals[face_idx], dtype=np.float64)  # unit, scale/shift-free

    # Inits: azimuth (object facing — the dominant unknown) x elevation.
    inits = [
        (_Rotation.from_euler("z", az, degrees=True)
         * _Rotation.from_euler("x", el, degrees=True)).as_matrix()
        for az in azimuths for el in elevations
    ]

    # --- Coarse pass: rank every init cheaply on downsampled clouds. ---
    def _sub(n):
        i = rng.choice(len(P_src), n, replace=False) if len(P_src) > n else slice(None)
        return P_src[i], N_src[i]
    src_c, ncs = _sub(800)
    tgt_c = P_tgt if len(P_tgt) <= 800 else P_tgt[rng.choice(len(P_tgt), 800, replace=False)]
    coarse = []
    for R0 in inits:
        R, s, t = _icp(src_c, ncs, tgt_c, R0, iters=15)
        coarse.append((_score(_chamfer(src_c, ncs, tgt_c, s, R, t), R), R0))
    coarse.sort(key=lambda x: x[0])

    def _confidence(R, s, t):
        vis = _visible(N_src, R)
        if vis.sum() < 4:
            return float("inf"), 0.0
        d, _ = cKDTree(s * (P_src[vis] @ R.T) + t).query(P_tgt, k=1)
        return float(np.sqrt((d ** 2).mean())), float((d < _INLIER_DIST).mean())

    # --- Refine pass: for each promising init, fit both a free-3D rotation and an
    # azimuth-only (level) rotation, on the full clouds. ---
    def _best(constrain):
        bR, bsc, bs, bt = identity, float("inf"), 1.0, np.zeros(3)
        for _, R0 in coarse[:n_refine]:
            R, s, t = _icp(P_src, N_src, P_tgt, R0, iters=80, constrain_up=constrain)
            score = _score(_chamfer(P_src, N_src, P_tgt, s, R, t), R)
            if score < bsc:
                bsc, bR, bs, bt = score, R, s, t
        return bR, bs, bt
    # Prefer the level (azimuth-only) fit: it removes MoGe's spurious out-of-plane tilt while
    # still recovering azimuth (the real unknown).  Only fall back to a free-3D tilt when the
    # level fit can't align at all but a tilted one can — i.e. a genuinely non-level shot.
    R_lvl, s_lvl, t_lvl = _best(True)
    rmse_lvl, inl_lvl = _confidence(R_lvl, s_lvl, t_lvl)
    best_R, best_s, best_t, rmse, inlier_frac = R_lvl, s_lvl, t_lvl, rmse_lvl, inl_lvl
    if inl_lvl < min_inlier_frac:
        R_free, s_free, t_free = _best(False)
        rmse_free, inl_free = _confidence(R_free, s_free, t_free)
        if inl_free >= inl_lvl + _LEVEL_MARGIN:
            best_R, best_s, best_t, rmse, inlier_frac = R_free, s_free, t_free, rmse_free, inl_free

    tilt_deg = float(np.degrees(np.arccos(np.clip(best_R[2, 2], -1.0, 1.0))))
    info.update(rmse=rmse, inlier_frac=inlier_frac, fallback=False, tilt_deg=tilt_deg)
    if inlier_frac < min_inlier_frac:
        # Not confident enough — applying a wrong rotation is worse than the front-view
        # assumption, so fall back to identity.
        info["fallback"] = True
        return identity, info
    return best_R.astype(np.float64), info
