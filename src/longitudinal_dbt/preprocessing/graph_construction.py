#!/usr/bin/env python3
"""Construct quality-controlled multiscale graphs from DBT TIFF volumes.

This module preserves the production three-scale CUDA supervoxel, feature,
edge, augmentation, and serialization pipeline while exposing dataset paths
and run controls through a portable command-line interface.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
import logging
import warnings
import math
import json
import multiprocessing as mp
import glob
import shutil
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
warnings.filterwarnings("ignore", message="`torch\\.cuda\\.amp\\.autocast\\(args\\.\\.\\.\\)` is deprecated")
warnings.filterwarnings("ignore", message="You are using `torch\\.load` with `weights_only=False`")

import numpy as np
import pandas as pd
import tifffile
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import generate_binary_structure, binary_closing, label as cc_label, find_objects

try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    torch.set_num_threads(1)
except Exception:
    pass

autocast = torch.amp.autocast

HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# SLIC backend is implemented with PyTorch CUDA to avoid CuPy/NVRTC/module issues.
GPU_SLIC = True
SLIC_BACKEND = "torch-cuda"

_TV = True
_TV_ERR = ""
_HAS_TV_WEIGHTS = False
try:
    import torchvision
    from torchvision.models import resnet50
    try:
        from torchvision.models import ResNet50_Weights
        _HAS_TV_WEIGHTS = True
    except Exception:
        ResNet50_Weights = None
        _HAS_TV_WEIGHTS = False
except Exception as e:
    _TV = False
    _TV_ERR = str(e)

log = logging.getLogger("longitudinal_dbt.graph_construction")
np.random.seed(123)
torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

_EMBEDDER_CACHE: Dict = {}
_PROJ_CACHE: Dict = {}


def configure_logging(out_dir: str, run_id: str, tag: str):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"run_{run_id}_{tag}.log")
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        logging.getLogger("longitudinal_dbt.graph_construction").info(f"Logging to {path}")


def tick():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter()


def safe_fp16(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, copy=False, nan=0.0, posinf=65500.0, neginf=-65500.0)
    np.clip(a, -65500.0, 65500.0, out=a)
    return a.astype(np.float16, copy=False)


def banner(cfg: Dict, device_id: int):
    g = torch.cuda.is_available()
    n = torch.cuda.device_count() if g else 0
    knn_backend = "faiss-gpu" if HAS_FAISS else "torch-gpu"
    weights_name = cfg["embed"].get("weights_path") or "imagenet"
    msg = [
        f"Torch CUDA: {g} (devices={n}) | Using device: cuda:{device_id}",
        f"GPU SLIC backend: {SLIC_BACKEND}",
        f"kNN backend: {knn_backend} (batch={cfg['knn']['batch']})",
        f"TorchVision: {_TV} (embed={cfg['embed']['enabled']}, amp={cfg['embed']['amp']}, weights={os.path.basename(weights_name)})",
    ]
    log.info(" | ".join(msg))


def load_volume(path: str) -> np.ndarray:
    def _to_slices(a: np.ndarray) -> list:
        a = np.asarray(a).squeeze()
        if a.ndim == 2:
            return [a.astype(np.float32, copy=False)]
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            g = a[..., :3].mean(axis=-1).astype(np.float32, copy=False)
            return [g]
        if a.ndim == 3:
            z_axis = int(np.argmin(a.shape))
            if z_axis != 0:
                a = np.moveaxis(a, z_axis, 0)
            return [a[z].astype(np.float32, copy=False) for z in range(a.shape[0])]
        if a.ndim == 4 and a.shape[-1] in (3, 4):
            a = a[..., :3].mean(axis=-1).squeeze()
            if a.ndim == 3:
                return [a[z].astype(np.float32, copy=False) for z in range(a.shape[0])]
        if a.ndim == 4:
            z_axis = int(np.argmin(a.shape))
            a = np.moveaxis(a, z_axis, 0).squeeze()
            if a.ndim == 3:
                return [a[z].astype(np.float32, copy=False) for z in range(a.shape[0])]
            if a.ndim > 3:
                index = (slice(None),) + tuple(0 for _ in range(a.ndim - 3))
                a = a[index]
                return [a[z].astype(np.float32, copy=False) for z in range(a.shape[0])]
        raise RuntimeError(f"Unsupported TIFF shape: {a.shape}")

    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".tif", ".tiff"))])
        sl = []
        for f in files:
            try:
                sl.extend(_to_slices(tifffile.imread(f)))
            except Exception:
                continue
        if not sl:
            return np.zeros((0, 0, 0), np.float32)
        H = min(s.shape[0] for s in sl)
        W = min(s.shape[1] for s in sl)
        vol = np.stack([s[:H, :W] for s in sl], 0).astype(np.float32, copy=False)
    else:
        arr = tifffile.imread(path)
        sl = _to_slices(arr)
        H = min(s.shape[0] for s in sl)
        W = min(s.shape[1] for s in sl)
        vol = np.stack([s[:H, :W] for s in sl], 0).astype(np.float32, copy=False)

    if vol.size == 0:
        return np.zeros_like(vol, np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    vol = (vol - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vol, np.float32)
    return vol


def fast_clahe_3d(vol: np.ndarray, clip: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=max(0.001, clip) * 40.0, tileGridSize=(8, 8))
    out = np.empty_like(vol, np.float32)
    for z in range(vol.shape[0]):
        sl = (vol[z] * 255.0).astype(np.uint8)
        out[z] = clahe.apply(sl).astype(np.float32) / 255.0
    return out


def breast_mask_3d(vol: np.ndarray) -> np.ndarray:
    Z, H, W = vol.shape
    m = np.zeros((Z, H, W), dtype=bool)
    for z in range(Z):
        sl = (vol[z] * 255.0).astype(np.uint8)
        t = threshold_otsu(sl) / 255.0
        mz = vol[z] > t
        mz = remove_small_objects(mz, min_size=1000)
        m[z] = mz
    struc = generate_binary_structure(3, 1)
    m = binary_closing(m, structure=struc, iterations=1)
    lab, num = cc_label(m, structure=struc)
    if num > 0:
        cnt = np.bincount(lab.ravel())
        cnt[0] = 0
        m = lab == cnt.argmax()
    return m.astype(np.uint8)


def adaptive_targets(tissue_mask: np.ndarray, spacing_mm: Tuple[float, float, float], lin_scales_mm: Tuple[float, float, float], K_min: Tuple[int, int, int], K_max: Tuple[int, int, int], K_cap_total: int) -> List[int]:
    sz, sy, sx = spacing_mm
    vox_vol = sz * sy * sx
    tissue_vox = int(tissue_mask.sum())
    tissue_mm3 = tissue_vox * vox_vol
    Ks = []
    for L, kmin, kmax in zip(lin_scales_mm, K_min, K_max):
        sv_mm3 = max(1e-6, L ** 3)
        K = int(round(tissue_mm3 / sv_mm3))
        Ks.append(int(np.clip(K, kmin, kmax)))
    S = sum(Ks)
    if S > K_cap_total:
        scale = K_cap_total / float(max(1, S))
        Ks = [max(1000, int(round(k * scale))) for k in Ks]
    return Ks


def _compress_assigned_labels(out: np.ndarray, K: int) -> Tuple[np.ndarray, int]:
    valid = out >= 0
    if not valid.any():
        return np.full_like(out, -1, dtype=np.int32), 0
    vals = out[valid].astype(np.int32, copy=False)
    used = np.bincount(vals, minlength=K)
    alive = used > 0
    remap = np.full(K, -1, np.int32)
    remap[alive] = np.arange(int(alive.sum()), dtype=np.int32)
    lab = np.full_like(out, -1, dtype=np.int32)
    lab[valid] = remap[vals]
    return lab, int(alive.sum())


def gpu_slic3d(vol, mask, n_segments, compactness, sigma, spacing, max_iter=8, batch_vox=400000, return_profile=False):
    """
    PyTorch-CUDA replacement for the previous CuPy SLIC-like 3D supervoxel code.

    Same inputs/outputs as the old function:
        lab, K
        or lab, K, profile when return_profile=True

    This removes the dependency on CuPy and libnvrtc.so.
    """
    import numpy as np

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for torch-cuda SLIC")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def ctick():
        torch.cuda.synchronize(device)
        return time.perf_counter()

    prof = {"init": 0.0, "iter": 0.0, "assign": 0.0, "relabel": 0.0, "total": 0.0}
    t_total0 = ctick()

    Z, H, W = vol.shape
    m = mask > 0
    n = int(m.sum())

    if n == 0 or int(n_segments) <= 0:
        lab_empty = np.full((Z, H, W), -1, np.int32)
        if return_profile:
            prof["total"] = ctick() - t_total0
            return lab_empty, 0, prof
        return lab_empty, 0

    t0 = ctick()

    zz, yy, xx = np.nonzero(m)
    zmin, zmax = int(zz.min()), int(zz.max())
    ymin, ymax = int(yy.min()), int(yy.max())
    xmin, xmax = int(xx.min()), int(xx.max())

    sz, sy, sx = map(float, spacing)
    tissue_mm3 = float(n) * (sz * sy * sx)
    K_req = int(min(int(n_segments), n))
    L_mm = float(max(1e-6, (tissue_mm3 / float(K_req)) ** (1.0 / 3.0)))

    dz = max(1, int(round(L_mm / sz)))
    dy = max(1, int(round(L_mm / sy)))
    dx = max(1, int(round(L_mm / sx)))

    z_start = zmin + dz // 2
    y_start = ymin + dy // 2
    x_start = xmin + dx // 2

    z_cent = np.arange(z_start, zmax + 1, dz, dtype=np.int32)
    y_cent = np.arange(y_start, ymax + 1, dy, dtype=np.int32)
    x_cent = np.arange(x_start, xmax + 1, dx, dtype=np.int32)

    Nz, Ny, Nx = int(len(z_cent)), int(len(y_cent)), int(len(x_cent))

    if Nz == 0 or Ny == 0 or Nx == 0:
        lab_empty = np.full((Z, H, W), -1, np.int32)
        if return_profile:
            prof["init"] = ctick() - t0
            prof["total"] = ctick() - t_total0
            return lab_empty, 0, prof
        return lab_empty, 0

    K = int(Nz * Ny * Nx)

    grid_idx_np = np.arange(K, dtype=np.int64).reshape(Nz, Ny, Nx)
    Zc, Yc, Xc = np.meshgrid(z_cent, y_cent, x_cent, indexing="ij")

    cz = torch.from_numpy(Zc.reshape(-1).astype(np.float32, copy=False)).to(device)
    cy = torch.from_numpy(Yc.reshape(-1).astype(np.float32, copy=False)).to(device)
    cx = torch.from_numpy(Xc.reshape(-1).astype(np.float32, copy=False)).to(device)
    grid_idx = torch.from_numpy(grid_idx_np).to(device=device, dtype=torch.long)

    v_np = vol.astype(np.float32, copy=False)
    if float(sigma) > 0.0:
        from scipy.ndimage import gaussian_filter
        v_np = gaussian_filter(v_np, sigma=float(sigma)).astype(np.float32, copy=False)

    vg = torch.from_numpy(v_np).to(device=device, dtype=torch.float32)

    cz_i = torch.clamp(torch.round(cz).long(), 0, Z - 1)
    cy_i = torch.clamp(torch.round(cy).long(), 0, H - 1)
    cx_i = torch.clamp(torch.round(cx).long(), 0, W - 1)
    ci = vg[cz_i, cy_i, cx_i].float()

    w = float(compactness) / float(max(1e-6, L_mm))
    w2 = float(w * w)

    zz = zz.astype(np.int64, copy=False)
    yy = yy.astype(np.int64, copy=False)
    xx = xx.astype(np.int64, copy=False)
    n = int(len(zz))
    batch_vox = max(1, int(batch_vox))

    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    z0 = int(z_start)
    y0 = int(y_start)
    x0 = int(x_start)

    prof["init"] = ctick() - t0

    t0 = ctick()

    with torch.inference_mode():
        for _ in range(int(max_iter)):
            counts = torch.zeros((K,), device=device, dtype=torch.float32)
            sum_z = torch.zeros((K,), device=device, dtype=torch.float32)
            sum_y = torch.zeros((K,), device=device, dtype=torch.float32)
            sum_x = torch.zeros((K,), device=device, dtype=torch.float32)
            sum_i = torch.zeros((K,), device=device, dtype=torch.float32)

            for s in range(0, n, batch_vox):
                e = min(s + batch_vox, n)

                zc = torch.from_numpy(zz[s:e]).to(device=device, dtype=torch.long, non_blocking=True)
                yc = torch.from_numpy(yy[s:e]).to(device=device, dtype=torch.long, non_blocking=True)
                xc = torch.from_numpy(xx[s:e]).to(device=device, dtype=torch.long, non_blocking=True)

                gz = torch.clamp((zc - z0) // dz, 0, Nz - 1)
                gy = torch.clamp((yc - y0) // dy, 0, Ny - 1)
                gx = torch.clamp((xc - x0) // dx, 0, Nx - 1)

                I = vg[zc, yc, xc].float()
                zcf = zc.float()
                ycf = yc.float()
                xcf = xc.float()

                best_d = torch.full((e - s,), float("inf"), device=device, dtype=torch.float32)
                best_k = torch.zeros((e - s,), device=device, dtype=torch.long)

                for doz, doy, dox in offsets:
                    gzz = torch.clamp(gz + int(doz), 0, Nz - 1)
                    gyy = torch.clamp(gy + int(doy), 0, Ny - 1)
                    gxx = torch.clamp(gx + int(dox), 0, Nx - 1)

                    kk = grid_idx[gzz, gyy, gxx]

                    czz = cz[kk]
                    cyy = cy[kk]
                    cxx = cx[kk]
                    cii = ci[kk]

                    dzm = (zcf - czz) * float(sz)
                    dym = (ycf - cyy) * float(sy)
                    dxm = (xcf - cxx) * float(sx)
                    ds2 = dzm * dzm + dym * dym + dxm * dxm
                    dc2 = (I - cii) * (I - cii)
                    d = dc2 + float(w2) * ds2

                    upd = d < best_d
                    best_d = torch.where(upd, d, best_d)
                    best_k = torch.where(upd, kk, best_k)

                counts += torch.bincount(best_k, minlength=K).to(torch.float32)
                sum_z += torch.bincount(best_k, weights=zcf, minlength=K).to(torch.float32)
                sum_y += torch.bincount(best_k, weights=ycf, minlength=K).to(torch.float32)
                sum_x += torch.bincount(best_k, weights=xcf, minlength=K).to(torch.float32)
                sum_i += torch.bincount(best_k, weights=I, minlength=K).to(torch.float32)

            nz = counts > 0
            denom = torch.clamp(counts, min=1.0)
            cz = torch.where(nz, sum_z / denom, cz).float()
            cy = torch.where(nz, sum_y / denom, cy).float()
            cx = torch.where(nz, sum_x / denom, cx).float()
            ci = torch.where(nz, sum_i / denom, ci).float()

    prof["iter"] = ctick() - t0

    t0 = ctick()
    out = np.full((Z, H, W), -1, np.int32)

    with torch.inference_mode():
        for s in range(0, n, batch_vox):
            e = min(s + batch_vox, n)

            zc = torch.from_numpy(zz[s:e]).to(device=device, dtype=torch.long, non_blocking=True)
            yc = torch.from_numpy(yy[s:e]).to(device=device, dtype=torch.long, non_blocking=True)
            xc = torch.from_numpy(xx[s:e]).to(device=device, dtype=torch.long, non_blocking=True)

            gz = torch.clamp((zc - z0) // dz, 0, Nz - 1)
            gy = torch.clamp((yc - y0) // dy, 0, Ny - 1)
            gx = torch.clamp((xc - x0) // dx, 0, Nx - 1)

            I = vg[zc, yc, xc].float()
            zcf = zc.float()
            ycf = yc.float()
            xcf = xc.float()

            best_d = torch.full((e - s,), float("inf"), device=device, dtype=torch.float32)
            best_k = torch.zeros((e - s,), device=device, dtype=torch.long)

            for doz, doy, dox in offsets:
                gzz = torch.clamp(gz + int(doz), 0, Nz - 1)
                gyy = torch.clamp(gy + int(doy), 0, Ny - 1)
                gxx = torch.clamp(gx + int(dox), 0, Nx - 1)

                kk = grid_idx[gzz, gyy, gxx]

                czz = cz[kk]
                cyy = cy[kk]
                cxx = cx[kk]
                cii = ci[kk]

                dzm = (zcf - czz) * float(sz)
                dym = (ycf - cyy) * float(sy)
                dxm = (xcf - cxx) * float(sx)
                ds2 = dzm * dzm + dym * dym + dxm * dxm
                dc2 = (I - cii) * (I - cii)
                d = dc2 + float(w2) * ds2

                upd = d < best_d
                best_d = torch.where(upd, d, best_d)
                best_k = torch.where(upd, kk, best_k)

            out[zz[s:e], yy[s:e], xx[s:e]] = best_k.detach().cpu().numpy().astype(np.int32, copy=False)

    prof["assign"] = ctick() - t0

    t0 = ctick()
    lab, K2 = _compress_assigned_labels(out, K)
    prof["relabel"] = ctick() - t0
    prof["total"] = ctick() - t_total0

    if return_profile:
        return lab, K2, prof
    return lab, K2


def slic3d_multiscale(vol, mask, targets, compactness, sigma, spacing):
    results = {}
    for target_K in targets:
        key = f"K{int(target_K)}"
        lab, K, prof = gpu_slic3d(vol, mask, int(target_K), float(compactness), float(sigma), tuple(spacing), return_profile=True)
        results[key] = {"lab": lab, "K": int(K), "profile": prof}
    return results


def slice_membership_and_bounds(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z, H, W = lab.shape
    valid = lab >= 0
    if not valid.any():
        return np.zeros(1, np.int32), np.zeros(0, np.int16), np.zeros((0, 6), np.int16), np.zeros(0, np.int32)

    K = int(lab.max()) + 1
    per_node = [[] for _ in range(K)]
    for z in range(Z):
        sl = lab[z]
        u = np.unique(sl[sl >= 0])
        for k in u.tolist():
            per_node[int(k)].append(z)

    slice_ptr = np.zeros(K + 1, np.int32)
    total = 0
    for k, lst in enumerate(per_node):
        total += len(lst)
        slice_ptr[k + 1] = total

    slice_idx = np.empty(total, np.int32)
    p = 0
    for lst in per_node:
        n = len(lst)
        if n:
            slice_idx[p:p + n] = lst
            p += n
    if slice_idx.size and int(slice_idx.max()) < (1 << 15):
        slice_idx = slice_idx.astype(np.int16, copy=False)

    sizes = np.bincount(lab[valid].ravel(), minlength=K).astype(np.int32, copy=False)
    lab01 = np.where(valid, lab + 1, 0).astype(np.int32, copy=False)
    objs = find_objects(lab01)
    b = np.zeros((K, 6), np.int32)
    for k, slc in enumerate(objs):
        if slc is None:
            continue
        zsl, ysl, xsl = slc
        b[k] = [zsl.start, ysl.start, xsl.start, zsl.stop, ysl.stop, xsl.stop]
    if b.size and int(b.max(initial=0)) < (1 << 15):
        b = b.astype(np.int16, copy=False)
    return slice_ptr, slice_idx, b, sizes


def features_3d(vol: np.ndarray, lab: np.ndarray, spacing: Tuple[float, float, float]):
    from skimage.feature import graycomatrix, graycoprops
    from scipy.stats import skew, kurtosis as scipy_kurtosis

    lab01 = np.where(lab >= 0, lab + 1, 0).astype(np.int32, copy=False)
    regs = regionprops(lab01, intensity_image=vol)
    Z, H, W = vol.shape
    sz, sy, sx = spacing

    gz, gy, gx = np.gradient(vol.astype(np.float32, copy=False), float(sz), float(sy), float(sx))
    gm = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32, copy=False)

    mask_vox = np.array(np.nonzero(lab >= 0), dtype=np.float32)
    breast_centroid = mask_vox.mean(axis=1) if mask_vox.size else np.zeros(3, dtype=np.float32)

    feats = []
    for r in regs:
        z0, y0, x0, z1, y1, x1 = r.bbox
        roi_mask = (lab[z0:z1, y0:y1, x0:x1] == r.label - 1)
        masked_vol = vol[z0:z1, y0:y1, x0:x1][roi_mask]
        masked_grad = gm[z0:z1, y0:y1, x0:x1][roi_mask]

        i_mean = float(np.mean(masked_vol)) if masked_vol.size else 0.0
        i_std = float(np.std(masked_vol)) if masked_vol.size else 0.0
        i_min = float(np.min(masked_vol)) if masked_vol.size else 0.0
        i_max = float(np.max(masked_vol)) if masked_vol.size else 0.0
        i_med = float(np.median(masked_vol)) if masked_vol.size else 0.0
        i_iqr = float(np.percentile(masked_vol, 75) - np.percentile(masked_vol, 25)) if masked_vol.size else 0.0

        if masked_vol.size > 2:
            i_skew = float(np.nan_to_num(skew(masked_vol), nan=0.0, posinf=0.0, neginf=0.0))
            i_kurt = float(np.nan_to_num(scipy_kurtosis(masked_vol), nan=0.0, posinf=0.0, neginf=0.0))
        else:
            i_skew = 0.0
            i_kurt = 0.0

        g_mean = float(np.mean(masked_grad)) if masked_grad.size else 0.0
        g_std = float(np.std(masked_grad)) if masked_grad.size else 0.0
        g_max = float(np.max(masked_grad)) if masked_grad.size else 0.0
        g_p90 = float(np.percentile(masked_grad, 90)) if masked_grad.size else 0.0

        voxel_count = float(r.area)
        bbox_vol = float(max(1, (z1 - z0) * (y1 - y0) * (x1 - x0)))
        compactness = voxel_count / bbox_vol
        dz_box, dy_box, dx_box = (z1 - z0), (y1 - y0), (x1 - x0)
        sorted_dims = sorted([dz_box, dy_box, dx_box])
        elongation = float(sorted_dims[-1]) / float(max(1, sorted_dims[0]))

        cz = float(r.centroid[0])
        cy = float(r.centroid[1])
        cxp = float(r.centroid[2])
        dist_mm = float(np.sqrt(((cz - breast_centroid[0]) * sz) ** 2 + ((cy - breast_centroid[1]) * sy) ** 2 + ((cxp - breast_centroid[2]) * sx) ** 2))
        norm_z = cz / max(1, Z - 1)
        norm_y = cy / max(1, H - 1)
        norm_x = cxp / max(1, W - 1)

        roi_2d = vol[z0:z1, y0:y1, x0:x1].max(axis=0)
        if roi_2d.size >= 16:
            roi_2d_u8 = (np.clip(roi_2d, 0.0, 1.0) * 63.0).astype(np.uint8, copy=False)
            try:
                glcm = graycomatrix(roi_2d_u8, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
                contrast = float(np.nan_to_num(graycoprops(glcm, 'contrast')[0, 0], nan=0.0, posinf=0.0, neginf=0.0))
                homogeneity = float(np.nan_to_num(graycoprops(glcm, 'homogeneity')[0, 0], nan=0.0, posinf=0.0, neginf=0.0))
                energy = float(np.nan_to_num(graycoprops(glcm, 'energy')[0, 0], nan=0.0, posinf=0.0, neginf=0.0))
                correlation = float(np.nan_to_num(graycoprops(glcm, 'correlation')[0, 0], nan=0.0, posinf=0.0, neginf=0.0))
            except Exception:
                contrast = 0.0
                homogeneity = 0.0
                energy = 0.0
                correlation = 0.0
        else:
            contrast = 0.0
            homogeneity = 0.0
            energy = 0.0
            correlation = 0.0

        f = [
            i_mean, i_std, i_min, i_max, i_med, i_iqr,
            i_skew, i_kurt,
            g_mean, g_std, g_max, g_p90,
            voxel_count, compactness, elongation,
            dist_mm, norm_z, norm_y, norm_x,
            contrast, homogeneity, energy, correlation,
            0.0,
        ]
        feats.append(f)

    X = np.array(feats, dtype=np.float32)
    names = [
        'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'median_intensity', 'iqr_intensity',
        'skewness', 'kurtosis',
        'mean_gradient', 'std_gradient', 'max_gradient', 'p90_gradient',
        'voxel_count', 'compactness', 'elongation',
        'dist_to_centroid_mm', 'norm_z', 'norm_y', 'norm_x',
        'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
        'scale_code',
    ]
    return X, names, regs


def _strip_known_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "teacher.",
        "student.",
        "resnet.",
        "net.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def _extract_best_state_dict(ckpt, own_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    candidates = []
    if isinstance(ckpt, dict):
        candidates.append(ckpt)
        for key in ("state_dict", "model", "teacher", "encoder", "backbone", "student", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                candidates.append(ckpt[key])
    best = {}
    best_n = -1
    for cand in candidates:
        current = {}
        for k, v in cand.items():
            if not torch.is_tensor(v):
                continue
            kk = _strip_known_prefixes(str(k))
            if kk in own_state and own_state[kk].shape == v.shape:
                current[kk] = v
        if len(current) > best_n:
            best = current
            best_n = len(current)
    return best


class ResNet25D:
    def __init__(self, device, n_slices=3, max_side=768, weights_path=None, amp=True):
        if not _TV:
            raise RuntimeError(f"torchvision unavailable: {_TV_ERR}")
        if weights_path:
            m = resnet50(weights=None) if _HAS_TV_WEIGHTS else resnet50(pretrained=False)
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            best = _extract_best_state_dict(ckpt, m.state_dict())
            if len(best) < 50:
                raise RuntimeError(f"Could not map enough checkpoint keys from {weights_path}; matched={len(best)}")
            missing, unexpected = m.load_state_dict(best, strict=False)
            log.info(f"Loaded embed weights from {weights_path} | matched={len(best)} missing={len(missing)} unexpected={len(unexpected)}")
        else:
            try:
                if _HAS_TV_WEIGHTS:
                    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                else:
                    m = resnet50(pretrained=True)
            except Exception:
                m = resnet50(weights=None) if _HAS_TV_WEIGHTS else resnet50(pretrained=False)

        if m.conv1.in_channels != n_slices:
            w = m.conv1.weight.data.mean(dim=1, keepdim=True)
            m.conv1 = nn.Conv2d(n_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                m.conv1.weight.copy_(w.repeat(1, n_slices, 1, 1))

        bb = nn.Sequential(*list(m.children())[:-2]).to(device).eval()
        if amp:
            bb = bb.half()
        for p in bb.parameters():
            p.requires_grad = False
        self.backbone = bb
        self.amp = bool(amp)
        self.n = int(n_slices)
        self.max_side = int(max_side)
        self.device = device
        base = torch.zeros(1, self.n, 1, 1, device=device) + 0.5
        self.mean = base.half() if amp else base
        self.std = base.half() if amp else base

    @torch.inference_mode()
    def _stack(self, vol, z, halfw):
        Z, H, W = vol.shape
        idx = [max(0, min(Z - 1, z + d)) for d in range(-halfw, halfw + 1)]
        return np.stack([vol[i] for i in idx], 0).astype(np.float32)

    @torch.inference_mode()
    def featmap_at(self, vol, z, halfw):
        img = self._stack(vol, z, halfw)
        C, H, W = img.shape
        scale = max(1.0, max(H, W) / float(self.max_side))
        H2, W2 = int(round(H / scale)), int(round(W / scale))
        arr = np.empty((C, H2, W2), np.float32)
        for c in range(C):
            arr[c] = cv2.resize(img[c], (W2, H2), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        t = (t - self.mean.float()) / self.std.float()
        with autocast("cuda", enabled=self.amp):
            t = t.half() if self.amp else t
            return self.backbone(t)


def get_embedder(device: torch.device, cfg_embed: Dict):
    key = (
        int(device.index if device.index is not None else 0),
        int(cfg_embed["n_slices"]),
        int(cfg_embed["max_side"]),
        str(cfg_embed.get("weights_path") or ""),
        bool(cfg_embed["amp"]),
    )
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = ResNet25D(
            device=device,
            n_slices=int(cfg_embed["n_slices"]),
            max_side=int(cfg_embed["max_side"]),
            weights_path=cfg_embed.get("weights_path", None),
            amp=bool(cfg_embed["amp"]),
        )
    return _EMBEDDER_CACHE[key]


def get_projection(device: torch.device, embed_dim: int):
    key = (int(device.index if device.index is not None else 0), int(embed_dim))
    if key not in _PROJ_CACHE:
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((2048, embed_dim)).astype(np.float32)
        q, _ = np.linalg.qr(raw)
        _PROJ_CACHE[key] = torch.from_numpy(q[:, :embed_dim].astype(np.float32, copy=False)).to(device)
    return _PROJ_CACHE[key]


@torch.inference_mode()
def sample_centroids(feat, centers_xy, H, W, device):
    _, C, hf, wf = feat.shape
    y = torch.from_numpy(centers_xy[:, 0].astype(np.float32)).to(device)
    x = torch.from_numpy(centers_xy[:, 1].astype(np.float32)).to(device)
    fy = (y / max(1, (H - 1))) * (hf - 1)
    fx = (x / max(1, (W - 1))) * (wf - 1)
    x0 = torch.clamp(fx.floor().long(), 0, wf - 1)
    x1 = torch.clamp(x0 + 1, 0, wf - 1)
    y0 = torch.clamp(fy.floor().long(), 0, hf - 1)
    y1 = torch.clamp(y0 + 1, 0, hf - 1)
    wx = (fx - x0.float()).unsqueeze(1)
    wy = (fy - y0.float()).unsqueeze(1)
    f = feat[0]
    f00 = f[:, y0, x0].T
    f01 = f[:, y1, x0].T
    f10 = f[:, y0, x1].T
    f11 = f[:, y1, x1].T
    return f00 * (1 - wx) * (1 - wy) + f01 * (1 - wx) * wy + f10 * wx * (1 - wy) + f11 * wx * wy

def deep_embed_all_scales(vol, regs_sets, device, embed_cfg: Dict):
    """
    Extract 2.5D ResNet features at the true supervoxel centroids.

    Important:
    The previous version sampled at bounding-box centers. For irregular
    supervoxels, and especially if a label contains disconnected islands,
    the bounding-box center can be outside the actual region. This version
    samples at regionprops centroid, consistent with graph positions.
    """
    if not _TV:
        raise RuntimeError(f"torchvision unavailable: {_TV_ERR}")

    net = get_embedder(device, embed_cfg)

    embed_dim = int(embed_cfg["dims"])
    n_slices = int(embed_cfg["n_slices"])

    Z, H, W = vol.shape
    halfw = (n_slices - 1) // 2

    z_needed = set()
    centers_all = []

    for regs in regs_sets:
        centers = []

        for r in regs:
            cz, cy, cx = r.centroid

            zc = int(np.rint(cz))
            yc = float(cy)
            xc = float(cx)

            zc = max(0, min(Z - 1, zc))
            yc = max(0.0, min(float(H - 1), yc))
            xc = max(0.0, min(float(W - 1), xc))

            z_needed.add(zc)
            centers.append((yc, xc, zc))

        centers_all.append(centers)

    fmap = {}
    for z in sorted(z_needed):
        fmap[int(z)] = net.featmap_at(vol, int(z), halfw)

    P = get_projection(device, embed_dim)
    names = [f"embed_{i}" for i in range(embed_dim)]

    outs = []

    for centers in centers_all:
        if not centers:
            outs.append((np.zeros((0, embed_dim), np.float32), names))
            continue

        cyx = np.array([(y, x) for (y, x, _) in centers], dtype=np.float32)
        zs = np.array([z for (_, _, z) in centers], dtype=np.int32)

        X = torch.zeros((len(centers), embed_dim), device=device)

        for z in np.unique(zs):
            idx = np.where(zs == z)[0]
            Fm = fmap[int(z)]
            V = sample_centroids(Fm, cyx[idx], H, W, device).float()
            X[idx, :] = V @ P

        outs.append((X.detach().cpu().numpy().astype(np.float32, copy=False), names))

    return outs


def rag_edges_and_contacts(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = lab >= 0
    if not valid.any():
        return np.empty((2, 0), np.int64), np.empty((0,), np.float32)

    K = int(lab.max()) + 1
    packed_parts = []

    for axis in (0, 1, 2):
        B = np.roll(lab, -1, axis=axis)
        m = (lab != B) & (lab >= 0) & (B >= 0)

        if axis == 0:
            m[-1, :, :] = False
        elif axis == 1:
            m[:, -1, :] = False
        else:
            m[:, :, -1] = False

        if not m.any():
            continue

        i = lab[m].astype(np.int64, copy=False)
        j = B[m].astype(np.int64, copy=False)

        a = np.minimum(i, j)
        b = np.maximum(i, j)
        packed_parts.append(a * K + b)

    if not packed_parts:
        return np.empty((2, 0), np.int64), np.empty((0,), np.float32)

    packed = np.concatenate(packed_parts)
    packed.sort()

    change = np.empty(packed.size, dtype=bool)
    change[0] = True
    change[1:] = packed[1:] != packed[:-1]

    idx = np.flatnonzero(change)
    uniq = packed[idx]

    counts_undirected = np.diff(np.r_[idx, packed.size]).astype(np.float32)

    src = (uniq // K).astype(np.int64, copy=False)
    dst = (uniq % K).astype(np.int64, copy=False)

    E = np.hstack([
        np.vstack([src, dst]),
        np.vstack([dst, src]),
    ]).astype(np.int64, copy=False)

    counts = np.concatenate([counts_undirected, counts_undirected]).astype(np.float32, copy=False)

    return E, counts


@torch.inference_mode()
def knn_edges_gpu_torch(X: np.ndarray, k: int, device: torch.device, batch: int) -> np.ndarray:
    if k <= 0 or X.shape[0] <= 1:
        return np.zeros((2, 0), np.int64)

    Xn = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    if HAS_FAISS:
        Xf = Xn.astype(np.float32, copy=False)
        norms = np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-6
        Xf = Xf / norms

        N, D = Xf.shape
        kk = min(k + 1, N)

        try:
            gpu_id = int(device.index if device.index is not None else torch.cuda.current_device())
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(D)
            index = faiss.index_cpu_to_gpu(res, gpu_id, index_flat)
            index.add(Xf)
            _, inds = index.search(Xf, kk)
            inds = inds.astype(np.int64, copy=False)
        except Exception:
            index = faiss.IndexFlatIP(D)
            index.add(Xf)
            _, inds = index.search(Xf, kk)
            inds = inds.astype(np.int64, copy=False)

        if kk <= 1:
            return np.zeros((2, 0), np.int64)

        neigh = inds[:, 1:kk]
        src = np.repeat(np.arange(N, dtype=np.int64), neigh.shape[1])
        dst = neigh.reshape(-1)

        edges = np.stack([src, dst], axis=0)
        rev = np.stack([dst, src], axis=0)

        return np.concatenate([edges, rev], axis=1)

    Xt = torch.from_numpy(Xn.astype(np.float32, copy=False)).to(device, non_blocking=True)
    Xt = F.normalize(Xt, p=2, dim=1)
    N = Xt.size(0)
    src_parts = []
    dst_parts = []
    step = max(1024, int(batch))

    for s in range(0, N, step):
        e = min(s + step, N)
        Q = Xt[s:e]
        with autocast("cuda", enabled=True):
            S = (Q.half() @ Xt.half().T).float()
        diag = torch.arange(s, e, device=device)
        S[torch.arange(e - s, device=device), diag] = -1e9
        _, inds = torch.topk(S, k=min(k, N - 1), dim=1, largest=True, sorted=False)
        src_parts.append(torch.arange(s, e, device=device).unsqueeze(1).expand_as(inds).reshape(-1))
        dst_parts.append(inds.reshape(-1))

    src = torch.cat(src_parts)
    dst = torch.cat(dst_parts)
    edges = torch.stack([src, dst], dim=0)
    rev = torch.stack([dst, src], dim=0)
    edges = torch.cat([edges, rev], dim=1)
    return edges.detach().cpu().numpy().astype(np.int64, copy=False)

def unique_edge_index(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return edge_index.astype(np.int64, copy=False)
    return np.unique(edge_index.T, axis=0).T.astype(np.int64, copy=False)


def compute_edge_features(pos_mm: np.ndarray, Xn: np.ndarray, edge_index: np.ndarray, contact_counts=None, sizes=None) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.zeros((0, 8), np.float32)
    i, j = edge_index[0], edge_index[1]
    dxyz = pos_mm[j] - pos_mm[i]
    dist = np.linalg.norm(dxyz, axis=1, keepdims=True)
    same_z = (np.abs(dxyz[:, 2]) < 1e-6).astype(np.float32).reshape(-1, 1)
    fdist = np.linalg.norm(Xn[j] - Xn[i], axis=1, keepdims=True)
    contact = np.zeros((edge_index.shape[1], 1), np.float32)
    cratio = np.zeros((edge_index.shape[1], 1), np.float32)
    if contact_counts is not None and sizes is not None:
        si = sizes[i].astype(np.float32, copy=False)
        sj = sizes[j].astype(np.float32, copy=False)
        contact = contact_counts.reshape(-1, 1).astype(np.float32, copy=False)
        cratio = contact / np.maximum(1.0, np.minimum(si, sj)).reshape(-1, 1)
    return np.concatenate([dxyz.astype(np.float32), dist.astype(np.float32), same_z, contact, cratio, fdist.astype(np.float32)], 1)


def map_parents_by_mode_fast(lab_child: np.ndarray, lab_parent: np.ndarray, K_child: int) -> np.ndarray:
    m = (lab_child >= 0) & (lab_parent >= 0)
    if not m.any():
        return np.full(K_child, -1, np.int32)
    child = lab_child[m].astype(np.int64, copy=False)
    parent = lab_parent[m].astype(np.int64, copy=False)
    base = int(parent.max()) + 1
    packed = child * base + parent
    packed.sort()
    change = np.empty(packed.size, dtype=bool)
    change[0] = True
    change[1:] = packed[1:] != packed[:-1]
    idx = np.flatnonzero(change)
    uniq = packed[idx]
    counts = np.diff(np.r_[idx, packed.size])
    ch = (uniq // base).astype(np.int64, copy=False)
    pa = (uniq % base).astype(np.int64, copy=False)
    out = np.full(K_child, -1, np.int32)
    starts = np.r_[0, np.flatnonzero(np.diff(ch)) + 1]
    ends = np.r_[starts[1:], len(ch)]
    for s, e in zip(starts, ends):
        j = s + int(np.argmax(counts[s:e]))
        out[int(ch[j])] = int(pa[j])
    return out


def cross_scale_edges(parent_map: np.ndarray,
                      child_offset: int,
                      parent_offset: int) -> np.ndarray:
    """
    Build bidirectional hierarchy edges between adjacent graph scales.

    Parameters
    ----------
    parent_map : np.ndarray, shape (K_child,)
        parent_map[c] gives the parent label index of child node c
        in the parent scale. Invalid children should have value -1.

    child_offset : int
        Global node-index offset of the child scale.

    parent_offset : int
        Global node-index offset of the parent scale.

    Returns
    -------
    np.ndarray, shape (2, 2 * number_of_valid_children)
        Directed bidirectional hierarchy edges:
        child_global -> parent_global and parent_global -> child_global.
    """
    valid = np.flatnonzero(parent_map >= 0)

    if valid.size == 0:
        return np.zeros((2, 0), np.int64)

    child_global = child_offset + valid.astype(np.int64, copy=False)
    parent_global = parent_offset + parent_map[valid].astype(np.int64, copy=False)

    return np.stack(
        [
            np.concatenate([child_global, parent_global]),
            np.concatenate([parent_global, child_global]),
        ],
        axis=0,
    ).astype(np.int64, copy=False)

def qc_metrics(N: int, edge_index_qc: np.ndarray, tissue_mask: np.ndarray, covered_mask: np.ndarray) -> Dict:
    tissue_voxels = int(tissue_mask.sum())
    labeled_voxels = int((covered_mask & (tissue_mask > 0)).sum())
    tissue_coverage = labeled_voxels / max(1, tissue_voxels)
    if N == 0 or edge_index_qc.shape[1] == 0:
        return {"tissue_coverage": tissue_coverage, "avg_degree": 0.0, "lcc_frac": 0.0}
    degrees = np.bincount(edge_index_qc.ravel(), minlength=N)
    avg_degree = float(np.mean(degrees))
    adj = sparse.coo_matrix((np.ones(edge_index_qc.shape[1]), (edge_index_qc[0], edge_index_qc[1])), shape=(N, N)).tocsr()
    n_components, labels = connected_components(adj, directed=False)
    lcc_size = int(np.bincount(labels).max()) if n_components > 0 else 0
    lcc_frac = lcc_size / float(N) if N > 0 else 0.0
    return {"tissue_coverage": tissue_coverage, "avg_degree": avg_degree, "lcc_frac": lcc_frac}

def relation_audit_metrics(K_fine: int,
                           K_med: int,
                           K_coarse: int,
                           edge_index: np.ndarray,
                           edge_type: np.ndarray) -> Dict:
    N = int(K_fine + K_med + K_coarse)

    out = {
        "edge_oob_count": 0,
        "self_loop_count": 0,
        "E_rag": 0,
        "E_knn": 0,
        "E_hier": 0,
        "hier_fine_med": 0,
        "hier_med_coarse": 0,
        "hier_fine_coarse_INVALID": 0,
        "hier_same_scale_INVALID": 0,
        "hier_invalid_frac": 0.0,
        "rag_missing_reverse_frac": 0.0,
    }

    if edge_index.shape[1] == 0:
        return out

    ei = edge_index.astype(np.int64, copy=False)
    et = edge_type.astype(np.int64, copy=False)

    out["edge_oob_count"] = int(((ei < 0) | (ei >= N)).sum())
    out["self_loop_count"] = int((ei[0] == ei[1]).sum())

    out["E_rag"] = int((et == 0).sum())
    out["E_knn"] = int((et == 1).sum())
    out["E_hier"] = int((et == 2).sum())

    scale = np.concatenate([
        np.zeros(int(K_fine), dtype=np.int8),
        np.ones(int(K_med), dtype=np.int8),
        2 * np.ones(int(K_coarse), dtype=np.int8),
    ])

    h = ei[:, et == 2]
    if h.shape[1] > 0 and out["edge_oob_count"] == 0:
        s0 = scale[h[0]]
        s1 = scale[h[1]]

        fine_med = ((s0 == 0) & (s1 == 1)) | ((s0 == 1) & (s1 == 0))
        med_coarse = ((s0 == 1) & (s1 == 2)) | ((s0 == 2) & (s1 == 1))
        fine_coarse = ((s0 == 0) & (s1 == 2)) | ((s0 == 2) & (s1 == 0))
        same_scale = s0 == s1

        out["hier_fine_med"] = int(fine_med.sum())
        out["hier_med_coarse"] = int(med_coarse.sum())
        out["hier_fine_coarse_INVALID"] = int(fine_coarse.sum())
        out["hier_same_scale_INVALID"] = int(same_scale.sum())
        out["hier_invalid_frac"] = float((fine_coarse.sum() + same_scale.sum()) / max(1, h.shape[1]))

    r = ei[:, et == 0]
    if r.shape[1] > 0:
        edge_set = set(map(tuple, r.T.tolist()))
        missing = 0
        for u, v in edge_set:
            if (v, u) not in edge_set:
                missing += 1
        out["rag_missing_reverse_frac"] = float(missing / max(1, len(edge_set)))

    return out

def gate(qc: Dict, cfg: Dict) -> Tuple[bool, str]:
    if qc["tissue_coverage"] < cfg["qc"]["min_tissue_coverage"]:
        return False, "low_tissue_coverage"
    if qc["lcc_frac"] < cfg["qc"]["min_lcc_frac"]:
        return False, "disconnected_graph"
    if qc["avg_degree"] < cfg["qc"]["min_avg_degree"]:
        return False, "low_avg_degree"
    return True, ""


def any_cached_for_group(out_dir: str, pid: str, view: str) -> Optional[str]:
    pat = os.path.join(out_dir, f"{pid}__{view}__*.pt")
    for pt in glob.glob(pat):
        npz = pt.replace(".pt", "__labels.npz")
        if os.path.exists(npz):
            return pt
    return None


def augmix_3d(vol: np.ndarray, severity=3, width=3, depth=-1, alpha=1.0, m=0.2) -> np.ndarray:
    ops = [
        lambda a, s: np.clip(a + np.random.normal(0, [0.01, 0.02, 0.03, 0.04, 0.05][max(1, s) - 1], a.shape), 0, 1),
        lambda a, s: np.clip((a - a.mean()) * ([0.9, 0.95, 1.05, 1.1, 1.2][max(1, s) - 1]) + a.mean(), 0, 1),
        lambda a, s: cv2.GaussianBlur(a, (0, 0), [0.5, 1.0, 1.5, 2.0, 2.5][max(1, s) - 1]),
    ]
    ws = np.random.dirichlet([alpha] * width)
    mix = np.zeros_like(vol, dtype=np.float32)
    for w in ws:
        aug = vol.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = ops[np.random.randint(len(ops))]
            aug = np.stack([np.clip(op(aug[z], severity), 0, 1) for z in range(aug.shape[0])], axis=0)
        mix += w * aug
    lam = np.random.beta(m, m)
    return (lam * vol + (1 - lam) * mix).astype(np.float32, copy=False)


def diffusion_like(vol: np.ndarray, sigma=0.05, denoiser="nlmeans", nlmeans_h=0.06) -> np.ndarray:
    sigma = float(sigma)
    noisy = np.clip(vol + np.random.normal(0, sigma, vol.shape), 0, 1).astype(np.float32)

    if denoiser == "nlmeans":
        from skimage.restoration import denoise_nl_means

        h = float(nlmeans_h)
        if h > 1.0:
            h = max(1e-6, 1.2 * sigma)

        out = np.empty_like(noisy, dtype=np.float32)

        for z in range(noisy.shape[0]):
            out[z] = denoise_nl_means(
                noisy[z],
                h=h,
                fast_mode=True,
                patch_size=5,
                patch_distance=6,
                channel_axis=None,
            ).astype(np.float32)

        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    return noisy


def craft_adversarial_graph_features_modelfree(data: Data, eps=0.5, steps=3, norm="linf") -> np.ndarray:
    x = data.x.detach().cpu().numpy().astype(np.float32)
    N = x.shape[0]

    ei = data.edge_index.detach().cpu().numpy()
    row, col = ei[0], ei[1]

    A = sparse.coo_matrix(
        (np.ones_like(row, dtype=np.float32), (row, col)),
        shape=(N, N),
    ).tocsr()
    A = A.maximum(A.T)

    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)

    names = list(getattr(data, "x_names", []))

    protected = {
        "voxel_count",
        "compactness",
        "elongation",
        "dist_to_centroid_mm",
        "norm_z",
        "norm_y",
        "norm_x",
        "scale_code",
    }

    if names and len(names) == x.shape[1]:
        perturb_idx = [i for i, name in enumerate(names) if name not in protected]
    else:
        perturb_idx = list(range(x.shape[1]))

    if len(perturb_idx) == 0:
        return x

    X0 = x[:, perturb_idx].copy()
    mu = X0.mean(axis=0, keepdims=True)
    sd = X0.std(axis=0, keepdims=True) + 1e-6

    Z = (X0 - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    alpha = float(eps) / float(max(1, int(steps)))

    for _ in range(max(1, int(steps))):
        AZ = A.dot(Z)
        LZ = (deg[:, None] * Z) - AZ

        if norm == "linf":
            step = alpha * np.sign(LZ)
        else:
            denom = np.linalg.norm(LZ, axis=1, keepdims=True) + 1e-6
            step = alpha * (LZ / denom)

        Z = Z + step
        Z = np.clip(Z, -3.0, 3.0)

    X_adv = x.copy()
    X_adv[:, perturb_idx] = (Z * sd + mu).astype(np.float32)

    if names and "scale_code" in names:
        scale_idx = names.index("scale_code")
        X_adv[:, scale_idx] = x[:, scale_idx]

    return X_adv.astype(np.float32, copy=False)

def build_graph_from_volume(vol: np.ndarray, tissue: np.ndarray, pid: str, view: str, label: int, cfg: Dict, aug_meta: Dict, device: torch.device, out_path: str, skip_group: bool = True) -> Tuple[Optional[str], Dict]:
    if os.path.exists(out_path) and os.path.exists(out_path.replace(".pt", "__labels.npz")):
        return out_path, {
            "PatientID": pid,
            "View": view,
            "status": "skip_exists",
            "t_slic": 0.0,
            "t_slic_fine": 0.0,
            "t_slic_med": 0.0,
            "t_slic_coarse": 0.0,
            "t_membership": 0.0,
            "t_features": 0.0,
            "t_deep": 0.0,
            "t_edges": 0.0,
            "t_edge_feats": 0.0,
            "t_cross_scale": 0.0,
            "t_qc": 0.0,
            "t_save": 0.0,
            "gpu_mem_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        }

    torch.cuda.reset_peak_memory_stats(device)
    T: Dict[str, float] = {}

    t0 = tick()
    Ks = adaptive_targets(tissue, cfg["spacing_mm"], cfg["slic"]["lin_scales_mm"], cfg["slic"]["K_min"], cfg["slic"]["K_max"], cfg["slic"]["K_cap_total"])
    scales = slic3d_multiscale(vol, tissue, Ks, cfg["slic"]["compactness"], cfg["slic"]["sigma"], tuple(cfg["spacing_mm"]))
    T["slic"] = tick() - t0
    T["slic_fine"] = scales[f"K{Ks[0]}"]["profile"]["total"]
    T["slic_med"] = scales[f"K{Ks[1]}"]["profile"]["total"]
    T["slic_coarse"] = scales[f"K{Ks[2]}"]["profile"]["total"]

    lab_fine, K_fine = scales[f"K{Ks[0]}"]["lab"], scales[f"K{Ks[0]}"]["K"]
    lab_med, K_med = scales[f"K{Ks[1]}"]["lab"], scales[f"K{Ks[1]}"]["K"]
    lab_coarse, K_coarse = scales[f"K{Ks[2]}"]["lab"], scales[f"K{Ks[2]}"]["K"]
    if K_fine <= 0 or K_med <= 0 or K_coarse <= 0:
        raise RuntimeError("no supervoxels")

    t0 = tick()
    spf, sif, nbf, szf = slice_membership_and_bounds(lab_fine)
    spm, sim, nbm, szm = slice_membership_and_bounds(lab_med)
    spc, sic, nbc, szc = slice_membership_and_bounds(lab_coarse)
    T["membership"] = tick() - t0

    t0 = tick()
    Xh_fine, names_h_fine, regs_fine = features_3d(vol, lab_fine, tuple(cfg["spacing_mm"]))
    Xh_med, names_h_med, regs_med = features_3d(vol, lab_med, tuple(cfg["spacing_mm"]))
    Xh_coarse, names_h_coarse, regs_coarse = features_3d(vol, lab_coarse, tuple(cfg["spacing_mm"]))
    T["features"] = tick() - t0

    if Xh_fine.shape[1] > 0:
        Xh_fine[:, -1] = 0.0
    if Xh_med.shape[1] > 0:
        Xh_med[:, -1] = 1.0
    if Xh_coarse.shape[1] > 0:
        Xh_coarse[:, -1] = 2.0

    t0 = tick()
    if cfg["embed"]["enabled"]:
        (Xe_fine, names_e_fine), (Xe_med, names_e_med), (Xe_coarse, names_e_coarse) = deep_embed_all_scales(vol, [regs_fine, regs_med, regs_coarse], device, cfg["embed"])
    else:
        Xe_fine, names_e_fine = np.zeros((len(regs_fine), 0), np.float32), []
        Xe_med, names_e_med = np.zeros((len(regs_med), 0), np.float32), []
        Xe_coarse, names_e_coarse = np.zeros((len(regs_coarse), 0), np.float32), []
    T["deep"] = tick() - t0

    X_fine = np.concatenate([Xh_fine, Xe_fine], 1) if Xe_fine.shape[1] else Xh_fine
    X_med = np.concatenate([Xh_med, Xe_med], 1) if Xe_med.shape[1] else Xh_med
    X_coarse = np.concatenate([Xh_coarse, Xe_coarse], 1) if Xe_coarse.shape[1] else Xh_coarse
    names_fine = names_h_fine + names_e_fine

    if cfg["embed"]["enabled"] and X_fine.shape[1] <= Xh_fine.shape[1]:
        raise RuntimeError("Deep embeddings were requested but were not added to node features")

    sz, sy, sx = cfg["spacing_mm"]

    def get_pos(regs):
        cents = np.array([r.centroid for r in regs], np.float32)
        pos_vox = np.column_stack([cents[:, 2], cents[:, 1], cents[:, 0]]).astype(np.float32)
        pos_mm = np.column_stack([cents[:, 2] * sx, cents[:, 1] * sy, cents[:, 0] * sz]).astype(np.float32)
        return pos_vox, pos_mm

    pvf, pmf = get_pos(regs_fine)
    pvm, pmm = get_pos(regs_med)
    pvc, pmc = get_pos(regs_coarse)

    t0 = tick()
    er_f, cc_f = rag_edges_and_contacts(lab_fine)
    er_m, cc_m = rag_edges_and_contacts(lab_med)
    er_c, cc_c = rag_edges_and_contacts(lab_coarse)
    ek_f = unique_edge_index(knn_edges_gpu_torch(X_fine, cfg["knn"]["k_fine"], device, cfg["knn"]["batch"]))
    ek_m = unique_edge_index(knn_edges_gpu_torch(X_med, cfg["knn"]["k_med"], device, cfg["knn"]["batch"]))
    ek_c = unique_edge_index(knn_edges_gpu_torch(X_coarse, cfg["knn"]["k_coarse"], device, cfg["knn"]["batch"]))
    T["edges"] = tick() - t0

    t0 = tick()
    Xn_f = (X_fine - X_fine.mean(0, keepdims=True)) / (X_fine.std(0, keepdims=True) + 1e-6)
    Xn_m = (X_med - X_med.mean(0, keepdims=True)) / (X_med.std(0, keepdims=True) + 1e-6)
    Xn_c = (X_coarse - X_coarse.mean(0, keepdims=True)) / (X_coarse.std(0, keepdims=True) + 1e-6)
    ef_r_f = compute_edge_features(pmf, Xn_f, er_f, cc_f, szf)
    ef_k_f = compute_edge_features(pmf, Xn_f, ek_f)
    ef_r_m = compute_edge_features(pmm, Xn_m, er_m, cc_m, szm)
    ef_k_m = compute_edge_features(pmm, Xn_m, ek_m)
    ef_r_c = compute_edge_features(pmc, Xn_c, er_c, cc_c, szc)
    ef_k_c = compute_edge_features(pmc, Xn_c, ek_c)
    T["edge_feats"] = tick() - t0

    t0 = tick()
    p_f2m = map_parents_by_mode_fast(lab_fine, lab_med, K_fine)
    p_m2c = map_parents_by_mode_fast(lab_med, lab_coarse, K_med)

    cross_fm = cross_scale_edges(
        p_f2m,
        child_offset=0,
        parent_offset=K_fine,
    )

    cross_mc = cross_scale_edges(
        p_m2c,
        child_offset=K_fine,
        parent_offset=K_fine + K_med,
    )
    T["cross_scale"] = tick() - t0

    X_all = np.vstack([X_fine, X_med, X_coarse])
    pv_all = np.vstack([pvf, pvm, pvc])
    pm_all = np.vstack([pmf, pmm, pmc])
    Nf, Nm = K_fine, K_med
    er_m_off = er_m + Nf
    er_c_off = er_c + Nf + Nm
    ek_m_off = ek_m + Nf
    ek_c_off = ek_c + Nf + Nm

    ei_rag = np.hstack([er_f, er_m_off, er_c_off])
    ea_rag = np.vstack([ef_r_f, ef_r_m, ef_r_c])
    et_rag = np.zeros(ei_rag.shape[1], np.int64)

    ei_knn = np.hstack([ek_f, ek_m_off, ek_c_off])
    ea_knn = np.vstack([ef_k_f, ef_k_m, ef_k_c])
    et_knn = np.ones(ei_knn.shape[1], np.int64)

    Xn_all = (X_all - X_all.mean(0, keepdims=True)) / (X_all.std(0, keepdims=True) + 1e-6)
    ea_fm = compute_edge_features(pm_all, Xn_all, cross_fm)
    ea_mc = compute_edge_features(pm_all, Xn_all, cross_mc)
    et_cross = np.full(cross_fm.shape[1] + cross_mc.shape[1], 2, np.int64)

    ei_all = np.hstack([ei_rag, ei_knn, cross_fm, cross_mc])
    ea_all = np.vstack([ea_rag, ea_knn, ea_fm, ea_mc])
    et_all = np.hstack([et_rag, et_knn, et_cross])

    audit = relation_audit_metrics(
    K_fine=K_fine,
    K_med=K_med,
    K_coarse=K_coarse,
    edge_index=ei_all,
    edge_type=et_all,
    )

    t0 = tick()
    ei_qc = np.hstack([ei_rag, cross_fm, cross_mc])
    covered = ((lab_fine >= 0) | (lab_med >= 0) | (lab_coarse >= 0)).astype(np.uint8)
    N_total = X_all.shape[0]
    qc = qc_metrics(N_total, ei_qc, tissue, covered)
    T["qc"] = tick() - t0

    ok, reason = gate(qc, cfg)
    if ok:
        if audit["edge_oob_count"] > 0:
            ok = False
            reason = "edge_index_oob"
        elif audit["self_loop_count"] > 0:
            ok = False
            reason = "self_loops"
        elif audit["hier_fine_coarse_INVALID"] > 0:
            ok = False
            reason = "invalid_fine_coarse_hierarchy"
        elif audit["hier_same_scale_INVALID"] > 0:
            ok = False
            reason = "invalid_same_scale_hierarchy"
        elif audit["rag_missing_reverse_frac"] > 0.0:
            ok = False
            reason = "rag_missing_reverse_edges"

    row = {
        "PatientID": pid,
        "View": view,
        "K_fine": K_fine,
        "K_med": K_med,
        "K_coarse": K_coarse,
        "K_total": N_total,
        "E_edges": ei_all.shape[1],
        "features_per_node": X_all.shape[1],
        "used_embed": int(X_all.shape[1] > Xh_fine.shape[1]),
        "aug_kind": aug_meta.get("kind", "clean"),
        "aug_params": json.dumps(aug_meta.get("params", {})),
        **qc,
        **audit,
        "status": "ok" if ok else f"reject:{reason}",
        "t_slic": T["slic"],
        "t_slic_fine": T["slic_fine"],
        "t_slic_med": T["slic_med"],
        "t_slic_coarse": T["slic_coarse"],
        "t_membership": T["membership"],
        "t_features": T["features"],
        "t_deep": T["deep"],
        "t_edges": T["edges"],
        "t_edge_feats": T["edge_feats"],
        "t_cross_scale": T["cross_scale"],
        "t_qc": T["qc"],
    }

    if not ok:
        row["out_path"] = ""
        row["gpu_mem_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        return None, row

    x16 = safe_fp16(X_all)
    ea16 = safe_fp16(ea_all)
    ei64 = ei_all.astype(np.int64, copy=False)
    et64 = et_all.astype(np.int64, copy=False)

    np.savez_compressed(
        out_path.replace(".pt", "__labels.npz"),
        lab_fine=lab_fine.astype(np.int32),
        lab_med=lab_med.astype(np.int32),
        lab_coarse=lab_coarse.astype(np.int32),
    )

    t0 = tick()
    data = Data(
        x=torch.from_numpy(x16),
        pos=torch.from_numpy(pm_all.astype(np.float32, copy=False)),
        edge_index=torch.from_numpy(ei64),
        edge_attr=torch.from_numpy(ea16),
        edge_type=torch.from_numpy(et64),
        y=torch.tensor([int(label)], dtype=torch.long),
    )
    data.x_names = list(names_fine)
    data.pos_vox = torch.from_numpy(pv_all.astype(np.float32, copy=False))
    data.slice_ptr_fine = torch.from_numpy(spf.astype(np.int32, copy=False))
    data.slice_idx_fine = torch.from_numpy(sif.astype(np.int16, copy=False) if sif.dtype not in (np.int16, np.int32) else sif)
    data.node_bounds_fine = torch.from_numpy(nbf.astype(np.int16, copy=False) if nbf.dtype != np.int16 else nbf)
    data.slice_ptr_med = torch.from_numpy(spm.astype(np.int32, copy=False))
    data.slice_idx_med = torch.from_numpy(sim.astype(np.int16, copy=False) if sim.dtype not in (np.int16, np.int32) else sim)
    data.node_bounds_med = torch.from_numpy(nbm.astype(np.int16, copy=False) if nbm.dtype != np.int16 else nbm)
    data.slice_ptr_coarse = torch.from_numpy(spc.astype(np.int32, copy=False))
    data.slice_idx_coarse = torch.from_numpy(sic.astype(np.int16, copy=False) if sic.dtype not in (np.int16, np.int32) else sic)
    data.node_bounds_coarse = torch.from_numpy(nbc.astype(np.int16, copy=False) if nbc.dtype != np.int16 else nbc)
    data.node_scale = torch.from_numpy(np.concatenate([np.zeros(K_fine, np.int8), np.ones(K_med, np.int8), 2 * np.ones(K_coarse, np.int8)]))
    data.vol_shape = tuple(int(s) for s in lab_fine.shape)
    data.spacing_mm = tuple(float(s) for s in cfg["spacing_mm"])
    data.K_fine = int(K_fine)
    data.K_med = int(K_med)
    data.K_coarse = int(K_coarse)
    data.num_relations = 3
    data.parent_fine_to_med = torch.from_numpy(p_f2m.astype(np.int32, copy=False))
    data.parent_med_to_coarse = torch.from_numpy(p_m2c.astype(np.int32, copy=False))

    data.patient_id = str(pid)
    data.view = str(view)
    data.label_int = int(label)
    data.aug_kind = str(aug_meta.get("kind", "clean"))
    data.aug_params = dict(aug_meta.get("params", {}))
    data.aug_idx = int(data.aug_params.get("idx", 0))
    data.storage_dtypes = {
    "x": "float16",
    "edge_attr": "float16",
    "edge_index": "int64",
    "edge_type": "int64",
    }
    torch.save(data, out_path)
    t_save = tick() - t0

    row["out_path"] = out_path
    row["t_save"] = t_save
    row["gpu_mem_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    return out_path, row


def append_summary(out_dir: str, row: Dict):
    path = os.path.join(out_dir, "summary.csv")
    lock = path + ".lock"
    os.makedirs(out_dir, exist_ok=True)

    with open(lock, "w") as lk:
        try:
            import fcntl
            fcntl.flock(lk, fcntl.LOCK_EX)

            row = dict(row)

            if os.path.exists(path) and os.path.getsize(path) > 0:
                old_cols = list(pd.read_csv(path, nrows=0).columns)
                new_cols = old_cols + [c for c in row.keys() if c not in old_cols]

                new_df = pd.DataFrame([row])
                for c in new_cols:
                    if c not in new_df.columns:
                        new_df[c] = np.nan
                new_df = new_df[new_cols]

                if new_cols != old_cols:
                    old_df = pd.read_csv(path)
                    for c in new_cols:
                        if c not in old_df.columns:
                            old_df[c] = np.nan
                    old_df = old_df[new_cols]
                    pd.concat([old_df, new_df], ignore_index=True).to_csv(path, index=False)
                else:
                    new_df.to_csv(path, mode="a", header=False, index=False)
            else:
                pd.DataFrame([row]).to_csv(path, index=False)

        except Exception:
            df = pd.DataFrame([row])
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _out_name(base_dir: str, pid: str, view: str, label: int, kind: Optional[str], idx: Optional[int]) -> str:
    if kind is None:
        return os.path.join(base_dir, f"{pid}__{view}__lbl{int(label)}.pt")
    return os.path.join(base_dir, f"{pid}__{view}__lbl{int(label)}__aug_{kind}{int(idx)}.pt")


def _is_nan_like(x):
    try:
        return pd.isna(x)
    except Exception:
        return False


def _row_flag(row: Dict, column: str) -> Optional[int]:
    value = row.get(column)
    if value is None or _is_nan_like(value) or str(value).strip() == "":
        return None
    parsed = int(float(value))
    if parsed not in (0, 1):
        raise ValueError(f"{column} must be binary, found {value!r}")
    return parsed


def _class_label_from_row(row: Dict, cfg: Dict) -> int:
    """Resolve graph targets as normal=0, malignant=1, and benign=2.

    ``GraphLabel`` is authoritative when present. Binary component columns are
    still evaluated when supplied so contradictory metadata fails loudly.
    """
    explicit = row.get("GraphLabel")
    has_explicit = explicit is not None and not _is_nan_like(explicit) and str(explicit).strip() != ""
    explicit_label = None
    if has_explicit:
        explicit_label = int(float(explicit))
        if explicit_label not in (0, 1, 2):
            raise ValueError(
                f"PatientID={row.get('PatientID')} View={row.get('View')} "
                f"GraphLabel must be one of 0, 1, or 2; found {explicit_label}"
            )

    normal = _row_flag(row, "Normal")
    benign = _row_flag(row, "Benign")
    cancer = _row_flag(row, "Cancer")
    has_component_labels = any(value is not None for value in (normal, benign, cancer))
    if explicit_label is not None and not has_component_labels:
        return explicit_label

    mask_path = row.get("Mask_path")
    has_mask = (
        mask_path is not None
        and not _is_nan_like(mask_path)
        and str(mask_path).strip() != ""
    )
    if benign == 1 and cancer == 1:
        raise ValueError(
            f"PatientID={row.get('PatientID')} View={row.get('View')} "
            "cannot be both benign and malignant"
        )
    # For a Normal+Benign overlap, a mask is the tie-breaker: a present mask
    # makes the view benign; without a mask it remains normal.
    if normal == 1 and benign == 1:
        label = 2 if has_mask else 0
    elif normal == 1:
        label = 0
    elif cancer == 1:
        label = 1
    elif benign == 1:
        label = 2
    elif normal is not None:
        # Preserve the previous abnormal fallback for non-benign, non-cancer
        # rows while giving explicit Benign and Cancer fields precedence.
        label = 1 - normal
    else:
        label = int(float(row.get(cfg["label_col"], 0)))

    if explicit_label is not None:
        if explicit_label != label:
            raise ValueError(
                f"PatientID={row.get('PatientID')} View={row.get('View')} "
                f"GraphLabel={explicit_label} conflicts with resolved label={label}"
            )
    return label

def set_output_seed(cfg: Dict, pid: str, view: str, kind: str, idx: int) -> int:
    import hashlib

    base = str(cfg.get("seed", 123))
    key = f"{base}|{pid}|{view}|{kind}|{idx}"
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

def build_one(row: Dict, cfg: Dict, device: torch.device, device_id: int):
    pid = str(row["PatientID"]).strip()
    view = str(row["View"]).strip()
    label = _class_label_from_row(row, cfg)

    out_clean = _out_name(cfg["out_dir"], pid, view, label, "clean", 0)

    t_total0 = tick()
    vol = load_volume(str(row["Raw_path"]).strip())
    t_load = tick() - t_total0
    t0 = tick()
    vol = fast_clahe_3d(vol, cfg["clahe_clip"])
    t_clahe = tick() - t0
    t0 = tick()
    tissue = breast_mask_3d(vol)
    t_mask = tick() - t0
    vol = vol * tissue

    out, row_clean = build_graph_from_volume(vol, tissue, pid, view, label, cfg, {"kind": "clean", "params": {"idx": 0}}, device, out_clean, skip_group=True)
    row_clean["t_load"] = t_load
    row_clean["t_clahe"] = t_clahe
    row_clean["t_mask"] = t_mask
    row_clean["t_total"] = tick() - t_total0
    append_summary(cfg["out_dir"], {**row_clean, "voxels_total": int(vol.size), "out_bytes": os.path.getsize(out_clean) if os.path.exists(out_clean) else 0})

    if out and row_clean.get("status", "").startswith("ok"):
        log.info(
            f"{pid} {view} total={row_clean['t_total']:.2f}s load={t_load:.2f} clahe={t_clahe:.2f} mask={t_mask:.2f} "
            f"slic={row_clean['t_slic']:.2f} fine={row_clean['t_slic_fine']:.2f} med={row_clean['t_slic_med']:.2f} coarse={row_clean['t_slic_coarse']:.2f} "
            f"memb={row_clean['t_membership']:.2f} feat={row_clean['t_features']:.2f} deep={row_clean['t_deep']:.2f} edges={row_clean['t_edges']:.2f} "
            f"edge_feats={row_clean['t_edge_feats']:.2f} cross={row_clean['t_cross_scale']:.2f} qc={row_clean['t_qc']:.2f} save={row_clean['t_save']:.2f} mem={row_clean['gpu_mem_mb']:.1f}MB"
        )
    elif row_clean.get("status") == "skip_exists":
        log.info(f"{pid} {view} SKIP (clean exists)")

    for i in range(1, cfg["augmix"]["n_per_view"] + 1):
        p = _out_name(cfg["out_dir"], pid, view, label, "augmix", i)
        if os.path.exists(p) and os.path.exists(p.replace(".pt", "__labels.npz")):
            continue
        aug_seed = set_output_seed(cfg, pid, view, "augmix", i)
        v_aug = augmix_3d(
            vol,
            severity=cfg["augmix"]["severity"],
            width=cfg["augmix"]["width"],
            depth=cfg["augmix"]["depth"],
            alpha=cfg["augmix"]["alpha"],
            m=cfg["augmix"]["m"],
        )
        v_aug = v_aug * tissue
        _, row_aug = build_graph_from_volume(v_aug, tissue, pid, view, label, cfg, {"kind": "augmix", "params": {"idx": i, "seed": aug_seed}}, device, p, skip_group=False)
        row_aug["t_total"] = 0.0
        append_summary(cfg["out_dir"], {**row_aug, "voxels_total": int(vol.size), "out_bytes": os.path.getsize(p) if os.path.exists(p) else 0})

    for i in range(1, cfg["diffusion"]["n_per_view"] + 1):
        p = _out_name(cfg["out_dir"], pid, view, label, "diffusion", i)
        if os.path.exists(p) and os.path.exists(p.replace(".pt", "__labels.npz")):
            continue
        aug_seed = set_output_seed(cfg, pid, view, "diffusion", i)
        v_aug = diffusion_like(
            vol,
            sigma=cfg["diffusion"]["sigma"],
            denoiser=cfg["diffusion"]["denoiser"],
            nlmeans_h=cfg["diffusion"]["nlmeans_h"],
        )
        v_aug = v_aug * tissue
        _, row_diff = build_graph_from_volume(v_aug, tissue, pid, view, label, cfg, {
            "kind": "diffusion",
            "params": {
                "idx": i,
                "sigma": cfg["diffusion"]["sigma"],
                "seed": aug_seed,
            },
        }, device, p, skip_group=False)
        row_diff["t_total"] = 0.0
        append_summary(cfg["out_dir"], {**row_diff, "voxels_total": int(vol.size), "out_bytes": os.path.getsize(p) if os.path.exists(p) else 0})

    for i in range(1, cfg["adversarial"]["n_per_view"] + 1):
        p = _out_name(cfg["out_dir"], pid, view, label, "adv", i)
        if os.path.exists(p) and os.path.exists(p.replace(".pt", "__labels.npz")):
            continue
        try:
            base = torch.load(out_clean, map_location="cpu", weights_only=False)
            x_adv = craft_adversarial_graph_features_modelfree(
                base,
                eps=cfg["adversarial"]["eps"],
                steps=cfg["adversarial"]["steps"],
                norm=cfg["adversarial"]["norm"],
            )

            adv = base.clone()
            adv.x = torch.from_numpy(safe_fp16(x_adv))
            adv = refresh_edge_feature_distance(adv)

            adv.aug_kind = "adv"
            adv.aug_idx = i
            adv.aug_params = {
                "idx": i,
                "eps": cfg["adversarial"]["eps"],
                "steps": cfg["adversarial"]["steps"],
                "norm": cfg["adversarial"]["norm"],
            }

            torch.save(adv, p)

            src_npz = out_clean.replace(".pt", "__labels.npz")
            dst_npz = p.replace(".pt", "__labels.npz")
            if os.path.exists(src_npz) and not os.path.exists(dst_npz):
                shutil.copy2(src_npz, dst_npz)
            append_summary(
                cfg["out_dir"],
                {
                    "PatientID": pid,
                    "View": view,
                    "K_total": int(base.x.shape[0]) if hasattr(base, "x") else None,
                    "E_edges": int(base.edge_index.shape[1]) if hasattr(base, "edge_index") else None,
                    "features_per_node": int(base.x.shape[1]) if hasattr(base, "x") else None,
                    "used_embed": int(
                        hasattr(base, "x_names")
                        and any(str(n).startswith("embed_") for n in list(base.x_names))
                    ),
                    "aug_kind": "adv",
                    "aug_params": json.dumps({"idx": i, "eps": cfg["adversarial"]["eps"], "steps": cfg["adversarial"]["steps"], "norm": cfg["adversarial"]["norm"]}),
                    "status": "ok",
                    "t_slic": 0.0,
                    "t_slic_fine": 0.0,
                    "t_slic_med": 0.0,
                    "t_slic_coarse": 0.0,
                    "t_membership": 0.0,
                    "t_features": 0.0,
                    "t_deep": 0.0,
                    "t_edges": 0.0,
                    "t_edge_feats": 0.0,
                    "t_cross_scale": 0.0,
                    "t_qc": 0.0,
                    "t_save": 0.0,
                    "out_path": p,
                    "gpu_mem_mb": 0.0,
                },
            )
        except Exception as e:
            log.warning(f"ADV build failed for {pid} {view}: {e}")

def refresh_edge_feature_distance(data: Data) -> Data:
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        return data
    if data.edge_attr.numel() == 0:
        return data

    x = data.x.detach().cpu().numpy().astype(np.float32)
    ei = data.edge_index.detach().cpu().numpy().astype(np.int64)
    ea = data.edge_attr.detach().cpu().numpy().astype(np.float32)

    Xn = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    i = ei[0]
    j = ei[1]
    fdist = np.linalg.norm(Xn[j] - Xn[i], axis=1).astype(np.float32)

    ea[:, -1] = fdist
    data.edge_attr = torch.from_numpy(safe_fp16(ea))

    return data

def worker_main(device_id: int, rows: List[Dict], cfg: Dict):
    configure_logging(cfg["out_dir"], cfg["run_id"], f"gpu{device_id}")

    torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{device_id}")
    banner(cfg, device_id)

    failures = []
    for r in rows:
        try:
            build_one(r, cfg, device, device_id)
        except Exception as e:
            log.error(f"FAILED {str(r.get('PatientID'))} {str(r.get('View'))}: {e}")
            failures.append((str(r.get("PatientID")), str(r.get("View")), str(e)))
    if failures:
        raise RuntimeError(f"{len(failures)} graph view(s) failed on cuda:{device_id}")


def _deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively update a nested configuration dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_run_config(path: Optional[str]) -> Dict:
    """Return an independent default configuration with an optional YAML overlay."""
    import copy

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg

    import yaml

    with open(path, encoding="utf-8") as handle:
        updates = yaml.safe_load(handle)
    if updates is None:
        return cfg
    if not isinstance(updates, dict):
        raise TypeError(f"Graph configuration root must be a mapping: {path}")
    return _deep_update(cfg, updates)


def _validate_manifest(df: pd.DataFrame, path: str) -> None:
    required = {"PatientID", "View", "Raw_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Graph manifest {path} is missing columns: {', '.join(sorted(missing))}")
    target_columns = {"GraphLabel", "Normal", "Cancer", "Benign"}
    if not target_columns.intersection(df.columns):
        raise ValueError(
            f"Graph manifest {path} must include GraphLabel or one of Normal, Cancer, and Benign"
        )


def _build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="longitudinal-dbt build-graphs",
        description="Construct multiscale PyTorch Geometric graphs from a DBT TIFF manifest.",
    )
    parser.add_argument("--manifest", required=True, help="CSV with PatientID, View, Raw_path, and target columns.")
    parser.add_argument("--output-dir", required=True, help="Destination for graph artifacts and QC summaries.")
    parser.add_argument("--config", default=None, help="Optional YAML overlay for graph-construction settings.")
    parser.add_argument(
        "--patients", default=None,
        help="Path to a text file with one PatientID per line, or comma-separated IDs. "
             "Only these patients will be processed. Omit to process all patients."
    )
    parser.add_argument("--embedding-weights", default=None, help="Optional ResNet-50 checkpoint path.")
    parser.add_argument("--no-embeddings", action="store_true", help="Use handcrafted node features only.")
    parser.add_argument("--no-augmentations", action="store_true", help="Construct only the clean graph variant.")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override the configured GPU worker count.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed.")
    return parser


def main(argv=None):
    mp.set_start_method("spawn", force=True)
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    cfg = _load_run_config(args.config)
    cfg["csv_path"] = os.path.abspath(os.path.expanduser(args.manifest))
    cfg["out_dir"] = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.embedding_weights is not None:
        cfg["embed"]["weights_path"] = os.path.abspath(os.path.expanduser(args.embedding_weights))
    if args.no_embeddings:
        cfg["embed"]["enabled"] = False
        cfg["embed"]["weights_path"] = None
    if args.no_augmentations:
        cfg["augmix"]["n_per_view"] = 0
        cfg["diffusion"]["n_per_view"] = 0
        cfg["adversarial"]["n_per_view"] = 0
    if args.num_gpus is not None:
        if args.num_gpus < 1:
            parser.error("--num-gpus must be at least 1")
        cfg["parallel"]["num_gpus"] = args.num_gpus
        cfg["parallel"]["enabled"] = args.num_gpus > 1
    if args.seed is not None:
        cfg["seed"] = args.seed

    if not os.path.isfile(cfg["csv_path"]):
        raise FileNotFoundError(f"Graph manifest does not exist: {cfg['csv_path']}")
    df = pd.read_csv(cfg["csv_path"])
    _validate_manifest(df, cfg["csv_path"])

    if args.patients is not None:
        p = args.patients.strip()

        if os.path.exists(p):
            with open(p) as f:
                patient_list = [line.strip() for line in f if line.strip()]
        else:
            patient_list = [x.strip() for x in p.split(",") if x.strip()]

        patient_set = set(str(x).strip() for x in patient_list)

        before = len(df)
        df["PatientID"] = df["PatientID"].astype(str).str.strip()
        df["View"] = df["View"].astype(str).str.strip()
        df = df[df["PatientID"].isin(patient_set)].reset_index(drop=True)

        print(
            f"Patient filter: {before} rows -> {len(df)} rows "
            f"({len(patient_set)} requested patients, {df['PatientID'].nunique()} matched)"
        )

    rows = df.to_dict("records")
    if not rows:
        log.info("No rows to process after filtering. Exiting.")
        return 0

    if cfg["embed"]["enabled"]:
        if not _TV:
            raise RuntimeError(f"embed.enabled=True but torchvision import failed: {_TV_ERR}")
        wp = cfg["embed"].get("weights_path")
        if wp is not None and str(wp).strip() != "" and not os.path.exists(wp):
            raise RuntimeError(f"embed weights_path does not exist: {wp}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    cfg["run_id"] = run_id
    configure_logging(cfg["out_dir"], run_id, "launcher")

    n_gpu = min(torch.cuda.device_count(), cfg["parallel"]["num_gpus"])
    log.info(f"Queuing {len(rows)} view(s) → {cfg['out_dir']} | parallel={cfg['parallel']['enabled']} workers={n_gpu} devs={[i for i in range(n_gpu)]}")

    if n_gpu <= 1 or not cfg["parallel"]["enabled"]:
        configure_logging(cfg["out_dir"], run_id, "gpu0")

        torch.cuda.set_device(0)

        device = torch.device("cuda:0")

        banner(cfg, 0)

        for i, r in enumerate(rows, 1):
            build_one(r, cfg, device, 0)
            log.info(f"[ {i:6d} / {len(rows):6d} ]")
    else:
        chunks = [rows[i::n_gpu] for i in range(n_gpu)]
        procs = []
        for gid, chunk in enumerate(chunks):
            p = mp.Process(target=worker_main, args=(gid, chunk, cfg), daemon=False)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed_workers = [p.exitcode for p in procs if p.exitcode != 0]
        if failed_workers:
            raise RuntimeError(f"Graph worker processes failed with exit codes: {failed_workers}")
    return 0


DEFAULT_CONFIG: Dict = {
    "seed": 123,
    "csv_path": None,
    "out_dir": None,
    "label_col": "Cancer",
    "spacing_mm": (1.0, 0.10, 0.10),
    "clahe_clip": 0.01,
    "slic": {
        "lin_scales_mm": (2.0, 3.5, 7.0),
        "K_min": (16000, 8000, 2000),
        "K_max": (64000, 32000, 8000),
        "K_cap_total": 100000,
        "compactness": 1.0,
        "sigma": 0.0,
    },
    "knn": {"k_fine": 6, "k_med": 3, "k_coarse": 0, "batch": 4096},
    "embed": {
        "enabled": True,
        "dims": 128,
        "max_side": 768,
        "n_slices": 3,
        "weights_path": os.environ.get("DBT_EMBED_WEIGHTS") or None,
        "amp": True,
    },
    "qc": {"min_tissue_coverage": 0.95, "min_lcc_frac": 0.85, "min_avg_degree": 2.0},
    "parallel": {"enabled": False, "num_gpus": 1},
    "augmix": {"n_per_view": 2, "severity": 3, "width": 3, "depth": -1, "alpha": 1.0, "m": 0.2},
    "diffusion": {"n_per_view": 1, "sigma": 0.05, "denoiser": "nlmeans", "nlmeans_h": 0.06},
    "adversarial": {"n_per_view": 1, "eps": 0.5, "steps": 3, "norm": "linf"},
}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
