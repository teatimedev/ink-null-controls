"""Run a v8-in family checkpoint (organiser v8-in, A-hn fine-tunes) on a stack.

Mirrors the tile recipe published with the PHerc. 0306B claim (github.com/SirProdigle/pherc0306b-first-letters,
`submission/code/ink_v8.py`, MIT): 64 px tiles on a stride-21 grid, 24 layers clipped at 200 and scaled by
1/255, sigmoid outputs stitched with a Gaussian (+-3 sigma) weight and divided by the per-pixel weight sum.
The model class comes from the `ink8um` loader published with v8-in; pass its folder as `v8_dir`.

Tile positions are computed once from the REAL stack and reused for every null, so an emptier null
input cannot hide a response by dropping tiles.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def gaussian_weight(size: int = 64, nsig: float = 3.0) -> np.ndarray:
    x = np.linspace(-nsig, nsig, size + 1)
    cdf = 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))
    k = np.outer(np.diff(cdf), np.diff(cdf))
    return (k / k.max()).astype(np.float32)


def positions(coverage: np.ndarray, min_cov: float = 0.25, stride: int = 21, tile: int = 64) -> np.ndarray:
    h, w = coverage.shape
    integral = np.pad(coverage.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys, xs = np.meshgrid(np.arange(0, h - tile + 1, stride), np.arange(0, w - tile + 1, stride), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    s = integral[ys + tile, xs + tile] - integral[ys, xs + tile] - integral[ys + tile, xs] + integral[ys, xs]
    keep = s >= min_cov * tile * tile
    return np.stack([ys[keep], xs[keep]], 1)


def load_model(v8_dir: str, weights: str, config: str, device: str):
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(Path(v8_dir)))
    from ink8um import InkDetector
    cfg = json.loads(Path(config).read_text())
    model = InkDetector(**cfg)
    model.load_state_dict(load_file(weights), strict=True)
    return model.to(device).eval()


def default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict(model, stack: np.ndarray, tile_positions: np.ndarray, layers: tuple[int, int] = (2, 26),
            device: str = "cpu", batch: int = 8, nsig: float = 3.0) -> np.ndarray:
    import torch
    slab = np.ascontiguousarray(np.minimum(stack[layers[0]:layers[1]], 200).transpose(1, 2, 0))
    weight = gaussian_weight(64, nsig)
    num = np.zeros(stack.shape[1:], np.float32)
    den = np.zeros(stack.shape[1:], np.float32)
    with torch.inference_mode():
        for i in range(0, len(tile_positions), batch):
            pts = tile_positions[i:i + batch]
            x = np.stack([slab[y:y + 64, x0:x0 + 64].transpose(2, 0, 1) for y, x0 in pts])
            x = torch.from_numpy(x).to(device).float().mul_(1 / 255.0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            p = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
            for q, (y, x0) in zip(p, pts):
                num[y:y + 64, x0:x0 + 64] += q * weight
                den[y:y + 64, x0:x0 + 64] += weight
    out = np.zeros_like(num)
    covered = den > 0
    out[covered] = num[covered] / den[covered]
    return out
