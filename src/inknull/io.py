"""Reading and writing stacks and maps."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def load_stack(path: str | Path) -> np.ndarray:
    """A (Z, H, W) stack from a zarr array or group (array '0'), a multi-page TIFF, a folder of
    per-layer TIFFs (sorted by name), or a .npy file."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".zarr" or (p / ".zarray").exists() or (p / "zarr.json").exists() or (p / ".zgroup").exists():
        import zarr
        node = zarr.open(str(p), mode="r")
        arr = node["0"] if hasattr(node, "keys") and "0" in node else node
        return np.asarray(arr)
    import tifffile
    if p.is_dir():
        layers = sorted(q for q in p.iterdir() if q.suffix.lower() in (".tif", ".tiff"))
        return np.stack([tifffile.imread(q) for q in layers])
    return tifffile.imread(p)


def save_stack(arr: np.ndarray, path: str | Path) -> None:
    """Write a stack as a zarr group with array '0', the layout villa's folder inference reads."""
    import zarr
    group = zarr.open_group(str(path), mode="w")
    chunks = (arr.shape[0], min(128, arr.shape[1]), min(128, arr.shape[2]))
    if hasattr(group, "create_array"):
        group.create_array("0", data=arr, chunks=chunks)
    else:
        group.create_dataset("0", data=arr, chunks=chunks)


def load_map(path: str | Path) -> np.ndarray:
    """A 2D prediction map from .npy, .tif/.tiff or .png. Integer maps are read as grey levels."""
    p = Path(path)
    if p.suffix == ".npy":
        m = np.load(p)
    elif p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        m = tifffile.imread(p)
    else:
        from PIL import Image
        m = np.asarray(Image.open(p).convert("L"))
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"{p}: expected a 2D map, got shape {m.shape}")
    if m.dtype == np.uint16:
        m = (m.astype(np.float32) / 65535.0)
    return m


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()
