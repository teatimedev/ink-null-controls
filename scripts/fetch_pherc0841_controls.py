"""Fetch two labelled PHerc. 0841 control crops from the public Vesuvius Challenge bucket.

Segment 20260220213127-w00, surface volume at native 9.366 um (28 layers, uncompressed 28x128x128
chunks), human ink labels v20260918 drawn on the 2.403 um scan and resized onto the native canvas.
The two 1024 px crops were picked from the labels alone (most labelled ink in the central 512 px core,
disjoint), before any model ran on them. Output, per crop i:

  data/pherc0841/crop{i}.zarr     (28, 1024, 1024) uint8 stack, zarr group with array '0'
  data/pherc0841/crop{i}-labels.npy  int8 1 ink / 0 no ink / -1 unlabelled

The script checks every array against the SHA-256 recorded when the crops were first made.
Data licence: CC BY-NC 4.0 (Vesuvius Challenge / EduceLab). Needs: numpy, zarr, opencv-python-headless.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
SEG = "PHerc0841/segments/20260220213127-w00/"
CT = SEG + "surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr/0/"
LABELS = SEG + "ink-labels/2.403um-volume-20260319124803/20260918/"
SHAPE = (28, 4220, 4760)
CROPS = [
    {"y": 2560, "x": 2816, "array_sha256": "7d538ee7f8f91c15d330cff5a9a2dd0cef51e1df46fe81c22ff43df6e335fc80",
     "label_sha256": "52dce6023ac638c4ed3eb889a1f43768b28e96636b5ecbfacf1a4920a94bbf72"},
    {"y": 1280, "x": 2816, "array_sha256": "10e3f067ac2a8a667f79e5682b4be65ae188a77fa69a780857a81888e08b9834",
     "label_sha256": "5ff89e81a5539b7257f2f21a3e9c253b78fdc277a6fcb7946a6532ddc94a8378"},
]


def get(key: str, attempts: int = 5) -> bytes | None:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(BUCKET + key, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None  # absent chunk = fill value 0
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def chunk(cy: int, cx: int) -> np.ndarray:
    raw = get(f"{CT}0/{cy}/{cx}")
    if raw is None:
        return np.zeros((28, 128, 128), np.uint8)
    return np.frombuffer(raw, np.uint8).reshape(28, 128, 128)


def label_store(name: str) -> np.ndarray:
    import zarr
    out = Path(".cache") / f"{name}.zarr"
    for rel in ("zarr.json", "c/0/0"):
        dest = out / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(get(f"{LABELS}{name}.zarr/0/{rel}"))
    return np.asarray(zarr.open_array(str(out), mode="r"))


def main(out_dir: str = "data/pherc0841") -> int:
    import cv2
    import zarr
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ink = cv2.resize(label_store("inklabels").astype(np.float32), (SHAPE[2], SHAPE[1]), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(label_store("validation"), (SHAPE[2], SHAPE[1]), interpolation=cv2.INTER_NEAREST) > 0
    ok = True
    for i, c in enumerate(CROPS):
        y, x = c["y"], c["x"]
        stack = np.zeros((28, 1024, 1024), np.uint8)
        keys = [(cy, cx) for cy in range(y // 128, (y + 1023) // 128 + 1) for cx in range(x // 128, (x + 1023) // 128 + 1)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for (cy, cx), block in zip(keys, pool.map(lambda k: chunk(*k), keys)):
                yy, xx = cy * 128 - y, cx * 128 - x
                stack[:, yy:yy + 128, xx:xx + 128] = block
        valid = (stack[14] > 0) & mask[y:y + 1024, x:x + 1024]
        labels = np.full((1024, 1024), -1, np.int8)
        labels[valid] = (ink[y:y + 1024, x:x + 1024][valid] > 127).astype(np.int8)
        for what, arr, want in (("stack", stack, c["array_sha256"]), ("labels", labels, c["label_sha256"])):
            got = hashlib.sha256(arr.tobytes()).hexdigest()
            ok &= got == want
            print(f"crop{i} {what}: {'OK' if got == want else 'MISMATCH ' + got}")
        group = zarr.open_group(str(out / f"crop{i}.zarr"), mode="w")
        (group.create_array if hasattr(group, "create_array") else group.create_dataset)("0", data=stack, chunks=(28, 128, 128))
        np.save(out / f"crop{i}-labels.npy", labels)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
