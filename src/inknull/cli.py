"""inknull: null-input controls for ink detection.

  inknull make STACK OUT_DIR             write null versions of a rendered surface stack
  inknull run-v8in STACK NULL_DIR OUT    predict real + nulls with a v8-in family checkpoint
  inknull score --real MAP --null-dir D  compare a real prediction with the null predictions

Any model can be used: run `make`, predict every null stack with your usual inference command,
save each map as <null-name>.(tif|npy|png) in one folder, then run `score`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import io, metrics, nulls, report


def _region(text: str | None, shape: tuple[int, int]) -> tuple[np.ndarray | None, tuple | None]:
    if not text:
        return None, None
    y0, y1, x0, x1 = (int(v) for v in text.split(","))
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask, (y0, y1, x0, x1)


def cmd_make(a) -> int:
    stack = io.load_stack(a.stack)
    if stack.ndim != 3 or stack.dtype != np.uint8:
        raise SystemExit(f"expected a (Z, H, W) uint8 stack, got {stack.shape} {stack.dtype}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tool": "inknull", "input": str(a.stack), "input_shape": list(stack.shape),
                "input_sha256": io.sha256_array(stack), "seed": a.seed, "nulls": {}}
    existing = out / "nulls.json"
    if existing.exists():  # add to an earlier set made from the same stack
        earlier = json.loads(existing.read_text())
        if earlier["input_sha256"] != manifest["input_sha256"]:
            raise SystemExit(f"{existing} was made from a different stack")
        manifest["nulls"] = earlier["nulls"]
    for name in a.nulls.split(","):
        t = time.monotonic()
        region = tuple(int(v) for v in a.region.split(",")) if a.region else None
        arr, params = nulls.make(name, stack, seed=a.seed, block=a.block, region=region)
        io.save_stack(arr, out / f"{name}.zarr")
        manifest["nulls"][name] = {"kind": "blank" if name in nulls.BLANK else "diagnostic", "params": params,
                                   "sha256": io.sha256_array(arr), "seconds": round(time.monotonic() - t, 3)}
        print(f"{name}: {out / (name + '.zarr')}")
    (out / "nulls.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


def _find_map(folder: Path, name: str) -> Path | None:
    for ext in (".npy", ".tif", ".tiff", ".png"):
        if (folder / f"{name}{ext}").exists():
            return folder / f"{name}{ext}"
    return None


def cmd_score(a) -> int:
    real = io.load_map(a.real)
    maps: dict[str, np.ndarray] = {}
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    if a.null_dir:
        for name in nulls.ALL:
            p = _find_map(Path(a.null_dir), name)
            if p is not None:
                maps[name] = io.load_map(p)
    for item in a.null or []:
        name, path = item.split("=", 1)
        maps[name] = io.load_map(path)
    if not maps:
        raise SystemExit("no null maps found")
    run_log = Path(a.null_dir) / "run.json" if a.null_dir else None
    if manifest and run_log and run_log.exists():  # every prediction must come from the null the manifest describes
        predicted = {r["name"]: r.get("input_sha256") for r in json.loads(run_log.read_text())["runs"]}
        for name in maps:
            want = manifest["nulls"].get(name, {}).get("sha256")
            got = predicted.get(name)
            if want and got and want != got:
                raise SystemExit(f"{name}: prediction was made from a different input ({got[:12]}) than "
                                 f"the manifest's null ({want[:12]}); re-run the prediction")
    if "block-shuffle" in maps:
        if manifest is None:
            raise SystemExit("block-shuffle needs --manifest (nulls.json) to move blocks back")
        maps["block-shuffle"] = nulls.unshuffle_map(maps["block-shuffle"], manifest["nulls"]["block-shuffle"]["params"])
    for name, m in maps.items():
        if m.shape != real.shape:
            raise SystemExit(f"{name}: shape {m.shape} differs from the real map {real.shape}")
    labels = np.load(a.labels) if a.labels else None
    region, box = _region(a.region, real.shape)
    result = metrics.score(real, maps, labels=labels, region=region, recall=a.recall,
                           real_quantile=a.real_quantile, null_quantile=a.null_quantile)
    result["inputs"] = {"real": str(a.real), "nulls": sorted(maps), "labels": a.labels, "region": box,
                        "manifest": a.manifest}
    text = report.markdown(result, a.title or Path(a.real).stem)
    print(text)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    if a.md:
        Path(a.md).write_text(text)
    if a.png:
        t = result.get("null_threshold", {}).get("threshold_gray")
        report.panel({"real": real, **maps}, a.png, threshold_gray=t, region=box)
    blank_bad = any(r["kind"] == "blank" and r["exceed_calibration"] > 0.01 for r in result["maps"])
    return 3 if blank_bad else 0


def cmd_run_v8in(a) -> int:
    from .adapters import v8in
    device = a.device or v8in.default_device()
    model = v8in.load_model(a.v8_dir, a.weights, a.config, device)
    real = io.load_stack(a.stack)
    pts = v8in.positions(real[a.coverage_layer] > 0, min_cov=0.25, stride=a.stride)
    if a.region:  # only tiles touching the window; values inside it are unchanged
        y0, y1, x0, x1 = (int(v) for v in a.region.split(","))
        pts = pts[(pts[:, 0] + 64 > y0) & (pts[:, 0] < y1) & (pts[:, 1] + 64 > x0) & (pts[:, 1] < x1)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [] if a.skip_real else [("real", Path(a.stack))]
    for name in (a.nulls.split(",") if a.nulls else nulls.ALL):
        p = Path(a.null_dir) / f"{name}.zarr"
        if p.exists():
            jobs.append((name, p))
    log = {"device": device, "weights": a.weights, "weights_sha256": io.sha256_file(a.weights),
           "tiles": int(len(pts)), "layers": [a.layer_start, a.layer_end], "runs": []}
    for name, path in jobs:  # load each input only when its turn comes, and record exactly what was read
        stack = real if name == "real" else io.load_stack(path)
        t = time.monotonic()
        prob = v8in.predict(model, stack, pts, layers=(a.layer_start, a.layer_end), device=device, batch=a.batch)
        np.save(out / f"{name}.npy", prob)
        log["runs"].append({"name": name, "seconds": round(time.monotonic() - t, 1),
                            "input_sha256": io.sha256_array(stack), "sha256": io.sha256_file(out / f"{name}.npy")})
        print(f"{name}: {time.monotonic() - t:.1f}s", flush=True)
    run_log = out / "run.json"
    if run_log.exists():  # keep the record of earlier runs into the same folder
        earlier = json.loads(run_log.read_text())
        log["runs"] = earlier.get("runs", []) + log["runs"]
    run_log.write_text(json.dumps(log, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inknull", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="write null versions of a stack")
    m.add_argument("stack")
    m.add_argument("out")
    m.add_argument("--nulls", default=",".join(nulls.ALL))
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--block", type=int, default=64, help="block-shuffle block size in pixels (default 64)")
    m.add_argument("--region", help="y0,y1,x0,x1: block-shuffle only within this evaluation window")
    m.set_defaults(func=cmd_make)

    s = sub.add_parser("score", help="compare a real prediction with null predictions")
    s.add_argument("--real", required=True)
    s.add_argument("--null-dir")
    s.add_argument("--null", action="append", help="NAME=PATH, repeatable")
    s.add_argument("--manifest", help="nulls.json from `make` (needed to un-shuffle block-shuffle)")
    s.add_argument("--labels", help=".npy int8 labels: 1 ink, 0 no ink, -1 ignore")
    s.add_argument("--region", help="y0,y1,x0,x1 evaluation window")
    s.add_argument("--recall", type=float, default=0.8)
    s.add_argument("--real-quantile", type=float, default=0.95)
    s.add_argument("--null-quantile", type=float, default=0.99)
    s.add_argument("--title")
    s.add_argument("--out", help="JSON report")
    s.add_argument("--md", help="Markdown report")
    s.add_argument("--png", help="panel image")
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("run-v8in", help="predict real + nulls with a v8-in family checkpoint (A-hn, v8-in)")
    r.add_argument("stack")
    r.add_argument("null_dir")
    r.add_argument("out")
    r.add_argument("--v8-dir", required=True, help="folder holding the ink8um package")
    r.add_argument("--weights", required=True, help="model.safetensors")
    r.add_argument("--config", required=True, help="config.json with the InkDetector arguments")
    r.add_argument("--nulls")
    r.add_argument("--layer-start", type=int, default=2)
    r.add_argument("--layer-end", type=int, default=26)
    r.add_argument("--coverage-layer", type=int, default=14)
    r.add_argument("--stride", type=int, default=21)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--region", help="y0,y1,x0,x1: run only the tiles that touch this window")
    r.add_argument("--skip-real", action="store_true", help="predict only the nulls (real map already done)")
    r.add_argument("--device")
    r.set_defaults(func=cmd_run_v8in)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
