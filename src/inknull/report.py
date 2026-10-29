"""Markdown table and a side-by-side panel of the real and null maps."""
from __future__ import annotations

import numpy as np

from .metrics import to_gray


def markdown(result: dict, title: str = "") -> str:
    cal = result["calibration"]
    labelled = cal["mode"] == "labels"
    head = f"### {title}\n\n" if title else ""
    if labelled:
        head += f"Calibration: grey {cal['threshold_gray']} keeps {cal['recall']:.0%} of labelled ink on the real map.\n\n"
    else:
        head += f"Calibration: grey {cal['threshold_gray']} is the real map's {cal['quantile']:.0%} quantile.\n\n"
    cols = ["map", "kind", "median", "p95−p05", "above calibration", "× real", "r vs real"]
    if labelled:
        cols += ["AUC", "AP"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in result["maps"]:
        cells = [r["name"], r["kind"], f"{r['median']:.3f}", f"{r['robust_range']:.3f}",
                 f"{r['exceed_calibration']:.2%}", f"{r['exceed_ratio_vs_real']:.2f}",
                 "" if "r_vs_real" not in r else f"{r['r_vs_real']:.3f}"]
        if labelled:
            cells += [f"{r['auc']:.3f}", f"{r['average_precision']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    body = head + "\n".join(lines) + "\n"
    at = result.get("null_threshold")
    if at:
        body += (f"\nNull threshold: grey {at['threshold_gray']} ({at['quantile']:.0%} quantile of the `{at['from']}` map, +1). "
                 f"{at['real_fraction_above']:.2%} of the real map is above it")
        if "recall" in at:
            body += (f"; it keeps {at['recall']:.1%} of labelled ink at a {at['false_positive_rate']:.1%} false-positive rate "
                     f"(calibration threshold: {at['recall_at_calibration']:.1%} at {at['false_positive_rate_at_calibration']:.1%})")
        body += ".\n"
    if result["flags"]:
        body += "\n" + "\n".join(f"- {m}" for m in result["flags"]) + "\n"
    return body


def panel(maps: dict[str, np.ndarray], path: str, threshold_gray: int | None = None,
          region: tuple[int, int, int, int] | None = None, width: int = 360) -> None:
    """One column per map on a fixed 0..1 grey scale. Pixels at or above `threshold_gray` are tinted red."""
    from PIL import Image, ImageDraw
    tiles = []
    for name, m in maps.items():
        g = to_gray(m)
        if region is not None:
            y0, y1, x0, x1 = region
            g = g[y0:y1, x0:x1]
        rgb = np.repeat(g[..., None], 3, axis=2)
        if threshold_gray is not None:
            hot = g >= threshold_gray
            rgb[hot] = (0.5 * rgb[hot] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
        im = Image.fromarray(rgb)
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, im.height + 18), "white")
        canvas.paste(im, (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), name, fill="black")
        tiles.append(canvas)
    out = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 6
    out.save(path)
