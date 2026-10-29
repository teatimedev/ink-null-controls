"""Scores for a real prediction map against prediction maps of null inputs.

All maps are compared as uint8 grey levels (0..255), the form ink maps are usually saved in.
Labels, when given, are int8: 1 ink, 0 papyrus without ink, -1 ignore.
"""
from __future__ import annotations

import numpy as np

from . import nulls as N


def to_gray(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    if p.dtype == np.uint8:
        return p
    p = np.nan_to_num(p.astype(np.float32), nan=0.0)
    return np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8)


def _hist(g: np.ndarray, sel: np.ndarray) -> np.ndarray:
    return np.bincount(g[sel], minlength=256).astype(np.float64)


def auc(g: np.ndarray, ink: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware ROC AUC of grey levels, ink vs non-ink pixels."""
    hp, hn = _hist(g, ink), _hist(g, neg)
    if hp.sum() == 0 or hn.sum() == 0:
        return float("nan")
    return float(np.sum(hp * (np.cumsum(hn) - 0.5 * hn)) / (hp.sum() * hn.sum()))


def average_precision(g: np.ndarray, ink: np.ndarray, neg: np.ndarray) -> float:
    hp, hn = _hist(g, ink)[::-1], _hist(g, neg)[::-1]
    tp, fp = np.cumsum(hp), np.cumsum(hn)
    if tp[-1] == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision * hp) / tp[-1])


def threshold_for_recall(g: np.ndarray, ink: np.ndarray, recall: float) -> int:
    """Highest grey level t with P(g >= t | ink) >= recall."""
    hp = _hist(g, ink)
    above = np.cumsum(hp[::-1])[::-1] / max(hp.sum(), 1)
    return int(np.nonzero(above >= recall)[0].max())


def summary(g: np.ndarray, sel: np.ndarray) -> dict:
    q = np.percentile(g[sel], [1, 5, 50, 95, 99]) / 255.0
    return {"p01": float(q[0]), "p05": float(q[1]), "median": float(q[2]), "p95": float(q[3]),
            "p99": float(q[4]), "robust_range": float(q[3] - q[1]), "mean": float(g[sel].mean() / 255.0)}


def exceed(g: np.ndarray, sel: np.ndarray, t: int) -> float:
    return float(np.mean(g[sel] >= t))


def pearson(a: np.ndarray, b: np.ndarray, sel: np.ndarray) -> float:
    x, y = a[sel].astype(np.float64), b[sel].astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def score(real: np.ndarray, null_maps: dict[str, np.ndarray], labels: np.ndarray | None = None,
          region: np.ndarray | None = None, recall: float = 0.8, real_quantile: float = 0.95,
          null_quantile: float = 0.99) -> dict:
    """Compare a real map with null maps over `region` (a 2D bool mask; default everything).

    Calibration threshold: the grey level keeping `recall` of labelled ink on the real map, or, without
    labels, the real map's `real_quantile`. Null threshold: the `null_quantile` grey level of the
    phase-scrambled input (falling back to constant input), i.e. a level that structure-free papyrus
    texture from this same region rarely reaches.
    """
    g_real = to_gray(real)
    sel = np.ones(g_real.shape, bool) if region is None else region.astype(bool)
    ink = neg = None
    if labels is not None:
        ink, neg = sel & (labels == 1), sel & (labels == 0)
        t_cal = threshold_for_recall(g_real, ink, recall)
        calibration = {"mode": "labels", "recall": recall, "threshold_gray": t_cal}
    else:
        t_cal = int(np.ceil(np.percentile(g_real[sel], real_quantile * 100)))
        calibration = {"mode": "real-quantile", "quantile": real_quantile, "threshold_gray": t_cal}

    gray = {"real": g_real} | {k: to_gray(v) for k, v in null_maps.items()}
    real_exceed = exceed(g_real, sel, t_cal)
    rows = []
    for name, g in gray.items():
        kind = "real" if name == "real" else ("blank" if name in N.BLANK else "diagnostic")
        row = {"name": name, "kind": kind, **summary(g, sel), "exceed_calibration": exceed(g, sel, t_cal)}
        row["exceed_ratio_vs_real"] = row["exceed_calibration"] / real_exceed if real_exceed else float("nan")
        if name != "real":
            row["r_vs_real"] = pearson(g, g_real, sel)
        if labels is not None:
            row["auc"] = auc(g, ink, neg)
            row["average_precision"] = average_precision(g, ink, neg)
        rows.append(row)

    blank = {k: g for k, g in gray.items() if k in N.TEXTURE} or {k: g for k, g in gray.items() if k in N.BIAS}
    out = {"calibration": calibration, "pixels": int(sel.sum()), "maps": rows}
    if blank:
        levels = {k: int(np.ceil(np.percentile(g[sel], null_quantile * 100))) for k, g in blank.items()}
        source = max(levels, key=levels.get)
        t_null = min(levels[source] + 1, 255)
        at_null = {"threshold_gray": t_null, "from": source, "quantile": null_quantile,
                   "real_fraction_above": exceed(g_real, sel, t_null)}
        if labels is not None:
            at_null["recall"] = exceed(g_real, ink, t_null)
            at_null["false_positive_rate"] = exceed(g_real, neg, t_null)
            at_null["recall_at_calibration"] = exceed(g_real, ink, t_cal)
            at_null["false_positive_rate_at_calibration"] = exceed(g_real, neg, t_cal)
        out["null_threshold"] = at_null
    out["flags"] = flags(out)
    return out


def flags(result: dict) -> list[str]:
    """Plain-language warnings. They are prompts to look closer, not verdicts on a scroll."""
    msgs = []
    t = result["calibration"]["threshold_gray"]
    for row in result["maps"]:
        if row["kind"] == "blank" and row["exceed_calibration"] > 0.01:
            msgs.append(f"{row['name']}: {row['exceed_calibration']:.1%} of structure-free input is above the "
                        f"calibration threshold (grey {t}); that threshold cannot separate ink from model bias here.")
        if row["kind"] == "blank" and row["robust_range"] > 0.05:
            msgs.append(f"{row['name']}: output varies by {row['robust_range']:.3f} (p95-p05) on structure-free input; "
                        "look for tile seams or invented structure.")
        if row["kind"] == "diagnostic" and row["exceed_ratio_vs_real"] >= 1:
            msgs.append(f"{row['name']}: as much or more area above threshold than the real input "
                        f"({row['exceed_ratio_vs_real']:.2f}x). Area or mean confidence cannot rank regions here.")
    at = result.get("null_threshold")
    if at and "recall" in at and at["recall"] < 0.5:
        msgs.append(f"At the null threshold (grey {at['threshold_gray']}) only {at['recall']:.0%} of labelled ink survives.")
    return msgs
