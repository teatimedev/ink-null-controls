"""Input-side null controls for a rendered surface stack.

A stack is a (Z, H, W) uint8 array of CT layers sampled along a surface normal, with 0 meaning
"no data". Each null keeps part of the real input and removes what a letter needs:

constant        nothing but one grey value. Any response is model bias.
phase-scramble  each layer keeps its exact grey-level histogram and power spectrum (fibre and
                sheet texture), but Fourier phases are randomised, so strokes and edges are gone.
                The same random phase is added to every layer, so depth coherence is kept.
pixel-shuffle   valid pixels are permuted in XY, jointly over depth: every depth profile survives,
                every spatial structure is gone. Out-of-distribution noise; a model should stay quiet.
block-shuffle   k x k blocks are moved around, jointly over depth. Local texture and the depth
                profile stay, letter-scale geometry is destroyed. Predictions can be un-shuffled.
depth-shuffle   the layers are permuted. Single layers keep any visible ink, the depth profile
                of the sheet surface is lost.
depth-reverse   the layers in reverse order (reads the other face).

constant and phase-scramble are "blank": neither can contain a letter. constant measures model bias and
tile seams. phase-scramble is the texture null that sets the null threshold, because it looks like
papyrus to a model without containing any stroke. The others are diagnostics: they can legitimately
keep some real ink signal.
"""
from __future__ import annotations

import numpy as np

BIAS = ("constant",)
TEXTURE = ("phase-scramble",)
BLANK = BIAS + TEXTURE + ("pixel-shuffle",)
DIAGNOSTIC = ("block-shuffle", "depth-shuffle", "depth-reverse")
ALL = BLANK + DIAGNOSTIC


def valid_mask(stack: np.ndarray) -> np.ndarray:
    """2D mask of pixels with data in at least half the layers."""
    return (stack > 0).mean(axis=0) >= 0.5


def constant(stack: np.ndarray, value: int | None = None) -> tuple[np.ndarray, dict]:
    valid = valid_mask(stack)
    if value is None:
        value = int(np.median(stack[:, valid])) if valid.any() else 0
    out = np.zeros_like(stack)
    out[:, valid] = value
    return out, {"value": int(value)}


def phase_scramble(stack: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    valid = valid_mask(stack)
    if not valid.any():
        return stack.copy(), {"seed": seed}
    z, h, w = stack.shape
    rng = np.random.default_rng(seed)
    # The phase of the FFT of real white noise is Hermitian, so the inverse transform stays real.
    phase = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal((h, w)))))
    out = np.zeros_like(stack)
    for k in range(z):
        layer = stack[k].astype(np.float64)
        fill = layer[valid].mean()
        layer[~valid] = fill
        scrambled = np.fft.irfft2(np.fft.rfft2(layer - fill) * phase, s=(h, w))
        out[k][valid] = _match_histogram(scrambled[valid], stack[k][valid])
    return out, {"seed": seed}


def _match_histogram(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Give `values` exactly the multiset of `reference`, keeping the rank order of `values`."""
    order = np.argsort(values, kind="stable")
    result = np.empty_like(reference)
    result[order] = np.sort(reference, kind="stable")
    return result


def pixel_shuffle(stack: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    valid = valid_mask(stack)
    idx = np.flatnonzero(valid)
    perm = np.random.default_rng(seed).permutation(idx.size)
    flat = stack.reshape(stack.shape[0], -1)
    out = flat.copy()
    out[:, idx] = flat[:, idx[perm]]
    return out.reshape(stack.shape), {"seed": seed}


def block_shuffle(stack: np.ndarray, block: int = 64, seed: int = 0,
                  region: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Permute fully valid k x k blocks jointly over depth. Partial blocks and margins stay put.

    With `region` (y0, y1, x0, x1), only blocks lying wholly inside it move, so a window scored on its
    own never borrows blocks (or missing predictions) from outside.
    """
    valid = valid_mask(stack)
    _, h, w = stack.shape
    y0, y1, x0, x1 = region or (0, h, 0, w)
    cells = [(by, bx) for by in range(-(-y0 // block), y1 // block) for bx in range(-(-x0 // block), x1 // block)
             if valid[by * block:(by + 1) * block, bx * block:(bx + 1) * block].all()]
    order = np.random.default_rng(seed).permutation(len(cells))
    out = stack.copy()
    moves = []
    for dst, src in zip(cells, (cells[i] for i in order)):
        (dy, dx), (sy, sx) = dst, src
        out[:, dy * block:(dy + 1) * block, dx * block:(dx + 1) * block] = \
            stack[:, sy * block:(sy + 1) * block, sx * block:(sx + 1) * block]
        moves.append([dy, dx, sy, sx])
    return out, {"seed": seed, "block": block, "region": list(region) if region else None, "moves": moves}


def unshuffle_map(pred: np.ndarray, params: dict) -> np.ndarray:
    """Move a 2D prediction of a block-shuffled stack back to the original block positions."""
    b = params["block"]
    out = pred.copy()
    for dy, dx, sy, sx in params["moves"]:
        out[sy * b:(sy + 1) * b, sx * b:(sx + 1) * b] = pred[dy * b:(dy + 1) * b, dx * b:(dx + 1) * b]
    return out


def depth_shuffle(stack: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    perm = np.random.default_rng(seed).permutation(stack.shape[0])
    return np.ascontiguousarray(stack[perm]), {"seed": seed, "permutation": perm.tolist()}


def depth_reverse(stack: np.ndarray) -> tuple[np.ndarray, dict]:
    return np.ascontiguousarray(stack[::-1]), {}


def make(name: str, stack: np.ndarray, seed: int = 0, block: int = 64,
         region: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    if name == "constant":
        return constant(stack)
    if name == "phase-scramble":
        return phase_scramble(stack, seed)
    if name == "pixel-shuffle":
        return pixel_shuffle(stack, seed)
    if name == "block-shuffle":
        return block_shuffle(stack, block, seed, region)
    if name == "depth-shuffle":
        return depth_shuffle(stack, seed)
    if name == "depth-reverse":
        return depth_reverse(stack)
    raise ValueError(f"unknown null {name!r}; choose from {', '.join(ALL)}")
