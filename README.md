# ink-null-controls (`inknull`)

**Before you believe a First Letters candidate, feed your ink model versions of the same surface
that cannot contain letters, and measure how much "ink" it still reports.**

The First Letters rules ask for false-positive mitigation. Existing checks compare a prediction with
known text (`inkcheck`, villa PR #2349) or benchmark a model on a separate blank region (Reader V3's
blank-papyrus test). `inknull` works on the region you are about to claim, with no labels:

1. `inknull make` writes null versions of your rendered surface stack.
2. You run your usual inference on them.
3. `inknull score` reports how much of each null is above your threshold, and the threshold that
   structure-free papyrus texture from this same region does not reach.

It works with any model that reads a (Z, H, W) stack and writes a 2D map (villa `ink_9um`, Hecate,
Reader V3, TRACE, v8-in, A-hn, …). `inknull run-v8in` runs v8-in family checkpoints directly.

## The nulls

| null | what survives | what is destroyed | a sound model should |
|---|---|---|---|
| `constant` | nothing (one grey value) | everything | output a flat, low map |
| `phase-scramble` | each layer's exact grey-level histogram and power spectrum: fibre and sheet texture | every stroke and edge (Fourier phases randomised, shared across depth) | stay below the real ink level |
| `pixel-shuffle` | every pixel's depth profile | all spatial structure | stay quiet (out-of-distribution noise) |
| `block-shuffle` | local texture and depth profile in 64 px blocks | letter-scale geometry | keep local stroke fragments, lose letters |
| `depth-shuffle` | each layer, including visible ink | the depth profile of the sheet surface | lose most of its response |
| `depth-reverse` | everything, read from the other face | reading direction | lose letter shapes |

`constant`, `phase-scramble` and `pixel-shuffle` cannot contain a letter. Any area they push above
your threshold is model bias or texture response. The **null threshold** is the 99th percentile of
the `phase-scramble` map plus one grey level: a level that papyrus texture from this region, with no
strokes in it, rarely reaches. The other nulls are diagnostics, because real ink can legitimately
survive them. All nulls keep the no-data mask of the input. Seeds and permutations go into
`nulls.json`, and `score` moves block-shuffled predictions back to their original positions.

## Quick start

```bash
pip install git+https://github.com/teatimedev/ink-null-controls     # core: numpy, zarr, tifffile, imagecodecs, pillow
inknull make render.zarr nulls/                                       # writes nulls/<null>.zarr + nulls.json
# run your model on render.zarr and on every nulls/<null>.zarr; save maps as maps/real.tif, maps/<null>.tif
inknull score --real maps/real.tif --null-dir maps --manifest nulls/nulls.json \
    --out score.json --md score.md --png panel.png
```

Add `--labels labels.npy` (int8: 1 ink, 0 no ink, −1 ignore) on known text to calibrate at a fixed
recall and get AUC/AP. Use `--region y0,y1,x0,x1` to score a window. The exit code is 3 when a blank
null crosses the calibration threshold on more than 1% of pixels, so it can gate a script.

## Results on real data

Two 1,024 px crops of PHerc. 0841 segment `20260220213127-w00` (native 9.366 µm, 28 layers), scored on
their central 512 px with the organisers' human ink labels (v20260918). `scripts/fetch_pherc0841_controls.py`
downloads them from the public bucket and checks every array against its SHA-256. The crops were
picked from the labels alone, before any model ran. They are engineering controls, not an unbiased
benchmark.

### TRACE v3B outputs "ink" on a blank input

Maps from a single RTX 3090 run, re-scored with `inknull score`. TRACE v3B, 17 layers, 128 px tiles,
via villa folder inference.

| crop | 80%-recall threshold | constant input: median output | constant input above threshold | real map above threshold |
|---|---|---|---|---|
| 0 | 0.376 | 0.224 | 0.00% | 34.2% |
| 1 | 0.220 | 0.224 | **71.3%** | 40.3% |

On blank input TRACE v3B outputs about 0.224 everywhere. That is **higher than its median output on
real papyrus in crop 1 (0.208)**. Wherever ink is faint enough that 80% recall needs a threshold
below about 0.22, the threshold flags most of a blank region as ink. Raising it to the constant-input
floor (grey 69) cuts crop 1's false-positive rate from 27.1% to 9.8%, at 65% recall.

The product TRACE v3B × A-hn (the PHerc. 0306B claim's recipe) and the A-hn seed mean pass the same
test: 0.00% of constant input above threshold on both crops.

Full tables: `results/pherc0841/earlier-gpu-runs/`.

How those maps were made:
- **Input:** each crop's `forward` layers 6–23.
- **Command:** villa
  `python -m vesuvius.ink_detection.inference.infer STACK.zarr v3B_best.pth OUT.tif --layer-start 6 --layer-end 23 --direction forward --batch-size 4 --num-workers 0 --overlap 0.5 --blend-mode hann --no-compile --amp-dtype auto`,
  using TRACE v3B release `v1.0`, sha256 `812ffe2b…e5da2`.
- **Constant input:** the crop's median grey value everywhere.
- **Depth shuffle:** a seeded permutation of the 28 layers.

These two nulls predate `inknull make`. Re-running them through `inknull make` changes only how the
no-data margin is filled.

### A-hn finds as much "ink" in stroke-free texture as in faint text

A-hn is the PHerc. 0306B claim's fine-tune of v8-in. It holds PHerc. 0841 out of training. Seed 0
`hn4_s0_final` was run with `inknull run-v8in` on an Apple M5 laptop GPU (MPS, FP32), on every null,
over the tiles touching each 512 px core. On the real crops these maps match an earlier RTX 3090 FP16
run of the same checkpoint to r ≥ 0.999999 (largest difference 0.0017). Reproduce with
`scripts/fetch_pherc0841_controls.py`, `scripts/run_pherc0841_ahn.sh` and
`scripts/run_pherc0841_followup.sh`. Each map takes about 16 minutes on the laptop.

Share of each input above the threshold that keeps 80% of labelled ink on the real map:

| input | crop 0 (clear ink) | crop 1 (faint ink) |
|---|---|---|
| real | 36.6% (AUC 0.899) | 45.7% (AUC 0.815) |
| constant | 0.0% | 0.0% |
| pixel-shuffle | 0.0% | 0.0% |
| **phase-scramble** (papyrus texture, no strokes) | 5.5% | **51.7%** |
| block-shuffle, moved back (r with real) | 34.1% (0.71) | 46.2% (0.64) |
| depth-shuffle | 0.0% | 0.2% |

- **A-hn is not fooled by blank input or pixel noise:** constant and pixel-shuffled input give flat,
  low maps.
- **On crop 1, structure-free papyrus texture gets *more* area above threshold than the real
  text** (1.13×). The median output on the scrambled input, 0.298, is above the real crop's median of
  0.282.
  - Where text is faint, the threshold that finds it also passes ordinary fibre and sheet texture.
  - So area above threshold and mean confidence cannot rank faint regions.
  - This is the regime a First Letters search on a new scroll operates in.
- **The null threshold trades recall for false positives.** Setting the threshold at the scrambled
  input's 99th percentile gives:
  - crop 0: 73.2% of the labelled ink at a 10.9% false-positive rate (80.2% at 18.4% before);
  - crop 1: only 42.6% at 5.1% (80.6% at 34.1% before).

  On faint text, most of what survives a texture-safe threshold is the strongest strokes.
- **block-shuffle keeps local strokes and destroys letters.** Moved back into place, it correlates
  with the real map (0.71 and 0.64) at a lower AUC. The model reads local strokes, as intended, so
  letter-scale structure has to come from the surface, not from the model's context.
- **depth-shuffle removes almost all response.** Ink needs its depth profile; the PHerc. 0306B claim
  reports the same for its own surface.

Panels, JSON and Markdown for each crop: `results/pherc0841/crop{0,1}/ahn-s0/`.

### Hecate on spatially shuffled CT (PHerc. 0343 positive control)

This result comes from the same campaign's earlier RTX 3090 run. The scripts here do not regenerate
it. Hecate 9.6 µm on a 1,080 × 1,440 px PHerc. 0343 crop with known text marked
**19.9%** of a pixel-shuffled input as ink (≥ 128/255, mean 67/255). On the real input it marked
**3.0%** (mean 14/255). On constant input it marked none. Raw confidence or "area of ink" cannot rank
regions for that model.

## Reading the output

- **A blank null above your threshold** means the threshold sits inside the model's bias. Raise it,
  or use a model that passes.
- **A diagnostic null with as much area above threshold as the real input** (`× real ≥ 1`) means area
  or mean confidence cannot rank candidate regions.
- **Recall at the null threshold** (labelled controls) is how much known text survives a threshold
  that structure-free texture from the same region does not reach. Compare models on it.
- A candidate whose letters do not stand out from its own `phase-scramble` and `depth-shuffle`
  maps should not be claimed.

None of this proves a reading. Passing these nulls is necessary, not sufficient.

## Limits

- The PHerc. 0841 results come from two crops of one segment, picked by label. TRACE v3B used this
  segment for checkpoint selection. A-hn holds PHerc. 0841 out (AUC 0.874 in its authors' report), but
  shares teacher models with TRACE.
- Thresholds are per model and per region. Re-run `make`/`score` on each candidate.
- `phase-scramble` keeps second-order texture statistics only. A model could still respond to
  higher-order texture this null does not reproduce, so it is a floor, not a full blank-papyrus model.

## Credits

- Labels and scans: Vesuvius Challenge / EduceLab (CC BY-NC 4.0). No CT data is stored in this
  repository. The result images and JSON in `results/` are model outputs derived from those scans, so
  they are CC BY-NC 4.0 too.
- A-hn and the tile recipe in `inknull/adapters/v8in.py`: Liam Warner and Nicola Bodill's PHerc. 0306B
  release (MIT code, CC BY-NC weights). v8-in: Youssef Nader.
- TRACE v3B: Jashanjot Singh Gill and Nhat Nam Tran. Hecate: Vesuvius Challenge.
- Related checks: `inkcheck` (villa PR #2349); Reader V3's blank-papyrus false-ink benchmark; the
  depth-shuffle and decoy controls in the PHerc. 0306B claim.

Built with AI assistance (Claude, Codex).

## Licence

Code: MIT. Derived result images and data in `results/`: CC BY-NC 4.0.
