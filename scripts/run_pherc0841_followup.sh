#!/usr/bin/env bash
# Second pass: region-bounded block-shuffle for crop 0 and pixel-shuffle for both crops, then re-score.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-.cache/models}; CORE=256,768,256,768
run() {  # run CROP NULLS
  inknull run-v8in data/pherc0841/crop$1.zarr data/pherc0841/crop$1-nulls results/pherc0841/crop$1/ahn-s0/maps \
    --nulls "$2" --skip-real --region $CORE --batch 2 \
    --v8-dir $M/v8in_loader --weights $M/hn4_s0_final/model.safetensors --config $M/hn4_s0_final/config.json
}
run 0 block-shuffle,pixel-shuffle
run 1 pixel-shuffle
for i in 0 1; do
  R=results/pherc0841/crop$i/ahn-s0
  cp data/pherc0841/crop$i-nulls/nulls.json "$R/nulls.json"
  inknull score --real "$R/maps/real.npy" --null-dir "$R/maps" --manifest "$R/nulls.json" \
    --labels data/pherc0841/crop$i-labels.npy --region $CORE --title "PHerc0841 crop $i, A-hn seed 0" \
    --out "$R/score.json" --md "$R/score.md" --png "$R/panel.png" || true
done
