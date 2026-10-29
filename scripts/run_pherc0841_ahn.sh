#!/usr/bin/env bash
# Null controls for A-hn (seed 0) on the two labelled PHerc. 0841 crops.
# Needs: scripts/fetch_pherc0841_controls.py run first; the ink8um loader and the A-hn checkpoint:
#   huggingface.co/Prodigle/pherc0306b-ink-ahn  checkpoints/hn4_s0_final/{model.safetensors,config.json}
#   sha256 of model.safetensors: 9a2812843924c50cafaa2b598f62c92ceca9f0d1f5d05b7bbf25764814ec7009
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-.cache/models}
NULLS=${NULLS:-phase-scramble,block-shuffle,constant,depth-shuffle}
CORE=256,768,256,768          # the central 512 px evaluation window of each 1024 px crop
for i in ${CROPS:-0 1}; do
  D=data/pherc0841; R=results/pherc0841/crop$i/ahn-s0
  mkdir -p "$R"
  [ -f $D/crop$i-nulls/nulls.json ] || inknull make $D/crop$i.zarr $D/crop$i-nulls --nulls "$NULLS" --seed 42 --block 64
  inknull run-v8in $D/crop$i.zarr $D/crop$i-nulls "$R/maps" --nulls "$NULLS" --region $CORE --batch 2 \
    --v8-dir $M/v8in_loader --weights $M/hn4_s0_final/model.safetensors --config $M/hn4_s0_final/config.json
  inknull score --real "$R/maps/real.npy" --null-dir "$R/maps" --manifest $D/crop$i-nulls/nulls.json \
    --labels $D/crop$i-labels.npy --region $CORE --title "PHerc0841 crop $i, A-hn seed 0" \
    --out "$R/score.json" --md "$R/score.md" --png "$R/panel.png" || true
  cp $D/crop$i-nulls/nulls.json "$R/nulls.json"
done
