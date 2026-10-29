# Evidence for the villa ink-inference worker-lifetime fix

Supporting images for the villa pull request that sets `persistent_workers=False` on the one-use
DataLoaders in `vesuvius/src/vesuvius/ink_detection/inference/infer.py`.

- `file-descriptors.png`: open file descriptors after each map, unchanged code (red) vs patched (green).
- `before-errno24.png`: the unchanged code stops after 25 maps with `OSError: [Errno 24] Too many open files`.
- `after-64-maps.png`: the patched code completes all 64 maps.
- `unchanged-output.png`: the same Reader V3 map before and after. All 25 shared TIFFs are SHA-256 identical.

Data: four native-resolution PHerc0139 crops, each run in both directions and repeated eight times
(64 maps), on an RTX 3090 with Torch 2.12.0+cu130, 4 workers, batch 1, FP16, seed 42. The soft
descriptor limit was lowered to 256 so the failure shows up quickly. Derived images are CC BY-NC 4.0.
