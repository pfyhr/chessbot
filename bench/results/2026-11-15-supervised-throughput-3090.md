# Supervised training throughput on a 3090

2026-11-15, aida GPU 0 (RTX 3090, shared only with an idle 0.9 GB llama-server),
cores 8-11, torch 2.11.0+cu128. `bench/supervised_throughput.py` on 4 test80
shards. GPU numbers reuse one batch, so they exclude the sampler; the sampler
number excludes the GPU. Full rows: `runs/bench/throughput.jsonl` on aida.

Sampler (one core, batch 1024): **91k positions/s**.

GPU step (unpack + forward + loss + backward + AdamW), positions/s:

| net    | batch | eager  | bf16   | compile+bf16 | peak GiB |
|--------|-------|--------|--------|--------------|----------|
| 6x96   | 1024  | 33.0k  | 47.2k  | 54.9k        | 0.6      |
| 6x96   | 4096  | 35.9k  | 63.7k  | **101.4k**   | 1.9      |
| 10x128 | 1024  | 17.8k  | 30.3k  | 37.4k        | 1.0      |
| 10x128 | 4096  | 17.9k  | 35.2k  | **58.2k**    | 3.4      |
| 15x192 | 1024  |  6.6k  | 12.7k  | 17.8k        | 1.9      |
| 15x192 | 4096  |  6.7k  | 13.8k  | **19.4k**    | 6.8      |

## What it says

- Jupiter (M2 Max, eager, batch 1024) trains 6x96 at 6.2k/s. The 3090 with
  compile+bf16 at batch 4096 is **16x** that.
- compile+bf16 is worth 1.6-3x over eager; it needs the large batch to pay off.
- At 6x96 the GPU (101k/s) outruns one sampler core (91k/s): the trainer needs
  two sampler workers there. At 10x128 and up one core is enough.
- Cost does not scale with tower FLOPs (1 : 3 : 10 predicted, 1 : 1.7 : 5.2
  measured): the fixed-size FC policy head and per-step overhead dominate small nets.
- One pass over 1.6B positions: 6x96 ~4.5 h, 10x128 ~7.6 h, 15x192 ~23 h. The
  whole 5x5 data/model grid fits in roughly two to three GPU-days. No rental needed.

Caveat: batch 4096 is not batch 1024 with a faster clock. The scaling study
must fix one batch/LR policy across sizes, or the comparison mixes in
optimisation effects.
