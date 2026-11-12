# On Apple Silicon the network is the cost, not the bridge; native MPSGraph runs it 1.2-1.4x faster than compiled PyTorch

> **Caveat added at merge (2026-11-12):** the packed-bridge rows below were
> measured before `d49208f`, which writes packed planes straight from the
> bitboards. At the time the packed path still encoded every position as
> dense floats and then scanned them into masks -- the CPU cost blamed here
> for packing losing. On aida that commit cut driver time per evaluation
> 3.2x. Re-measure packed on Apple Silicon before treating "packing loses"
> as settled. The pools result is unaffected.

**Date:** 2026-11-12 · mercury (M1 Pro, 16 GB, macOS 27.0), torch 2.15.0, fp32
unless stated, random-init 6x96 (res trunk, fc policy head), `sims 32 /
considered 32`. Every number is a **single run**. `ps` was snapshotted before
each step (`top 4 by CPU`). A browser tab (WebKit content process) swung
between ~15% and ~90% CPU throughout and could not be stopped. A foreign
`cargo test` (another agent) overlapped the start of the first fp32 per-call
run, which was re-run. Not measured on Jupiter (M2 Max).

## Part A: where self-play time goes on MPS

Self-play: `generation_profile.py --games G --concurrency G --max-plies 80
--steps 5`. Eager PyTorch, as the profiler runs it by default. `--max-plies 80`
(not 400) keeps 8 runs inside an hour. All games run at once and stop
together, so batches stay full: this measures steady state, not the tail.

| games | bridge | pools | self-play s | forward s | driver s | evals / s | positions / wall s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | dense | 1 | 112.2 | 101.3 | 10.6 | 10,585 | **320** |
| 512 | dense | 2 | ~136 | (overlap) | 13.1 | ~8,820 | 267 |
| 512 | packed | 1 | 117.1 | 103.7 | 13.0 | 10,035 | 304 |
| 512 | packed | 2 | ~139 | (overlap) | 15.6 | ~8,690 | 263 |
| 1024 | dense | 1 | 202.4 | 179.1 | 22.7 | 11,674 | **354** |
| 1024 | dense | 2 | ~225 | (overlap) | 30.9 | ~10,680 | 324 |
| 1024 | packed | 1 | 221.1 | 190.6 | 29.9 | 10,990 | 333 |
| 1024 | packed | 2 | ~229 | (overlap) | 43.5 | ~10,450 | 317 |

`~`: for pools 2 the total is derived as positions / (positions per wall s)
minus ~0.7 s of training, which the pools-1 rows calibrate.

**Neither CUDA fix pays here.** Both lose at both sizes:

* **Pools: -8% to -17%.** Self-play is ~88% GPU on this machine (the 3090 was
  about half GPU), so there is little CPU work to overlap. Splitting the games
  halves each pool's batch, and MPS throughput drops at smaller batches.
* **Packed: -5% to -6%.** Unified memory means there is no PCIe transfer to
  save. Packing costs CPU (driver +2.4 s at 512 games, +7 s at 1024), and the
  unpack and gather on the device cost about as much as the copy they replace.

`unpack_obs` (int64 shifts) is exact on MPS: 540 real batches matched the dense
planes bit for bit, and random masks with bit 63 set match a numpy reference.
`tests/test_pipeline.py::test_unpack_obs_is_exact_on_every_device` now runs on
every available device. No fix was needed.

### Raw forward and the per-call cost

The `--sweep` loop was unbounded on MPS. MPS has no back-pressure, so 5 s of
enqueueing took several minutes to drain. It now synchronises every 4 calls.

| batch | sweep (pipelined) | per call, net only | per call, packed | per call, dense |
|---:|---:|---:|---:|---:|
| 128 | 9,771 | 8,276 | 7,953 | 8,761 |
| 256 | 11,418 | 10,964 | 10,753 | 10,137 |
| 512 | 12,906 | 13,294 | 12,595 | 12,564 |
| 1024 | 13,975 | 14,266 | 13,607 | 14,345 |
| 2048 | 15,126 | 15,348 | 14,473 | 14,918 |

Per call: `bench/forward_call_bench.py`, synchronous, packed inputs with legal
width 64. A one-element round trip costs **0.25-0.35 ms**. The "~2 ms fixed
cost per call" from the earlier note does not reproduce on torch 2.15.

**The gap.** At 1024 games self-play does 11,674 evals/s against 13,975 raw:
84%. The other 16% of wall time splits as driver 11%, forward time above raw
5% (bridge copies, plus batches below 1024 as games finish), and fixed
per-call cost about 0.4% (~2,300 calls x 0.25 ms). **Per-call and bridge
overhead are a small share on MPS.** The network itself is the cost.

## Part B: native MPSGraph (`crates/az-metal`)

Same network, built as one MPSGraph from an exported `.azmw` file with
BatchNorm folded in. The input is the packed format, unpacked on the GPU. The
legal logits are gathered on the GPU. Inputs and outputs are shared-storage
`MTLBuffer`s. Each call is synchronous and includes the copies in and out,
exactly what self-play would pay.

**Parity** (`cargo test -p az-metal --release`): random init with non-trivial
BatchNorm statistics, 512 real self-play positions, compared against PyTorch
CPU fp32.

| precision | max abs err, logits | WDL | value | bound |
|---|---:|---:|---:|---:|
| fp32 | 7.2e-7 | 7.5e-8 | 3.0e-8 | 1e-5 x max(1, max \|logit\|) |
| fp16 | 4.3e-4 | 6.2e-5 | 2.4e-5 | 1e-2 x same |

The max \|logit\| was 0.80. The fp32 bound was shown to fail on a 1e-3 nudge
to one bias (error 2.6e-5).

**Throughput.** Positions/s, per call, all in one session (17:21-17:25,
machine idle apart from the browser tab):

| batch | PyTorch eager, packed | + fold BN | torch.compile, packed | compile + fold BN | **az-metal fp32** | az-metal fp16 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 7,953 | 9,165 | 7,105 | 10,986 | **16,123** | 14,133 |
| 256 | 10,753 | 11,666 | 14,495 | 15,246 | **19,510** | 22,185 |
| 512 | 12,595 | 13,714 | 17,675 | 18,213 | **23,097** | 25,450 |
| 1024 | 13,607 | 14,935 | 17,541 | 16,725 | **23,961** | 25,432 |
| 2048 | 14,473 | 15,359 | 20,449 | 20,938 | **24,892** | 26,395 |

PyTorch fp16 autocast was no faster than fp32 (12,037 packed at 1024). The
first session's az-metal run read 23,668 at 1024 and the end-of-session check
24,246, so the native numbers are stable within about 2%. The compiled-PyTorch
numbers are noisier: 1024 came out below 512 in both compile runs.

* Against **eager** PyTorch, which is what this profiler runs: **1.7-2.0x** at
  128-2048. Folding BatchNorm is only ~10% of that.
* Against **compiled** PyTorch, which training uses by default (`--compile`,
  bucketed shapes): **1.2-1.4x** at 512-2048.
* fp16 adds 5-10% at batch 512 and above.

## Recommendation

Do not turn on `--pools` or `--bridge packed` on the Mac. Both are measured
losses here.

Native Metal is a real but moderate gain: about 1.3x on the forward pass
against compiled PyTorch. At ~88% GPU share that is roughly 1.2-1.25x on
self-play wall time (Amdahl, not measured end to end). Measure on Jupiter
before integrating, with `az-metal-bench` and `forward_call_bench.py
--compile` in one session. The M2 Max has twice the GPU cores, so the ratio
may move either way. The integration cost is real: export weights each
generation, and give the Rust driver its own forward path, plus its own
bucketing of batch and legal width (graphs are cached per shape).
