# Should we rent a GPU to train faster?

Draft plan, 2026-10-26. Nothing here has been rented, and nothing should be
until the measurement session in §1 has run and its numbers have been read
against the thresholds declared *before* it.

## The short version

The instinct is "rent a faster GPU". The repo's own measurements say the
binding constraint is somewhere else, and that picking hardware on GPU class
would waste most of the money.

Two numbers already on record:

* `crates/az-core/src/selfplay.rs:225` — on an **RTX 3090**, the Rust driver was
  **29 s of a 92 s generation**, 31.5% of the wall clock.
* `bench/results/2026-10-06-the-ceiling-was-the-other-machines-cpu.md` — Jupiter's
  M2 Max did the same driver work in **9.3 s**.

The driver is CPU work, it is **single-threaded on purpose**, and a rented
core is typically 2–3× slower per thread than an M2 Max P-core. So the rented
box starts with a handicap on the one part of the generation that money cannot
buy away.

Amdahl, with Jupiter's current 97 s/generation and a driver cost somewhere
near 9 s:

| box | GPU share of a generation | driver | total | vs Jupiter |
|---|---:|---:|---:|---:|
| Jupiter (M2 Max) | ~88 s | 9 s | 97 s | 1.0× |
| 4090, fast core | ~9 s | ~18 s | ~27 s | **3.6×** |
| 4090, slow core | ~9 s | ~29 s | ~38 s | 2.6× |
| H100, fast core | ~3.5 s | ~18 s | ~22 s | **4.4×** |
| H100, slow core | ~3.5 s | ~29 s | ~33 s | 2.9× |

**An H100 costs 2–4× a 4090 and buys about 20% more throughput**, because at
that point the GPU is idle waiting for one CPU thread. The table is arithmetic
on two measured points, not a measurement — which is exactly what §1 is for.

Caveat on the 29 s: it dates from around the Gumbel-noise change (`d576652`)
that cut the driver 14.6 s → 9.3 s on Jupiter. If that landed after the 3090
reading, the real figure today is nearer 18 s and every row above improves.
We do not know which, and guessing is how the last four ceiling estimates went
wrong.

## 1. The measurement session

One box, a few hours, before any commitment. `bench/vast_deploy.sh` already
does most of this.

```bash
./bench/vast_deploy.sh <host> <port> check    # inspect only, installs nothing
./bench/vast_deploy.sh <host> <port> probe    # build, then measure
```

Measure, in this order, stopping early if a threshold below is missed:

1. **Egress to Jupiter.** Already in the script, and it must stay first. On
   2026-10-21 a host advertising 857 Mbps up delivered **0.2 Mbps** to Sweden.
   Neither the listing, DLPerf, nor a download test caught it.
2. **Real core count** from the cgroup quota, not `nproc`. A box advertising
   12.8 cores reports 64.
3. **`drivercost`, single-threaded**, the same invocation as
   `bench/results/2026-08-20-driver-ceiling.md`:
   `./target/release/drivercost --sims 32 --concurrency 512 --games 4096`.
   Jupiter: 292,646 evals/s. This one number predicts most of the rest.
4. **`drivercost` multi-threaded**, for the record. Expect no gain — the
   driver is sequential by design and rayon measured *slower* — but it costs
   a minute and settles whether a high core count is worth paying for.
5. **Generation profile** at the long run's settings, which is the only number
   that decides anything:
   `bench/generation_profile.py --device cuda --games 256 --steps 250`,
   then `--sweep 512,1024,2048,4096` and `--tf32`.
6. **`torch.compile` on CUDA**, on and off. Worth 1.36× on Jupiter; unknown
   here, and compile times on CUDA are longer.
7. **bf16/fp16.** On MPS this bought 1.03–1.05× — nothing, because at 8×8
   spatial the work is launch- and bandwidth-bound. Tensor cores may change
   that, which is the one genuine reason to retest it on CUDA.

### Declared in advance

**Worth renting if**, at the long run's settings:

* generations/hour ≥ **3.0×** Jupiter's 37.1, i.e. ≥ 111 gens/h, **and**
* GPU-bound fraction ≥ **65%** (the GPU is the thing being paid for), **and**
* sustained egress to Jupiter ≥ **10 Mbps** (52 GB of checkpoints from a
  10,000-generation run at `--keep-every 25` needs ~12 h at that rate).

**Kills the idea if:**

* generations/hour < **2.0×** Jupiter. Below that the money buys a long
  weekend's wait, not a different class of experiment.
* GPU-bound fraction < **50%**. Then we are renting a GPU to watch it idle,
  and the right next move is to make the driver parallel at game granularity —
  free, and it would help Jupiter too.
* egress < 5 Mbps. The results cannot come home.
* `drivercost` single-thread < **120,000 evals/s** (under 0.4× Jupiter). The
  box's core is too slow for the driver no matter what card is bolted to it.

Note the asymmetry: two of the four kill criteria point at *our code* rather
than at the hardware. That is the honest reading of the measurements we have.

## 2. Hardware to try

Prices are **estimates as of 2026-10-26**, from the sources at the end; Vast is
a host-set marketplace where rates move daily and vary several-fold between
hosts, so treat these as a bracket to verify live, not a quote.

| class | est. $/h (Vast, on-demand) | cores to ask for | why |
|---|---|---|---|
| **RTX 4090** | $0.30 – 0.75 | ≥ 8 advertised (≈ 4 real) | The expected sweet spot. The driver is the ceiling well before this card is. |
| **L40S** | $0.40 – 0.90 | ≥ 8 | Datacentre part, often on better-provisioned hosts; worth one data point for CPU quality as much as for the GPU. |
| **H100** | $0.90 – 2.00 | ≥ 16 | Only to **falsify** the table above. If it is not ≥ 1.5× the 4090 on gens/hour, that settles it and we never rent one again. |

**Select on single-thread CPU, not on GPU.** Prefer a listed CPU with a high
base clock over a high core count; the driver uses one core and `rayon` has
already been measured as a loss. The core-count column is only there because
Vast allocates cores in proportion to GPUs, and because fastchess matches (a
separate job) do scale with cores.

The earlier 1.2× headroom estimate for a 4090 came from
`2026-08-20-driver-ceiling.md` — driver 292,646 evals/s against an estimated
237,000 the card can consume. It assumed an M2-class core doing the driving. On
a rented core at a third of that, the headroom is **below 1.0× and the driver is
already the ceiling**, which is what the 3090's 29 s is showing.

## 3. What it costs

Jupiter's marginal cost is electricity, so this is really buying *time*, not
throughput that was otherwise unavailable.

**Measurement session.** 3–4 h on one box, plus 1 h of build and fetch time.
At $0.75/h, **under $5**. Doing all three classes: **$10–15**. This is cheap
enough that the only real cost is attention.

**A 72 h run**, assuming the 4090 row above holds at 3.6×:

| | Jupiter | 4090 @ 3.6× |
|---|---:|---:|
| generations/hour | 37.1 | ~133 |
| generations in 72 h | 2,601 | ~9,600 |
| cost | ~0 | ~$36 (72 h × $0.50) |
| wall clock for 2,601 gens | 72 h | **20 h** |

**On Elo, carefully.** `long-oct` gained **+144.92 ± 50.73** over 2,601
generations — 0.056 Elo/generation. Multiplying that by 9,600 gives +535, and
that number is almost certainly wrong. Within that same run the first half
measured +159 and the second +95, so the rate roughly halved across one
doubling. On that shape, 3.7× the generations plausibly buys **+250 to +350**,
not +535. It could be less: nothing here says the curve does not flatten
further, and the run has only ever been observed over one 72 h window.

The defensible claim is the cheap one: **~$36 buys the same result in 20 hours
instead of 72**, and makes a multi-day run a weekend rather than a fortnight.
Any Elo figure beyond that is extrapolation, and this project has a long record
of extrapolations that did not survive the games.

## 4. What changes in the code

Less than expected — `--device auto` already resolves CUDA
(`train_chess.py:47`), and `--compile` is on by default.

* **Device.** Nothing. `pick_device` returns `"cuda"` when it is available.
* **TF32.** Not currently set anywhere. Worth a flag:
  `torch.backends.cuda.matmul.allow_tf32 = True` plus
  `torch.backends.cudnn.allow_tf32 = True`. `generation_profile.py` already
  takes `--tf32`, so the measurement path exists and the training path does not.
* **bf16.** Only if §1.7 says it earns anything. It bought nothing on MPS.
* **Data loading.** Observations go `numpy → torch.from_numpy(...).to(device)`
  every batch (`train_chess.py:123-129`). On CUDA that is a pageable host→device
  copy per tensor per step. Pinned memory and `non_blocking=True` are the
  obvious change, and they are worth measuring rather than assuming — the
  tensors are small and this may be noise.
* **`--keep-every`.** Default is 1. A long rented run must set it to 25 or more
  or it will write 132 MB per generation and fill the disk. Vast instances
  default to 16 GB; that is 121 checkpoints.

### Risks

* **Core count lies.** `nproc` reports the host's cores, not the slice. The
  deploy script already reads the cgroup quota; anything written by hand must
  do the same.
* **Egress lies.** See §1.1. Test it before committing hours.
* **Disk.** 16 GB default. Set it at rent time; it cannot be grown later.
* **Checkpoint sync.** Pull them as they land, not at the end. A run whose
  results are stranded on an instance that gets reclaimed is a total loss.
  `bench/results` already records this as a standing lesson.
* **The torch in the image.** Vast's PyTorch template keeps torch in `/venv/main`,
  not system python. Build with `--system-site-packages` and reuse it rather
  than downloading a second multi-GB torch.
* **Reproducibility.** A CUDA run is not bit-comparable with an MPS run. Any
  A/B that spans both machines measures the machines as well as the arms —
  the same trap as `ab-arms-must-share-the-machine`.

## 5. What Pontus has to do

I cannot rent anything, pay for anything, or create an account, and I will not
ask another session to do it on my behalf.

1. **Create or fund the Vast.ai account.** Add credit — $20 covers the
   measurement session across all three GPU classes with room to spare.
2. **Pick the offer.** Filter for the GPU class, then sort by CPU single-thread
   quality rather than core count or DLPerf. Set **disk to at least 100 GB at
   rent time** — it cannot be changed afterwards.
3. **Hand over SSH host and port**, which is all `bench/vast_deploy.sh` takes:
   `./bench/vast_deploy.sh <host> <port> check` first, which installs nothing.
4. **Decide after the numbers, not before.** The thresholds in §1 are written
   down so that the decision is made against them rather than against how much
   effort has already gone in.

## Honest summary

The measurement session is cheap and worth doing. The 72 h run is a reasonable
buy *if* the 4090 row survives contact. But the most likely outcome of §1, on
the evidence already in this repo, is that the driver turns out to be the
ceiling on rented hardware — and the right response to that is a few days of
work making self-play parallel at game granularity, which costs nothing, helps
Jupiter too, and would have to happen before an H100 was ever worth its price.

### Sources for the price estimates (2026-10-26)

* [Thunder Compute — Vast.ai GPU pricing](https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute)
* [Vast.ai — GPU cloud product page](https://vast.ai/products/gpu-cloud)
* [Cloud GPU pricing guide 2026](https://dev.to/abdulrahman_maslmany/cloud-gpu-pricing-guide-2026-runpod-vs-lambda-vs-aws-vs-vastai-29e4)
* [computecomparison.com — Vast.ai rates](https://computecomparison.com/provider/vast-ai)
* [gpuperhour.com — RTX 4090](https://gpuperhour.com/rent/rtx-4090)

Sources disagree by several-fold (one lists an H100 at $10.84/h, far outside
every other estimate). They are a bracket for budgeting, not a quote.
