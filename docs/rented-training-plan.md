# Should we rent a GPU to train faster?

Draft plan, 2026-10-26. Nothing here has been rented, and nothing should be
until the measurement session in §1 has run and its numbers have been read
against the thresholds declared *before* it.

## The short version

The instinct is "rent a faster GPU". The repo's own measurements say the
binding constraint is partly elsewhere, and that picking hardware on GPU class
alone would waste money.

The one real datapoint is in commit `86b661a`, which profiled **the same
workload on both machines on the same day**:

> the single-threaded Rust driver costs 29.1 s of a 92 s generation there
> [an RTX 3090] against 14.6 s of 190 s on an M2 Max

Everything below is arithmetic on those two lines.

| | M2 Max | RTX 3090 |
|---|---:|---:|
| driver (CPU, single-threaded) | 14.6 s | **29.1 s** |
| everything else (GPU forward + training) | 175.4 s | 62.9 s |
| generation | 190 s | 92 s |

Two facts fall out, and they point in opposite directions:

* **The 3090's GPU work was 2.79× the M2 Max's.** Against a vendor peak-fp32
  ratio of 2.62× that is ~107% scaling efficiency, so at this net size the GPU
  side does scale with peak fp32 — which makes extrapolating to other cards
  defensible.
* **The 3090's CPU core was 2.0× slower on the driver.** The driver is
  single-threaded *on purpose* — rayon was tried the same day (`86b661a`,
  reverted in `2131d58`) and measured slower, 4.8 s → 5.4 s at 4 threads,
  because a call carries ~5 ms of work across at most 256 slots. More cores
  buy nothing.

`2131d58` drew the conclusion itself: *"the driver is 32% of wall clock on a
fast GPU and single-threaded, which caps even a 10x card at 2.5x."*

### The corrected table

Jupiter today is 97.0 s/generation with `torch.compile`, of which ~9.3 s is
driver (CPU work, unaffected by compile) and ~87.7 s is GPU. Scaling the
measured 3090 point by vendor peak fp32, and taking the rented driver cost as
29.1 s reduced by the 1.56× the Gumbel-noise change (`d576652`) bought after
that profile was taken:

| card | peak fp32 | GPU share | driver | generation | vs Jupiter |
|---|---:|---:|---:|---:|---:|
| RTX 3090 | 35.6 TF | 31.4 s | 18.7 s | 50.1 s | **1.9×** |
| RTX 4090 | 82.6 TF | 13.6 s | 18.7 s | 32.2 s | **3.0×** |
| L40S | 91.6 TF | 12.2 s | 18.7 s | 30.9 s | **3.1×** |
| H100 SXM | 66.9 TF | 16.7 s | 18.7 s | 35.4 s | **2.7×** |

With a notably fast rented core (driver 14 s) the 4090 row reaches 3.5×; with a
slow one (driver 29 s, i.e. no better than the 3090's) it falls to 2.3×.

**Two things to take from this.** First, the expected 4090 result is ~3.0×,
*not* the 3.6× an earlier draft of this document claimed — that figure rested
on `2026-08-20-driver-ceiling.md`'s "est. 10× M2 Max" for a 4090, which was an
estimate and is contradicted by the measured 3090 at 2.8×. Second, and more
usefully:

> **On fp32, an H100 is slower than a 4090 here** — 66.9 TFLOPS against 82.6 —
> while costing 2–4× as much. Its advantage is tensor cores, which only pay if
> TF32 or bf16 earns something. On MPS reduced precision bought 1.03–1.05×,
> i.e. nothing.

So the H100 is worth renting for one hour to test TF32/bf16, and is otherwise
the worst buy on the list.

### How confident is this

The GPU-side scaling rests on **one** measured card and a peak-fp32
extrapolation. The driver cost on a rented core rests on **one** core, scaled
by a factor measured on a different machine. Call it ±30% on the "vs Jupiter"
column, which is enough to move the 4090 row across the decision threshold in
§1 in either direction. That is precisely why §1 exists and why it is cheap.

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
7. **TF32 and bf16.** On MPS reduced precision bought 1.03–1.05× — nothing,
   because at 8×8 spatial the work is launch- and bandwidth-bound. On CUDA this
   is the single highest-value unknown: it is the *only* thing that would make
   an H100 worth more than a 4090, since the H100's fp32 peak is lower. If
   TF32 gives a large win the hardware ranking changes completely; if it gives
   nothing, never rent an H100 for this workload.

### Declared in advance

The table above puts the expected 4090 result at ~3.0×, which sits exactly on
the "worth it" line below. That is not a hedge — it means the session is
genuinely decisive rather than confirmatory, and that the thresholds must be
read as written rather than argued with afterwards.

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

The 1.2× headroom figure in `2026-08-20-driver-ceiling.md` — driver 292,646
evals/s against an estimated 237,000 a 4090 can consume — carries two
assumptions that do not survive. It assumed an M2-class core doing the driving,
where a rented core measured 2× slower; and its "est. 10× M2 Max" for the card
is contradicted by the measured 3090 at 2.8×, which scales to ~6.5× for a 4090.
Netted out, the driver on a rented box is roughly **half** the generation at
4090 class, not a 1.2× margin. Treat the whole of that document's rented-GPU
column as superseded by the measured point in `86b661a`.

## 3. What it costs

Jupiter's marginal cost is electricity, so this is really buying *time*, not
throughput that was otherwise unavailable.

**Measurement session.** 3–4 h on one box, plus 1 h of build and fetch time.
At $0.75/h, **under $5**. Doing all three classes: **$10–15**. This is cheap
enough that the only real cost is attention.

**A 72 h run**, at the corrected 4090 figure of 3.0× (range 2.3–3.5×):

| | Jupiter | 4090 @ 3.0× | 4090 @ 2.3× (slow core) |
|---|---:|---:|---:|
| generations/hour | 37.1 | ~112 | ~85 |
| generations in 72 h | 2,601 | ~7,800 | ~6,000 |
| cost | ~0 | ~$36 (72 h × $0.50) | ~$36 |
| wall clock for 2,601 gens | 72 h | **24 h** | **31 h** |

**On Elo, carefully.** `long-oct` gained **+144.92 ± 50.73** over 2,601
generations — 0.056 Elo/generation. Multiplying that by 7,800 gives +437, and
that number is almost certainly wrong. Within that same run the first half
measured +159 and the second +95, so the rate roughly halved across one
doubling. On that shape, 3× the generations plausibly buys **+220 to +300**,
not +437. It could be less: nothing here says the curve does not flatten
further, and the run has only ever been observed over one 72 h window.

The defensible claim is the cheap one: **~$36 buys the same result in about a
day instead of three**, and makes a multi-day run a weekend rather than a
fortnight. Any Elo figure beyond that is extrapolation, and this project has a
long record of extrapolations that did not survive the games.

Note how much of the money buys an idle GPU. At the 4090 row, 18.7 s of every
32.2 s generation is one CPU core doing tree descent — **58% of the rental**.
That is the strongest argument for spending a few days on game-granularity
parallelism *before* spending money, and it gets stronger with every faster
card.

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

## A correction worth recording

An earlier draft of this document had the 4090 at 3.6× and a "$36 buys 72 h in
20 h" headline. Both rested on `2026-08-20-driver-ceiling.md`'s **estimate**
that a 4090 is 10× an M2 Max, which was never measured and which the 3090
profile contradicts.

A reviewer caught it, and their own arithmetic contained the mirror-image
mistake worth naming: they compared the 3090's 62.9 s of non-driver time
against *Jupiter's current* ~87.7 s, concluding the 3090 was only 1.4× an M2
Max. But 87.7 s is post-`torch.compile` and post-optimisation; the
contemporaneous M2 figure, from the same commit message as the 3090 number, is
**175.4 s**. Like-for-like the ratio is 2.79×, not 1.4×.

Both errors are the same error: taking one number from one date and one from
another because they describe "the same thing". They do not. Any ratio here has
to come from two measurements of the same workload at the same commit, which is
why `86b661a` is the only pair in this document that is quoted together.

## Honest summary

The measurement session is cheap and worth doing. The 72 h run is a marginal
buy: the expected 4090 result sits on the decision threshold, and a slow rented
core pushes it under.

The most useful finding is not about hardware at all. At the 4090 row, **58% of
the rental is one CPU core doing tree descent while the GPU waits**, and that
fraction rises with every faster card — which is why an H100 is predicted to be
*slower* than a 4090 here on fp32. The highest-value work is therefore making
self-play parallel at game granularity: it costs nothing, it helps Jupiter too,
and it is the prerequisite for any rented card above 4090 class being worth its
price. Renting first would be paying to discover that.

### Sources for the price estimates (2026-10-26)

* [Thunder Compute — Vast.ai GPU pricing](https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute)
* [Vast.ai — GPU cloud product page](https://vast.ai/products/gpu-cloud)
* [Cloud GPU pricing guide 2026](https://dev.to/abdulrahman_maslmany/cloud-gpu-pricing-guide-2026-runpod-vs-lambda-vs-aws-vs-vastai-29e4)
* [computecomparison.com — Vast.ai rates](https://computecomparison.com/provider/vast-ai)
* [gpuperhour.com — RTX 4090](https://gpuperhour.com/rent/rtx-4090)

Sources disagree by several-fold (one lists an H100 at $10.84/h, far outside
every other estimate). They are a bracket for budgeting, not a quote.
