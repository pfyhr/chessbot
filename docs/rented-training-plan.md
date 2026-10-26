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

* **The 3090's GPU work was 2.79× the M2 Max's.** Close to its peak-fp32 ratio
  of 2.62× — but also close to its memory-bandwidth ratio of 2.34×, and one
  measured point cannot say which of the two it is. That ambiguity is the
  subject of the next section, and it is worth a factor of two on the cards we
  would actually rent.
* **The 3090's CPU core was 2.0× slower on the driver.** The driver was
  single-threaded, and rayon was tried the same day (`86b661a`, reverted in
  `2131d58`) and measured slower — 4.8 s → 5.4 s at 4 threads, because a call
  carries ~5 ms of work across at most 256 slots. That measurement was taken at
  *slot* granularity; a game-granularity driver is now in the tree
  (`bench/driver_threads.sh`) and its result replaces the 18.7 s below.

`2131d58` drew the conclusion itself: *"the driver is 32% of wall clock on a
fast GPU and single-threaded, which caps even a 10x card at 2.5x."*

### Which regime are we in?

Everything downstream turns on one unmeasured question: **is this net limited by
arithmetic or by memory bandwidth?** The 3090 point cannot answer it, because on
that card the two predictions are 2.62× and 2.34× — 12% apart, against a
measurement with no error bars at all.

On the cards worth renting the two answers diverge completely. Vendor figures,
checked 2026-10-26, dense (non-sparsity) throughput:

| | fp32 | TF32 tensor | bf16 tensor (fp32 acc.) | bandwidth | fp32 ridge |
|---|---:|---:|---:|---:|---:|
| M2 Max (Jupiter), 38-core | 13.6 TF | — | — | 400 GB/s | 34 |
| RTX 3090 | 35.6 TF | 35.6 TF | 71.0 TF | 936 GB/s | 38 |
| RTX 4090 | 82.6 TF | 82.6 TF | 165.2 TF | 1008 GB/s | 82 |
| L40S | 91.6 TF | 183 TF | 362 TF | 864 GB/s | 106 |
| H100 SXM | 67.0 TF | 494.5 TF | 989 TF | 3350 GB/s | 20 |

Three things in that table were not obvious and change the plan:

1. **TF32 is nothing on a GeForce card.** On the 3090 and the 4090 the TF32
   tensor peak *equals* the fp32 peak. It is a real speedup only on datacentre
   silicon: 2.0× on an L40S, 7.4× on an H100. So the H100's whole case rests on
   a knob that does not exist on the card we expected to buy.
2. **PyTorch has been using TF32 all along, and nobody wrote that down.**
   `torch.backends.cudnn.allow_tf32` defaults to **True**; only
   `cuda.matmul.allow_tf32` defaults to False (verified on torch 2.13.0). This
   net is almost entirely convolution, so a CUDA run that sets nothing has been
   running TF32 convolutions. On the 3090 that happens to change nothing,
   because its TF32 peak equals its fp32 peak — but "the 3090 ran fp32" was a
   guess that happened to be harmless, not a fact.
3. **bf16 cannot be worth its tensor peak here.** A 3×3 convolution, 96→96
   channels at 8×8, batch 512, is 5.44 GFLOP against 25.5 MB of traffic — 213
   FLOP/byte fused, 72 unfused with BatchNorm and ReLU. Against bf16 ridge
   points of 164 (4090), 295 (H100) and 419 (L40S), the big cards fall *off* the
   compute roof the moment bf16 doubles their arithmetic rate. Their bf16
   advantage is capped by bandwidth, far short of the headline number.

**Which way does the arithmetic lean?** Toward compute, in fp32. The same
roofline that caps bf16 puts this net at 72–213 FLOP/byte against fp32 ridge
points of 34–106, so in fp32 every candidate card — including the 3090 and the
M2 Max — should be compute-bound, and the measured 2.79× lands nearer the
compute prediction (2.62×) than the bandwidth one (2.34×).

That is a reason to expect the fp32 ranking to hold, and it is **not** a reason
to skip §1.8. It is arithmetic on a datasheet: it assumes both machines reach
the same fraction of their roof, it ignores everything in a generation that is
not a convolution, and it is contradicted by the one precision measurement this
project actually has — fp16 and bf16 on MPS bought 1.03–1.05×, which is what a
*launch*-bound workload looks like and neither roof explains. Thirty seconds of
batch sweep settles what an afternoon of this cannot.

### The table, both ways

Roofline — attainable = min(peak, intensity × bandwidth) — at the two ends of
the fusion range, with Jupiter's 87.7 s GPU share and a rented driver at 18.7 s:

| card | fp32 strict | TF32 | bf16 |
|---|---|---|---|
| RTX 3090 | 52.2 s · 1.9× | 52.2 s · 1.9× | 35.5–36.4 s · 2.7× |
| **RTX 4090** | 33.1–35.1 s · **2.8–2.9×** | 33.1–35.1 s · 2.8–2.9× | 25.9–35.1 s · 2.8–3.7× |
| L40S | 31.7–37.9 s · 2.6–3.1× | 25.2–37.9 s · 2.6–3.9× | 25.2–37.9 s · 2.6–3.9× |
| **H100 SXM** | 36.5 s · **2.7×** | 21.1–23.6 s · **4.1–4.6×** | 20.4–23.6 s · **4.1–4.8×** |

**The earlier claim that an H100 is slower than a 4090 holds only in strict
fp32**, where it is 2.7× against 2.8–2.9× — a wash, not a loss. Enable the TF32
that PyTorch enables by itself and the H100 is the fastest card on the list by a
clear margin, because TF32 is worth 7.4× to it and nothing at all to a 4090.
That is a reversal, and it is the reason to rent one of each rather than two of
the cheap one.

What does *not* move is the thing that matters most:

> **The driver is 53–92% of every row in that table.** At the 4090's fp32 row,
> 18.7 s of 33.1 s. At the H100's bf16 row, 18.7 s of 20.4 s — the card finishes
> its share in 1.7 s and waits. Every precision win makes this worse, not
> better.

### How confident is this

Not very. The GPU side rests on **one** measured card, and the table above
spreads from 1.9× to 4.8× depending on a regime question that one card cannot
settle. The driver cost on a rented core rests on **one** core, scaled by a
factor measured on a different machine.

±30% on any single cell is the honest claim, and that is enough to move the
4090 across the decision threshold in §1 in either direction. Note also that
every number here is a *peak-ratio* extrapolation: it assumes both machines
reach the same fraction of their roof. The 3090 reached 107% of its predicted
fp32 ratio, which is reassuring and is also a sample of one.

That is precisely why §1 exists and why it is cheap.

## 1. The measurement session

One box, a few hours, before any commitment. `bench/vast_deploy.sh measure`
runs everything below end to end, so the box is never idle while someone decides
what to run next.

```bash
./bench/vast_deploy.sh <host> <port> check      # inspect + egress, installs nothing
./bench/vast_deploy.sh <host> <port> measure    # build, then everything below
```

`check` is non-destructive and includes the egress test, so a bad box can be
abandoned before it has cost twenty minutes of rent. (`probe` is a different
thing — the two-card sims A/B — and is not this.)

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
   `bench/generation_profile.py --device cuda --games 256 --steps 250`.
6. **`torch.compile` on CUDA**, on and off. Worth 1.36× on Jupiter; unknown
   here, and compile times on CUDA are longer.
7. **All four precision regimes**, because the ranking in §2 depends on which
   one the workload lives in. `generation_profile.py --precision` takes:

   | | what it sets | why |
   |---|---|---|
   | `strict` | TF32 off on matmul **and** cudnn | the only honest fp32, and the only one the old table's ranking applies to |
   | `default` | whatever torch ships | what every past CUDA measurement actually ran: TF32 convs, fp32 matmuls |
   | `tf32` | TF32 on everywhere | a no-op on GeForce; the H100's entire case |
   | `bf16` | TF32 on, plus bf16 autocast | the upper bound, capped by bandwidth |

   The profiler prints the flags **as torch reports them afterwards**, not as
   they were asked for, so a setting that silently did not apply is visible in
   the log rather than folded into the result.
8. **A batch sweep**, `--sweep 256,512,1024,2048,4096`, in `strict` and in
   `bf16`. This is the measurement that separates the two regimes: if
   throughput keeps climbing with batch size the card is launch-bound; if it
   flattens at the same TFLOPS in both precisions it is bandwidth-bound; if
   bf16 lifts the plateau it is compute-bound and the tensor cores are
   reachable. Thirty seconds, and it settles a question the whole hardware
   table rests on.
9. **bf16 as a training change, not just a speed one.** Only if 7 and 8 say it
   is worth something. Speed is not the test: a log-softmax over 4672 logits in
   bf16 has about three significant digits. Run a short fixed-length loss-curve
   comparison against fp32 from the same checkpoint and the same seed, then a
   quick match. `--precision bf16` already keeps both losses in fp32 under the
   autocast for exactly this reason, so what is being tested is the forward and
   backward, not the loss.

### Declared in advance

The table above puts the 4090 at 2.8–3.7× and the H100 at 2.7–4.8×, straddling
the "worth it" line below in both cases. That is not a hedge — it means the
session is genuinely decisive rather than confirmatory, and that the thresholds
must be read as written rather than argued with afterwards.

**Which number is read against them.** The best precision regime that passes the
quality check in §1.9 — not the fastest number on the sheet. A bf16 result that
has not been through a loss curve and a match is a speed measurement of a
different training run, and may not be quoted.

**Worth renting if**, at the long run's settings:

* generations/hour ≥ **3.0×** Jupiter's 37.1, i.e. ≥ 111 gens/h, **and**
* GPU-bound fraction ≥ **65%** (the GPU is the thing being paid for), **and**
* sustained egress to Jupiter ≥ **10 Mbps** (52 GB of checkpoints from a
  10,000-generation run at `--keep-every 25` needs ~12 h at that rate).

**Kills the idea if:**

* generations/hour < **2.0×** Jupiter. Below that the money buys a long
  weekend's wait, not a different class of experiment.
* GPU-bound fraction < **50%**, *measured with driver threading already on*.
  Then we are renting a GPU to watch it idle. This criterion was written when
  the next move would have been to parallelise the driver at game granularity;
  that is now done, so a failure here means the threading did not deliver on a
  rented core and the box is the wrong shape.
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
| **RTX 4090** | $0.30 – 0.75 | ≥ 8 advertised (≈ 4 real) | The expected sweet spot, and the cheap end of the bracket. TF32 buys it nothing, so its fp32 and TF32 rows are the same number — which makes it the clean control. |
| **L40S or H100** | $0.40 – 2.00 | ≥ 8 / ≥ 16 | **Not optional.** GeForce and datacentre silicon disagree by 2–7× on exactly the knob that decides the ranking, and no pair of GeForce cards can tell us which regime we are in. One of these is the whole point of the session. |

Rent the 4090 **and** one datacentre card, in that order. The 4090 establishes
the baseline and the driver cost on a rented core; the datacentre card is the
only way to find out whether tensor cores are reachable at this net size. Two
boxes for an hour each is under $5 and answers a question that a week of
arithmetic cannot.

**Select on single-thread CPU first, core count second.** Prefer a listed CPU
with a high base clock: the driver's per-game work is sequential whatever the
thread count, and a slow core is a slow core on every one of them. Core count
now matters as well — it did not when the driver was single-threaded — but only
up to whatever `bench/driver_threads.sh` measured as the point where the
sharded driver stops scaling. Pay for cores up to that number and not beyond.
Vast allocates cores in proportion to GPUs in any case, and fastchess matches
(a separate job) scale with them too.

The 1.2× headroom figure in `2026-08-20-driver-ceiling.md` — driver 292,646
evals/s against an estimated 237,000 a 4090 can consume — carries two
assumptions that do not survive. It assumed an M2-class core doing the driving,
where a rented core measured 2× slower; and its "est. 10× M2 Max" for the card
is contradicted by the measured 3090 at 2.8×, which scales to ~6.5× for a 4090.
Netted out, the driver on a rented box is roughly **half** the generation at
4090 class, not a 1.2× margin. Treat the whole of that document's rented-GPU
column as superseded by the measured point in `86b661a`.

Its "precision is not a lever here" section is a narrower kind of stale. The
measurement behind it is sound — fp16 and bf16 on **MPS** bought 1.03–1.05× —
but it was generalised into a claim about the workload, and MPS has no tensor
cores to turn on. It says nothing about what TF32 does on an H100, which is the
question in §1.7.

## 3. What it costs

Jupiter's marginal cost is electricity, so this is really buying *time*, not
throughput that was otherwise unavailable.

**Measurement session.** 3–4 h on one box, plus 1 h of build and fetch time.
At $0.75/h, **under $5**. Doing all three classes: **$10–15**. This is cheap
enough that the only real cost is attention.

**A 72 h run**, at the 4090's bracket of 2.8–3.7×, and at the H100's TF32 row
of 4.1–4.6× if the datacentre card turns out to be reachable:

| | Jupiter | 4090 @ 2.8× | 4090 @ 3.7× | H100 @ 4.3× |
|---|---:|---:|---:|---:|
| generations/hour | 37.1 | ~104 | ~137 | ~160 |
| generations in 72 h | 2,601 | ~7,500 | ~9,900 | ~11,500 |
| $/h assumed | — | $0.50 | $0.50 | $1.50 |
| cost of 72 h | ~0 | ~$36 | ~$36 | ~$108 |
| wall clock for 2,601 gens | 72 h | **26 h** | **20 h** | **17 h** |

The H100 column buys 9 hours over the slow 4090 for three times the money, and
most of what it is paid for is a card waiting on one CPU core. It is worth an
hour to measure and is almost certainly not worth 72.

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

Note how much of the money buys an idle GPU. At the 4090's fp32 row, 18.7 s of
every 33.1 s generation is one CPU core doing tree descent — **56% of the
rental**. At the H100's bf16 row it is **92%**: the card does its share in 1.7
seconds and waits eighteen.

This is the one row of the analysis that no precision question touches, because
every precision win makes it *worse*. It is the argument for doing
game-granularity parallelism first — which is now in the tree, measured by
`bench/driver_threads.sh`, and whose result replaces the 18.7 s above.

## 4. What changes in the code

Less than expected — `--device auto` already resolves CUDA
(`train_chess.py:47`), and `--compile` is on by default.

* **Device.** Nothing. `pick_device` returns `"cuda"` when it is available.
* **TF32.** Half of it is already on and was never written down.
  `torch.backends.cudnn.allow_tf32` defaults to **True**, so the convolutions —
  which is nearly the whole net — have been running TF32 on CUDA all along;
  only `cuda.matmul.allow_tf32` defaults to False. Training should set both
  explicitly, in whichever direction §1.7 picks, so that the numerics are a
  recorded decision rather than a library default that can change under us.
  `generation_profile.py --precision` does this already.
* **bf16.** Only if §1.7 says it earns anything *and* §1.9 says it costs
  nothing. It bought 1.03–1.05× on MPS, and the roofline says the big cards
  cannot reach their bf16 peak at this net size anyway.
* **Driver threads.** `--driver-threads K` shards the self-play games across K
  threads. A seed reproduces the same games at any K (asserted in
  `az_core::selfplay`'s tests), so it is a speed knob and not an experimental
  arm. This is the change that moves the rows above, and it is free.
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
   `./bench/vast_deploy.sh <host> <port> check` first, which installs nothing
   and will say within a minute whether the box's egress disqualifies it. Then
   `measure`, which runs §1 end to end unattended.
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

**A second correction, 2026-10-26.** The same reviewer then challenged the
precision assumptions, and was right again. The document had said "this workload
runs fp32" and ranked the cards on fp32 peak. Neither survived checking:
`torch.backends.cudnn.allow_tf32` defaults to **True**, so the convolutions had
been running TF32 all along; and ranking by fp32 peak quietly assumed the
workload is compute-bound, which the one measured card cannot establish because
its compute and bandwidth ratios against an M2 Max are 2.62× and 2.34× — too
close to tell apart.

The conclusion that moved: **"an H100 is slower than a 4090 here" is true only
in strict fp32**, and is reversed by the TF32 that PyTorch enables by itself.
It was stated as a finding; it was a consequence of an unexamined default.

The pattern is the same a third time: a number was taken from a context where it
was true and used in one where it was not. Here the context was not a date but a
*configuration* — which is harder to notice, because a configuration does not
appear in the commit message at all. Hence §1.7, which prints the flags torch
reports back rather than the ones that were asked for.

## Honest summary

The measurement session is cheap and worth doing, and it now has a second
question to answer as well as the first.

**Question one, unchanged: is a rented card worth it?** The 72 h run is a
marginal buy. The 4090 lands at 2.8–3.7×, straddling the decision threshold,
and a slow rented core pushes it under.

**Question two, new: are tensor cores reachable at this net size?** The answer
decides which card, and the two answers are opposite. In strict fp32 an H100 is
no faster than a 4090 and costs three times as much. With TF32 — which PyTorch
turns on for convolutions by default, and which is worth 7.4× on an H100 and
*exactly nothing* on a GeForce card — it is the fastest thing on the list. One
hour on one datacentre card settles it; no amount of further arithmetic will.

The most useful finding is still not about hardware. **The driver is 53–92% of
every row in the table**, and every precision win makes that worse rather than
better: an H100 running bf16 would spend 92% of the rental waiting on one CPU
core doing tree descent. So the order is:

1. **Game-granularity driver threading.** Done and in the tree — shard the games
   across K threads, each owning whole games, its own trees and its own rows of
   the batch. Reproducible at any K by construction, and measured by
   `bench/driver_threads.sh`. Jupiter is ~90% GPU so the gain *here* is small by
   design; the number that transfers is driver seconds per generation.
2. **Then the measurement session**, with the new driver cost substituted into
   the tables above.
3. **Then, and only then, a decision about renting.**

Renting first would have been paying to discover step 1.

### Sources for the hardware figures (2026-10-26)

Dense (non-sparsity) throughput and bandwidth, checked rather than quoted from
memory. Where a vendor table carries an asterisk it is a sparsity figure and is
**half** what is usable here; several third-party sites reprint the starred
number without the note.

* [NVIDIA H100 datasheet (Cisco mirror)](https://bxin.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-h100-80-gpu.pdf) — SXM: fp32 67 TF, TF32 989 TF\* (494.5 dense), bf16 1,979 TF\* (989 dense), 3,350 GB/s
* [NVIDIA Ada GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) — RTX 4090 fp16 tensor with **fp32 accumulate** is 165.2 TF, half the 330.3 TF fp16-accumulate rate. The GeForce fp32-accumulate penalty is real and is what the bf16 column above uses.
* [Lenovo Press, L40S](https://lenovopress.lenovo.com/lp1812.pdf) — TF32 183 TF dense / 366\*, bf16 362 TF dense / 733\*, 864 GB/s
* [NVIDIA Ampere GA102 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) — RTX 3090: TF32 35.6 TF dense, equal to its fp32 peak; fp16/bf16 with fp32 accumulate 71 TF; 936 GB/s
* [NVIDIA developer forum, TF32 on GeForce Ampere](https://forums.developer.nvidia.com/t/tf32-tflops-of-geforce-rtx-3090-vs-a40/265828) — confirms the 3090's TF32 figure and the formula behind it
* M2 Max (38-core): 13.6 TFLOPS fp32, 400 GB/s, Apple's published figures.

### Sources for the price estimates (2026-10-26)

* [Thunder Compute — Vast.ai GPU pricing](https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute)
* [Vast.ai — GPU cloud product page](https://vast.ai/products/gpu-cloud)
* [Cloud GPU pricing guide 2026](https://dev.to/abdulrahman_maslmany/cloud-gpu-pricing-guide-2026-runpod-vs-lambda-vs-aws-vs-vastai-29e4)
* [computecomparison.com — Vast.ai rates](https://computecomparison.com/provider/vast-ai)
* [gpuperhour.com — RTX 4090](https://gpuperhour.com/rent/rtx-4090)

Sources disagree by several-fold (one lists an H100 at $10.84/h, far outside
every other estimate). They are a bracket for budgeting, not a quote.
