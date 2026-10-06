# The 6.3x GPU ceiling was the borrowed machine's CPU, not our code

Date: 2026-10-06. `bench/generation_profile.py --games 256 --steps 250`, M2 Max,
same configuration both times.

The status board claimed that an infinitely fast GPU caps at 6.3× the laptop,
because "30 seconds of every generation is CPU and cannot be bought away". That
was wrong twice.

**First, the 30s was the wrong machine's.** It is what aida's Threadripper core
costs in the driver. Our own M2 did the same work in 14.8s at the same moment.
The 6.3× divided one machine's total by the other machine's floor.

**Second, the floor then moved.** Deriving Gumbel noise on demand instead of
drawing 4,672 per move (`d576652`) cut it again:

| | before | after |
|---|---|---|
| self-play, network forward | 150.4s | 126.5s |
| self-play, Rust driver + bridge | **14.6s** | **9.3s** |
| python/numpy overhead | 0.2s | 0.2s |
| training, 250 steps | 24.9s | 25.6s |
| generation total | 190.1s | 161.5s |
| GPU-bound fraction | 92.2% | 94.1% |
| inference throughput | 10,730 pos/sec | 10,575 pos/sec |

Driver 14.6 → 9.3s is **1.56×**, which is exactly what `drivercost` measured in
isolation. Worth noting against the standing rule that microbenchmarks overstate:
this one did not, because the change removed work rather than speeding work up,
and removed work does not reappear when the loop is reassembled.

The forward is lower too, but positions produced fell with it (49,812 → 41,259)
and throughput is flat, so that is the games playing out differently, not a
speedup.

**Corrected:** a generation is 161.5s with 9.5s fixed, so the ceiling on this
laptop is **17×**, not 6.3×. Nothing was bought to get there.

The general point is the one the old panel obscured: the ceiling is not a
property of our code alone. It is set by the CPU paired with the card *and* by
the driver, and both are ours to choose.
