"""How far is one M2 Max from the hardware AlphaZero actually used?

Measures the real AlphaZero network (19 blocks x 256 filters, ~46M params) on
this machine, rather than extrapolating from FLOP counts.
"""
import time, torch, sys
sys.path.insert(0, "/Users/pontus/repos/chessbot/python")
from chessbot.net import Net

dev = "mps" if torch.backends.mps.is_available() else "cpu"

def bench(blocks, channels, batch, label):
    net = Net(119, (8, 8), 4672, blocks, channels).to(dev).eval()
    params = sum(p.numel() for p in net.parameters())
    x = torch.zeros(batch, 119, 8, 8, device=dev)
    def fwd():
        with torch.no_grad():
            net(x)
        if dev == "mps":
            torch.mps.synchronize()
    fwd()
    best = min(( (lambda t0: (fwd(), time.perf_counter()-t0)[1])(time.perf_counter()) ) for _ in range(5))
    pps = batch / best
    print(f"{label:<28} {params/1e6:>6.1f}M params  batch {batch:>4}  {pps:>9,.0f} pos/sec")
    return pps

print(f"device: {dev}\n")
small = bench(6, 96, 512, "ours (6x96, Gumbel)")
big   = bench(19, 256, 256, "AlphaZero net (19x256)")

print()
PLIES = 135  # typical AlphaZero self-play game length

for label, pps, sims in (("AlphaZero settings", big, 800), ("our settings", small, 32)):
    evals_per_game = sims * PLIES
    gps = pps / evals_per_game
    print(f"{label:<22} {sims:>4} sims  {gps:>10.4f} games/sec  {gps*3600:>10,.0f} games/hour")
    years = 44e6 / gps / 3600 / 24 / 365
    days  = 44e6 / gps / 3600 / 24
    print(f"{'':22} 44M games would take {days:,.0f} days ({years:,.1f} years)\n")
