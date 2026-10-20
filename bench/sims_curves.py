#!/usr/bin/env python3
"""Build the equal-data and equal-wall-clock match lists for a sims A/B.

The first attempt was analysed by hand and two things went wrong: the
wall-clock pairs were nearly missed entirely (they were assumed destroyed by
the deadlock when only the post-deadlock ones were), and the matched
generations were eyeballed. This does both mechanically from history.json.

    python bench/sims_curves.py runs/ab-sims32 runs/ab-sims64 [--deadline-h H]

--deadline-h marks the point after which an arm ran alone. It constrains the
WALL-CLOCK pairs only: past it contention stops being symmetric and an hour
stops meaning the same thing on both sides. Equal-data pairs are unaffected,
because running alone changes how fast a generation arrives, not what it
contains.
"""
import argparse, json, pathlib, sys


def load(run):
    p = pathlib.Path(run)
    hist = json.loads((p / "history.json").read_text())
    saved = sorted(int(f.stem[3:]) for f in p.glob("gen*.pt") if f.stem[3:].isdigit())
    hours = {r["gen"]: r["elapsed_total"] / 3600 for r in hist}
    return {"path": p, "hist": hist, "saved": saved, "hours": hours}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a"); ap.add_argument("arm_b")
    ap.add_argument("--deadline-h", type=float, default=None)
    a = ap.parse_args()
    A, B = load(a.arm_a), load(a.arm_b)

    print(f"{a.arm_a}: {len(A['hist'])} generations, {len(A['saved'])} checkpoints, "
          f"{A['hist'][-1]['elapsed_total']/3600:.2f}h")
    print(f"{a.arm_b}: {len(B['hist'])} generations, {len(B['saved'])} checkpoints, "
          f"{B['hist'][-1]['elapsed_total']/3600:.2f}h")

    # Equal data: the same generation is the same number of self-play games,
    # PROVIDED both arms used the same --games. Check it rather than assume.
    ga = {r["games"] for r in A["hist"]}; gb = {r["games"] for r in B["hist"]}
    if ga != gb:
        print(f"\nWARNING: games per generation differ ({ga} vs {gb}); matched "
              f"generations are NOT equal data.", file=sys.stderr)

    both = sorted(set(A["saved"]) & set(B["saved"]))
    print(f"\n=== EQUAL DATA: {len(both)} matched generations ===")
    # No deadline flag here, deliberately. Losing contention symmetry changes
    # how FAST a generation arrives, not what it contains: the network at gen N
    # has seen N*games positions however long that took. Equal-data pairs stay
    # valid past the point where one arm ran alone. Only the wall-clock pairs
    # below care, and conflating the two is what nearly threw away the salvage.
    for g in both:
        print(f"  gen{g:<6} s32 {A['hours'].get(g,0):6.2f}h   s64 {B['hours'].get(g,0):6.2f}h")

    # Equal wall clock: every pair of saved checkpoints, ranked by how closely
    # their elapsed times agree.
    pairs = []
    for ga_ in A["saved"]:
        ta = A["hours"].get(ga_)
        if ta is None or (a.deadline_h and ta > a.deadline_h): continue
        for gb_ in B["saved"]:
            tb = B["hours"].get(gb_)
            if tb is None or (a.deadline_h and tb > a.deadline_h): continue
            pairs.append((abs(ta - tb), ga_, gb_, ta, tb))
    pairs.sort()

    # Greedily take well-separated pairs so the curve spans the range instead of
    # clustering at whichever hour happens to match best.
    chosen, used_a, used_b = [], set(), set()
    for d, ga_, gb_, ta, tb in pairs:
        if d * 60 > 20: break
        if ga_ in used_a or gb_ in used_b: continue
        if any(abs(ta - t) < 0.5 for _, _, _, t, _ in chosen): continue
        chosen.append((d, ga_, gb_, ta, tb)); used_a.add(ga_); used_b.add(gb_)
    chosen.sort(key=lambda r: r[3])

    print(f"\n=== EQUAL WALL CLOCK: {len(chosen)} pairs within 20 min ===")
    for d, ga_, gb_, ta, tb in chosen:
        lead = "s32" if ta > tb else "s64"
        print(f"  ~{ta:5.2f}h   s32-gen{ga_:<6} ({ta:5.2f}h)  vs  s64-gen{gb_:<6} ({tb:5.2f}h)"
              f"   gap {d*60:4.1f}min, {lead} +{abs(ta-tb)/min(ta,tb)*100:.0f}%")
    if not chosen:
        print("  none -- the arms' clocks never line up on saved checkpoints.")


if __name__ == "__main__":
    main()
