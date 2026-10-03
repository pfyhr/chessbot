"""Networks for the AlphaZero loop.

Small SE-ResNet with a policy head and a win/draw/loss head. WDL rather than a
scalar value because chess is draw-heavy and a single tanh output blurs "solidly
drawn" together with "unclear"; Connect4 uses the same head so the two share a
code path and a set of bugs.

The moves-left head from Lc0 is deliberately absent for now -- it earns its keep
in chess endgames and has nothing to do on a 6x7 board.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-excitation: cheap, and reliably worth real Elo in this setting."""

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(4, channels // ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        scale, bias = s[:, :c], s[:, c:]
        return x * torch.sigmoid(scale).view(b, c, 1, 1) + bias.view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.se(self.b2(self.c2(y)))
        return F.relu(x + y)


class AttnBlock(nn.Module):
    """Pre-norm transformer block over the board's 64 squares.

    A convolution relates two squares only by relaying through the squares
    between them, and squeeze-excitation pools the board flat and broadcasts one
    modulation everywhere -- so neither can express that the bishop on a1 bears
    on the king on h8. Attention says it in a single layer.

    At 64 tokens it is also the cheaper layer: 7.9M multiply-accumulates against
    a residual block's 10.6M, with the quadratic term that makes attention
    expensive elsewhere costing 10% of the block. A chess board is short enough
    that the usual trade does not apply.
    """

    def __init__(self, channels: int, heads: int = 4, mult: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(channels)
        # Projections written out rather than nn.MultiheadAttention so the
        # fused scaled_dot_product_attention kernel is reachable: measured 23%
        # faster on MPS, which is small but free.
        self.attn_qkv = nn.Linear(channels, 3 * channels)
        self.attn_proj = nn.Linear(channels, channels)
        self.n2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, mult * channels),
            nn.GELU(),
            nn.Linear(mult * channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = (
            self.attn_qkv(self.n1(x))
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + self.attn_proj(o)
        return x + self.ff(self.n2(x))


class ConvPolicyHead(nn.Module):
    """AlphaZero's policy head: a convolution emitting one plane per move type.

    The flat policy index is `plane * 64 + from_square` (see `chess.rs`), which
    is exactly the order a `(planes, H, W)` tensor flattens to, so this drops in
    with no reindexing anywhere.

    It replaces a `Linear(32*H*W, policy_len)` holding 87% of the network's
    parameters while doing no spatial reasoning. The saving is ~106x, but the
    point is weight sharing: a Linear learns each move's row independently, from
    only those positions where that move was plausible, whereas these filters
    see all 64 squares of every position.

    Only usable when `policy_len` is a whole number of board-sized planes --
    true for chess (4672 = 73*64), false for Connect4 (7 vs 42), which keeps
    the fully-connected path.
    """

    def __init__(self, channels: int, planes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(channels, planes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x)).flatten(1)


class Net(nn.Module):
    """Policy + WDL over an arbitrary (C, H, W) board."""

    def __init__(
        self,
        in_planes: int,
        board: tuple[int, int],
        policy_len: int,
        blocks: int = 4,
        channels: int = 64,
        policy_head: str = "fc",
        trunk: str = "res",
        policy_bottleneck: int = 32,
    ):
        super().__init__()
        h, w = board
        self.head_kind = policy_head
        self.trunk_kind = trunk
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        if trunk == "attn":
            self.tower = nn.Sequential(*[AttnBlock(channels) for _ in range(blocks)])
            # Attention is permutation-invariant: without this the network cannot
            # tell a1 from h8 at all. One learned vector per square.
            self.pos = nn.Parameter(torch.zeros(1, h * w, channels))
            nn.init.normal_(self.pos, std=0.02)
        else:
            self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        if policy_head == "conv":
            if policy_len % (h * w):
                raise ValueError(
                    f"conv policy head needs policy_len ({policy_len}) to be a "
                    f"multiple of the board ({h}x{w}={h * w})"
                )
            self.policy_head = ConvPolicyHead(channels, policy_len // (h * w))
        else:
            b = policy_bottleneck
            self.policy_conv = nn.Sequential(
                nn.Conv2d(channels, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU()
            )
            self.policy_fc = nn.Linear(b * h * w, policy_len)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.value_fc = nn.Sequential(nn.Linear(32 * h * w, 128), nn.ReLU())
        self.wdl = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        if self.trunk_kind == "attn":
            b, c, h, w = x.shape
            # (B,C,H,W) -> (B,HW,C): token i is square i, matching the
            # `plane * 64 + square` order the Rust core encodes everything in.
            t = self.tower(x.flatten(2).transpose(1, 2) + self.pos)
            x = t.transpose(1, 2).reshape(b, c, h, w)
        else:
            x = self.tower(x)
        if self.head_kind == "conv":
            p = self.policy_head(x)
        else:
            p = self.policy_fc(self.policy_conv(x).flatten(1))
        v = self.value_fc(self.value_conv(x).flatten(1))
        return p, self.wdl(v)


def wdl_to_scalar(wdl_logits: torch.Tensor) -> torch.Tensor:
    """P(win) - P(loss), the scalar value MCTS backs up."""
    p = F.softmax(wdl_logits, dim=-1)
    return p[:, 0] - p[:, 2]


def z_to_wdl_target(z: torch.Tensor) -> torch.Tensor:
    """Game result in {-1, 0, 1} to a win/draw/loss class index."""
    return (1 - z).long()  # 1 -> 0 (win), 0 -> 1 (draw), -1 -> 2 (loss)


def save_checkpoint(path, net, optimizer=None, meta: dict | None = None) -> None:
    """Write a checkpoint that can actually be resumed from.

    Earlier checkpoints held only the network. Resuming from one restarts AdamW
    with zeroed moments against an already-converged network, which is a
    perturbation rather than a continuation. Keeping the optimizer state makes
    `--init` mean what it says.
    """
    import torch as _torch

    _torch.save(
        {
            "net": net.state_dict(),
            "opt": optimizer.state_dict() if optimizer is not None else None,
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[dict, dict | None, dict]:
    """Read either checkpoint format.

    Returns `(net_state, optimizer_state, meta)`. Older files are a bare
    `state_dict`, and every tool that plays the network has to keep reading them.
    """
    import torch as _torch

    blob = _torch.load(path, map_location=map_location)
    if isinstance(blob, dict) and "net" in blob:
        return blob["net"], blob.get("opt"), blob.get("meta", {})
    return blob, None, {}


def infer_arch(state: dict) -> dict:
    """Recover blocks, channels and policy-head kind from the weights themselves.

    Every tool that plays a network took --blocks/--channels on the command line
    and built whatever shape those said, so a mismatch with the file on disk
    surfaced as an unreadable load error at best. The checkpoint already knows
    what it is; two architectures now coexist, so it has to be asked.
    """
    channels = state["stem.0.weight"].shape[0]
    tower = [int(k.split(".")[1]) for k in state if k.startswith("tower.")]
    blocks = max(tower) + 1 if tower else 0
    head = "conv" if any(k.startswith("policy_head.") for k in state) else "fc"
    # Keyed on a parameter only AttnBlock has. A looser match (".attn.") broke
    # silently when the projections were renamed, and a checkpoint that loads as
    # the wrong architecture is a run thrown away.
    trunk = "attn" if any("attn_qkv" in k for k in state) else "res"
    # The flat head's bottleneck width is configurable, so read it rather than
    # assume the 32 it happened to be when only one width existed.
    bottleneck = (
        state["policy_conv.0.weight"].shape[0] if "policy_conv.0.weight" in state else 32
    )
    return {
        "blocks": blocks,
        "channels": channels,
        "policy_head": head,
        "trunk": trunk,
        "policy_bottleneck": bottleneck,
    }


def net_from_checkpoint(path, planes, board, policy_len, device="cpu"):
    """Build the network a checkpoint actually contains, and load it."""
    state, _, _ = load_checkpoint(path, device)
    arch = infer_arch(state)
    net = Net(
        planes, board, policy_len,
        arch["blocks"], arch["channels"],
        policy_head=arch["policy_head"],
        trunk=arch["trunk"],
        policy_bottleneck=arch["policy_bottleneck"],
    ).to(device)
    net.load_state_dict(state)
    return net, arch


def losses(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    policy_target: torch.Tensor,
    z: torch.Tensor,
    value_mask: torch.Tensor | None = None,
    policy_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy against the search's improved policy, and against the result.

    The policy target is a full distribution, not a label, so this is a soft
    cross-entropy: -sum(target * log_softmax(logits)). Illegal moves carry zero
    target mass and therefore contribute nothing.

    `value_mask` zeroes the value term for positions whose game was stopped by the
    ply limit. Those games have no result, so `z` there is invented; training on it
    teaches the value head an outcome that never happened.

    The *policy* target survives truncation untouched. The search's improved policy
    is a statement about the position, not about how the game later ended.

    `policy_mask` is the mirror image, and zeroes the *policy* term for moves that
    playout cap randomization played on the cheap budget. A search that shallow is
    not worth imitating: the loop only improves because search beats the raw
    network, and at eight simulations it barely does. Those positions keep their
    value target, because the game result is just as true there -- which is the
    whole reason cheap moves are worth playing.

    The two masks are independent. A position can have a trustworthy policy target
    and an invented result, or an honest result and a policy target too cheap to
    learn from.
    """
    logp = F.log_softmax(policy_logits, dim=-1)
    per_policy = -(policy_target * logp).sum(dim=-1)
    if policy_mask is None:
        policy_loss = per_policy.mean()
    else:
        policy_loss = (per_policy * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)

    per_position = F.cross_entropy(wdl_logits, z_to_wdl_target(z), reduction="none")
    if value_mask is None:
        return policy_loss, per_position.mean()
    return policy_loss, (per_position * value_mask).sum() / value_mask.sum().clamp(min=1.0)
