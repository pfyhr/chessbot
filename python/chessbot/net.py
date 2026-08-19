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


class Net(nn.Module):
    """Policy + WDL over an arbitrary (C, H, W) board."""

    def __init__(
        self,
        in_planes: int,
        board: tuple[int, int],
        policy_len: int,
        blocks: int = 4,
        channels: int = 64,
    ):
        super().__init__()
        h, w = board
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.policy_fc = nn.Linear(32 * h * w, policy_len)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.value_fc = nn.Sequential(nn.Linear(32 * h * w, 128), nn.ReLU())
        self.wdl = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))
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


def losses(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    policy_target: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy against the search's improved policy, and against the result.

    The policy target is a full distribution, not a label, so this is a soft
    cross-entropy: -sum(target * log_softmax(logits)). Illegal moves carry zero
    target mass and therefore contribute nothing.
    """
    logp = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(policy_target * logp).sum(dim=-1).mean()
    value_loss = F.cross_entropy(wdl_logits, z_to_wdl_target(z))
    return policy_loss, value_loss
