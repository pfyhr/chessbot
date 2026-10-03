"""Two policy-head architectures now coexist, so checkpoints must be self-describing.

The fully-connected head holds 87% of the network's parameters in a layer that
does no spatial reasoning. The convolutional one emits 73 planes of 8x8, which
is exactly the `plane * 64 + from_square` order the Rust core encodes moves in.

What these tests protect is the thing that would silently break everything:
every tool that plays a network used to build its architecture from CLI flags.
It now reads the weights instead, and a network that cannot be loaded back is a
training run thrown away.
"""

from __future__ import annotations

import chessbot_core as cc
import pytest
import torch

from chessbot.net import Net, infer_arch, load_checkpoint, save_checkpoint


def build(kind: str, blocks: int = 2, channels: int = 32, **kw) -> Net:
    planes, h, w = cc.OBS_SHAPE
    return Net(planes, (h, w), cc.POLICY_LEN, blocks, channels, policy_head=kind, **kw)


@pytest.mark.parametrize("kind", ["fc", "conv"])
def test_both_heads_emit_the_same_policy_vector(kind: str) -> None:
    planes, h, w = cc.OBS_SHAPE
    policy, wdl = build(kind)(torch.zeros(2, planes, h, w))
    assert policy.shape == (2, cc.POLICY_LEN)
    assert wdl.shape == (2, 3)


def test_conv_head_is_the_cheaper_one() -> None:
    fc = sum(p.numel() for p in build("fc").parameters())
    conv = sum(p.numel() for p in build("conv").parameters())
    assert conv < fc / 5, f"conv {conv} is not decisively smaller than fc {fc}"


@pytest.mark.parametrize("kind", ["fc", "conv"])
def test_architecture_round_trips_through_a_checkpoint(tmp_path, kind: str) -> None:
    net = build(kind, blocks=3, channels=32)
    path = tmp_path / "gen000.pt"
    save_checkpoint(path, net, None, {"gen": 0})

    arch = infer_arch(load_checkpoint(path)[0])
    assert arch == {"blocks": 3, "channels": 32, "policy_head": kind,
                    "trunk": "res", "policy_bottleneck": 32}

    # The inferred architecture must be loadable, not merely descriptive.
    planes, h, w = cc.OBS_SHAPE
    rebuilt = Net(planes, (h, w), cc.POLICY_LEN,
                  arch["blocks"], arch["channels"], policy_head=arch["policy_head"],
                  trunk=arch["trunk"], policy_bottleneck=arch["policy_bottleneck"])
    rebuilt.load_state_dict(load_checkpoint(path)[0])


def test_a_bare_state_dict_is_read_as_the_old_architecture(tmp_path) -> None:
    """Checkpoints written before this change are bare state dicts with an fc head."""
    path = tmp_path / "old.pt"
    torch.save(build("fc").state_dict(), path)
    assert infer_arch(load_checkpoint(path)[0])["policy_head"] == "fc"


def test_conv_head_refuses_a_board_it_cannot_tile() -> None:
    """Connect4's 7 moves are not a whole number of 6x7 planes, so it keeps the fc head."""
    with pytest.raises(ValueError, match="multiple of the board"):
        Net(2, (6, 7), cc.CONNECT4_POLICY_LEN, 2, 16, policy_head="conv")


@pytest.mark.parametrize(
    "kw",
    [
        {"policy_head": "fc", "trunk": "res", "policy_bottleneck": 8},     # C
        {"policy_head": "conv", "trunk": "attn", "policy_bottleneck": 32},  # D
    ],
    ids=["C-narrow-bottleneck", "D-attention-trunk"],
)
def test_candidate_architectures_round_trip(tmp_path, kw) -> None:
    """C and D must survive a save/load cycle or a 12h run cannot be measured."""
    planes, h, w = cc.OBS_SHAPE
    kind = kw.pop("policy_head")
    net = build(kind, blocks=3, channels=32, **kw)
    path = tmp_path / "gen000.pt"
    save_checkpoint(path, net, None, {"gen": 0})

    arch = infer_arch(load_checkpoint(path)[0])
    assert arch["policy_head"] == kind
    assert arch["trunk"] == kw["trunk"]
    assert arch["policy_bottleneck"] == kw["policy_bottleneck"]

    rebuilt = Net(planes, (h, w), cc.POLICY_LEN, arch["blocks"], arch["channels"],
                  policy_head=arch["policy_head"], trunk=arch["trunk"],
                  policy_bottleneck=arch["policy_bottleneck"])
    rebuilt.load_state_dict(load_checkpoint(path)[0])


def test_attention_trunk_actually_mixes_across_squares() -> None:
    """The whole case for D is that one square can reach another in one layer.

    Perturb a single square of the input and require the output at a distant
    square to move. A residual trunk of this depth would also pass eventually,
    but an attention trunk must pass with one block -- and if positional
    embeddings or the token reshape were wrong, it would not.
    """
    planes, h, w = cc.OBS_SHAPE
    net = build("conv", blocks=1, channels=32, trunk="attn").eval()
    a = torch.zeros(1, planes, h, w)
    b = a.clone()
    b[0, :, 0, 0] = 1.0  # perturb square a1 only

    with torch.no_grad():
        pa = net(a)[0].reshape(73, h * w)
        pb = net(b)[0].reshape(73, h * w)

    far = (pa[:, 63] - pb[:, 63]).abs().max().item()  # h8, as far as it gets
    assert far > 1e-6, "a1 did not reach h8 through a single attention block"


def test_narrow_bottleneck_shrinks_the_flat_head() -> None:
    wide = sum(p.numel() for p in build("fc", policy_bottleneck=32).parameters())
    narrow = sum(p.numel() for p in build("fc", policy_bottleneck=8).parameters())
    assert narrow < wide / 2
