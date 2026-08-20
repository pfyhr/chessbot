"""The Rust core, re-exported.

The compiled extension is built as ``chessbot_core._core`` rather than as a
top-level module. That is what maturin's mixed layout requires: the Rust
extension has to live inside the Python source tree so that the pure-Python
``chessbot`` package can be installed alongside it. Without that, ``chessbot``
was never installed at all and every entry point needed ``PYTHONPATH`` set.

The names are listed explicitly rather than star-imported, so this file doubles
as the public surface of the core.
"""

from ._core import (
    # chess
    Position,
    encode_batch,
    legal_mask_batch,
    POLICY_LEN,
    OBS_SHAPE,
    OBS_PLANES,
    HISTORY,
    # connect4
    Connect4,
    Connect4SelfPlay,
    Connect4Search,
    connect4_encode_batch,
    CONNECT4_POLICY_LEN,
    CONNECT4_OBS_SHAPE,
)

__all__ = [
    "Position",
    "encode_batch",
    "legal_mask_batch",
    "POLICY_LEN",
    "OBS_SHAPE",
    "OBS_PLANES",
    "HISTORY",
    "Connect4",
    "Connect4SelfPlay",
    "Connect4Search",
    "connect4_encode_batch",
    "CONNECT4_POLICY_LEN",
    "CONNECT4_OBS_SHAPE",
]
