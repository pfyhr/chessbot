"""How a checkpoint is named in files the build publishes.

`manifest.json` ships as a release asset and `reference.json`'s checkpoint
field is echoed into a *public* CI log by `parity_check.mjs`. An absolute path
in either one publishes the directory layout -- and the home directory -- of
whoever ran the build, for no benefit: nobody outside that machine can use it.
"""

from __future__ import annotations

from pathlib import Path


def provenance(checkpoint: Path) -> str:
    """`run/file.pt`, e.g. `long-oct/gen2600.pt`.

    Identifies the checkpoint exactly as well as an absolute path for anyone
    who has the repo, and `path.parse(...).name` in parity_check.mjs still
    resolves it to the ONNX basename.
    """
    return f"{Path(checkpoint).parent.name}/{Path(checkpoint).name}"
