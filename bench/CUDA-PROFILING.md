# Running the profiler on a borrowed CUDA box

    git clone https://github.com/pfyhr/chessbot && cd chessbot
    python3 -m venv .venv && .venv/bin/pip install -U pip maturin torch numpy
    .venv/bin/maturin develop --release      # builds the Rust core, needs a rust toolchain

Then, in order:

    # 1. the comparable number -- same settings as the laptop
    .venv/bin/python bench/generation_profile.py --device cuda

    # 2. is the GPU even saturated? an M2-sized batch will not fill a 3090
    .venv/bin/python bench/generation_profile.py --device cuda --sweep 256,512,1024,2048,4096

    # 3. tensor cores, which MPS never gave us
    .venv/bin/python bench/generation_profile.py --device cuda --tf32

The number that decides the rental is "GPU-bound fraction" in run 1.
Laptop baseline for comparison is in bench/results/.
