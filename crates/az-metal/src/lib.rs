//! Native Metal inference for the chess network, without Python on the path.
//!
//! A prototype: it builds the incumbent 6x96 SE-ResNet (fc policy head, WDL
//! head) as an MPSGraph from an `.azmw` file, takes the packed bridge format
//! (`PackedBatch`: u64 plane masks + scales, legal policy indices), unpacks it
//! on the GPU and returns only the legal logits and the value. Not wired into
//! self-play or training.
//!
//! Everything Metal is `cfg(target_os = "macos")`; elsewhere only the weight
//! file reader exists, so the workspace keeps building on Linux.

pub mod weights;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "macos")]
pub use metal::{MetalNet, Outputs, Precision};
pub use weights::{Arch, Weights};
