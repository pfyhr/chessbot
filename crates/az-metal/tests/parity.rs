//! Same weights, same packed inputs: the MPSGraph network must match PyTorch.
//!
//! The fixture (real self-play positions in the packed format, plus PyTorch
//! CPU fp32 outputs for a random-init 6x96 with non-trivial BatchNorm) is
//! written by `python -m chessbot.export_metal`. If `AZ_METAL_FIXTURE` is not
//! set, the test generates one with the repo's `.venv`, and skips loudly if
//! there is no venv to do it with.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use az_metal::{MetalNet, Precision, Weights};

/// fp32: the logits are sums of ~2000 products through 13 convolutions, and
/// MPSGraph is free to reassociate them (and folds BatchNorm differently from
/// PyTorch's separate op), so this is a relative tolerance on the scale of the
/// outputs, not bit equality.
const F32_TOL: f32 = 1e-5;
/// fp16: weights and activations in half precision; a sanity bound, not a
/// claim about quality (measured ~1e-3 relative on random init).
const F16_TOL: f32 = 1e-2;

fn read<T: Copy>(p: &Path) -> Vec<T> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let n = b.len() / std::mem::size_of::<T>();
    let mut v = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len());
        v.set_len(n);
    }
    v
}

fn fixture() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AZ_METAL_FIXTURE") {
        return Some(PathBuf::from(p));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let py = root.join(".venv/bin/python");
    if !py.exists() {
        eprintln!("SKIPPED: no .venv and no AZ_METAL_FIXTURE");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("az-metal-fixture-{}", std::process::id()));
    let ok = Command::new(py)
        .current_dir(&root)
        .args(["-m", "chessbot.export_metal", "--positions", "512"])
        .arg("--out")
        .arg(dir.join("net.azmw"))
        .arg("--fixture")
        .arg(&dir)
        .status()
        .expect("run exporter")
        .success();
    assert!(ok, "exporter failed");
    Some(dir)
}

struct Errs {
    logits: f32,
    legal: f32,
    wdl: f32,
    value: f32,
    scale: f32,
}

fn compare(dir: &Path, precision: Precision, n: usize) -> Errs {
    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let m: Vec<usize> = meta.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (total, planes, width, plen) = (m[0], m[1], m[2], m[3]);
    assert!(n <= total);
    let bits: Vec<u64> = read(&dir.join("bits.u64"));
    let scale: Vec<f32> = read(&dir.join("scale.f32"));
    let legal: Vec<i32> = read(&dir.join("legal.i32"));
    let logits: Vec<f32> = read(&dir.join("logits.f32"));
    let wdl: Vec<f32> = read(&dir.join("wdl.f32"));
    let value: Vec<f32> = read(&dir.join("value.f32"));

    let w = Weights::load(dir.join("net.azmw")).unwrap();
    let mut net = MetalNet::new(w, precision).unwrap();
    let out = net.forward_packed(
        n,
        width,
        &bits[..n * planes],
        &scale[..n * planes],
        &legal[..n * width],
        true,
    );
    let maxdiff = |a: &[f32], b: &[f32]| {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    };
    let got_logits = out.logits.unwrap();
    let ref_legal: Vec<f32> = (0..n * width)
        .map(|k| logits[(k / width) * plen + legal[k] as usize])
        .collect();
    let scale_of = logits[..n * plen].iter().fold(0.0f32, |m, x| m.max(x.abs()));
    Errs {
        logits: maxdiff(got_logits, &logits[..n * plen]),
        legal: maxdiff(out.legal_logits, &ref_legal),
        wdl: maxdiff(out.wdl, &wdl[..n * 3]),
        value: maxdiff(out.values, &value[..n]),
        scale: scale_of,
    }
}

#[test]
fn matches_pytorch() {
    let Some(dir) = fixture() else { return };
    for (prec, tol) in [(Precision::F32, F32_TOL), (Precision::F16, F16_TOL)] {
        // Two batch sizes, because each compiles its own graph.
        for n in [512, 37] {
            let e = compare(&dir, prec, n);
            eprintln!(
                "{prec:?} n={n}: max |logit| {:.3}; max abs err logits {:.2e} legal {:.2e} wdl {:.2e} value {:.2e}",
                e.scale, e.logits, e.legal, e.wdl, e.value
            );
            let bound = tol * e.scale.max(1.0);
            assert!(e.logits <= bound, "{prec:?} logits off by {} (> {bound})", e.logits);
            assert!(e.legal <= bound, "{prec:?} legal logits off by {}", e.legal);
            assert!(e.wdl <= bound, "{prec:?} wdl off by {}", e.wdl);
            assert!(e.value <= tol, "{prec:?} value off by {}", e.value);
            // The gather must be exactly the full logits at the legal indices.
            assert!(e.legal <= e.logits);
        }
    }
    if std::env::var_os("AZ_METAL_FIXTURE").is_none() {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
