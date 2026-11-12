//! The `.azmw` weight file written by `python -m chessbot.export_metal`.
//!
//! A flat list of named little-endian float32 tensors, BatchNorm already folded
//! into the convolutions. Plain Rust, so it builds and is tested everywhere.

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// The network's shape, read off the tensors rather than trusted from a header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    pub in_planes: usize,
    pub channels: usize,
    pub blocks: usize,
    pub se_hidden: usize,
    pub policy_bottleneck: usize,
    pub policy_len: usize,
    pub value_planes: usize,
    pub value_hidden: usize,
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub arch: Arch,
    pub tensors: HashMap<String, Tensor>,
}

fn take<'a>(buf: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
    if buf.len() < n {
        return Err("truncated weight file".into());
    }
    let (a, b) = buf.split_at(n);
    *buf = b;
    Ok(a)
}

fn u32le(buf: &mut &[u8]) -> Result<u32, String> {
    Ok(u32::from_le_bytes(take(buf, 4)?.try_into().unwrap()))
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| format!("{}: {e}", path.as_ref().display()))?;
        Self::parse(&bytes)
    }

    pub fn parse(mut buf: &[u8]) -> Result<Self, String> {
        let b = &mut buf;
        if take(b, 4)? != b"AZMW" {
            return Err("not an .azmw file".into());
        }
        let version = u32le(b)?;
        if version != 1 {
            return Err(format!("unsupported .azmw version {version}"));
        }
        let count = u32le(b)?;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let nl = u16::from_le_bytes(take(b, 2)?.try_into().unwrap()) as usize;
            let name = String::from_utf8(take(b, nl)?.to_vec()).map_err(|e| e.to_string())?;
            let ndim = u32le(b)? as usize;
            let shape = (0..ndim).map(|_| u32le(b).map(|d| d as usize)).collect::<Result<Vec<_>, _>>()?;
            let n: usize = shape.iter().product();
            let data = take(b, n * 4)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            tensors.insert(name, Tensor { shape, data });
        }
        if !buf.is_empty() {
            return Err("trailing bytes after the last tensor".into());
        }
        let arch = Self::infer(&tensors)?;
        Ok(Self { arch, tensors })
    }

    fn infer(t: &HashMap<String, Tensor>) -> Result<Arch, String> {
        let get = |k: &str| t.get(k).ok_or_else(|| format!("missing tensor {k}"));
        let stem = &get("stem.w")?.shape;
        let blocks = (0..).take_while(|i| t.contains_key(&format!("tower.{i}.c1.w"))).count();
        let arch = Arch {
            channels: stem[0],
            in_planes: stem[1],
            blocks,
            se_hidden: if blocks > 0 { get("tower.0.se.fc1.w")?.shape[0] } else { 0 },
            policy_bottleneck: get("policy_conv.w")?.shape[0],
            policy_len: get("policy_fc.w")?.shape[0],
            value_planes: get("value_conv.w")?.shape[0],
            value_hidden: get("value_fc.w")?.shape[0],
        };
        Ok(arch)
    }

    pub fn get(&self, name: &str) -> &Tensor {
        self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(out: &mut Vec<u8>, name: &str, shape: &[u32]) {
        out.extend((name.len() as u16).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((shape.len() as u32).to_le_bytes());
        for d in shape {
            out.extend(d.to_le_bytes());
        }
        let n: u32 = shape.iter().product();
        for i in 0..n {
            out.extend((i as f32).to_le_bytes());
        }
    }

    #[test]
    fn parses_and_infers_the_architecture() {
        let names: &[(&str, &[u32])] = &[
            ("stem.w", &[8, 119, 3, 3]),
            ("tower.0.c1.w", &[8, 8, 3, 3]),
            ("tower.0.se.fc1.w", &[4, 8]),
            ("policy_conv.w", &[32, 8, 1, 1]),
            ("policy_fc.w", &[4672, 2048]),
            ("value_conv.w", &[32, 8, 1, 1]),
            ("value_fc.w", &[128, 2048]),
        ];
        let mut f = b"AZMW".to_vec();
        f.extend(1u32.to_le_bytes());
        f.extend((names.len() as u32).to_le_bytes());
        for (n, s) in names {
            tensor(&mut f, n, s);
        }
        let w = Weights::parse(&f).unwrap();
        assert_eq!(w.arch.channels, 8);
        assert_eq!(w.arch.in_planes, 119);
        assert_eq!(w.arch.blocks, 1);
        assert_eq!(w.arch.policy_len, 4672);
        assert_eq!(w.get("tower.0.se.fc1.w").data[5], 5.0);
        assert!(Weights::parse(&f[..f.len() - 1]).is_err());
    }
}
