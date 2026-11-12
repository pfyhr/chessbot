//! The network as an MPSGraph, fed and read through shared-memory buffers.
//!
//! One graph is built and cached per `(batch, legal width, full logits)`, with
//! its input and output `MTLBuffer`s allocated once in shared storage. A call
//! copies the packed batch into those buffers (about 1.4 KB a position), runs
//! the graph synchronously on one command queue, and hands back slices of the
//! output buffers -- no Python, no framework tensors, no per-call allocation.
//!
//! On the device: the u64 plane masks are read as pairs of u32 words and
//! expanded with shifts, multiplied by their scales (exact in fp32: every
//! square is 0 or 1 times one value), run through the network, and the policy
//! is gathered down to each row's legal indices before anything is read back.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber};
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphPaddingStyle, MPSGraphTensor,
    MPSGraphTensorData, MPSGraphTensorDataDictionary, MPSGraphTensorNamedDataLayout,
};

use crate::weights::{Arch, Weights};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    fn dtype(self) -> MPSDataType {
        match self {
            Precision::F32 => MPSDataType::Float32,
            Precision::F16 => MPSDataType::Float16,
        }
    }
}

/// What one call returns: views into the graph's output buffers, valid until
/// the next call.
pub struct Outputs<'a> {
    /// `(n, width)`: the logits at `legal`, row-major.
    pub legal_logits: &'a [f32],
    /// `(n,)`: P(win) - P(loss), what MCTS backs up.
    pub values: &'a [f32],
    /// `(n, 3)`: the raw WDL logits.
    pub wdl: &'a [f32],
    /// `(n, policy_len)`, only when asked for (parity checks).
    pub logits: Option<&'a [f32]>,
}

struct Compiled {
    graph: Retained<MPSGraph>,
    feeds: Retained<MPSGraphTensorDataDictionary>,
    results: Retained<MPSGraphTensorDataDictionary>,
    bits: Buffer,
    scale: Buffer,
    legal: Buffer,
    legal_out: Buffer,
    value_out: Buffer,
    wdl_out: Buffer,
    logits_out: Option<Buffer>,
}

pub struct MetalNet {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    weights: Weights,
    precision: Precision,
    board: (usize, usize),
    cache: HashMap<(usize, usize, bool), Compiled>,
}

fn shape(dims: &[isize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = dims.iter().map(|&d| NSNumber::new_isize(d)).collect();
    NSArray::from_retained_slice(&v)
}

fn buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("MTLBuffer allocation failed")
}

fn contents<T>(b: &Buffer) -> NonNull<T> {
    b.contents().cast()
}

fn tensor_data(b: &Buffer, dims: &[isize], dt: MPSDataType) -> Retained<MPSGraphTensorData> {
    unsafe {
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
            MPSGraphTensorData::alloc(),
            b,
            &shape(dims),
            dt,
        )
    }
}

/// Graph-building helpers over one `MPSGraph` and one set of weights.
struct Builder<'a> {
    g: &'a MPSGraph,
    w: &'a Weights,
    dt: MPSDataType,
}

impl Builder<'_> {
    fn constant(&self, data: &[f32], dims: &[isize]) -> Retained<MPSGraphTensor> {
        let bytes: Vec<u8> = match self.dt {
            MPSDataType::Float16 => data
                .iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect(),
            _ => data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        };
        let d = NSData::with_bytes(&bytes);
        unsafe { self.g.constantWithData_shape_dataType(&d, &shape(dims), self.dt) }
    }

    fn conv(&self, x: &MPSGraphTensor, name: &str) -> Retained<MPSGraphTensor> {
        let w = self.w.get(&format!("{name}.w"));
        let b = self.w.get(&format!("{name}.b"));
        let (o, i, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        let pad = kh / 2;
        let wt = self.constant(&w.data, &[o as isize, i as isize, kh as isize, kw as isize]);
        let desc = unsafe {
            MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                1, 1, 1, 1, 1, pad, pad, pad, pad,
                MPSGraphPaddingStyle::Explicit,
                MPSGraphTensorNamedDataLayout::NCHW,
                MPSGraphTensorNamedDataLayout::OIHW,
            )
        }
        .expect("conv descriptor");
        let _ = kw;
        let y = unsafe { self.g.convolution2DWithSourceTensor_weightsTensor_descriptor_name(x, &wt, &desc, None) };
        let bias = self.constant(&b.data, &[1, o as isize, 1, 1]);
        self.add(&y, &bias)
    }

    /// `x @ W.T + b` with PyTorch's `(out, in)` weight, transposed here once.
    fn linear(&self, x: &MPSGraphTensor, name: &str) -> Retained<MPSGraphTensor> {
        let w = self.w.get(&format!("{name}.w"));
        let b = self.w.get(&format!("{name}.b"));
        let (o, i) = (w.shape[0], w.shape[1]);
        let mut t = vec![0.0f32; o * i];
        for r in 0..o {
            for c in 0..i {
                t[c * o + r] = w.data[r * i + c];
            }
        }
        let wt = self.constant(&t, &[i as isize, o as isize]);
        let y = unsafe { self.g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(x, &wt, None) };
        let bias = self.constant(&b.data, &[1, o as isize]);
        self.add(&y, &bias)
    }

    fn add(&self, a: &MPSGraphTensor, b: &MPSGraphTensor) -> Retained<MPSGraphTensor> {
        unsafe { self.g.additionWithPrimaryTensor_secondaryTensor_name(a, b, None) }
    }
    fn mul(&self, a: &MPSGraphTensor, b: &MPSGraphTensor) -> Retained<MPSGraphTensor> {
        unsafe { self.g.multiplicationWithPrimaryTensor_secondaryTensor_name(a, b, None) }
    }
    fn relu(&self, a: &MPSGraphTensor) -> Retained<MPSGraphTensor> {
        unsafe { self.g.reLUWithTensor_name(a, None) }
    }
    fn reshape(&self, a: &MPSGraphTensor, dims: &[isize]) -> Retained<MPSGraphTensor> {
        unsafe { self.g.reshapeTensor_withShape_name(a, &shape(dims), None) }
    }
    fn cast(&self, a: &MPSGraphTensor, dt: MPSDataType) -> Retained<MPSGraphTensor> {
        unsafe { self.g.castTensor_toType_name(a, dt, None) }
    }
    fn slice(&self, a: &MPSGraphTensor, dim: usize, start: isize, len: isize) -> Retained<MPSGraphTensor> {
        unsafe { self.g.sliceTensor_dimension_start_length_name(a, dim, start, len, None) }
    }

    /// `SEBlock`: x * sigmoid(scale) + bias, both from the pooled board.
    fn se(&self, x: &MPSGraphTensor, n: isize, c: isize, name: &str) -> Retained<MPSGraphTensor> {
        let axes = NSArray::from_retained_slice(&[NSNumber::new_isize(2), NSNumber::new_isize(3)]);
        let s = unsafe { self.g.meanOfTensor_axes_name(x, &axes, None) };
        let s = self.reshape(&s, &[n, c]);
        let s = self.relu(&self.linear(&s, &format!("{name}.fc1")));
        let s = self.linear(&s, &format!("{name}.fc2"));
        let scale = self.reshape(&self.slice(&s, 1, 0, c), &[n, c, 1, 1]);
        let bias = self.reshape(&self.slice(&s, 1, c, c), &[n, c, 1, 1]);
        let gate = unsafe { self.g.sigmoidWithTensor_name(&scale, None) };
        self.add(&self.mul(x, &gate), &bias)
    }
}

impl MetalNet {
    pub fn new(weights: Weights, precision: Precision) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let a = weights.arch;
        let hw = weights.get("policy_fc.w").shape[1] / a.policy_bottleneck;
        if hw != 64 {
            return Err(format!("expected an 8x8 board, policy_fc implies {hw} squares"));
        }
        Ok(Self { device, queue, weights, precision, board: (8, 8), cache: HashMap::new() })
    }

    pub fn arch(&self) -> Arch {
        self.weights.arch
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    fn build(&self, n: usize, width: usize, full: bool) -> Compiled {
        let a = self.weights.arch;
        let (h, w) = self.board;
        let hw = (h * w) as isize;
        let (ni, p, c) = (n as isize, a.in_planes as isize, a.channels as isize);
        let dt = self.precision.dtype();
        let g = unsafe { MPSGraph::new() };
        let bld = Builder { g: &g, w: &self.weights, dt };

        // Inputs: u64 masks as (n, planes, 2) u32 words, little-endian, so
        // word 0 holds squares 0..32 and word 1 squares 32..64.
        let bits_ph = unsafe { g.placeholderWithShape_dataType_name(Some(&shape(&[ni, p, 2])), MPSDataType::UInt32, None) };
        let scale_ph = unsafe { g.placeholderWithShape_dataType_name(Some(&shape(&[ni, p])), MPSDataType::Float32, None) };
        let legal_ph = unsafe { g.placeholderWithShape_dataType_name(Some(&shape(&[ni, width as isize])), MPSDataType::Int32, None) };

        // Unpack on the device: ((word >> k) & 1) for k in 0..32, then times scale.
        let shifts: Vec<u8> = (0u32..32).flat_map(|k| k.to_le_bytes()).collect();
        let shifts = unsafe { g.constantWithData_shape_dataType(&NSData::with_bytes(&shifts), &shape(&[1, 1, 1, 32]), MPSDataType::UInt32) };
        let one = unsafe { g.constantWithScalar_dataType(1.0, MPSDataType::UInt32) };
        let words = bld.reshape(&bits_ph, &[ni, p, 2, 1]);
        let bit = unsafe { g.bitwiseRightShiftWithPrimaryTensor_secondaryTensor_name(&words, &shifts, None) };
        let bit = unsafe { g.bitwiseANDWithPrimaryTensor_secondaryTensor_name(&bit, &one, None) };
        let planes = bld.reshape(&bld.cast(&bit, MPSDataType::Float32), &[ni, p, hw]);
        let scale = bld.reshape(&scale_ph, &[ni, p, 1]);
        let x = bld.mul(&planes, &scale);
        let x = bld.cast(&bld.reshape(&x, &[ni, p, h as isize, w as isize]), dt);

        // Trunk.
        let mut x = bld.relu(&bld.conv(&x, "stem"));
        for i in 0..a.blocks {
            let y = bld.relu(&bld.conv(&x, &format!("tower.{i}.c1")));
            let y = bld.conv(&y, &format!("tower.{i}.c2"));
            let y = bld.se(&y, ni, c, &format!("tower.{i}.se"));
            x = bld.relu(&bld.add(&x, &y));
        }

        // Policy: 1x1 conv to the bottleneck, flatten in NCHW order (as
        // PyTorch's `flatten(1)`), one big linear.
        let pb = a.policy_bottleneck as isize;
        let pol = bld.relu(&bld.conv(&x, "policy_conv"));
        let logits = bld.linear(&bld.reshape(&pol, &[ni, pb * hw]), "policy_fc");
        let legal_logits = unsafe { g.gatherAlongAxis_withUpdatesTensor_indicesTensor_name(1, &logits, &legal_ph, None) };
        let legal_logits = bld.cast(&legal_logits, MPSDataType::Float32);

        // Value: WDL logits, and P(win) - P(loss) for the search.
        let vp = a.value_planes as isize;
        let v = bld.relu(&bld.conv(&x, "value_conv"));
        let v = bld.relu(&bld.linear(&bld.reshape(&v, &[ni, vp * hw]), "value_fc"));
        let wdl = bld.cast(&bld.linear(&v, "wdl"), MPSDataType::Float32);
        let prob = unsafe { g.softMaxWithTensor_axis_name(&wdl, 1, None) };
        let value = unsafe {
            g.subtractionWithPrimaryTensor_secondaryTensor_name(&bld.slice(&prob, 1, 0, 1), &bld.slice(&prob, 1, 2, 1), None)
        };
        let value = bld.reshape(&value, &[ni]);

        let dev = &*self.device;
        let bits = buffer(dev, n * a.in_planes * 8);
        let scale_b = buffer(dev, n * a.in_planes * 4);
        let legal = buffer(dev, n * width * 4);
        let legal_out = buffer(dev, n * width * 4);
        let value_out = buffer(dev, n * 4);
        let wdl_out = buffer(dev, n * 12);
        let logits_out = full.then(|| buffer(dev, n * a.policy_len * 4));

        let feeds = NSDictionary::from_retained_objects(
            &[&*bits_ph, &*scale_ph, &*legal_ph],
            &[
                tensor_data(&bits, &[ni, p, 2], MPSDataType::UInt32),
                tensor_data(&scale_b, &[ni, p], MPSDataType::Float32),
                tensor_data(&legal, &[ni, width as isize], MPSDataType::Int32),
            ],
        );
        let mut keys: Vec<&MPSGraphTensor> = vec![&legal_logits, &value, &wdl];
        let mut vals = vec![
            tensor_data(&legal_out, &[ni, width as isize], MPSDataType::Float32),
            tensor_data(&value_out, &[ni], MPSDataType::Float32),
            tensor_data(&wdl_out, &[ni, 3], MPSDataType::Float32),
        ];
        let logits32;
        if let Some(lb) = &logits_out {
            logits32 = bld.cast(&logits, MPSDataType::Float32);
            keys.push(&logits32);
            vals.push(tensor_data(lb, &[ni, a.policy_len as isize], MPSDataType::Float32));
        }
        let results = NSDictionary::from_retained_objects(&keys, &vals);

        Compiled {
            graph: g,
            feeds,
            results,
            bits,
            scale: scale_b,
            legal,
            legal_out,
            value_out,
            wdl_out,
            logits_out,
        }
    }

    /// Evaluate `n` positions in the packed bridge format.
    ///
    /// `bits`/`scale` are `(n, planes)` as `PackedBatch` holds them; `legal`
    /// is `(n, width)`, padded with any valid index. `full_logits` also returns
    /// the whole policy, for parity checks; self-play never needs it.
    pub fn forward_packed(
        &mut self,
        n: usize,
        width: usize,
        bits: &[u64],
        scale: &[f32],
        legal: &[i32],
        full_logits: bool,
    ) -> Outputs<'_> {
        let a = self.weights.arch;
        assert_eq!(bits.len(), n * a.in_planes);
        assert_eq!(scale.len(), n * a.in_planes);
        assert_eq!(legal.len(), n * width);
        assert!(legal.iter().all(|&i| (i as usize) < a.policy_len), "legal index out of range");
        let key = (n, width, full_logits);
        if !self.cache.contains_key(&key) {
            let c = autoreleasepool(|_| self.build(n, width, full_logits));
            self.cache.insert(key, c);
        }
        let c = &self.cache[&key];
        unsafe {
            std::ptr::copy_nonoverlapping(bits.as_ptr(), contents::<u64>(&c.bits).as_ptr(), bits.len());
            std::ptr::copy_nonoverlapping(scale.as_ptr(), contents::<f32>(&c.scale).as_ptr(), scale.len());
            std::ptr::copy_nonoverlapping(legal.as_ptr(), contents::<i32>(&c.legal).as_ptr(), legal.len());
        }
        autoreleasepool(|_| unsafe {
            c.graph.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                &self.queue,
                &c.feeds,
                None,
                &c.results,
            );
        });
        // SAFETY: the run above is synchronous, so the GPU is done with these
        // shared buffers, and they live as long as `self`.
        unsafe {
            let sl = |b: &Buffer, len: usize| std::slice::from_raw_parts(contents::<f32>(b).as_ptr(), len);
            Outputs {
                legal_logits: sl(&c.legal_out, n * width),
                values: sl(&c.value_out, n),
                wdl: sl(&c.wdl_out, n * 3),
                logits: c.logits_out.as_ref().map(|b| sl(b, n * a.policy_len)),
            }
        }
    }
}
