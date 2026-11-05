//! avatar-rs patch: two kernels tract runs slowly on MediaPipe's NHWC face models.
//!
//! - PRELU: one pass with the per-channel slope read in the inner loop. The generic
//!   decomposition (min, mul, add) took three passes and a slope tensor spread to the full
//!   activation shape (14.5 MB for the face mesh, 5x the model).
//! - DEPTHWISE_CONV_2D: tract's depthwise kernel walks channel by channel over the image,
//!   which for NHWC means strided loads and no SIMD. Here the inner loop runs over the
//!   contiguous channels of one pixel, so it vectorizes.
//!
//! Face mesh, 7800X3D, x86-64-v3: PRELU 2.9 -> 0.4 ms, DEPTHWISE 9.2 -> 3.3 ms per run; the
//! whole model 21.7 -> 12.1 ms and 39 -> 22 MB.

use tract_core::internal::*;

/// y = x for x >= 0, else slope[c] * x, with c the last (channel) axis.
#[derive(Debug, Clone)]
pub struct ChannelPrelu {
    pub slope: Arc<Tensor>,
}

impl Op for ChannelPrelu {
    fn name(&self) -> StaticName {
        "ChannelPrelu".into()
    }
    op_as_typed_op!();
}

impl EvalOp for ChannelPrelu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let mut x = input.into_tensor();
        let slope = self.slope.as_slice::<f32>()?;
        for px in x.as_slice_mut::<f32>()?.chunks_exact_mut(slope.len()) {
            for (v, a) in px.iter_mut().zip(slope) {
                *v = if *v < 0.0 { *v * a } else { *v };
            }
        }
        Ok(tvec!(x.into_tvalue()))
    }
}

impl TypedOp for ChannelPrelu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].without_value()))
    }
    as_op!();
}

/// NHWC depthwise convolution, depth multiplier 1, no dilation. `kernel` is [kh, kw, c].
#[derive(Debug, Clone)]
pub struct NhwcDepthwise {
    pub kernel: Arc<Tensor>,
    pub bias: Arc<Tensor>,
    pub kh: usize,
    pub kw: usize,
    pub stride: (usize, usize),
    pub same: bool,
}

impl NhwcDepthwise {
    /// Output size and leading padding along one axis (TFLite SAME / VALID).
    fn geometry(&self, input: usize, k: usize, stride: usize) -> (usize, usize) {
        if self.same {
            let out = input.div_ceil(stride);
            let pad = ((out - 1) * stride + k).saturating_sub(input);
            (out, pad / 2)
        } else {
            ((input - k) / stride + 1, 0)
        }
    }
}

impl Op for NhwcDepthwise {
    fn name(&self) -> StaticName {
        "NhwcDepthwise".into()
    }
    op_as_typed_op!();
}

impl EvalOp for NhwcDepthwise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let &[n, h, w, c] = input.shape() else { bail!("NhwcDepthwise wants rank 4") };
        let (oh, top) = self.geometry(h, self.kh, self.stride.0);
        let (ow, left) = self.geometry(w, self.kw, self.stride.1);
        let x = input.as_slice::<f32>()?;
        let (k, bias) = (self.kernel.as_slice::<f32>()?, self.bias.as_slice::<f32>()?);
        let mut out = vec![0f32; n * oh * ow * c];
        for b in 0..n {
            let x = &x[b * h * w * c..][..h * w * c];
            for (oy, row) in out[b * oh * ow * c..][..oh * ow * c].chunks_exact_mut(ow * c).enumerate() {
                for (ox, o) in row.chunks_exact_mut(c).enumerate() {
                    o.copy_from_slice(bias);
                    for ky in 0..self.kh {
                        let Some(iy) = (oy * self.stride.0 + ky).checked_sub(top).filter(|&y| y < h) else { continue };
                        for kx in 0..self.kw {
                            let Some(ix) = (ox * self.stride.1 + kx).checked_sub(left).filter(|&x| x < w) else { continue };
                            let (px, kk) = (&x[(iy * w + ix) * c..][..c], &k[(ky * self.kw + kx) * c..][..c]);
                            for ((o, p), q) in o.iter_mut().zip(px).zip(kk) {
                                *o += p * q;
                            }
                        }
                    }
                }
            }
        }
        Ok(tvec!(Tensor::from_shape(&[n, oh, ow, c], &out)?.into_tvalue()))
    }
}

impl TypedOp for NhwcDepthwise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let s = inputs[0].shape.as_concrete().context("NhwcDepthwise wants a concrete shape")?;
        let (oh, ow) = (self.geometry(s[1], self.kh, self.stride.0).0, self.geometry(s[2], self.kw, self.stride.1).0);
        Ok(tvec!(f32::fact([s[0], oh, ow, s[3]])))
    }
    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5 }).collect()
    }

    /// Same result as tract's own grouped Conv, for SAME/VALID, stride 1/2, odd sizes.
    #[test]
    fn depthwise_matches_tract_conv() {
        for (h, w, c, k, stride, same) in [(9, 7, 8, 3, 1, true), (10, 11, 16, 5, 2, true), (9, 9, 4, 3, 2, false), (6, 5, 3, 3, 1, false)] {
            let x = Tensor::from_shape(&[1, h, w, c], &noise(h * w * c, 1)).unwrap();
            let kern = noise(k * k * c, 2);
            let bias = noise(c, 3);
            let fast = NhwcDepthwise {
                kernel: Tensor::from_shape(&[k, k, c], &kern).unwrap().into_arc_tensor(),
                bias: Tensor::from_shape(&[c], &bias).unwrap().into_arc_tensor(),
                kh: k, kw: k, stride: (stride, stride), same,
            };
            let got = fast.eval(tvec!(x.clone().into_tvalue())).unwrap().remove(0);
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NHWC,
                    kernel_shape: tvec!(k, k),
                    padding: if same { PaddingSpec::SameUpper } else { PaddingSpec::Valid },
                    strides: Some(tvec!(stride, stride)),
                    dilations: None,
                    input_channels: c,
                    output_channels: c,
                },
                kernel_fmt: KernelFormat::OHWI,
                group: c,
                q_params: None,
            };
            let mut model = TypedModel::default();
            let src = model.add_source("x", f32::fact([1, h, w, c])).unwrap();
            let kw = model.add_const("k", Tensor::from_shape(&[1, k, k, c], &kern).unwrap()).unwrap();
            let bw = model.add_const("b", Tensor::from_shape(&[c], &bias).unwrap()).unwrap();
            let out = model.wire_node("conv", conv, &[src, kw, bw]).unwrap();
            model.set_output_outlets(&out).unwrap();
            let want = model.into_runnable().unwrap().run(tvec!(x.into_tvalue())).unwrap().remove(0);
            assert_eq!(got.shape(), want.shape(), "{h}x{w}x{c} k{k} s{stride} same {same}");
            let diff = got.as_slice::<f32>().unwrap().iter().zip(want.as_slice::<f32>().unwrap()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(diff < 1e-5, "{h}x{w}x{c} k{k} s{stride} same {same}: off by {diff}");
        }
    }

    #[test]
    fn prelu_scales_only_negatives_per_channel() {
        let op = ChannelPrelu { slope: Tensor::from_shape(&[2], &[0.5f32, 2.0]).unwrap().into_arc_tensor() };
        let x = Tensor::from_shape(&[1, 2, 2], &[-1.0f32, -1.0, 3.0, -2.0]).unwrap();
        let y = op.eval(tvec!(x.into_tvalue())).unwrap().remove(0);
        assert_eq!(y.as_slice::<f32>().unwrap(), &[-0.5, -2.0, 3.0, -4.0]);
    }
}
