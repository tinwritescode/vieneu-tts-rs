//! STFT pair of the resemble-enhance denoiser: hann-periodic, centered with
//! reflect padding, last frame dropped.

use std::sync::Arc;

use anyhow::Result;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

pub const N_FFT: usize = 1680;
pub const HOP: usize = 420;
pub const BINS: usize = N_FFT / 2 + 1;

pub struct Stft {
    window: Vec<f64>,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

impl Default for Stft {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        let tau = 2.0 * std::f64::consts::PI;
        Self {
            window: (0..N_FFT).map(|i| 0.5 - 0.5 * (tau * i as f64 / N_FFT as f64).cos()).collect(),
            forward: planner.plan_fft_forward(N_FFT),
            inverse: planner.plan_fft_inverse(N_FFT),
        }
    }
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f64> {
    let last = x.len() - 1;
    let at = |i: isize| {
        let period = 2 * last.max(1) as isize;
        let m = i.rem_euclid(period) as usize;
        f64::from(x[if m > last { 2 * last - m } else { m }])
    };
    (-(pad as isize)..(x.len() + pad) as isize).map(at).collect()
}

impl Stft {
    /// Frames of `BINS` complex values.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<Vec<Complex<f64>>>> {
        let padded = reflect_pad(x, N_FFT / 2);
        let frames = 1 + (padded.len() - N_FFT) / HOP;
        let mut out = Vec::with_capacity(frames - 1);
        let mut buffer = vec![0.0; N_FFT];
        for frame in 0..frames - 1 {
            let segment = &padded[frame * HOP..frame * HOP + N_FFT];
            for ((b, s), w) in buffer.iter_mut().zip(segment).zip(&self.window) {
                *b = s * w;
            }
            let mut spectrum = self.forward.make_output_vec();
            self.forward.process(&mut buffer, &mut spectrum)?;
            out.push(spectrum);
        }
        Ok(out)
    }

    /// Inverse of [`Self::forward`]; the last frame is replicated back.
    pub fn inverse(&self, frames: &[Vec<Complex<f64>>], out_len: usize) -> Result<Vec<f32>> {
        let count = frames.len() + 1;
        let length = N_FFT + HOP * (count - 1);
        let (mut out, mut weight) = (vec![0.0; length], vec![0.0; length]);
        let mut buffer = vec![0.0; N_FFT];
        for index in 0..count {
            let mut spectrum = frames[index.min(frames.len() - 1)].clone();
            spectrum[0].im = 0.0;
            spectrum[BINS - 1].im = 0.0;
            self.inverse.process(&mut spectrum, &mut buffer)?;
            let start = index * HOP;
            for i in 0..N_FFT {
                let w = self.window[i];
                out[start + i] += buffer[i] / N_FFT as f64 * w;
                weight[start + i] += w * w;
            }
        }
        let body = N_FFT / 2..length - N_FFT / 2;
        let mut result: Vec<f32> = out[body.clone()]
            .iter()
            .zip(&weight[body])
            .map(|(o, w)| (o / w.max(1e-11)) as f32)
            .collect();
        result.resize(out_len, 0.0);
        Ok(result)
    }
}
