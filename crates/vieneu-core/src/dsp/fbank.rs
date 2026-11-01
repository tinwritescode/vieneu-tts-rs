//! 80-bin Kaldi log-mel filterbank (povey window, snip_edges) with cepstral
//! mean normalization: the speaker encoder's front end at 16 kHz.

use anyhow::{bail, Result};
use realfft::RealFftPlanner;

pub const N_MELS: usize = 80;
pub const SAMPLE_RATE: usize = 16000;
const FRAME_LEN: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_LEN: usize = 512;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;

fn mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// One `(first_bin, weights)` triangle per mel bin.
fn mel_filters() -> Vec<(usize, Vec<f32>)> {
    let high = SAMPLE_RATE as f32 / 2.0;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(high));
    let delta = (mel_high - mel_low) / (N_MELS + 1) as f32;
    let hz_per_bin = SAMPLE_RATE as f32 / FFT_LEN as f32;
    (0..N_MELS)
        .map(|bin| {
            let left = mel_low + bin as f32 * delta;
            let center = left + delta;
            let right = center + delta;
            let mut first = None;
            let mut weights = Vec::new();
            for i in 0..FFT_LEN / 2 {
                let m = mel(hz_per_bin * i as f32);
                if m > left && m < right {
                    first.get_or_insert(i);
                    let slope = if m <= center { (m - left) / delta } else { (right - m) / delta };
                    weights.push(slope);
                }
            }
            (first.unwrap_or(0), weights)
        })
        .collect()
}

/// `(frames, N_MELS)` flattened, mean-normalized per bin.
pub fn speaker_fbank(wav: &[f32]) -> Result<Vec<f32>> {
    if wav.len() < FRAME_LEN {
        bail!("audio too short for fbank: {} samples", wav.len());
    }
    let frames = 1 + (wav.len() - FRAME_LEN) / FRAME_SHIFT;
    let window: Vec<f32> = (0..FRAME_LEN)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (FRAME_LEN - 1) as f64;
            (0.5 - 0.5 * phase.cos()).powf(0.85) as f32
        })
        .collect();
    let filters = mel_filters();
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_LEN);
    let mut spectrum = fft.make_output_vec();
    let mut buffer = vec![0.0f32; FFT_LEN];
    let mut features = Vec::with_capacity(frames * N_MELS);
    for index in 0..frames {
        let start = index * FRAME_SHIFT;
        buffer.fill(0.0);
        buffer[..FRAME_LEN].copy_from_slice(&wav[start..start + FRAME_LEN]);
        let mean = buffer[..FRAME_LEN].iter().sum::<f32>() / FRAME_LEN as f32;
        buffer[..FRAME_LEN].iter_mut().for_each(|s| *s -= mean);
        for i in (1..FRAME_LEN).rev() {
            buffer[i] -= PREEMPH * buffer[i - 1];
        }
        buffer[0] -= PREEMPH * buffer[0];
        buffer[..FRAME_LEN].iter_mut().zip(&window).for_each(|(s, w)| *s *= w);
        fft.process(&mut buffer, &mut spectrum)?;
        for (first, weights) in &filters {
            let energy: f32 = weights
                .iter()
                .zip(&spectrum[*first..])
                .map(|(w, c)| w * c.norm_sqr())
                .sum();
            features.push(energy.max(f32::EPSILON).ln());
        }
    }
    for bin in 0..N_MELS {
        let mean = features.iter().skip(bin).step_by(N_MELS).sum::<f32>() / frames as f32;
        features.iter_mut().skip(bin).step_by(N_MELS).for_each(|v| *v -= mean);
    }
    Ok(features)
}
