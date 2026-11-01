//! Pitch-preserving time stretch by Waveform Similarity Overlap-Add.

const NEUTRAL: f32 = 1e-3;
const FLOOR: f64 = 1e-8;

fn hann_fade(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / (len - 1) as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn resample_linear(input: &[f32], target: usize) -> Vec<f32> {
    let step = (input.len() - 1) as f32 / (target.max(2) - 1) as f32;
    (0..target)
        .map(|i| {
            let position = i as f32 * step;
            let low = position as usize;
            let high = (low + 1).min(input.len() - 1);
            input[low] + (input[high] - input[low]) * (position - low as f32)
        })
        .collect()
}

fn window(frame_len: usize) -> Vec<f32> {
    let taper = frame_len / 4;
    let mut win = vec![1.0f32; frame_len];
    for i in 0..taper {
        let value = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / taper as f32).cos());
        win[i] = value;
        win[frame_len - 1 - i] = value;
    }
    win
}

/// Best-matching segment start in `[low, high]` by normalized correlation.
fn best_match(input: &[f32], target: &[f32], low: usize, high: usize) -> usize {
    let len = target.len();
    let square = |s: f32| f64::from(s) * f64::from(s);
    let mut energy: f64 = input[low..low + len].iter().copied().map(square).sum();
    let (mut best, mut best_score) = (low, f64::NEG_INFINITY);
    for start in low..=high {
        let correlation: f32 = input[start..start + len].iter().zip(target).map(|(a, b)| a * b).sum();
        let score = f64::from(correlation) / energy.max(FLOOR).sqrt();
        if score > best_score {
            (best, best_score) = (start, score);
        }
        if start < high {
            energy += square(input[start + len]) - square(input[start]);
        }
    }
    best
}

/// Stretch `input` so it plays `rate` times faster, keeping the pitch.
pub fn time_stretch(input: &[f32], rate: f32, sample_rate: u32) -> Vec<f32> {
    let n = input.len();
    if (rate - 1.0).abs() < NEUTRAL || n == 0 {
        return input.to_vec();
    }
    let target_len = ((n as f64 / f64::from(rate)).round() as usize).max(1);
    if n < 256 {
        return resample_linear(input, target_len);
    }
    let mut frame_len = (sample_rate as f32 * 0.025) as usize;
    frame_len += frame_len % 2;
    if n < frame_len * 2 {
        frame_len = ((n / 3) & !1).max(32);
    }
    let hop = frame_len / 2;
    let delta = (frame_len / 2).min((sample_rate as f32 * 0.015) as usize) as isize;
    let win = window(frame_len);

    let frames = target_len.div_ceil(hop) + 2;
    let mut output = vec![0.0f32; (frames + 2) * hop + frame_len];
    let mut norm = output.clone();
    let mut overlap = |start: usize, position: usize| {
        for i in 0..frame_len {
            output[start + i] += input[position + i] * win[i];
            norm[start + i] += win[i];
        }
    };
    overlap(0, 0);
    let mut best = 0usize;
    for frame in 1..frames {
        let target = best + hop;
        if target + frame_len > n {
            break;
        }
        let nominal = (frame as f64 * hop as f64 * f64::from(rate)).round_ties_even() as isize;
        let last = (n - frame_len) as isize;
        let low = (nominal - delta).max(0);
        let high = last.min(nominal + delta);
        best = if low >= high {
            nominal.clamp(0, last) as usize
        } else {
            best_match(input, &input[target..target + frame_len], low as usize, high as usize)
        };
        overlap(frame * hop, best);
        if frame * hop + hop >= target_len {
            break;
        }
    }
    let mut result: Vec<f32> = output
        .iter()
        .zip(&norm)
        .take(target_len)
        .map(|(o, w)| if *w > 1e-4 { o / w } else { *o })
        .collect();

    let fade_len = (result.len() / 4).min((sample_rate as f32 * 0.002) as usize);
    if fade_len > 1 {
        let fade = hann_fade(fade_len);
        let end = result.len();
        for (i, gain) in fade.iter().enumerate() {
            result[i] *= gain;
            result[end - 1 - i] *= gain;
        }
    }
    result
}
