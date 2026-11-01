//! Whole-buffer band-limited resampling with a Kaiser-windowed sinc. Zero
//! phase, so output sample `i` sits exactly at input time `i * from / to`.

use anyhow::{ensure, Result};

const ZERO_CROSSINGS: f64 = 32.0;
const CUTOFF: f64 = 0.97;
const BETA: f64 = 12.0;
const TABLE_PER_CROSSING: f64 = 512.0;

/// Modified Bessel function of the first kind, order zero.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..64 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

fn kernel(distance: f64, width: f64, cutoff: f64) -> f64 {
    let ratio = distance / width;
    if ratio.abs() >= 1.0 {
        return 0.0;
    }
    let x = std::f64::consts::PI * distance * cutoff;
    let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
    let window = bessel_i0(BETA * (1.0 - ratio * ratio).sqrt()) / bessel_i0(BETA);
    cutoff * sinc * window
}

pub fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    ensure!(from > 0 && to > 0, "sample rates must be positive");
    if from == to {
        return Ok(input.to_vec());
    }
    let step = f64::from(from) / f64::from(to);
    let cutoff = (1.0 / step).min(1.0) * CUTOFF;
    let width = ZERO_CROSSINGS / cutoff;
    let scale = TABLE_PER_CROSSING * cutoff;
    let table: Vec<f64> = (0..=(width * scale).ceil() as usize + 1)
        .map(|i| kernel(i as f64 / scale, width, cutoff))
        .collect();
    let lookup = |distance: f64| {
        let position = distance.abs() * scale;
        let low = position as usize;
        table[low] + (table[low + 1] - table[low]) * (position - low as f64)
    };
    let length = (input.len() as f64 / step).round() as usize;
    let last = input.len() as isize - 1;
    let output = (0..length)
        .map(|i| {
            let center = i as f64 * step;
            let low = (center - width).ceil().max(0.0) as isize;
            let high = ((center + width).floor() as isize).min(last);
            let sum: f64 = (low..=high)
                .map(|n| f64::from(input[n as usize]) * lookup(center - n as f64))
                .sum();
            sum as f32
        })
        .collect();
    Ok(output)
}
