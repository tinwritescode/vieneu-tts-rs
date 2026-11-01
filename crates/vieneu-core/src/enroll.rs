//! Voice cloning: reference clip to speaker embedding and reference codes.
//! Graphs load per call and are dropped after, since enrollment is rare.

use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::value::Tensor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as DecodeError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::dsp::stft::{Stft, BINS};
use crate::dsp::{fbank, resample};
use crate::engine::{session, SAMPLE_RATE};
use crate::voice::Voice;

const MAX_REFERENCE_SECONDS: f32 = 8.0;
const SPEAKER_MAX_SECONDS: f32 = 30.0;
const DENOISE_RATE: u32 = 44100;
const DENOISE_PAD: usize = 441;
const TOP_DB: f32 = 30.0;
const TRIM_FRAME: usize = 2048;
const TRIM_HOP: usize = 512;
const NORM_FLOOR: f64 = 1e-7;

/// Decode any supported audio file to mono float samples.
pub fn read_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        stream,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("no audio track")?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.context("unknown sample rate")?;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(DecodeError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let channels = decoded.spec().channels.count();
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);
        samples.extend(
            buffer.samples().chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32),
        );
    }
    Ok((samples, rate))
}

/// Cut leading and trailing frames quieter than `top_db` below the peak frame.
pub fn trim_silence(wav: &[f32], top_db: f32) -> &[f32] {
    let frames = wav.len() / TRIM_HOP + 1;
    let rms: Vec<f32> = (0..frames)
        .map(|f| {
            let center = f * TRIM_HOP;
            let start = center.saturating_sub(TRIM_FRAME / 2);
            let end = (center + TRIM_FRAME / 2).min(wav.len());
            let sum: f32 = wav[start..end].iter().map(|s| s * s).sum();
            (sum / TRIM_FRAME as f32).sqrt()
        })
        .collect();
    let peak = rms.iter().copied().fold(0.0, f32::max);
    let threshold = peak * 10f32.powf(-top_db / 20.0);
    let mut loud = rms.iter().enumerate().filter(|(_, v)| **v > threshold).map(|(i, _)| i);
    let Some(first) = loud.next() else { return wav };
    let last = loud.next_back().unwrap_or(first);
    &wav[(first * TRIM_HOP).min(wav.len())..((last + 1) * TRIM_HOP).min(wav.len())]
}

fn seconds(wav: &[f32], rate: u32, limit: f32) -> &[f32] {
    &wav[..wav.len().min((limit * rate as f32) as usize)]
}

pub fn speaker_embedding(models: &Path, wav: &[f32], rate: u32, threads: usize) -> Result<Vec<f32>> {
    let clip = resample::resample(seconds(wav, rate, SPEAKER_MAX_SECONDS), rate, fbank::SAMPLE_RATE as u32)?;
    let features = fbank::speaker_fbank(&clip)?;
    let frames = features.len() / fbank::N_MELS;
    let mut graph = session(&models.join("vieneu/speaker_encoder.onnx"), threads)?;
    let input = Tensor::from_array(([1, frames, fbank::N_MELS], features))?;
    let outputs = graph.run(ort::inputs! { "input" => input })?;
    let (_, embedding) = outputs["output"].try_extract_tensor::<f32>()?;
    Ok(embedding.to_vec())
}

/// Denoise to 44.1 kHz with the resemble-enhance core.
pub fn denoise(models: &Path, wav: &[f32], rate: u32, threads: usize) -> Result<Vec<f32>> {
    let resampled = resample::resample(wav, rate, DENOISE_RATE)?;
    let length = resampled.len();
    let peak = resampled.iter().fold(0.0f32, |p, s| p.max(s.abs())).max(1e-7);
    let mut input: Vec<f32> = resampled.iter().map(|s| s / peak).collect();
    input.resize(length + DENOISE_PAD, 0.0);
    let inner = f64::from(input.iter().fold(0.0f32, |p, s| p.max(s.abs()))) + NORM_FLOOR;
    input.iter_mut().for_each(|s| *s = (f64::from(*s) / inner) as f32);

    let stft = Stft::default();
    let frames = stft.forward(&input)?;
    let count = frames.len();
    let shape = [1, BINS, count];
    let mut parts = [
        vec![0.0f32; BINS * count],
        vec![0.0f32; BINS * count],
        vec![0.0f32; BINS * count],
    ];
    for (t, frame) in frames.iter().enumerate() {
        for (f, value) in frame.iter().enumerate() {
            let at = f * count + t;
            let phase = value.im.atan2(value.re);
            parts[0][at] = value.norm() as f32;
            parts[1][at] = phase.cos() as f32;
            parts[2][at] = phase.sin() as f32;
        }
    }
    let [mag, cos, sin] = parts;
    let mut graph = session(&models.join("vieneu/denoiser.onnx"), threads)?;
    let outputs = graph.run(ort::inputs! {
        "mag" => Tensor::from_array((shape, mag))?,
        "cos" => Tensor::from_array((shape, cos))?,
        "sin" => Tensor::from_array((shape, sin))?,
    })?;
    let (_, sep_mag) = outputs[0].try_extract_tensor::<f32>()?;
    let (_, sep_cos) = outputs[1].try_extract_tensor::<f32>()?;
    let (_, sep_sin) = outputs[2].try_extract_tensor::<f32>()?;
    let mut spectrum = frames;
    for (t, frame) in spectrum.iter_mut().enumerate() {
        for (f, value) in frame.iter_mut().enumerate() {
            let at = f * count + t;
            let magnitude = f64::from(sep_mag[at]);
            value.re = magnitude * f64::from(sep_cos[at]);
            value.im = magnitude * f64::from(sep_sin[at]);
        }
    }
    let output = stft.inverse(&spectrum, input.len())?;
    Ok(output[..length].iter().map(|s| s * peak).collect())
}

/// MOSS codec reference codes, `(frames, n_vq)`.
pub fn reference_codes(models: &Path, wav: &[f32], rate: u32, threads: usize) -> Result<Vec<Vec<i64>>> {
    let audio = resample::resample(wav, rate, SAMPLE_RATE)?;
    let length = audio.len();
    let stereo = [audio.as_slice(), audio.as_slice()].concat();
    let mut graph = session(&models.join("codec/moss_audio_tokenizer_encode.onnx"), threads)?;
    let outputs = graph.run(ort::inputs! {
        "waveform" => Tensor::from_array(([1, 2, length], stereo))?,
        "input_lengths" => Tensor::from_array(([1usize], vec![length as i32]))?,
    })?;
    let (shape, codes) = outputs[0].try_extract_tensor::<i32>()?;
    if shape.len() != 3 {
        bail!("unexpected codec output shape {shape:?}");
    }
    Ok(codes
        .chunks_exact(shape[2] as usize)
        .map(|frame| frame.iter().map(|c| i64::from(*c)).collect())
        .collect())
}

/// Enroll a clip: trim, cap, optionally denoise, then embed and encode.
pub fn enroll(models: &Path, clip: &Path, use_denoiser: bool, threads: usize) -> Result<Voice> {
    let (wav, rate) = read_mono(clip)?;
    if wav.is_empty() {
        bail!("{} has no audio", clip.display());
    }
    let wav = seconds(trim_silence(&wav, TOP_DB), rate, MAX_REFERENCE_SECONDS);
    let (wav, rate) = if use_denoiser {
        (denoise(models, wav, rate, threads)?, DENOISE_RATE)
    } else {
        (wav.to_vec(), rate)
    };
    Ok(Voice {
        speaker_emb: speaker_embedding(models, &wav, rate, threads)?,
        ref_codes: reference_codes(models, &wav, rate, threads)?,
    })
}
