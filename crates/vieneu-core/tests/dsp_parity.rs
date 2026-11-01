mod common;
use common::{fixture, models};
use vieneu_core::dsp::{fbank, resample, wsola};
use vieneu_core::enroll::{self, read_mono};

#[test]
fn fbank_matches_kaldi() {
    let fx = fixture("engine.json");
    let tone: Vec<f32> = (0..16000)
        .map(|i| {
            let t = i as f64 / 16000.0;
            (0.3 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() as f32)
                + (0.1 * (2.0 * std::f64::consts::PI * 1800.0 * t).sin() as f32)
        })
        .collect();
    let features = fbank::speaker_fbank(&tone).unwrap();
    assert_eq!(features.len() / fbank::N_MELS, fx["fbank16_shape"][0].as_u64().unwrap() as usize);
    let want: Vec<f32> = serde_json::from_value(fx["fbank16_row"].clone()).unwrap();
    let row = &features[7 * fbank::N_MELS..8 * fbank::N_MELS];
    for (got, want) in row.iter().zip(&want) {
        assert!((got - want).abs() < 1e-2, "{got} {want}");
    }
}

#[test]
fn wsola_matches_python() {
    let sr = 48000.0;
    let sig: Vec<f32> = (0..28800)
        .map(|i| {
            let t = i as f64 / sr;
            let tau = 2.0 * std::f64::consts::PI;
            (0.4 * (tau * 180.0 * t).sin() + 0.2 * (tau * 410.0 * t).sin()) as f32
        })
        .collect();
    let fx = fixture("wsola.json");
    for rate in ["0.75", "1.3"] {
        let y = wsola::time_stretch(&sig, rate.parse().unwrap(), 48000);
        assert_eq!(y.len() as u64, fx[rate]["len"].as_u64().unwrap());
        let sum: f64 = y.iter().map(|s| s.abs() as f64).sum();
        let want = fx[rate]["sum_abs"].as_f64().unwrap();
        assert!((sum - want).abs() / want < 1e-3, "{sum} {want}");
    }
}

#[test]
fn resampling_keeps_length_and_tone() {
    let tone: Vec<f32> = (0..24000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();
    let out = resample::resample(&tone, 24000, 16000).unwrap();
    assert_eq!(out.len(), 16000);
    let want = |i: usize| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
    let worst = (2000..14000).map(|i| (out[i] - want(i)).abs()).fold(0.0, f32::max);
    assert!(worst < 0.01, "{worst}");
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "needs the downloaded models"]
fn cloning_matches_python() {
    let fx = fixture("engine.json");
    let clip = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ref.wav");
    let (wav, sr) = read_mono(clip.as_ref()).unwrap();
    assert_eq!(sr as u64, fx["ref_wav_sr"].as_u64().unwrap());

    let want: Vec<f32> = serde_json::from_value(fx["speaker_emb"].clone()).unwrap();
    let got = enroll::speaker_embedding(&models(), &wav, sr, 2).unwrap();
    let similarity = cosine(&got, &want);
    assert!(similarity > 0.995, "{similarity}");

    let clean = enroll::denoise(&models(), &wav, sr, 2).unwrap();
    assert_eq!(clean.len() as u64, fx["denoise_len"].as_u64().unwrap());
    let rms = (clean.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / clean.len() as f64).sqrt();
    let want_rms = fx["denoise_rms"].as_f64().unwrap();
    assert!((rms - want_rms).abs() / want_rms < 0.03, "{rms} {want_rms}");

    let codes = enroll::reference_codes(&models(), &wav, sr, 2).unwrap();
    assert_eq!(codes.len() as u64, fx["ref_codes_shape"][0].as_u64().unwrap());
    assert_eq!(codes[0].len() as u64, fx["ref_codes_shape"][1].as_u64().unwrap());
}

#[test]
fn trim_silence_drops_quiet_edges() {
    let mut wav = vec![0.0f32; 24000];
    wav.extend((0..24000).map(|i| 0.5 * (i as f32 * 0.05).sin()));
    wav.extend(vec![0.0f32; 24000]);
    let trimmed = enroll::trim_silence(&wav, 30.0);
    assert!(trimmed.len() < 30000 && trimmed.len() >= 24000, "{}", trimmed.len());
}
