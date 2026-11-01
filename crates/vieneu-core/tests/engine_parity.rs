mod common;
use std::sync::atomic::AtomicBool;

use common::{fixture, models};
use vieneu_core::engine::{Engine, Flow, Pacing};
use vieneu_core::sampling::Sampling;
use vieneu_core::voice::Voice;

fn voice(name: &str) -> Voice {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/voices_v3_turbo.json");
    let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    serde_json::from_value(data["presets"][name].clone()).unwrap()
}

#[test]
#[ignore = "needs the downloaded models"]
fn greedy_generation_matches_python() {
    let fx = fixture("engine.json");
    let mut engine = Engine::open(&models(), 4).unwrap();
    let voice = voice(fx["voice"].as_str().unwrap());
    let phonemes = fx["phonemes"].as_str().unwrap();

    let ids: Vec<u64> = engine.encode_phonemes(phonemes).unwrap().iter().map(|i| *i as u64).collect();
    let expected: Vec<u64> = fx["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(ids, expected);

    let anchor = engine.speaker_anchor(&voice.speaker_emb).unwrap();
    let python_anchor: Vec<f32> = serde_json::from_value(fx["anchor"].clone()).unwrap();
    for (got, want) in anchor.iter().zip(&python_anchor) {
        assert!((got - want).abs() < 1e-5);
    }

    let sampling = Sampling { temperature: 0.0, ..Default::default() };
    let cancel = AtomicBool::new(false);
    let mut audio = Vec::new();
    let (flow, frames) = engine
        .synthesize_anchored(phonemes, &python_anchor, &voice, &sampling, Pacing::Whole, &cancel, &mut |pcm| audio.extend(pcm))
        .unwrap();
    assert_eq!(flow, Flow::Finished);
    let want: Vec<Vec<i64>> = serde_json::from_value(fx["greedy_frames"].clone()).unwrap();
    assert_eq!(frames, want);
    assert_eq!(audio.len() as u64, fx["wav_len"].as_u64().unwrap());
    let rms = (audio.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / audio.len() as f64).sqrt();
    assert!((rms - fx["wav_rms"].as_f64().unwrap()).abs() < 1e-4, "{rms}");

    // Streaming decodes the same frames in groups and must yield the same length.
    let mut streamed = Vec::new();
    engine
        .synthesize_anchored(phonemes, &python_anchor, &voice, &sampling, Pacing::Adaptive { max_frames: 25 }, &cancel, &mut |pcm| streamed.extend(pcm))
        .unwrap();
    assert_eq!(streamed.len(), audio.len());
    let drift = streamed.iter().zip(&audio).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(drift < 1e-3, "{drift}");
}
