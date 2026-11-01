//! Text to audio: plans chunks, synthesizes each, and joins them with the
//! silence their boundary calls for.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use anyhow::{bail, Result};

use crate::dsp::wsola::time_stretch;
use crate::engine::{Engine, Flow, Pacing, SAMPLE_RATE};
use crate::frontend::Frontend;
use crate::sampling::Sampling;
use crate::text::Gap;
use crate::voice::Voice;

pub const MAX_CHUNK_CHARS: usize = 256;
const STREAM_GROUP_FRAMES: usize = 12;
const NEUTRAL_SPEED: f32 = 1e-3;

pub struct Speech<'a> {
    pub text: &'a str,
    pub voice: &'a Voice,
    /// Playback rate, 1.0 is natural; pitch is preserved.
    pub speed: f32,
    /// Minimum silence after a sentence or paragraph, in seconds.
    pub pause: f32,
    pub sampling: Sampling,
    /// Emit audio as frames decode rather than per chunk.
    pub live: bool,
}

pub enum SpeechEvent {
    Planned { chunks: usize },
    Chunk { index: usize },
    Audio(Vec<f32>),
}

fn gap_seconds(gap: Gap, pause: f32) -> f32 {
    match gap {
        Gap::Minor => gap.silence(),
        Gap::Sentence | Gap::Para => gap.silence().max(pause),
    }
}

fn silence(seconds: f32) -> Vec<f32> {
    vec![0.0; (seconds * SAMPLE_RATE as f32) as usize]
}

pub struct Synthesizer {
    frontend: Frontend,
    engine: Engine,
}

impl Synthesizer {
    pub fn open(models: &Path, threads: usize) -> Result<Self> {
        Ok(Self {
            frontend: Frontend::open(&models.join("sea_g2p.bin"))?,
            engine: Engine::open(models, threads)?,
        })
    }

    pub fn seed(&mut self, seed: u64) {
        self.engine.seed(seed);
    }

    pub fn speak(
        &mut self,
        speech: &Speech<'_>,
        cancel: &AtomicBool,
        sink: &mut dyn FnMut(SpeechEvent),
    ) -> Result<Flow> {
        let plan = self.frontend.plan(speech.text, MAX_CHUNK_CHARS);
        if plan.chunks.is_empty() {
            bail!("the text has nothing to read");
        }
        sink(SpeechEvent::Planned { chunks: plan.chunks.len() });
        let stretch = (speech.speed - 1.0).abs() >= NEUTRAL_SPEED;
        let pacing = if speech.live && !stretch {
            Pacing::Adaptive { max_frames: STREAM_GROUP_FRAMES }
        } else {
            Pacing::Whole
        };
        for (index, chunk) in plan.chunks.iter().enumerate() {
            sink(SpeechEvent::Chunk { index });
            let phonemes = self.frontend.phonemize(chunk);
            let mut audio = Vec::new();
            let (flow, _) = self.engine.synthesize(
                &phonemes,
                speech.voice,
                &speech.sampling,
                pacing,
                cancel,
                &mut |block| {
                    if matches!(pacing, Pacing::Whole) {
                        audio.extend(block);
                    } else {
                        sink(SpeechEvent::Audio(block));
                    }
                },
            )?;
            if flow == Flow::Cancelled {
                return Ok(flow);
            }
            if !audio.is_empty() {
                sink(SpeechEvent::Audio(if stretch {
                    time_stretch(&audio, speech.speed, SAMPLE_RATE)
                } else {
                    audio
                }));
            }
            if let Some(gap) = plan.gaps.get(index) {
                sink(SpeechEvent::Audio(silence(gap_seconds(*gap, speech.pause))));
            }
        }
        Ok(Flow::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_extends_sentence_and_paragraph_gaps_only() {
        assert_eq!(gap_seconds(Gap::Sentence, 0.5), 0.5);
        assert_eq!(gap_seconds(Gap::Para, 0.5), 0.5);
        assert_eq!(gap_seconds(Gap::Para, 0.0), 0.35);
        assert_eq!(gap_seconds(Gap::Minor, 0.5), 0.04);
    }
}
