//! WAV output and waveform peaks. Audio goes straight to disk so a long take
//! never sits in memory.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::engine::SAMPLE_RATE;

const SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: SAMPLE_RATE,
    bits_per_sample: 16,
    sample_format: SampleFormat::Int,
};

/// Writes to a sibling `.part` file, renamed into place by [`Self::finish`].
pub struct WavFile {
    writer: WavWriter<BufWriter<File>>,
    part: PathBuf,
    path: PathBuf,
    frames: u64,
}

impl WavFile {
    pub fn create(path: &Path) -> Result<Self> {
        let part = path.with_extension("wav.part");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let writer = WavWriter::create(&part, SPEC)
            .with_context(|| format!("creating {}", part.display()))?;
        Ok(Self { writer, part, path: path.to_path_buf(), frames: 0 })
    }

    pub fn write(&mut self, audio: &[f32]) -> Result<()> {
        for sample in audio {
            self.writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        self.frames += audio.len() as u64;
        Ok(())
    }

    pub fn seconds(&self) -> f32 {
        self.frames as f32 / SAMPLE_RATE as f32
    }

    pub fn finish(self) -> Result<PathBuf> {
        self.writer.finalize()?;
        fs::rename(&self.part, &self.path)?;
        Ok(self.path)
    }

    pub fn abort(self) {
        drop(self.writer);
        // Best effort: a leftover .part is overwritten by the next take.
        let _ = fs::remove_file(&self.part);
    }
}

/// Peak amplitude per bin, `0.0..=1.0`, for drawing a waveform.
pub fn peaks(path: &Path, bins: usize) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let per_bin = (reader.duration() as usize / bins.max(1)).max(1);
    let mut out = Vec::with_capacity(bins);
    let (mut peak, mut seen) = (0u16, 0);
    for sample in reader.samples::<i16>() {
        peak = peak.max(sample?.unsigned_abs());
        seen += 1;
        if seen == per_bin {
            out.push(f32::from(peak) / 32768.0);
            (peak, seen) = (0, 0);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_atomically_and_reports_peaks() {
        let dir = std::env::temp_dir().join(format!("vieneu-wav-{}", std::process::id()));
        let path = dir.join("take.wav");
        let mut wav = WavFile::create(&path).unwrap();
        wav.write(&[0.5; 100]).unwrap();
        wav.write(&[-1.0; 100]).unwrap();
        assert!(!path.exists());
        assert_eq!(wav.finish().unwrap(), path);
        let bins = peaks(&path, 2).unwrap();
        assert_eq!(bins.len(), 2);
        assert!((bins[0] - 0.5).abs() < 0.01 && bins[1] > 0.99);
        fs::remove_dir_all(dir).unwrap();
    }
}
