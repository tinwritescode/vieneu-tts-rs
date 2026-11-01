//! Audio output. Takes play from a file so seeking works; live synthesis
//! appends blocks to a queue. Without an output device the app still
//! synthesizes and exports, and `available` says so.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::{mpsc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use rodio::buffer::SamplesBuffer;
use rodio::source::Source;
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use serde::Serialize;
use vieneu_core::engine::SAMPLE_RATE;

struct Playing {
    sink: Sink,
    /// File position where this sink started; `None` for a live queue.
    start: Option<f32>,
}

#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    Idle,
    Playing,
    Paused,
}

#[derive(Serialize)]
pub struct PlayerStatus {
    available: bool,
    transport: Transport,
    /// Seconds into the take; `None` while playing a live queue.
    position: Option<f32>,
}

pub struct Player {
    output: Option<OutputStreamHandle>,
    playing: Mutex<Option<Playing>>,
}

fn open_output() -> Option<OutputStreamHandle> {
    let (sender, receiver) = mpsc::channel();
    // `OutputStream` is not Send: a parked thread keeps it alive for the app's life.
    std::thread::spawn(move || match OutputStream::try_default() {
        Ok((stream, handle)) => {
            let _ = sender.send(Some(handle));
            let _stream = stream;
            loop {
                std::thread::park();
            }
        }
        Err(_) => {
            let _ = sender.send(None);
        }
    });
    receiver.recv().ok().flatten()
}

impl Player {
    pub fn new() -> Self {
        Self { output: open_output(), playing: Mutex::new(None) }
    }

    fn sink(&self) -> Result<Sink> {
        let output = self.output.as_ref().context("no audio output device")?;
        Ok(Sink::try_new(output)?)
    }

    pub fn play_file(&self, path: &Path, from: f32) -> Result<()> {
        let sink = self.sink()?;
        let decoder = Decoder::new_wav(BufReader::new(File::open(path)?))?;
        sink.append(decoder.skip_duration(Duration::from_secs_f32(from.max(0.0))));
        *self.lock() = Some(Playing { sink, start: Some(from.max(0.0)) });
        Ok(())
    }

    /// Queue live audio, starting a queue when none is running.
    pub fn append(&self, audio: Vec<f32>) {
        let mut playing = self.lock();
        if !playing.as_ref().is_some_and(|p| p.start.is_none()) {
            let Ok(sink) = self.sink() else { return };
            *playing = Some(Playing { sink, start: None });
        }
        if let Some(current) = playing.as_ref() {
            current.sink.append(SamplesBuffer::new(1, SAMPLE_RATE, audio));
        }
    }

    pub fn set_paused(&self, paused: bool) {
        if let Some(current) = self.lock().as_ref() {
            if paused {
                current.sink.pause();
            } else {
                current.sink.play();
            }
        }
    }

    pub fn stop(&self) {
        *self.lock() = None;
    }

    pub fn status(&self) -> PlayerStatus {
        let mut playing = self.lock();
        if playing.as_ref().is_some_and(|p| p.sink.empty()) {
            *playing = None;
        }
        let (transport, position) = match playing.as_ref() {
            None => (Transport::Idle, None),
            Some(p) => (
                if p.sink.is_paused() { Transport::Paused } else { Transport::Playing },
                p.start.map(|start| start + p.sink.get_pos().as_secs_f32()),
            ),
        };
        PlayerStatus { available: self.output.is_some(), transport, position }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Playing>> {
        self.playing.lock().expect("player lock is never held across a panic")
    }
}
