//! The worker thread owns the synthesizer, so one job runs at a time. Models
//! unload after an idle spell: an open app that is not speaking stays small.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use vieneu_core::engine::{default_threads, Flow};
use vieneu_core::library::{Library, VoiceInfo};
use vieneu_core::sampling::Sampling;
use vieneu_core::speech::{Speech, SpeechEvent, Synthesizer};
use vieneu_core::wav::{peaks, WavFile};
use vieneu_core::enroll;

use crate::player::Player;

const IDLE_UNLOAD: Duration = Duration::from_secs(120);
const WAVEFORM_BINS: usize = 600;
const AUDITION_SEED: u64 = 11;
pub const AUDITION_TEXT: &str = "Xin chào, đây là giọng đọc của tôi. Bạn thấy thế nào?";

pub struct SpeakJob {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub pause: f32,
    pub live: bool,
    pub threads: usize,
}

pub enum Job {
    Speak(SpeakJob),
    Audition { voice: String, threads: usize },
    Enroll { name: String, clip: PathBuf, denoise: bool, threads: usize },
}

#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum JobEvent {
    Loading,
    Planned { chunks: usize },
    Chunk { index: usize },
    Finished { seconds: f32, peaks: Vec<f32> },
    Cancelled,
    Auditioned,
    Enrolled { voice: VoiceInfo },
    Failed { message: String },
}

pub struct Paths {
    pub models: PathBuf,
    pub voices: PathBuf,
    pub take: PathBuf,
    pub audition: PathBuf,
}

impl Paths {
    pub fn audition_clip(&self, voice: &str) -> PathBuf {
        self.audition.join(format!("{}.wav", voice.replace(':', "-")))
    }
}

struct Loaded {
    synthesizer: Synthesizer,
    threads: usize,
}

pub struct Worker {
    app: AppHandle,
    paths: Arc<Paths>,
    cancel: Arc<AtomicBool>,
    loaded: Option<Loaded>,
}

impl Worker {
    pub fn new(app: AppHandle, paths: Arc<Paths>, cancel: Arc<AtomicBool>) -> Self {
        Self { app, paths, cancel, loaded: None }
    }

    pub fn run(mut self, jobs: Receiver<Job>) {
        loop {
            match jobs.recv_timeout(IDLE_UNLOAD) {
                Ok(job) => {
                    if let Err(error) = self.handle(job) {
                        self.emit(JobEvent::Failed { message: format!("{error:#}") });
                    }
                }
                Err(RecvTimeoutError::Timeout) => self.loaded = None,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }

    fn emit(&self, event: JobEvent) {
        // The window is gone only during shutdown, when nobody is listening.
        let _ = self.app.emit("job", event);
    }

    fn player(&self) -> tauri::State<'_, Player> {
        self.app.state::<Player>()
    }

    fn handle(&mut self, job: Job) -> Result<()> {
        match job {
            Job::Speak(job) => self.speak(job),
            Job::Audition { voice, threads } => self.audition(&voice, threads),
            Job::Enroll { name, clip, denoise, threads } => {
                self.loaded = None;
                self.emit(JobEvent::Loading);
                let threads = if threads == 0 { default_threads() } else { threads };
                let voice = enroll::enroll(&self.paths.models, &clip, denoise, threads)?;
                let info = Library::new(self.paths.voices.clone()).save_clone(&name, voice)?;
                // A replaced clone must not replay the old voice's sample.
                let _ = std::fs::remove_file(self.paths.audition_clip(&info.id));
                self.emit(JobEvent::Enrolled { voice: info });
                Ok(())
            }
        }
    }

    fn synthesizer(&mut self, threads: usize) -> Result<&mut Synthesizer> {
        if self.loaded.as_ref().is_none_or(|l| l.threads != threads) {
            self.loaded = None;
            self.emit(JobEvent::Loading);
            let synthesizer = Synthesizer::open(&self.paths.models, threads)?;
            self.loaded = Some(Loaded { synthesizer, threads });
        }
        Ok(&mut self.loaded.as_mut().expect("loaded above").synthesizer)
    }

    /// Render to `out`; the duration in seconds, or `None` when cancelled.
    fn render(&mut self, job: &SpeakJob, out: &std::path::Path) -> Result<Option<f32>> {
        let voice = Library::new(self.paths.voices.clone()).voice(&job.voice)?;
        let (app, cancel) = (self.app.clone(), self.cancel.clone());
        let mut wav = WavFile::create(out)?;
        let speech = Speech {
            text: &job.text,
            voice: &voice,
            speed: job.speed,
            pause: job.pause,
            sampling: Sampling::default(),
            live: job.live,
        };
        let mut failure = None;
        let flow = self.synthesizer(job.threads)?.speak(&speech, &cancel, &mut |event| match event {
            SpeechEvent::Planned { chunks } => {
                let _ = app.emit("job", JobEvent::Planned { chunks });
            }
            SpeechEvent::Chunk { index } => {
                let _ = app.emit("job", JobEvent::Chunk { index });
            }
            SpeechEvent::Audio(audio) => {
                if let Err(error) = wav.write(&audio) {
                    failure.get_or_insert(error);
                }
                if job.live {
                    app.state::<Player>().append(audio);
                }
            }
        });
        match (flow, failure) {
            (Ok(Flow::Finished), None) => {
                let seconds = wav.seconds();
                wav.finish()?;
                Ok(Some(seconds))
            }
            (Ok(Flow::Cancelled), None) => {
                wav.abort();
                Ok(None)
            }
            (Err(error), _) | (_, Some(error)) => {
                wav.abort();
                Err(error)
            }
        }
    }

    fn speak(&mut self, job: SpeakJob) -> Result<()> {
        self.player().stop();
        let take = self.paths.take.clone();
        let Some(seconds) = self.render(&job, &take)? else {
            self.player().stop();
            self.emit(JobEvent::Cancelled);
            return Ok(());
        };
        if !job.live {
            // No output device still leaves a finished take to export.
            let _ = self.player().play_file(&take, 0.0);
        }
        self.emit(JobEvent::Finished { seconds, peaks: peaks(&take, WAVEFORM_BINS)? });
        Ok(())
    }

    fn audition(&mut self, voice: &str, threads: usize) -> Result<()> {
        let clip = self.paths.audition_clip(voice);
        if !clip.exists() {
            let job = SpeakJob {
                text: AUDITION_TEXT.to_string(),
                voice: voice.to_string(),
                speed: 1.0,
                pause: 0.0,
                live: false,
                threads,
            };
            self.synthesizer(threads)?.seed(AUDITION_SEED);
            self.cancel.store(false, Ordering::Relaxed);
            if self.render(&job, &clip)?.is_none() {
                self.emit(JobEvent::Cancelled);
                return Ok(());
            }
        }
        self.player().play_file(&clip, 0.0)?;
        self.emit(JobEvent::Auditioned);
        Ok(())
    }
}
