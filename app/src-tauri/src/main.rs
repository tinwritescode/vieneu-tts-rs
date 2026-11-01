#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod player;
mod worker;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use vieneu_core::engine::default_threads;
use vieneu_core::library::{Library, VoiceInfo};
use vieneu_core::{import, models};

use player::{Player, PlayerStatus};
use worker::{Job, Paths, SpeakJob, Worker};

const THREADS_LIMIT: usize = 32;

struct Shared {
    jobs: Sender<Job>,
    cancel: Arc<AtomicBool>,
    installing: AtomicBool,
    paths: Arc<Paths>,
}

type Reply<T> = Result<T, String>;

fn text_error(error: impl std::fmt::Display) -> String {
    format!("{error:#}")
}

impl Shared {
    fn submit(&self, job: Job) -> Reply<()> {
        self.cancel.store(false, Ordering::Relaxed);
        self.jobs.send(job).map_err(text_error)
    }
}

/// Threads to use: the request when valid, else the default.
fn clamp_threads(threads: usize) -> usize {
    if (1..=THREADS_LIMIT).contains(&threads) {
        threads
    } else {
        default_threads()
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatus {
    installed: bool,
    total_bytes: u64,
    default_threads: usize,
    default_voice: String,
    version: &'static str,
}

#[tauri::command]
fn model_status(shared: State<'_, Shared>) -> ModelStatus {
    ModelStatus {
        installed: models::is_installed(&shared.paths.models),
        total_bytes: models::total_bytes(),
        default_threads: default_threads(),
        default_voice: Library::default_voice(),
        version: env!("CARGO_PKG_VERSION"),
    }
}

#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum InstallEvent {
    Progress { bytes: u64 },
    Finished,
    Failed { message: String },
}

#[tauri::command]
fn install_models(app: AppHandle, shared: State<'_, Shared>) {
    if shared.installing.swap(true, Ordering::SeqCst) {
        return;
    }
    let dir = shared.paths.models.clone();
    std::thread::spawn(move || {
        let emit = |event: InstallEvent| {
            let _ = app.emit("install", event);
        };
        let mut reported = 0;
        let result = models::install(&dir, &mut |bytes| {
            if bytes.abs_diff(reported) >= 1 << 20 {
                reported = bytes;
                emit(InstallEvent::Progress { bytes });
            }
        });
        emit(match result {
            Ok(()) => InstallEvent::Finished,
            Err(error) => InstallEvent::Failed { message: format!("{error:#}") },
        });
        app.state::<Shared>().installing.store(false, Ordering::SeqCst);
    });
}

#[tauri::command(async)]
fn list_voices(shared: State<'_, Shared>) -> Vec<VoiceInfo> {
    Library::new(shared.paths.voices.clone()).list()
}

#[derive(Deserialize)]
struct SpeakArgs {
    text: String,
    voice: String,
    speed: f32,
    pause: f32,
    live: bool,
    threads: usize,
}

#[tauri::command]
fn speak(shared: State<'_, Shared>, args: SpeakArgs) -> Reply<()> {
    if !(0.5..=2.0).contains(&args.speed) || !(0.0..=2.0).contains(&args.pause) {
        return Err("speed or pause is out of range".into());
    }
    let SpeakArgs { text, voice, speed, pause, live, threads } = args;
    let threads = clamp_threads(threads);
    shared.submit(Job::Speak(SpeakJob { text, voice, speed, pause, live, threads }))
}

#[tauri::command]
fn cancel_job(shared: State<'_, Shared>) {
    shared.cancel.store(true, Ordering::Relaxed);
}

#[tauri::command]
fn audition(shared: State<'_, Shared>, voice: String, threads: usize) -> Reply<()> {
    shared.submit(Job::Audition { voice, threads: clamp_threads(threads) })
}

#[tauri::command]
fn enroll_voice(shared: State<'_, Shared>, name: String, clip: PathBuf, denoise: bool, threads: usize) -> Reply<()> {
    shared.submit(Job::Enroll { name, clip, denoise, threads: clamp_threads(threads) })
}

#[tauri::command]
fn remove_voice(shared: State<'_, Shared>, id: String) -> Reply<()> {
    Library::new(shared.paths.voices.clone()).remove_clone(&id).map_err(text_error)?;
    // Best effort: the sample may never have been made.
    let _ = fs::remove_file(shared.paths.audition_clip(&id));
    Ok(())
}

#[tauri::command(async)]
fn import_document(path: PathBuf) -> Reply<String> {
    import::import_document(&path).map_err(text_error)
}

#[tauri::command(async)]
fn export_take(shared: State<'_, Shared>, destination: PathBuf) -> Reply<()> {
    fs::copy(&shared.paths.take, destination).map(|_| ()).map_err(text_error)
}

#[tauri::command]
fn player_play(shared: State<'_, Shared>, player: State<'_, Player>, from: f32) -> Reply<()> {
    player.play_file(&shared.paths.take, from).map_err(text_error)
}

#[tauri::command]
fn player_pause(player: State<'_, Player>, paused: bool) {
    player.set_paused(paused);
}

#[tauri::command]
fn player_stop(player: State<'_, Player>) {
    player.stop();
}

#[tauri::command]
fn player_status(player: State<'_, Player>) -> PlayerStatus {
    player.status()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data = models::app_data_dir();
            let cache = app.path().app_cache_dir()?;
            fs::create_dir_all(&cache)?;
            let paths = Arc::new(Paths {
                models: models::default_models_dir(),
                voices: data.join("voices"),
                take: cache.join("take.wav"),
                audition: cache.join("audition"),
            });
            let cancel = Arc::new(AtomicBool::new(false));
            let (jobs, queue) = channel();
            app.manage(Player::new());
            app.manage(Shared {
                jobs,
                cancel: cancel.clone(),
                installing: AtomicBool::new(false),
                paths: paths.clone(),
            });
            let worker = Worker::new(app.handle().clone(), paths, cancel);
            std::thread::spawn(move || worker.run(queue));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            model_status,
            install_models,
            list_voices,
            speak,
            cancel_job,
            audition,
            enroll_voice,
            remove_voice,
            import_document,
            export_take,
            player_play,
            player_pause,
            player_stop,
            player_status,
        ])
        .run(tauri::generate_context!())
        .expect("the app window could not start");
}
