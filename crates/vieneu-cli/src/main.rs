use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use vieneu_core::engine::Flow;
use vieneu_core::library::Library;
use vieneu_core::models;
use vieneu_core::sampling::Sampling;
use vieneu_core::speech::{Speech, SpeechEvent, Synthesizer};
use vieneu_core::wav::WavFile;
use vieneu_core::{enroll, import};

#[derive(Parser)]
#[command(name = "vieneu", about = "On-device Vietnamese/English text-to-speech")]
struct Cli {
    /// ONNX threads; 0 picks half the cores, up to 8.
    #[arg(long, global = true, default_value_t = 0)]
    threads: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download the model files.
    Install,
    /// List voices.
    Voices,
    /// Synthesize text or a document to a WAV file.
    Say {
        /// Text to read; omit when using --file.
        text: Option<String>,
        /// Read this document instead (.txt .md .docx .pdf .srt).
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(short, long, default_value = "out.wav")]
        out: PathBuf,
        #[arg(short, long)]
        voice: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        /// Minimum pause after a sentence, in seconds.
        #[arg(long, default_value_t = 0.0)]
        pause: f32,
        /// Sampling seed, for reproducible output.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Clone a voice from a 3-8 second clip.
    Clone {
        name: String,
        clip: PathBuf,
        #[arg(long)]
        no_denoise: bool,
    },
}

fn library() -> Library {
    Library::new(models::app_data_dir().join("voices"))
}

fn require_models(dir: &std::path::Path) -> Result<()> {
    if !models::is_installed(dir) {
        bail!("model files are missing; run `vieneu install`");
    }
    Ok(())
}

struct Say {
    text: String,
    out: PathBuf,
    voice: String,
    speed: f32,
    pause: f32,
    seed: Option<u64>,
    threads: usize,
}

fn say(args: Say) -> Result<()> {
    let dir = models::default_models_dir();
    require_models(&dir)?;
    let voice = library().voice(&args.voice)?;
    let mut synthesizer = Synthesizer::open(&dir, args.threads)?;
    if let Some(seed) = args.seed {
        synthesizer.seed(seed);
    }
    let speech = Speech {
        text: &args.text,
        voice: &voice,
        speed: args.speed,
        pause: args.pause,
        sampling: Sampling::default(),
        live: false,
    };
    let mut wav = WavFile::create(&args.out)?;
    let mut chunks = 0;
    let mut failure = None;
    let started = std::time::Instant::now();
    let flow = synthesizer.speak(&speech, &AtomicBool::new(false), &mut |event| match event {
        SpeechEvent::Planned { chunks: count } => chunks = count,
        SpeechEvent::Chunk { index } => eprintln!("chunk {}/{chunks}", index + 1),
        SpeechEvent::Audio(audio) => {
            if let Err(e) = wav.write(&audio) {
                failure.get_or_insert(e);
            }
        }
    })?;
    if let Some(e) = failure {
        wav.abort();
        return Err(e);
    }
    debug_assert_eq!(flow, Flow::Finished);
    let seconds = wav.seconds();
    let path = wav.finish()?;
    let spent = started.elapsed().as_secs_f32();
    eprintln!("{}: {seconds:.1}s audio in {spent:.1}s", path.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Install => {
            let total = models::total_bytes();
            let dir = models::default_models_dir();
            let mut shown = 0;
            models::install(&dir, &mut |bytes| {
                let percent = bytes * 100 / total;
                if percent != shown {
                    shown = percent;
                    eprintln!("{percent}%");
                }
            })?;
            println!("{}", dir.display());
        }
        Command::Voices => {
            for voice in library().list() {
                println!("{}\t{}", voice.id, voice.description);
            }
        }
        Command::Say { text, file, out, voice, speed, pause, seed } => {
            let text = match (text, file) {
                (Some(text), None) => text,
                (None, Some(file)) => import::import_document(&file)?,
                _ => bail!("give either text or --file"),
            };
            let voice = voice.unwrap_or_else(Library::default_voice);
            say(Say { text, out, voice, speed, pause, seed, threads: cli.threads })?;
        }
        Command::Clone { name, clip, no_denoise } => {
            let dir = models::default_models_dir();
            require_models(&dir)?;
            let threads = if cli.threads == 0 { vieneu_core::engine::default_threads() } else { cli.threads };
            let voice = enroll::enroll(&dir, &clip, !no_denoise, threads)?;
            println!("{}", library().save_clone(&name, voice)?.id);
        }
    }
    Ok(())
}
