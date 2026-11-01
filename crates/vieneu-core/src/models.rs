//! Model files: where they live and how they are fetched. Every file is pinned
//! to a revision and a SHA-256, so a finished install is reproducible.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

const VIENEU: &str =
    "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/resolve/61b85e3d937fbbacb387714180e8182823512523";
const CODEC: &str =
    "https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/resolve/ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae";
const G2P: &str =
    "https://raw.githubusercontent.com/pnnbao97/sea-g2p/e825173f235d08ea19315b2b279fb11153b44cea/python/sea_g2p";

struct ModelFile {
    base: &'static str,
    remote: &'static str,
    local: &'static str,
    size: u64,
    sha256: &'static str,
}

macro_rules! model {
    ($base:expr, $remote:literal, $local:literal, $size:literal, $sha:literal) => {
        ModelFile { base: $base, remote: $remote, local: $local, size: $size, sha256: $sha }
    };
}

const FILES: &[ModelFile] = &[
    model!(VIENEU, "onnx_int8/config.json", "vieneu/config.json", 2152, "a9f8d9c4b4736448ab355d1a98cfe48f5e39aecf2916c37b0806c228612e9a2d"),
    model!(VIENEU, "onnx_int8/tokenizer.json", "vieneu/tokenizer.json", 22320, "6cc6bcbe380b8c37bd9f2514e37c5dfa3e00e122c6e3125dae5c4afe48e39158"),
    model!(VIENEU, "onnx_int8/vieneu_acoustic_cached.onnx", "vieneu/vieneu_acoustic_cached.onnx", 7207223, "f631e3387c788c3d8b9a5ac5df94952af5bc4c4d1049ff8a751e76a246fff2d4"),
    model!(VIENEU, "onnx_int8/vieneu_backbone_shared.data", "vieneu/vieneu_backbone_shared.data", 103891968, "bb683925f7c8d826fadca4f8a0252ae4d5fc5b7837c14f6857e18f4c6666588d"),
    model!(VIENEU, "onnx_int8/vieneu_decode_step.onnx", "vieneu/vieneu_decode_step.onnx", 1062040, "2c5b30bd8ccb751c58d651f44c074df10c4113efd08719adaa8e3dec6a6ce2ca"),
    model!(VIENEU, "onnx_int8/vieneu_prefill.onnx", "vieneu/vieneu_prefill.onnx", 1090823, "c6a80dabf67c820de798f8deb7d4e0f37d81b5d76e33fbe20ab5a67f2d371f4e"),
    model!(VIENEU, "onnx_int8/vieneu_v3_heads.npz", "vieneu/vieneu_v3_heads.npz", 52219622, "fb22484baa424bbb775133a6e5f0d00d6299b2b256fbe3312a864b85b9aed01e"),
    model!(VIENEU, "speaker_encoder.onnx", "vieneu/speaker_encoder.onnx", 28303423, "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"),
    model!(VIENEU, "denoiser.onnx", "vieneu/denoiser.onnx", 42661414, "b7621953291cfe05e695a9c0ff4255aa2f93239fc17c26627e18b7b6b8f72f0b"),
    model!(CODEC, "codec_browser_onnx_meta.json", "codec/codec_browser_onnx_meta.json", 17036, "3e291c883bb7d11ff2fe8e964e3e495519760358859f35c951254c7741592731"),
    model!(CODEC, "moss_audio_tokenizer_decode_shared.data", "codec/moss_audio_tokenizer_decode_shared.data", 44198912, "e69d52e0f4e84ca27850557ee54face46632d3a5a16c89bd246c7c408466dcad"),
    model!(CODEC, "moss_audio_tokenizer_decode_step.onnx", "codec/moss_audio_tokenizer_decode_step.onnx", 351400, "9527c86a29e1837edec1f74db57d5eeaadb3a715af3382703566460afed25855"),
    model!(CODEC, "moss_audio_tokenizer_encode.data", "codec/moss_audio_tokenizer_encode.data", 44507136, "aa751265b2bab2887eac224484546b194875aa7494b607115439b3dc6b228a2c"),
    model!(CODEC, "moss_audio_tokenizer_encode.onnx", "codec/moss_audio_tokenizer_encode.onnx", 815775, "eadea4a645abdcf98714c7aead122ee2ce7da6e080f9f80b977cd1ca8e19473a"),
    model!(G2P, "sea_g2p.bin", "sea_g2p.bin", 62829820, "4346e690d0711ebc5231e7a42c5c88aaf6e40377e894b4617c018fd81c6f4096"),
];

pub fn total_bytes() -> u64 {
    FILES.iter().map(|f| f.size).sum()
}

/// True when every file has its pinned size. Hashes are checked at download.
pub fn is_installed(models: &Path) -> bool {
    FILES.iter().all(|f| fs::metadata(models.join(f.local)).is_ok_and(|m| m.len() == f.size))
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn fetch(file: &ModelFile, part: &Path, done: u64, progress: &mut dyn FnMut(u64)) -> Result<()> {
    let have = fs::metadata(part).map_or(0, |m| m.len());
    let url = format!("{}/{}", file.base, file.remote);
    let response = ureq::get(&url)
        .set("Range", &format!("bytes={have}-"))
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let resumed = response.status() == 206;
    let mut out = OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(part)?;
    let mut written = if resumed { have } else { 0 };
    let mut reader = response.into_reader();
    let mut buffer = vec![0; 1 << 16];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        out.write_all(&buffer[..read])?;
        written += read as u64;
        progress(done + written);
    }
}

/// Download whatever is missing, resuming partial files. Running it again
/// after a crash, or when complete, is safe. `progress` receives total bytes.
pub fn install(models: &Path, progress: &mut dyn FnMut(u64)) -> Result<()> {
    let mut done = 0;
    for file in FILES {
        let path = models.join(file.local);
        if fs::metadata(&path).is_ok_and(|m| m.len() == file.size) {
            done += file.size;
            progress(done);
            continue;
        }
        fs::create_dir_all(path.parent().context("model path has a parent")?)?;
        let part = path.with_extension("part");
        if fs::metadata(&part).is_ok_and(|m| m.len() >= file.size) {
            fs::remove_file(&part)?;
        }
        fetch(file, &part, done, progress)?;
        let actual = sha256(&part)?;
        if actual != file.sha256 {
            fs::remove_file(&part)?;
            bail!("{} failed verification: got {actual}", file.local);
        }
        fs::rename(&part, &path)?;
        done += file.size;
    }
    Ok(())
}

/// `<data dir>/vieneu-tts-rs/models`, overridable with `VIENEU_MODELS`.
pub fn default_models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("VIENEU_MODELS") {
        return PathBuf::from(dir);
    }
    app_data_dir().join("models")
}

/// Per-user application data directory.
pub fn app_data_dir() -> PathBuf {
    let var = |name| std::env::var_os(name).map(PathBuf::from);
    let home = || var("HOME").unwrap_or_default();
    let base = if cfg!(target_os = "macos") {
        home().join("Library/Application Support")
    } else if cfg!(windows) {
        var("APPDATA").unwrap_or_default()
    } else {
        var("XDG_DATA_HOME").unwrap_or_else(|| home().join(".local/share"))
    };
    base.join("vieneu-tts-rs")
}
