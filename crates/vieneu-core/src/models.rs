//! Where model files live on disk.

use std::path::PathBuf;

/// `<data dir>/vieneu-tts-rs/models`, overridable with `VIENEU_MODELS`.
pub fn default_models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("VIENEU_MODELS") {
        return PathBuf::from(dir);
    }
    data_dir().join("vieneu-tts-rs").join("models")
}

fn data_dir() -> PathBuf {
    let var = |name| std::env::var_os(name).map(PathBuf::from);
    let home = || var("HOME").unwrap_or_default();
    if cfg!(target_os = "macos") {
        home().join("Library/Application Support")
    } else if cfg!(windows) {
        var("APPDATA").unwrap_or_default()
    } else {
        var("XDG_DATA_HOME").unwrap_or_else(|| home().join(".local/share"))
    }
}
