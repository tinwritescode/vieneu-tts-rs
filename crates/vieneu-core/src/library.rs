//! The voice library: bundled presets plus clones saved as one JSON file each.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::voice::Voice;

const PRESETS: &str = include_str!("../assets/voices_v3_turbo.json");

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VoiceKind {
    Preset,
    Clone,
}

#[derive(Clone, Debug, Serialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub kind: VoiceKind,
    pub gender: String,
    pub region: String,
    pub description: String,
}

#[derive(Deserialize)]
struct Preset {
    #[serde(default)]
    description: String,
    #[serde(default)]
    gender: String,
    #[serde(default)]
    region: String,
    #[serde(flatten)]
    voice: Voice,
}

#[derive(Deserialize)]
struct Presets {
    default_voice: String,
    presets: BTreeMap<String, Preset>,
}

#[derive(Serialize, Deserialize)]
struct SavedClone {
    name: String,
    #[serde(flatten)]
    voice: Voice,
}

const CLONE_PREFIX: &str = "clone:";

fn parse_presets() -> Presets {
    serde_json::from_str(PRESETS).expect("bundled voices are valid")
}

fn slug(name: &str) -> String {
    let hash = name.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
    format!("{hash:016x}")
}

pub struct Library {
    clones: PathBuf,
}

impl Library {
    pub fn new(clones: PathBuf) -> Self {
        Self { clones }
    }

    pub fn default_voice() -> String {
        parse_presets().default_voice
    }

    fn clone_path(&self, id: &str) -> Option<PathBuf> {
        let key = id.strip_prefix(CLONE_PREFIX)?;
        key.bytes().all(|b| b.is_ascii_hexdigit()).then(|| self.clones.join(format!("{key}.json")))
    }

    fn read_clones(&self) -> Vec<(String, SavedClone)> {
        let Ok(entries) = fs::read_dir(&self.clones) else { return Vec::new() };
        entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let key = path.file_stem()?.to_str()?.to_string();
                // A damaged file must not hide the rest of the library.
                let saved = serde_json::from_slice(&fs::read(&path).ok()?).ok()?;
                Some((format!("{CLONE_PREFIX}{key}"), saved))
            })
            .collect()
    }

    pub fn list(&self) -> Vec<VoiceInfo> {
        let presets = parse_presets().presets.into_iter().map(|(name, p)| VoiceInfo {
            id: name.clone(),
            name,
            kind: VoiceKind::Preset,
            gender: p.gender,
            region: p.region,
            description: p.description,
        });
        let clones = self.read_clones().into_iter().map(|(id, saved)| VoiceInfo {
            id,
            name: saved.name,
            kind: VoiceKind::Clone,
            gender: String::new(),
            region: String::new(),
            description: String::new(),
        });
        let mut all: Vec<_> = presets.chain(clones).collect();
        all.sort_by_key(|voice| (voice.kind == VoiceKind::Preset, voice.name.clone()));
        all
    }

    pub fn voice(&self, id: &str) -> Result<Voice> {
        if let Some(path) = self.clone_path(id) {
            let saved: SavedClone = serde_json::from_slice(&fs::read(&path)?)?;
            return Ok(saved.voice);
        }
        let mut presets = parse_presets().presets;
        Ok(presets.remove(id).with_context(|| format!("unknown voice: {id}"))?.voice)
    }

    /// Save a clone; saving the same name again replaces it.
    pub fn save_clone(&self, name: &str, voice: Voice) -> Result<VoiceInfo> {
        let name = name.trim();
        if name.is_empty() {
            bail!("a voice needs a name");
        }
        if parse_presets().presets.contains_key(name) {
            bail!("a preset is already called {name}");
        }
        fs::create_dir_all(&self.clones)?;
        let key = slug(name);
        let saved = SavedClone { name: name.to_string(), voice };
        let (path, part) = (self.clones.join(format!("{key}.json")), self.clones.join(format!("{key}.part")));
        fs::write(&part, serde_json::to_vec(&saved)?)?;
        fs::rename(&part, &path)?;
        Ok(VoiceInfo {
            id: format!("{CLONE_PREFIX}{key}"),
            name: saved.name,
            kind: VoiceKind::Clone,
            gender: String::new(),
            region: String::new(),
            description: String::new(),
        })
    }

    pub fn remove_clone(&self, id: &str) -> Result<()> {
        let path = self.clone_path(id).with_context(|| format!("not a clone: {id}"))?;
        match fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library(tag: &str) -> Library {
        let dir = std::env::temp_dir().join(format!("vieneu-lib-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        Library::new(dir)
    }

    fn sample() -> Voice {
        Voice { speaker_emb: vec![1.0; 192], ref_codes: vec![vec![1; 16]] }
    }

    #[test]
    fn lists_presets_with_the_default_among_them() {
        let voices = library("list").list();
        assert_eq!(voices.len(), 20);
        assert!(voices.iter().any(|v| v.id == Library::default_voice()));
        assert_eq!(library("v").voice("Adam").unwrap().speaker_emb.len(), 192);
    }

    #[test]
    fn clones_round_trip_and_sort_first() {
        let library = library("clone");
        let info = library.save_clone("  Giọng tôi ", sample()).unwrap();
        assert_eq!(info.name, "Giọng tôi");
        assert_eq!(library.list()[0].id, info.id);
        assert_eq!(library.voice(&info.id).unwrap().ref_codes.len(), 1);
        library.remove_clone(&info.id).unwrap();
        library.remove_clone(&info.id).unwrap();
        assert_eq!(library.list().len(), 20);
    }

    #[test]
    fn refuses_bad_names_and_path_tricks() {
        let library = library("bad");
        assert!(library.save_clone("  ", sample()).is_err());
        assert!(library.save_clone("Adam", sample()).is_err());
        assert!(library.remove_clone("clone:../../etc").is_err());
    }
}
