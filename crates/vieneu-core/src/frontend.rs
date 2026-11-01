//! Text front end: normalization and G2P through `sea-g2p`, plus the inline
//! emotion cues of the v3 Turbo checkpoint. Port of `vieneu_utils.phonemize_text`.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use regex::Regex;
use sea_g2p_rs::g2p::G2PEngine;
use sea_g2p_rs::lang::vi::Normalizer;
use sea_g2p_rs::punc::apply_punc_norm;

use crate::text::{
    classify_gap, merge_short_chunks, pack_sentences_into_chunks, split_into_sentences, Gap, Plan,
};

const MIN_CHUNK_CHARS: usize = 20;
const ATTACHING_PUNCT: &str = ".,!?;:…)]}\"'’”";

static CUE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[[^\]]+\]|<\|emotion_\d+\|>").unwrap());

/// A piece of text: either plain, or a bracketed span / emotion token.
enum Part<'a> {
    Text(&'a str),
    Cue(&'a str),
}

fn split_cues(text: &str) -> Vec<Part<'_>> {
    let mut parts = Vec::new();
    let mut last = 0;
    for m in CUE.find_iter(text) {
        parts.push(Part::Text(&text[last..m.start()]));
        parts.push(Part::Cue(m.as_str()));
        last = m.end();
    }
    parts.push(Part::Text(&text[last..]));
    parts
}

fn emotion_token(tag: &str) -> Option<String> {
    let tag = tag.trim();
    if tag.starts_with("<|") {
        return Some(tag.to_string());
    }
    let inner = tag[1..tag.len() - 1].trim().to_lowercase();
    let k = match inner.as_str() {
        "chuckle" | "cười" | "cuoi" => 1,
        "sigh" | "thở dài" | "tho dai" => 2,
        "clear throat" | "hắng giọng" | "hang giong" => 3,
        _ => return None,
    };
    Some(format!("<|emotion_{k}|>"))
}

fn has_cues(text: &str) -> bool {
    text.contains('[') || text.contains("<|emotion_")
}

pub struct Frontend {
    normalizer: Normalizer,
    g2p: G2PEngine,
}

impl Frontend {
    pub fn open(dict: &Path) -> Result<Self> {
        let path = dict.to_str().context("dictionary path is not UTF-8")?;
        let g2p = G2PEngine::new(path)
            .with_context(|| format!("opening dictionary {}", dict.display()))?;
        Ok(Self { normalizer: Normalizer::new("vi", Some(path)), g2p })
    }

    fn run(&self, text: &str, punc_norm: bool) -> String {
        if text.is_empty() {
            return String::new();
        }
        self.g2p.phonemize(&self.normalizer.normalize(text, punc_norm))
    }

    /// Phonemize, keeping inline cues as `<|emotion_k|>` tokens with the
    /// spacing of the training data.
    pub fn phonemize(&self, text: &str) -> String {
        if !has_cues(text) {
            return self.run(text, true);
        }
        let mut out = String::new();
        for part in split_cues(text) {
            let piece = match part {
                Part::Cue(tag) => match emotion_token(tag) {
                    Some(token) => {
                        if !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str(&token);
                        continue;
                    }
                    None => self.run(tag, false),
                },
                Part::Text(text) if text.trim().is_empty() => continue,
                Part::Text(text) => self.run(text, false),
            };
            let Some(first) = piece.chars().next() else { continue };
            if !out.is_empty() && !ATTACHING_PUNCT.contains(first) {
                out.push(' ');
            }
            out.push_str(&piece);
        }
        apply_punc_norm(&out)
    }

    fn normalize_keep_cues(&self, sentence: &str) -> String {
        if !has_cues(sentence) {
            return self.normalizer.normalize(sentence, false);
        }
        let mut parts = Vec::new();
        for part in split_cues(sentence) {
            match part {
                Part::Cue(tag) => parts.push(emotion_token(tag).unwrap_or_else(|| tag.to_string())),
                Part::Text(text) if !text.trim().is_empty() => {
                    parts.push(self.normalizer.normalize(text, false));
                }
                Part::Text(_) => {}
            }
        }
        parts.retain(|p| !p.is_empty());
        parts.join(" ")
    }

    /// Split raw text into normalized chunks of about `max_chars`, with the
    /// boundary kind between them.
    pub fn plan(&self, text: &str, max_chars: usize) -> Plan {
        let mut plan = Plan::default();
        let paragraphs = text.split(['\r', '\n']).filter(|p| !p.trim().is_empty());
        for paragraph in paragraphs {
            let sentences: Vec<String> = split_into_sentences(paragraph)
                .iter()
                .map(|s| self.normalize_keep_cues(s))
                .collect();
            let chunks = pack_sentences_into_chunks(&sentences, max_chars);
            if chunks.is_empty() {
                continue;
            }
            if !plan.chunks.is_empty() {
                plan.gaps.push(Gap::Para);
            }
            for (j, chunk) in chunks.into_iter().enumerate() {
                if j > 0 {
                    plan.gaps.push(Gap::Sentence);
                }
                plan.chunks.push(chunk);
            }
        }
        plan.chunks = plan.chunks.iter().map(|c| apply_punc_norm(c)).collect();
        for (i, gap) in plan.gaps.iter_mut().enumerate() {
            if *gap != Gap::Para {
                *gap = classify_gap(&plan.chunks[i]);
            }
        }
        merge_short_chunks(&mut plan, MIN_CHUNK_CHARS);
        plan
    }
}
