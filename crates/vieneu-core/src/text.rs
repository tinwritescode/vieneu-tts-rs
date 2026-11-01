//! Text planning: sentence splitting and chunk packing. Pure port of
//! `vieneu_utils.core_utils`; lengths are in code points, as in Python.

use std::sync::LazyLock;

use regex::Regex;

static EN_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<en>.*?</en>|\S+").unwrap());
static FRAME_MARKUP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|emotion_\d+\|>|</?en>").unwrap());
pub(crate) static EMOTION_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|emotion_\d+\|>").unwrap());

const OPENERS: &str = "([{“‘«‹「『";
const CLOSERS: &str = ")]}”’»›」』";
const SENTENCE_END: &str = ".!?…";
const TRAILING_CLOSE_EXTRA: &str = "\"'’”";
const MINOR_PUNCT: &str = ",;:-–—";
const CONN_STRIP: &[char] = &['"', '\'', '“', '”', '‘', '’', '(', ')', '[', ']', '«', '»', '…'];
const CONN_WORDS: &[&str] =
    &["và", "nhưng", "hoặc", "song", "rồi", "nên", "vì", "nếu", "khi", "để", "do", "bởi"];
const CONN_PAIRS: &[(&str, &str)] = &[
    ("sau", "khi"), ("trước", "khi"), ("trong", "khi"), ("mỗi", "khi"),
    ("đến", "khi"), ("tới", "khi"), ("cho", "nên"), ("cho", "đến"),
    ("bởi", "vì"), ("nếu", "như"), ("tuy", "nhiên"), ("thế", "nhưng"),
    ("vì", "vậy"), ("vì", "thế"), ("do", "đó"), ("sau", "đó"),
];

const FRAME_CAP_SLACK: usize = 24;
const MAX_FRAMES_PER_PHONE: usize = 2;
const SINGLE_WORD_MAX_FRAMES: usize = 13;
const SINGLE_WORD_MAX_PHONES: usize = 24;

/// The kind of boundary between two consecutive chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    Para,
    Sentence,
    Minor,
}

impl Gap {
    pub fn name(self) -> &'static str {
        match self {
            Gap::Para => "para",
            Gap::Sentence => "sentence",
            Gap::Minor => "minor",
        }
    }

    /// Silence inserted for this boundary, in seconds.
    pub fn silence(self) -> f32 {
        match self {
            Gap::Para => 0.35,
            Gap::Sentence => 0.18,
            Gap::Minor => 0.04,
        }
    }
}

/// Chunks to synthesize, with `gaps.len() == chunks.len() - 1` when non-empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub chunks: Vec<String>,
    pub gaps: Vec<Gap>,
}

pub(crate) fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn scan_sentences(text: &str, quote_aware: bool) -> (Vec<String>, bool) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let is_trailing =
        |c: char| CLOSERS.contains(c) || TRAILING_CLOSE_EXTRA.contains(c);
    let mut sentences = Vec::new();
    let (mut start, mut i, mut depth, mut in_quote) = (0, 0, 0usize, false);
    while i < n {
        let ch = chars[i];
        if quote_aware && ch == '"' {
            in_quote = !in_quote;
        } else if quote_aware && OPENERS.contains(ch) {
            depth += 1;
        } else if quote_aware && CLOSERS.contains(ch) {
            depth = depth.saturating_sub(1);
        } else if SENTENCE_END.contains(ch) && depth == 0 && !in_quote {
            let mut j = i + 1;
            while j < n && SENTENCE_END.contains(chars[j]) {
                j += 1;
            }
            while j < n && is_trailing(chars[j]) {
                j += 1;
            }
            if j >= n || chars[j].is_whitespace() {
                sentences.push(chars[start..j].iter().collect::<String>());
                start = j;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    if start < n {
        sentences.push(chars[start..].iter().collect::<String>());
    }
    let cleaned = sentences
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (cleaned, depth == 0 && !in_quote)
}

/// Split into sentences without cutting inside brackets or quotes. Unbalanced
/// text is rescanned ignoring them, so a stray quote cannot swallow the rest.
pub fn split_into_sentences(text: &str) -> Vec<String> {
    let (sentences, balanced) = scan_sentences(text, true);
    if balanced {
        sentences
    } else {
        scan_sentences(text, false).0
    }
}

fn conn_key(token: &str) -> String {
    token.trim_matches(CONN_STRIP).to_lowercase()
}

fn is_conn_pair(a: &str, b: &str) -> bool {
    CONN_PAIRS.contains(&(a, b))
}

fn natural_cut(words: &[&str], start: usize, end: usize, min_left: usize) -> Option<usize> {
    let lens: usize = words[start..end].iter().map(|w| char_len(w)).sum();
    let mut left = (lens + (end - start - 1)) as isize;
    for j in (start + 1..end).rev() {
        left -= char_len(words[j]) as isize + 1;
        if left < min_left as isize {
            return None;
        }
        let (key, prev) = (conn_key(words[j]), conn_key(words[j - 1]));
        if is_conn_pair(&prev, &key) || CONN_WORDS.contains(&prev.as_str()) {
            continue;
        }
        let next = words.get(j + 1).map(|w| conn_key(w)).unwrap_or_default();
        if CONN_WORDS.contains(&key.as_str()) || is_conn_pair(&key, &next) {
            return Some(j);
        }
    }
    None
}

fn split_long_part(part: &str, max_chars: usize) -> Vec<String> {
    let words: Vec<&str> = EN_TOKEN.find_iter(part).map(|m| m.as_str()).collect();
    let min_left = max_chars / 2;
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let (mut end, mut length) = (start, 0);
        while end < words.len() {
            let word = char_len(words[end]);
            let add = if end > start { length + 1 + word } else { word };
            if end > start && add > max_chars {
                break;
            }
            length = add;
            end += 1;
        }
        if end < words.len() {
            if let Some(cut) = natural_cut(&words, start, end, min_left) {
                end = cut;
            } else {
                while end > start + 1
                    && is_conn_pair(&conn_key(words[end - 1]), &conn_key(words[end]))
                {
                    end -= 1;
                }
            }
        }
        pieces.push(words[start..end].join(" "));
        start = end;
    }
    pieces
}

/// `re.split(r'(?<=[,;:\-–—])\s+', s)`.
fn split_minor_punct(sentence: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = sentence.chars().peekable();
    let mut prev = None;
    while let Some(c) = chars.next() {
        let after_punct = prev.is_some_and(|p| MINOR_PUNCT.contains(p));
        if c.is_whitespace() && after_punct {
            while chars.peek().is_some_and(|n| n.is_whitespace()) {
                chars.next();
            }
            parts.push(std::mem::take(&mut current));
            prev = None;
            continue;
        }
        current.push(c);
        prev = Some(c);
    }
    parts.push(current);
    parts
}

fn pack_long_sentence(
    sentence: &str,
    max_chars: usize,
    buffer: &mut String,
    out: &mut Vec<String>,
) {
    if !buffer.is_empty() {
        out.push(std::mem::take(buffer));
    }
    for part in split_minor_punct(sentence) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if char_len(buffer) + 1 + char_len(part) <= max_chars {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(part);
            continue;
        }
        if !buffer.is_empty() {
            out.push(std::mem::take(buffer));
        }
        *buffer = part.to_string();
        if char_len(buffer) > max_chars {
            let mut pieces = split_long_part(buffer, max_chars);
            *buffer = pieces.pop().unwrap_or_default();
            out.extend(pieces);
        }
    }
}

/// Greedily pack sentences into chunks of at most `max_chars`, in order.
pub fn pack_sentences_into_chunks(sentences: &[String], max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buffer = String::new();
    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        if char_len(sentence) > max_chars {
            pack_long_sentence(sentence, max_chars, &mut buffer, &mut out);
        } else if !buffer.is_empty() && char_len(&buffer) + 1 + char_len(sentence) > max_chars {
            out.push(std::mem::replace(&mut buffer, sentence.to_string()));
        } else {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(sentence);
        }
    }
    if !buffer.is_empty() {
        out.push(buffer);
    }
    out.iter().map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect()
}

/// Classify the boundary right after `chunk` from its final punctuation.
pub(crate) fn classify_gap(chunk: &str) -> Gap {
    match chunk.trim_end().chars().last() {
        Some('.' | '!' | '?') => Gap::Sentence,
        _ => Gap::Minor,
    }
}

fn effective_len(chunk: &str) -> usize {
    char_len(EMOTION_TOKEN.replace_all(chunk, "").trim())
}

/// Merge chunks shorter than `min_chars` into a neighbour: isolated one or two
/// word chunks make the model miss its stop token and hallucinate.
pub(crate) fn merge_short_chunks(plan: &mut Plan, min_chars: usize) {
    let Plan { chunks, gaps } = plan;
    while chunks.len() > 1 {
        let shortest = (0..chunks.len())
            .filter(|&i| effective_len(&chunks[i]) < min_chars)
            .min_by_key(|&i| effective_len(&chunks[i]));
        let Some(i) = shortest else { break };
        let key = |gap: usize, neighbour: usize| {
            (gaps[gap] != Gap::Para, -(char_len(&chunks[neighbour]) as isize))
        };
        let merge_right = if i == 0 {
            true
        } else if i == chunks.len() - 1 {
            false
        } else {
            key(i, i + 1) >= key(i - 1, i - 1)
        };
        let (left, gap) = if merge_right { (i, i) } else { (i - 1, i - 1) };
        let right = chunks.remove(left + 1);
        chunks[left].push(' ');
        chunks[left].push_str(&right);
        gaps.remove(gap);
    }
}

/// Upper bound on plausible frames for one chunk of `phonemes`.
pub fn max_expected_frames(phonemes: &str) -> usize {
    let stripped = FRAME_MARKUP.replace_all(phonemes, "");
    let len = char_len(&stripped);
    let cap = FRAME_CAP_SLACK + MAX_FRAMES_PER_PHONE * len;
    let single_word = stripped.split_whitespace().count() <= 1
        && len <= SINGLE_WORD_MAX_PHONES
        && !phonemes.contains("<|emotion_");
    if single_word {
        cap.min(SINGLE_WORD_MAX_FRAMES)
    } else {
        cap
    }
}
