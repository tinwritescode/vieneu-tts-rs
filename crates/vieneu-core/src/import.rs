//! Document import: `.txt`, `.md`, `.docx`, `.pdf`, `.srt` to plain text.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{bail, Context, Result};
use regex::Regex;

pub const SUPPORTED_EXTENSIONS: &[&str] = &["txt", "md", "docx", "pdf", "srt"];
/// Larger documents are refused, never truncated: dropped text would silently
/// yield audio for only part of the document.
pub const CHAR_LIMIT: usize = 200_000;

static XML_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<w:t(?: [^>]*)?>(.*?)</w:t>|<w:tab/>|<w:br/>|(</w:p>)").unwrap()
});
static CUE_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]*>|\{\\[^}]*\}").unwrap());
static SOUND_CUE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[[^\]]*\]|\([^)]*\)").unwrap());

fn unescape_xml(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn docx_text(xml: &str) -> String {
    let mut out = String::new();
    for caps in XML_RUN.captures_iter(xml) {
        if let Some(run) = caps.get(1) {
            out.push_str(&unescape_xml(run.as_str()));
        } else if caps.get(2).is_some() {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    out.trim_end().to_string()
}

fn read_docx(path: &Path) -> Result<String> {
    let file = fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file).context("not a valid .docx file")?;
    let mut xml = String::new();
    archive.by_name("word/document.xml").context("not a valid .docx file")?.read_to_string(&mut xml)?;
    Ok(docx_text(&xml))
}

fn read_utf8(bytes: &[u8]) -> Result<String> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    Ok(std::str::from_utf8(bytes).context("the file is not UTF-8 text")?.to_string())
}

fn srt_text(source: &str) -> Result<String> {
    if !source.contains("-->") {
        bail!("no subtitle timecodes found: this is not an .srt file");
    }
    let mut lines = Vec::new();
    for block in source.replace("\r\n", "\n").split("\n\n") {
        let Some(arrow) = block.lines().position(|line| line.contains("-->")) else { continue };
        let body = block.lines().skip(arrow + 1).collect::<Vec<_>>().join(" ");
        let body = CUE_TAG.replace_all(&body, "");
        let spoken = SOUND_CUE.replace_all(&body, " ");
        if spoken.chars().any(char::is_alphanumeric) {
            lines.push(body.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }
    Ok(lines.join("\n"))
}

pub fn import_document(path: &Path) -> Result<String> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let text = match extension.as_str() {
        "txt" | "md" => read_utf8(&fs::read(path)?)?,
        "srt" => srt_text(&read_utf8(&fs::read(path)?)?)?,
        "docx" => read_docx(path)?,
        "pdf" => pdf_extract::extract_text(path).context("could not read this PDF")?,
        other => bail!("unsupported file type: .{other}"),
    };
    let chars = text.chars().count();
    if chars > CHAR_LIMIT {
        bail!("the document has {chars} characters; the limit is {CHAR_LIMIT}");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docx_keeps_paragraphs_and_entities() {
        let xml = r#"<w:p><w:r><w:t>Xin </w:t></w:r><w:r><w:t xml:space="preserve">chào &amp; tạm biệt</w:t></w:r></w:p><w:p><w:r><w:t>Dòng hai</w:t></w:r></w:p>"#;
        assert_eq!(docx_text(xml), "Xin chào & tạm biệt\nDòng hai");
    }

    #[test]
    fn srt_keeps_speech_and_drops_sound_cues() {
        let srt = "1\r\n00:00:01,000 --> 00:00:02,000\r\n<i>Xin chào</i>\r\nbạn\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\n[MUSIC]\r\n\r\n3\r\n00:00:05,000 --> 00:00:06,000\r\n♪\r\n\r\n4\r\n00:00:07,000 --> 00:00:08,000\r\nTạm biệt\r\n";
        assert_eq!(srt_text(srt).unwrap(), "Xin chào bạn\nTạm biệt");
    }

    #[test]
    fn prose_is_refused_as_srt() {
        assert!(srt_text("một\n\nhai").is_err());
    }

    #[test]
    fn utf8_bom_is_dropped() {
        assert_eq!(read_utf8(b"\xef\xbb\xbfabc").unwrap(), "abc");
        assert!(read_utf8(&[0xff, 0xfe]).is_err());
    }
}
