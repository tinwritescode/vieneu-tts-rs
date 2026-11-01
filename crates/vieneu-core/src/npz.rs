//! Minimal `.npz` reader for little-endian float arrays.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let at = header.find(key).with_context(|| format!("npy header has no {key}"))?;
    Ok(header[at + key.len()..].trim_start_matches([':', ' ', '\'']))
}

fn parse_npy(bytes: &[u8]) -> Result<Array> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let (start, len) = if bytes[6] == 1 {
        (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else {
        (12, u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize)
    };
    let header = std::str::from_utf8(&bytes[start..start + len])?;
    let shape_text = header_value(header, "'shape'")?;
    let shape_text = &shape_text[1..shape_text.find(')').context("bad npy shape")?];
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().context("bad npy dimension"))
        .collect::<Result<Vec<_>>>()?;
    let body = &bytes[start + len..];
    let descr = header_value(header, "'descr'")?;
    let data = if descr.starts_with("<f4") {
        body.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    } else if descr.starts_with("<f8") {
        body.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b) as f32).collect()
    } else {
        bail!("unsupported npy dtype: {descr}");
    };
    Ok(Array { shape, data })
}

pub fn load(path: &Path) -> Result<HashMap<String, Array>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut out = HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;
        out.insert(name, parse_npy(&bytes)?);
    }
    Ok(out)
}
