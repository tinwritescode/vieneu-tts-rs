//! A voice: the speaker embedding plus optional reference codes it conditions on.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Voice {
    pub speaker_emb: Vec<f32>,
    /// `(frames, n_vq)` reference codes.
    #[serde(rename = "codes")]
    pub ref_codes: Vec<Vec<i64>>,
}
