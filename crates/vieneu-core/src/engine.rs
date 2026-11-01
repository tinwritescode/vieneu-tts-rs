//! VieNeu-TTS v3 Turbo inference on ONNX Runtime. Port of the SDK's
//! `OnnxV3LiteEngine`, with `decode_step` as the only codec decoder: it is
//! bit-exact to `decode_full`, so loading both would only cost memory.
//!
//! The engine is single-owner: `ort` sessions need `&mut` to run.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::ep::CPU;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::npz;
use crate::sampling::{argmax, sample, ChannelHistory, Sampling};
use crate::text::max_expected_frames;
use crate::voice::Voice;

pub const SAMPLE_RATE: u32 = 48_000;
const STREAM_LEADIN_FRAMES: usize = 4;

#[derive(Deserialize)]
struct Config {
    n_vq: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    local_num_hidden_layers: usize,
    local_num_attention_heads: usize,
    audio_pad_token_id: i64,
    text_prompt_start_token_id: i64,
    text_prompt_end_token_id: i64,
    speech_generation_start_token_id: i64,
    speech_generation_end_token_id: usize,
    audio_ref_slot_token_id: i64,
    default_style_token_id: i64,
}

/// How generated frames are grouped into decoded audio.
#[derive(Clone, Copy, Debug)]
pub enum Pacing {
    /// Decode once, after the last frame.
    Whole,
    /// Decode small groups, growing as playback lead builds up.
    Adaptive { max_frames: usize },
}

impl Pacing {
    fn target(self, lead_seconds: Option<f32>) -> usize {
        let Pacing::Adaptive { max_frames } = self else { return usize::MAX };
        let cap = max_frames.max(1);
        let wanted = match lead_seconds {
            Some(lead) if lead >= 1.10 => cap,
            Some(lead) if lead >= 0.55 => 8,
            Some(lead) if lead >= 0.20 => 6,
            _ => STREAM_LEADIN_FRAMES,
        };
        wanted.min(cap)
    }
}

/// Whether generation ran to its end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Finished,
    Cancelled,
}

struct Xvec {
    w: Vec<f32>,
    b: Vec<f32>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    ln_eps: f32,
}

struct Heads {
    text_emb: Vec<f32>,
    audio_emb: Vec<f32>,
    audio_vocab: usize,
    xvec: Xvec,
}

impl Heads {
    fn load(path: &Path) -> Result<Self> {
        let mut z = npz::load(path)?;
        let mut take = |name: &str| z.remove(name).with_context(|| format!("heads: no {name}"));
        let text_emb = take("text_emb")?.data;
        let audio = take("audio_emb")?;
        let audio_vocab = *audio.shape.get(1).context("audio_emb has no vocab axis")?;
        let xvec = Xvec {
            w: take("xvec_w")?.data,
            b: take("xvec_b")?.data,
            ln_w: take("xvec_ln_w")?.data,
            ln_b: take("xvec_ln_b")?.data,
            ln_eps: take("xvec_ln_eps")?.data[0],
        };
        Ok(Self { text_emb, audio_emb: audio.data, audio_vocab, xvec })
    }
}

#[derive(Deserialize, Clone)]
struct TransformerOffset {
    input_name: String,
    output_name: String,
    shape: Vec<i64>,
}

#[derive(Deserialize, Clone)]
struct AttentionCache {
    offset_input_name: String,
    offset_output_name: String,
    cached_keys_input_name: String,
    cached_keys_output_name: String,
    cached_values_input_name: String,
    cached_values_output_name: String,
    cached_positions_input_name: String,
    cached_positions_output_name: String,
    offset_shape: Vec<i64>,
    cache_shape: Vec<i64>,
    positions_shape: Vec<i64>,
}

#[derive(Deserialize, Clone)]
struct StreamSpec {
    transformer_offsets: Vec<TransformerOffset>,
    attention_caches: Vec<AttentionCache>,
}

#[derive(Deserialize)]
struct CodecMeta {
    streaming_decode: StreamSpec,
}

fn filled<T>(shape: &[i64], value: T) -> Result<DynValue>
where
    T: Clone + std::fmt::Debug + ort::value::PrimitiveTensorElementType + 'static,
{
    let count: i64 = shape.iter().product();
    let tensor = Tensor::from_array((shape.to_vec(), vec![value; count as usize]))?;
    Ok(tensor.into_dyn())
}

fn dot_into(row: &[f32], matrix: &[f32], out: &mut [f32]) {
    let width = row.len();
    for (logit, weights) in out.iter_mut().zip(matrix.chunks_exact(width)) {
        *logit = row.iter().zip(weights).map(|(a, b)| a * b).sum();
    }
}

// ONNX Runtime allows one environment per process.
static ENVIRONMENT: LazyLock<Result<ort::environment::Environment, String>> =
    LazyLock::new(|| ort::init().with_telemetry(false).build().map_err(|e| e.to_string()));

/// Session options the app uses for every graph.
pub(crate) fn session(path: &Path, threads: usize) -> Result<Session> {
    let environment = ENVIRONMENT.as_ref().map_err(|e| anyhow!("onnx runtime: {e}"))?;
    let open = || -> ort::Result<Session> {
        // On macOS the default policy registers CoreML, which rejects the
        // acoustic graph's empty KV cache.
        Session::builder(environment)?
            .with_auto_device(AutoDevicePolicy::PreferCPU)?
            .with_execution_providers([CPU::default().with_arena_allocator(false).build()])?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_inter_threads(1)?
            .with_intra_threads(threads)?
            .with_intra_op_spinning(false)?
            .commit_from_file(path)
    };
    open().with_context(|| format!("loading {}", path.display()))
}

pub fn default_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(8, usize::from);
    (cores / 2).clamp(1, 8)
}

pub struct Engine {
    cfg: Config,
    heads: Heads,
    tokenizer: Tokenizer,
    prefill: Session,
    decode: Session,
    acoustic: Session,
    codec: Session,
    spec: StreamSpec,
    rng: fastrand::Rng,
    logits: Vec<f32>,
    scratch: Vec<(f32, u32)>,
}

impl Engine {
    /// `models` holds `vieneu/` and `codec/`; `threads == 0` picks a default.
    pub fn open(models: &Path, threads: usize) -> Result<Self> {
        let threads = if threads == 0 { default_threads() } else { threads };
        let (vieneu, codec) = (models.join("vieneu"), models.join("codec"));
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(vieneu.join("config.json"))?)?;
        let heads = Heads::load(&vieneu.join("vieneu_v3_heads.npz"))?;
        let tokenizer = Tokenizer::from_file(vieneu.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer: {e}"))?;
        let meta: CodecMeta = serde_json::from_str(&std::fs::read_to_string(
            codec.join("codec_browser_onnx_meta.json"),
        )?)?;
        let logits = vec![0.0; heads.audio_vocab.max(heads.text_emb.len() / cfg.hidden_size)];
        Ok(Self {
            prefill: session(&vieneu.join("vieneu_prefill.onnx"), threads)?,
            decode: session(&vieneu.join("vieneu_decode_step.onnx"), threads)?,
            acoustic: session(&vieneu.join("vieneu_acoustic_cached.onnx"), threads)?,
            codec: session(&codec.join("moss_audio_tokenizer_decode_step.onnx"), threads)?,
            spec: meta.streaming_decode,
            rng: fastrand::Rng::new(),
            cfg,
            heads,
            tokenizer,
            logits,
            scratch: Vec::new(),
        })
    }

    /// Seed the sampler, for reproducible output.
    pub fn seed(&mut self, seed: u64) {
        self.rng.seed(seed);
    }

    pub fn encode_phonemes(&self, phonemes: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(phonemes, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// 192-d x-vector to the `(hidden,)` anchor added to every embedding.
    pub fn speaker_anchor(&self, speaker_emb: &[f32]) -> Result<Vec<f32>> {
        if speaker_emb.iter().all(|v| *v == 0.0) {
            bail!("speaker embedding is all zero");
        }
        let x = &self.heads.xvec;
        let mut v: Vec<f32> = x
            .w
            .chunks_exact(speaker_emb.len())
            .zip(&x.b)
            .map(|(row, b)| row.iter().zip(speaker_emb).map(|(a, s)| a * s).sum::<f32>() + b)
            .collect();
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let scale = (var + x.ln_eps).sqrt();
        for (value, (w, b)) in v.iter_mut().zip(x.ln_w.iter().zip(&x.ln_b)) {
            *value = (*value - mean) / scale * w + b;
        }
        Ok(v)
    }

    fn audio_row(&self, channel: usize, code: usize) -> &[f32] {
        let h = self.cfg.hidden_size;
        let start = (channel * self.heads.audio_vocab + code) * h;
        &self.heads.audio_emb[start..start + h]
    }

    fn text_row(&self, id: usize) -> &[f32] {
        let h = self.cfg.hidden_size;
        &self.heads.text_emb[id * h..(id + 1) * h]
    }

    fn embed_row(&self, text_id: i64, codes: Option<&[i64]>, anchor: &[f32], out: &mut Vec<f32>) {
        let start = out.len();
        out.extend_from_slice(self.text_row(text_id as usize));
        let row = &mut out[start..];
        for (channel, &code) in codes.unwrap_or(&[]).iter().enumerate() {
            if code == self.cfg.audio_pad_token_id {
                continue;
            }
            let weights = self.audio_row(channel, code as usize);
            row.iter_mut().zip(weights).for_each(|(a, b)| *a += b);
        }
        row.iter_mut().zip(anchor).for_each(|(a, b)| *a += b);
    }

    fn prompt_embeds(&self, phonemes: &str, voice: &Voice, anchor: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let ids = self.encode_phonemes(phonemes)?;
        let mut text = vec![cfg.default_style_token_id, cfg.text_prompt_start_token_id];
        text.extend(ids.iter().map(|id| i64::from(*id)));
        text.push(cfg.text_prompt_end_token_id);
        let mut out = Vec::with_capacity((text.len() + voice.ref_codes.len()) * cfg.hidden_size);
        for id in text {
            self.embed_row(id, None, anchor, &mut out);
        }
        for codes in &voice.ref_codes {
            self.embed_row(cfg.audio_ref_slot_token_id, Some(codes), anchor, &mut out);
        }
        Ok(out)
    }

    fn acoustic_inputs(
        &self,
        token_emb: Vec<f32>,
        positions: Vec<i64>,
        past: Vec<DynValue>,
    ) -> Result<Vec<(String, SessionInputValue<'static>)>> {
        let tokens = token_emb.len() / self.cfg.hidden_size;
        let mut inputs: Vec<(String, SessionInputValue<'static>)> = vec![
            ("token_emb".into(), Tensor::from_array(([1, tokens, self.cfg.hidden_size], token_emb))?.into()),
            ("position_ids".into(), Tensor::from_array(([1, tokens], positions))?.into()),
        ];
        let layers = self.cfg.local_num_hidden_layers;
        for (i, value) in past.into_iter().enumerate() {
            let kind = if i < layers { "k" } else { "v" };
            inputs.push((format!("past_{kind}_{}", i % layers), value.into()));
        }
        Ok(inputs)
    }

    fn empty_past(&self) -> Result<Vec<DynValue>> {
        let cfg = &self.cfg;
        let heads = cfg.local_num_attention_heads as i64;
        let dim = (cfg.hidden_size / cfg.local_num_attention_heads) as i64;
        (0..2 * cfg.local_num_hidden_layers).map(|_| filled(&[1, heads, 0, dim], 0.0f32)).collect()
    }

    fn run_acoustic(
        &mut self,
        inputs: Vec<(String, SessionInputValue<'static>)>,
    ) -> Result<(Vec<f32>, Vec<DynValue>)> {
        let names: Vec<String> = self.acoustic.outputs().iter().map(|o| o.name().to_string()).collect();
        let mut outputs = self.acoustic.run(inputs)?;
        let (_, hidden) = outputs[names[0].as_str()].try_extract_tensor::<f32>()?;
        let hidden = hidden.to_vec();
        let past = names[1..]
            .iter()
            .map(|name| outputs.remove(name.as_str()).context("acoustic output missing"))
            .collect::<Result<Vec<_>>>()?;
        Ok((hidden, past))
    }

    fn sample_channel(
        &mut self,
        channel: usize,
        hidden: &[f32],
        sampling: &Sampling,
        history: &mut [ChannelHistory],
    ) -> usize {
        let h = self.cfg.hidden_size;
        let vocab = self.heads.audio_vocab;
        let start = channel * vocab * h;
        let matrix = &self.heads.audio_emb[start..start + vocab * h];
        let logits = &mut self.logits[..vocab];
        dot_into(hidden, matrix, logits);
        let code = sample(logits, sampling, &history[channel], &mut self.rng, &mut self.scratch);
        history[channel].add(code);
        code
    }

    /// One frame of codes from backbone state `state`, plus the stop decision.
    fn acoustic_frame(
        &mut self,
        state: &[f32],
        sampling: &Sampling,
        history: &mut [ChannelHistory],
    ) -> Result<(Vec<i64>, bool)> {
        let h = self.cfg.hidden_size;
        let start = self.cfg.speech_generation_start_token_id as usize;
        let mut tokens = state.to_vec();
        tokens.extend_from_slice(self.text_row(start));
        let inputs = self.acoustic_inputs(tokens, vec![0, 1], self.empty_past()?)?;
        let (hidden, mut past) = self.run_acoustic(inputs)?;

        let vocab = self.heads.text_emb.len() / h;
        let logits = &mut self.logits[..vocab];
        dot_into(&hidden[..h], &self.heads.text_emb, logits);
        let stop = argmax(logits) == self.cfg.speech_generation_end_token_id;

        let mut code = self.sample_channel(0, &hidden[h..2 * h], sampling, history);
        let mut codes = vec![code as i64];
        for channel in 1..self.cfg.n_vq {
            let emb = self.audio_row(channel - 1, code).to_vec();
            let inputs = self.acoustic_inputs(emb, vec![channel as i64 + 1], past)?;
            let (hidden, next) = self.run_acoustic(inputs)?;
            past = next;
            code = self.sample_channel(channel, &hidden[..h], sampling, history);
            codes.push(code as i64);
        }
        Ok((codes, stop))
    }


    fn new_codec_state(&self) -> Result<HashMap<String, DynValue>> {
        let mut state = HashMap::new();
        for t in &self.spec.transformer_offsets {
            state.insert(t.input_name.clone(), filled(&t.shape, 0i32)?);
        }
        for a in &self.spec.attention_caches {
            state.insert(a.offset_input_name.clone(), filled(&a.offset_shape, 0i32)?);
            state.insert(a.cached_keys_input_name.clone(), filled(&a.cache_shape, 0.0f32)?);
            state.insert(a.cached_values_input_name.clone(), filled(&a.cache_shape, 0.0f32)?);
            state.insert(a.cached_positions_input_name.clone(), filled(&a.positions_shape, -1i32)?);
        }
        Ok(state)
    }

    fn decode_group(
        &mut self,
        frames: &[Vec<i64>],
        state: &mut HashMap<String, DynValue>,
    ) -> Result<Vec<f32>> {
        let codes: Vec<i32> = frames.iter().flatten().map(|c| *c as i32).collect();
        let shape = [1, frames.len(), self.cfg.n_vq];
        let mut inputs: Vec<(String, SessionInputValue<'static>)> = vec![
            ("audio_codes".into(), Tensor::from_array((shape, codes))?.into()),
            ("audio_code_lengths".into(), Tensor::from_array(([1usize], vec![frames.len() as i32]))?.into()),
        ];
        inputs.extend(state.drain().map(|(name, value)| (name, value.into())));
        let mut outputs = self.codec.run(inputs)?;

        let spec = self.spec.clone();
        let mut carry = |input: &str, output: &str| -> Result<()> {
            let value = outputs.remove(output).with_context(|| format!("codec output {output}"))?;
            state.insert(input.to_string(), value);
            Ok(())
        };
        for t in &spec.transformer_offsets {
            carry(&t.input_name, &t.output_name)?;
        }
        for a in &spec.attention_caches {
            carry(&a.offset_input_name, &a.offset_output_name)?;
            carry(&a.cached_keys_input_name, &a.cached_keys_output_name)?;
            carry(&a.cached_values_input_name, &a.cached_values_output_name)?;
            carry(&a.cached_positions_input_name, &a.cached_positions_output_name)?;
        }
        let (_, lengths) = outputs["audio_lengths"].try_extract_tensor::<i32>()?;
        let length = lengths[0] as usize;
        let (shape, audio) = outputs["audio"].try_extract_tensor::<f32>()?;
        let channels = shape[1] as usize;
        let width = shape[2] as usize;
        let mono = (0..length)
            .map(|i| (0..channels).map(|c| audio[c * width + i]).sum::<f32>() / channels as f32)
            .collect();
        Ok(mono)
    }

    /// Synthesize one chunk of phonemes, handing audio to `sink` as it is
    /// decoded. Stops early when `cancel` is set.
    pub fn synthesize(
        &mut self,
        phonemes: &str,
        voice: &Voice,
        sampling: &Sampling,
        pacing: Pacing,
        cancel: &AtomicBool,
        sink: &mut dyn FnMut(Vec<f32>),
    ) -> Result<(Flow, Vec<Vec<i64>>)> {
        let anchor = self.speaker_anchor(&voice.speaker_emb)?;
        self.synthesize_anchored(phonemes, &anchor, voice, sampling, pacing, cancel, sink)
    }

    /// [`Self::synthesize`] with a precomputed anchor. The quantized graphs
    /// amplify 1-ULP input changes, so reproducing another implementation's
    /// frames needs its exact anchor.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_anchored(
        &mut self,
        phonemes: &str,
        anchor: &[f32],
        voice: &Voice,
        sampling: &Sampling,
        pacing: Pacing,
        cancel: &AtomicBool,
        sink: &mut dyn FnMut(Vec<f32>),
    ) -> Result<(Flow, Vec<Vec<i64>>)> {
        let cfg_hidden = self.cfg.hidden_size;
        let prompt = self.prompt_embeds(phonemes, voice, anchor)?;
        let prompt_len = prompt.len() / cfg_hidden;
        let frame_limit = sampling.max_new_frames.min(max_expected_frames(phonemes));

        let names: Vec<String> = self.prefill.outputs().iter().map(|o| o.name().to_string()).collect();
        let tensor = Tensor::from_array(([1, prompt_len, cfg_hidden], prompt))?;
        let mut outputs = self.prefill.run(ort::inputs! { "inputs_embeds" => tensor })?;
        let (_, hidden) = outputs[names[0].as_str()].try_extract_tensor::<f32>()?;
        let mut state = hidden[hidden.len() - cfg_hidden..].to_vec();
        let mut past = names[1..]
            .iter()
            .map(|name| outputs.remove(name.as_str()).context("prefill output missing"))
            .collect::<Result<Vec<_>>>()?;
        drop(outputs);

        let vocab = self.heads.audio_vocab;
        let window = sampling.repetition_window;
        let mut history: Vec<_> =
            (0..self.cfg.n_vq).map(|_| ChannelHistory::new(vocab, window)).collect();
        let mut codec_state = self.new_codec_state()?;
        let mut all_frames = Vec::new();
        let mut pending: Vec<Vec<i64>> = Vec::new();
        let mut first_audio: Option<Instant> = None;
        let mut emitted = 0usize;

        for t in 0..frame_limit {
            if cancel.load(Ordering::Relaxed) {
                return Ok((Flow::Cancelled, all_frames));
            }
            let (codes, stop) = self.acoustic_frame(&state, sampling, &mut history)?;
            pending.push(codes.clone());
            all_frames.push(codes);
            let last = stop || t + 1 == frame_limit;
            if !last {
                (state, past) = self.backbone_step(all_frames.last().expect("just pushed"), anchor, prompt_len + t, past)?;
            }
            let lead = first_audio
                .map(|at| emitted as f32 / SAMPLE_RATE as f32 - at.elapsed().as_secs_f32());
            if last || pending.len() >= pacing.target(lead) {
                let audio = self.decode_group(&pending, &mut codec_state)?;
                pending.clear();
                if !audio.is_empty() {
                    first_audio.get_or_insert_with(Instant::now);
                    emitted += audio.len();
                    sink(audio);
                }
            }
            if stop {
                break;
            }
        }
        Ok((Flow::Finished, all_frames))
    }

    fn backbone_step(
        &mut self,
        codes: &[i64],
        anchor: &[f32],
        position: usize,
        past: Vec<DynValue>,
    ) -> Result<(Vec<f32>, Vec<DynValue>)> {
        let h = self.cfg.hidden_size;
        let layers = self.cfg.num_hidden_layers;
        let mut embeds = Vec::with_capacity(h);
        self.embed_row(self.cfg.speech_generation_start_token_id, Some(codes), anchor, &mut embeds);
        let mut inputs: Vec<(String, SessionInputValue<'static>)> = vec![
            ("inputs_embeds".into(), Tensor::from_array(([1, 1, h], embeds))?.into()),
            ("position_ids".into(), Tensor::from_array(([1usize, 1], vec![position as i64]))?.into()),
        ];
        for (i, value) in past.into_iter().enumerate() {
            let kind = if i < layers { "k" } else { "v" };
            inputs.push((format!("past_{kind}_{}", i % layers), value.into()));
        }
        let names: Vec<String> = self.decode.outputs().iter().map(|o| o.name().to_string()).collect();
        let mut outputs = self.decode.run(inputs)?;
        let (_, hidden) = outputs[names[0].as_str()].try_extract_tensor::<f32>()?;
        let hidden = hidden.to_vec();
        let past = names[1..]
            .iter()
            .map(|name| outputs.remove(name.as_str()).context("decode output missing"))
            .collect::<Result<Vec<_>>>()?;
        Ok((hidden, past))
    }
}
