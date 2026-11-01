//! Code sampling and the sliding-window repetition history.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// Zero or below means greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Frames a code stays penalized.
    pub repetition_window: usize,
    pub max_new_frames: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 25,
            top_p: 0.95,
            repetition_penalty: 1.2,
            repetition_window: 64,
            max_new_frames: 300,
        }
    }
}

/// Codes one codebook produced in the last `window` frames.
pub struct ChannelHistory {
    counts: Vec<u32>,
    order: VecDeque<usize>,
    window: usize,
}

impl ChannelHistory {
    pub fn new(vocab: usize, window: usize) -> Self {
        Self { counts: vec![0; vocab], order: VecDeque::with_capacity(window + 1), window }
    }

    pub fn add(&mut self, code: usize) {
        self.counts[code] += 1;
        self.order.push_back(code);
        if self.order.len() > self.window {
            if let Some(old) = self.order.pop_front() {
                self.counts[old] -= 1;
            }
        }
    }

    fn penalize(&self, logits: &mut [f32], penalty: f32) {
        for (logit, _) in logits.iter_mut().zip(&self.counts).filter(|(_, count)| **count > 0) {
            *logit = if *logit < 0.0 { *logit * penalty } else { *logit / penalty };
        }
    }
}

pub(crate) fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = i;
        }
    }
    best
}

/// Pick a code from `logits`, which are consumed as scratch space.
pub fn sample(
    logits: &mut [f32],
    params: &Sampling,
    history: &ChannelHistory,
    rng: &mut fastrand::Rng,
    scratch: &mut Vec<(f32, u32)>,
) -> usize {
    if (params.repetition_penalty - 1.0).abs() > 1e-6 {
        history.penalize(logits, params.repetition_penalty);
    }
    if params.temperature <= 0.0 {
        return argmax(logits);
    }
    scratch.clear();
    scratch.extend(logits.iter().enumerate().map(|(i, l)| (*l / params.temperature, i as u32)));
    let descending = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0);
    if params.top_k > 0 && params.top_k < scratch.len() {
        scratch.select_nth_unstable_by(params.top_k - 1, descending);
        scratch.truncate(params.top_k);
    }
    scratch.sort_unstable_by(descending);

    let top = scratch[0].0;
    for candidate in scratch.iter_mut() {
        candidate.0 = (candidate.0 - top).exp();
    }
    let total: f32 = scratch.iter().map(|c| c.0).sum();
    let mut seen = 0.0;
    let mut kept = 0.0;
    let mut end = 0;
    for candidate in scratch.iter_mut() {
        candidate.0 /= total;
        if seen < params.top_p {
            end += 1;
            kept += candidate.0;
        }
        seen += candidate.0;
    }
    let mut point = rng.f32() * kept;
    for candidate in &scratch[..end] {
        point -= candidate.0;
        if point <= 0.0 {
            return candidate.1 as usize;
        }
    }
    scratch[end - 1].1 as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(logits: &[f32], params: &Sampling, history: &ChannelHistory) -> usize {
        let mut rng = fastrand::Rng::with_seed(7);
        sample(&mut logits.to_vec(), params, history, &mut rng, &mut Vec::new())
    }

    #[test]
    fn greedy_avoids_a_recently_penalized_code() {
        let params = Sampling { temperature: 0.0, repetition_penalty: 2.0, ..Default::default() };
        let mut history = ChannelHistory::new(3, 2);
        assert_eq!(draw(&[1.0, 3.0, 2.0], &params, &history), 1);
        history.add(1);
        assert_eq!(draw(&[1.0, 3.0, 2.0], &params, &history), 2);
    }

    #[test]
    fn penalty_expires_after_the_window() {
        let params = Sampling { temperature: 0.0, repetition_penalty: 2.0, ..Default::default() };
        let mut history = ChannelHistory::new(3, 2);
        history.add(1);
        history.add(0);
        history.add(0);
        assert_eq!(draw(&[1.0, 3.0, 2.0], &params, &history), 1);
    }

    #[test]
    fn negative_logits_get_more_negative() {
        let params = Sampling { temperature: 0.0, repetition_penalty: 2.0, ..Default::default() };
        let mut history = ChannelHistory::new(2, 4);
        history.add(0);
        assert_eq!(draw(&[-1.0, -1.5], &params, &history), 1);
    }

    #[test]
    fn top_k_only_returns_top_candidates() {
        let params = Sampling { top_k: 2, top_p: 1.0, repetition_penalty: 1.0, ..Default::default() };
        let history = ChannelHistory::new(5, 4);
        let mut rng = fastrand::Rng::with_seed(1);
        for _ in 0..200 {
            let mut logits = [0.0, 5.0, 0.1, 5.0, 0.2];
            let code = sample(&mut logits, &params, &history, &mut rng, &mut Vec::new());
            assert!(code == 1 || code == 3, "{code}");
        }
    }

    #[test]
    fn top_p_cuts_the_tail() {
        let params = Sampling { top_k: 0, top_p: 0.5, repetition_penalty: 1.0, ..Default::default() };
        let history = ChannelHistory::new(3, 4);
        let mut rng = fastrand::Rng::with_seed(1);
        for _ in 0..200 {
            let mut logits = [10.0, 0.0, 0.0];
            assert_eq!(sample(&mut logits, &params, &history, &mut rng, &mut Vec::new()), 0);
        }
    }
}
