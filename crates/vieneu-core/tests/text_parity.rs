mod common;
use common::{fixture, models, strs};
use vieneu_core::frontend::Frontend;
use vieneu_core::text::{max_expected_frames, pack_sentences_into_chunks, split_into_sentences};

#[test]
fn sentences_and_packing_match_python() {
    for case in fixture("sentences.json").as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let sentences = split_into_sentences(text);
        assert_eq!(sentences, strs(&case["sentences"]), "{text}");
        assert_eq!(pack_sentences_into_chunks(&sentences, 40), strs(&case["packed_40"]), "{text}");
    }
}

#[test]
#[ignore = "needs the downloaded dictionary"]
fn chunks_gaps_and_phonemes_match_python() {
    let frontend = Frontend::open(&models().join("sea_g2p.bin")).unwrap();
    for case in fixture("chunks.json").as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let max_chars = case["max_chars"].as_u64().unwrap() as usize;
        let plan = frontend.plan(text, max_chars);
        let chunks: Vec<String> = plan.chunks.clone();
        assert_eq!(chunks, strs(&case["chunks"]), "{text} @ {max_chars}");
        let gaps: Vec<String> = plan.gaps.iter().map(|g| g.name().to_string()).collect();
        assert_eq!(gaps, strs(&case["gaps"]), "{text} @ {max_chars}");
        for (i, chunk) in chunks.iter().enumerate() {
            let ph = frontend.phonemize(chunk);
            assert_eq!(ph, case["phonemes"][i].as_str().unwrap());
            assert_eq!(max_expected_frames(&ph) as u64, case["frame_caps"][i].as_u64().unwrap());
        }
    }
}
