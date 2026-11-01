"""Generate golden fixtures from the reference Python `vieneu` SDK (3.3.0).

The Rust port is tested for parity against these outputs. Run with a venv that
has `vieneu==3.3.0` and its runtime deps installed:

    python tools/gen_fixtures.py <models_dir> <ref_wav> <out_dir>

`models_dir` has the layout the Rust app downloads: `vieneu/`, `codec/`.
"""
import json
import sys
from pathlib import Path

import numpy as np

from vieneu_utils.core_utils import max_expected_frames, split_into_sentences, pack_sentences_into_chunks
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

TEXTS = [
    "Xin chào các bạn [cười], hôm nay là 12/3/2024.",
    "Có phải ... kiểu như: \"Rồi sao nữa? Mình phải làm đến bao giờ?\", đúng không anh? Thế à!",
    "Chương một.\nNgày xửa ngày xưa, ở một ngôi làng nhỏ ven sông, có một cậu bé tên là An. "
    "Cậu rất thích đọc sách và thường ngồi dưới gốc cây đa đầu làng mỗi buổi chiều [thở dài]. "
    "Giá vàng hôm nay tăng 3.5 triệu đồng mỗi lượng, tương đương 2,5%.\n\nHết.",
    "Tôi dùng iPhone 15 Pro Max và laptop MacBook để làm việc với Python và Rust mỗi ngày.",
    "Chào.",
    "Một câu rất dài không có dấu ngắt nào để bám vào và nó cứ kéo dài mãi mãi cho đến khi "
    "người đọc cảm thấy mệt mỏi nhưng vẫn phải tiếp tục đọc vì nếu dừng lại thì sẽ không hiểu "
    "được ý nghĩa của toàn bộ đoạn văn này sau khi đã đọc xong phần mở đầu rất dài dòng",
]


def chunk_fixtures():
    out = []
    for text in TEXTS:
        for max_chars in (256, 60):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
            phs = [phonemize_text_with_emotions(c) for c in chunks]
            out.append({
                "text": text, "max_chars": max_chars, "chunks": chunks, "gaps": gaps,
                "phonemes": phs, "frame_caps": [max_expected_frames(p) for p in phs],
            })
    return out


def sentence_fixtures():
    out = []
    for text in TEXTS:
        sents = split_into_sentences(text)
        out.append({"text": text, "sentences": sents,
                    "packed_40": pack_sentences_into_chunks(sents, max_chars=40)})
    return out


def engine_fixtures(models: Path, ref_wav: str):
    from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine

    eng = OnnxV3LiteEngine(checkpoint_path=str(models / "vieneu"), onnx_dir=str(models / "vieneu"),
                           codec_dir=str(models / "codec"), threads=4)
    voices = json.loads((Path(__import__("vieneu").__file__).parent / "assets" / "voices_v3_turbo.json")
                        .read_text(encoding="utf-8"))
    v = voices["presets"][voices["default_voice"]]
    emb = np.asarray(v["speaker_emb"], np.float32)
    codes = np.asarray(v["codes"], np.int64)
    ph = phonemize_text_with_emotions("Xin chào các bạn.")
    token_ids = eng.tokenizer.encode(ph, add_special_tokens=False).ids
    anchor = eng._speaker_anchor(emb)

    # Greedy decode is deterministic, so Rust must reproduce the same frames.
    # Capture them at the codec input.
    captured = {}
    orig_decode = eng._decode_codes

    def cap(c):
        captured["codes"] = np.asarray(c).tolist()
        return orig_decode(c)

    eng._decode_codes = cap
    wav = eng.infer(phonemes=ph, ref_codes=codes, speaker_emb=emb, temperature=0.0,
                    repetition_penalty=1.2)
    frames = captured["codes"]

    import soundfile as sf
    ref, sr = sf.read(ref_wav, dtype="float32", always_2d=True)
    ref = ref.mean(axis=1)
    spk = eng.extract_speaker_emb(ref, sr)
    from vieneu._v3_turbo_engine.speaker import extract_speaker_fbank
    fb = extract_speaker_fbank(ref, sample_rate=sr)
    den = eng.denoiser.denoise(ref, sr)
    ref_codes = eng._encode_ref_wav(ref[: int(8 * sr)], sr)

    from vieneu._v3_turbo_engine.speaker.audio_utils import extract_fbank
    t = np.arange(16000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)
    fb16 = extract_fbank(tone, sample_rate=16000, n_mels=80, mean_norm=True)

    return {
        "voice": voices["default_voice"],
        "phonemes": ph,
        "token_ids": token_ids,
        "anchor": anchor.tolist(),
        "greedy_frames": frames,
        "wav_len": int(wav.size),
        "wav_rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))),
        "ref_wav_sr": int(sr),
        "fbank_shape": list(fb.shape),
        "fbank16_shape": list(fb16.shape),
        "fbank16_row": fb16[7].tolist(),
        "speaker_emb": spk.tolist(),
        "denoise_len": int(den.size),
        "denoise_rms": float(np.sqrt(np.mean(den.astype(np.float64) ** 2))),
        "ref_codes_shape": list(ref_codes.shape),
        "ref_codes_head": ref_codes[:2].tolist(),
    }


def wsola_fixture():
    sys.path.insert(0, str(Path(sys.argv[4]) if len(sys.argv) > 4 else Path(".")))
    try:
        from vienetts_app.core.audio import time_stretch_audio
    except ImportError:
        return None
    sr = 48000
    t = np.arange(int(0.6 * sr)) / sr
    sig = (0.4 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 410 * t)).astype(np.float32)
    out = {}
    for rate in (0.75, 1.3):
        y = time_stretch_audio(sig, rate, sr)
        out[str(rate)] = {"len": int(y.size), "head": y[1000:1010].tolist(),
                          "sum_abs": float(np.abs(y.astype(np.float64)).sum())}
    return out


def main():
    models, ref_wav, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(json.dumps(chunk_fixtures(), ensure_ascii=False, indent=1), "utf-8")
    (out_dir / "sentences.json").write_text(json.dumps(sentence_fixtures(), ensure_ascii=False, indent=1), "utf-8")
    ws = wsola_fixture()
    if ws is not None:
        (out_dir / "wsola.json").write_text(json.dumps(ws, indent=1), "utf-8")
    (out_dir / "engine.json").write_text(json.dumps(engine_fixtures(models, ref_wav), ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
