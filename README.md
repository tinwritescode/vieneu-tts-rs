# VieNeuTTS Rust

Offline Vietnamese and English text-to-speech for the desktop, written in Rust.
This is a rewrite of [VieNeuTTSApp](https://github.com/NguyenSiTrung/VieNeuTTSApp)
that aims for lower memory use. It runs the VieNeu-TTS v3 Turbo ONNX models on
the CPU; after the first model download, nothing leaves your machine.

## Features

- Streaming speech synthesis at 48 kHz, with speed and pause controls
- Emotion cues: `[cười]`, `[thở dài]`, `[hắng giọng]`
- 20 preset voices, with search and filters
- Voice cloning from a short recording
- Import from `.txt`, `.md`, `.docx`, `.pdf` and `.srt`
- WAV export
- Vietnamese and English interface
- Desktop app (Tauri) and command-line tool

Not included from the original app: audiobooks, Studio, speech recognition,
Qwen, video dubbing, MP3 export and GPU acceleration.

## Memory and speed

Measured on an Apple Silicon Mac with a short sentence:

| | Peak memory |
|---|---|
| Python app | 1.39 GB |
| Rust CLI | 626 MB |
| Desktop app, models not loaded | 105 MB |
| Desktop app, generating | 675 MB |

Synthesis runs at about 5–6× realtime. The desktop app unloads the models after
two minutes idle.

## Build

You need a recent stable Rust toolchain and Node.js. The app has only been built
and run on macOS so far.

Desktop app:

```sh
cd app
npm ci
npx tauri build --bundles app
```

The bundle is written to `target/release/bundle/macos/VieNeuTTS.app`. On first
launch the app offers to download the models (388 MB).

Command-line tool:

```sh
cargo build --release -p vieneu-cli
```

## Command-line use

```sh
vieneu install                          # download the models
vieneu voices                           # list voices
vieneu say "Xin chào các bạn." -o hello.wav
vieneu say --file story.txt -o story.wav -v Adam --speed 1.2 --pause 0.3
vieneu clone "My voice" recording.wav   # add --no-denoise to skip noise removal
```

`--threads` sets the CPU thread count. `VIENEU_MODELS` overrides the model folder.

## Project layout

| Path | Contents |
|---|---|
| `crates/vieneu-core` | Text front end, ONNX engine, DSP, voices, importers, model installer |
| `crates/vieneu-cli` | The `vieneu` command |
| `app/src-tauri` | Desktop backend: synthesis worker and audio playback |
| `app/src` | Desktop interface (TypeScript) |
| `tools/gen_fixtures.py` | Generates test fixtures from the Python SDK |

## Tests

```sh
cargo test --workspace --release                       # models not needed
cargo test --workspace --release -- --include-ignored  # needs installed models
cd app && npm test
```

The ignored tests compare the port against golden outputs from the Python
`vieneu` 3.3.0 SDK. Audio is not sample-identical to Python: the quantized models
amplify tiny floating-point differences. Given Python's exact speaker anchor,
generated frames match exactly.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Models are downloaded
from Hugging Face under their own licenses.
