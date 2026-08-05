# runpod-s2s-gpu-worker

RunPod Serverless GPU worker for a realtime speech-to-speech interpreter. One
request carries one utterance of audio; the worker transcribes it, translates
it with a local LLM, speaks the translation, and streams the pieces back as
they are produced.

It is driven by the CPU relay in
https://github.com/higeshogun/runpod-s2s-relay, which owns the client
WebSocket and all turn segmentation. This repository is only the GPU half.

## Pipeline

```
WAV (16 kHz mono PCM16)
  -> faster-whisper large-v3-turbo        transcript
  -> Gemma 4 E4B (Q4_0 GGUF, llama.cpp)   translated sentences, streamed
  -> Kokoro-82M                           24 kHz audio, per sentence
```

RunPod builds the image straight from this repository, so every push to
`main` triggers a new build and a rolling deploy. Builds take roughly 20
minutes, mostly compiling llama-cpp-python with CUDA and downloading model
weights. Bundle related changes onto one branch and merge once, rather than
pushing three commits and paying for three serial builds.

## Streaming

`handler` is a generator, registered with `return_aggregate_stream=True`, so
every dict it yields is delivered through RunPod's `/stream` endpoint the
moment it is produced instead of the whole turn being withheld until synthesis
finishes.

Chunk types yielded, in order:

| Type | Payload |
| --- | --- |
| `warmup` | `ready` - response to a no-op warmup request |
| `transcript` | `transcript`, `detected_language` |
| `text` | `text` - one translated sentence |
| `audio` | `audio_base64`, `sample_rate` - about 200 ms of PCM16 |
| `error` | `message` |
| `done` | always last, including on failure |

Sentences are emitted as soon as the LLM finishes them, so the first syllable
can be spoken while the rest of the reply is still being generated. That is
what turns a batch turn into a streaming one.

## Request format

```json
{
  "input": {
    "audio_base64": "<base64 WAV, 16 kHz mono PCM16>",
    "instructions": "<system prompt for this session>",
    "languages": ["en", "ja"],
    "history": []
  }
}
```

`{"input": {"warmup": true}}` is a no-op used to bring a worker up before the
first real utterance.

`history` is deliberately always empty. This is an interpreter, not a chat
assistant: feeding it conversation history makes it answer questions instead
of translating them.

## Models, and why these ones

**STT - faster-whisper `large-v3-turbo`.** Same encoder as large-v3 with the
decoder distilled from 32 layers down to 4, so it is large-v3-class accurate
while still fast enough for realtime. It stays multilingual, unlike the
`distil-*` checkpoints, which are English only and would break one direction
of the interpreter. The earlier default was `base`, which looped hallucinated
phrases and returned empty transcripts on a large share of Japanese turns.

Decoding is greedy (`beam_size=1`). Beam search costs real latency and buys
very little on short single-utterance clips.

Language ID is restricted to the session's two languages. Whisper's detector
runs over all 99 and gets short clips wrong often enough to matter - a
Japanese sentence detected as Korean comes back as garbage. `transcribe()`
inspects the detection before consuming the segment generator, so when the
guess falls outside the pair it can re-run with the right language forced
without ever decoding the wrong text.

**LLM - Gemma 4 E4B, Q4_0 GGUF, via llama-cpp-python with all layers on GPU.**
Translation is not a creative task, so decoding is near-greedy
(`temperature=0.2`, `repeat_penalty=1.15`) to keep the model literal instead
of paraphrasing or restating.

**TTS - Kokoro-82M.** The interpreter is bidirectional, so the output language
changes turn to turn and each language family needs its own pipeline and
voice. `detect_kokoro_target()` checks script ranges first, because langdetect
is unreliable on short strings and cannot reliably separate Japanese, Chinese
and Korean.

## Output cleanup

Small quantized models and neural vocoders both produce artefacts that are
obvious in a voice interface:

- `collapse_repetition()` folds a transcript that is nothing but N repeats of
  a shorter phrase back down to one occurrence.
- `_LABEL_PREFIX` strips leading "Translation:" style labels from the first
  sentence.
- `strip_wrapping_quotes()` removes quotes the model wraps around its output.
- Identical consecutive sentences are skipped, because the model sometimes
  restates itself.
- `trim_silence()` removes the near-silent padding Kokoro appends to every
  utterance, which otherwise became a dead gap between sentences.
- `apply_fades()` applies a 5 ms ramp at each end so concatenated sentences do
  not click.
- `normalize()` peak-normalizes toward a fixed target with the gain clamped,
  so a quiet sentence cannot be blown up.
- `to_pcm16_bytes()` rounds rather than truncates; `astype()` alone throws away
  up to a full LSB on every sample.

Cleanup runs on the whole sentence before it is sliced into chunks, because
trimming and fading have to see the real start and end of the utterance rather
than an arbitrary chunk edge.

## Build-time model download

Whisper, Gemma and Kokoro weights, the unidic dictionary for the Japanese G2P,
and the English and Japanese Kokoro pipelines are all fetched and warmed
during the Docker build. Downloading them at runtime meant every fresh worker
hit the Hugging Face Hub over the network, adding highly variable and
sometimes multi-minute delays, made worse by unauthenticated rate limiting.

This makes the image large and the build slow, and that is the correct
trade: it is paid once per build instead of once per cold worker.

## Failure handling

A missing or broken G2P dependency for one language must never take down the
worker. `get_tts_pipeline()` falls back to English for any language whose
pipeline fails to initialize, and caches the fallback. English itself is the
one exception and is allowed to raise, since without it there is no TTS at
all.

Likewise, a synthesis failure on one sentence is logged and skipped rather
than aborting the turn, and `done` is always yielded so the relay can close
the turn cleanly.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `STT_MODEL_SIZE` | `large-v3-turbo` | also baked into the image at build time |
| `STT_COMPUTE_TYPE` | `float16` | |
| `STT_BEAM_SIZE` | `1` | greedy |
| `LLM_GGUF_PATH` | `/app/gemma-4-E4B-it-Q4_0.gguf` | |
| `LLM_CONTEXT_SIZE` | `4096` | |
| `SYSTEM_PROMPT` | generic assistant | only used if the request omits `instructions` |
| `TTS_SPEED` | `1.0` | |
| `TTS_TARGET_PEAK` | `0.95` | |

## Endpoint configuration

- Queue based, built from this repo on branch `main`
- GPU type 24 GB, GPU count 1
- Max workers 3, active workers 0 so it scales to zero
- Idle timeout 45 s, execution timeout 600 s

Idle timeout is the main latency/cost dial. At 45 s roughly a quarter of
requests hit a cold start. Raising it cuts cold starts but bills idle GPU
seconds during natural pauses in conversation.

Observed timings: a cold worker takes about 40 s to become ready and 40 to 60 s
to first audio; a warm worker returns first audio in 1.5 to 4.5 seconds.
