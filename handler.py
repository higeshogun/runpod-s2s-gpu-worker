import os
import re
import base64
import tempfile
import traceback

import runpod
from faster_whisper import WhisperModel
from llama_cpp import Llama
import numpy as np
from kokoro import KPipeline
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic langdetect results

# large-v3-turbo is Whisper large-v3 with the decoder distilled from 32 layers
# down to 4: it keeps large-v3-class accuracy while decoding fast enough for
# realtime, and it stays multilingual (unlike the distil-* checkpoints, which
# are English only) so it serves both directions of the interpreter. The old
# default was "base", which looped hallucinated phrases and returned empty
# transcripts on a large share of Japanese turns.
STT_MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "large-v3-turbo")
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "float16")
# Greedy decoding: beam search costs real latency and buys very little on the
# short single-utterance clips the relay sends.
STT_BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", "1"))
stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type=STT_COMPUTE_TYPE)

LLM_GGUF_PATH = os.environ.get("LLM_GGUF_PATH", "/app/gemma-4-E4B-it-Q4_0.gguf")
llm = Llama(
    model_path=LLM_GGUF_PATH,
    n_gpu_layers=-1,
    n_ctx=int(os.environ.get("LLM_CONTEXT_SIZE", "4096")),
    verbose=False,
)

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful, concise voice assistant. Keep replies short and conversational.",
)

TTS_SAMPLE_RATE = 24000
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))
# ~200 ms of audio per streamed chunk: small enough that playback starts as soon
# as the first syllable exists, large enough not to flood the relay.
TTS_CHUNK_SAMPLES = int(TTS_SAMPLE_RATE * 0.2)
TTS_FADE_SAMPLES = int(TTS_SAMPLE_RATE * 0.005)  # 5 ms de-click ramp
TTS_KEEP_SILENCE_SAMPLES = int(TTS_SAMPLE_RATE * 0.02)
TTS_SILENCE_FLOOR = 0.004
TTS_TARGET_PEAK = float(os.environ.get("TTS_TARGET_PEAK", "0.95"))

# The interpreter is bidirectional (e.g. English<->Japanese), so the TTS output
# language changes turn to turn. Kokoro needs a pipeline + matching voice per
# language family (Japanese/Mandarin need the misaki[ja]/misaki[zh] extras -
# see requirements.txt), so we route each response to the right one instead of
# always using the English pipeline (which just garbles non-Latin scripts).
LANG_TO_KOKORO = {
    "en": ("a", "af_heart"),
    "ja": ("j", "jf_alpha"),
    "zh-cn": ("z", "zf_xiaobei"),
    "zh": ("z", "zf_xiaobei"),
    "fr": ("f", "ff_siwis"),
    "es": ("e", "ef_dora"),
    "it": ("i", "if_sara"),
    "pt": ("p", "pf_dora"),
    "hi": ("h", "hf_alpha"),
}
DEFAULT_KOKORO = ("a", "af_heart")

# Human-readable language names for building an explicit "translate INTO X"
# instruction when we have to correct a turn the model got wrong.
LANG_LABELS = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "de": "German",
    "hi": "Hindi",
    "ru": "Russian",
    "ar": "Arabic",
}

_tts_pipelines = {}

def get_tts_pipeline(kokoro_lang_code):
    pipeline = _tts_pipelines.get(kokoro_lang_code)
    if pipeline is not None:
        return pipeline
    try:
        pipeline = KPipeline(lang_code=kokoro_lang_code)
    except Exception as e:
        # Never let a missing/broken G2P dependency for one language (e.g. a
        # non-English tokenizer's dictionary not being downloaded) crash or
        # permanently break the worker. Fall back to English instead of
        # raising, except for English itself which must always work.
        print(f"[handler] failed to init Kokoro pipeline for lang_code={kokoro_lang_code!r}: {e}", flush=True)
        if kokoro_lang_code == DEFAULT_KOKORO[0]:
            raise
        fallback = get_tts_pipeline(DEFAULT_KOKORO[0])
        _tts_pipelines[kokoro_lang_code] = fallback
        return fallback
    _tts_pipelines[kokoro_lang_code] = pipeline
    return pipeline

_HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30ff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_HANGUL = re.compile(r"[\uac00-\ud7a3]")

def detect_kokoro_target(text):
    # Script-based checks first: langdetect is unreliable on short strings and
    # can't tell Japanese/Chinese/Korean apart reliably by itself.
    if _HIRAGANA_KATAKANA.search(text):
        return LANG_TO_KOKORO["ja"]
    if _HANGUL.search(text):
        return DEFAULT_KOKORO  # Korean isn't supported by Kokoro; avoid mangling it
    if _CJK.search(text):
        return LANG_TO_KOKORO["zh"]
    try:
        code = detect(text)
    except Exception:
        return DEFAULT_KOKORO
    return LANG_TO_KOKORO.get(code, DEFAULT_KOKORO)

def text_script_language(text):
    # Best-effort language code for a piece of text, using unambiguous scripts
    # first (the same ordering detect_kokoro_target relies on) and falling back
    # to langdetect for Latin-script languages. Returns None when undecidable.
    if _HIRAGANA_KATAKANA.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    if _CJK.search(text):
        return "zh"
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return detect(stripped)
    except Exception:
        return None

def other_language(detected, allowed):
    # The interpreter is strictly two-way, so the target language is simply the
    # session language that is NOT the one that was spoken. Fall back to English
    # when the pair is unknown or degenerate.
    codes = [c for c in (allowed or []) if c]
    for code in codes:
        if code != detected:
            return code
    return "en" if detected != "en" else (codes[0] if codes else "en")

def looks_like_echo(output_text, source_lang, target_lang):
    # True when the model returned text in the SOURCE language instead of the
    # target - i.e. it echoed/paraphrased the input rather than translating it.
    # This is the JA->EN failure: Whisper hears Japanese, the model restates it
    # in Japanese, and script-based TTS then speaks it back in Japanese.
    lang = text_script_language(output_text)
    if lang is None:
        return False
    # Normalize Chinese variants so "zh-cn" and "zh" compare equal.
    norm = lambda c: "zh" if (c or "").startswith("zh") else c
    if norm(lang) == norm(target_lang):
        return False
    if norm(lang) == norm(source_lang):
        return True
    # For CJK targets, a Latin-script answer is fine (names/loanwords); only
    # treat an unambiguous source-script match as an echo. Anything else we
    # leave alone rather than risk fighting correct output.
    return False

# Pre-warm English at import time (required - if this fails there's no TTS at
# all, so we let it raise). Pre-warm Japanese too, but never let a failure
# here take down the whole worker - get_tts_pipeline() will fall back to
# English for every request until this is fixed, instead of crash-looping.
get_tts_pipeline("a")
try:
    get_tts_pipeline("j")
except Exception as e:
    print(f"[handler] Japanese TTS pipeline unavailable, will fall back to English: {e}", flush=True)

def collapse_repetition(text):
    # Whisper occasionally hallucinates by looping the same word/phrase over and
    # over on short or ambiguous audio. If the *entire* transcript is just N
    # back-to-back repeats of a shorter phrase, collapse it to one occurrence.
    words = text.split()
    n = len(words)
    if n < 2:
        return text
    for size in range(1, n // 2 + 1):
        if n % size != 0:
            continue
        pattern = words[:size]
        if pattern * (n // size) == words:
            return " ".join(pattern)
    return text

_LABEL_PREFIX = re.compile(
    r"^\s*(translation|translated|english|japanese|output|answer|reply)\s*[:\-]\s*",
    re.IGNORECASE,
)
_OPEN_QUOTES = "\"'\u201c\u300c"
_CLOSE_QUOTES = "\"'\u201d\u300d"

def strip_wrapping_quotes(text):
    text = text.strip()
    if len(text) >= 2 and text[0] in _OPEN_QUOTES and text[-1] in _CLOSE_QUOTES:
        return text[1:-1].strip()
    return text

_STT_DECODE_OPTS = dict(
    beam_size=STT_BEAM_SIZE,
    condition_on_previous_text=False,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    vad_filter=True,
)

def choose_language(info, allowed):
    # Whisper's language ID runs over all 99 languages and gets short clips wrong
    # often enough to matter - a Japanese sentence detected as Korean comes back
    # as garbage. A session only ever involves two languages, so restrict the
    # choice to those and take the most probable one.
    if not allowed:
        return None
    if info.language in allowed:
        return info.language
    for code, _prob in (getattr(info, "all_language_probs", None) or []):
        if code in allowed:
            return code
    return allowed[0]

def transcribe(audio_path, allowed_languages=None):
    allowed = [code for code in (allowed_languages or []) if code]
    # transcribe() returns the detected-language info immediately; decoding only
    # happens as the segment generator is consumed. So we can inspect the
    # detection and, when it lands outside the session's languages, redo the
    # call with the right one forced without ever decoding the wrong text.
    segments, info = stt_model.transcribe(audio_path, language=None, **_STT_DECODE_OPTS)
    language = choose_language(info, allowed)
    if language and language != info.language:
        print(f"[handler] detected {info.language!r} outside session {allowed}; forcing {language!r}", flush=True)
        segments, info = stt_model.transcribe(audio_path, language=language, **_STT_DECODE_OPTS)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return collapse_repetition(text), info.language

def build_system_prompt(base, target_lang):
    # Reinforce the required output language on top of whatever bidirectional
    # persona the client sent. A small quantized model (Gemma-4-E4B) will often
    # echo/paraphrase the source when translating INTO English from a non-Latin
    # language, because copying is the locally cheapest continuation; naming the
    # target explicitly makes the correct direction the expected one.
    base = base or SYSTEM_PROMPT
    label = LANG_LABELS.get(target_lang)
    if not label:
        return base
    return (
        f"{base}\n\nThe person just spoke. Output ONLY the {label} translation of "
        f"what they said. Do not reply in any other language. Do not repeat or "
        f"paraphrase their words in the original language. Output only the "
        f"{label} translation, with no preface, label, or commentary."
    )

def run_llm(user_text, system_prompt, temperature=0.2, repeat_penalty=1.15):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
    out = []
    for part in llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=repeat_penalty,
        stream=True,
    ):
        choice = (part.get("choices") or [{}])[0]
        content = (choice.get("delta") or {}).get("content")
        if content:
            out.append(content)
    return "".join(out).strip()

def translate(user_text, base_instructions, source_lang, target_lang):
    # Do NOT feed conversation history: this is a stateless per-turn interpreter.
    # First pass uses the (target-reinforced) persona. If the model echoed the
    # source language instead of translating, re-run once with a harder, more
    # explicit instruction and slightly different decoding to break the copy.
    prompt = build_system_prompt(base_instructions, target_lang)
    text = run_llm(user_text, prompt)
    if target_lang and looks_like_echo(text, source_lang, target_lang):
        label = LANG_LABELS.get(target_lang, target_lang)
        print(f"[handler] output looked like {source_lang!r} echo; forcing {target_lang!r} retranslate", flush=True)
        hard_prompt = (
            f"You are a translation engine. Translate the user's message into {label}. "
            f"Respond with ONLY the {label} translation and nothing else. "
            f"Never answer, never reply in the original language, never add commentary."
        )
        retry = run_llm(user_text, hard_prompt, temperature=0.3, repeat_penalty=1.3)
        if retry and not looks_like_echo(retry, source_lang, target_lang):
            text = retry
        elif retry:
            text = retry  # still not ideal, but the hard prompt is our best shot
    return text

# Sentence terminators for Latin and CJK punctuation, including any closing
# quote/bracket that follows them.
_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f\uff0e][\"'\u201d\u300d)\uff09]*")
MIN_SENTENCE_CHARS = 8
MAX_SENTENCE_CHARS = 140

def stream_sentences(text):
    # Split a completed translation into speakable sentences so each can be
    # synthesized and streamed as soon as it is ready. (The translation is now
    # produced whole - rather than token-streamed - because the echo guard has
    # to see the full output before we commit to synthesizing it.)
    buf = text
    while True:
        match = None
        for candidate in _SENTENCE_END.finditer(buf):
            if candidate.end() >= MIN_SENTENCE_CHARS:
                match = candidate
                break
        if match is None:
            break
        sentence, buf = buf[:match.end()].strip(), buf[match.end():]
        if sentence:
            yield sentence
        while len(buf) >= MAX_SENTENCE_CHARS:
            cut = buf.rfind(" ", 0, MAX_SENTENCE_CHARS)
            if cut <= 0:
                cut = MAX_SENTENCE_CHARS
            head, buf = buf[:cut].strip(), buf[cut:]
            if head:
                yield head
    tail = buf.strip()
    if tail:
        yield tail

def trim_silence(samples):
    # Kokoro pads every utterance with a stretch of near-silence. Left in place
    # it becomes a dead gap between each sentence, which is a large part of why
    # the interpreter sounded slow and disjointed.
    if not samples.size:
        return samples
    loud = np.nonzero(np.abs(samples) > TTS_SILENCE_FLOOR)[0]
    if not loud.size:
        return samples[:0]
    start = max(0, int(loud[0]) - TTS_KEEP_SILENCE_SAMPLES)
    end = min(samples.size, int(loud[-1]) + 1 + TTS_KEEP_SILENCE_SAMPLES)
    return samples[start:end]

def apply_fades(samples):
    # A sentence that starts or ends on a non-zero sample clicks audibly when the
    # player concatenates it with the next one.
    n = min(TTS_FADE_SAMPLES, samples.size // 2)
    if n <= 0:
        return samples
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = samples.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out

def normalize(samples):
    # Kokoro's output level varies between voices and between sentences, which
    # made the translated speech quiet and inconsistent. Peak-normalize toward a
    # fixed target, with the gain clamped so a quiet sentence can't be blown up.
    if not samples.size:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-4:
        return samples
    gain = min(max(TTS_TARGET_PEAK / peak, 0.5), 3.0)
    return samples * gain

def to_pcm16_bytes(samples):
    if not samples.size:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    # Round rather than truncate: astype() alone throws away up to a full LSB on
    # every sample, which is just added quantization noise.
    return np.round(clipped * 32767.0).astype(np.int16).tobytes()

def synthesize_chunks(text):
    kokoro_lang_code, voice = detect_kokoro_target(text)
    pipeline = get_tts_pipeline(kokoro_lang_code)
    parts = []
    for _, _, audio in pipeline(text, voice=voice, speed=TTS_SPEED):
        if audio is None:
            continue
        parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not parts:
        return
    samples = parts[0] if len(parts) == 1 else np.concatenate(parts)
    # Clean up the whole sentence before slicing it: trimming and fading have to
    # see the real start and end of the utterance, not an arbitrary chunk edge.
    samples = normalize(apply_fades(trim_silence(samples)))
    for start in range(0, samples.size, TTS_CHUNK_SAMPLES):
        pcm = to_pcm16_bytes(samples[start:start + TTS_CHUNK_SAMPLES])
        if pcm:
            yield pcm

def handler(job):
    # Generator handler: every yielded dict is delivered to the relay through
    # RunPod's /stream endpoint as soon as it is produced, instead of the whole
    # turn being withheld until synthesis finishes.
    job_input = job.get("input") or {}

    if job_input.get("warmup"):
        # Cheap no-op request used to bring a worker up before the first real
        # utterance, so the user doesn't pay the cold start mid-conversation.
        yield {"type": "warmup", "ready": True}
        return

    audio_b64 = job_input.get("audio_base64")
    if not audio_b64:
        yield {"type": "error", "message": "audio_base64 is required"}
        return

    instructions = job_input.get("instructions")
    languages = job_input.get("languages")

    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            user_text, detected_language = transcribe(tmp.name, languages)

        yield {"type": "transcript", "transcript": user_text, "detected_language": detected_language}
        if not user_text:
            yield {"type": "done"}
            return

        # The target is the other language of the two-way session. Passing it
        # explicitly (and re-checking the output) is what fixes JA->EN, where
        # the model used to echo Japanese back and TTS then spoke Japanese.
        target_language = other_language(detected_language, languages)
        translated = translate(user_text, instructions, detected_language, target_language)
        if not translated:
            print("[handler] translation was empty", flush=True)
            yield {"type": "done"}
            return

        spoken_any = False
        last_sentence = None
        first = True
        for sentence in stream_sentences(translated):
            if first:
                sentence = _LABEL_PREFIX.sub("", sentence)
                first = False
            sentence = strip_wrapping_quotes(sentence)
            if not sentence or sentence == last_sentence:
                continue  # a small quantized model sometimes restates a sentence
            last_sentence = sentence

            yield {"type": "text", "text": sentence}
            try:
                for pcm in synthesize_chunks(sentence):
                    spoken_any = True
                    yield {
                        "type": "audio",
                        "audio_base64": base64.b64encode(pcm).decode(),
                        "sample_rate": TTS_SAMPLE_RATE,
                    }
            except Exception as e:
                print(f"[handler] synthesis failed for a sentence: {e}", flush=True)
                traceback.print_exc()

        if not spoken_any:
            print("[handler] turn produced no audio", flush=True)
        yield {"type": "done"}
    except Exception as e:
        print(f"[handler] turn failed: {e}", flush=True)
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
        yield {"type": "done"}

runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
