import os
import re
import base64
import io
import tempfile

import runpod
from faster_whisper import WhisperModel
from llama_cpp import Llama
import soundfile as sf
import numpy as np
from kokoro import KPipeline
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic langdetect results

STT_MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "base")
stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")

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
    # faster-whisper (especially the small "base" model) occasionally
    # hallucinates by looping the same word/phrase over and over on short or
    # ambiguous audio (e.g. "1.5% 1.5% 1.5% ... x7"). If the *entire*
    # transcript is just N back-to-back repeats of a shorter phrase, collapse
    # it down to a single occurrence rather than translating/speaking the
    # repeated text N times.
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


def collapse_sentence_repetition(text):
    # Same idea one level up: a small quantized model asked to translate can
    # emit the translated sentence two or three times in a row, which then gets
    # synthesized and sounds like the reply is repeating itself.
    parts = [p.strip() for p in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s*", text) if p.strip()]
    if len(parts) < 2:
        return text
    deduped = []
    for part in parts:
        if not deduped or part != deduped[-1]:
            deduped.append(part)
    return " ".join(deduped)


_LABEL_PREFIX = re.compile(
    r"^\s*(translation|translated|english|japanese|output|answer|reply)\s*[:\-]\s*",
    re.IGNORECASE,
)


def clean_response(text):
    text = text.strip()
    text = _LABEL_PREFIX.sub("", text)
    # Strip a wrapping pair of quotes the model sometimes adds around the
    # translation; spoken aloud these become audible artifacts.
    if len(text) >= 2 and text[0] in "\"'\u201c\u300c" and text[-1] in "\"'\u201d\u300d":
        text = text[1:-1].strip()
    return collapse_sentence_repetition(text)


def transcribe(audio_path):
    segments, info = stt_model.transcribe(
        audio_path,
        beam_size=5,
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    text = collapse_repetition(text)
    return text, info.language


def call_llm(history, user_text, system_prompt=None):
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}, *history,
                {"role": "user", "content": user_text}]
    # Translation is not a creative task. The default sampling temperature made
    # the model paraphrase, add commentary, and sometimes restate the same
    # sentence; near-greedy decoding with a repeat penalty keeps it literal.
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.15,
    )
    return clean_response(result["choices"][0]["message"]["content"])


def synthesize(text):
    kokoro_lang_code, voice = detect_kokoro_target(text)
    pipeline = get_tts_pipeline(kokoro_lang_code)
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        return np.zeros(1, dtype=np.float32), 24000
    return np.concatenate(chunks), 24000


def handler(job):
    job_input = job["input"]
    audio_b64 = job_input.get("audio_base64")
    if not audio_b64:
        return {"error": "audio_base64 is required"}

    history = job_input.get("history", [])
    instructions = job_input.get("instructions")

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        user_text, detected_language = transcribe(tmp.name)

    if not user_text:
        return {"transcript": "", "response_text": "", "response_audio_base64": ""}

    response_text = call_llm(history, user_text, system_prompt=instructions)
    if not response_text:
        return {"transcript": user_text, "response_text": "", "response_audio_base64": ""}

    audio_array, sample_rate = synthesize(response_text)

    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV")
    response_audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "transcript": user_text,
        "detected_language": detected_language,
        "response_text": response_text,
        "response_audio_base64": response_audio_b64,
        "sample_rate": sample_rate,
    }


runpod.serverless.start({"handler": handler})
