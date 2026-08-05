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

TTS_SAMPLE_RATE = 24000

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
    # it down to a single occurrence.
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
    return collapse_repetition(text), info.language


def llm_deltas(history, user_text, system_prompt=None):
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}, *history,
                {"role": "user", "content": user_text}]
    # Translation is not a creative task: near-greedy decoding with a repeat
    # penalty keeps the model literal instead of paraphrasing or restating.
    for part in llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.15,
        stream=True,
    ):
        choice = (part.get("choices") or [{}])[0]
        content = (choice.get("delta") or {}).get("content")
        if content:
            yield content


# Sentence terminators for Latin and CJK punctuation, including any closing
# quote/bracket that follows them.
_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f\uff0e][\"'\u201d\u300d)\uff09]*")
MIN_SENTENCE_CHARS = 8
MAX_SENTENCE_CHARS = 140


def stream_sentences(delta_iter):
    # Emit complete sentences as soon as the model produces them so the first
    # chunk of speech can be synthesized and played while the rest is still
    # being generated. This is what turns a batch turn into a streaming one.
    buf = ""
    for delta in delta_iter:
        buf += delta
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
        if len(buf) >= MAX_SENTENCE_CHARS:
            cut = buf.rfind(" ", 0, MAX_SENTENCE_CHARS)
            if cut <= 0:
                cut = MAX_SENTENCE_CHARS
            sentence, buf = buf[:cut].strip(), buf[cut:]
            if sentence:
                yield sentence
    tail = buf.strip()
    if tail:
        yield tail


def to_pcm16_bytes(audio):
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def synthesize_chunks(text):
    kokoro_lang_code, voice = detect_kokoro_target(text)
    pipeline = get_tts_pipeline(kokoro_lang_code)
    for _, _, audio in pipeline(text, voice=voice):
        if audio is None:
            continue
        pcm = to_pcm16_bytes(audio)
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

    history = job_input.get("history", [])
    instructions = job_input.get("instructions")

    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            user_text, detected_language = transcribe(tmp.name)

        yield {"type": "transcript", "transcript": user_text, "detected_language": detected_language}
        if not user_text:
            yield {"type": "done"}
            return

        spoken_any = False
        last_sentence = None
        first = True
        for sentence in stream_sentences(llm_deltas(history, user_text, system_prompt=instructions)):
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
