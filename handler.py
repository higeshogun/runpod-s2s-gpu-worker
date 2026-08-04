import os
import base64
import io
import tempfile

import runpod
from faster_whisper import WhisperModel
from llama_cpp import Llama
import soundfile as sf
import numpy as np
from kokoro import KPipeline

STT_MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "base")
stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")

tts_pipeline = KPipeline(lang_code=os.environ.get("TTS_LANG_CODE", "a"))  # "a" = American English

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

def transcribe(audio_path):
    segments, info = stt_model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language

def call_llm(history, user_text, system_prompt=None):
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}, *history,
                {"role": "user", "content": user_text}]
    result = llm.create_chat_completion(messages=messages, max_tokens=200)
    return result["choices"][0]["message"]["content"].strip()

def synthesize(text, voice="af_heart"):
    chunks = [audio for _, _, audio in tts_pipeline(text, voice=voice)]
    if not chunks:
        return np.zeros(1, dtype=np.float32), 24000
    return np.concatenate(chunks), 24000

def handler(job):
    job_input = job["input"]
    audio_b64 = job_input.get("audio_base64")
    if not audio_b64:
        return {"error": "audio_base64 is required"}

    history = job_input.get("history", [])
    voice = job_input.get("voice", "af_heart")
    instructions = job_input.get("instructions")

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        user_text, detected_language = transcribe(tmp.name)

    if not user_text:
        return {"transcript": "", "response_text": "", "response_audio_base64": ""}

    response_text = call_llm(history, user_text, system_prompt=instructions)
    audio_array, sample_rate = synthesize(response_text, voice=voice)

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
