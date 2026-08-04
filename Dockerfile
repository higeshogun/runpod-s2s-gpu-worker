FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip espeak-ng ffmpeg wget git cmake ninja-build build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip3 install --no-cache-dir -r requirements.txt

# Download the quantized Gemma 4 GGUF model at build time so cold starts don't re-download it
RUN wget -O /app/gemma-4-E4B-it-Q4_0.gguf \
    https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf

# Pre-download the faster-whisper STT model and Kokoro TTS weights at build time so cold
# starts never have to hit the Hugging Face Hub over the network (this was previously
# happening on every fresh worker, adding highly variable and sometimes multi-minute delays,
# worsened by unauthenticated-request rate limiting).
ARG STT_MODEL_SIZE=base
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-${STT_MODEL_SIZE}')"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hexgrad/Kokoro-82M')"

# misaki's Japanese G2P uses fugashi/MeCab, which needs the unidic dictionary data
# downloaded separately (the pip package ships without it). Without this, MeCab
# fails to initialize at runtime with "Failed initializing MeCab" and crashes the
# worker on every startup.
RUN python3 -m unidic download

# Pre-warm the Kokoro/misaki G2P pipelines used by this deployment (English + Japanese)
# so the first request in either language doesn't pay pipeline-init latency mid-request.
RUN python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a'); KPipeline(lang_code='j')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
