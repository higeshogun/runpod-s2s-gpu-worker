FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip espeak-ng ffmpeg wget git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip3 install --no-cache-dir -r requirements.txt

# Download the quantized Gemma 4 GGUF model at build time so cold starts don't re-download it
RUN wget -O /app/gemma-4-E4B-it-Q4_0.gguf \
    https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
