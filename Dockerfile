# Stage 1: Download models and extensions
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget bash && \
    mkdir -p /downloads/models/Stable-diffusion /downloads/models/Lora /downloads/extensions

COPY models.txt /downloads/models.txt
COPY extensions.txt /downloads/extensions.txt

# Download models
RUN bash -c 'set -e; \
    if [ -s /downloads/models.txt ]; then \
        while read -r type url; do \
            filename=$(basename "$(wget --content-disposition --spider --server-response "$url" 2>&1 | awk "/Content-Disposition/ {print \$0}" | grep -o -E "filename=.*" | cut -d= -f2- | tr -d "\"")"); \
            if [ -z "$filename" ]; then \
                filename=$(basename "$url"); \
            fi; \
            case "$type" in \
                checkpoint) target_dir="/downloads/models/Stable-diffusion" ;; \
                lora) target_dir="/downloads/models/Lora" ;; \
                vae) target_dir="/downloads/models/VAE" ;; \
                embedding) target_dir="/downloads/embeddings" ;; \
                *) echo "Unknown model type: $type" && exit 1 ;; \
            esac; \
            mkdir -p "$target_dir"; \
            wget -v --content-disposition --show-progress -O "${target_dir}/${filename}" "$url" || { echo "Failed to download $filename"; exit 1; }; \
        done < /downloads/models.txt; \
    fi'

# Download extensions
RUN bash -c 'set -e; \
    if [ -s /downloads/extensions.txt ]; then \
        while read -r exturl; do \
            cd /downloads/extensions && git clone "$exturl"; \
        done < /downloads/extensions.txt; \
    fi'

# Stage 2: Build the final image
FROM python:3.10.14-slim as build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget \
    libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 \
    build-essential python3-dev && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    pip install runpod~=1.7.9 insightface==0.7.3 onnx "onnxruntime-gpu>=1.16.1" opencv-python tqdm && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

COPY --from=download /downloads/models/Stable-diffusion /stable-diffusion-webui/models/Stable-diffusion
COPY --from=download /downloads/models/Lora /stable-diffusion-webui/models/Lora
COPY --from=download /downloads/extensions /stable-diffusion-webui/extensions

COPY test_input.json .
COPY src .

RUN chmod +x /start.sh
CMD /start.sh
