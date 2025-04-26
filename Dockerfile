# Stage 1: Download models
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget && \
    mkdir /models && \
    wget -v --content-disposition --show-progress -O /models/RealisticVision.safetensors "https://civitai.com/api/download/models/501240?token=89f98b0d1d7c074688fc6958add259af&type=Model&format=SafeTensor&size=pruned&fp=fp16" 2>&1 || { echo "Failed to download RealisticVision"; exit 1; } && \
    wget -v --content-disposition --show-progress -O /models/WaifuReaper.safetensors "https://civitai.com/api/download/models/648218?token=89f98b0d1d7c074688fc6958add259af&type=Model&format=SafeTensor&size=pruned&fp=fp16" 2>&1 || { echo "Failed to download WaifuReaper"; exit 1; } && \
    mkdir /loras && \
    wget -v --content-disposition --show-progress -O /loras/AddDetail.safetensors "https://civitai.com/api/download/models/1506035?token=89f98b0d1d7c074688fc6958add259af&type=Model&format=SafeTensor" 2>&1 || { echo "Failed to download AddDetail"; exit 1; }

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
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

COPY --from=download /models /stable-diffusion-webui/models/Stable-diffusion
COPY --from=download /loras /stable-diffusion-webui/models/Lora

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .
COPY src .

RUN chmod +x /start.sh
CMD /start.sh
