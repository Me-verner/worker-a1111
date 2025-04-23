# Stage 1: Download public models
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget && \
    mkdir /models && \
    wget -q -O /models/AnimeV1.safetensors https://civitai.com/api/download/models/712448 && \
    wget -q -O /models/CinematicV1.safetensors https://civitai.com/api/download/models/501240 && \
    wget -q -O /models/RealisticV1.safetensors https://civitai.com/api/download/models/1633727 && \
    mkdir /loras && \
    wget -q -O /loras/DetailEnhancer.safetensors https://civitai.com/api/download/models/532451

# Stage 2: Download login-required models
FROM alpine/git:2.43.0 as download_auth

ARG CIVITAI_TOKEN
RUN apk add --no-cache wget && \
    mkdir /models && \
    wget -q -O /models/AnimeV2.safetensors "https://civitai.com/api/download/models/1572570?token=$CIVITAI_TOKEN" && \
    wget -q -O /models/FantasyV1.safetensors "https://civitai.com/api/download/models/547268?token=$CIVITAI_TOKEN" && \
    wget -q -O /models/PhotorealV1.safetensors "https://civitai.com/api/download/models/671503?token=$CIVITAI_TOKEN" && \
    wget -q -O /models/ArtisticV1.safetensors "https://civitai.com/api/download/models/1031794?token=$CIVITAI_TOKEN" && \
    mkdir /loras && \
    wget -q -O /loras/StyleBooster.safetensors "https://civitai.com/api/download/models/151465?token=$CIVITAI_TOKEN"

# Stage 3: Build the final image
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
COPY --from=download_auth /models /stable-diffusion-webui/models/Stable-diffusion
COPY --from=download /loras /stable-diffusion-webui/models/Lora
COPY --from=download_auth /loras /stable-diffusion-webui/models/Lora

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .
COPY src .

RUN chmod +x /start.sh
CMD /start.sh
