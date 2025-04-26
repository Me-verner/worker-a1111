# Stage 1: Download base
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget

# Stage 2: Final Build
FROM python:3.10.14-slim as build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System Packages
RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget \
    build-essential python3-dev libgoogle-perftools-dev libtcmalloc-minimal4 \
    procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && apt-get clean -y

# Clone WebUI
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy Files
COPY requirements.txt .
COPY download.py .
COPY models.txt .
COPY extensions.txt .
COPY test_input.json .
COPY src .
COPY start.sh .

RUN chmod +x /start.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

CMD /start.sh
