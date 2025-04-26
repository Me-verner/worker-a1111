# Stage 1: Download Stage
FROM python:3.10-slim as download_stage

RUN apt-get update && apt-get install -y git wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY models.txt extensions.txt download.py ./

RUN python3 download.py

# Stage 2: Build final image
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
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

COPY --from=download_stage /workspace/models /stable-diffusion-webui/models
COPY --from=download_stage /workspace/extensions /stable-diffusion-webui/extensions

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /src

RUN chmod +x /src/start.sh
CMD /src/start.sh
