# Stage 1: Base Image.
FROM python:3.10.14-slim

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install dependencies
RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget build-essential python3-dev libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

# Clone WebUI
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy requirement files and downloader script
COPY requirements.txt /
COPY models.txt /
COPY extensions.txt /
COPY download.sh /

# Install python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /requirements.txt

# Download models and extensions
RUN chmod +x /download.sh && /download.sh

# Copy your scripts
COPY src /src
COPY test_input.json .

RUN chmod +x /src/start.sh

CMD /src/start.sh
