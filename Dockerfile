# Stage 1: Download models and extensions list
FROM alpine:latest as download

RUN apk add --no-cache wget curl bash git

WORKDIR /workspace

COPY models.txt models.txt
COPY extensions.txt extensions.txt

# Download models
RUN mkdir -p /models && \
    if [ -s models.txt ]; then \
      echo "\nDownloading models..."; \
      while IFS= read -r url || [ -n "$url" ]; do \
        echo "Downloading model from: $url"; \
        wget --content-disposition --show-progress -P /models "$url" || { echo "\nFailed to download model: $url"; exit 1; }; \
      done < models.txt; \
    else \
      echo "\nmodels.txt is empty, skipping model download."; \
    fi

# Clone extensions
RUN mkdir -p /extensions && \
    if [ -s extensions.txt ]; then \
      echo "\nCloning extensions..."; \
      while IFS= read -r repo || [ -n "$repo" ]; do \
        echo "Cloning extension from: $repo"; \
        git clone "$repo" /extensions/$(basename "$repo") || { echo "\nFailed to clone extension: $repo"; exit 1; }; \
      done < extensions.txt; \
    else \
      echo "\nextensions.txt is empty, skipping extensions cloning."; \
    fi

# Stage 2: Build final image
FROM python:3.10.14-slim as build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget curl \
    libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 \
    build-essential python3-dev && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

# Clone and prepare A1111 repo
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy models and extensions from previous stage
COPY --from=download /models /stable-diffusion-webui/models/Stable-diffusion
COPY --from=download /extensions /stable-diffusion-webui/extensions

# Install Python requirements (your own)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy additional files (your handler, starter script)
COPY src/ src/
COPY test_input.json ./

RUN chmod +x /start.sh

CMD ["/start.sh"]
