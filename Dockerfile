# Stage 1: Download models and extensions
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget bash

WORKDIR /workspace

COPY models.txt .
COPY extensions.txt .

RUN mkdir -p /models && \
    if [ -s models.txt ]; then \
        echo "\nDownloading models..."; \
        while IFS= read -r url || [ -n "$url" ]; do \
            if [ -n "$url" ]; then \
                echo "\nDownloading model from: $url"; \
                wget --content-disposition --show-progress -P /models "$url" || echo "Failed to download model: $url (skipping)"; \
            fi; \
        done < models.txt; \
    else \
        echo "\nmodels.txt is empty, skipping model download."; \
    fi

RUN mkdir -p /extensions && \
    if [ -s extensions.txt ]; then \
        echo "\nDownloading extensions..."; \
        while IFS= read -r repo || [ -n "$repo" ]; do \
            if [ -n "$repo" ]; then \
                echo "\nCloning extension from: $repo"; \
                git clone "$repo" /extensions/$(basename "$repo") || echo "Failed to clone extension: $repo (skipping)"; \
            fi; \
        done < extensions.txt; \
    else \
        echo "\nextensions.txt is empty, skipping extensions download."; \
    fi

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
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy downloaded models
COPY --from=download /models /stable-diffusion-webui/models/Stable-diffusion

# Copy downloaded extensions
COPY --from=download /extensions /stable-diffusion-webui/extensions

# Copy app files
COPY src /src
COPY requirements.txt .
COPY test_input.json .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

RUN chmod +x /src/start.sh
CMD /src/start.sh
