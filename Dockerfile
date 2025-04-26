# Stage 1: Download models and extensions
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget

WORKDIR /workspace

COPY models.txt extensions.txt . 

# Download models
RUN mkdir -p models && \
    if [ -s models.txt ]; then \
        echo "\nDownloading models..."; \
        while IFS= read -r url || [ -n "$url" ]; do \
            echo "Downloading model from: $url"; \
            wget --content-disposition --show-progress -P models "$url" || echo "Failed to download model: $url"; \
        done < models.txt; \
    else \
        echo "\nmodels.txt is empty, skipping model download."; \
    fi

# Download extensions
RUN mkdir -p extensions && \
    if [ -s extensions.txt ]; then \
        echo "\nDownloading extensions..."; \
        while IFS= read -r url || [ -n "$url" ]; do \
            echo "Cloning extension from: $url"; \
            git clone "$url" extensions/ || echo "Failed to clone extension: $url"; \
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

# Copy downloaded models and extensions
COPY --from=download /workspace/models /stable-diffusion-webui/models/Stable-diffusion
COPY --from=download /workspace/extensions /stable-diffusion-webui/extensions

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY src /src
COPY test_input.json .

RUN chmod +x /src/start.sh
CMD /src/start.sh
