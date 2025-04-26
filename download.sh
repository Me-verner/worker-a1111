#!/usr/bin/env bash
set -e

echo "🔽 Downloading Models..."
mkdir -p /stable-diffusion-webui/models/Stable-diffusion
while IFS= read -r url || [ -n "$url" ]; do
    echo "Downloading model: $url"
    wget --content-disposition --show-progress -P /stable-diffusion-webui/models/Stable-diffusion "$url" || { echo "❌ Failed to download model: $url"; exit 1; }
done < /models.txt

echo "🔽 Downloading Extensions..."
mkdir -p /stable-diffusion-webui/extensions
while IFS= read -r url || [ -n "$url" ]; do
    echo "Cloning extension: $url"
    git clone "$url" /stable-diffusion-webui/extensions/$(basename "$url" .git) || { echo "❌ Failed to clone extension: $url"; exit 1; }
done < /extensions.txt
