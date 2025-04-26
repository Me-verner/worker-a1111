#!/bin/bash
set -e

echo "🔽 Downloading Models..."
mkdir -p /stable-diffusion-webui/models/Stable-diffusion
if [ -s /models.txt ]; then
    while IFS= read -r url || [ -n "$url" ]; do
        echo "Downloading model: $url"
        wget --content-disposition --show-progress -P /stable-diffusion-webui/models/Stable-diffusion "$url" || { echo "❌ Failed to download model: $url"; exit 1; }
    done < /models.txt
else
    echo "ℹ️ No models to download."
fi

echo "🔽 Downloading Extensions..."
mkdir -p /stable-diffusion-webui/extensions
if [ -s /extensions.txt ]; then
    while IFS= read -r url || [ -n "$url" ]; do
        echo "Cloning extension: $url"
        git clone "$url" "/stable-diffusion-webui/extensions/$(basename "$url" .git)" || { echo "❌ Failed to clone extension: $url"; exit 1; }
    done < /extensions.txt
else
    echo "ℹ️ No extensions to download."
fi

echo "✅ Downloads finished!"
