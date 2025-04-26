#!/usr/bin/env python3

import os
import requests
import shutil

def download(url, dest_folder):
    local_filename = url.split("/")[-1].split("?")[0]
    local_path = os.path.join(dest_folder, local_filename)
    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            raise Exception(f"❌ Download failed: {url} | Status: {r.status_code}")
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print(f"✅ Downloaded: {local_filename} to {dest_folder}")

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Paths
model_base = "/stable-diffusion-webui/models"
extensions_base = "/stable-diffusion-webui/extensions"

# Read models.txt
if os.path.isfile("models.txt"):
    print("\n🔽 Downloading Models...")
    with open("models.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"⚠️ Invalid line in models.txt: {line}")
                continue
            model_type, url = parts
            if model_type == "checkpoint":
                dest = os.path.join(model_base, "Stable-diffusion")
            elif model_type == "lora":
                dest = os.path.join(model_base, "Lora")
            elif model_type == "vae":
                dest = os.path.join(model_base, "VAE")
            elif model_type == "embedding":
                dest = os.path.join(model_base, "embeddings")
            else:
                print(f"⚠️ Unknown model type: {model_type}")
                continue
            ensure_folder(dest)
            download(url, dest)
else:
    print("\n⚠️ models.txt not found, skipping model download.")

# Read extensions.txt
if os.path.isfile("extensions.txt"):
    print("\n🔽 Downloading Extensions...")
    with open("extensions.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            repo_url = line
            repo_name = repo_url.rstrip("/").split("/")[-1]
            dest = os.path.join(extensions_base, repo_name)
            ensure_folder(extensions_base)
            os.system(f"git clone --depth=1 {repo_url} {dest}")
            print(f"✅ Cloned: {repo_name}")
else:
    print("\n⚠️ extensions.txt not found, skipping extension download.")
