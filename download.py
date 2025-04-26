import os
import requests

model_dirs = {
    "checkpoint": "/stable-diffusion-webui/models/Stable-diffusion",
    "lora": "/stable-diffusion-webui/models/Lora",
    "vae": "/stable-diffusion-webui/models/VAE",
    "embedding": "/stable-diffusion-webui/embeddings"
}

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        filename = r.url.split("/")[-1].split("?")[0]
        with open(os.path.join(save_path, filename), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {filename} to {save_path}")

def download_models():
    if not os.path.exists("models.txt"):
        print("models.txt not found!")
        return
    with open("models.txt") as f:
        for line in f:
            if "|" not in line:
                continue
            type_, url = line.strip().split("|", 1)
            if type_ in model_dirs:
                download_file(url, model_dirs[type_])

def install_extensions():
    if not os.path.exists("extensions.txt"):
        print("extensions.txt not found!")
        return
    with open("extensions.txt") as f:
        for url in f:
            url = url.strip()
            if url:
                cmd = f"git clone {url} /stable-diffusion-webui/extensions/{url.split('/')[-1]}"
                os.system(cmd)
                print(f"Installed extension from {url}")

if __name__ == "__main__":
    download_models()
    install_extensions()
