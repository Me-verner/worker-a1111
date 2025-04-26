import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import os
import json
import subprocess
import shutil
import re

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

# Configure session with retries
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# Directory mappings for model types
directories = {
    "checkpoints": ("/stable-diffusion-webui/models/Stable-diffusion", [".safetensors", ".ckpt"]),
    "loras": ("/stable-diffusion-webui/models/Lora", [".safetensors", ".pt"]),
    "vaes": ("/stable-diffusion-webui/models/VAE", [".safetensors", ".pt"]),
    "embeddings": ("/stable-diffusion-webui/embeddings", [".pt", ".bin", ".safetensors"]),
}

# Refresh endpoints for each model type
refresh_endpoints = {
    "checkpoints": "refresh-checkpoints",
    "loras": "refresh-loras",
    "vaes": "refresh-vae",
    "embeddings": "refresh-embeddings"
}

# Paths to models.txt and extensions.txt
MODELS_FILE = "/app/src/models.txt"
EXTENSIONS_FILE = "/app/src/extensions.txt"

def format_size(size):
    """Convert file size to human-readable format."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"

def wait_for_service(url):
    """Wait for the WebUI service to be ready."""
    retries = 0
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)

def set_model(model_name):
    """Set the checkpoint model by its filename, stripping extension for title."""
    model_dir = directories["checkpoints"][0]
    model_path = os.path.join(model_dir, model_name)
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file {model_name} not found at {model_path}")

    model_title = os.path.splitext(model_name)[0]
    payload = {"sd_model_checkpoint": model_title}
    response = automatic_session.post(f"{LOCAL_URL}/options", json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to set model: {response.text}")

def refresh_model_type(model_type):
    """Refresh the model list for a specific type."""
    if model_type not in refresh_endpoints:
        raise ValueError(f"Invalid model type for refresh: {model_type}")
    endpoint = refresh_endpoints[model_type]
    response = automatic_session.post(f"{LOCAL_URL}/{endpoint}", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to refresh {model_type}: {response.text}")
    print(f"{model_type.capitalize()} refreshed successfully.")

def server_restart():
    """Restart the WebUI server."""
    response = automatic_session.post(f"{LOCAL_URL}/server-restart", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to restart server: {response.text}")
    return {"status": "server restarted"}

def install_all():
    """Install all models and extensions from models.txt and extensions.txt."""
    install_models_from_file()
    install_extensions_from_file()
    server_restart()
    return {"status": "all installed and server restarted"}

def install_models_from_file():
    """Install models from models.txt."""
    if not os.path.exists(MODELS_FILE):
        raise ValueError(f"{MODELS_FILE} not found")

    with open(MODELS_FILE, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line:
            model_type, url = line.split("|", 1)
            model_type = model_type.strip().lower()
            if model_type in directories:
                download_model({"type": model_type, "url": url})
            else:
                raise ValueError(f"Invalid model type in {MODELS_FILE}: {model_type}")

    return {"status": "models installed from file"}

def install_extensions_from_file():
    """Install extensions from extensions.txt."""
    if not os.path.exists(EXTENSIONS_FILE):
        raise ValueError(f"{EXTENSIONS_FILE} not found")

    with open(EXTENSIONS_FILE, "r") as f:
        lines = f.readlines()

    for line in lines:
        url = line.strip()
        if url:
            install_extension({"url": url})

    return {"status": "extensions installed from file"}

def install_extension(input_data):
    """Install an extension from a Git URL and restart the server."""
    url = input_data["url"]
    extension_name = url.split("/")[-1].replace(".git", "")
    extensions_dir = "/stable-diffusion-webui/extensions"
    target_dir = os.path.join(extensions_dir, extension_name)

    if os.path.exists(target_dir):
        raise ValueError(f"Extension {extension_name} already exists")

    subprocess.run(["git", "clone", url, target_dir], check=True)

    # Restart the server to load the new extension
    server_restart()

    return {"status": "extension installed", "extension": extension_name}

def delete_extension(input_data):
    """Delete an extension and restart the server."""
    extension_name = input_data["extension_name"]
    extensions_dir = "/stable-diffusion-webui/extensions"
    target_dir = os.path.join(extensions_dir, extension_name)

    if not os.path.exists(target_dir):
        raise ValueError(f"Extension {extension_name} not found")

    shutil.rmtree(target_dir)

    # Restart the server to reflect the change
    server_restart()

    return {"status": "extension deleted", "extension": extension_name}

def list_extensions():
    """List all installed extensions."""
    extensions_dir = "/stable-diffusion-webui/extensions"
    if not os.path.exists(extensions_dir):
        return {"extensions": []}

    extensions = [d for d in os.listdir(extensions_dir) if os.path.isdir(os.path.join(extensions_dir, d))]
    return {"extensions": extensions}

def extract_filename(response):
    """Extract the filename from the response headers or URL."""
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        filename_match = re.search(r'filename="?(.+?)"?(;|$)', content_disposition)
        if filename_match:
            return filename_match.group(1)
    url_filename = os.path.basename(response.url)
    return url_filename if url_filename else "downloaded_model.safetensors"

def download_model(input_data):
    """Download a model file from a URL, save it, and refresh the model list."""
    model_type = input_data["type"]
    url = input_data["url"]
    filename = input_data.get("filename")
    token = input_data.get("token")

    if model_type not in directories:
        raise ValueError(f"Invalid model type: {model_type}")

    dir_path, extensions = directories[model_type]

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code} {response.reason}")

    if not filename:
        filename = extract_filename(response)
        if not any(filename.endswith(ext) for ext in extensions):
            raise ValueError(f"Extracted filename {filename} does not have a valid extension for {model_type}")

    target_path = os.path.join(dir_path, filename)

    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Automatically refresh the model list for this type after successful download
    refresh_model_type(model_type)

    size = os.path.getsize(target_path)
    modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(target_path)))
    return {
        "name": filename,
        "type": model_type[:-1],  # Singularize type
        "size": format_size(size),
        "modified": modified,
        "path": target_path
    }

def inference_handler(input_data):
    """Handle image generation with flexible parameters."""
    model_name = input_data.get("model_name")
    if model_name:
        set_model(model_name)

    override_settings = input_data.get("override_settings", {})

    inference_request = {
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt", ""),
        "steps": input_data.get("steps", 20),
        "width": input_data.get("width", 512),
        "height": input_data.get("height", 512),
        "cfg_scale": input_data.get("cfg_scale", 7.5),
        "seed": input_data.get("seed", -1),
        "override_settings": override_settings,
    }

    if input_data.get("enable_hr", False):
        inference_request["enable_hr"] = True
        inference_request["hr_scale"] = input_data.get("hr_scale", 2.0)
        inference_request["hr_upscaler"] = input_data.get("hr_upscaler", "Latent")
        inference_request["hr_second_pass_steps"] = input_data.get("hr_second_pass_steps", 20)
        inference_request["denoising_strength"] = input_data.get("denoising_strength", 0.55)

    response = automatic_session.post(f"{LOCAL_URL}/txt2img", json=inference_request, timeout=600)
    if response.status_code != 200:
        raise Exception(f"Failed to run inference: {response.text}")
    return response.json()

def handler(event):
    """Main handler function to route actions."""
    input_data = event["input"]
    action = input_data.get("action", "inference")

    if action == "inference":
        return inference_handler(input_data)
    elif action == "get_models":
        return get_models()
    elif action == "download_model":
        return download_model(input_data)
    elif action == "install_all":
        return install_all()
    elif action == "install_models":
        return install_models_from_file()
    elif action == "install_extensions":
        return install_extensions_from_file()
    elif action == "install_extension":
        return install_extension(input_data)
    elif action == "delete_extension":
        return delete_extension(input_data)
    elif action == "list_extensions":
        return list_extensions()
    elif action == "server_restart":
        return server_restart()
    else:
        raise ValueError(f"Unknown action: {action}")

def get_models():
    """Placeholder for get_models function (not fully implemented in original)."""
    response = automatic_session.get(f"{LOCAL_URL}/sd-models", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to get models: {response.text}")
    return response.json()

if __name__ == "__main__":
    wait_for_service(url=f"{LOCAL_URL}/sd-models")
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
