import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import os
import json
import time as time_module
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
    "embeddings": ("/stable-diffusion-webui/embeddings", [".pt", ".bin"]),
}

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

def get_models():
    """List all available models with details."""
    result = {}
    for model_type, (dir_path, extensions) in directories.items():
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(tuple(extensions)) and os.path.isfile(os.path.join(dir_path, f))]
            file_list = []
            for file in files:
                path = os.path.join(dir_path, file)
                size = os.path.getsize(path)
                modified = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime(os.path.getmtime(path)))
                file_list.append({
                    "name": file,
                    "type": model_type[:-1],  # Singularize type (e.g., "checkpoints" -> "checkpoint")
                    "size": format_size(size),
                    "modified": modified,
                    "path": path
                })
            result[model_type] = {
                "count": len(files),
                "files": file_list
            }
        else:
            result[model_type] = {
                "count": 0,
                "files": []
            }
    return result

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
    """Download a model file from a URL and save it to the appropriate directory."""
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

    size = os.path.getsize(target_path)
    modified = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime(os.path.getmtime(target_path)))
    return {
        "name": filename,
        "type": model_type[:-1],  # Singularize type
        "size": format_size(size),
        "modified": modified,
        "path": target_path
    }

def rename_model(input_data):
    """Rename a specified model file."""
    model_type = input_data["type"]
    old_filename = input_data["old_filename"]
    new_filename = input_data["new_filename"]

    if model_type not in directories:
        raise ValueError(f"Invalid model type: {model_type}")

    dir_path, extensions = directories[model_type]
    old_path = os.path.join(dir_path, old_filename)
    new_path = os.path.join(dir_path, new_filename)

    if not os.path.exists(old_path):
        raise ValueError(f"File {old_filename} not found in {dir_path}")

    if not any(new_filename.endswith(ext) for ext in extensions):
        raise ValueError(f"New filename {new_filename} does not have a valid extension for {model_type}")

    os.rename(old_path, new_path)
    return {"status": "renamed", "old_filename": old_filename, "new_filename": new_filename}

def delete_model(input_data):
    """Delete a specified model file."""
    model_type = input_data["type"]
    filename = input_data["filename"]

    if model_type not in directories:
        raise ValueError(f"Invalid model type: {model_type}")

    dir_path, _ = directories[model_type]
    target_path = os.path.join(dir_path, filename)

    if not os.path.exists(target_path):
        raise ValueError(f"File {filename} not found in {dir_path}")

    os.remove(target_path)
    return {"status": "deleted", "filename": filename}

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
    elif action == "rename_model":
        return rename_model(input_data)
    elif action == "delete_model":
        return delete_model(input_data)
    else:
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    wait_for_service(url=f"{LOCAL_URL}/sd-models")
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
