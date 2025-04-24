import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import base64
import json
import os
import shutil
import mimetypes

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

MODEL_DIRS = {
    "checkpoint": "/stable-diffusion-webui/models/Stable-diffusion",
    "lora": "/stable-diffusion-webui/models/Lora",
    "embedding": "/stable-diffusion-webui/embeddings",
    "vae": "/stable-diffusion-webui/models/VAE",
    "controlnet": "/stable-diffusion-webui/extensions/sd-webui-controlnet/models",
    "t2i_adapter": "/stable-diffusion-webui/extensions/sd-webui-controlnet/models"
}

VALID_EXTENSIONS = {'.ckpt', '.safetensors', '.pt', '.bin'}

def wait_for_service(url):
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

def refresh_checkpoints():
    response = automatic_session.post(f"{LOCAL_URL}/refresh-checkpoints", timeout=60)
    if response.status_code != 200:
        print(f"Warning: Failed to refresh checkpoints: {response.text}")
    else:
        print("Checkpoints refreshed successfully")

def get_available_models():
    response = automatic_session.get(f"{LOCAL_URL}/sd-models", timeout=60)
    if response.status_code == 200:
        models = response.json()
        model_names = []
        for model in models:
            # Prefer model_name, fall back to title, remove .safetensors if present
            name = model.get('model_name', model.get('title', ''))
            if name.endswith('.safetensors'):
                name = name[:-11]
            if name:
                model_names.append(name)
        print(f"Available models: {model_names}")
        return model_names
    else:
        raise Exception(f"Failed to fetch models: {response.text}")

def get_available_loras():
    response = automatic_session.get(f"{LOCAL_URL}/loras", timeout=60)
    if response.status_code == 200:
        loras = response.json()
        lora_names = [lora['name'] for lora in loras if lora.get('name')]
        print(f"Available LoRAs: {lora_names}")
        return lora_names
    else:
        raise Exception(f"Failed to fetch LoRAs: {response.text}")

def get_available_embeddings():
    response = automatic_session.get(f"{LOCAL_URL}/embeddings", timeout=60)
    if response.status_code == 200:
        embeddings = response.json()
        embedding_names = list(embeddings.get('loaded', {}).keys())
        print(f"Available embeddings: {embedding_names}")
        return embedding_names
    else:
        raise Exception(f"Failed to fetch embeddings: {response.text}")

def get_available_vaes():
    vae_dir = MODEL_DIRS["vae"]
    if os.path.exists(vae_dir):
        vae_names = [f for f in os.listdir(vae_dir) if f.endswith(('.pt', '.safetensors'))]
        print(f"Available VAEs: {vae_names}")
        return vae_names
    print("No VAEs found")
    return []

def get_available_controlnet_models():
    response = automatic_session.get(f"{LOCAL_URL}/controlnet/model_list", timeout=60)
    if response.status_code == 200:
        models = response.json().get('model_list', [])
        print(f"Available ControlNet models: {models}")
        return models
    else:
        raise Exception(f"Failed to fetch ControlNet models: {response.text}")

def get_available_controlnet_modules():
    response = automatic_session.get(f"{LOCAL_URL}/controlnet/module_list", timeout=60)
    if response.status_code == 200:
        modules = response.json().get('module_list', [])
        print(f"Available ControlNet modules: {modules}")
        return modules
    else:
        raise Exception(f"Failed to fetch ControlNet modules: {response.text}")

def list_models(model_type=None):
    if model_type:
        model_dirs = {model_type: MODEL_DIRS.get(model_type)}
        if not model_dirs[model_type]:
            raise ValueError(f"Invalid model type: {model_type}")
    else:
        model_dirs = MODEL_DIRS

    result = {}
    for m_type, m_dir in model_dirs.items():
        if os.path.exists(m_dir):
            files = [f for f in os.listdir(m_dir) if os.path.splitext(f)[1] in VALID_EXTENSIONS]
            result[m_type] = [
                {
                    "name": f,
                    "size": os.path.getsize(os.path.join(m_dir, f)),
                    "modified": os.path.getmtime(os.path.join(m_dir, f)),
                    "path": os.path.join(m_dir, f)
                } for f in files
            ]
        else:
            result[m_type] = []
    print(f"Model list for {model_type or 'all types'}: {result}")
    return result if not model_type else result[model_type]

def download_model(url, model_type=None, civitai_token=None, custom_dir=None):
    headers = {}
    if civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"

    response = requests.get(url, headers=headers, stream=True, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Failed to download model: {response.status_code} - {response.text}")

    content_disposition = response.headers.get('content-disposition')
    if content_disposition and 'filename=' in content_disposition:
        filename = content_disposition.split('filename=')[1].strip('";')
    else:
        filename = url.split("/")[-1] or "downloaded_model"

    extension = os.path.splitext(filename)[1].lower()
    if extension not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}")

    if not model_type:
        if extension in {'.ckpt', '.safetensors'}:
            model_type = "checkpoint" if "checkpoint" in url.lower() else "lora" if "lora" in url.lower() else "vae"
        elif extension == '.pt':
            model_type = "embedding" if "embedding" in url.lower() else "lora"
        elif extension == '.bin':
            model_type = "controlnet" if "controlnet" in url.lower() else "t2i_adapter"
        else:
            raise ValueError("Cannot infer model type from URL and extension")

    model_dir = custom_dir if custom_dir and os.path.exists(custom_dir) else MODEL_DIRS.get(model_type)
    if not model_dir:
        raise ValueError(f"Invalid model type: {model_type} or invalid custom directory")

    filepath = os.path.join(model_dir, filename)
    os.makedirs(model_dir, exist_ok=True)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Model downloaded to {filepath}")
    return filepath

def delete_model(model_name=None, model_type=None, path=None):
    if path:
        if os.path.exists(path):
            os.remove(path)
            print(f"Model at {path} deleted")
            return {"message": f"Model at {path} deleted"}
        raise ValueError(f"File not found at path: {path}")

    if not model_name or not model_type:
        raise ValueError("model_name and model_type are required unless path is provided")

    model_dir = MODEL_DIRS.get(model_type)
    if not model_dir:
        raise ValueError(f"Invalid model type: {model_type}")
    filepath = os.path.join(model_dir, model_name)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Model {model_name} of type {model_type} deleted")
        return {"message": f"Model {model_name} of type {model_type} deleted"}
    raise ValueError(f"Model {model_name} not found in {model_type} directory")

def construct_prompt(base_prompt, loras, embeddings):
    prompt = base_prompt
    for lora in loras:
        prompt += f" <lora:{lora['name']}:{lora['weight']}>"
    for embedding in embeddings:
        prompt += f" {embedding}"
    print(f"Constructed prompt: {prompt}")
    return prompt

def get_override_settings(model_name, vae_name, extra_override_settings):
    override_settings = {}
    if model_name:
        override_settings["sd_model_checkpoint"] = model_name
    if vae_name:
        override_settings["sd_vae"] = vae_name
    override_settings.update(extra_override_settings)
    print(f"Override settings: {override_settings}")
    return override_settings

def get_alwayson_scripts(controlnet_units, extra_alwayson_scripts):
    alwayson_scripts = {}
    if controlnet_units:
        args = []
        for unit in controlnet_units:
            args.append({
                "input_image": unit.get("input_image"),
                "model": unit.get("model"),
                "module": unit.get("module"),
                "weight": unit.get("weight", 1.0),
                "resize_mode": unit.get("resize_mode", "Just Resize"),
                "control_mode": unit.get("control_mode", "Balanced"),
                "pixel_perfect": unit.get("pixel_perfect", False),
                "mask": unit.get("mask"),
                "lowvram": unit.get("lowvram", False),
                "processor_res": unit.get("processor_res", 64),
                "threshold_a": unit.get("threshold_a", 64),
                "threshold_b": unit.get("threshold_b", 64),
                "guidance_start": unit.get("guidance_start", 0.0),
                "guidance_end": unit.get("guidance_end", 1.0)
            })
        alwayson_scripts["controlnet"] = {"args": args}
    alwayson_scripts.update(extra_alwayson_scripts)
    print(f"Alwayson scripts: {alwayson_scripts}")
    return alwayson_scripts

def run_inference(payload):
    response = automatic_session.post(f"{LOCAL_URL}/txt2img", json=payload, timeout=600)
    if response.status_code != 200:
        raise Exception(f"Failed to generate image: {response.status_code} - {response.text}")
    print("Image generation successful")
    return response.json()

def handler(event):
    input_data = event["input"]
    action = input_data.get("action", "generate_image")

    if action == "list_components":
        return {
            "models": get_available_models(),
            "loras": get_available_loras(),
            "embeddings": get_available_embeddings(),
            "vaes": get_available_vaes(),
            "controlnet_models": get_available_controlnet_models(),
            "controlnet_modules": get_available_controlnet_modules()
        }
    elif action == "list_models":
        model_type = input_data.get("model_type")
        return list_models(model_type)
    elif action == "download_model":
        url = input_data.get("url")
        model_type = input_data.get("model_type")
        civitai_token = input_data.get("civitai_token")
        custom_dir = input_data.get("custom_dir")
        if not url:
            raise ValueError("url is required for download_model action")
        filepath = download_model(url, model_type, civitai_token, custom_dir)
        return {"message": f"Model downloaded to {filepath}"}
    elif action == "delete_model":
        model_name = input_data.get("model_name")
        model_type = input_data.get("model_type")
        path = input_data.get("path")
        return delete_model(model_name, model_type, path)
    elif action == "generate_image":
        prompt = input_data.get("prompt", "")
        negative_prompt = input_data.get("negative_prompt", "")
        model_name = input_data.get("model_name", None)
        loras = input_data.get("loras", [])
        embeddings = input_data.get("embeddings", [])
        vae_name = input_data.get("vae_name", None)
        controlnet_units = input_data.get("controlnet_units", [])
        width = input_data.get("width", 512)
        height = input_data.get("height", 512)
        steps = input_data.get("steps", 20)
        seed = input_data.get("seed", -1)
        sampler_index = input_data.get("sampler_index", "Euler a")
        batch_size = input_data.get("batch_size", 1)
        n_iter = input_data.get("n_iter", 1)
        cfg_scale = input_data.get("cfg_scale", 7.5)
        extra_override_settings = input_data.get("extra_override_settings", {})
        extra_alwayson_scripts = input_data.get("extra_alwayson_scripts", {})

        # Validate inputs
        available_models = get_available_models()
        if model_name and model_name not in available_models:
            raise ValueError(f"Model {model_name} not found. Available models: {available_models}")

        available_loras = get_available_loras()
        for lora in loras:
            if 'name' not in lora or 'weight' not in lora:
                raise ValueError("Each LoRA must have 'name' and 'weight' fields")
            if lora['name'] not in available_loras:
                raise ValueError(f"LoRA {lora['name']} not found. Available LoRAs: {available_loras}")

        available_embeddings = get_available_embeddings()
        for embedding in embeddings:
            if embedding not in available_embeddings:
                raise ValueError(f"Embedding {embedding} not found. Available embeddings: {available_embeddings}")

        available_vaes = get_available_vaes()
        if vae_name and vae_name not in available_vaes:
            raise ValueError(f"VAE {vae_name} not found. Available VAEs: {available_vaes}")

        available_controlnet_models = get_available_controlnet_models()
        available_controlnet_modules = get_available_controlnet_modules()
        for unit in controlnet_units:
            if 'model' not in unit or 'module' not in unit:
                raise ValueError("Each controlnet unit must have 'model' and 'module' fields")
            if unit['model'] not in available_controlnet_models:
                raise ValueError(f"ControlNet model {unit['model']} not found. Available models: {available_controlnet_models}")
            if unit['module'] not in available_controlnet_modules:
                raise ValueError(f"ControlNet module {unit['module']} not found. Available modules: {available_controlnet_modules}")

        # Construct full prompt
        full_prompt = construct_prompt(prompt, loras, embeddings)

        # Get override settings
        override_settings = get_override_settings(model_name, vae_name, extra_override_settings)

        # Get alwayson scripts
        alwayson_scripts = get_alwayson_scripts(controlnet_units, extra_alwayson_scripts)

        # Construct payload
        payload = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "sampler_index": sampler_index,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "cfg_scale": cfg_scale,
            "override_settings": override_settings,
            "alwayson_scripts": alwayson_scripts
        }

        # Make API call
        result = run_inference(payload)
        images = result["images"]
        return {"images": images}
    else:
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    wait_for_service(f"{LOCAL_URL}/sd-models")
    refresh_checkpoints()
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
