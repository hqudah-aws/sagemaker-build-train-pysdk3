import json
import os
import io
import base64
import torch
from djl_python import Input, Output

_pipeline = None


def handle(inputs: Input) -> Output:
    """
    DJL Serving handler for Stable Diffusion 3.5 Medium with LoRA weights.

    Expects JSON input:
    {
        "prompt": "a boy Malcom and his dog Ben",
        "num_inference_steps": 30,       // optional, default 30
        "guidance_scale": 7.5,           // optional, default 7.5
        "height": 512,                   // optional, default 512
        "width": 512,                    // optional, default 512
        "seed": 42                       // optional, for reproducibility
    }

    Returns JSON with base64-encoded PNG image.
    """
    global _pipeline

    # ── Model loading (called once at startup) ──────────────────────────────
    if inputs.is_empty():
        properties = inputs.get_properties()
        model_dir = properties.get("model_dir", "/opt/ml/model")

        from diffusers import StableDiffusion3Pipeline

        # Load the base SD 3.5 Medium pipeline
        base_model_id = "stabilityai/stable-diffusion-3.5-medium"
        _pipeline = StableDiffusion3Pipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
        )

        # Load LoRA weights from the model artifact
        lora_weights_path = model_dir
        _pipeline.load_lora_weights(lora_weights_path)

        _pipeline.to("cuda")
        _pipeline.set_progress_bar_config(disable=True)

        return None

    # ── Inference ───────────────────────────────────────────────────────────
    content_type = inputs.get_property("Content-Type") or "application/json"
    request_body = inputs.get_as_string()

    if content_type == "application/json":
        params = json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    prompt = params.get("prompt", "")
    num_inference_steps = params.get("num_inference_steps", 30)
    guidance_scale = params.get("guidance_scale", 7.5)
    height = params.get("height", 512)
    width = params.get("width", 512)
    seed = params.get("seed", None)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        result = _pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

    image = result.images[0]

    # Encode image as base64 PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    output = Output()
    output.add_as_json({
        "generated_image": img_b64,
        "prompt": prompt,
        "parameters": {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "seed": seed,
        },
    })
    return output
