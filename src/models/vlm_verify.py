# src/models/vlm_verify.py
"""
Qwen2.5-VL structured visual verification layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image as PILImage


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_vlm(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    load_4bit: bool = True,
):
    """
    Load Qwen2.5-VL and its processor.

    Returns (model, processor). Falls back to fp16 if 4-bit fails.
    """
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = ModelClass.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            return model, processor
        except Exception as e:
            print(f"4-bit load failed ({e}). Falling back to fp16.")

    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


VLM_SYSTEM_PROMPT = (
    "You are a plant-pathology visual QA auditor. You will be shown a leaf image and a "
    "deep-learning-computed lesion-area severity percentage. Judge ONLY whether that number "
    "is visually plausible for the leaf shown. Return STRICT JSON only, no prose, with keys: "
    '{"visual_severity_bracket": one of ["healthy(0-2%)","low(2-10%)","moderate(10-25%)",'
    '"high(25-50%)","severe(>50%)"], '
    '"agrees_with_dl_score": true|false, '
    '"confidence": float 0-1, '
    '"flags": array of short strings from ["lighting_glare","soil_or_background_noise",'
    '"overlapping_leaves","boundary_occlusion","blur","none"]}'
)


def vlm_verify(
    image: Union[str, Path, PILImage.Image],
    dl_severity_pct: float,
    model,
    processor,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Ask the VLM to critique a DL severity score.

    Returns a parsed dict with keys:
      visual_severity_bracket, agrees_with_dl_score, confidence, flags
    or {"error": ..., "raw": ...} on failure.
    """
    from qwen_vl_utils import process_vision_info

    if isinstance(image, (str, Path)):
        image_path = str(image)
        pil_image = None
    else:
        image_path = None
        pil_image = image

    content = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    else:
        content.append({"type": "image", "image": pil_image})

    content.append(
        {
            "type": "text",
            "text": f"DL-computed severity: {dl_severity_pct:.2f}%. Return the JSON verdict.",
        }
    )

    messages = [
        {"role": "system", "content": VLM_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    parsed = _extract_json(raw)
    if parsed is None:
        return {"error": "unparsable", "raw": raw}
    return parsed
