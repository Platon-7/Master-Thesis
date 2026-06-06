import warnings

import torch
from qwen_vl_utils import process_vision_info


def _load_qwen3_vl(model_id):
    """Load a Qwen3-VL model. Prefers flash_attention_2; falls back if unavailable."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    except (ImportError, ValueError) as e:
        warnings.warn(
            f"flash_attention_2 unavailable for {model_id} ({e}); "
            "falling back to default attention. Install flash-attn for full throughput."
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    return model, processor


def get_qwen3():
    return _load_qwen3_vl("Qwen/Qwen3-VL-32B-Instruct")


def get_qwen3_8b():
    return _load_qwen3_vl("Qwen/Qwen3-VL-8B-Instruct")


def prompt_qwen(model, processor, messages, prompt_kwargs=None, debug=True):

    if prompt_kwargs is None:
        prompt_kwargs = dict(max_new_tokens=200, do_sample=True, top_p=0.9, top_k=50, temperature=0.7)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    # Qwen processor is happier with None than empty lists
    image_inputs = image_inputs or None
    video_inputs = video_inputs or None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    eos = processor.tokenizer.eos_token_id
    pad = processor.tokenizer.pad_token_id
    if eos is None:
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if pad is None:
        pad = eos
    gen_kwargs = dict(
        eos_token_id=eos,
        pad_token_id=pad,
        use_cache=True,
    )
    gen_kwargs.update(prompt_kwargs)

    with torch.inference_mode():
        generation = model.generate(**inputs, **gen_kwargs)
    raw_output = processor.tokenizer.decode(generation[0], skip_special_tokens=True)

    if debug:
        print("RAW OUPUT:")
        print(raw_output)
        print("---")

    split_token = "assistant"
    if split_token in raw_output:
        output_text = raw_output.split(split_token, 1)[-1].strip()
    else:
        output_text = raw_output.strip()
    return output_text
