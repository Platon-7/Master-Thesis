# local_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import uvicorn
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# --- Load Model (Runs ONCE when server starts) ---
print("Loading Qwen3-VL-8B-Instruct (4-bit)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


import os

# Allow overriding the model ID with a local path env var
# On the cluster, we don't have internet access during the run. Download previously offline to scratch
model_id = os.getenv("LOCAL_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
print(f"Loading Model from: {model_id}")

# Use AutoModelForVision2Seq (Safe for Qwen 3)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True  # Required for Qwen 3
)

# Processor also needs trust_remote_code=True
processor = AutoProcessor.from_pretrained(
    model_id, 
    min_pixels=128*28*28, 
    max_pixels=1280*28*28, 
    trust_remote_code=True
)

print("Model Loaded & Ready!")

# --- Data Structures ---
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]

class GenerateRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7

# --- API Endpoint ---
@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        # 1. Convert Pydantic format back to Qwen format
        qwen_messages = []
        for msg in req.messages:
            qwen_messages.append(msg.dict())

        # 2. Processor
        text = processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 3. Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                
                # If temp is 0, use Greedy Decoding (do_sample=False). 
                # Otherwise use sampling.
                # do_sample=False enables Greedy Decoding. The model always picks the statistically
                # most likely word. If you run the same image 100 times, you get the exact same
                # answer 100 times. This is vital for stable training.
                do_sample=(req.temperature > 0), 
                temperature=req.temperature if req.temperature > 0 else None,
                top_p=None if req.temperature == 0 else 0.9,
            )

        # 4. Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("\n========== QWEN OUTPUT START ==========", flush=True)
        print(output_text[0], flush=True)
        print("========== QWEN OUTPUT END ============\n", flush=True)
        # -----------------------------------
        
        return {"text": output_text[0]}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os as _os
    _host = _os.getenv("SERVER_HOST", "0.0.0.0")
    _port = int(_os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(app, host=_host, port=_port)