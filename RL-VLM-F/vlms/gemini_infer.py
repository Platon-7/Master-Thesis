import requests
import base64
import time
from PIL import Image
from io import BytesIO
import numpy as np
import re
import os

# Build server URL from env for concurrent runs
_server_url = os.getenv("SERVER_URL")
if _server_url is None:
    _host = os.getenv("SERVER_HOST", "localhost")
    _port = os.getenv("SERVER_PORT", "8000")
    _server_url = f"http://{_host}:{_port}/generate"
SERVER_URL = _server_url

# Take the pixels → Save them as a PNG image in memory → Converts that image into a massive string
# of random-looking characters (Base64).
def _encode_image(image):
    """Helper to convert numpy or PIL images to base64 strings"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')


# Converts the lazy list from the RL codebase into the strict dictionary format the server expects.
def _build_payload(query_list):
    """Converts the mixed [text, image, text] list into the format the server expects"""
    content = []
    for item in query_list:
        if isinstance(item, str):
            content.append({"type": "text", "text": item})
        else:
            # It's an image (PIL or numpy)
            base64_str = _encode_image(item)
            content.append({"type": "image", "image": base64_str})
    
    return [{"role": "user", "content": content}]

def gemini_query_1(query_list, temperature=0):
    """
    Called by the codebase to get a direct answer (e.g. Reward).
    """
    try:
        start = time.time()
        messages = _build_payload(query_list)

        # We modify the last text part of the prompt to ask for brevity.
        # This stops Qwen from writing 300+ tokens.
        if isinstance(messages[0]['content'][-1], dict) and messages[0]['content'][-1].get('type') == 'text':
             messages[0]['content'][-1]['text'] += "\n(Answer concisely in 2 sentences.)"
        
        # Send request to your local server
        response = requests.post(SERVER_URL, json={
            "messages": messages,
            "max_tokens": 512, # cut in half for speed, open to reduce more later
            "temperature": temperature
        })
        
        # Qwen likes to think before answering, like <think>I see a robot arm...</think> Answer: 0.
        # This line finds anything starting with <think> and ending with </think> and deletes it.
        if response.status_code == 200:
            text = response.json()['text']
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            print(f"Time: {time.time()-start:.2f}s")
            return text.strip()
        else:
            print(f"Server Error: {response.text}")
            return -1
    except Exception as e:
        print(f"Connection Failed: {e}")
        return -1

def gemini_query_2(query_list, summary_prompt, temperature=0):
    """
    Called when the codebase wants a description + a summary.
    """
    # 1. Get Vision Description
    vision_text = gemini_query_1(query_list, temperature)
    if vision_text == -1: return -1

    # 2. Get Summary (Text-only call)
    try:
        # We format the summary request as a text-only prompt
        messages = [{"role": "user", "content": [{"type": "text", "text": summary_prompt.format(vision_text)}]}]
        
        response = requests.post(SERVER_URL, json={
            "messages": messages,
            "max_tokens": 16, # STRICT LIMIT (We only need a digit) ---
            "temperature": temperature
        })
        
        if response.status_code == 200:
            raw_text = response.json()['text']

            # A. Remove <think> tags if Qwen 3 uses them
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)

            # B. Regex Search: Prefer explicit '-1', then '0' or '1'
            # Use robust token detection so '-1' is not misread as '1'.
            m_neg1 = re.search(r'(?<!\d)-1(?!\d)', clean_text)
            if m_neg1:
                return "-1"
            m_zero = re.search(r'\b0\b', clean_text)
            if m_zero:
                return "0"
            m_one = re.search(r'\b1\b', clean_text)
            if m_one:
                return "1"

            # C. Fallback: First line trimming + keyword heuristics
            first_line = clean_text.strip().split('\n')[0]
            # Treat common uncertainty phrases as -1
            uncertain_phrases = [
                "uncertain", "unsure", "not sure", "can't tell", "cannot tell",
                "no difference", "equal", "tie", "both", "neither"
            ]
            if any(p in first_line.lower() for p in uncertain_phrases):
                return "-1"
            if "-1" in first_line: return "-1"
            if "0" in first_line: return "0"
            if "1" in first_line: return "1"

            return first_line 
        else:
            return -1
    except:
        return -1