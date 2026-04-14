# RoboMeter Failure Scoring Pipeline
# Requires: Docker + nvidia-container-toolkit, 2x GPU with ≥24 GB VRAM each
#
# Build:
#   docker build -t robometer-pipeline .
#
# Run:
#   docker run --gpus all --rm -it \
#     -e HF_TOKEN=hf_xxx \
#     -v /your/data:/data \
#     robometer-pipeline \
#     bash run.sh jesbu1_oxe_rfm_oxe_bridge_v2

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# vLLM — handles both VLM and LLM inference
RUN pip install vllm==0.4.3

# Other dependencies
RUN pip install \
    transformers>=4.40.0 \
    accelerate \
    huggingface_hub \
    Pillow \
    numpy \
    opencv-python-headless \
    tqdm \
    anthropic

# Copy pipeline scripts
WORKDIR /pipeline
COPY . /pipeline/

# Make scripts executable
RUN chmod +x /pipeline/run.sh

# Default HF cache inside /data (bind-mounted by user)
ENV HF_HOME=/data/hf_cache
ENV TORCH_HOME=/data/torch_cache

ENTRYPOINT ["/bin/bash"]
