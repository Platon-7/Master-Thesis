#!/bin/bash
# Setup Ctrl-World + PlayWorld checkpoint for failure generation
# Run this once to prepare the environment

set -e

THESIS_DIR="/home/pkarageorgis/DAS-5/Master-Thesis"
CTRLWORLD_DIR="${THESIS_DIR}/Ctrl-World"
SCRATCH="/scratch-shared/pkarageorgis"
PLAYWORLD_DIR="${SCRATCH}/playworld_data"

echo "============================================"
echo "PlayWorld/Ctrl-World Setup"
echo "============================================"

# 1. Clone Ctrl-World repo under Master-Thesis
if [ ! -d "$CTRLWORLD_DIR" ]; then
    echo "Cloning Ctrl-World into ${CTRLWORLD_DIR}..."
    git clone https://github.com/Robert-gyj/Ctrl-World.git "$CTRLWORLD_DIR"
else
    echo "Ctrl-World already cloned at $CTRLWORLD_DIR"
fi

# 2. Create conda environment
if ! conda env list | grep -q "ctrl-world"; then
    echo "Creating ctrl-world conda environment..."
    conda create -n ctrl-world python=3.11 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ctrl-world
    pip install -r "$CTRLWORLD_DIR/requirements.txt"
else
    echo "ctrl-world conda env already exists"
fi

# 3. Download PlayWorld dataset + checkpoint from HuggingFace
if [ ! -f "$PLAYWORLD_DIR/checkpoint-100000.pt" ]; then
    echo "Downloading PlayWorld dataset + checkpoint..."
    mkdir -p "$PLAYWORLD_DIR"
    huggingface-cli download tennyyyin/playworld_dataset_preview \
        --repo-type dataset \
        --local-dir "$PLAYWORLD_DIR" \
        --local-dir-use-symlinks False
else
    echo "PlayWorld data already at $PLAYWORLD_DIR"
fi

# 4. Download pretrained SVD model (required by Ctrl-World, ~8GB)
SVD_DIR="${SCRATCH}/hf_cache/stable-video-diffusion-img2vid"
if [ ! -d "$SVD_DIR" ]; then
    echo "Downloading Stable Video Diffusion..."
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid \
        --local-dir "$SVD_DIR" \
        --local-dir-use-symlinks False
else
    echo "SVD already at $SVD_DIR"
fi

# 5. Download CLIP model (required by Ctrl-World, ~600MB)
CLIP_DIR="${SCRATCH}/hf_cache/clip-vit-base-patch32"
if [ ! -d "$CLIP_DIR" ]; then
    echo "Downloading CLIP..."
    huggingface-cli download openai/clip-vit-base-patch32 \
        --local-dir "$CLIP_DIR" \
        --local-dir-use-symlinks False
else
    echo "CLIP already at $CLIP_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "  Ctrl-World repo: $CTRLWORLD_DIR"
echo "  PlayWorld data:  $PLAYWORLD_DIR"
echo "  SVD model:       $SVD_DIR"
echo "  CLIP model:      $CLIP_DIR"
echo "============================================"
