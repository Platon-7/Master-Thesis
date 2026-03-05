import os
import pickle
from datasets import load_dataset, Value

# --- CONFIGURATION ---
OUTPUT_DIR = "/var/scratch/pkarageo/roboreward_dataset"
NUM_SAMPLES_TO_SAVE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Initializing dataset stream...")
dataset = load_dataset("teetone/RoboReward", split="train", streaming=True)

# --- THE FIX: CAST VIDEO COLUMN TO STRING ---
# We force the 'video' column (or 'mp4') to be treated as a simple string (URL/Path).
# This completely bypasses the need for torchcodec/decord/av.
print("Forcing video column to be text-only (bypassing decoding)...")

# We cast BOTH common keys just to be safe. It won't hurt if they don't exist.
features_to_cast = {}
if "video" in dataset.features:
    features_to_cast["video"] = Value("string")
if "mp4" in dataset.features:
    features_to_cast["mp4"] = Value("string")

if features_to_cast:
    dataset = dataset.cast_column("video", Value("string")) if "video" in features_to_cast else dataset
    dataset = dataset.cast_column("mp4", Value("string")) if "mp4" in features_to_cast else dataset
    print("-> Successfully cast video columns to string.")
else:
    print("-> No video column found (keys might be different).")

# -------------------------------------------

print(f"Fetching {NUM_SAMPLES_TO_SAVE} samples...")
# We use iter() now that the schema is safe
dataset_iter = iter(dataset)

for i in range(NUM_SAMPLES_TO_SAVE):
    try:
        example = next(dataset_iter)
        
        if i == 0:
            print("\n--- SAMPLE 0 INSPECTION ---")
            print(f"Keys: {list(example.keys())}")
            print(f"Task: {example.get('task', 'N/A')}")
            # This will now print a path/url string instead of crashing!
            print(f"Video Data: {str(example.get('video', 'N/A'))[:50]}...") 
            print("---------------------------\n")

        # Save to disk
        filename = os.path.join(OUTPUT_DIR, f"sample_{i}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(example, f)
        
        print(f"Saved sample {i}")

    except StopIteration:
        print("End of dataset stream.")
        break

print(f"\nDone! Check {OUTPUT_DIR}")