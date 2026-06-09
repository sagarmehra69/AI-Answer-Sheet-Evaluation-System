from datasets import load_dataset
import os

save_path = "data/processed/semantic_datasets/asag2024"

# Create folders automatically
os.makedirs(save_path, exist_ok=True)

dataset = load_dataset("Meyerger/ASAG2024")

dataset.save_to_disk(save_path)

print("Dataset downloaded successfully.")