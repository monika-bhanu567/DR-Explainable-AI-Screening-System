from pathlib import Path
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
    / "train_labels.csv"
)


# Read training labels
df = pd.read_csv(LABEL_FILE)

labels = df["Retinopathy grade"].astype(int)

# Count samples in each class
class_counts = labels.value_counts().sort_index()

print("CLASS DISTRIBUTION")
print("------------------------------")

for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count}")

# Calculate total number of classes
num_classes = 5

# Calculate class weights
total_samples = len(labels)

weights = []

for class_id in range(num_classes):
    count = class_counts.get(class_id, 0)

    if count == 0:
        weight = 0.0
    else:
        weight = total_samples / (num_classes * count)

    weights.append(weight)


class_weights = torch.tensor(
    weights,
    dtype=torch.float32
)

print()
print("CLASS WEIGHTS")
print("------------------------------")

for class_id, weight in enumerate(class_weights):
    print(f"Class {class_id}: {weight:.4f}")

print()
print("Class weights created successfully!")