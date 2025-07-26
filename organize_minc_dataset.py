import os
import shutil
import random

# Paths
base_dir = 'C:/Users/acer/Documents/yollo/minc-2500/materials_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Create train and val folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Parameters
val_split = 0.2  # 20% for validation

# Loop through each class
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path) or class_name in ['train', 'val']:
        continue

    # Get image list and shuffle
    images = os.listdir(class_path)
    random.shuffle(images)

    val_size = int(len(images) * val_split)
    val_images = images[:val_size]
    train_images = images[val_size:]

    # Create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move files
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("✅ Done splitting into train and val sets.")

