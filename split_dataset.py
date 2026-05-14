import os
import shutil
import random

base_dir = "image_dataset_new"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

classes = ["phishing", "legitimate"]

split_ratio = 0.2  # 20% for validation

for cls in classes:
    class_train_path = os.path.join(train_dir, cls)
    class_val_path = os.path.join(val_dir, cls)

    images = os.listdir(class_train_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    val_images = images[:split_index]

    for img in val_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)

    print(f"{cls}: moved {len(val_images)} images to validation")

print("Dataset split completed.")