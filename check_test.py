import os
from src.config import TEST_DIR, CLASSES

print(f"Test directory: {TEST_DIR}")
for cls in CLASSES:
    cls_path = os.path.join(TEST_DIR, cls)
    if os.path.exists(cls_path):
        n_images = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f"{cls}: {n_images} images")
    else:
        print(f"{cls}: folder NOT FOUND!")