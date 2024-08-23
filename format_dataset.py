import os
import random
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

# Define paths
image_dir_in = 'datasets/trash_detection'
annotation_dir_in = 'datasets/trash_detection/labels'
output_dir = 'datasets/trash_detection_split'

# Define YOLO folder structure
image_dir_out = os.path.join(output_dir, 'images')
labels_dir_out = os.path.join(output_dir, 'labels')

# Create YOLO folder structure if not already present
for dir_path in [image_dir_out, labels_dir_out]:
    os.makedirs(os.path.join(dir_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'test'), exist_ok=True)

# Collect all images and corresponding annotation files
image_files = []
annotation_files = []

for root, _, files in os.walk(image_dir_in):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, file))
            annotation_files.append(os.path.join(annotation_dir_in, os.path.splitext(file)[0] + '.txt'))

# Ensure the number of image files matches the number of annotation files
assert len(image_files) == len(annotation_files), "Mismatch between image and annotation files count."

print(f"Parsed {len(image_files)} images and annotations.")

# Build a class distribution for stratified splitting
class_distribution = defaultdict(list)

# print(image_files[:10])
# print(annotation_files[:10])

for img, ann in zip(image_files, annotation_files):
    with open(ann, 'r') as f:
        classes = set(line.split()[0] for line in f.readlines() if line.strip())
    for cls in classes:
        class_distribution[cls].append((img, ann))

# Prepare stratified data split
train_files = []
val_files = []
test_files = []

print(f"Length of class distribution dict: {len(class_distribution)}")
for i in range(len(class_distribution)):
    print(f"For class {i} there are {len(class_distribution[str(i)])} instances")

for cls, files in class_distribution.items():
    train, temp = train_test_split(files, test_size=0.3, random_state=SEED) # 70% for train
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED) # 15% and 15% for val and test
    train_files.extend(train)
    val_files.extend(val)
    test_files.extend(test)

# Helper function to copy files to their respective directories
def copy_files(file_list, subset_dir):
    for img_path, ann_path in file_list:
        shutil.copy(img_path, os.path.join(image_dir_out, subset_dir, os.path.basename(img_path)))
        shutil.copy(ann_path, os.path.join(labels_dir_out, subset_dir, os.path.basename(ann_path)))

# Copy files to train, val, and test directories
print("Copying images...")
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")
print("Finished!")

print("Dataset split and files copied successfully!")
