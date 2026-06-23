import os

# Uses whole images as bounding boxes for simplicity
# Assumes folder names are integer IDs starting from 1, they represent the class IDs with 0-based indexing


# Paths to image folders
train_dir = "datasets/train/images"

# Function to create YOLO label files
def create_yolo_labels(image_dir):
    for root, dirs, files in os.walk(image_dir):
        for dir_name in dirs:
            try:
                dir_id = int(dir_name)  # Extract patient ID from folder name
                class_id = dir_id - 1  # Class is folder number minus 1
            except ValueError:
                print(f"Skipping folder with invalid ID: {dir_name}")
                continue

            label_dir = os.path.join(image_dir.replace("images", "labels"), dir_name)
            os.makedirs(label_dir, exist_ok=True)

            for img_file in os.listdir(os.path.join(root, dir_name)):
                if img_file.endswith(".jpg"):  # Adjust if using PNG or other formats
                    label_path = os.path.join(label_dir, f"{os.path.splitext(img_file)[0]}.txt")

                    print(f"Processing image: {img_file}")
                    print(f"Class ID: {class_id}")
                    print(f"Label directory: {label_dir}")
                    print(f"Label path: {label_path}")
                    with open(label_path, "w") as f:
                        # Assuming a full-image bounding box for simplicity
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Create label files for both train and test
create_yolo_labels(train_dir)
# create_yolo_labels(test_dir)

print("YOLO label files creation process completed!")