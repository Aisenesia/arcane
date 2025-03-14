import os

# Changes the class IDs in the label files to be zero-based

# Define the path to the labels directory
labels_path = 'datasets/train/labels/'

# Function to update labels in a file
def update_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            parts[0] = str(int(parts[0]) - 1)
            file.write(' '.join(parts) + '\n')

# Iterate through all folders and files in the labels directory
for class_dir in os.listdir(labels_path):
    class_dir_path = os.path.join(labels_path, class_dir)
    if os.path.isdir(class_dir_path):
        for label_file in os.listdir(class_dir_path):
            label_file_path = os.path.join(class_dir_path, label_file)
            update_label_file(label_file_path)

print("Labels updated successfully.")
