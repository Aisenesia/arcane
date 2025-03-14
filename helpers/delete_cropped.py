import os

def delete_cropped_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'cropped' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == "__main__":
    base_directory = 'datasets/train/images'
    for i in range(1, 21):
        directory = os.path.join(base_directory, str(i))
        if os.path.exists(directory):
            print(f"Processing folder: {directory}")
            delete_cropped_images(directory)
        else:
            print(f"Folder does not exist: {directory}")