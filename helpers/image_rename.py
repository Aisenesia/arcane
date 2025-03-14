import os
import shutil

def rename_files(directory):
    files = [f for f in os.listdir(directory) if f.startswith('img') and f.endswith('.jpg')]
    files.sort()  # Sort files to ensure consistent renaming

    # Copy files to temporary names
    temp_files = []
    for index, filename in enumerate(files, start=1):
        temp_name = f"temp_img{index}.jpg"
        shutil.copy(os.path.join(directory, filename), os.path.join(directory, temp_name))
        temp_files.append(temp_name)

    # Remove original files
    for filename in files:
        os.remove(os.path.join(directory, filename))

    # Rename temporary files to final names
    for index, temp_name in enumerate(temp_files, start=1):
        new_name = f"img{index}.jpg"
        os.rename(os.path.join(directory, temp_name), os.path.join(directory, new_name))
        print(f"Renamed {temp_name} to {new_name}")

if __name__ == "__main__":
    base_directory = "datasets/train/images"
    for i in range(1, 21):
        directory = os.path.join(base_directory, str(i))
        if os.path.exists(directory):
            print(f"Processing folder: {directory}")
            rename_files(directory)
        else:
            print(f"Folder does not exist: {directory}")