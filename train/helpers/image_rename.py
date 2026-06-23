import os
import shutil

def rename_files(directory):
    files = [f for f in os.listdir(directory) if f.startswith('frame') and f.endswith('.png')]
    files.sort()  # Sort files to ensure consistent renaming

    # Copy files to temporary names
    temp_files = []
    for index, filename in enumerate(files, start=1):
        temp_name = f"temp_img{index}.png"
        shutil.copy(os.path.join(directory, filename), os.path.join(directory, temp_name))
        temp_files.append(temp_name)

    # Remove original files
    for filename in files:
        os.remove(os.path.join(directory, filename))

    # Rename temporary files to final names
    for index, temp_name in enumerate(temp_files, start=1):
        new_name = f"{directory}_frame{index}.png"
        os.rename(os.path.join(directory, temp_name), os.path.join(directory, new_name))
        print(f"Renamed {temp_name} to {new_name}")

if __name__ == "__main__":
    base_directory = "cropped_dataset3"
    for i in range(0, 20):
        directory = os.path.join(base_directory, str(i))
        if os.path.exists(directory):
            print(f"Processing folder: {directory}")
            rename_files(directory)
        else:
            print(f"Folder does not exist: {directory}")