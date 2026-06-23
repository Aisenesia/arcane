import os

def modify_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            parts[0] = '0'
            file.write(' '.join(parts) + '\n')

def modify_labels_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                modify_label_file(file_path)

if __name__ == '__main__':
    base_dir = 'c:/Users/Hakan/cv/arcane-2/datasets/labels'
    for sub_dir in ['train', 'test', 'val']:
        modify_labels_in_directory(os.path.join(base_dir, sub_dir))