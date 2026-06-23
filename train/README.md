# Arcane Dice Detection

Arcane is a Dice detection machine learning model trained using the YOLO framework. It is designed to identify and classify dice faces in images, making it useful for applications such as automated dice counting and game management.

## Features

- Detects and classifies dice faces in real-time
- Trained on a diverse dataset to ensure accuracy
- Easy to integrate with existing projects

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/arcane.git
    cd arcane
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have a camera connected to your system.

2. Run the dice detection script:
    ```sh
    python detect_dice.py
    ```

3. The script will open a window displaying the camera feed with detected dice faces highlighted.

## Directory Structure
