"""
    Author: Sami Almnayer
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""

import os
import torch
import numpy as np
from torchvision import transforms
import random
import glob
from PIL import Image
IMAGE_DIMENSION = 100


def create_arrays_from_image(image_array: np.ndarray, offset: tuple, spacing: tuple):
    image_array = np.transpose(image_array, (2, 0, 1))  # (3,H,W)

    mask = np.zeros((1, image_array.shape[1], image_array.shape[2]), dtype=np.float32)
    mask[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1.0

    corrupted = image_array.copy()
    corrupted[mask.repeat(3, axis=0) == 0] = 0.0

    return corrupted, mask



def resize(img: Image):
    resize_transforms = transforms.Compose([
        transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION)),
        transforms.CenterCrop((IMAGE_DIMENSION,IMAGE_DIMENSION))
        # Data augmentation?
    ])
    return resize_transforms(img)

def preprocessing(input_array: np.ndarray):
    # additional normalization?
    return np.array(input_array, dtype=np.float32) / 255.0

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder
    """
    def __init__(self, datafolder: str):
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder,"**","*.jpg"),recursive=True))

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, idx: int):
        index = int(idx)

        image = Image.open(self.imagefiles[index])
        image = np.asarray(resize(image))
        image = preprocessing(image)

        spacing_x = random.randint(2, 6)
        spacing_y = random.randint(2, 6)
        offset_x = random.randint(0, 8)
        offset_y = random.randint(0, 8)

        spacing = (spacing_x, spacing_y)
        offset = (offset_x, offset_y)


        # Wichtig!!!!!!!!
        input_array, known_array = create_arrays_from_image(image.copy(), offset, spacing)
        # creates (3, 100, 100), (1, 100, 100)
        target_image = torch.from_numpy(np.transpose(image, (2, 0, 1))) # makes (3, 100, 100) from (100, 100, 3)

        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(known_array)

        input_array = torch.cat((input_array, known_array), dim=0) # (4, 100, 100)

        return input_array, target_image


    pass