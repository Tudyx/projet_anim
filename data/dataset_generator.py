import os
import numpy as np
import pandas as pd

from tensorflow import keras
from skimage.io import imread
from skimage import img_as_float
from collections import Counter
from .utils import crop_center


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, dataset_root_path, batch_size=32, n_channels=3, shuffle=True):
        """Initialization"""
        self.dataset_root_path = dataset_root_path
        csv_file = pd.read_csv(os.path.join(self.dataset_root_path, "results.csv"))
        
        self.filenames = csv_file.values[:, 1]
        self.n_images = len(self.filenames)
        self.labels = {
            filename: label for filename, label in zip(csv_file.values[:, 1], csv_file.values[:, -1])
        }
        self.nucleus_positions = {
            filename: (int(cx), int(cy))
            for filename, cx, cy in zip(csv_file.values[:, 1], csv_file.values[:, 2], csv_file.values[:, 3])
        }
        
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = len(Counter(self.labels.values()))
        self.shuffle = shuffle
        self.idx = 0
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return self.n_images // self.batch_size

    def __getitem__(self, i):
        """Generate one batch of data"""
        batch_filenames = self.filenames[i * self.batch_size: (i + 1) * self.batch_size]
        image, label = self.__data_generation(batch_filenames)

        return image, label

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        np.random.shuffle(self.filenames)

    def __data_generation(self, batch_filenames):
        """Generates data containing batch_size samples"""
        image_batch = []
        label_batch = [self.labels[filename] for filename in batch_filenames]

        for filename in batch_filenames:
            # Get center coordinates
            cx, cy = self.nucleus_positions[filename]

            # Compose image path
            image_filepath = os.path.join(self.dataset_root_path, filename)
            
            # Read image
            image_uint8 = imread(image_filepath)
            image_float = img_as_float(image_uint8)

            # Apply preprocessing functions
            image_float = crop_center(image_float, cx, cy)

            # Append to batch
            image_batch.append(image_float)

        return np.array(image_batch), label_batch

    def __next__(self):
        self.idx = (self.idx + 1) % len(self)
        return self.__getitem__(self.idx)
