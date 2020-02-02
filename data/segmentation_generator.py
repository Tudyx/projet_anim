import io
import os
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow import keras
from skimage.io import imread
from data import pad_to_center
from collections import Counter
from skimage import img_as_float
from skimage.transform import resize


class SegmentationGenerator(keras.utils.Sequence):
    
    def __init__(self, dataset_root_path, augmentation_dict=None, batch_size=32, n_channels=3, shuffle=True):
        """Initialization"""
        self.dataset_root_path = dataset_root_path
        
        self.image_filenames = os.listdir(os.path.join(dataset_root_path, "image"))
        self.label_filenames = os.listdir(os.path.join(dataset_root_path, "label"))
        self.n_images = len(self.image_filenames)
        
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.augmentation_dict = augmentation_dict
        self.shuffle = shuffle
        self.idx = 0
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return self.n_images // self.batch_size

    def __getitem__(self, i):
        """Generate one batch of data"""
        image_batch_filenames = self.image_filenames[i * self.batch_size: (i + 1) * self.batch_size]
        label_batch_filenames = self.label_filenames[i * self.batch_size: (i + 1) * self.batch_size]
        image, label = self.__data_generation(image_batch_filenames, label_batch_filenames)

        return image, label

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        indices = np.arange(0, len(self.image_filenames))
        np.random.shuffle(indices)
        self.image_filenames = [self.image_filenames[index] for index in indices]
        self.label_filenames = [self.label_filenames[index] for index in indices]

    def __data_generation(self, image_batch_filenames, label_batch_filenames):
        """Generates data containing batch_size samples"""
        image_batch = []
        label_batch = []

        for image_filename, label_filename in zip(image_batch_filenames, label_batch_filenames):
            # Compose image path
            image_filepath = os.path.join(self.dataset_root_path, "image", image_filename)
            label_filepath = os.path.join(self.dataset_root_path, "label", label_filename)
            
            # Read image
            image_uint8 = imread(image_filepath)
            image_float = img_as_float(image_uint8)
            
            # Read segmentation labels
            with open(label_filepath, 'rb') as f:
                bytesArray = io.BytesIO(f.read())
            labels = np.array(Image.open(bytesArray))

            # Pad to center
            # image_float = pad_to_center(image_float)
            # labels = pad_to_center(labels)
            image_float = resize(image_float, (200, 200, 3))
            labels = resize(labels, (200, 200, 3))

            image_batch.append(image_float)
            label_batch.append(labels)


        return np.array(image_batch), np.array(label_batch)

    def __next__(self):
        self.idx = (self.idx + 1) % len(self)
        return self.__getitem__(self.idx)

