import numpy as np

from tensorflow import keras
from skimage.io import imread
from skimage import img_as_float


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        """Initialization
		
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        pass

    def __getitem__(self, index):
        """Generate one batch of data"""
        pass

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        pass

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        pass