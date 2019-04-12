from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Object that gets a batch of x-rays during training, validation en testing
class XRay_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, imgSize):
        self.image_filenames, self.labels  = image_filenames, labels
        self.batch_size, self.imgSize = batch_size, imgSize

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (self.imgSize, self.imgSize,3))
            for file_name in batch_x]), \
               np.array(batch_y)
