"""Utilities for real-time data augmentation on image data.
Refer to the following files
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/numpy_array_iterator.py
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/iterator.py
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/utils.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import warnings
import keras.backend as K
from PIL import Image
from keras.preprocessing.image import Iterator, img_to_array
from Augmentor import Pipeline

class ImageIterator(Iterator):
    """Iterator yielding data from image file paths.
    """

    def __init__(self,
                 image_paths,
                 labels=None,
                 augmentation_pipeline=None,
                 batch_size=64,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 rescale=None,
                 pregen_augmented_images=False,
                 preprocessing_function=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 dtype='float32'):

        self.image_paths = image_paths
        self.rescale = rescale
        self.pregen_augmented_images = pregen_augmented_images
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype

        if labels is not None and len(image_paths) != len(labels):
            raise ValueError('`image_paths` and `labels` '
                             'should have the same length. '
                             'Found: len(image_paths) = %s, len(labels) = %s' %
                             (len(image_paths), len(labels)))

        if sample_weight is not None and len(image_paths) != len(sample_weight):
            raise ValueError('`image_paths` and `sample_weight` '
                             'should have the same length. '
                             'Found: x.shape = %s, sample_weight.shape = %s' %
                             (len(image_paths), len(sample_weight)))

        if labels is not None:
            self.labels = np.asarray(labels)
        else:
            self.labels = None

        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight)
        else:
            self.sample_weight = None

        self.augmentation_pipeline = augmentation_pipeline
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format
        self.save_to_dir = save_to_dir
        if save_to_dir is not None and not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        self.save_prefix = save_prefix
        self.save_format = save_format

        if self.pregen_augmented_images:
            self.augmented_images = self._generate_augmented_images()

        super(ImageIterator, self).__init__(len(image_paths), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = [None] * len(index_array)

        if self.pregen_augmented_images:
            # Use augmented images directly
            for i, j in enumerate(index_array):
                batch_x[i] = self.augmented_images[j]
        else:
            for i, j in enumerate(index_array):
                x = Image.open(self.image_paths[j]) # PIL Image
                if self.augmentation_pipeline:
                    x = self.augmentation_pipeline.perform_operations(x)
                batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = batch_x[i]
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # Converts each PIL Image instance to a Numpy array.
        for i in range(len(batch_x)):
            img = batch_x[i]
            img = img_to_array(img, data_format=self.data_format, dtype=self.dtype)
            img = self._standardize(img)
            batch_x[i] = np.expand_dims(img, axis=0)

        # All images dimensions in the batch match exactly
        output = (np.vstack(batch_x),)
        if self.labels is None:
            return output[0]
        output += (self.labels[index_array],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output

    def _standardize(self, x):
        """Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standarize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:
        standarize(np.copy(x))
        # Arguments
            x: Batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """

        if self.preprocessing_function:
            x = self.preprocessing_function(x, data_format=self.data_format)
        if self.rescale:
            x *= self.rescale

        return x

    def _generate_augmented_images(self):
        augmented_images = []

        for i in range(len(self.image_paths)):
            img = Image.open(self.image_paths[i])
            if self.augmentation_pipeline:
                img = self.augmentation_pipeline.perform_operations(img)
            augmented_images.append(img)
        
        return augmented_images
