import math
import os
from Augmentor import Pipeline
from Augmentor.Operations import CropPercentageRange
from image_iterator import ImageIterator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import keras.backend as K
import tensorflow as tf

class LesionClassifier():
    """Base class of skin lesion classifier.
    # Arguments
        batch_size: Integer, size of a batch.
        image_data_format: String, either 'channels_first' or 'channels_last'.
    """
    def __init__(self, input_size, image_data_format, batch_size=32, rescale=None, preprocessing_func=None,
        num_classes=None, image_paths_train=None, categories_train=None, image_paths_val=None, categories_val=None):

        self.input_size = input_size
        self.image_data_format = image_data_format
        self.batch_size = batch_size
        self.rescale = rescale
        self.preprocessing_func = preprocessing_func
        self.num_classes = num_classes
        self.image_paths_train = image_paths_train
        self.categories_train = categories_train
        self.image_paths_val = image_paths_val
        self.categories_val = categories_val

        self.aug_pipeline_train, self.aug_pipeline_val = self._create_aug_pipeline()
        self.generator_train, self.generator_val = self._create_image_generator()

    def _create_aug_pipeline(self):
        ### Image Augmentation Pipeline for Training Set
        p_train = Pipeline()
        # Resize the image to 1.25 times of the desired input size of the model
        resize_target_size = tuple(math.ceil(1.25*x) for x in self.input_size)
        p_train.resize(probability=1, width=resize_target_size[0], height=resize_target_size[1])
        # Random crop
        p_train.add_operation(CropPercentageRange(probability=1, min_percentage_area=0.8, max_percentage_area=1, centre=False))
        # Resize the image to the desired input size of the model
        p_train.resize(probability=1, width=self.input_size[0], height=self.input_size[1])
        # Rotate the image by either 90, 180, or 270 degrees randomly
        p_train.rotate_random_90(probability=0.5)
        # Flip the image along its vertical axis
        p_train.flip_top_bottom(probability=0.5)
        # Flip the image along its horizontal axis
        p_train.flip_left_right(probability=0.5)
        # Random change brightness of the image
        p_train.random_brightness(probability=0.5, min_factor=0.9, max_factor=1.1)
        # Random change saturation of the image
        p_train.random_color(probability=0.5, min_factor=0.9, max_factor=1.1)
        print('Image Augmentation Pipeline for Training Set')
        p_train.status()

        ### Image Augmentation Pipeline for Validation Set
        p_val = Pipeline()
        # Center Crop
        p_val.crop_centre(probability=1, percentage_area=0.9)
        # Resize the image to the desired input size of the model
        p_val.resize(probability=1, width=self.input_size[0], height=self.input_size[1])
        print('Image Augmentation Pipeline for Validation Set')
        p_val.status()

        return p_train, p_val

    def _create_image_generator(self):
        ### Training Image Generator
        generator_train = ImageIterator(
            image_paths=self.image_paths_train,
            labels=self.categories_train,
            augmentation_pipeline=self.aug_pipeline_train,
            batch_size=self.batch_size,
            shuffle=True,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_func,
            pregen_augmented_images=False,
            data_format=self.image_data_format
        )

        ### Validation Image Generator
        generator_val = ImageIterator(
            image_paths=self.image_paths_val,
            labels=self.categories_val,
            augmentation_pipeline=self.aug_pipeline_val,
            batch_size=self.batch_size,
            shuffle=True,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_func,
            pregen_augmented_images=True, # Since the augmentation pipeline only contains center crop and resize operations.
            data_format=self.image_data_format
        )

        return generator_train, generator_val

    def _train(self, epoch_num, model_name, class_weight=None, workers=1):
        self.model.fit_generator(
            self.generator_train,
            class_weight=class_weight,
            max_queue_size=10,
            workers=workers,
            use_multiprocessing=False,
            steps_per_epoch=len(self.image_paths_train)//self.batch_size,
            epochs=epoch_num,
            verbose=1,
            callbacks=self._create_callbacks(model_name),
            validation_data=self.generator_val,
            validation_steps=len(self.image_paths_val)//self.batch_size)

    def _create_callbacks(self, model_name):
        """Create the functions to be applied at given stages of the training procedure."""

        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        checkpoint_balanced_acc = ModelCheckpoint(
            filepath="saved_models/{}_best_balanced_acc.hdf5".format(model_name),
            monitor='val_balanced_accuracy',
            verbose=1,
            save_best_only=True)
        
        checkpoint_balanced_acc_latest = ModelCheckpoint(
            filepath="saved_models/{}_latest_balanced_acc.hdf5".format(model_name),
            monitor='val_balanced_accuracy',
            verbose=1,
            save_best_only=False)

        checkpoint_loss = ModelCheckpoint(
            filepath="saved_models/{}_best_loss.hdf5".format(model_name),
            monitor='val_loss',
            verbose=1,
            save_best_only=True)
        
        # Callback that streams epoch results to a csv file.
        csv_logger = CSVLogger("logs/{}.training.csv".format(model_name), append=True)
        
        # Reduce learning rate when the validation loss has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)

        # Stop training when the validation loss has stopped improving.
        early_stop = EarlyStopping(monitor='val_loss', patience=22, verbose=1)
        
        return [checkpoint_balanced_acc, checkpoint_balanced_acc_latest, checkpoint_loss, csv_logger, reduce_lr, early_stop]

    @property
    def model(self):
        """CNN Model"""
        raise NotImplementedError(
            '`model` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def model_name(self):
        """Name of the CNN Model"""
        raise NotImplementedError(
            '`model_name` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )