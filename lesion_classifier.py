import math
import os
import pandas as pd
import numpy as np
from Augmentor import Pipeline
from Augmentor.Operations import CropPercentageRange
from image_iterator import ImageIterator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K
import tensorflow as tf
from callbacks import MyModelCheckpoint

class LesionClassifier():
    """Base class of skin lesion classifier.
    # Arguments
        batch_size: Integer, size of a batch.
        image_data_format: String, either 'channels_first' or 'channels_last'.
    """
    def __init__(self, input_size, image_data_format=None, batch_size=64, max_queue_size=10, rescale=None, preprocessing_func=None,
        num_classes=None, image_paths_train=None, categories_train=None, image_paths_val=None, categories_val=None):

        self.log_folder = 'logs'
        self.saved_model_folder = 'saved_models'
        self.input_size = input_size
        if image_data_format is None:
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.rescale = rescale
        self.preprocessing_func = preprocessing_func
        self.num_classes = num_classes
        self.image_paths_train = image_paths_train
        self.categories_train = categories_train
        self.image_paths_val = image_paths_val
        self.categories_val = categories_val

        self.aug_pipeline_train = LesionClassifier.create_aug_pipeline_train(self.input_size)
        print('Image Augmentation Pipeline for Training Set')
        self.aug_pipeline_train.status()

        self.aug_pipeline_val = LesionClassifier.create_aug_pipeline_val(self.input_size)
        print('Image Augmentation Pipeline for Validation Set')
        self.aug_pipeline_val.status()

        self.generator_train, self.generator_val = self._create_image_generator()

    @staticmethod
    def create_aug_pipeline_train(input_size):
        """Image Augmentation Pipeline for Training Set."""
        
        # p_train = Pipeline()
        # # Resize the image to 1.25 times of the desired input size of the model
        # resize_target_size = tuple(math.ceil(1.25*x) for x in input_size)
        # p_train.resize(probability=1, width=resize_target_size[0], height=resize_target_size[1])
        # # Random crop
        # p_train.add_operation(CropPercentageRange(probability=1, min_percentage_area=0.8, max_percentage_area=1, centre=False))
        # # Resize the image to the desired input size of the model
        # p_train.resize(probability=1, width=input_size[0], height=input_size[1])
        # # Rotate the image by either 90, 180, or 270 degrees randomly
        # p_train.rotate_random_90(probability=0.5)
        # # Flip the image along its vertical axis
        # p_train.flip_top_bottom(probability=0.5)
        # # Flip the image along its horizontal axis
        # p_train.flip_left_right(probability=0.5)
        # # Random change brightness of the image
        # p_train.random_brightness(probability=0.5, min_factor=0.9, max_factor=1.1)
        # # Random change saturation of the image
        # p_train.random_color(probability=0.5, min_factor=0.9, max_factor=1.1)

        p_train = Pipeline()
        # Random crop
        p_train.add_operation(CropPercentageRange(probability=1, min_percentage_area=0.8, max_percentage_area=1, centre=False))
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
        # Resize the image to the desired input size of the model
        p_train.resize(probability=1, width=input_size[0], height=input_size[1])

        return p_train

    @staticmethod
    def create_aug_pipeline_val(input_size):
        """Image Augmentation Pipeline for Validation Set."""
        p_val = Pipeline()
        # # Center Crop
        # p_val.crop_centre(probability=1, percentage_area=0.9)
        # Resize the image to the desired input size of the model
        p_val.resize(probability=1, width=input_size[0], height=input_size[1])
        return p_val

    @staticmethod
    def predict_dataframe(model, df, x_col='path', y_col='category', id_col='image', category_names=None,
                          augmentation_pipeline=None, preprocessing_function=None,
                          batch_size=64, workers=1, save_file_name=None):
        generator = ImageIterator(
            image_paths=df[x_col].tolist(),
            labels=None,
            augmentation_pipeline=augmentation_pipeline,
            batch_size=batch_size,
            shuffle=False,  # shuffle must be False otherwise will get a wrong balanced accuracy
            rescale=None,
            preprocessing_function=preprocessing_function,
            pregen_augmented_images=False,  # Only 1 epoch.
            data_format=K.image_data_format()
        )

        # Predict
        predicted_vector = model.predict_generator(generator, verbose=1, workers=workers)

        # Save the predicted results as a csv file
        df_pred = pd.DataFrame(predicted_vector, columns=category_names)
        df_pred[y_col] = df[y_col].to_numpy()
        df_pred['pred_'+y_col] = np.argmax(predicted_vector, axis=1)
        df_pred.insert(0, id_col, df[id_col].to_numpy())
        if save_file_name is not None:
            df_pred.to_csv(path_or_buf=save_file_name, index=False)
        return df_pred

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
            pregen_augmented_images=True, # Since there is no randomness in the augmentation pipeline.
            data_format=self.image_data_format
        )

        return generator_train, generator_val

    def _create_checkpoint_callbacks(self, model, model_name):
        """Create the functions to be applied at given stages of the training procedure."""

        if not os.path.exists(self.saved_model_folder):
            os.makedirs(self.saved_model_folder)
        
        checkpoint_balanced_acc = MyModelCheckpoint(
            model=model,
            filepath=os.path.join(self.saved_model_folder, "{}_best_balanced_acc.hdf5".format(model_name)),
            monitor='val_balanced_accuracy',
            verbose=1,
            save_best_only=True)
        
        checkpoint_latest = MyModelCheckpoint(
            model=model,
            filepath=os.path.join(self.saved_model_folder, "{}_latest.hdf5".format(model_name)),
            verbose=1,
            save_best_only=False)

        checkpoint_loss = MyModelCheckpoint(
            model=model,
            filepath=os.path.join(self.saved_model_folder, "{}_best_loss.hdf5".format(model_name)),
            monitor='val_loss',
            verbose=1,
            save_best_only=True)
        
        return [checkpoint_balanced_acc, checkpoint_latest, checkpoint_loss]

    def _create_csvlogger_callback(self, model_name):
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        return CSVLogger(filename=os.path.join(self.log_folder, "{}.training.csv".format(model_name)), append=True)

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
