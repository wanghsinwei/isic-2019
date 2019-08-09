import os
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import imagenet_utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from lesion_classifier import LesionClassifier
from base_model_param import BaseModelParam

class VanillaClassifier(LesionClassifier):
    """Skin lesion classifier based on transfer learning.

    # Arguments
        base_model_param: Instance of `BaseModelParam`.
    """

    def __init__(self, model_folder, input_size=(224, 224), image_data_format=None, num_classes=None, batch_size=32, max_queue_size=10, class_weight=None,
        metrics=None, image_paths_train=None, categories_train=None, image_paths_val=None, categories_val=None):

        if num_classes is None:
            raise ValueError('num_classes cannot be None')

        self._model_name = 'Vanilla'

        # Define vanilla CNN
        self._model = Sequential()

        self._model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(input_size[0], input_size[1], 3)))
        self._model.add(MaxPooling2D(pool_size=2))

        self._model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        self._model.add(MaxPooling2D(pool_size=2))

        self._model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self._model.add(MaxPooling2D(pool_size=2))

        self._model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self._model.add(MaxPooling2D(pool_size=2))

        self._model.add(Dropout(rate=0.3))
        self._model.add(GlobalAveragePooling2D())
        self._model.add(Dense(num_classes, name='dense_pred'))
        self._model.add(Activation('softmax', name='probs'))

        # Compile the model
        self._model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=metrics)

        super().__init__(
            model_folder=model_folder, input_size=input_size, preprocessing_func=VanillaClassifier.preprocess_input, class_weight=class_weight, num_classes=num_classes,
            image_data_format=image_data_format, batch_size=batch_size, max_queue_size=max_queue_size,
            image_paths_train=image_paths_train, categories_train=categories_train,
            image_paths_val=image_paths_val, categories_val=categories_val)

    def train(self, epoch_num, workers=1):
        ### Callbacks
        checkpoints = super()._create_checkpoint_callbacks()
        
        # Reduce learning rate when the validation loss has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-6, verbose=1)

        # Stop training when the validation loss has stopped improving.
        early_stop = EarlyStopping(monitor='val_loss', patience=16, verbose=1)

        # Callback that streams epoch results to a csv file.
        csv_logger = super()._create_csvlogger_callback()

        return self.model.fit_generator(
            self.generator_train,
            class_weight=self.class_weight,
            max_queue_size=self.max_queue_size,
            workers=workers,
            use_multiprocessing=False,
            steps_per_epoch=len(self.image_paths_train)//self.batch_size,
            epochs=epoch_num,
            verbose=1,
            callbacks=(checkpoints + [reduce_lr, early_stop, csv_logger]),
            validation_data=self.generator_val,
            validation_steps=len(self.image_paths_val)//self.batch_size)

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return self._model_name

    @staticmethod
    def preprocess_input(x, **kwargs):
        """Preprocesses a numpy array encoding a batch of images.
        # Arguments
            x: a 4D numpy array consists of RGB values within [0, 255].
        # Returns
            Preprocessed array.
        """
        return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)