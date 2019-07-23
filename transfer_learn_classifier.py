from importlib import import_module
from lesion_classifier import LesionClassifier
from base_model_param import BaseModelParam
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model

class TransferLearnClassifier(LesionClassifier):
    """Skin lesion classifier based on transfer learning.

    # Arguments
        base_model_param: Instance of `BaseModelParam`.
    """

    def __init__(self, base_model_param, fc_layers=None, num_classes=None, dropout=None, batch_size=64, max_queue_size=10, image_data_format=None, metrics=None,
        gpus=None, image_paths_train=None, categories_train=None, image_paths_val=None, categories_val=None):

        if num_classes is None:
            raise ValueError('num_classes cannot be None')

        self._model_name = base_model_param.class_name
        self.metrics = metrics

        if image_data_format is None:
            image_data_format = K.image_data_format()
            
        if image_data_format == 'channels_first':
            input_shape = (3, base_model_param.input_size[0], base_model_param.input_size[1])
        else:
            input_shape = (base_model_param.input_size[0], base_model_param.input_size[1], 3)

        # Dynamically get the class name of base model
        module = import_module(base_model_param.module_name)
        class_ = getattr(module, base_model_param.class_name)
        
        self.gpus = gpus
        self.start_lr = 1e-4

        if gpus == 1:
            device_name = '/device:GPU:0'
        else:
            # If gpus >= 2, the model's weights are merged on CPU.
            # Reference https://keras.io/utils/#multi_gpu_model
            device_name = '/cpu:0'

        with tf.device(device_name):
            # create an instance of base model which is pre-trained on the ImageNet dataset.
            if base_model_param.class_name == 'ResNeXt50':
                # A workaround to use ResNeXt in Keras 2.2.4.
                # See http://donghao.org/2019/02/22/using-resnext-in-keras-2-2-4/
                self._base_model = class_(include_top=False, weights='imagenet', input_shape=input_shape,
                                         backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            else:
                self._base_model = class_(include_top=False, weights='imagenet', input_shape=input_shape)

            # Freeze all layers in the base model
            for layer in self._base_model.layers:
                layer.trainable = False

            x = self._base_model.output
            x = GlobalAveragePooling2D()(x)
            # Add fully connected layers
            for fc in fc_layers:
                x = Dense(fc, activation='relu')(x)
                if dropout is not None:
                    x = Dropout(rate=dropout)(x)

            # Final layer with softmax activation
            predictions = Dense(num_classes, activation='softmax')(x)
            # Create the model
            model = Model(inputs=self._base_model.input, outputs=predictions)
            # Compile the model
            model.compile(optimizer=Adam(lr=self.start_lr), loss='categorical_crossentropy', metrics=self.metrics)

        self._model_for_checkpoint = model

        if gpus is not None and gpus >= 2:
            try:
                self._model = multi_gpu_model(model, gpus=gpus)
                # Compile the model
                self._model.compile(optimizer=Adam(lr=self.start_lr), loss='categorical_crossentropy', metrics=self.metrics)
                print('===== Training using multiple GPUs =====')
            except ValueError:
                self._model = model
                print('===== Training using CPU(s) =====')
        elif gpus == 1:
            self._model = model
            print('===== Training using single GPU =====')
        else:
            self._model = model
            print('===== Training using CPU(s) =====')

        super().__init__(
            input_size=base_model_param.input_size, preprocessing_func=base_model_param.preprocessing_func,
            image_data_format=image_data_format, batch_size=batch_size, max_queue_size=max_queue_size,
            image_paths_train=image_paths_train, categories_train=categories_train,
            image_paths_val=image_paths_val, categories_val=categories_val)

    def train(self, epoch_num, class_weight=None, workers=1):
        
        feature_extract_epochs = 5

        # Checkpoint Callbacks
        checkpoints = self._create_checkpoint_callbacks(self._model_name)

        # This ReduceLROnPlateau is just a workaround to make csv_logger record learning rate, and won't affect learning rate during feature extraction epochs.
        reduce_lr = ReduceLROnPlateau(patience=feature_extract_epochs+10, verbose=1)

        # Callback that streams epoch results to a csv file.
        csv_logger = self._create_csvlogger_callback(self._model_name)

        ### Feature extraction
        self._model.fit_generator(
            self.generator_train,
            class_weight=class_weight,
            max_queue_size=self.max_queue_size,
            workers=workers,
            use_multiprocessing=False,
            steps_per_epoch=len(self.image_paths_train)//self.batch_size,
            epochs=feature_extract_epochs,
            verbose=1,
            callbacks=(checkpoints + [reduce_lr, csv_logger]),
            validation_data=self.generator_val,
            validation_steps=len(self.image_paths_val)//self.batch_size)

        ### Fine tuning. It should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable.
        print('===== Unfreeze the base model =====')
        for layer in self._base_model.layers:
            layer.trainable = True
        
        # Compile the model
        if self.gpus is not None and self.gpus >= 2:
            self._model_for_checkpoint.compile(optimizer=Adam(lr=self.start_lr), loss='categorical_crossentropy', metrics=self.metrics)
        self._model.compile(optimizer=Adam(lr=self.start_lr), loss='categorical_crossentropy', metrics=self.metrics)
        self._model.summary()

        # Reduce learning rate when the validation loss has stopped improving.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1)

        # Stop training when the validation loss has stopped improving.
        early_stop = EarlyStopping(monitor='val_loss', patience=25, verbose=1)

        self.generator_train.reset()
        self._model.fit_generator(
            self.generator_train,
            class_weight=class_weight,
            max_queue_size=self.max_queue_size,
            workers=workers,
            use_multiprocessing=False,
            steps_per_epoch=len(self.image_paths_train)//self.batch_size,
            epochs=epoch_num,
            verbose=1,
            callbacks=(checkpoints + [reduce_lr, early_stop, csv_logger]),
            validation_data=self.generator_val,
            validation_steps=len(self.image_paths_val)//self.batch_size,
            initial_epoch=feature_extract_epochs)

    @property
    def model(self):
        return self._model

    @property
    def model_for_checkpoint(self):
        return self._model_for_checkpoint

    @property
    def model_name(self):
        return self._model_name
