from importlib import import_module
from lesion_classifier import LesionClassifier
from base_model_param import BaseModelParam
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

class TransferLearnClassifier(LesionClassifier):
    """Skin lesion classifier based on transfer learning.

    # Arguments
        base_model_param: Instance of `BaseModelParam`.
    """

    def __init__(self, base_model_param, fc_layers=None, num_classes=None, dropout=None, batch_size=40, max_queue_size=10, image_data_format=None, metrics=None,
        gpus=None, image_paths_train=None, categories_train=None, image_paths_val=None, categories_val=None):

        if num_classes is None:
            raise ValueError('num_classes cannot be None')

        self._model_name = base_model_param.class_name

        # Dynamically create an instance of base model
        module = import_module(base_model_param.module_name)
        class_ = getattr(module, base_model_param.class_name)
        
        if image_data_format is None:
            image_data_format = K.image_data_format()
            
        if image_data_format == 'channels_first':
            input_shape = (3, base_model_param.input_size[0], base_model_param.input_size[1])
        else:
            input_shape = (base_model_param.input_size[0], base_model_param.input_size[1], 3)
        
        if gpus == 1:
            device_name = '/device:GPU:0'
        else:
            # If gpus >= 2, the model's weights are merged on CPU.
            # Reference https://keras.io/utils/#multi_gpu_model
            device_name = '/cpu:0'

        with tf.device(device_name):
            base_model = class_(include_top=False, weights='imagenet', input_shape=input_shape)
            # Whether to freeze all layers in the base model
            for layer in base_model.layers:
                layer.trainable = base_model_param.layers_trainable

            x = base_model.output
            # x = Flatten()(x)
            x = GlobalAveragePooling2D()(x)
            for fc in fc_layers:
                x = Dense(fc, activation='relu')(x)
                if dropout is not None:
                    x = Dropout(rate=dropout)(x)

            # Final layer with softmax activation
            predictions = Dense(num_classes, activation='softmax')(x) 
            model = Model(inputs=base_model.input, outputs=predictions)

        self._model_for_checkpoint = model

        if gpus is not None and gpus >= 2:
            try:
                self._model = multi_gpu_model(model, gpus=gpus)
                print("Training using multiple GPUs.")
            except ValueError:
                self._model = model
                print("Training using CPUs.")
        elif gpus == 1:
            self._model = model
            print("Training using single GPU.")
        else:
            self._model = model
            print("Training using CPUs.")

        self._model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=metrics)

        super().__init__(
            input_size=base_model_param.input_size, preprocessing_func=base_model_param.preprocessing_func,
            image_data_format=image_data_format, batch_size=batch_size, max_queue_size=max_queue_size,
            image_paths_train=image_paths_train, categories_train=categories_train,
            image_paths_val=image_paths_val, categories_val=categories_val)

    def train(self, epoch_num, class_weight=None, workers=1):
        super()._train(epoch_num, self._model_name, class_weight, workers)

    @property
    def model(self):
        return self._model

    @property
    def model_for_checkpoint(self):
        return self._model_for_checkpoint

    @property
    def model_name(self):
        return self._model_name