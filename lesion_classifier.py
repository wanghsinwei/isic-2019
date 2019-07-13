from Augmentor import Pipeline
from Augmentor.Operations import CropPercentageRange
from image_iterator import ImageIterator
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout
from keras.models import Sequential, Model

class LesionClassifier():
    """Skin lesion classifier.
    # Arguments
        batch_size: Integer, size of a batch.
        image_data_format: String, either 'channels_first' or 'channels_last'.
    """
    def __init__(self, batch_size=40, preprocessing_func=None, image_data_format='channels_last', input_size=(224, 224),
        image_paths_train=None, categories_train=None, image_paths_val=None, categories_val=None):

        self.batch_size = batch_size
        self.preprocessing_func = preprocessing_func
        self.image_data_format = image_data_format
        self.input_size = input_size
        self.image_paths_train = image_paths_train
        self.categories_train = categories_train
        self.image_paths_val = image_paths_val
        self.categories_val = categories_val

        self.aug_pipeline_train, self.aug_pipeline_val = self._create_aug_pipeline()
        self.generator_train, self.generator_val = self._create_image_generator()

    def _create_aug_pipeline(self):
        ### Training Image Augmentation Pipeline
        p_train = Pipeline()
        # Random crop
        p_train.add_operation(CropPercentageRange(probability=1, min_percentage_area=0.8, max_percentage_area=1, centre=False))
        # Rotate an image by either 90, 180, or 270 degrees randomly
        p_train.rotate_random_90(probability=0.5)
        # Flip the image along its vertical axis
        p_train.flip_top_bottom(probability=0.5)
        # Flip the image along its horizontal axis
        p_train.flip_left_right(probability=0.5)
        # Random change brightness of an image
        p_train.random_brightness(probability=0.5, min_factor=0.9, max_factor=1.1)
        # Random change saturation of an image
        p_train.random_color(probability=0.5, min_factor=0.9, max_factor=1.1)
        # Resize an image
        p_train.resize(probability=1, width=self.input_size[0], height=self.input_size[1])

        ### Validation Image Augmentation Pipeline
        p_val = Pipeline()
        # Resize an image
        p_val.resize(probability=1, width=self.input_size[0], height=self.input_size[1])

        return p_train, p_val

    def _create_image_generator(self):
        ### Training Image Generator
        generator_train = ImageIterator(
            image_paths=self.image_paths_train,
            labels=self.categories_train,
            augmentation_pipeline=self.aug_pipeline_train,
            batch_size=self.batch_size,
            shuffle=True,
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
            preprocessing_function=self.preprocessing_func,
            pregen_augmented_images=True, #Since the augmentation pipeline only contains a resize operation.
            data_format=self.image_data_format
        )

        return generator_train, generator_val

    def create_model(self, base_model=None, fc_layers=None, num_classes=None, dropout=0.3, base_model_layers_trainable=False):
        if base_model is None:
            raise ValueError('base_model cannot be None')

        if num_classes is None:
            raise ValueError('num_classes cannot be None')

        # Whether to freeze all layers in the base model
        for layer in base_model.layers:
            layer.trainable = base_model_layers_trainable

        x = base_model.output
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        for fc in fc_layers:
            # A fully-connected layer
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(dropout)(x)

        # Final layer with softmax activation
        predictions = Dense(num_classes, activation='softmax')(x) 
        
        return Model(inputs=base_model.input, outputs=predictions)