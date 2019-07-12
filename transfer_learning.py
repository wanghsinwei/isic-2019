from Augmentor import Pipeline
from Augmentor.Operations import CropPercentageRange
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout
from keras.models import Sequential, Model

def build_aug_pipeline(input_size=(224, 224)):
    ### Training Data Generator
    p_train = Pipeline()
    # Random crop
    p_train.add_operation(CropPercentageRange(probability=1, min_percentage_area=0.8, max_percentage_area=1, centre=False))
    # Rotate an image by either 90, 180, or 270 degrees randomly
    p_train.rotate_random_90(probability=0.5)
    # Resize an image
    p_train.resize(probability=1, width=input_size[0], height=input_size[1])
    # Flip the image along its vertical axis
    p_train.flip_top_bottom(probability=0.5)
    # Flip the image along its horizontal axis
    p_train.flip_left_right(probability=0.5)
    # Random change brightness of an image
    p_train.random_brightness(probability=0.5, min_factor=0.9, max_factor=1.1)
    # Random change saturation of an image
    p_train.random_color(probability=0.5, min_factor=0.9, max_factor=1.1)

    ### Validation Data Generator
    p_val = Pipeline()
    # Resize an image
    p_val.resize(probability=1, width=input_size[0], height=input_size[1])

    return p_train, p_val

def build_finetune_model(base_model, fc_layers, dropout, num_classes, base_model_layers_trainable=False):
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