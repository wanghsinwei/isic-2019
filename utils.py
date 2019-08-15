from keras.preprocessing import image
from keras import backend as K
import numpy as np
import pandas as pd
import os
import cv2

def path_to_tensor(img_path, size=(224, 224)):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=(224, 224)):
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def calculate_mean_std(img_paths):
    """
    Calculate the image per channel mean and standard deviation.

    # References
        https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
    """
    
    # Number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
    channel_num = 3
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for path in img_paths:
        im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im/255.
        pixel_num += (im.size/channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std

def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images. Each image is normalized by subtracting the mean and dividing by the standard deviation channel-wise.
    This function only implements the 'torch' mode which scale pixels between 0 and 1 and then will normalize each channel with respect to the training dataset of approach 1 (not include validation set).

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    # References
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)

    # Mean and STD from ImageNet
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Mean and STD calculated over the Training Set
    # Mean:[0.6236094091893962, 0.5198354883713194, 0.5038435406338101]
    # STD:[0.2421814437693499, 0.22354427793687906, 0.2314805420919389]
    x /= 255.
    mean = [0.6236, 0.5198, 0.5038]
    std = [0.2422, 0.2235, 0.2315]

    if data_format is None:
        data_format = K.image_data_format()

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def preprocess_input_2(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images. Each image is normalized by subtracting the mean and dividing by the standard deviation channel-wise.
    This function only implements the 'torch' mode which scale pixels between 0 and 1 and then will normalize each channel with respect to the training dataset of approach 2 (not include validation set).

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    # References
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)

    # Mean and STD calculated over the training set of approach 2
    # Mean:[0.6296238064420809, 0.5202302775509949, 0.5032952297664738]
    # STD:[0.24130893564897463, 0.22150225707876617, 0.2297057828857888]
    x /= 255.
    mean = [0.6296, 0.5202, 0.5033]
    std = [0.2413, 0.2215, 0.2297]

    if data_format is None:
        data_format = K.image_data_format()

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def ensemble_predictions(result_folder, category_names, save_file=True,
                         model_names=['DenseNet201', 'Xception', 'ResNeXt50'],
                         postfixes=['best_balanced_acc', 'best_loss', 'latest']):
    """ Ensemble predictions of different models. """
    for postfix in postfixes:
        # Load models' predictions
        df_dict = {model_name : pd.read_csv(os.path.join(result_folder, "{}_{}.csv".format(model_name, postfix))) for model_name in model_names}

        # Check row number
        for i in range(1, len(model_names)):
            if len(df_dict[model_names[0]]) != len(df_dict[model_names[i]]):
                raise ValueError("Row numbers are inconsistent between {} and {}".format(model_names[0], model_names[i]))

        # Check whether values of image column are consistent
        for i in range(1, len(model_names)):
            inconsistent_idx = np.where(df_dict[model_names[0]].image != df_dict[model_names[i]].image)[0]
            if len(inconsistent_idx) > 0:
                raise ValueError("{} values of image column are inconsistent between {} and {}"
                                .format(len(inconsistent_idx), model_names[0], model_names[i]))

        # Copy the first model's predictions
        df_ensemble = df_dict[model_names[0]].drop(columns=['pred_category'])

        # Add up predictions
        for category_name in category_names:
            for i in range(1, len(model_names)):
                df_ensemble[category_name] = df_ensemble[category_name] + df_dict[model_names[i]][category_name]

        # Take average of predictions
        for category_name in category_names:
            df_ensemble[category_name] = df_ensemble[category_name] / len(model_names)

        # Ensemble Predictions
        df_ensemble['pred_category'] = np.argmax(np.array(df_ensemble.iloc[:,1:(1+len(category_names))]), axis=1)

        # Save Ensemble Predictions
        if save_file:
            ensemble_file = os.path.join(result_folder, "Ensemble_{}.csv".format(postfix))
            df_ensemble.to_csv(path_or_buf=ensemble_file, index=False)
            print('Save "{}"'.format(ensemble_file))
    return df_ensemble

def logistic(x, x0=0, L=1, k=1):
    """ Calculate the value of a logistic function.

    # Arguments
        x0: The x-value of the sigmoid's midpoint.
        L: The curve's maximum value.
        k: The logistic growth rate or steepness of the curve.
    # References https://en.wikipedia.org/wiki/Logistic_function
    """

    return L / (1 + np.exp(-k*(x-x0)))
