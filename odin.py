import os
import sys
import numpy as np
import pandas as pd
from typing import NamedTuple
from keras.models import load_model
from keras import backend as K
from base_model_param import get_transfer_model_param_map
from image_iterator import ImageIterator
from metrics import balanced_accuracy
from keras_numpy_backend import softmax
from lesion_classifier import LesionClassifier
from tqdm import tqdm, trange

def compute_odin_softmax_scores(pred_result_folder, derm_image_folder, out_dist_pred_result_folder, out_dist_image_folder, saved_model_folder, num_classes):
    ModelAttr = NamedTuple('ModelAttr', [('model_name', str), ('postfix', str)])
    # model_names = ['DenseNet201', 'Xception', 'ResNeXt50']
    # postfixes = ['best_balanced_acc', 'best_loss', 'latest']
    model_names = ['DenseNet201']
    postfixes = ['best_balanced_acc']
    distributions = ['In', 'Out']

    softmax_score_root_folder = 'softmax_scores'
    os.makedirs(softmax_score_root_folder, exist_ok=True)
    # This file is used for recording what parameter combinations were already computed.
    progress_file = os.path.join(softmax_score_root_folder, 'Done.txt')
    done_set = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            done_set = set(line.rstrip('\n') for line in f)

    f_done = open(progress_file, 'a')
    
    # ODIN parameters
    OdinParam = NamedTuple('OdinParam', [('temperature', int), ('magnitude', float)])
    temperatures = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
    magnitudes = np.round(np.arange(0, 0.0041, 0.0002), 4)

    model_param_map = get_transfer_model_param_map()
    image_data_format = K.image_data_format()
    learning_phase = 0 # 0 = test, 1 = train

    for modelattr in (ModelAttr(x, y) for x in model_names for y in postfixes):
        # Load model
        model_filepath = os.path.join(saved_model_folder, "{}_{}.hdf5".format(modelattr.model_name, modelattr.postfix))
        print('Load model: ', model_filepath)
        model = load_model(filepath=model_filepath, custom_objects={'balanced_accuracy': balanced_accuracy(num_classes)})
        need_norm_perturbations = (modelattr.model_name == 'DenseNet201' or modelattr.model_name == 'ResNeXt50')

        # In-distribution data
        df_in = pd.read_csv(os.path.join(pred_result_folder, "{}_{}.csv".format(modelattr.model_name, modelattr.postfix)))
        df_in['path'] = df_in.apply(lambda row : os.path.join(derm_image_folder, row['image']+'.jpg'), axis=1)
        generator_in = ImageIterator(
                image_paths=df_in['path'].tolist(),
                labels=None,
                augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(model_param_map[modelattr.model_name].input_size),
                preprocessing_function=model_param_map[modelattr.model_name].preprocessing_func,
                batch_size=1,
                shuffle=False,
                rescale=None,
                pregen_augmented_images=False,
                data_format=image_data_format)
        images_in = []
        for i in trange(df_in.shape[0], desc='In-distribution Images'):
            images_in.append(next(generator_in))

        # Out-distribution data
        df_out = pd.read_csv(os.path.join(out_dist_pred_result_folder, "{}_{}.csv".format(modelattr.model_name, modelattr.postfix)))
        df_out['path'] = df_out.apply(lambda row : os.path.join(out_dist_image_folder, row['image']+'.jpg'), axis=1)
        generator_out = ImageIterator(
                image_paths=df_out['path'].tolist(),
                labels=None,
                augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(model_param_map[modelattr.model_name].input_size),
                preprocessing_function=model_param_map[modelattr.model_name].preprocessing_func,
                batch_size=1,
                shuffle=False,
                rescale=None,
                pregen_augmented_images=False,
                data_format=image_data_format)
        images_out = []
        for i in trange(df_out.shape[0], desc='Out-distribution Images'):
            images_out.append(next(generator_out))

        for odinparam in (OdinParam(x, y) for x in temperatures for y in magnitudes):
            for dist in distributions:
                # Skip if the parameter combination has done
                param_comb_id = "{}_{}, {}, {}, {}".format(modelattr.model_name, modelattr.postfix, dist, odinparam.temperature, odinparam.magnitude)
                if param_comb_id in done_set:
                    print('Skip ', param_comb_id)
                    continue

                if dist == 'In':
                    df = df_in
                    images = images_in
                else:
                    df = df_out
                    images = images_out

                print("Computing Temperature: {}, Magnitude: {}, {}-Distribution".format(odinparam.temperature, odinparam.magnitude, dist))
                softmax_score_folder = os.path.join(softmax_score_root_folder, "{}_{}".format(odinparam.temperature, odinparam.magnitude))
                os.makedirs(softmax_score_folder, exist_ok=True)

                # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
                with open(os.path.join(softmax_score_folder, "{}_{}_Base_{}.txt".format(modelattr.model_name, modelattr.postfix, dist)), 'w') as f:
                    for _, row in df.iterrows():
                        softmax_probs = row[1:9]
                        softmax_score = np.max(softmax_probs)
                        f.write("{}, {}, {}\n".format(odinparam.temperature, odinparam.magnitude, softmax_score))

                ### Define Keras Functions
                # Compute loss based on model's outputs of last two layers and temperature scaling
                dense_pred_layer_output = model.get_layer('dense_pred').output
                scaled_dense_pred_output = dense_pred_layer_output / odinparam.temperature
                label_tensor = K.one_hot(K.argmax(dense_pred_layer_output), num_classes)
                loss = K.categorical_crossentropy(label_tensor, K.softmax(scaled_dense_pred_output))

                # Compute gradient of loss with respect to inputs
                grad_loss = K.gradients(loss, model.inputs)

                # The learning phase flag is a bool tensor (0 = test, 1 = train)
                compute_perturbations = K.function(model.inputs + [K.learning_phase()], grad_loss)

                # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
                get_dense_pred_layer_output = K.function(model.inputs + [K.learning_phase()], [dense_pred_layer_output])

                f = open(os.path.join(softmax_score_folder, "{}_{}_ODIN_{}.txt".format(modelattr.model_name, modelattr.postfix, dist)), 'w')
                for image in tqdm(images, desc="{}-Distribution Softmax Scores".format(dist)):
                    perturbations = compute_perturbations([image, learning_phase])[0]

                    # Get sign of perturbations
                    perturbations = np.sign(perturbations)
                    
                    # Normalize the perturbations to the same space of image
                    # https://github.com/facebookresearch/odin/issues/5
                    # Perturbations divided by ISIC Training Set STD
                    if need_norm_perturbations:
                        perturbations = norm_perturbations(perturbations, image_data_format)
                    
                    # Add perturbations to image
                    perturbative_image = image - odinparam.magnitude * perturbations
                    
                    # Calculate the confidence after adding perturbations
                    dense_pred_output = get_dense_pred_layer_output([perturbative_image, learning_phase])[0]
                    dense_pred_output = dense_pred_output / odinparam.temperature
                    softmax_probs = softmax(dense_pred_output)
                    softmax_score = np.max(softmax_probs)

                    f.write("{}, {}, {}\n".format(odinparam.temperature, odinparam.magnitude, softmax_score))
                f.close()
                f_done.write("{}\n".format(param_comb_id))
                f_done.flush()
        del model
        K.clear_session()
    f_done.close()

def norm_perturbations(x, image_data_format):
    std = [0.2422, 0.2235, 0.2315]

    if image_data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] /= std[0]
            x[1, :, :] /= std[1]
            x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] /= std[0]
            x[:, 1, :, :] /= std[1]
            x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x

def compute_tpr95(scores_in, scores_out, delta_start, delta_end, delta_num=100000):
    tpr95_min = 0.9495
    tpr95_max = 0.9505
    delta_step = (delta_end - delta_start)/delta_num
    tpr95_delta_min = sys.float_info.max
    tpr95_delta_max = sys.float_info.min
    scores_in_count = np.float(len(scores_in))
    scores_out_count = np.float(len(scores_out))
    tpr95_delta_count = 0
    fpr = 0.0

    for delta in tqdm(np.arange(delta_start, delta_end, delta_step), desc='Compute TPR95'):
        tpr = np.sum(scores_in >= delta) / scores_in_count
        error = np.sum(scores_out > delta) / scores_out_count
        if tpr >= tpr95_min and tpr <= tpr95_max:
            # print("delta:{}, tpr:{}, error:{}".format(delta, tpr, error))
            tpr95_delta_min = min(tpr95_delta_min, delta)
            tpr95_delta_max = max(tpr95_delta_max, delta)
            fpr += error
            tpr95_delta_count += 1
    fpr = fpr/tpr95_delta_count
    print("fpr:{}, tpr95_delta_count:{}, tpr95_delta_min:{}, tpr95_delta_max:{}".format(
        fpr, tpr95_delta_count, tpr95_delta_min, tpr95_delta_max))
    return fpr, tpr95_delta_count, tpr95_delta_min, tpr95_delta_max
    

def tpr95(base_file_in, base_file_out, odin_file_in, odin_file_out):
    """
    calculate the falsepositive error when tpr is 95%
    """
    # calculate baseline
    base_in = np.loadtxt(base_file_in, delimiter=',')
    base_out = np.loadtxt(base_file_out, delimiter=',')

    fpr_base, tpr95_delta_count, tpr95_delta_min, tpr95_delta_max = compute_tpr95(
        scores_in=base_in[:, 2],
        scores_out=base_out[:, 2],
        delta_start=0.01,
        delta_end=1)

    # calculate ODIN algorithm
    odin_in = np.loadtxt(odin_file_in, delimiter=',')
    odin_out = np.loadtxt(odin_file_out, delimiter=',')

    fpr_odin, tpr95_delta_count, tpr95_delta_min, tpr95_delta_max = compute_tpr95(
        scores_in=odin_in[:, 2],
        scores_out=odin_out[:, 2],
        delta_start=0.01,
        delta_end=0.2)

    return fpr_base, fpr_odin
