import os
import sys
import math
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
from tqdm import trange

ModelAttr = NamedTuple('ModelAttr', [('model_name', str), ('postfix', str)])
OdinParam = NamedTuple('OdinParam', [('temperature', int), ('magnitude', float)])

def compute_odin_softmax_scores(pred_result_folder, derm_image_folder, out_dist_pred_result_folder, out_dist_image_folder, saved_model_folder, num_classes, batch_size):
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
    
    # ODIN parameters
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
                batch_size=batch_size,
                shuffle=False,
                rescale=None,
                pregen_augmented_images=True,
                data_format=image_data_format)

        # Out-distribution data
        df_out = pd.read_csv(os.path.join(out_dist_pred_result_folder, "{}_{}.csv".format(modelattr.model_name, modelattr.postfix)))
        df_out['path'] = df_out.apply(lambda row : os.path.join(out_dist_image_folder, row['image']+'.jpg'), axis=1)
        generator_out = ImageIterator(
                image_paths=df_out['path'].tolist(),
                labels=None,
                augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(model_param_map[modelattr.model_name].input_size),
                preprocessing_function=model_param_map[modelattr.model_name].preprocessing_func,
                batch_size=batch_size,
                shuffle=False,
                rescale=None,
                pregen_augmented_images=True,
                data_format=image_data_format)

        for odinparam in (OdinParam(x, y) for x in temperatures for y in magnitudes):
            for dist in distributions:
                # Skip if the parameter combination has done
                param_comb_id = "{}_{}, {}, {}, {}".format(modelattr.model_name, modelattr.postfix, dist, odinparam.temperature, odinparam.magnitude)
                if param_comb_id in done_set:
                    print('Skip ', param_comb_id)
                    continue

                if dist == 'In':
                    df = df_in
                    generator = generator_in
                else:
                    df = df_out
                    generator = generator_out

                print("\n===== Temperature: {}, Magnitude: {}, {}-Distribution =====".format(odinparam.temperature, odinparam.magnitude, dist))
                softmax_score_folder = os.path.join(softmax_score_root_folder, "{}_{}".format(odinparam.temperature, odinparam.magnitude))
                os.makedirs(softmax_score_folder, exist_ok=True)

                # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
                # Here just copy the original prediction results
                with open(os.path.join(softmax_score_folder, "{}_{}_Base_{}.txt".format(modelattr.model_name, modelattr.postfix, dist)), 'w') as f:
                    for _, row in df.iterrows():
                        softmax_probs = row[1:9]
                        softmax_score = np.max(softmax_probs)
                        f.write("{}, {}, {}\n".format(odinparam.temperature, odinparam.magnitude, softmax_score))

                ### Define Keras Functions
                # Compute loss based on the second last layer's output and temperature scaling
                dense_pred_layer_output = model.get_layer('dense_pred').output
                scaled_dense_pred_output = dense_pred_layer_output / odinparam.temperature
                label_tensor = K.one_hot(K.argmax(model.outputs), num_classes)
                # ODIN implementation uses torch.nn.CrossEntropyLoss
                # Keras will call tf.nn.softmax_cross_entropy_with_logits when from_logits is True
                loss = K.categorical_crossentropy(label_tensor, scaled_dense_pred_output, from_logits=True)

                # Compute gradient of loss with respect to inputs
                grad_loss = K.gradients(loss, model.inputs)

                # The learning phase flag is a bool tensor (0 = test, 1 = train)
                compute_perturbations = K.function(model.inputs + [K.learning_phase()], grad_loss)

                # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
                get_dense_pred_layer_output = K.function(model.inputs + [K.learning_phase()], [dense_pred_layer_output])

                steps = math.ceil(df.shape[0] / batch_size)
                generator.reset()
                f = open(os.path.join(softmax_score_folder, "{}_{}_ODIN_{}.txt".format(modelattr.model_name, modelattr.postfix, dist)), 'w')
                for _ in trange(steps):
                    images = next(generator)
                    perturbations = compute_perturbations([images, learning_phase])[0]
                    # Get sign of perturbations
                    perturbations = np.sign(perturbations)
                    
                    # Normalize the perturbations to the same space of image
                    # https://github.com/facebookresearch/odin/issues/5
                    # Perturbations divided by ISIC Training Set STD
                    if need_norm_perturbations:
                        perturbations = norm_perturbations(perturbations, image_data_format)
                    
                    # Add perturbations to images
                    perturbative_images = images - odinparam.magnitude * perturbations
                    
                    # Calculate the confidence after adding perturbations
                    dense_pred_outputs = get_dense_pred_layer_output([perturbative_images, learning_phase])[0]
                    dense_pred_outputs = dense_pred_outputs / odinparam.temperature
                    softmax_probs = softmax(dense_pred_outputs)
                    softmax_scores = np.max(softmax_probs, axis=-1)
                    for s in softmax_scores:
                        f.write("{}, {}, {}\n".format(odinparam.temperature, odinparam.magnitude, s))
                f.close()

                with open(progress_file, 'a') as f_done:
                    f_done.write("{}\n".format(param_comb_id))
        del model
        K.clear_session()


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
    """
    calculate the false positive rate (FPR) when true positive rate (TPR) is 95%
    """
    delta_step = (delta_end - delta_start)/delta_num
    delta_best = None
    scores_in_count = np.float(len(scores_in))
    scores_out_count = np.float(len(scores_out))
    tpr95_count = 0
    fpr_min = sys.float_info.max
    fpr_sum = 0.0

    for delta in np.arange(delta_start, delta_end, delta_step):
        tpr = np.sum(scores_in >= delta) / scores_in_count
        if 0.9495 <= tpr <= 0.9505:
            fpr = np.sum(scores_out > delta) / scores_out_count
            # print("delta:{}, tpr:{}, fpr:{}".format(delta, tpr, fpr))
            if fpr < fpr_min:
                delta_best = delta  # The optimal delta is chosen to minimize the FPR at TPR 95%
            fpr_sum += fpr
            tpr95_count += 1
    fpr_at_tpr95 = fpr_sum / tpr95_count
    return fpr_at_tpr95, delta_best


def tpr95_base(in_dist_file, out_dist_file):
    """ Calculate baseline """
    base_in = np.loadtxt(in_dist_file, delimiter=',')
    base_out = np.loadtxt(out_dist_file, delimiter=',')

    fpr, delta = compute_tpr95(
        scores_in=base_in[:, 2],
        scores_out=base_out[:, 2],
        delta_start=0.01,
        delta_end=1)
    return fpr, delta


def tpr95_odin(in_dist_file, out_dist_file):
    """ Calculate ODIN method """
    odin_in = np.loadtxt(in_dist_file, delimiter=',')
    odin_out = np.loadtxt(out_dist_file, delimiter=',')

    fpr, delta = compute_tpr95(
        scores_in=odin_in[:, 2],
        scores_out=odin_out[:, 2],
        delta_start=0.1,
        delta_end=0.2)
    return fpr, delta
