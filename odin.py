import os
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

def compute_odin_softmax_scores(pred_result_folder, derm_image_folder, out_dist_pred_result_folder, out_dist_image_folder, saved_model_folder, num_classes):
    ModelAttr = NamedTuple('ModelAttr', [('model_name', str), ('postfix', str)])
    model_names = ['DenseNet201', 'Xception', 'ResNeXt50']
    postfixes = ['best_balanced_acc', 'best_loss', 'latest']
    distributions = ['In', 'Out']
    
    # ODIN parameters
    OdinParam = NamedTuple('OdinParam', [('temperature', int), ('magnitude', float)])
    temperatures = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    magnitudes = np.round(np.arange(0, 0.0041, 0.0002), 4)

    model_param_map = get_transfer_model_param_map()
    image_data_format = K.image_data_format()

    for modelattr in (ModelAttr(x, y) for x in model_names for y in postfixes):
        # Load model
        model_filepath = os.path.join(saved_model_folder, "{}_{}.hdf5".format(modelattr.model_name, modelattr.postfix))
        print('Load model: ', model_filepath)
        model = load_model(filepath=model_filepath, custom_objects={'balanced_accuracy': balanced_accuracy(num_classes)})
        for odinparam in (OdinParam(x, y) for x in temperatures for y in magnitudes):
            print("Temperature: {}, Magnitude: {}".format(odinparam.temperature, odinparam.magnitude))
            softmax_score_folder = os.path.join('softmax_scores', "{}_{}".format(odinparam.temperature, odinparam.magnitude))
            os.makedirs(softmax_score_folder, exist_ok=True)
            for dist in distributions:
                # Load predicted results
                if dist == 'In':
                    df = pd.read_csv(os.path.join(pred_result_folder, "{}_{}.csv".format(modelattr.model_name, modelattr.postfix)))
                    df['path'] = df.apply(lambda row : os.path.join(derm_image_folder, row['image']+'.jpg'), axis=1)
                else:
                    df = pd.read_csv(os.path.join(out_dist_pred_result_folder, "{}_{}.csv".format(modelattr.model_name, modelattr.postfix)))
                    df['path'] = df.apply(lambda row : os.path.join(out_dist_image_folder, row['image']+'.jpg'), axis=1)
            
                # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
                with open(os.path.join(softmax_score_folder, "{}_{}_Base_{}.txt".format(modelattr.model_name, modelattr.postfix, dist)), 'w') as f:
                    for _, row in df.iterrows():
                        softmax_probs = row[1:9]
                        softmax_score = np.max(softmax_probs)
                        f.write("{}, {}, {}\n".format(odinparam.temperature, odinparam.magnitude, softmax_score))

                ### Define Keras Functions
                # Compute loss based on model's outputs of last two layers and temperature scaling
                scaled_dense_pred_output = model.get_layer('dense_pred').output / odinparam.temperature
                label_tensor = K.one_hot(K.argmax(model.outputs), num_classes)
                loss = K.categorical_crossentropy(label_tensor, K.softmax(scaled_dense_pred_output))

                # Compute gradient of loss with respect to inputs
                grad_loss = K.gradients(loss, model.inputs)

                # The learning phase flag is a bool tensor (0 = test, 1 = train)
                compute_perturbations = K.function(model.inputs + [K.learning_phase()],
                                                   grad_loss)

                # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
                get_dense_pred_layer_output = K.function(model.inputs + [K.learning_phase()],
                                                        [model.get_layer('dense_pred').output])

                # 0 = test, 1 = train
                learning_phase = 0

                generator = ImageIterator(
                    image_paths=df['path'].tolist(),
                    labels=None,
                    augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(model_param_map[modelattr.model_name].input_size),
                    preprocessing_function=model_param_map[modelattr.model_name].preprocessing_func,
                    batch_size=1,
                    shuffle=False,
                    rescale=None,
                    pregen_augmented_images=False,
                    data_format=image_data_format)

                f = open(os.path.join(softmax_score_folder, "{}_{}_Odin_{}.txt".format(modelattr.model_name, modelattr.postfix, dist)), 'w')
                for image in generator:
                    perturbations = compute_perturbations([image, learning_phase])[0]
                    # Get sign of perturbations
                    perturbations = np.sign(perturbations)
                    
                    # Normalize the perturbations to the same space of image
                    # https://github.com/facebookresearch/odin/issues/5
                    # Perturbations divided by ISIC Training Set STD
                    if modelattr.model_name == 'DenseNet201' or modelattr.model_name == 'ResNeXt50':
                        perturbations[0][0] = perturbations[0][0] / 0.2422
                        perturbations[0][1] = perturbations[0][1] / 0.2235
                        perturbations[0][2] = perturbations[0][2] / 0.2315
                    
                    # Add perturbations to image
                    perturbative_image = image - odinparam.magnitude * perturbations
                    
                    # Calculate the confidence after adding perturbations
                    dense_pred_output = get_dense_pred_layer_output([perturbative_image, learning_phase])[0]
                    dense_pred_output = dense_pred_output / odinparam.temperature
                    softmax_probs = softmax(dense_pred_output)
                    softmax_score = np.max(softmax_probs)

                    f.write("{}, {}, {}\n".format(odinparam.temperature, odinparam.magnitude, softmax_score))
                f.close()
        del model
        K.clear_session()
