import argparse
import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
# from keras.applications.densenet import preprocess_input as preprocess_input_densenet
# from keras_applications.resnext import preprocess_input as preprocess_input_resnext
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from utils import preprocess_input as preprocess_input_trainset
import keras
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
from keras_numpy_backend import softmax
from data import load_isic_data, train_validation_split, compute_class_weight_dict
from vanilla_classifier import VanillaClassifier
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import BaseModelParam
from lesion_classifier import LesionClassifier
from image_iterator import ImageIterator
from typing import NamedTuple

def main():
    parser = argparse.ArgumentParser(description='ISIC-2019 Skin Lesion Classifiers')
    parser.add_argument('data', metavar='DIR', help='path to data foler')
    parser.add_argument('--batchsize', type=int, help='Batch size (default: %(default)s)', default=32)
    parser.add_argument('--maxqueuesize', type=int, help='Maximum size for the generator queue (default: %(default)s)', default=10)
    parser.add_argument('--epoch', type=int, help='Number of epochs (default: %(default)s)', default=100)
    parser.add_argument('--vanilla', dest='vanilla', action='store_true', help='Train Vanilla CNN')
    parser.add_argument('--transfer', dest='transfer_models', nargs='*', help='Models for Transfer Learning')
    parser.add_argument('--autoshutdown', dest='autoshutdown', action='store_true', help='Automatically shutdown the computer after everything is done')
    parser.add_argument('--skiptraining', dest='skiptraining', action='store_true', help='Skip training processes')
    parser.add_argument('--skippredict', dest='skippredict', action='store_true', help='Skip predicting validation set')
    parser.add_argument('--skipodin', dest='skipodin', action='store_true', help='Skip computing ODIN softmax scores')
    parser.add_argument('--temperature', type=int, help='Temperature (default: %(default)s)', default=1000)
    parser.add_argument('--magnitude', type=float, help='Noise/Perturbation Magnitude (default: %(default)s)', default=0.0014)
    args = parser.parse_args()
    print(args)

    # Write command to a file
    with open('Cmd_History.txt', 'a') as f:
        f.write("{}\t{}\n".format(str(datetime.datetime.utcnow()), str(args)))

    data_folder = args.data
    pred_result_folder = 'predict_results'
    if not os.path.exists(pred_result_folder):
        os.makedirs(pred_result_folder)
    saved_model_folder = 'saved_models'
    batch_size = args.batchsize
    max_queue_size = args.maxqueuesize
    epoch_num = args.epoch

    derm_image_folder = os.path.join(data_folder, 'ISIC_2019_Training_Input')
    ground_truth_file = os.path.join(data_folder, 'ISIC_2019_Training_GroundTruth.csv')
    df_ground_truth, category_names = load_isic_data(derm_image_folder, ground_truth_file)
    known_category_num = len(category_names)
    df_train, df_val = train_validation_split(df_ground_truth)
    class_weight_dict, _ = compute_class_weight_dict(df_train)

    out_dist_image_folder = os.path.join(data_folder, 'ISIC_Archive_Out_Distribtion')
    out_dist_pred_result_folder = 'out_dist_predict_results'

    # Models used to predict validation set
    models_to_predict_val = []

    # Train Vanilla CNN
    if args.vanilla:
        input_size_vanilla = (224, 224)
        if not args.skiptraining:
            train_vanilla(df_train, df_val, known_category_num, class_weight_dict, batch_size, max_queue_size, epoch_num, input_size_vanilla)
        models_to_predict_val.append({'model_name': 'Vanilla',
                                      'input_size': input_size_vanilla,
                                      'preprocessing_function': VanillaClassifier.preprocess_input})
    
    # Train models by Transfer Learning
    if args.transfer_models:
        model_param_map = get_transfer_model_param_map()
        base_model_params = [model_param_map[x] for x in args.transfer_models]
        if not args.skiptraining:
            train_transfer_learning(base_model_params, df_train, df_val, known_category_num, class_weight_dict, batch_size, max_queue_size, epoch_num)
        for base_model_param in base_model_params:
            models_to_predict_val.append({'model_name': base_model_param.class_name,
                                        'input_size': base_model_param.input_size,
                                        'preprocessing_function': base_model_param.preprocessing_func})

    # Predict validation set
    if not args.skippredict:
        workers = os.cpu_count()
        postfixes = ['best_balanced_acc', 'best_loss', 'latest']
        for postfix in postfixes:
            for m in models_to_predict_val:
                model_filepath = os.path.join(saved_model_folder, "{}_{}.hdf5".format(m['model_name'], postfix))
                if os.path.exists(model_filepath):
                    print("===== Predict validation set using \"{}_{}\" model =====".format(m['model_name'], postfix))
                    model = load_model(filepath=model_filepath, custom_objects={'balanced_accuracy': balanced_accuracy(known_category_num)})
                    LesionClassifier.predict_dataframe(model=model, df=df_val,
                                                       category_names=category_names,
                                                       augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(m['input_size']),
                                                       preprocessing_function=m['preprocessing_function'],
                                                       workers=workers,
                                                       save_file_name=os.path.join(pred_result_folder, "{}_{}.csv").format(m['model_name'], postfix))
                    del model
                    K.clear_session()
                else:
                    print("\"{}\" doesn't exist".format(model_filepath))

    # Compute ODIN Softmax Scores
    if not args.skipodin:
        compute_odin_softmax_scores(pred_result_folder=pred_result_folder, derm_image_folder=derm_image_folder,
                                    out_dist_pred_result_folder=out_dist_pred_result_folder, out_dist_image_folder=out_dist_image_folder,
                                    saved_model_folder=saved_model_folder,
                                    num_classes=known_category_num,
                                    temperature=args.temperature,
                                    noise_magnitude=args.magnitude)

    # Shutdown
    if args.autoshutdown:
        os.system("sudo shutdown -h +2")


def get_transfer_model_param_map():
    base_model_params = {
        'DenseNet201': BaseModelParam(module_name='keras.applications.densenet',
                                      class_name='DenseNet201',
                                      input_size=(224, 224),
                                      preprocessing_func=preprocess_input_trainset),
        'Xception': BaseModelParam(module_name='keras.applications.xception',
                                   class_name='Xception',
                                   input_size=(299, 299),
                                   preprocessing_func=preprocess_input_xception),
        'NASNetLarge': BaseModelParam(module_name='keras.applications.nasnet',
                                      class_name='NASNetLarge',
                                      input_size=(331, 331),
                                      preprocessing_func=preprocess_input_nasnet),
        'InceptionResNetV2': BaseModelParam(module_name='keras.applications.inception_resnet_v2',
                                            class_name='InceptionResNetV2',
                                            input_size=(299, 299),
                                            preprocessing_func=preprocess_input_inception_resnet_v2),
        'ResNeXt50': BaseModelParam(module_name='keras_applications.resnext',
                                    class_name='ResNeXt50',
                                    input_size=(224, 224),
                                    preprocessing_func=preprocess_input_trainset)
    }
    return base_model_params


def train_vanilla(df_train, df_val, known_category_num, class_weight_dict, batch_size, max_queue_size, epoch_num, input_size):
    workers = os.cpu_count()

    classifier = VanillaClassifier(
        input_size=input_size,
        image_data_format=K.image_data_format(),
        num_classes=known_category_num,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        class_weight=class_weight_dict,
        metrics=[balanced_accuracy(known_category_num), 'accuracy'],
        image_paths_train=df_train['path'].tolist(),
        categories_train=np_utils.to_categorical(df_train['category'], num_classes=known_category_num),
        image_paths_val=df_val['path'].tolist(),
        categories_val=np_utils.to_categorical(df_val['category'], num_classes=known_category_num)
    )
    classifier.model.summary()
    print('Begin to train Vanilla CNN')
    classifier.train(epoch_num=epoch_num, workers=workers)
    del classifier
    K.clear_session()


def train_transfer_learning(base_model_params, df_train, df_val, known_category_num, class_weight_dict, batch_size, max_queue_size, epoch_num):
    workers = os.cpu_count()

    for model_param in base_model_params:
        classifier = TransferLearnClassifier(
            base_model_param=model_param,
            fc_layers=[512], # e.g. [512]
            num_classes=known_category_num,
            dropout=0.3, # e.g. 0.3
            batch_size=batch_size,
            max_queue_size=max_queue_size,
            image_data_format=K.image_data_format(),
            metrics=[balanced_accuracy(known_category_num), 'accuracy'],
            class_weight=class_weight_dict,
            image_paths_train=df_train['path'].tolist(),
            categories_train=np_utils.to_categorical(df_train['category'], num_classes=known_category_num),
            image_paths_val=df_val['path'].tolist(),
            categories_val=np_utils.to_categorical(df_val['category'], num_classes=known_category_num)
        )
        classifier.model.summary()
        print("Begin to train {}".format(model_param.class_name))
        classifier.train(epoch_num=epoch_num, workers=workers)
        del classifier
        K.clear_session()


def compute_odin_softmax_scores(pred_result_folder, derm_image_folder, out_dist_pred_result_folder, out_dist_image_folder, saved_model_folder, num_classes,
                                temperature, noise_magnitude):
    softmax_score_folder = 'softmax_scores'
    if not os.path.exists(softmax_score_folder):
        os.makedirs(softmax_score_folder)

    ModelAttr = NamedTuple('ModelAttr', [('model_name', str), ('postfix', str)])
    model_names = ['DenseNet201', 'Xception', 'ResNeXt50']
    postfixes = ['best_balanced_acc', 'best_loss', 'latest']
    distributions = ['In', 'Out']
    model_param_map = get_transfer_model_param_map()
    image_data_format = K.image_data_format()

    for modelattr in (ModelAttr(x, y) for x in model_names for y in postfixes):
        # Load model
        model_filepath = os.path.join(saved_model_folder, "{}_{}.hdf5".format(modelattr.model_name, modelattr.postfix))
        print('Load model: ', model_filepath)
        model = load_model(filepath=model_filepath, custom_objects={'balanced_accuracy': balanced_accuracy(num_classes)})

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
                    f.write("{}, {}, {}\n".format(temperature, noise_magnitude, softmax_score))

            ### Define Keras Functions
            # Compute loss based on model's outputs of last two layers and temperature scaling
            scaled_dense_pred_output = model.get_layer('dense_pred').output / temperature
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
                perturbative_image = image - noise_magnitude * perturbations
                
                # Calculate the confidence after adding perturbations
                dense_pred_output = get_dense_pred_layer_output([perturbative_image, learning_phase])[0]
                dense_pred_output = dense_pred_output / temperature
                softmax_probs = softmax(dense_pred_output)
                softmax_score = np.max(softmax_probs)

                f.write("{}, {}, {}\n".format(temperature, noise_magnitude, softmax_score))
            f.close()
        del model
        K.clear_session()

if __name__ == '__main__':
    main()
