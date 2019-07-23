import argparse
import os
import tensorflow as tf
# from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras_applications.resnext import preprocess_input as preprocess_input_resnext
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.models import load_model
from utils import preprocess_input as preprocess_input_trainset
from keras import backend as K
from keras.utils import np_utils
from data import load_isic_data, train_validation_split, compute_class_weight_dict
from vanilla_classifier import VanillaClassifier
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import BaseModelParam
from lesion_classifier import LesionClassifier

def main():
    parser = argparse.ArgumentParser(description='ISIC-2019 Skin Lesion Classifiers')
    parser.add_argument('data', metavar='DIR', help='path to data foler')
    parser.add_argument('--batchsize', type=int, help='Batch size (default: %(default)s)', default=64)
    parser.add_argument('--maxqueuesize', type=int, help='Maximum size for the generator queue (default: %(default)s)', default=10)
    parser.add_argument('--epoch', type=int, help='Number of epochs', required=True)
    parser.add_argument('--vanilla', dest='vanilla', action='store_true', help='Train Vanilla CNN')
    parser.add_argument('--transfer', dest='transfer_models', nargs='*', help='Models for Transfer Learning')
    parser.add_argument('--gpus', type=int, help='Number of GPUs')
    parser.add_argument('--autoshutdown', dest='autoshutdown', action='store_true', help='Automatically shutdown the computer after training is done')
    args = parser.parse_args()
    print(args)

    # Check if GPU available
    gpus = args.gpus
    if gpus is not None and not tf.test.is_gpu_available():
        print('GPU is not available!')
        gpus = None

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
    df_train, df_val = train_validation_split(df_ground_truth)
    class_weight_dict = compute_class_weight_dict(df_train)

    # Models used to predict validation set
    models_to_predict_val = []

    # Train Vanilla CNN
    if args.vanilla:
        input_size_vanilla = (224, 224)
        train_vanilla(df_train, df_val, len(category_names), class_weight_dict, batch_size, max_queue_size, epoch_num, input_size_vanilla)
        models_to_predict_val.append({'model_name': 'Vanilla',
                                      'input_size': input_size_vanilla,
                                      'preprocessing_function': VanillaClassifier.preprocess_input})
    
    # Train models by Transfer Learning
    if args.transfer_models:
        model_param_map = get_transfer_model_param_map()
        base_model_params = [model_param_map[x] for x in args.transfer_models]
        train_transfer_learning(base_model_params, df_train, df_val, len(category_names), class_weight_dict, batch_size, max_queue_size, epoch_num, gpus)
        for base_model_param in base_model_params:
            models_to_predict_val.append({'model_name': base_model_param.class_name,
                                        'input_size': base_model_param.input_size,
                                        'preprocessing_function': base_model_param.preprocessing_func})

    # Predict validation set
    workers = os.cpu_count()
    postfixes = ['best_balanced_acc', 'best_loss', 'latest']
    for postfix in postfixes:
        for m in models_to_predict_val:
            print("Predict validation set using {} model".format(m['model_name']))
            model = load_model(filepath=os.path.join(saved_model_folder, "{}_{}.hdf5".format(m['model_name'], postfix)),
                            custom_objects={'balanced_accuracy': balanced_accuracy})
            LesionClassifier.predict_dataframe(model=model, df=df_val,
                                            category_names=category_names,
                                            augmentation_pipeline=LesionClassifier.create_aug_pipeline_val(m['input_size']),
                                            preprocessing_function=m['preprocessing_function'],
                                            workers=workers,
                                            save_file_name=os.path.join(pred_result_folder, "{}_{}.csv").format(m['model_name'], postfix))
    
    # Shutdown
    if args.autoshutdown:
        print('Shutdown in 5 minutes! It can be canceled by executing "shutdown -c"')
        os.system("sudo shutdown -h +5")


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
                                      preprocessing_func=preprocess_input_resnext)
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
        metrics=[balanced_accuracy, 'accuracy'],
        image_paths_train=df_train['path'].tolist(),
        categories_train=np_utils.to_categorical(df_train['category'], num_classes=known_category_num),
        image_paths_val=df_val['path'].tolist(),
        categories_val=np_utils.to_categorical(df_val['category'], num_classes=known_category_num)
    )
    classifier.model.summary()
    print('Begin to train Vanilla CNN')
    classifier.train(epoch_num=epoch_num, class_weight=class_weight_dict, workers=workers)
    del classifier


def train_transfer_learning(base_model_params, df_train, df_val, known_category_num, class_weight_dict, batch_size, max_queue_size, epoch_num, gpus=None):
    workers = os.cpu_count()

    for model_param in base_model_params:
        classifier = TransferLearnClassifier(
            base_model_param=model_param,
            fc_layers=[512],
            num_classes=known_category_num,
            dropout=0.3,
            batch_size=batch_size,
            max_queue_size=max_queue_size,
            image_data_format=K.image_data_format(),
            metrics=[balanced_accuracy, 'accuracy'],
            gpus=gpus,
            image_paths_train=df_train['path'].tolist(),
            categories_train=np_utils.to_categorical(df_train['category'], num_classes=known_category_num),
            image_paths_val=df_val['path'].tolist(),
            categories_val=np_utils.to_categorical(df_val['category'], num_classes=known_category_num)
        )
        classifier.model.summary()
        print("Begin to train {}".format(model_param.class_name))
        classifier.train(epoch_num=epoch_num, class_weight=class_weight_dict, workers=workers)
        del classifier


if __name__ == '__main__':
    main()
