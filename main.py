import argparse
import os
from keras.applications.densenet import DenseNet201, preprocess_input as preprocess_input_densenet
from keras.applications.xception  import Xception, preprocess_input as preprocess_input_xception
from keras.applications.nasnet import NASNetLarge, preprocess_input as preprocess_input_nasnet
from keras import backend as K
from keras.utils import np_utils
from data import load_isic_data, train_validation_split, compute_class_weight_dict
from vanilla_classifier import VanillaClassifier
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import BaseModelParam

def main():
    parser = argparse.ArgumentParser(description='ISIC-2019 Skin Lesion Classifiers')
    parser.add_argument('data', metavar='DIR', help='path to data foler')
    parser.add_argument('--batchsize', type=int, help='Batch size (default: %(default)s)', default=40)
    parser.add_argument('--epoch', type=int, help='Number of epochs', required=True)
    parser.add_argument('--vanilla', dest='vanilla', action='store_true', help='Vanilla CNN')
    parser.add_argument('--transfer', dest='transfer_learning', action='store_true', help='Transfer Learning')
    parser.add_argument('--finetune', dest='fine_tuning', action='store_true', help='Fine-Tuning Transfer Learning')
    args = parser.parse_args()
    print(args)
    # return

    data_folder = args.data
    batch_size = args.batchsize
    epoch_num = args.epoch

    derm_image_folder = os.path.join(data_folder, 'ISIC_2019_Training_Input')
    ground_truth_file = os.path.join(data_folder, 'ISIC_2019_Training_GroundTruth.csv')
    df_ground_truth, category_names = load_isic_data(derm_image_folder, ground_truth_file)
    df_train, df_val = train_validation_split(df_ground_truth)
    class_weight_dict = compute_class_weight_dict(df_train)

    if args.vanilla:
        train_vanilla(df_train, df_val, len(category_names), class_weight_dict, batch_size, epoch_num)
    if args.transfer_learning:
        train_transfer_learning(df_train, df_val, len(category_names), class_weight_dict, batch_size, epoch_num, args.fine_tuning)

def train_vanilla(df_train, df_val, known_category_num, class_weight_dict, batch_size, epoch_num):
    input_size = (224, 224)
    rescale=1./255
    workers = os.cpu_count()

    classifier = VanillaClassifier(
        input_size=input_size,
        image_data_format=K.image_data_format(),
        num_classes=known_category_num,
        batch_size=batch_size,
        rescale=rescale,
        metrics=[balanced_accuracy, 'accuracy'],
        image_paths_train=df_train['path'].tolist(),
        categories_train=np_utils.to_categorical(df_train['category'], num_classes=known_category_num),
        image_paths_val=df_val['path'].tolist(),
        categories_val=np_utils.to_categorical(df_val['category'], num_classes=known_category_num)
    )
    classifier.model.summary()
    classifier.train(epoch_num=epoch_num, class_weight=class_weight_dict, workers=workers)


def train_transfer_learning(df_train, df_val, known_category_num, class_weight_dict, batch_size, epoch_num, fine_tuning=True):
    workers = os.cpu_count()

    base_model_params = [
        BaseModelParam(module_name='keras.applications.densenet',
                    class_name='DenseNet201',
                    input_size=(224, 224),
                    layers_trainable=fine_tuning,
                    preprocessing_func=preprocess_input_densenet),
        BaseModelParam(module_name='keras.applications.xception',
                    class_name='Xception',
                    input_size=(299, 299),
                    layers_trainable=fine_tuning,
                    preprocessing_func=preprocess_input_xception),
        BaseModelParam(module_name='keras.applications.nasnet',
                    class_name='NASNetLarge',
                    input_size=(331, 331),
                    layers_trainable=fine_tuning,
                    preprocessing_func=preprocess_input_nasnet)
    ]

    for model_param in base_model_params:
        classifier = TransferLearnClassifier(
            base_model_param=model_param,
            fc_layers=[],
            num_classes=known_category_num,
            dropout=None,
            batch_size=batch_size,
            image_data_format=K.image_data_format(),
            metrics=[balanced_accuracy, 'accuracy'],
            image_paths_train=df_train['path'].tolist(),
            categories_train=np_utils.to_categorical(df_train['category'], num_classes=known_category_num),
            image_paths_val=df_val['path'].tolist(),
            categories_val=np_utils.to_categorical(df_val['category'], num_classes=known_category_num)
        )
        # classifier.model.summary()
        classifier.train(epoch_num=epoch_num, class_weight=class_weight_dict, workers=workers)

if __name__ == '__main__':
    main()