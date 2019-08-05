import argparse
import os
import datetime
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
from data import load_isic_data, train_validation_split, compute_class_weight_dict
from vanilla_classifier import VanillaClassifier
from transfer_learn_classifier import TransferLearnClassifier
from metrics import balanced_accuracy
from base_model_param import get_transfer_model_param_map
from lesion_classifier import LesionClassifier
from odin import compute_baseline_softmax_scores, compute_odin_softmax_scores

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
    args = parser.parse_args()
    print(args)

    # Write command to a file
    with open('Cmd_History.txt', 'a') as f:
        f.write("{}\t{}\n".format(str(datetime.datetime.utcnow()), str(args)))

    data_folder = args.data
    pred_result_folder = 'predict_results'
    os.makedirs(pred_result_folder, exist_ok=True)
    saved_model_folder = 'saved_models'
    softmax_score_folder = 'softmax_scores'
    os.makedirs(softmax_score_folder, exist_ok=True)
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
                                                       batch_size=batch_size,
                                                       workers=workers,
                                                       save_file_name=os.path.join(pred_result_folder, "{}_{}.csv").format(m['model_name'], postfix))
                    del model
                    K.clear_session()
                else:
                    print("\"{}\" doesn't exist".format(model_filepath))

    # Compute Baseline and ODIN Softmax Scores
    if not args.skipodin:
        compute_baseline_softmax_scores(pred_result_folder=pred_result_folder,
                                        out_dist_pred_result_folder=out_dist_pred_result_folder,
                                        softmax_score_folder=softmax_score_folder)
                                        
        compute_odin_softmax_scores(pred_result_folder=pred_result_folder, derm_image_folder=derm_image_folder,
                                    out_dist_pred_result_folder=out_dist_pred_result_folder, out_dist_image_folder=out_dist_image_folder,
                                    saved_model_folder=saved_model_folder, softmax_score_folder=softmax_score_folder,
                                    num_classes=known_category_num, batch_size=batch_size)

    # Shutdown
    if args.autoshutdown:
        os.system("sudo shutdown -h +2")


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


if __name__ == '__main__':
    main()
