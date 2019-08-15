from typing import NamedTuple
from types import FunctionType
# from keras.applications.densenet import preprocess_input as preprocess_input_densenet
# from keras_applications.resnext import preprocess_input as preprocess_input_resnext
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from utils import preprocess_input as preprocess_input_trainset, preprocess_input_2 as preprocess_input_trainset_2

BaseModelParam = NamedTuple('BaseModelParam', [
    ('module_name', str),
    ('class_name', str),
    ('input_size', tuple),
    ('preprocessing_func', FunctionType)
])

def get_transfer_model_param_map():
    """For approach 1"""
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

def get_transfer_model_param_map_2():
    """For approach 2"""
    base_model_params = {
        'DenseNet201': BaseModelParam(module_name='keras.applications.densenet',
                                      class_name='DenseNet201',
                                      input_size=(224, 224),
                                      preprocessing_func=preprocess_input_trainset_2),
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
                                    preprocessing_func=preprocess_input_trainset_2)
    }
    return base_model_params