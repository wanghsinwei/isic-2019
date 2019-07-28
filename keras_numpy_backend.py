"""Clone some funtions from Keras.
# References https://github.com/keras-team/keras/blob/master/keras/backend/numpy_backend.py
"""

import numpy as np

def categorical_crossentropy(target, output, class_weights=None, from_logits=False):
    if from_logits:
        output = softmax(output)
    else:
        output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    if class_weights is None:
        return np.sum(target * -np.log(output), axis=-1, keepdims=False)
    else:
        return np.sum(target * -np.log(output) * class_weights, axis=-1, keepdims=False)

def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)