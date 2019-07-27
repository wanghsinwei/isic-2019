import keras.backend as K
import tensorflow as tf

def balanced_accuracy(y_true, y_pred):
    """
    Calculates the mean of the per-class accuracies.
    Same as sklearn.metrics.balanced_accuracy_score and sklearn.metrics.recall_score with macro average
    # References
        https://stackoverflow.com/a/45947435/2437361
        https://stackoverflow.com/a/52163410/2437361
    """
    y_true_argmax = K.argmax(y_true, axis=1)
    y_pred_argmax = K.argmax(y_pred, axis=1)
    mean_accuracy, update_op = tf.metrics.mean_per_class_accuracy(y_true_argmax, y_pred_argmax, 8) # 8 Know skin lesion categories
    K.get_session().run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    with tf.control_dependencies([update_op]):
       mean_accuracy = tf.identity(mean_accuracy)
    
    return mean_accuracy