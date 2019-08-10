import keras.backend as K
import tensorflow as tf

def balanced_accuracy(num_classes):
    """
    Calculates the mean of the per-class accuracies.
    Same as sklearn.metrics.balanced_accuracy_score and sklearn.metrics.recall_score with macro average
    
    # References
        https://stackoverflow.com/a/41717938/2437361
        https://stackoverflow.com/a/50266195/2437361
    """
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        class_acc_total = 0
        seen_classes = 0
        
        for c in range(num_classes):
            accuracy_mask = K.cast(K.equal(class_id_true, c), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * accuracy_mask
            accuracy_mask_sum = K.sum(accuracy_mask)
            class_acc = K.cast(K.sum(class_acc_tensor) / K.maximum(accuracy_mask_sum, 1), K.floatx())
            class_acc_total += class_acc
            
            condition = K.equal(accuracy_mask_sum, 0)
            seen_classes = K.switch(condition, seen_classes, seen_classes+1)
            
        return class_acc_total / K.cast(seen_classes, K.floatx())
    fn.__name__ = 'balanced_accuracy'
    return fn
