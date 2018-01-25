import tensorflow as tf
import numpy as np
import keras.backend as K

def f1_score(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # How many selected items are relevant?
    precision = c1 / c2
    return precision


def my_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def my_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


y_true = []
y_true.append([0.,0.,1.])
y_true.append([1.,1.,0.])
y_true.append([0.,0.,1.])
y_true = np.array(y_true)
# y_true.astype(np.float32)


y_pred = []
y_pred.append([1.,0.,1.])
y_pred.append([1.,1.,0.])
y_pred.append([1.,0.,1.])
y_pred = np.array(y_pred)
# y_pred.astype(np.float32)


y_pred = tf.convert_to_tensor(y_pred, np.float32)
y_true = tf.convert_to_tensor(y_true, np.float32)

pr = my_precision(y_true, y_pred)
re = my_recall(y_true, y_pred)
print('precesion: %.3f, recall: %.3f' %(K.eval(pr), K.eval(re)))
print('its done ...')
