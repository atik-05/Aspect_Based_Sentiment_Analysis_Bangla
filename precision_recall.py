import tensorflow as tf
import numpy as np
import keras.backend as K

# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def f1_score(y_true, y_pred):
#     pr = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     f1_score = 2 * (pr * rec) / (pr + rec)
#     return f1_score
#
# y_true = []
# y_true.append([0.2,0.,1.])
# y_true.append([1.,1.,0.])
# y_true.append([1.,0.2,1.])
#
# y_true = np.array(y_true)
# # y_true.astype(np.float32)
#
#
# y_pred = []
# y_pred.append([0.3,0.,1.])
# y_pred.append([0.,1.,0.51])
# y_pred.append([0.,0.9,0.])
#
# y_pred = np.array(y_pred)
# # y_pred.astype(np.float32)
#
#
# y_pred = tf.convert_to_tensor(y_pred, np.float32)
# y_true = tf.convert_to_tensor(y_true, np.float32)
#
# pr = precision(y_true, y_pred)
# re = recall(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# print('For the full model: precesion: %.3f, recall: %.3f, f1_score: %.3f' %(K.eval(pr), K.eval(re), K.eval(f1)))


print('\nIndividual class evaluation....\n')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
con_matrix = confusion_matrix(y_true, y_pred)
print(con_matrix)

target_names = ['class 0', 'class 1', 'class 2']

report = classification_report(y_true, y_pred, target_names=target_names)
print(report)


print('its done ...')
