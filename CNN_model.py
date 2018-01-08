#source: https://github.com/keras-team/keras/tree/master/examples
#keras source: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

import pandas as pd
import numpy as np
import logging

import keras.backend as K
import keras

from keras.preprocessing import sequence
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam


# max_features = 200  # number of words we want to keep
# maxlen = 100  # max length of the comments in the model
batch_size = 10  # batch size for the model
embedding_dims = 50  # dimension of the hidden variable, i.e. the embedding dimension
filters = 32
kernel_size = 3
model_type = 'cnn_static'

logging.basicConfig(filename='log.txt', level=logging.INFO)

# source: https://keras.io/metrics/     and   https://www.quora.com/How-do-I-customize-the-decision-threshold-in-Keras-for-training-on-imbalanced-data
def accuracy_with_threshold(y_true, y_pred):
   threshold = 0.7
   y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
   return K.mean(K.equal(y_true, y_pred))


def get_data_and_lebel():
    reviews = pd.read_csv('restaurant.csv')

    x = reviews['text'].values
    y = reviews['category'].values

    my_set = list(sorted(set(y)))
    label = np.zeros(len(my_set))
    if y[1] == 'food':
        ind = my_set.index('food')
        label[ind] = 1
    print(label)

    prev_rev = ''
    prev_lebel = []
    my_review = []
    my_label = []
    i = 0
    for review in x:

        if review == prev_rev:
            label = prev_lebel
            for index, category in enumerate(my_set):
                if y[i] == category:
                    label[index] = 1
            my_label[-1] = label
            i += 1
            prev_lebel = label
            continue

        my_review.append(review)
        label = np.zeros(len(my_set))

        for index, category in enumerate(my_set):
            if y[i] == category:
                label[index] = 1

        my_label.append(label)
        i += 1
        prev_rev = review
        prev_lebel = label
    return my_review, my_label


x, y = get_data_and_lebel()
max_document_length = max([len(text.split(" ")) for text in x])
x = np.array(x)
y = np.array(y)

train_len = int(len(x) * 0.9)
x_train = x[:train_len]
y_train = y[:train_len]
x_test = x[train_len:]
y_test = y[train_len:]

test_len = len(x_test) -5


tok = Tokenizer()
tok.fit_on_texts(list(x_train) + list(x_test))
vocab_size = len(tok.word_index) +1
word_index = tok.word_index
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))


x_train = sequence.pad_sequences(x_train, maxlen=max_document_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_document_length)


x_small = x_test[test_len:]
y_small = y_test[test_len:]


print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# using word2vec
embeddings_index = {}
f = open('glove_sample.txt', 'r', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.uniform(-0.5, 0.5, embedding_dims)


my_model = Sequential()
if model_type == 'cnn_static':
    em = Embedding(len(word_index)+1, embedding_dims, weights=[embedding_matrix], input_length=max_document_length, trainable=False)
else:
    em = Embedding(vocab_size, embedding_dims, input_length=max_document_length)

my_model.add(em)
my_model.add(Dropout(0.2))
my_model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
my_model.add(GlobalMaxPooling1D())
my_model.add(Dense(5, activation='sigmoid'))
my_model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

hist = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))
print(hist.history)
logging.info(hist.history)


preds = my_model.predict(x_small)
# preds[preds>=0.5] = 1
# preds[preds<0.5] = 0
print(preds)

acc = my_model.evaluate(x_test, y_test)
testing = my_model.metrics_names
print('metrics name: ', testing)

# print('Test score:', score)
print('Test accuracy:', acc)

# print ('keras prediction', K.eval(keras.metrics.binary_accuracy(x_small, y_small)))

# logging.info('Test score: %f and Test accuracy: %f' %(score, acc))


y_small = y_small.astype(np.float32)
my_accuracy = accuracy_with_threshold(y_small, preds)
print('my accuracy with threshold: ', K.eval(my_accuracy))



print('its done...')