import pandas as pd
import numpy as np
import Text_preprocessor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



review_dataset = 'data/restaurant.csv'

number_of_category = 5
train_percentage = 0.8

def evaluate_model(target_true,target_predicted):
    print (classification_report(target_true,target_predicted))
    print ("The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted)))

def get_tfidf_value(data):
    count_vectorizer = CountVectorizer(binary='true', stop_words='english')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data).toarray()
    return tfidf_data

def get_data_and_lebel():
    reviews = pd.read_csv(review_dataset, encoding='ISO-8859-1')

    x = reviews['text'].values
    y = reviews['category'].values

    my_set = list(sorted(set(y)))
    label = np.zeros(len(my_set))
    # if y[1] == 'food':
    #     ind = my_set.index('food')
    #     label[ind] = 1

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
# mlb = MultiLabelBinarizer()
# y_enc = mlb.fit_transform(y)
x = [Text_preprocessor.clean_english_string(text) for text in x]
x = get_tfidf_value(x)
y = np.array(y)
y = y.astype(int)

train_len = int(len(x) * train_percentage)
x_train = x[:train_len]
y_train = y[:train_len]
x_test = x[train_len:]
y_test = y[train_len:]
print('train test ready')

from sklearn.preprocessing import MultiLabelBinarizer
# Y = MultiLabelBinarizer().fit_transform(yvalues)

clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

evaluate_model(y_test, predictions)

print('going well ...')

print('for random forest....')
cls = RandomForestClassifier()
cls.fit(x_train, y_train)
pred = cls.predict(x_test)
evaluate_model(y_test, pred)

print('KNN...')
cls = KNeighborsClassifier(n_neighbors=3)
cls.fit(x_train, y_train)
pred = cls.predict(x_test)
evaluate_model(y_test, pred)

