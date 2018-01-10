from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize
import numpy as np
import re
import string
from nltk.stem import PorterStemmer
import pandas as pd
from gensim import models

with open('stop_words.txt', 'r') as f:
    stop_words = [line.rstrip() for line in f]

def clean_str(text):

    text = re.sub("\d", "", text)           #remove decimal number
    translator = str.maketrans('', '', string.punctuation)   #remove punctuation
    text = text.translate(translator)

    text = word_tokenize(text)
    filtered_words = text[:]  # make a copy of the word_list
    for word in text:  # iterate over word_list
        if word in stop_words:
            filtered_words.remove(word)

    return filtered_words


def stemming(str):
    stemmed_string = []
    stemmer = PorterStemmer()
    for word in str:
        stemmed_string.append(stemmer.stem(word))
    return stemmed_string

def word_counter(text):
    text = text.lower()
    text = clean_str(text)

    # text = stemming(text)
    word_count = Counter(text)
    return word_count

def add_word_count(list_of_dictionary):
    length = len(list_of_dictionary)
    all_word = list_of_dictionary[0]
    for i in  range(1, length):
        for key1, value1 in all_word.items():
            found = 0
            for key2, value2 in list_of_dictionary[i].items():
                if key1 == key2:
                    all_word[key1] = value1 + value2
                    found = 1
                    break
            if found == 0:
                all_word[key1] = value1
    return all_word

def substract_word_count(dictionary1, dictionary2):
    word_dict = {}
    for key1, value1 in dictionary1.items():
        found = 0
        for key2, value2 in dictionary2.items():
            if key1 == key2:
                word_dict[key1] = value1 - value2
                found = 1
                break
        if found == 0:
            word_dict[key1] = value1
    return word_dict

def divide_single_label(selected_label, x, y):
    selected_text = []
    other_text = []
    all_labels = list(sorted(set(y)))

    for index, label in enumerate(y):
        if label == selected_label:
            selected_text.append(x[index])

    for review in x:
        found = 0
        for selected_review in selected_text:
            if review == selected_review:
                found = 1
        if found == 0:
            other_text.append(review)

    return selected_text, other_text


def useful_word_for_label(label, x, y):
    selected_text_list, other_text_list = divide_single_label(label, x, y)
    selected_text = ''
    other_text = ''

    for text in selected_text_list:

        selected_text += text
        selected_text += ' '

    for text in other_text_list:

        other_text += text
        other_text += ' '

    selected_count = word_counter(selected_text)
    other_count = word_counter(other_text)
    substracted_values = substract_word_count(selected_count, other_count)
    useful_words = OrderedDict(sorted(substracted_values.items(), key=lambda kv: kv[1], reverse=True))
    # print('useful words with count: ', useful_words)

    top_useful_words = list(useful_words.keys())[:10]

    return top_useful_words


reviews = pd.read_csv('restaurant.csv')

x = reviews['text'].values
y = reviews['category'].values
length_of_corpus = len(x)
labels = list(sorted(set(y)))
doc_of_classes = ['']*len(labels)

text_of_labels = []
for label in labels:
    useful_words = useful_word_for_label(label, x, y)
    text_of_labels.append(useful_words)


test_review = 'I just wonder how you can have such a delicious meal for such little money.'
test_review = test_review.lower()
test_review = clean_str(test_review)
# test_review = stemming(test_review)

distances_from_food = []
distance_from_price = []
label_distance = {}
word2vec_model = models.KeyedVectors.load_word2vec_format('glove.txt', binary=False)

for i in range(0, len(labels)):
    most_similar_word = ''
    most_similar_distance = 0.0
    for base_word in text_of_labels[i]:
        for word in test_review:
            similarity = word2vec_model.wv.similarity(word, base_word)
            if similarity > most_similar_distance:
                most_similar_distance = similarity
                most_similar_word = word
    most_similar_distance = format(most_similar_distance, '.3f')
    label_distance[most_similar_distance] = most_similar_word


print('ok')


