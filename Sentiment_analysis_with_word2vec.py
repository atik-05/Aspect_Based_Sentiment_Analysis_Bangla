import numpy as np
from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize
import re
import string
from nltk.stem import PorterStemmer
import pandas as pd
from gensim import models


percentage_threshold = 10

positive_text = ''
negative_text = ''


with open('data/stop_words.txt', 'r') as f:
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


def word_counter(text):
    text = text.lower()
    text = clean_str(text)
    # text = stemming(text)
    word_count = Counter(text)
    return word_count


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


def calculate_percentage(positive_count, negative_count):
    pos_percentage_dict = {}
    neg_percentage_dict = {}
    for key1, value1 in positive_count.items():
        for key2, value2 in negative_count.items():
            if key1 == key2:
                total_count = value1+value2
                pos_percentage_dict[key1] = (value1/total_count)*100
                neg_percentage_dict[key1] = (value2/total_count)*100

    return pos_percentage_dict, neg_percentage_dict


def select_top_percentage(pos_percentage, neg_percentage):
    select_top = []
    for key, value in pos_percentage.items():
        if value>percentage_threshold and neg_percentage[key]>percentage_threshold:
            select_top.append(key)
    return select_top

def find_useful_words(positive_count, negative_count):

    substracted_values = substract_word_count(positive_count, negative_count)
    useful_words = OrderedDict(sorted(substracted_values.items(), key=lambda kv: kv[1], reverse=True))
    # print('useful words with count: ', useful_words)
    top_useful_words = list(useful_words.keys())[:50]
    return top_useful_words

def remove_common_words(positive_count, negative_count, common_words):
    filtered_positive = positive_count
    filtered_negative = negative_count
    for word in list(positive_count.keys()):
        if word in common_words:
            filtered_positive.pop(word)

    for word in list(negative_count.keys()):
        if word in common_words:
            filtered_negative.pop(word)

    return filtered_positive, filtered_negative

def remove_common_test(text, common_words):
    filtered_words = text[:]  # make a copy of the word_list
    for word in text:  # iterate over word_list
        if word in stop_words:
            filtered_words.remove(word)
    return filtered_words

with open('data/pos.txt', 'r') as f:
    positive = f.readlines()

with open('data/neg.txt', 'r') as f:
    negative = f.readlines()

for text in positive:
    positive_text += text
    positive_text += ' '

for text in negative:
    negative_text += text
    negative_text += ' '

positive_count = word_counter(positive_text)
negative_count = word_counter(negative_text)

pos_percentage, neg_percentage = calculate_percentage(positive_count, negative_count)
common_words = select_top_percentage(pos_percentage, neg_percentage)

positive_count, negative_count = remove_common_words(positive_count, negative_count, common_words)

positive_words = find_useful_words(positive_count, negative_count)
# total_count = positive_count[positive_words[97]]+negative_count[positive_words[97]]
# perc = (positive_count[positive_words[97]]/total_count)*100
# print(perc)

negative_words = find_useful_words(negative_count, positive_count)

test_review = 'too sincere to exploit its subjects and too honest to manipulate its audience .'
test_review = test_review.lower()
test_review = clean_str(test_review)
test_review = remove_common_test(test_review, common_words)
# test_review = stemming(test_review)
print(test_review)

distances_from_food = []
distance_from_price = []
label_distance = {}


word2vec_model = models.KeyedVectors.load_word2vec_format('data/glove.txt', binary=False)

print('word2vec loaded')


text_of_labels = []
text_of_labels.append(positive_words)
text_of_labels.append(negative_words)
print(text_of_labels)

matched_words = []
for i in range(0, 2):
    most_similar_word = ''
    matched = ''
    most_similar_distance = 0.0
    for base_word in text_of_labels[i]:
        for word in test_review:

            similarity = word2vec_model.wv.similarity(word, base_word)
            if similarity > most_similar_distance:
                most_similar_distance = similarity
                most_similar_word = word
                matched = base_word
    most_similar_distance = format(most_similar_distance, '.3f')
    label_distance[most_similar_distance] = most_similar_word
    matched_words.append(matched)

print(label_distance)
print(matched_words)

print('done...')

