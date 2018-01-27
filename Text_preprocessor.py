from nltk.tokenize import word_tokenize
import numpy as np
import re
import string
from nltk.stem import PorterStemmer


with open('data/english_stop_words.txt', 'r') as en:
    english_stop_words = [line.rstrip() for line in en]

# bangla_stop_words = list(open('data/bangla_stop_words.txt', encoding='utf8').readline())

with open('data/bangla_stop_words.txt', 'r', encoding='utf8') as bn:
    bangla_stop_words = [line.strip() for line in bn]

def clean_english_string(text):
    text = text.lower()
    text = re.sub("\d", "", text)           #remove decimal number
    translator = str.maketrans('', '', string.punctuation)   #remove punctuation
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in english_stop_words])      #remove stop words
    return text


def clean_bangla_string(text):
    text = re.sub("\d", "", text)           #remove decimal number
    text = re.sub('[a-zA-Z]', '', text)     #remove english word
    text = re.sub('[।]', ' ', text)         #remove ।
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in bangla_stop_words])  # remove stop words
    return text

def stemming(str):
    stemmed_string = []
    stemmer = PorterStemmer()
    for word in str:
        stemmed_string.append(stemmer.stem(word))
    return stemmed_string



