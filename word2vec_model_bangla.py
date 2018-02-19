import numpy as np
from gensim import models
from nltk.tokenize import word_tokenize
import re
import string
import logging
from datetime import datetime

logging.basicConfig(filename='data/word_embeddings/log_of_bangla_wv.txt', level=logging.INFO)




def get_all_bangla_text():
    bangla_corpus = []
    for i in range(3, 8):
        file_name = 'data/bangla_newspaper/paper_201' + str(i) + '.txt'
        with open(file_name, 'r', encoding='utf8') as file:
            bangla_corpus += file.readlines()
    return bangla_corpus

def count_word(text_list):
    counter = 0
    for line in text_list:
        counter += len(line.split())
    return counter

def clean_str(text):
    text = re.sub("\d", "", text)           #remove decimal number
    text = re.sub('[a-zA-Z]', '', text)     #remove english word
    text = re.sub('[।]', ' ', text)         #remove ।
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text



bangla_corpus = get_all_bangla_text()
bangla_corpus = [clean_str(i) for i in bangla_corpus]
number_of_words = count_word(bangla_corpus)
print('total number of words: ', number_of_words)

text = [word_tokenize(i) for i in bangla_corpus]

model = models.Word2Vec(text, size=50, window=5, min_count=4, sg=1)
model.wv.save_word2vec_format('data/word_embeddings/bangla_sg_50.txt', binary=False)
print('bangla model created')

logging.info('present word vectors is created named bangla_wv_cbow_window5_min4 at %s...' %datetime.now() )

print('its done...')

# model = models.KeyedVectors.load_word2vec_format('bangla_word2vec.txt', binary=False)
# print('word2vec model loaded')
# word_vec = model.most_similar('চেয়ারম্যান')
# print(word_vec)
