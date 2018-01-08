import numpy as np
from gensim import models
from nltk.tokenize import word_tokenize
import re
import string


def clean_str(text):
    text = re.sub("\d", "", text)           #remove decimal number
    text = re.sub('[a-zA-Z]', '', text)     #remove english word
    text = re.sub('[।]', ' ', text)         #remove ।
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

# bangla_text = list(open('paper.txt', encoding='utf8').readlines())
#
#
# text = [clean_str(i) for i in bangla_text]
# text = [word_tokenize(i) for i in text]
#
# model = models.Word2Vec(text, size=50, window=3, min_count=2, workers=4)
# model.wv.save_word2vec_format('bangla_word2vec.txt', binary=False)
# print('bangla model created')

model = models.KeyedVectors.load_word2vec_format('bangla_word2vec.txt', binary=False)
print('word2vec model loaded')
word_vec = model.most_similar('চেয়ারম্যান')
print(word_vec)
