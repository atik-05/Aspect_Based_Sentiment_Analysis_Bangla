from gensim import models
import numpy as np

str = 'THis is AR rahman'
print(str.lower())



from nltk.corpus import stopwords
# ...
filtered_words = [word for word in word_list if word not in stopwords.words('english')]