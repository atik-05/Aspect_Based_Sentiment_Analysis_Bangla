import numpy as np
#
# model = KeyedVectors.load_word2vec_format('data/google_news.bin', binary=True)
# model.save_word2vec_format('data/google_news_text.txt', binary=False)
# print('loaded....')

# embeddings_index = {}
# file = open('data/google_news_text.txt', 'r', encoding='utf8')
# file = list(file)
# file.pop(0)
#
# for line in file:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# file.close()


embedding_matrix = np.zeros((5, 10))
print(embedding_matrix)
print('test')
