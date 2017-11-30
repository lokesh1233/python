import gensim
# includes lots of NLP applications. Use Cython for speed UP
# create data folder if it doesn't exist
import os
if os.path.exists("data/"):
    os.makedirs("data/")

# load word vectors
w2v = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
dog = w2v['dog']
print()