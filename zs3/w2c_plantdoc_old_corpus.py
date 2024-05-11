from gensim.models import Word2Vec
import numpy as np

corpus = [["background", "is", "everything", "that", "is", "not", "the", "plant"],
          ["diseased", "plant", "area", "shows", "signs", "of", "disease"],
          ["healthy", "plant", "area", "is", "green", "and", "vibrant"]]
model = Word2Vec(corpus, vector_size=300, window=5, min_count=1, workers=4)
embeddings = {word: model.wv[word] for word in model.wv.key_to_index}

np.save('/Users/alex/Documents/GitHub/ZS3/zs3/embeddings/plantdoc/plantdoc_class_w2c.npy', embeddings)
