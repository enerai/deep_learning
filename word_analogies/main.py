import numpy as np
from w2v_utils import *
from cosine_similarity import *
from complete_analogy import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

complete_analogy(word_a, word_b, word_c, word_to_vec_map)
