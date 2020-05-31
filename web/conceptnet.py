import requests
import time
import random
import numpy as np

from .embeddings import load_embedding

BASE_URL = 'http://api.conceptnet.io/relatedness?node1=/c/%s/%s&node2=/c/%s/%s'
HARDCODED_EMBEDDINGS_PATH = '/Users/alexpulich/wrk/uni/nlp/numberbatch_wo_prefix_fin.txt'


def get_similarity(word1, word2, lang='th'):
    score = requests.get(BASE_URL % (lang, word1, lang, word2)).json()['value']
    time.sleep(random.randint(0, 3))
    return score if score > 0 else None


def get_similarity_from_dict(scores, word1, word2):
    return scores.get(word1 + word2)


class ConceptNetNumberbatch:
    def __init__(self):
        self.w = load_embedding(HARDCODED_EMBEDDINGS_PATH,
                                format='word2vec',
                                normalize=True,
                                lower=False,
                                clean_words=False,
                                load_kwargs={})

    def get_similarity(self, word1, word2):
        v1 = self._get_vector(word1)
        if v1 is None:
            return None
        self.w[word1] = v1

        v2 = self._get_vector(word2)
        if v2 is None:
            return None
        self.w[word2] = v2

        return v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _get_vector(self, word):
        vector = self.w.get(word)
        if vector is None:
            vector = self._get_vector_for_ovv(word)
        if vector is None:
            return None
        return vector

    def _get_vector_for_ovv(self, word):
        words = self.w.vocabulary.word_id
        cut_word = word

        words_with_same_prefix = set()
        while len(cut_word) and cut_word not in words:
            cut_word = cut_word[:-1]
            # collectings words with the same prefix
            for vocabulary_word in self.w:
                if vocabulary_word[0].startswith(cut_word):
                    words_with_same_prefix.add(vocabulary_word[0])
            if len(words_with_same_prefix):
                break
        if words_with_same_prefix:
            token_vecs = [self.w.get(t) for t in words_with_same_prefix]
            return np.mean(token_vecs, axis=0)
        return None
