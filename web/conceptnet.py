import requests
import time
import random
import pickle

BASE_URL = 'http://api.conceptnet.io/relatedness?node1=/c/%s/%s&node2=/c/%s/%s'


def get_similarity(word1, word2, lang='th'):
    score = requests.get(BASE_URL % (lang, word1, lang, word2)).json()['value']
    time.sleep(random.randint(0, 3))
    return score if score > 0 else None

def get_similarity_from_dict(scores, word1, word2):
    return scores.get(word1 + word2)