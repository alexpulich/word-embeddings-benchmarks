# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging, sys
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.datasets.similarity import fetch_TWS65, fetch_thai_wordsim353 
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity

fname = sys.argv[1] # path to the embedding file
format = sys.argv[2] # see ../scripts/evaluate_on_all.py for formats

def toBool(para):
    if para == 'True': return True
    elif para == 'False': return False
    else: raise Exception()

# if a thai word is not found in the embedding (OOV), tokenize it with deepcut, and try to use the sum vector of its parts?
TOKENIZE_OOV_WORDS_WITH_DEEPCUT = toBool(sys.argv[3])

# remove a word pair if one of the words was not found in the embedding vocabulary
FILTER_NOT_FOUND = toBool(sys.argv[4])


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

#w = load_embedding(fname, format=format, normalize=True, lower=True, clean_words=False, load_kwargs={})
w = load_embedding(fname, format=format, normalize=True, lower=False, clean_words=False, load_kwargs={})

# Define tasks
tasks = {
    #"MEN": fetch_MEN(),
        #"WS353": fetch_WS353(),
        #"SIMLEX999": fetch_SimLex999(),
    "TWS65": fetch_TWS65(),
    "T-WS353": fetch_thai_wordsim353()
}

# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))

# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(w, data.X, data.y, 
                                                           tokenize_oov_words_with_deepcut=TOKENIZE_OOV_WORDS_WITH_DEEPCUT, 
                                                           filter_not_found=FILTER_NOT_FOUND)))

