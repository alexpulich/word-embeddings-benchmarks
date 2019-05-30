# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging, sys
import scipy.stats
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.datasets.similarity import fetch_TWS65, fetch_thai_wordsim353, fetch_thai_semeval2017_task2, fetch_thai_simlex999
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
    "TH-WS353": fetch_thai_wordsim353(),
    "TH-SemEval2017T2": fetch_thai_semeval2017_task2(),
    "TH-SimLex999": fetch_thai_simlex999()
}

# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))

# Calculate results using helper function for the various word similarity datasets
for name, data in iteritems(tasks):
    print("\n", "NEW TASK:", name)
    result = evaluate_similarity(w, data.X, data.y, tokenize_oov_words_with_deepcut=TOKENIZE_OOV_WORDS_WITH_DEEPCUT, 
                                                           filter_not_found=FILTER_NOT_FOUND)

    try:
        hm = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
    except:
        hm = -999 ## undefined
    perc_oov_words = 100 * (result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))
 
    print('num_found_words and num_missing_words are just the plain counts in the datasets')
    print('num_oov_created is the set of words created/replace with a new vectors (based on tokenization with deepcut)')
    print("""Dataset {:17}: Spearman/Pearson/HarmMean: {:4.3f} {:4.3f} {:4.3f}, OOV-word-pairs: {:4d}, perc_oov_words: {:3.1f}, num_found_words: {:4d}, num_missing_words: {:4d}, num_oov_created: {:4d}, y.shape: {}""".format(name, 
          round(result['spearmanr'],3), 
          round(result['pearsonr'],3),
          hm,
          result['num_oov_word_pairs'],
          perc_oov_words,
          result['num_found_words'],
          result['num_missing_words'],
          result['num_oov_created'],
          str(result['y.shape'])
    ))

    ## for using in the paper -- LaTeX output
    print('LaTeX 1 out: DS: {:17}: {:4.3f}~~{:4.3f}~~{:4.3f} & {:3.1f}~~{:4d} &'.format(name, round(result['spearmanr'],3), round(result['pearsonr'],3), hm, perc_oov_words, result['num_oov_word_pairs']))

    ## pairs left
    print('LaTeX 2 out: DS: {:17}: {:4.3f}~~{:4.3f}~~{:4.3f} & {:3.1f}~~{:4d} &'.format(name, round(result['spearmanr'],3), round(result['pearsonr'],3), hm, perc_oov_words, result['y.shape'][0]))

