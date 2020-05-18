# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging, sys
import numpy as np
import scipy.stats
from six import iteritems
from pprint import pprint
from web.datasets.similarity import (
    fetch_TWS65,
    fetch_thai_wordsim353,
    fetch_thai_semeval2017_task2,
    fetch_thai_simlex999
)
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity
from sklearn.datasets.base import Bunch

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


class ThaiEvaluation:
    WORDNET = 'wn'
    CONCEPTNET = 'cn'

    def __init__(self,
                 fname: str,
                 format: str,
                 ):
        self.fname = fname
        self.format = format

        self.tasks = {}
        self.w = None

        self.latex1 = ''
        self.latex2 = ''

        self._fetch_tasks()
        self._load_embeddings()

    def _fetch_tasks(self):
        self.tasks = {
            "TH-WS353": fetch_thai_wordsim353(),
            "TH-SemEval2017T2": fetch_thai_semeval2017_task2(),
            "TH-SimLex999": fetch_thai_simlex999(),
            "TWS65": fetch_TWS65()
        }

    def _print_sample_data(self):
        for name, data in iteritems(self.tasks):
            print(
                "Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(
                    name,
                    data.X[0][0],
                    data.X[0][1],
                    data.y[0])
            )

    def _load_embeddings(self):
        self.w = load_embedding(self.fname,
                                format=self.format,
                                normalize=True,
                                lower=False,
                                clean_words=False,
                                load_kwargs={})

    def _evaluate_structed(self,
                           data: Bunch,
                           struct_info: str,
                           tokenize_oov_with_deepcut: bool,
                           filter_not_found: bool):
        results = []
        for coef in np.arange(0.00, 1.05, 0.05):
            result = evaluate_similarity(self.w, data.X, data.y,
                                         tokenize_oov_words_with_deepcut=tokenize_oov_with_deepcut,
                                         filter_not_found=filter_not_found,
                                         include_structured_sources=struct_info,
                                         structed_sources_coef=coef)
            result['coef'] = coef
            try:
                result['hm'] = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
            except:
                result['hm'] = -999  ## undefined
            results.append(result)

        pprint(results)
        result = max(results, key=lambda x: x['hm'])
        hm = result['hm']
        print('BEST COEF: {}'.format(result['coef']))
        print('STRUCTED OOV : {}'.format(result['structed_oov_pairs']))
        return result, hm

    def _evaluate_unstructed(self,
                             data: Bunch,
                             struct_info: str,
                             tokenize_oov_with_deepcut: bool,
                             filter_not_found: bool):
        result = evaluate_similarity(self.w, data.X, data.y,
                                     tokenize_oov_words_with_deepcut=tokenize_oov_with_deepcut,
                                     filter_not_found=filter_not_found,
                                     include_structured_sources=struct_info,
                                     structed_sources_coef=None)

        try:
            hm = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
        except:
            hm = -999  ## undefined

        return result, hm

    def _print_report(self, name, hm, result):
        perc_oov_words = 100 * (
                result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))
        print('num_found_words and num_missing_words are just the plain counts in the datasets')
        print(
            'num_oov_created is the set of words created/replace with a new vectors (based on tokenization with deepcut)')
        print(
            """Dataset {:17}: Spearman/Pearson/HarmMean: {:4.3f} {:4.3f} {:4.3f}, OOV-word-pairs: {:4d}, perc_oov_words: {:3.1f}, num_found_words: {:4d}, num_missing_words: {:4d}, num_oov_created: {:4d}, y.shape: {}""".format(
                name,
                round(result['spearmanr'], 3),
                round(result['pearsonr'], 3),
                hm,
                result['num_oov_word_pairs'],
                perc_oov_words,
                result['num_found_words'],
                result['num_missing_words'],
                result['num_oov_created'],
                str(result['y.shape'])
            ))

        ## for using in the paper -- LaTeX output
        self.latex1 += '{:4.3f}~~{:4.3f}~~{:4.3f}          & {:3.1f}~~{:3d}  & '.format(round(result['spearmanr'], 3),
                                                                                        round(result['pearsonr'], 3),
                                                                                        hm,
                                                                                        perc_oov_words,
                                                                                        result['num_oov_word_pairs'])

        ## pairs left
        self.latex2 += '{:4.3f}~~{:4.3f}~~{:4.3f}          & {:3.1f}~~{:3d}  & '.format(round(result['spearmanr'], 3),
                                                                                        round(result['pearsonr'], 3),
                                                                                        hm,
                                                                                        perc_oov_words,
                                                                                        result['y.shape'][0])

    def _print_latex(self):
        ## print final latex string
        print('{:17} & {} \\\\  % LaTeX 1 for Table 1 and 2'.format(self.fname, self.latex1[:-2]))
        ## pairs left
        print('{:17} & {} \\\\  % LaTeX 2 for Table 3 and 4'.format(self.fname, self.latex2[:-2]))

    def evaluate(self,
                 tokenize_oov_with_deepcut: bool,
                 filter_not_found: bool,
                 struct_info: str
                 ):
        # Print sample data
        self._print_sample_data()

        # Calculate results using helper function for the various word similarity datasets
        for name, data in iteritems(self.tasks):
            print("\n", "NEW TASK:", name)
            if struct_info:
                result, hm = self._evaluate_structed(data, struct_info, tokenize_oov_with_deepcut, filter_not_found)
            else:
                result, hm = self._evaluate_unstructed(data, struct_info, tokenize_oov_with_deepcut, filter_not_found)

            self._print_report(name, hm, result)

        self._print_latex()
        self.latex1, self.latex2 = '', ''
