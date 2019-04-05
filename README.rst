Word Embeddings Benchmarks - for Thai datasets
=====

This is a fork, with the goal to provide an easy way to evaluate Thai word embeddings with new word similarity datasets.
The accompanying publication which describes the new Thai datasets is currently under review.
The Thai datasets are translations of popular existing datasets: WordSim-353, SimLex-999 and the dataset from SemEval 2017 (task 2).
The task is **word similarity**, which is often used for intrinsic evaluation of word embedding models.
In the fork we added Spearman rho as additional evaluation measure, and added the option to tokenize 
out-of-vocabulary words with the `deepcut` library.

First, please follow the installation guide from the original repo which is duplicated below as well.
Then execute following commands to evaluate your own Thai word embedding file::
	cd examples
	chmod +x call_thai.sh
    bash examples/call_thai.sh <path_to_your_embedding_file>

The datasets were created by KMITL University, Ladkrabang, Thailand (Dr. Ponrudee Netisopakul) together with ITMO University, St. Petersburg, Russia (Dr. Gerhard Wohlgenannt,
Aleksei Pulich).
Please cite our work:

    P. Netisopakul, G. Wohlgenannt and A. Pulich, Word Similarity Datasets for Thai: Construction and Evaluation, IEEE Access, 2019, under review


Below please find the description of the original repository by **kudkudak**, which includes general info,
info on installation, etc.



Word Embeddings Benchmarks
=====

.. image:: https://travis-ci.org/kudkudak/word-embeddings-benchmarks.svg?branch=master

Word Embedding Benchmark (web) package is focused on providing methods for easy evaluating and reporting
results on common benchmarks (analogy, similarity and categorization).

Research goal of the package is to help drive research in word embeddings by easily accessible reproducible
results (as there is a lot of contradictory results in the literature right now).
This should also help to answer question if we should devise new methods for evaluating word embeddings.

To evaluate your embedding (converted to word2vec or python dict pickle)
on all fast-running benchmarks execute `./scripts/eval_on_all.py <path-to-file>`.
See `here <https://github.com/kudkudak/word-embeddings-benchmarks/wiki>`_ results for embeddings available in the package.

Warnings and Disclaimers:

* Analogy test does not normalize internally word embeddings.
* **Package is currently under development, and we expect within next few months an official release**. The main issue that might hit you at the moment is rather long embeddings loading times (especially if you use fetchers).

Please also refer to our recent publication on evaluation methods https://arxiv.org/abs/1702.02170.

Features:

* scikit-learn API and conventions
* 18 popular datasets
* 11 word embeddings (word2vec, HPCA, morphoRNNLM, GloVe, LexVec, ConceptNet, HDC/PDC and others)
* methods to solve analogy, similarity and categorization tasks

Included datasets:

* TR9856
* WordRep
* Google Analogy
* MSR Analogy
* SemEval2012
* AP 
* BLESS
* Battig
* ESSLI (2b, 2a, 1c)
* WS353
* MTurk
* RG65
* RW
* SimLex999
* MEN

Note: embeddings are not hosted currently on a proper server, if the download is too slow consider downloading embeddings manually from original sources referred in docstrings.

Dependencies
======

Please see the requirements.txt and pip_requirements.txt file.

Install
======

This package uses setuptools. You can install it running:

    python setup.py install

If you have problems during this installation. First you may need to install the dependencies:

    pip install -r requirements.txt

If you already have the dependencies listed in requirements.txt installed,
to install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

You can also install it in development mode with::

    python setup.py develop


Examples
========
See `examples` folder.

License
=======
Code is licensed under MIT, however available embeddings distributed within package might be under different license. If you are unsure please reach to authors (references are included in docstrings)

