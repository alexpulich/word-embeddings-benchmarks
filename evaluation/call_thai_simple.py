import sys
from run.evaluate_similarity_thai import ThaiEvaluation

if __name__ == '__main__':
    fname = sys.argv[1]
    try:
        format = sys.argv[2]
    except IndexError:
        format = 'word2vec'

    evaluator = ThaiEvaluation(fname, format)

    # basic evaluation, OOV words will be represented by average vector of all words in the model
    evaluator.evaluate(tokenize_oov_with_deepcut=False,
                       cut_letters_for_oov=False,
                       filter_not_found=False,
                       struct_info=None)

    # OOV words will be tokenized with deepcut, and the word is represented by the average of in-vocabulary components
    # evaluator.evaluate(tokenize_oov_with_deepcut=True,
    #                    cut_letters_for_oov=False,
    #                    filter_not_found=False,
    #                    struct_info=None)
    evaluator.evaluate(tokenize_oov_with_deepcut=False,
                       cut_letters_for_oov=True,
                       filter_not_found=False,
                       struct_info=None)

    # completely remove OOV words
    evaluator.evaluate(tokenize_oov_with_deepcut=False,
                       cut_letters_for_oov=False,
                       filter_not_found=True,
                       struct_info=None)

    # first apply deepcut, for the rest of OOV words: remove them
    # evaluator.evaluate(tokenize_oov_with_deepcut=True,
    #                    cut_letters_for_oov=False,
    #                    filter_not_found=True,
    #                    struct_info=None)

    evaluator.evaluate(tokenize_oov_with_deepcut=False,
                       cut_letters_for_oov=True,
                       filter_not_found=True,
                       struct_info=None)
