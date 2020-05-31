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
    evaluator.evaluate(False, False, False, ThaiEvaluation.CONCEPTNET)

    # OOV words will be tokenized with deepcut, and the word is represented by the average of in-vocabulary components
    evaluator.evaluate(True, False, False, ThaiEvaluation.CONCEPTNET)

    # completely remove OOV words
    evaluator.evaluate(False, False, True, ThaiEvaluation.CONCEPTNET)

    # first apply deepcut, for the rest of OOV words: remove them
    evaluator.evaluate(True, False, True, ThaiEvaluation.CONCEPTNET)
