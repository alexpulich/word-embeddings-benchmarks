# argv: embeddingfile   -   emb_format  - TOKENIZE_OOV_WORDS_WITH_DEEPCUT (default:False)  -  FILTER_NOT_FOUND (default: False)

if [ $# -eq 0 ]
  then
    echo "Wrong usage. Please, pass the path to the embeddings."
    echo "./call_thai.sh <some_path>"
    exit 1
fi
EMB=$1

################################################################################################################################

## basic evaluation, OOV words will be represented by average vector of all words in the model
python3 evaluate_similarity_thai.py $EMB  word2vec False False False

## OOV words will be tokenized with deepcut, and the word is represented by the average of in-vocabulary components
python3 evaluate_similarity_thai.py $EMB  word2vec True False False

# ## completely remove OOV words 
python3 evaluate_similarity_thai.py $EMB  word2vec False True False
# 
# ## first apply deepcut, for the rest of OOV words: remove them
python3 evaluate_similarity_thai.py $EMB  word2vec True True False

exit

