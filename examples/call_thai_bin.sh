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
python evaluate_similarity_thai.py $EMB  word2vec_bin False False 

## OOV words will be tokenized with deepcut, and the word is represented by the average of in-vocabulary components
python evaluate_similarity_thai.py $EMB  word2vec_bin True False 

# ## completely remove OOV words 
python evaluate_similarity_thai.py $EMB  word2vec_bin False True 
# 
# ## first apply deepcut, for the rest of OOV words: remove them
python evaluate_similarity_thai.py $EMB  word2vec_bin True True 

exit

