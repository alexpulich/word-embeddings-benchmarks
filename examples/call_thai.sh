# argv: embeddingfile   -   emb_format  - TOKENIZE_OOV_WORDS_WITH_DEEPCUT (default:False)  -  FILTER_NOT_FOUND (default: False)

# to evaluate your own model, fill in and uncomment:
# you only need to change this next line
EMB=path-to-the-embedding-model

################################################################################################################################

## basic evaluation, OOV words will be represented by average vector of all words in the model
python evaluate_similarity_thai.py $EMB  word2vec False False 

## OOV words will be tokenized with deepcut, and the word is represented by the average of in-vocabulary components
python evaluate_similarity_thai.py $EMB  word2vec True False 

# ## completely remove OOV words 
python evaluate_similarity_thai.py $EMB  word2vec False True 
# 
# ## first apply deepcut, for the rest of OOV words: remove them
python evaluate_similarity_thai.py $EMB  word2vec True True 

exit

