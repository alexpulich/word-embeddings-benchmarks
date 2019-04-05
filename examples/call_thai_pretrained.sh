# argv: embeddingfile   -   emb_format  - TOKENIZE_OOV_WORDS_WITH_DEEPCUT (default:False)  -  FILTER_NOT_FOUND (default: False)

############### below here are the evaluation calls for the publication about the datasets, you can use them to reproduce the results ######

# download the model here: https://github.com/Kyubyong/wordvectors -- fasttext (f) from within .gz file
EMB=/home/wohlg/itmo/datasets/th.vec-f
NAME=Kyubyong_th_f
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 


# download the model here: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.th.vec 
EMB=/home/wohlg/itmo/datasets/wiki.th.vec
NAME=ft_wiki
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 

# download the model here: https://github.com/cstorm125/thai2fit/ 
EMB=/home/wohlg/itmo/kmitl2019/embeddings/thai2fit/v0.1/word2vec/thai2vec.vec
NAME=t2v
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 

# download the model here: https://fasttext.cc/docs/en/crawl-vectors.html
EMB=/home/wohlg/itmo/kmitl2019/embeddings/cc.th.300.vec
NAME=big_cc
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 

# download the model here: https://github.com/Kyubyong/wordvectors -- word2vec (w) converted from .tsv file 
EMB=/home/wohlg/itmo/datasets/th.vec-w
NAME=Kyubyong_th_w
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 

