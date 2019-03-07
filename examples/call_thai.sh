# argv: embeddingfile   -   emb_format  - TOKENIZE_OOV_WORDS_WITH_DEEPCUT (default:False)  -  FILTER_NOT_FOUND (default: False)

EMB=/home/wohlg/itmo/word-embeddings-benchmarks-orig/web/datasets/wiki.th.vec
NAME=ft_wiki
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 

EMB=/home/wohlg/itmo/kmitl2019/embeddings/thai2fit/v0.1/word2vec/thai2vec.vec
NAME=t2v
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 


EMB=/home/wohlg/itmo/kmitl2019/embeddings/cc.th.300.vec
NAME=big_cc
python evaluate_similarity_thai.py $EMB  word2vec False False >/tmp/${NAME}_ff
python evaluate_similarity_thai.py $EMB  word2vec True  False >/tmp/${NAME}_tf
python evaluate_similarity_thai.py $EMB  word2vec False True  >/tmp/${NAME}_ft
python evaluate_similarity_thai.py $EMB  word2vec True  True  >/tmp/${NAME}_tt 

