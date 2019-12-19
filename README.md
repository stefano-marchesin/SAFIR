# Learning Methods for Knowledge-Enhanced Word Embeddings

This repository contains code, plots, and runs for the paper: 
<p align="center">
<b><i>Learning Unsupervised Knowledge-Enhanced Representations to Reduce the Semantic Gap in Information Retrieval</i></b>
 </p>
Submitted to ACM Transactions on Information Systems by M. Agosti, S. Marchesin, and G. Silvello 

### Requirements

- ElasticSearch 6.6
- Python 3
  - Numpy
  - Gensim
  - TensorFlow >= 1.13
  - Whoosh
  - SQLite3
  - Cvangysel
  - Pytrec_Eval
  - Scikit-Learn
  - Tqdm
  - QuickUMLS
  - Elasticsearch
  - Elasticsearch_dsl
  - Pubmed_parser
- UMLS 2018AA

### Notes
The folder structure required to run experiments can be seen in folder ``example``. Python files need to be put in root.  
Qrels file needs to be in ``.txt`` format.  
To train/evaluate SAFIR run ``safir_train.py``.  
To train/evaluate word2vec run ``gensim_word2vec.py``, to train/evaluate doc2vec/cdoc2vec run ``gensim_doc2vec.py``, to train/evaluate rword2vec and rdoc2vec run ``retrofit_word_vecs.py`` and ``retrofit_doc_vecs.py``, respectively.  
To run BM25 or QLM, run ``lexical_search.py``.  
The code for the model combination task is within ``re_ranking`` and ``rank_fusion`` directories.  

### Additional Notes
``server.py`` needs to be substitued within QuickUMLS folder as it contains a modified version required to run knowledge-enhanced models.  
