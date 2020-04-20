# Learning Unsupervised Knowledge-Enhanced Representations to Reduce the Semantic Gap in Information Retrieval

<b>NOTE: New data and fixes will be available asap</b>

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
- Trec_eval
- Sample_eval

### Notes 
To train/evaluate SAFIR run ``safir_train.py``.  
To train/evaluate word2vec run ``gensim_word2vec.py``.  
To train/evaluate doc2vec and cdoc2vec run ``gensim_doc2vec.py``.  
To train/evaluate rword2vec and rdoc2vec run ``retrofit_word_vecs.py`` and ``retrofit_doc_vecs.py``, respectively.  
To run BM25 or QLM, run ``lexical_search.py``.  
The code for the model combination task is within ``re_ranking`` and ``rank_fusion`` directories.  
The folder structure required to run experiments can be seen in folder ``example``. Python files need to be put in root.  
Qrels file needs to be in ``.txt`` format.  
Collections need to be named as: ``OHSUMED``, ``TREC_CDS14_15``, and ``TREC_CDS16``.  
To evaluate models ``trec_eval`` and ``sample_eval`` from NIST are required.  


### Additional Notes
``server.py`` needs to be substituted within QuickUMLS folder as it contains a modified version required to run knowledge-enhanced models.  
