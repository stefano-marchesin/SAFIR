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
- UMLS 2018AA

### Additional Notes
<!-- ``server.py`` needs to be substitued within QuickUMLS folder as it contains a modified version required to run knowledge-enhanced models.  
The folder structure required to run experiments can be seen in folder ``example``. Python files need to be put in root.  
Qrels file needs to be in ``.txt`` format.  
To train SAFIR run ``safir_train.py`` and to test it run ``safir_test.py``.  
To run BM25 or QLM, run ``lexical_search.py``. -->
Code and data TBA.
