from elasticsearch import Elasticsearch


class ESC(object):
    """define constants for ElasticSearch modules"""

    def __init__(self, ix_name):
        """initialize constants"""
        self.ES_HOST = {'host': 'localhost', 'port': 9200}
        self.INDEX = ix_name.lower()
        self.ANALYZER = 'analyzer'
        self.DOC = 'abstract'
        self.DOC_FIELD = 'body'
        self.RANKER = 'BM25'
        self.STOPWORDS = 'indri_stopwords.txt'  # refers to stopwords path
        self.ES = Elasticsearch([self.ES_HOST])

