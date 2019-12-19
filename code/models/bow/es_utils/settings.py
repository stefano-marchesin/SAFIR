from . import utils
from . import constants


class ESS(object):
    """define settings for ElasticSearch modules"""

    def __init__(self, ix_name):
        """initialize settings"""
        self.constants = constants.ESC(ix_name)
        # define settings and mapping used to index with es
        self.properties = {
            'settings': {
                'number_of_shards': '1',
                'number_of_replicas': '0',
                "index": {
                    "blocks": {
                        "read_only_allow_delete": "false"
                    }
                },
                'analysis': {
                    'filter': {
                        'custom_stopwords': {
                            'type': 'stop',
                            'stopwords': utils.load_stopwords('./' + self.constants.STOPWORDS)
                        },
                        'length_filter': {
                            'type': 'length',
                            'min': 3
                        },
                        'possessive_stemmer': {
                            'type': 'stemmer',
                            'language': 'possessive_english'
                        },
                        'porter_stemmer': {
                            'type': 'stemmer',
                            'language': 'english'
                        }
                    },
                    'analyzer': {
                        self.constants.ANALYZER: {
                            'tokenizer': 'classic',
                            'filter': [
                                    # 'possessive_stemmer',
                                'lowercase',
                                'length_filter',
                                'custom_stopwords',
                                    # 'porter_stemmer'
                            ]
                        }
                    }
                },
                'similarity': {
                    'custom_model': {
                        'type': self.constants.RANKER
                    }
                }
            },
            'mappings': {
                self.constants.DOC: {
                    'properties': {
                        self.constants.DOC_FIELD: {
                            'type': 'text',
                            'similarity': 'custom_model',
                            'analyzer': self.constants.ANALYZER
                        }
                    }
                }
            }
        }
