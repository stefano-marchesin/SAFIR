import glob
import os
import math
import string
import json
import itertools
import numpy as np
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from elasticsearch import helpers
from elasticsearch_dsl import Search, Q
from pubmed_parser.pubmed_oa_parser import list_xml_path, parse_pubmed_xml


class Index(object):
	"""define an index instance and its associated methods"""

	def __init__(self, constants, properties):
		"""initialize index variables"""
		self.es = constants.ES
		self.index = constants.INDEX
		self.analyzer = constants.ANALYZER
		self.doc = constants.DOC
		self.field = constants.DOC_FIELD
		self.settings = properties

	def get_trec_corpus(self, corpus_path):
		"""convert trec style corpus into (valid) xml"""
		docs = glob.glob(corpus_path + '/**/*.txt', recursive=True)
		for doc in docs:
			with open(doc, 'r') as f:  # read doc
				xml = f.read()
			# convert into true xml
			xml = '<ROOT>' + xml + '</ROOT>'
			# fix bad-formed tokens
			xml = xml.replace('&', '&amp;')
			yield xml

	def gen_trec_doc(self, corpus_path):
		"""generate doc from batch of docs"""
		ohsu = self.get_trec_corpus(corpus_path)
		# loop over batches
		for batch in ohsu:
			# parse xml
			root = ETree.fromstring(batch)
			# loop through each doc in the batch
			for doc in tqdm(root):
				docno = ''
				body = ''
				# loop through each element (tag, value)
				for elem in doc:
					if elem.tag == 'DOCNO':
						docno = elem.text.strip()
					else:
						body = elem.text.strip()
				# return doc to index
				yield docno, body

	def gen_cds_doc(self, corpus_path):
		"""generate doc from batch of NXML-style docs"""
		cds = list_xml_path(corpus_path)
		# loop over doc paths
		for doc_path in tqdm(cds):
			# parse doc and get required data
			doc_data = parse_pubmed_xml(doc_path)
			docno = doc_data['pmc'].strip()
			title = doc_data['full_title'].strip()
			abstract = doc_data['abstract'].strip()
			body = ' '.join([title, abstract])
			# return doc to index
			yield docno, body

	def index_corpus(self, corpus_path):
		"""index given corpus (word-based indexing)"""
		if self.es.indices.exists(index=self.index):
			print('index already exists')
		else:
			print('create index')
			self.es.indices.create(index=self.index, body=self.settings)
			# index corpus docs
			print('indexing {} ... '.format(self.index))
			# create generator to iterate over docs
			if 'OHSUMED' in self.index.upper():
				i = ({'_index': self.index, '_type': self.doc, '_id': docno, '_source': {self.field: body}} for docno, body in self.gen_trec_doc(corpus_path))
			elif 'CDS' in self.index.upper():
				i = ({'_index': self.index, '_type': self.doc, '_id': docno, '_source': {self.field: body}} for docno, body in self.gen_cds_doc(corpus_path))
			# index bulks of docs
			helpers.bulk(self.es, i)
			print('indexing {} finished!'.format(self.index))
		return True

	def get_ix_terms(self):
		"""return list of indexed terms"""
		terms = self.es.search(index=self.index, body={'aggs': {self.doc: {'terms': {'field': self.field, 'size': 999999}}}})
		return [term['key'] for term in terms['aggregations'][self.doc]['buckets']]

	def get_terms_stats(self):
		"""get stats for indexed terms"""
		terms = self.get_ix_terms()
		synt_doc = ' '.join(terms)
		# get terms stats from es index
		terms_stats = self.es.termvectors(index=self.index, doc_type=self.doc, 
			term_statistics=True, field_statistics=False, positions=False, offsets=False, 
			body={'doc': {self.field: synt_doc}})
		# return terms stats
		return [(term, stats['doc_freq'], stats['ttf']) for term, stats in terms_stats['term_vectors'][self.field]['terms'].items()] 

	def get_doc_ids(self):
		"""return list of doc ids"""
		s = Search(using=self.es, index=self.index, doc_type=self.doc)
		src = s.source([])
		return [h.meta.id for h in src.scan()]

	def get_doc_terms(self, doc_id):
		"""return list of (positionally-ordered) doc terms given a doc id"""
		doc_terms = self.es.termvectors(index=self.index, doc_type=self.doc, fields=[self.field], id=doc_id, 
			positions=True, term_statistics=False, field_statistics=False, offsets=False)['term_vectors'][self.field]['terms']
		# get term positions within doc: {term: [pos1, pos2, ...]}
		doc_pos = {term: stats['tokens'] for term, stats in doc_terms.items()}
		# reverse doc_pos associating each position with the corresponding term
		terms_pos = {}
		for term, positions in doc_pos.items():
			for pos in positions:
				terms_pos[pos['position']] = term
		# return positionally-ordered doc terms
		return [terms_pos.get(i) for i in range(min(terms_pos), max(terms_pos) + 1) if terms_pos.get(i) != None]

	def analyze_query(self, query):
		"""analyze query using index analyzer"""
		res = self.es.indices.analyze(index=self.index, body={'analyzer': self.analyzer, 'text': query})
		return [term['token'] for term in res['tokens']]

	def change_model(self, model, **kwargs):
		"""change similarity model for current index"""
		model_settings = {'type': model}
		if kwargs is not None:
			for key, value in kwargs.items():
				model_settings[key] = value
		# close index before updating
		self.es.indices.close(index=self.index)
		# update settings
		similarity_settings = {'similarity': {'custom_model': model_settings}}
		self.es.indices.put_settings(index=self.index, body=similarity_settings)
		# re-open index after updating
		self.es.indices.open(index=self.index)
		return True

	def lexical_ranking(self, queries, qfield):
		"""perform search over queries using lexical models and return ranking as a dict {qid: {docno: doc_score, ...}, ...}"""
		print('searching over batch of {} queries'.format(len(queries)))
		ranking = {}
		# search over queries
		for qid, qbody in tqdm(queries.items()):
			qres = self.es.search(index=self.index, size=1000, body={'query': {'match': {self.field: qbody[qfield]}}})
			ranking[qid] = {rank['_id']: rank['_score'] for rank in qres['hits']['hits']}
		# return ranking
		return ranking

	def lexical_search(self, queries, qfield, rank_path, run_name):
		"""perform search over queries using lexical models and store ranking"""
		out = open(rank_path + '/' + run_name + '.txt', 'w')
		print('searching over batch of {} queries'.format(len(queries)))
		# search over queries
		for qid, qbody in tqdm(queries.items()):
			qres = self.es.search(index=self.index, size=1000, body={'query': {'match': {self.field: qbody[qfield]}}})
			for idx, rank in enumerate(qres['hits']['hits']):
				out.write('%s %s %s %d %f %s\n' % (qid, 'Q0', rank['_id'], idx, rank['_score'], run_name))
		out.close()
		return True
