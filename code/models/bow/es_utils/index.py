import glob
import os
import math
import string
import json
import operator
import itertools
import numpy as np
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from collections import defaultdict, Counter
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

	### RM3 PSEUDO-RELEVANCE FEEDBACK RELATED FUNCTIONS ###

	def perform_rm3_prf(self, queries, qfield, rank_path, run_name, fb_docs=10, fb_terms=10, qweight=0.5):
		"""perform pseudo-relevance feedback w/ RM3 model"""
		
		out = open(rank_path + '/' + run_name + '.txt', 'w')
		# loop over queries and perform RM3 expansion
		for qid, qbody in tqdm(queries.items()):
			# issue first quey retrieving 'fb_docs' feedback docs
			first_qres = self.es.search(index=self.index, size=fb_docs, body={'query': {'match': {self.field: qbody[qfield]}}})
			
			# store ids and scores of the returned feedback docs
			ids_and_scores = dict()
			for res in first_qres['hits']['hits']:
				ids_and_scores[res['_id']] = res['_score']
			
			# get query feature vector and normalize to L1 norm
			qfv = self.scale_to_L1_norm(Counter(self.analyze_query(qbody[qfield])))
			# get relevance model feature vector (i.e., RM1)
			rm1 = self.estimate_relevance_model(ids_and_scores, fb_terms)
			# interpolate qfv and rm1 (i.e., RM3)
			rm3 = self.interpolate(qfv, rm1, qweight)
			
			# build boosted term queries
			term_queries = [{'term': {self.field: {'value': term, 'boost': score}}} for term, score in rm3.items()]
			# combine term queries w/ SHOULD operator
			expanded_query = {'query': { 'bool': {'should': term_queries}}}
			# issue expanded query and return 1000 docs constituting final ranking list
			prf_qres = self.es.search(index=self.index, size=1000, body=expanded_query)
			for idx, rank in enumerate(prf_qres['hits']['hits']):
				out.write('%s %s %s %d %f %s\n' % (qid, 'Q0', rank['_id'], idx, rank['_score'], run_name))
		out.close()
		return True

	def interpolate(self, qfv, rm1, qweight):
		"""interpolate two feature vectors w/ given weight"""
		
		# set variables
		rm3 = defaultdict(float)
		vocab = set()
		
		# update vocab w/ terms from both feature vectors
		vocab.update(qfv.keys())
		vocab.update(rm1.keys())
		
		# interpolate
		for term in vocab:
			weight = qweight * qfv[term] + (1 - qweight) * rm1[term]
			rm3[term] = weight
		return rm3

	def estimate_relevance_model(self, ids_and_scores, fb_terms):
		"""estimate RM1"""
		
		# set variables
		rm1_vec = list()
		vocab = set()
		doc_vecs = dict()
	   
		# create document feature vectors for each feedback doc
		for doc in self.es.mtermvectors(index=self.index, 
										doc_type=self.doc, 
										body=dict(ids=list(ids_and_scores.keys()), 
												  parameters=dict(term_statistics=True, 
																  field_statistics=False, 
																  positions=False, 
																  payloads=False, 
																  offsets=False, 
																  fields=[self.field])))['docs']:
			# extract term stats from current feedback doc
			fields = doc['term_vectors']
			term_stats = fields[self.field]['terms']
			
			# create document feature vector
			dfv = self.create_feature_vector(term_stats)
			# keep top 'fb_terms' from dfv
			dfv = defaultdict(int, sorted(dfv, key=lambda x: (-x[1], x[0]))[:fb_terms])  # -x[1] represents descending order

			# update vocab with top 'fb_terms' terms contained within feedback docs and store document feature vectors
			vocab.update(dfv.keys())
			doc_vecs[doc['_id']] = dfv

		# compute L1 norm for each document feature vector
		norms = {doc_id: sum(dfv.values()) for doc_id, dfv in doc_vecs.items()}

		# loop over terms in vocab and compute RM1
		for term in vocab:
			fb_weight = 0.0
			# loop over document feature vectors 
			for doc_id in doc_vecs.keys():
				if norms[doc_id] > 0.001:  # avoids zero-length feedback docs which cause division by zero when computing term weights
					# sum the score of current term across different docs to fb_weight
					fb_weight += (doc_vecs[doc_id][term] / norms[doc_id]) * ids_and_scores[doc_id]  # ids_and_scores[doc_id] is the score obtained for current doc w/ the original query
			# assign term w/ weight to RM1 feature vector
			rm1_vec.append((term, fb_weight))

		# keep top 'fb_terms' from rm1_vec
		rm1_vec = defaultdict(float, sorted(rm1_vec, key=lambda x: (-x[1], x[0]))[:fb_terms])  # -x[1] represents descending order
		# scale rm1_vec to L1 norm
		return self.scale_to_L1_norm(rm1_vec)

	def create_feature_vector(self, term_stats):
		"""create the feature vector out of doc terms"""
		
		tfv = list()
		# get corpus length (n. of docs)
		num_docs = self.es.count(index=self.index)['count']
		for term, stats in term_stats.items():
			# filter out terms w/ length lt 2 or length gt 20
			if len(term) < 2 or len(term) > 20:
				continue
			# filter out non-alphabetical terms
			if not term.isalpha():
				continue
			# get document frequency 
			df = stats['doc_freq']
			# compute ratio between df and num_docs
			ratio = df / num_docs
			if ratio > 0.1:  # skip term - requires tuning: check if it's okay to keep it as is
				continue
			# get term frequency within current doc
			freq = stats['term_freq']
			# append term w/ term_freq to tfv
			tfv.append((term, freq))
		return tfv

	def scale_to_L1_norm(self, vec): 
		"""scale input vector using L1 norm"""
		
		norm = sum(vec.values())
		for term, score in vec.items():
			vec[term] = score / norm
		return vec