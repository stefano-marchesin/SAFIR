import os
import math
import itertools
import gensim
import glob
import json
import string
import numpy as np

from tqdm import tqdm
from textwrap import wrap
from collections import Counter
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import strip_tags
from nltk import sent_tokenize
from whoosh.analysis import StandardAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

import safir_utils
import umls

from quickumls_conn import QuickUMLS
from QuickUMLS.quickumls.client import get_quickumls_client


class Index(object):
	"""define an index instance along with its associated methods"""

	def __init__(self, stops, minsize=3):
		"""initialize index variables"""
		self.ix = None
		self.tokenizer = StandardAnalyzer(stoplist=stops, minsize=minsize)
		self.umls = umls.UMLSLookup()
		self.term_dict = {}
		self.token2cuis = {}
		self.concept_dict = {"__NULL__": 0}
		self.synsets = {}

	def get_doc_ids(self, corpus_path, corpus_name): 
		"""get doc ids from corpus"""
		if "OHSUMED" in corpus_name:
			docs = safir_utils.gen_trec_doc(corpus_path)
		elif "TREC_CDS" in corpus_name: 
			docs = safir_utils.gen_cds_doc(corpus_path)
		return [docno for docno, doc in docs]

	def only_digits(self, token):
		"""check whether input token contains only digits and/or punctuation"""
		return all(char.isdigit() or char in string.punctuation for char in token)
	
	def preprocess_text(self, text, tags=False, remove_digits=True): 
		"""preprocess text: tokenize docs, lowerize text, remove words with length < min_size, remove tags, remove only-digits tokens and remove stopwords"""
		if tags:  # remove tags 
			text = strip_tags(text)
		if remove_digits:  # tokenize and remove digits-only tokens
			text = [token.text for token in self.tokenizer(text) if not self.only_digits(token.text)]
		else:  # tokenize and keep digits-only tokens
			text = [token.text for token in self.tokenizer(text)]
		# return preprocessed doc
		return text

	def preprocess_corpus(self, corpus_path, corpus_name, out_corpus, out_ids): 
		"""preprocess corpus: apply preprocess_text to each doc within corpus"""
		if "OHSUMED" in corpus_name:
			docs = safir_utils.gen_trec_doc(corpus_path)
		elif "TREC_CDS" in corpus_name: 
			docs = safir_utils.gen_cds_doc(corpus_path)
		# tokenize docs
		print("pre processing docs...")
		#pproc_corpus = [self.preprocess_text(doc) for docno, doc in docs]
		pproc_corpus = []
		doc_ids = []
		# iterate over docs and store pre processed docs and docnos
		for docno, doc in docs:
			pproc_corpus.append(self.preprocess_text(doc))
			doc_ids.append(docno)
		print("pre processing finished!")
		# store pproc_corpus
		print("store pre processed corpus in {}".format(out_corpus))
		with open(out_corpus, 'w') as outf:
			json.dump(pproc_corpus, outf)
		# store docnos
		print("store doc_ids in {}".format(out_ids))
		with open(out_ids, 'w') as outf:
			json.dump(doc_ids, outf)
		# return pproc_corpus and doc_ids
		return pproc_corpus, doc_ids

	def load_pproc_corpus(self, fname):
		"""load stored pre processed corpus"""
		with open(fname, 'r') as inf:
			pproc_corpus = json.load(inf)
		return pproc_corpus

	def load_doc_ids(self, fname):
		"""load stored doc ids"""
		with open(fname, 'r') as inf:
			doc_ids = json.load(inf)
		return doc_ids

	def index_corpus(self, pproc_corpus, fname): 
		"""index pre processed corpus using gensim dictionary - fast doc2bow, doc2idx conversion"""
		self.ix = Dictionary(pproc_corpus)
		self.ix.save_as_text(fname)
		return True

	def load_index(self, fname):
		"""load stored index""" 
		self.ix = Dictionary.load_from_text(fname)
		return True

	def build_term_dict(self, pproc_corpus, fname, dict_size=131072, remove_digits=True, min_df=2, max_df=0.5):
		"""create term dictionary"""
		ttf = {}
		# filter terms with df lower than 2 and greater than 0.5 (in %) and store their ttf
		for doc in tqdm(pproc_corpus):
			# get doc in bow format
			bow = self.ix.doc2bow(doc)
			for idx, tf in bow:
				if self.ix.dfs[idx] >= 2 and self.ix.dfs[idx] / self.ix.num_docs <= 0.5:
					if idx in ttf:
						ttf[idx] += tf
					else:
						ttf[idx] = tf
		# convert ttf dict into counter and keep dict_size most frequent terms
		count = Counter(ttf).most_common(dict_size)
		# create term dict - two-levels encoding (i.e. self.term_dict[self.ix.token2id[token]])
		for idx, ttf in count:
			self.term_dict[idx] = len(self.term_dict) 
		# store term dictionary
		with open(fname, 'w') as outf:
			json.dump(self.term_dict, outf)
		return True

	def load_term_dict(self, fname):
		"""load term dictionary"""
		with open(fname, 'r') as inf:
			self.term_dict = json.load(inf)
		# convert keys from str back to int - json stores dict keys as str
		self.term_dict = {int(ix_term): dict_term for ix_term, dict_term in self.term_dict.items()}
		return True

	def get_pos2token(self, text):
		"""split text into tokens and return {pos: [token, ["__NULL__"]]}"""
		pos2token = {}
		tokens = text.split()  # split on whitespaces as text has been already pre processed
		# set text index
		index = text.index
		running_offset = 0
		# loop over tokens
		for token in tokens:
			token_offset = index(token, running_offset)
			token_len = len(token)
			# update running offset
			running_offset = token_offset + token_len
			pos2token[token_offset] = [self.ix.token2id[token], ["__NULL__"]]  # note: ["__NULL__"] is for later use
		return pos2token

	def associate_token2cuis(self, pos2token, terms_candidate_cuis):
		"""return list of (token, [cui1, cui2, ...]) pairs given token position and candidate concepts""" 
		for term_cuis in terms_candidate_cuis:
			# get positional information
			start = term_cuis[0]['start']
			# check whether 'start' matches with any pos2token key
			if start in pos2token:
				# update ["__NULL__"] with candidate cuis
				pos2token[start][1] = [concept['cui'] for concept in term_cuis]
		# return pos2token values only - i.e. (term, [cui1, cui2, ...]) pairs
		return list(pos2token.values())

	def map_token2cuis(self, fname, threshold=1.0, stypes_fname=None): 
		"""map candidate cuis to each token in the index"""
		terms_str = ' '.join(list(self.ix.token2id.keys()))
		# split term_str into substrings of length <= 999999 - max length allowed by scipy parser
		substrs = wrap(terms_str, width=999999, break_long_words=False, break_on_hyphens=False)
		if stypes_fname is not None:  # load user-specified UMLS semantic types
			print("user-specified UMLS semantic types for QuickUMLS enabled")
			semtypes = ','.join(safir_utils.load_semtypes(stypes_fname))
		else:  # keep default QuickUMLS semantic types
			semtypes = None
		# initialize QuickUMLS server
		server = QuickUMLS(window=1, threshold=threshold, semtypes=semtypes)
		server.launch_quickumls()
		# initialize concept matcher
		matcher = get_quickumls_client()
		token2cuis = []
		# extract concepts from substrs
		for substr in substrs:
			terms_candidate_cuis = matcher.match(substr)
			# get position dict: {pos: [token, ["__NULL__"]]} given substr
			pos2token = self.get_pos2token(substr)
			# associate each token with its candidate concepts
			token2cuis += self.associate_token2cuis(pos2token, terms_candidate_cuis)
		# close connection with QuickUMLS server
		server.close_quickumls()
		# store token2cuis as dict
		self.token2cuis = dict(token2cuis)
		# store token2cuis 
		with open(fname, 'w') as outf:
			json.dump(self.token2cuis, outf)
		return True

	def load_token2cuis(self, fname):
		"""load token2cuis"""
		with open(fname, 'r') as inf:
			self.token2cuis = json.load(inf)
		# convert keys from str back to int - json stores dict keys as str
		self.token2cuis = {int(token): cuis for token, cuis in self.token2cuis.items()}
		return True

	def update_concept_dict(self, cui):
		"""update concept dictionary"""
		if cui in self.concept_dict:
				return True
		else:
				self.concept_dict[cui] = len(self.concept_dict)
				return True

	def load_concept_dict(self, fname):
		"""load concept dictionary"""
		with open(fname, 'r') as inf:
			self.concept_dict = json.load(inf)
		return True

	def update_synsets(self, cui, idx):
		"""update synonyms set"""
		if self.concept_dict[cui] in self.synsets:  # add term to set of synonyms for the given cui
			self.synsets[self.concept_dict[cui]].add(self.term_dict[idx])
			return True
		elif self.concept_dict[cui] != self.concept_dict["__NULL__"]:  # initialize set of synsets for given cui
			self.synsets[self.concept_dict[cui]] = {self.term_dict[idx]}
			return True
		else:  # do not update synsets
			return False

	def load_synsets(self, fname):
		"""load synsets"""
		with open(fname, 'r') as inf:
			self.synsets = json.load(inf) 
		# convert keys from str back to int - json stores dict keys as str
		self.synsets = {int(cui): syns for cui, syns in self.synsets.items()}
		return True

	def get_sense_pairs(self):
		"""return senses as (term, cui) 2-dim np array"""
		syns = [list(itertools.product(self.synsets[cui], [cui])) for cui in self.synsets]
		synp = [list(itertools.combinations(syn, 2)) for syn in syns]
		return np.array(list(itertools.chain.from_iterable(synp)))

	def s_wsd(self, doc, table_name, query=False):
		"""shallow word-sense disambiguation: disambiguate polysemous terms based on shallow word-concept connectivity within UMLS"""
		doc_cuis = {}
		# convert doc into doc2idx format
		doc2idx = self.ix.doc2idx(doc)
		# get cuis from doc tokens
		for idx in doc2idx:
			if idx in self.token2cuis and self.token2cuis[idx] != ["__NULL__"]: 
				for cui in self.token2cuis[idx]:
					if cui in doc_cuis:  # increase cui count
						doc_cuis[cui] += 1
					else:  # initialize cui count
						doc_cuis[cui] = 1
		# perform shallow word-sense disambiguation
		enc_doc = []
		for idx in doc2idx:
			if idx in self.term_dict:  # disambiguate only for terms contained within self.term_dict
				max_edges = 0  # relative maximum connections (edges)
				if len(self.token2cuis[idx]) == 1:  # monosemous term
					ref_cui = self.token2cuis[idx][0]
					if not query:  # update concept dict and synsets
						self.update_concept_dict(ref_cui)
						self.update_synsets(ref_cui, idx)
					# encode (term, cui) pair
					enc_doc.append([self.term_dict[idx], self.concept_dict[ref_cui]])
				else:  # polysemous term
					candidates = []
					# loop over cadidate concepts
					for subj_cui in self.token2cuis[idx]:
						num_edges = 0  # number of edges
						if doc_cuis[subj_cui] == 1:  # subj_cui is only associated with current term (idx)
							obj_cuis = list(set(doc_cuis.keys()).difference({subj_cui}))
						else:  # subj_cui is associated with other terms in the doc too
							obj_cuis = list(doc_cuis.keys())
						num_edges += self.umls.compute_num_edges(obj_cuis, subj_cui, table_name)  # remember that subj and obj are inverted within UMLS <s, p, o> triples
						# verify connectivity
						if num_edges > max_edges:
							# set candidates to subj_cui
							candidates = [subj_cui]
							# update max_edges
							max_edges = num_edges
						else:
							# append subj_cui to candidates
							candidates.append(subj_cui)
					# keep head candidate - when disambiguation is not complete, it allows to get the most likely concept based on QuickUMLS ordering
					ref_cui = candidates[0]
					if not query:  # update concept dict and synsets
						self.update_concept_dict(ref_cui)
						self.update_synsets(ref_cui, idx)
					# encode (term, cui) pair
					enc_doc.append([self.term_dict[idx], self.concept_dict[ref_cui]])
			else:  # term oov
				continue
		return enc_doc

	def encode_corpus(self, pproc_corpus, corpus_name, ecorpus_fname, t2c_fname, cdict_fname, syn_fname, threshold=0.7, stypes_fname=None): 
		"""perform semantic indexing and encode corpus"""
		print("map UMLS concepts to (indexed) tokens")
		self.map_token2cuis(t2c_fname, threshold=threshold, stypes_fname=stypes_fname)
		# get UMLS concepts mapped to (indexed) tokens
		ix_concepts = {cui for cuis in self.token2cuis.values() for cui in cuis if cui != "__NULL__"}
		# create sql table to store relations between concepts associated to indexed tokens - allows for fast accessing compared to MRREL table
		print("create table to store UMLS relations between concepts associated to (indexed) tokens - fast access is enabled by indexes")
		self.umls.restrict_to_ix_concepts(ix_concepts, corpus_name)
		# create indexes to speed up requests
		self.umls.create_index("CUI2_" + corpus_name, ["CUI2"], corpus_name)  # create index for subject column
		self.umls.create_index("CUI1_" + corpus_name, ["CUI1"], corpus_name)  # create index for object column
		self.umls.create_index("CUI2_CUI1_" + corpus_name, ["CUI2", "CUI1"], corpus_name)  # create multicolumn index (subj, obj)
		# encode corpus
		print("disambiguate polysemous tokens and encode corpus")
		enc_corpus = [self.s_wsd(doc, corpus_name, query=False) for doc in tqdm(pproc_corpus)]
		# store synsets as dict of lists - enables json encoding
		self.synsets = {cui: list(syns) for cui, syns in self.synsets.items()}
		# store semantic data and encoded corpus
		with open(ecorpus_fname, 'w') as outf:
			json.dump(enc_corpus, outf)
		with open(cdict_fname, 'w') as outf:
			json.dump(self.concept_dict, outf)
		with open(syn_fname, 'w') as outf:
			json.dump(self.synsets, outf)
		# return encoded corpus
		return enc_corpus

	def load_enc_corpus(self, fname):
		"""load encoded corpus"""
		with open(fname, 'r') as inf:
			enc_corpus = json.load(inf)
		return enc_corpus

	def preprocess_query(self, query):
		"""pre process query"""
		pproc_query = self.preprocess_text(query)
		return pproc_query

	def encode_query(self, pproc_query, corpus_name):
		"""disambiguate polysemous terms and encode query"""
		enc_query = self.s_wsd(pproc_query, corpus_name, query=True)
		if not enc_query:
			print("query does not contain known terms")
			return None
		else:
			return np.array(enc_query)

	def project_query(self, query, corpus_name, word_embs, proj_weights, concept_embs=None):
		"""project encoded query into dense vector of size [1, doc_embs]"""
		enc_query = self.encode_query(self.preprocess_query(query), corpus_name)
		if enc_query is None:
			return None
		else:
			if concept_embs is None:  # only terms are considered
				return np.matmul(proj_weights, np.mean(word_embs[enc_query[:, 0]], axis=0))
			else:  # terms + concepts are considered (i.e. senses)
				return np.matmul(proj_weights, np.mean(np.add(word_embs[enc_query[:, 0]], concept_embs[enc_query[:, 1]]), axis=0))

	def semantic_search(self, doc_ids, docs, query_ids, queries, ranking_folder, ranking_name):
		"""perform search over queries using neural semantic models and return ranking"""
		doc_ids = np.array(doc_ids)
		print("compute similarities between docs and queries")
		similarities = cosine_similarity(docs, queries)
		out = open(ranking_folder + '/' + ranking_name + '.txt', 'w')
		for i in tqdm(range(similarities.shape[1])):
			rank = np.argsort(-similarities[:, i])[:1000]
			docs_rank = doc_ids[rank]
			qid = query_ids[i]
			if qid.isdigit():  # cast to integer - this operation avoids storing topic ids as '0##' instead of '##'
					qid = str(int(qid))  # convert to int and then back to str
			for j in range(len(docs_rank)):
					out.write('%s %s %s %d %f %s\n' % (qid, 'Q0', docs_rank[j], j, similarities[rank[j]][i], ranking_name))
		out.close()
		return True
