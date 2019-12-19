import string
import subprocess
import glob 
import collections
import operator
import json
import itertools
import pytrec_eval
import numpy as np
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from textwrap import wrap
from whoosh.analysis import SimpleAnalyzer, StandardAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models.callbacks import CallbackAny2Vec

import umls

from QuickUMLS.quickumls.client import get_quickumls_client
from quickumls_conn import QuickUMLS


class Utils(object):
	"""utils functions for Gensim semantic models"""

	def __init__(self, seed):
		"""set random seed - to enable repro"""
		np.random.seed(seed)

	def load_stopwords(self, stopwords_path):
		"""read stopwords file into list"""
		with open(stopwords_path, 'r') as sl:
			stop_words = {stop.strip() for stop in sl}
		return stop_words

	def load_stypes(self, stypes_path):
		"""read semantic types as list"""
		with open(stypes_path, 'r') as stf:
			stypes = [stype.split('|')[1] for stype in stf]
		return stypes

	def read_corpus(self, corpus_folder):  # @smarchesin TODO: extend to CDS collections too
		"""read corpus and yield (docno, doc) pairs"""
		docs = glob.glob(corpus_folder + '/**/*.txt', recursive=True)
		for doc in docs:
			with open(doc, 'r') as f:  # read doc
				xml = f.read()
			# convert into true xml and fix malformed tokens
			xml = '<ROOT>' + xml.replace('&', '&amp;') + '</ROOT>'
			# parse xml 
			root = ETree.fromstring(xml)
			# loop through each oc within root
			for doc in tqdm(root):
				docno = ''
				body = ''
				# loop through each element (tag, value)
				for elem in doc:
					if elem.tag == 'DOCNO':
						docno = elem.text.strip()
					else:
						body = elem.text.strip()
				# yield docno and body 
				yield docno, body

	def tokenize_corpus(self, corpus_folder, stopwords, out_path, remove_digits=False, minsize=3):
		"""tokenize corpus into list of lists and return list of doc ids"""
		corpus = self.read_corpus(corpus_folder)
		# set tokenizer
		tokenizer = StandardAnalyzer(stoplist=stopwords, minsize=minsize)
		# tokenize corpus as list of lists
		print('tokenizing corpus ...')
		docs = {}
		for docno, doc in corpus:
			# tokenize doc
			doc_tokens = self.tokenize_doc(doc, tokenizer, remove_digits)
			docs[docno] = doc_tokens
		print('corpus tokenized!')
		print("store processed data")
		with open(out_path + '/docs.json', 'w') as file_docs:
			json.dump(docs, file_docs)
		return docs	

	def tokenize_doc(self, doc, tokenizer, remove_digits=False):
		"""tokenize doc given tokenizer"""
		if remove_digits:
			return [token.text for token in tokenizer(doc) if not self.is_only_digits(token.text)]
		else:
			return [token.text for token in tokenizer(doc)]

	def is_only_digits(self, token):
		"""check whether token contains only digits and/or punctuation"""
		return all(char.isdigit() or char in string.punctuation for char in token)

	def build_vocab(self, docs, min_cut_freq, out_path):
		"""build vocabulary required to train any2vec models"""
		words = [word for doc in docs for word in doc]
		count = []
		# keep all words within corpus
		count.extend(collections.Counter(words).most_common())
		word_dict = {}
		for word, freq in count:
			if freq >= min_cut_freq:  # cut words with freq < min_cut_freq
				word_dict[word] = len(word_dict)
		# dump word dictionary 
		with open(out_path + '/word_dict.json', 'w') as file_dict:
			json.dump(word_dict, file_dict)
		return word_dict

	def build_cvocab(self, cdocs, min_cut_freq, out_path):
		"""build concept vocabulary required to train (concept-based) any2vec models"""
		concepts = [concept for cdoc in cdocs for concept in cdoc]
		count = []
		# keep all concepts within corpus
		count.extend(collections.Counter(concepts).most_common())
		concept_dict = {}
		for concept, freq in count:
			if freq >= min_cut_freq:  # cut concepts with freq < min_cut_freq
				concept_dict[concept] = len(concept_dict)
		# dump concept dictionary
		with open(out_path + '/concept_dict.json', 'w') as file_dict:
			json.dump(concept_dict, file_dict)
		return concept_dict

	def get_pos2term(self, text):
		"""split text into terms and return {pos: [term, ["__NULL__"]]}"""
		pos2term = {}
		terms = text.split()  # split on whitespaces as text has been already pre processed
		# set text index
		index = text.index
		running_offset = 0
		# loop over terms
		for term in terms:
			term_offset = index(term, running_offset)
			term_len = len(term)
			# update running offset
			running_offset = term_offset + term_len
			pos2term[term_offset] = [term, "__NULL__"]  # note: "__NULL__" is for later use
		return pos2term

	def map_term2cui(self, pos2term, cuis):
		"""return list of (term, cui) pairs given term position and cuis"""
		for cui in cuis:
			# get positional info
			start = cui[0]['start']
			# check whether 'start' matches any pos2term key
			if start in pos2term:
				# update ["__NULL__"] w/ cui in first position (best candidate from QuickUMLS)
				pos2term[start][1] = cui[0]['cui']
		# return pos2term values only - i.e. (term, CUI) pairs
		return list(pos2term.values())

	def get_term2cui(self, word_dict, out_path, threshold=0.7, stypes_fname=None):
		"""map candidate CUIs to each indexed word"""
		terms = ' '.join(list(word_dict.keys()))
		# split terms into substrings of length <= 999999 -- max length allowed by scipy parser
		subs = wrap(terms, width=999999, break_long_words=False, break_on_hyphens=False)
		if stypes_fname is not None:  # load user-specified UMLS semantic types
			print("user-specified UMLS semantic types for QuickUMLS enabled")
			stypes = ','.join(self.load_stypes(stypes_fname))
		else:  # keep default QuickUMLS semantic types
			stypes = None
		# initialize QuickUMLS server
		server = QuickUMLS(window=1, threshold=threshold, semtypes=stypes)
		server.launch_quickumls()
		# initialize concept matcher
		matcher = get_quickumls_client()
		term2cui = []
		# extract concepts
		for sub in subs:
			cuis = matcher.match(sub)
			# get position dict {pos: [term, ["__NULL__"]]} given sub
			pos2term = self.get_pos2term(sub) 
			# associate each term to its candidate CUIs
			term2cui += self.map_term2cui(pos2term, cuis)
		# close connection w/ QuickUMLS server
		server.close_quickumls()
		# store term2cui as a dictionary
		print("store (word, cui) pairs as a dictionary")
		term2cui = dict(term2cui)
		with open(out_path + '/term2cui.json', 'w') as file_t2c:
			json.dump(term2cui, file_t2c)
		return term2cui

	def cui2source(self, term2cui, source='MSH'):
		"""keep only CUIs presenting an entry in the given 'source' lexicon"""
		cui2source = {}
		umls_lookup = umls.UMLSLookup()
		for term, cui in tqdm(term2cui.items()):
			if cui == '__NULL__':  # skip __NULL__ concepts
				cui2source[term] = '__NULL__'
			else:
				# lookup codes and sources from UMLS 
				codes_and_sources = umls_lookup.lookup_code(cui=cui, preferred=False)
				source_code = [code for code, src, _ in codes_and_sources if src == source]
				if source_code:  # CUI in source - keep it
					cui2source[term] = cui
				else:  # CUI not in source - discard it
					cui2source[term] = '__NULL__'
		# return cui2source
		return cui2source

	def get_syns(self, term2cui, term_dict):
		"""get synonymic relations between words within corpus (derived from a semantic lexicon)"""
		syns = {}
		umls_lookup = umls.UMLSLookup()
		analyzer = SimpleAnalyzer()
		for term, cui in term2cui.items():
			if term in term_dict:
				if cui != '__NULL__':
					# get synset composed of single-word terms (reference term excluded)
					synset = {syn[0].lower() for syn in umls_lookup.lookup_synonyms(cui, preferred=False) if len(list(analyzer(syn[0]))) == 1 and syn[0].lower() in term_dict and syn[0].lower() != term}
					if len(synset) > 0:
						syns[term] = list(synset)
					else:
						syns[term] = list()
				else:
					syns[term] = list()
		return syns

	def conceptualize_corpus(self, corpus, term_dict, term2cui, out_path):
		"""convert corpus from words to concepts (CUIs)"""
		cdocs = {}
		print('converting corpus words into concepts...')
		for docno, doc in tqdm(corpus.items()):
			cdocs[docno] = [term2cui[term].lower() for term in doc if term in term_dict and term2cui[term] != '__NULL__']
		print('conversion to concepts completed!')
		print('store processed data')
		with open(out_path + '/cdocs.json', 'w') as file_docs:
			json.dump(cdocs, file_docs)
		return cdocs	

	def compute_idfs(self, word_dict):
		"""compute IDF scores for words within corpus"""
		print('computing IDF scores for words within corpus')
		# obtain document frequencies
		dfreqs = dict(word_dict.dfs)
		# return IDF scores
		return {word_dict[idx]: np.log(word_dict.num_docs / (1 + float(dfreq))) for idx, dfreq in dfreqs.items()}

	def compute_doc_embs(self, corpus, model, weights=None):
		"""compute doc embeddings as the (weighted) sum of their word embeddings"""
		doc_embs = {}
		# loop over docs
		for docno, doc in tqdm(corpus.items()):
			if doc:  # doc is not empty
				word_embs = model.wv[[word for word in doc if word in model.wv.vocab]]
				if weights is None:  # compute doc embs as the sum of its word embs
					doc_embs[docno] = np.sum(word_embs, axis=0)
				else:  # compute doc embs as the weighted sum of its word embs
					wweights = np.array([weights[word] for word in doc if word in model.wv.vocab])
					doc_embs[docno] = np.sum(word_embs * wweights[:, np.newaxis], axis=0)  # add an extra dim to allow vector-scalar mul
			else:  # doc is empty - skip it
				continue
		return doc_embs

	def read_ohsu_queries(self, query_path):
		"""read queries and return a dict[id] = {title: <string>, desc: <string>}"""
		with open(query_path, 'r') as qf:
			queries = qf.read()
		queries = [query.split('\n') for query in queries.split('\n\n') if query]
		# loop through each query and fill query dict
		q = {}
		for query in queries:
			qid = query[1].split()[-1]
			q[qid] = {}
			q[qid]['title'] = query[2].split('<title>')[1].strip()
			q[qid]['desc'] = query[4]
		return q

	def read_cds_queries(self, query_path):
		"""read query file and return a dict[id] = {note: <string>, description: <string>, summary: <string>}"""
		with open(query_path, 'r') as qf:
			queries = qf.read()
		# convert queries to xml
		q = ETree.fromstring(queries)
		# loop through each query and fill dict
		qdict = dict()
		for query in q:
			qid = query.attrib['number']
			qdict[qid] = dict()
			# store query versions (i.e. note, description, summary)
			for version in query:
				qdict[qid][version.tag] = version.text.strip()
		return qdict

	def conceptualize_queries(self, queries, qfield, term_dict, term2cui):
		"""convert queries from words to concepts (CUIs)"""
		cqueries = {}
		for qid, qtext in queries.items():
			cqueries[qid] = {qfield: []}
			# convert target query field from words to CUIs
			q_tokens = self.tokenize_query(qtext[qfield])
			q_concepts = [term2cui[term].lower() for term in q_tokens if term in term_dict and term2cui[term] != '__NULL__']
			cqueries[qid][qfield] = ' '.join(q_concepts) 
		return cqueries 

	def disambiguate_query(self, ix, term_dict, concept_dict, token2cuis, query, table_name): 
		"""shallow word-sense disambiguation: disambiguate polysemous terms based on shallow word-concept connectivity within UMLS"""
		qcuis = {}
		umls_lookup = umls.UMLSLookup()
		# tokenize query 
		q = self.tokenize_query(query)
		# convert query into gensim doc2idx format
		q2idx = ix.doc2idx(q)
		# get cuis from query tokens
		for idx in q2idx:
			if idx in token2cuis and token2cuis[idx] != ["__NULL__"]: 
				for cui in token2cuis[idx]:
					if cui in qcuis:  # increase cui count
						qcuis[cui] += 1
					else:  # initialize cui count
						qcuis[cui] = 1
		# perform shallow word-sense disambiguation
		enc_query = []
		for idx in q2idx:
			if idx in term_dict:  # disambiguate only for terms contained within term_dict
				max_edges = 0  # relative maximum connections (edges)
				if len(token2cuis[idx]) == 1:  # monosemous term
					ref_cui = token2cuis[idx][0]
					# encode (term, cui) pair
					enc_query.append([term_dict[idx], concept_dict[ref_cui]])
				else:  # polysemous term
					candidates = []
					# loop over cadidate concepts
					for subj_cui in token2cuis[idx]:
						num_edges = 0  # number of edges
						if qcuis[subj_cui] == 1:  # subj_cui is only associated with current term (idx)
							obj_cuis = list(set(qcuis.keys()).difference({subj_cui}))
						else:  # subj_cui is associated with other terms in the query too
							obj_cuis = list(qcuis.keys())
						num_edges += umls_lookup.compute_num_edges(obj_cuis, subj_cui, table_name)  # remember that subj and obj are inverted within UMLS <s, p, o> triples
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
					# encode (term, cui) pair
					enc_query.append([term_dict[idx], concept_dict[ref_cui]])
			else:  # term oov
				continue
		return enc_query

	def tokenize_query(self, query, use_concepts=False):
		"""tokenize query"""
		analyzer = SimpleAnalyzer()
		if use_concepts:  # uppercase tokens since CUIs are in uppercase 
			return [token.text.upper() for token in analyzer(query)]
		else:
			return [token.text for token in analyzer(query)]

	def embed_query(self, query_words, model):
		"""project query words within the embedding space and return the query embedding as the sum of its word embeddings"""
		query_words = [term for term in query_words if term in model.wv.vocab]
		if not query_words:
			return None
		else:
			word_embs = model.wv[query_words]
			return np.sum(word_embs, axis=0)

	def query2emb(self, query, model, use_concepts=False):
		"""tokenize query and transform it into dense vector of size [1, opts.embs_size]"""
		query_words = self.tokenize_query(query, use_concepts)
		q_emb = self.embed_query(query_words, model)
		return q_emb

	def semantic_search(self, doc_ids, doc_embs, q_ids, q_embs, rankings_folder, ranking_name):
		"""perform semantic search over docs given queries"""
		print("compute similarities between doc and query embeddings")
		similarities = cosine_similarity(doc_embs, q_embs)
		# open file to write results
		rf = open(rankings_folder + '/' + ranking_name + '.txt', 'w')
		# write results in ranking file
		for i in tqdm(range(similarities.shape[1])):
			rank = np.argsort(-similarities[:, i])[:1000]
			doc_ranks = doc_ids[rank]
			qid = q_ids[i]
			if qid.isdigit():  # cast to integer - this operation avoids storing q_ids as '059' instead of '59'
				qid = str(int(qid))  # convert to int and then back to str
			for j in range(len(doc_ranks)):
				# write into .run file
				rf.write('%s %s %s %d %f %s\n' % (qid, 'Q0', doc_ranks[j], j, similarities[rank[j]][i], ranking_name))
		rf.close()
		return True

	def get_score(self, run, qrels, measure):
		"""return np array of scores for a given measure"""
		if "P_" in measure:
			cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + " " + qrels + " " + run
		elif "ndcg_cut" in measure:
			cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
		else:
			cmd = "./trec_eval/trec_eval -q -m " + measure + " " + qrels + " " + run
		# run trev_eval as a subprocess
		process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
		result = process.stdout.decode('utf-8').split('\n')
		# get scores
		scores = np.array([(score.split('\t')[-2], score.split('\t')[-1]) for score in result if score.split('\t')[0].strip() == measure and score.split('\t')[-2] != 'all'])
		return scores

	def get_averaged_measure_score(self, run, qrels, measure):
		"""return averaged measure score over topics"""
		if "P_" in measure:
			cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + " " + qrels + " " + run
		elif "ndcg_cut" in measure:
			cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
		else:
			cmd = "./trec_eval/trec_eval -m " + measure + " " + qrels + " " + run
		process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
		result = process.stdout.decode('utf-8').split('\n')
		if 'recall.' in measure:
			measure = '_'.join(measure.split('.'))
		qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
		qscore = qscore.astype(np.float)[0]
		return qscore

	def get_averaged_inferred_measure_score(self, run, qrels, measure):
		"""return averaged measure score over topics"""
		cmd = "perl sample_eval.pl " + qrels + " " + run
		process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
		result = process.stdout.decode('utf-8').split('\n')
		qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
		qscore = qscore.astype(np.float)[0]
		return qscore

	def evaluate(self, measures, rankings_folder, ranking_name, qrels_folder, qrels_name):
		"""evaluate models on given measures"""
		scores = []
		print('evaluate model ranking')
		if type(measures) == list:  # list of measures provided
			for measure in measures:
				scores.append(self.get_averaged_measure_score(rankings_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure)) 
		else:  # single measure provided
			scores.append(self.get_averaged_measure_score(rankings_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
		print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
		return {measure: score for measure, score in zip(measures, scores)}

	def evaluate_inferred(self, measures, rankings_folder, ranking_name, qrels_folder, qrels_name):
		"""evaluate models on given inferred measures"""
		scores = []
		print('evaluate model ranking')
		if type(measures) == list:  # list of inferred measures provided
			for measure in measures:
				scores.append(self.get_averaged_inferred_measure_score(rankings_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure))
		else:  # single inferred measure provided
			scores.append(self.get_averaged_inferred_measure_score(rankings_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
		print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
		return {measure: score for measure, score in zip(measures, scores)}


class EpochRanker(CallbackAny2Vec):
	"""callback to perform retrieval and evaluation during training"""

	def __init__(self, corpus, queries, weights, utils, include_docs, opts, model_folder, rankings_folder, qrels_folder):
		"""initialize required data"""
		self.epoch = 0
		self.best_epoch = 0
		self.best_score = 0.0
		self.corpus = corpus
		self.queries = queries
		self.weights = weights
		self.utils = utils
		self.include_docs = include_docs
		self.opts = opts
		self.model_folder = model_folder
		self.rankings_folder = rankings_folder
		self.qrels_folder = qrels_folder

	def on_epoch_begin(self, model):
		print('training model at epoch: {}'.format(self.epoch + 1))

	def on_epoch_end(self, model):
		# store model for current epoch
		print('store model at epoch: {}'.format(self.epoch + 1))
		model.save(self.model_folder + '/' + self.opts.model_name + '_' + str(self.epoch + 1) + '.model')
		# evaluate model for current epoch 
		print('evaluation at epoch: {}'.format(self.epoch + 1))
		
		# compute doc embs
		if self.include_docs:  # get doc embs from Doc2Vec models
			print('get document embeddings')
			doc_embs = model.docvecs.vectors_docs
			doc_ids = np.array(list(model.docvecs.doctags.keys()))
		else:  # compute doc embs
			print('compute document embeddings')
			doc_ids_embs = self.utils.compute_doc_embs(self.corpus, model, self.weights)
			# get doc embs and ids from doc_ids_embs
			doc_embs = np.array(list(doc_ids_embs.values()))
			doc_ids = np.array(list(doc_ids_embs.keys()))
		
		q_ids = list()
		q_embs = list()
		# loop over queries and generate query embs
		for qid, qtext in self.queries.items():
			# compute query emb as the sum of its word embs
			q_emb = self.utils.query2emb(qtext[self.opts.qfield], model, use_concepts=self.opts.train_on_concepts)
			if q_emb is None:
				print('query {} does not contain known terms'.format(qid))
			else:
				q_embs.append(q_emb)
				q_ids.append(qid)
		# convert q_embs to numpy
		q_embs = np.array(q_embs)

		# perform semantic search with trained embs
		self.utils.semantic_search(doc_ids, doc_embs, q_ids, q_embs, self.rankings_folder, self.opts.model_name + '_' + str(self.epoch + 1))
		
		if "OHSUMED" in self.opts.corpus_name:  # evaluate OHSUMED collection
			# evaluate ranking list
			scores = self.utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], self.rankings_folder, self.opts.model_name + '_' + str(self.epoch + 1), self.qrels_folder, self.opts.qrels_fname)
		elif "TREC_CDS" in self.opts.corpus_name:  # evaluate TREC CDS collections
			print('evaluate model for CDS 2016')
			scores = self.utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], self.rankings_folder, self.opts.model_name + '_' + str(self.epoch + 1), self.qrels_folder, self.opts.qrels_fname)
			inf_scores = self.utils.evaluate_inferred(['infNDCG'], self.rankings_folder, self.opts.model_name + '_' + str(self.epoch + 1), self.qrels_folder, self.opts.infqrels_fname)
			if 'inf' in self.opts.ref_measure:  # optimize model w/ reference inf measure
				scores = inf_scores
		# check if current scores[self.opts.ref_measure] is greater than or equal to best_score
		if scores[self.opts.ref_measure] >= self.best_score:  # update self.best_score and self.best_epoch
			self.best_score = scores[self.opts.ref_measure]
			self.best_epoch = self.epoch + 1
		# update epoch 
		self.epoch += 1
