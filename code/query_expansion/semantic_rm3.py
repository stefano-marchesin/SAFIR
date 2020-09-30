import os
import argparse
import json
import gensim
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from gensim.corpora import Dictionary
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from es_utils import settings
from es_utils import index as es_index
from es_utils import utils
from safir_index import Index
from gensim_utils import Utils


flags = argparse.ArgumentParser()

flags.add_argument("--corpus_name", default="OHSUMED", type=str, help="Used corpus.")
flags.add_argument("--query_fname", default="topics_all", type=str, help="Queries file name.")
flags.add_argument("--qrels_fname", default="qrels_all", type=str, help="Qrels file name.")
flags.add_argument("--inf_qrels_fname", default="", type=str, help="Inferred qrels file name.")
flags.add_argument("--query_field", default="desc", type=str, help="Query field to consider when performing retrieval.")
flags.add_argument("--lexical", default=False, dest='lexical', action='store_true', help="Whether to consider lexical models for the second round of RM3 prf.")
flags.add_argument("--model_name", default="BM25", type=str, help="Model name. Tip: model names are, among the others, 'BM25', 'LMDirichlet', 'DFR'.")
flags.add_argument("--retrofitted", default=False, dest='retrofitted', action='store_true', help="Whether to consider Word2Vec or rWord2Vec.")
flags.add_argument("--semantic_path", default="/mnt/c/Users/stefa/Desktop/TOIS_MAJOR/rm3/corpus/OHSUMED/models/word2vec_ohsu14.model", type=str, help="Path to trained semantic model.")
flags.add_argument("--feedback_docs", default=10, type=int, help="Number of feedback documents considered for pseudo-releance feedback.")
flags.add_argument("--feedback_terms", default=10, type=int, help="Number of feedback terms considered for pseudo-releance feedback.")
flags.add_argument("--query_weight", default=0.5, type=float, help="The weight applied to terms belonging to the original query when interpolating.")
flags.add_argument("--run_name", default="word2vec_rm3_word2vec", type=str, help="Run name.")

FLAGS = flags.parse_args()


class Options(object):
	"""options used to perform re-ranking w/ gensim models"""

	def __init__(self):
		# semantic model
		self.semantic_model = FLAGS.semantic_path


def scale_to_L1_norm(vec):
	"""scale input vector using L1 norm"""

	norm = sum(vec.values())
	for term, score in vec.items():
		vec[term] = score / norm
	return vec


def interpolate(qfv, rm1, qweight):
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


def create_feature_vector(ix, term_dict, bow): 
	"""create the feature vector out of doc terms"""

	tfv = list()
	# get corpus length (n. of docs)
	num_docs = ix.num_docs
	for idx, tf in bow:
		# get term from dict index
		term = ix[idx]
		# filter out terms not contained in self.term_dict
		if term not in term_dict:
			continue
		# filter out terms w/ length gt 20
		if len(term) > 20:
			continue
		# filter out non-alphabetical terms
		if not term.isalpha():
			continue
		# get document frequency 
		df = ix.dfs[idx]
		# compute ratio between df and num_docs
		ratio = df / num_docs
		if ratio > 0.1:  # skip term - requires tuning: check if it's okay to keep it as is
			continue
		# append term w/ tf to tfv
		tfv.append((term, tf))
	return tfv


def create_query_vector(ix, term_dict, bow):
	"""create the feature vector out of query terms"""

	qfv = list()
	for idx, tf in bow:
		# get term from dict index
		term = ix[idx]
		# filter out terms not contained in self.term_dict
		if term not in term_dict:
			continue
		# append term w/ tf to tfv
		qfv.append((term, tf))
	return scale_to_L1_norm(defaultdict(float, qfv))


def neural_relevance_model(ix, term_dict, ids_and_scores, corpus, fb_terms):
	"""estimate RM1 based on neural models"""

	# set variables
	rm1_vec = list()
	vocab = set()
	doc_vecs = dict()

	# create document feature vectors for each feedback doc
	for doc_id in ids_and_scores.keys():
		# convert current doc to bow format
		bow = ix.doc2bow(corpus[doc_id])
		# create document feature vector
		dfv = create_feature_vector(ix, term_dict, bow)
		# keep top 'fb_terms' from dfv
		dfv = defaultdict(int, sorted(dfv, key=lambda x: (-x[1], x[0]))[:fb_terms])  # -x[1] represents descending order
		# update vocab with top 'fb_terms' terms contained within feedback docs and store document feature vectors
		vocab.update(dfv.keys())
		doc_vecs[doc_id] = dfv

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
	return scale_to_L1_norm(rm1_vec)


def neural_rm3_prf(neu, ix, word_embs, doc_embs, corpus_name, corpus, doc_ids, queries, qfield, rank_path, run_name, es=None, fb_docs=10, fb_terms=10, qweight=0.5):
	"""perform pseudo-relevance feedback w/ RM3 model using neural models at first round and neural/lexical at second round"""

	# project queries within doc latent space
	proj_queries = list()
	query_ids = list()
	for qid, qbody in queries.items():
		proj_query = neu.query2emb(qbody[qfield], word_embs)
		if proj_query is not None:  # keep query
			proj_queries.append(proj_query)
			query_ids.append(qid)

	# compute the cosine similarity between docs and queries
	scores = cosine_similarity(doc_embs, proj_queries)

	if es is not None:  # open ranking 
		out = open(rank_path + '/' + run_name + '.txt', 'w')

	# loop over queries and perform RM3 expansion
	exp_queries = list()
	# loop over the cosine similarity scores
	for qix in tqdm(range(scores.shape[1])):
		# get the index of the top fb_docs documents from result list
		first_qres = np.argsort(-scores[:, qix])[:fb_docs]
		# store ids and scores of the retrieved feedback documents
		ids_and_scores = dict()
		for res in first_qres:
			if es is None:  # perform RM3 using neural model after pseudo-relevance feedback
				ids_and_scores[res] = scores[res][qix]
			else:  # perform RM3 using lexical model after pseudo-relevance feedback
				ids_and_scores[doc_ids[res]] = scores[res][qix]

		if es is None:  # perform RM3 using neural model after pseudo-relevance feedback
			# get query feature vector and normalize to L1 norm
			qfv = create_query_vector(ix, word_embs.wv.vocab, ix.doc2bow(neu.tokenize_query(queries[query_ids[qix]][qfield])))
			# get relevance model feature vector (i.e., RM1)
			rm1 = neural_relevance_model(ix, word_embs.wv.vocab, ids_and_scores, corpus, fb_terms)
			# interpolate qfv and rm1 (i.e., RM3)
			rm3 = interpolate(qfv, rm1, qweight)

			# extract terms and scores from rm3
			rm3_terms = list(rm3.keys())
			rm3_scores = np.array(list(rm3.values()))
			# project expanded query into document latent space
			proj_query = np.sum(np.multiply(word_embs.wv[rm3_terms], rm3_scores[:, np.newaxis]), axis=0)
			# append projected query to exp_queries
			exp_queries.append(proj_query)
		else:  # perform RM3 using lexical model after pseudo-relevance feedback
			# get query feature vector and normalize to L1 norm
			qfv = es.scale_to_L1_norm(Counter(es.analyze_query(queries[query_ids[qix]][qfield])))
			# get relevance model feature vector (i.e., RM1)
			rm1 = es.estimate_relevance_model(ids_and_scores, fb_terms)
			# interpolate qfv and rm1 (i.e., RM3)
			rm3 = es.interpolate(qfv, rm1, qweight)

			# build boosted term queries
			term_queries = [{'term': {es.field: {'value': term, 'boost': score}}} for term, score in rm3.items()]
			# combine term queries w/ SHOULD operator
			expanded_query = {'query': { 'bool': {'should': term_queries}}}

			# perform lexical search after pseudo-relevance feedback
			prf_qres = es.es.search(index=es.index, size=1000, body=expanded_query)
			for idx, rank in enumerate(prf_qres['hits']['hits']):
				out.write('%s %s %s %d %f %s\n' % (query_ids[qix], 'Q0', rank['_id'], idx, rank['_score'], run_name))

	if es is None:  # perform neural search after pseudo-relevance feedback
		neu.semantic_search(doc_ids, doc_embs, query_ids, exp_queries, rank_path, run_name)
	else:  # close ranking
		out.close()
	return True


def main():

	index_folder = './corpus/' + FLAGS.corpus_name + '/index/'
	data_folder = './corpus/' + FLAGS.corpus_name + '/data/'
	model_folder = './corpus/' + FLAGS.corpus_name + '/models/'
	ranking_folder = './corpus/' + FLAGS.corpus_name + '/rankings/rm3_runs_orig/'
	query_folder = './corpus/' + FLAGS.corpus_name + '/queries/'
	qrels_folder = './corpus/' + FLAGS.corpus_name + '/qrels/'

	# create folders
	if not os.path.exists(ranking_folder):
		os.makedirs(ranking_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# load options 
	opts = Options()
	# set Neural index
	neu = Utils()
	# set random seed
	np.random.seed(42)
		

	# load required data
	print('load index')
	ix = gensim.corpora.Dictionary.load_from_text(index_folder + '/' + FLAGS.corpus_name + '_ix.txt')
	ix.id2token = {v: k for k, v in ix.token2id.items()}
	print('load safir term dict')
	with open(data_folder + '/term_dict.json', 'r') as std:
		ix2dict = json.load(std)
	# cast keys to integers - json stores dict keys always as string
	ix2dict = {v: int(k) for k, v in ix2dict.items()}
	print('load encoded corpus')
	with open(data_folder + '/enc_corpus.json', 'r') as corpus_file:
		docs = json.load(corpus_file)
	print('convert corpus to words-only')
	docs = [[ix.id2token[ix2dict[word]] for word, _ in doc] for doc in tqdm(docs)]
	print('converted corpus becomes ... "{}" ...'.format(' '.join(docs[0][:10])))
	print('load doc ids')
	with open(data_folder + '/doc_ids.json', 'r') as docids_file:
		doc_ids = json.load(docids_file)
	# create term dictionary 
	term_dict = gensim.corpora.Dictionary(docs)
	# compute IDF scores for words within corpus
	weights = neu.compute_idfs(term_dict)

	# restore model and get required vectors
	print('load semantic model')
	if FLAGS.retrofitted:  # load rWord2Vec
		model = gensim.models.word2vec.Word2VecKeyedVectors.load(opts.semantic_model)
	else:  # load Word2Vec
		model = gensim.models.Word2Vec.load(opts.semantic_model)
	# get document vectors
	print('compute doc vectors for Word2Vec models')
	doc_ids_embs = neu.compute_doc_embs(dict(zip(doc_ids, docs)), model, weights)
	del docs  # free memory space
	# get doc embs and ids from doc_ids_embs
	doc_embs = np.array(list(doc_ids_embs.values()))
	doc_ids = np.array(list(doc_ids_embs.keys()))
	print('model loaded!')

	print("load pre-processed corpus")
	with open(data_folder + 'pproc_corpus.json', 'r') as f:
			pproc_corpus = json.load(f)

	if FLAGS.lexical:
		# set ElasticSearch constants and properties
		es_settings = settings.ESS(FLAGS.corpus_name)
		constants = es_settings.constants
		properties = es_settings.properties
		del es_settings

		# set ElasticSearch index
		es = es_index.Index(constants, properties)

		# set lexical model
		if FLAGS.model_name == 'BM25':
			print('change model to BM25 with k2={} and b={}'.format(1.2, 0.75))
			es.change_model(model='BM25', k2=1.2, b=0.75)
		elif FLAGS.model_name == 'LMDirichlet':
			print('change model to QLM (Dirichlet) with mu={}'.format(2000))
			es.change_model(model='LMDirichlet', mu=2000)
		elif FLAGS.model_name == 'DFR':
			print('change model to DFR with basic_model={}, after_effect={}, and normalization={}'.format('if', 'b', 'h2'))
			es.change_model(model='DFR', basic_model='if', after_effect='b', normalization='h2')
		else: 
			print('select a model between BM25, LMDirichlet, and DFR')
			return False
	else:
		es = None

	# read queries
	if FLAGS.corpus_name == 'OHSUMED':
		queries = neu.read_ohsu_queries(query_folder + '/' + FLAGS.query_fname)
	else:
		queries = neu.read_cds_queries(query_folder + '/' + FLAGS.query_fname)

	# project queries within doc latent space
	proj_queries = list()
	query_ids = list()
	for qid, qbody in queries.items():
		proj_query = neu.query2emb(qbody[FLAGS.query_field], model)
		if proj_query is not None:  # keep query
			proj_queries.append(proj_query)
			query_ids.append(qid)
	# perform neural search at first round
	neu.semantic_search(doc_ids, doc_embs, query_ids, proj_queries, ranking_folder, FLAGS.run_name)
	# evaluate baseline run
	print('evaluate baseline search')
	neu.evaluate(['ndcg_cut_1000', 'ndcg_cut_100', 'ndcg_cut_10', 'P_10', 'recall_1000'], ranking_folder, FLAGS.run_name, qrels_folder, FLAGS.qrels_fname)
	if 'CDS' in FLAGS.corpus_name:
		neu.evaluate_inferred(['infNDCG'], ranking_folder, FLAGS.run_name, qrels_folder, FLAGS.inf_qrels_fname) 

	# perform RM3 pseudo-relevance feedback w/ neural models
	neural_rm3_prf(neu, ix, model, doc_embs, FLAGS.corpus_name, pproc_corpus, doc_ids, queries, FLAGS.query_field, ranking_folder, FLAGS.run_name, es=es, 
				   fb_docs=FLAGS.feedback_docs, fb_terms=FLAGS.feedback_terms, qweight=FLAGS.query_weight)
	# evaluate RM3 run
	print('evaluate RM3-enhanced search')
	neu.evaluate(['ndcg_cut_1000', 'ndcg_cut_100', 'ndcg_cut_10', 'P_10', 'recall_1000'], ranking_folder, FLAGS.run_name, qrels_folder, FLAGS.qrels_fname)
	if 'CDS' in FLAGS.corpus_name:
		neu.evaluate_inferred(['infNDCG'], ranking_folder, FLAGS.run_name, qrels_folder, FLAGS.inf_qrels_fname) 


if __name__ == "__main__":
	main()