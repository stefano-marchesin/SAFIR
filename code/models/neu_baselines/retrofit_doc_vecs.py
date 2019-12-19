import os
import gensim
import sys
import json
import argparse
import numpy as np 

from tqdm import tqdm

from gensim_utils import Utils


flags = argparse.ArgumentParser()

flags.add_argument("--wdoc2vec", default="", type=str, 
				   help="Path to word-based Doc2Vec model.")
flags.add_argument("--cdoc2vec", default="", type=str, 
				   help="Path to concept-based Doc2Vec model.")
flags.add_argument("--init", default="unif", type=str, help="Initialization strategy for retrofitted doc vectors. Choose between 'unif' and 'wdoc2vec'.")
flags.add_argument("--norm", default=False, type=bool, help="Whether to normalize doc vectors prior retrofitting.")
flags.add_argument("--beta", default=None, type=float, help="Regularization parameter to retrofit doc vectors using word-based Doc2Vec and concept-based Doc2Vec models.")
flags.add_argument("--sweep", default=0.1, type=float, help="Sweeping parameter to perform rDoc2Vec optimization over beta.")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--qrels_fname", default="qrels-treceval-2015", type=str, help="Qrels filename.")
flags.add_argument("--infqrels_fname", default="qrels-sampleval-2015", type=str, help="Inferred qrels file name.")
flags.add_argument("--query_fname", default="topics2015A.xml", type=str, help="Query filename.")
flags.add_argument("--query_field", default="summary", type=str, help="Query field to consider for retrieval.")
flags.add_argument("--reference_measure", default="infNDCG", type=str, help="Reference measure to consider for optimization.")
flags.add_argument("--corpus_name", default="TREC_CDS14_15", type=str, help="Corpus to consider.")
flags.add_argument("--model_name", default="rdoc2vec_cds15", type=str, help="Model name.")

FLAGS = flags.parse_args()


class Options(object):
	"""options used by Doc2Vec and ConceptualDoc2Vec models"""
	def __init__(self):
		# word-based doc2vec path
		self.wdoc2vec_path = FLAGS.wdoc2vec
		# concept-based doc2vec path
		self.cdoc2vec_path = FLAGS.cdoc2vec
		# initialization strategy
		self.init = FLAGS.init
		# normalization 
		self.norm = FLAGS.norm
		# beta
		self.beta = FLAGS.beta
		# sweep 
		self.sweep = FLAGS.sweep
		# seed
		self.seed = FLAGS.seed
		# corpus name
		self.corpus_name = FLAGS.corpus_name
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname 
		# inferred qrels file name
		self.infqrels_fname = FLAGS.infqrels_fname
		# query file name
		self.query_fname = FLAGS.query_fname
		# query field
		self.qfield = FLAGS.query_field
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# model name
		self.model_name = FLAGS.model_name


def z_normalize(vector):
	"""perform z-normalization over vector"""
	mean = np.mean(vector)
	std = np.std(vector)
	if std != 0:
		vector = (vector - mean) / std
	else: 
		vector = vector - vector
	return vector


def retrofit(wdoc2vec, cdoc2vec, init='unif', norm=False, beta=0.6):
	"""retrofit doc vectors using word- and concept-based doc vectors"""
	try:
		# sanity check on doc lengths
		wdoc_size = len(wdoc2vec.docvecs[0])
		cdoc_size = len(cdoc2vec.docvecs[0]) 
		assert wdoc_size == cdoc_size
	except Exception as e:
			print('inconsistent vector size between the 2 models: {} vs {}'.format(wdoc_size, cdoc_size))
	# get docnos from txt_d2v_model
	docnos = wdoc2vec.docvecs.doctags.keys()
	
	# vector inizialization for retrofitted docs
	retro_docs = {}
	for docno in docnos:
		retro_docs[docno] = np.random.uniform(-1, 1, wdoc_size)
		if init == 'wdoc2vec':  # initialize doc vectors as word-based doc2vec vectors
			retro_docs = wdoc2vec.docvecs[docno]
		if norm:  # normalize doc vectors prior retrofitting
			retro_docs[docno] = z_normalize(retro_docs[docno])

	# retrofit doc vectors using closed formula solution
	for docno in tqdm(docnos):
		if docno in wdoc2vec.docvecs and docno in cdoc2vec.docvecs:
			retro_docs[docno] = beta * wdoc2vec.docvecs[docno] + (1-beta) * cdoc2vec.docvecs[docno]
		elif docno in wdoc2vec.docvecs:
			retro_docs[docno] = beta * wdoc2vec.docvecs[docno]
		elif docno in cdoc2vec.docvecs:
			retro_docs[docno] = (1-beta) * cdoc2vec.docvecs[docno]
	return retro_docs


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set model folders
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	data_folder = 'corpus/' + opts.corpus_name + '/data'
	model_folder = 'corpus/' + opts.corpus_name + '/models/' + opts.model_name
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings/' + opts.model_name
	
	# create folders 
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# load utils functions
	utils = Utils()
	# set random seed - enable repro
	np.random.seed(opts.seed)

	# load queries
	print('load {} queries'.format(opts.corpus_name))
	if 'OHSUMED' in opts.corpus_name:  # load ohsu queries
		qdict = utils.read_ohsu_queries(query_folder + '/' + opts.query_fname)
	elif 'TREC_CDS' in opts.corpus_name:  # load cds queries
		qdict = utils.read_cds_queries(query_folder + '/' + opts.query_fname)
	
	"""
	QUERY PREPROCESSING
	"""

	# load required data
	print('load index')
	ix = gensim.corpora.Dictionary.load_from_text(index_folder + '/' + opts.corpus_name + '_ix.txt')
	ix.id2token = {v: k for k, v in ix.token2id.items()}

	print('load safir candidate cuis')
	with open(data_folder + '/token2cuis.json', 'r') as t2cf:
		token2cuis = json.load(t2cf)
	# cast keys to integers - json stores dict keys always as string
	token2cuis = {int(k): v for k, v in token2cuis.items()}
	print('load safir term dict')
	with open(data_folder + '/term_dict.json', 'r') as std:
		ix2ix = json.load(std)
	# cast keys to integers - json stores dict keys always as string
	ix2ix = {int(k): v for k, v in ix2ix.items()}
	print('load safir concept dict')
	with open(data_folder + '/concept_dict.json', 'r') as scd:
		ix2dict = json.load(scd)

	cqdict = {}
	# disambiguate query
	for qid, qbody in qdict.items():
		enc_query = utils.disambiguate_query(ix, ix2ix, ix2dict, token2cuis, qbody[opts.qfield], opts.corpus_name)
		cqdict[qid] = {opts.qfield: enc_query}

	# remove '__NULL__' concept from safir concept dict
	ix2dict.pop('__NULL__', None)
	# reverse safir concept dict
	ix2dict = {v: k for k, v in ix2dict.items()}
	print('convert queries to concepts-only')
	for qid, qbody in cqdict.items():
		cuis_query = ' '.join([ix2dict[cui] for _, cui in cqdict[qid][opts.qfield] if cui in ix2dict])
		cqdict[qid][opts.qfield] = cuis_query
	print('converted TREC CDS queries become ... "{}" ...'.format(list(cqdict.values())[0][opts.qfield]))

	"""
	LOADING MODELS
	"""

	# load doc2vec models
	print('load word-based doc2vec model')
	wdoc2vec = gensim.models.Doc2Vec.load(opts.wdoc2vec_path)
	print('load concept-based doc2vec model')
	cdoc2vec = gensim.models.Doc2Vec.load(opts.cdoc2vec_path)

	"""
	RETROFITTING AND EVALUATION
	"""

	if opts.beta:
		# retrofit doc vectors
		print('retrofit doc vectors with initialization: {}, normalization: {}, and beta: {}'.format(opts.init, opts.norm, opts.beta))
		doc_ids_embs = retrofit(wdoc2vec, cdoc2vec, opts.init, opts.norm, opts.beta)
		# convert retrofitted doc vectors in gensim doc2vec format
		rdoc2vec = gensim.models.doc2vec.Doc2VecKeyedVectors(vector_size=wdoc2vec.vector_size, mapfile_path=None)
		# store retrofitted doc vectors within rdoc2vec
		rdoc2vec.add(entities=list(doc_ids_embs.keys()), weights=list(doc_ids_embs.values()))
		# store rdoc2vec model
		print('store rDoc2Vec model')
		rdoc2vec.save(model_folder + '/' + opts.model_name + '_beta_' + str(opts.beta) + '_' + opts.ref_measure + '.model.docs')
		
		# convert word/concept vectors in gensim word2vec format
		rword2vec = gensim.models.word2vec.Word2Vec(size=wdoc2vec.vector_size)
		# store word/concept vectors within rword2vec
		if opts.beta < 0.5:  # concept vectors
			rword2vec.wv = cdoc2vec.wv
		else:  # word vectors
			rword2vec.wv = wdoc2vec.wv
		print('store word/concept vectors used by rDoc2Vec model')
		rword2vec.save(model_folder + '/' + opts.model_name + '_beta_' + str(opts.beta) + '_' + opts.ref_measure + '.model.tokens')

		# get doc embs and ids from doc_ids_embs
		doc_embs = np.array(list(doc_ids_embs.values()))
		doc_ids = np.array(list(doc_ids_embs.keys()))

		q_ids = list()
		q_embs = list()

		# loop over queries and generate query embs
		if beta < 0.5:  # cqdict 
			for qid, qtext in cqdict.items():
				# compute query emb as the sum of its word/concept embs
				q_emb = utils.query2emb(qtext[opts.qfield], rword2vec, use_concepts=True)
				if q_emb is None:
					print('query {} does not contain known concepts'.format(qid))
				else:
					q_embs.append(q_emb)
					q_ids.append(qid)
		else:  # qdict
			for qid, qtext in qdict.items():
				# compute query emb as the sum of its word/concept embs
				q_emb = utils.query2emb(qtext[opts.qfield], rword2vec)
				if q_emb is None:
					print('query {} does not contain known terms'.format(qid))
				else:
					q_embs.append(q_emb)
					q_ids.append(qid)
		
		# convert q_embs to numpy
		q_embs = np.array(q_embs)

		# perform semantic search with trained embs
		utils.semantic_search(doc_ids, doc_embs, q_ids, q_embs, rankings_folder, opts.model_name + '_beta_' + str(opts.beta) + '_' + opts.ref_measure)
		# evaluate ranking list
		scores = utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder, opts.model_name + '_beta_' + str(opts.beta) + '_' + opts.ref_measure, qrels_folder, opts.qrels_fname)
	else:
		print('perform optimization over beta with sweep: {}'.format(opts.sweep))
		best_score = 0.0
		best_beta = 0
		for beta in np.arange(opts.sweep, 1.0, opts.sweep):
			# set beta to 4-decimal points
			beta = round(beta, 4)
			# retrofit doc vectors
			print('retrofit doc vectors with initialization: {}, normalization: {}, and beta: {}'.format(opts.init, opts.norm, beta))
			doc_ids_embs = retrofit(wdoc2vec, cdoc2vec, opts.init, opts.norm, beta)
			# convert retrofitted doc vectors in gensim doc2vec format
			rdoc2vec = gensim.models.doc2vec.Doc2VecKeyedVectors(vector_size=wdoc2vec.vector_size, mapfile_path=None)
			# store retrofitted doc vectors within rdoc2vec
			rdoc2vec.add(entities=list(doc_ids_embs.keys()), weights=list(doc_ids_embs.values()))

			# convert word/concept vectors in gensim word2vec format
			rword2vec = gensim.models.word2vec.Word2Vec(size=wdoc2vec.vector_size)
			# store word/concept vectors within rword2vec
			if beta < 0.5:  # concept vectors
				rword2vec.wv = cdoc2vec.wv
			else:  # word vectors
				rword2vec.wv = wdoc2vec.wv
	
			# get doc embs and ids from doc_ids_embs
			doc_embs = np.array(list(doc_ids_embs.values()))
			doc_ids = np.array(list(doc_ids_embs.keys()))

			q_ids = list()
			q_embs = list()

			# loop over queries and generate query embs
			if beta < 0.5:  # cqdict 
				for qid, qtext in cqdict.items():
					# compute query emb as the sum of its word/concept embs
					q_emb = utils.query2emb(qtext[opts.qfield], rword2vec, use_concepts=True)
					if q_emb is None:
						print('query {} does not contain known concepts'.format(qid))
					else:
						q_embs.append(q_emb)
						q_ids.append(qid)
			else:  # qdict
				for qid, qtext in qdict.items():
					# compute query emb as the sum of its word/concept embs
					q_emb = utils.query2emb(qtext[opts.qfield], rword2vec)
					if q_emb is None:
						print('query {} does not contain known terms'.format(qid))
					else:
						q_embs.append(q_emb)
						q_ids.append(qid)
		
			# convert q_embs to numpy
			q_embs = np.array(q_embs)

			# perform semantic search with trained embs
			utils.semantic_search(doc_ids, doc_embs, q_ids, q_embs, rankings_folder, opts.model_name + '_beta_' + str(beta) + '_' + opts.ref_measure)

			if "OHSUMED" in opts.corpus_name:  # evaluate OHSUMED collection
				# evaluate ranking list
				scores = utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder, opts.model_name + '_beta_' + str(beta) + '_' + opts.ref_measure, qrels_folder, opts.qrels_fname)
			elif "TREC_CDS" in opts.corpus_name:  # evaluate TREC CDS collections
				print('evaluate model for CDS 2016')
				scores = utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder, opts.model_name + '_beta_' + str(beta) + '_' + opts.ref_measure, qrels_folder, opts.qrels_fname)
				inf_scores = utils.evaluate_inferred(['infNDCG'], rankings_folder, opts.model_name + '_beta_' + str(beta) + '_' + opts.ref_measure, qrels_folder, opts.infqrels_fname)
				if 'inf' in opts.ref_measure:  # optimize model w/ reference inf measure
					scores = inf_scores
			
			# update best score and best beta 
			if scores[opts.ref_measure] >= best_score:
				best_score = scores[opts.ref_measure]
				best_beta = beta

		print('retrofitted Doc2Vec found for beta={} with {}={}'.format(best_beta, opts.ref_measure, best_score))
		# train and store rDoc2Vec using best_beta
		print('retrofit doc vectors with initialization: {}, normalization: {}, and best beta: {}'.format(opts.init, opts.norm, best_beta))
		doc_ids_embs = retrofit(wdoc2vec, cdoc2vec, opts.init, opts.norm, best_beta)
		# convert retrofitted doc vectors in gensim doc2vec format
		rdoc2vec = gensim.models.doc2vec.Doc2VecKeyedVectors(vector_size=wdoc2vec.vector_size, mapfile_path=None)
		# store retrofitted doc vectors within rdoc2vec
		rdoc2vec.add(entities=list(doc_ids_embs.keys()), weights=list(doc_ids_embs.values()))
		# store rdoc2vec model
		print('store rDoc2Vec model')
		rdoc2vec.save(model_folder + '/' + opts.model_name + '_beta_' + str(best_beta) + '_' + opts.ref_measure + '.model.docs')
		
		# convert word/concept vectors in gensim word2vec format
		rword2vec = gensim.models.word2vec.Word2Vec(size=wdoc2vec.vector_size)
		# store word/concept vectors within rword2vec
		if best_beta < 0.5:  # concept vectors
			rword2vec.wv = cdoc2vec.wv
		else:  # word vectors
			rword2vec.wv = wdoc2vec.wv
		print('store word/concept vectors used by rDoc2Vec model')
		rword2vec.save(model_folder + '/' + opts.model_name + '_beta_' + str(best_beta) + '_' + opts.ref_measure + '.model.tokens') 


if __name__ == "__main__":
	main()