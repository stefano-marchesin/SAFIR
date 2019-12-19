import os
import argparse
import json
import math
import gensim
import numpy as np 

from copy import deepcopy
from tqdm import tqdm
from gensim.corpora import Dictionary

from gensim_utils import Utils


flags = argparse.ArgumentParser()

flags.add_argument("--word2vec", default="", type=str, help="Path to stored semantic model.")
flags.add_argument("--iterations", default=10, type=int, help="Number of iteration to retrofit word vectors using lexicon information.")
flags.add_argument("--weighting_scheme", default=True, type=bool, help="Whether to consider IDF weighting scheme to compose document embeddings.")
flags.add_argument("--qrels_fname", default="qrels_all", type=str, help="Qrels filename.")
flags.add_argument("--infqrels_fname", default="qrels-sampleval-2015", type=str, help="Inferred qrels file name.")
flags.add_argument("--query_fname", default="topics_all", type=str, help="Query filename.")
flags.add_argument("--qfield", default="desc", type=str, help="Query field to consider for retrieval.")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--reference_measure", default="ndcg", type=str, help="Measure used to optimize (base) word2vec model.")
flags.add_argument("--corpus_name", default="OHSUMED", type=str, help="Corpus to consider.")
flags.add_argument("--model_name", default="", type=str, help="Model name.")

FLAGS = flags.parse_args()


class Options(object):
	"""options used by the retrofitted word2vec model"""
	def __init__(self):
		# word2vec model path
		self.word2vec_path = FLAGS.word2vec
		# number of iterations to retrofit word vectors
		self.iterations = FLAGS.iterations
		# weighting scheme
		self.weighting = FLAGS.weighting_scheme
		# qrels filename
		self.qrels_fname = FLAGS.qrels_fname
		# inferred qrels file name
		self.infqrels_fname = FLAGS.infqrels_fname
		# query filename
		self.query_fname = FLAGS.query_fname
		# query field
		self.qfield = FLAGS.qfield
		# seed 
		self.seed = FLAGS.seed
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# corpus name
		self.corpus_name = FLAGS.corpus_name
		# model name
		self.model_name = FLAGS.model_name


def norm_word(word):
	return word.lower()


def norm_embs(word_embs):
	"""read all the word vectors and normalize them"""
	norm_embs = []
	for emb in word_embs:
		# normalize weight vector
		norm_embs.append(emb / math.sqrt((emb**2).sum() + 1e-6))
	return np.array(norm_embs)


def retrofit(model, syns, num_iters, alpha=1.0):
	"""retrofit word vectors to a lexicon"""
	new_embs = {word: model.wv[word] for word, _ in model.wv.vocab.items()}
	for it in tqdm(range(num_iters)):
		# loop through every node also in ontology (else just use data estimate)
		for word, synset in syns.items():
			num_syns = len(synset)
			# no synonyms, pass - use data estimate
			if num_syns == 0:
				continue
			# the weight of the data estimate if the number of neighbours is > 0
			new_emb = (num_syns * alpha) * model.wv[word]
			# loop over synonyms and add to new vector
			weights = []
			for ix, syn in enumerate(synset):
				syn_weight = 1 / num_syns  # synonyms weights are set to degree(ix)^-1 -- i.e. number of synonyms that ix has
				weights.append(syn_weight)
				new_emb += new_embs[syn] * weights[ix]
			new_embs[word] = new_emb / (np.sum(weights) + (num_syns * alpha))
	return new_embs


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set folders
	corpus_folder = 'corpus/' + opts.corpus_name + '/' + opts.corpus_name
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	model_folder = 'corpus/' + opts.corpus_name + '/models/' + opts.model_name
	data_folder = 'corpus/' + opts.corpus_name + '/data'
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
		
	"""MODEL PRE-PROCESSING"""

	# load required data
	print('load index')
	ix = Dictionary.load_from_text(index_folder + '/' + opts.corpus_name + '_ix.txt')
	ix.id2token = {v: k for k, v in ix.token2id.items()}
	print('load safir dict')
	with open(data_folder + '/term_dict.json', 'r') as std:
		ix2dict = json.load(std)
	# reverse safir dict
	ix2dict = {v: int(k) for k, v in ix2dict.items()}
	print('load safir candidate cuis')
	with open(data_folder + '/token2cuis.json', 'r') as t2cf:
		token2cuis = json.load(t2cf)
	# cast keys to integers - json stores dict keys always as string - and keep first CUI as candidate CUI
	token2cui = {ix.id2token[int(k)]: v[0] for k, v in token2cuis.items()}
	del token2cuis  # free memory space
	print('load encoded corpus')
	with open(data_folder + '/enc_corpus.json', 'r') as corpus_file:
		docs = json.load(corpus_file)
	print('convert corpus to words-only')
	docs = [[ix.id2token[ix2dict[word]] for word, _ in doc] for doc in tqdm(docs)]
	print('converted corpus becomes ... "{}" ...'.format(' '.join(docs[0][:10])))
	print('load doc ids')
	with open(data_folder + '/doc_ids.json', 'r') as docids_file:
		doc_ids = json.load(docids_file)

	# print statistics
	print("corpus size: {}".format(ix.num_docs))
	print("corpus unique tokens: {}".format(len(ix.values())))

	# create term dictionary 
	term_dict = Dictionary(docs)
	# sanity check on corpus vocab - term_dict must be equal to safir dict
	assert set(term_dict.token2id.keys()) == set([ix.id2token[term] for _, term in ix2dict.items()])
	assert ix.num_docs == term_dict.num_docs

	if opts.weighting:
		# compute IDF scores for words within corpus
		weights = utils.compute_idfs(term_dict)
	else: 
		weights = None

	"""
	LOADING MODEL
	"""	

	# load semantic model
	print('load word2vec model')
	word2vec = gensim.models.Word2Vec.load(opts.word2vec_path)
		
	"""
	RETROFITTING
	"""

	# get synonyms for each word within vocabulary
	print('get synonyms for each word within gensim vocabulary')
	syns = utils.get_syns(token2cui, word2vec.wv.vocab)
	# retrofit word vectors 
	print('retrofit word vectors for {} iterations'.format(opts.iterations))
	word_embs = retrofit(word2vec, syns, opts.iterations, alpha=1.0)
	# convert retrofitted word vectors in gensim word2vec format
	rword2vec = gensim.models.word2vec.Word2VecKeyedVectors(vector_size=word2vec.vector_size)
	# store retrofitted word vectors within rword2vec
	rword2vec.add(entities=list(word_embs.keys()), weights=list(word_embs.values()))
	# store rword2vec model
	print('store rWord2Vec model')
	rword2vec.save(model_folder + '/' + opts.model_name + '_' + opts.ref_measure + '.model')

	# compute doc embeddings
	print('compute document vectors w/ retrofitted word vectors')
	doc_ids_embs = utils.compute_doc_embs(dict(zip(doc_ids, docs)), rword2vec, weights)
	# get doc embs and ids from doc_ids_embs
	doc_embs = np.array(list(doc_ids_embs.values()))
	doc_ids = np.array(list(doc_ids_embs.keys()))

	"""
	TRAINING AND EVALUATION
	"""

	q_ids = list()
	q_embs = list()
	# loop over queries and generate query embs
	for qid, qtext in qdict.items():
		# compute query emb as the sum of its word embs
		q_emb = utils.query2emb(qtext[opts.qfield], rword2vec)
		if q_emb is None:
			print('query {} does not contain known terms'.format(qid))
		else:
			q_embs.append(q_emb)
			q_ids.append(qid)
	# convert q_embs to numpy
	q_embs = np.array(q_embs)

	# perform semantic search with trained embs
	utils.semantic_search(doc_ids, doc_embs, q_ids, q_embs, rankings_folder, opts.model_name + '_' + opts.ref_measure)
	
	if "OHSUMED" in opts.corpus_name:  # evaluate OHSUMED collection
		# evaluate ranking list
		scores = utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder, opts.model_name + '_' + opts.ref_measure, qrels_folder, opts.qrels_fname)
	elif "TREC_CDS" in opts.corpus_name:  # evaluate TREC CDS collections
		print('evaluate model for CDS 2016')
		scores = utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder, opts.model_name + '_' + opts.ref_measure, qrels_folder, opts.qrels_fname)
		inf_scores = utils.evaluate_inferred(['infNDCG'], rankings_folder, opts.model_name + '_' + opts.ref_measure, qrels_folder, opts.infqrels_fname)


if __name__ == "__main__":
	main() 
	