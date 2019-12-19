import os
import argparse
import gensim
import json
import multiprocessing
import numpy as np

from tqdm import tqdm 
from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec

from gensim_utils import Utils, EpochRanker


flags = argparse.ArgumentParser()

flags.add_argument("--embs_size", default=256, type=int, help="The embedding dimension size.")
flags.add_argument("--epochs", default=15, type=int,
				   help="Number of epochs to train. Each epoch processes the training data once completely.")
flags.add_argument("--train_on_concepts", default=False, type=bool, 
				   help="Whether to train Word2Vec or ConceptualWord2Vec.")
flags.add_argument("--negative_samples", default=10, type=int, help="Negative samples per training example.")
flags.add_argument("--learning_rate", default=0.025, type=float, help="Learning rate.")
flags.add_argument("--linear_decay", default=0.0001, type=float, help="Minimum learning rate after linear decay.")
flags.add_argument("--window", default=8, type=int, help="The number of words to predict to the left/right of the target word.")
flags.add_argument("--min_count", default=1, type=int, help="Ignores all words with total frequency lower than this.")
flags.add_argument("--model_type", default=1, type=int, help="CBOW: 0; skip-gram: 1.")
flags.add_argument("--weighting_scheme", default=True, type=bool,
				   help="Whether to consider IDF weighting scheme to compose document embeddings.")
flags.add_argument("--minsize", default=1, type=int, help="Minimum word's size allowed.")
flags.add_argument("--seed", default=0, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--corpus", default="TREC_CDS16", type=str, help="Corpus to use.")
flags.add_argument("--query_fname", default="topics2016.xml", type=str, help="Queries file name.")
flags.add_argument("--qrels_fname", default="qrels-treceval-2016", type=str, help="Qrels file name.")
flags.add_argument("--infqrels_fname", default="qrels-sampleval-2016", type=str, help="Inferred qrels file name.")
flags.add_argument("--query_field", default="summary", type=str, help="Query field to consider when performing retrieval.")
flags.add_argument("--reference_measure", default="P_10", type=str, help="Reference measure to optimize.")
flags.add_argument("--model_name", default="", type=str, help="Model name.")
FLAGS = flags.parse_args()


class Options(object):
	"""options used by word2vec models"""

	def __init__(self):
		# embeddings dimension
		self.embs_size = FLAGS.embs_size
		# epochs to train
		self.epochs = FLAGS.epochs
		# train on concepts
		self.train_on_concepts = FLAGS.train_on_concepts
		# number of negative samples per example
		self.neg_samples = FLAGS.negative_samples
		# learning rate
		self.learn_rate = FLAGS.learning_rate
		# linear decay
		self.lin_decay = FLAGS.linear_decay
		# window size
		self.window = FLAGS.window
		# min count
		self.min_count = FLAGS.min_count
		# model
		self.model_type = FLAGS.model_type
		# weighting scheme
		self.weighting = FLAGS.weighting_scheme
		# minimum word size
		self.minsize = FLAGS.minsize
		# seed
		self.seed = FLAGS.seed
		# corpus name
		self.corpus_name = FLAGS.corpus
		# query file name
		self.qfname = FLAGS.query_fname
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname 
		# inferred qrels file name
		self.infqrels_fname = FLAGS.infqrels_fname
		# query field
		self.qfield = FLAGS.query_field
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# model name
		self.model_name = FLAGS.model_name


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
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
	if not os.path.exists(index_folder):
		os.makedirs(index_folder)
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# load utils functions - set random seed
	utils = Utils(opts.seed)
	# compute available number of CPUs
	cpu_count = multiprocessing.cpu_count()

	# load queries
	print('load {} queries'.format(opts.corpus_name))
	if 'OHSUMED' in opts.corpus_name:  # load ohsu queries
		qdict = utils.read_ohsu_queries(query_folder + '/' + opts.qfname)
	elif 'TREC_CDS' in opts.corpus_name:  # load cds queries
		qdict = utils.read_cds_queries(query_folder + '/' + opts.qfname)
	elif 'TREC_PM' in opts.corpus_name:  # load pm queries
		qdict = utils.read_pm_queries(query_folder + '/' + opts.qfname)

	"""MODEL PRE-PROCESSING"""

	# load required data
	print('load index')
	ix = Dictionary.load_from_text(index_folder + '/' + opts.corpus_name + '_ix.txt')
	ix.id2token = {v: k for k, v in ix.token2id.items()}

	if opts.train_on_concepts:
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
	else: 
		print('load safir term dict')
		with open(data_folder + '/term_dict.json', 'r') as std:
			ix2dict = json.load(std)

	print('load encoded corpus')
	with open(data_folder + '/enc_corpus.json', 'r') as corpus_file:
		docs = json.load(corpus_file)

	if opts.train_on_concepts:
		print('disambiguate queries and return them as lists of (word, cui) pairs')
		for qid, qbody in qdict.items():
			enc_query = utils.disambiguate_query(ix, ix2ix, ix2dict, token2cuis, qbody[opts.qfield], opts.corpus_name)
			qdict[qid][opts.qfield] = enc_query
		# remove '__NULL__' concept from safir concept dict
		ix2dict.pop('__NULL__', None)
		# reverse safir concept dict
		ix2dict = {v: k for k, v in ix2dict.items()}
		print('convert queries to concepts-only')
		for qid, qbody in qdict.items():
			cuis_query = ' '.join([ix2dict[cui] for _, cui in qdict[qid][opts.qfield] if cui in ix2dict])
			qdict[qid][opts.qfield] = cuis_query
		print('converted queries become ... "{}" ...'.format(list(qdict.values())[0][opts.qfield]))
		print('convert corpus to concepts-only')
		docs = [[ix2dict[cui] for _, cui in doc if cui in ix2dict] for doc in tqdm(docs)]
		print('converted corpus becomes ... "{}" ...'.format(' '.join(docs[0][:10])))
	else: 
		# reverse safir term dict
		ix2dict = {v: int(k) for k, v in ix2dict.items()}
		print('convert corpus to words-only')
		docs = [[ix.id2token[ix2dict[word]] for word, _ in doc] for doc in tqdm(docs)]
		print('converted corpus becomes ... "{}" ...'.format(' '.join(docs[0][:10])))
	print('load doc ids')
	with open(data_folder + '/doc_ids.json', 'r') as docids_file:
		docnos = json.load(docids_file)

	# print statistics
	print("corpus size: {}".format(ix.num_docs))
	print("corpus unique tokens: {}".format(len(ix.values())))

	# create term dictionary 
	term_dict = Dictionary(docs)
	if opts.train_on_concepts:
		# sanity check on corpus vocab - term_dict must be equal to safir concept dict
		assert set(term_dict.token2id.keys()) == set([cui for _, cui in ix2dict.items()])
	else:
		# sanity check on corpus vocab - term_dict must be equal to safir term dict
		assert set(term_dict.token2id.keys()) == set([ix.id2token[term] for _, term in ix2dict.items()])
	assert ix.num_docs == term_dict.num_docs

	if opts.weighting:
		# compute IDF scores for words within corpus
		weights = utils.compute_idfs(term_dict)
	else: 
		weights = None

	"""TRAINING AND EVALUATION"""

	# initialize Word2Vec model
	print('initialize model')
	model = Word2Vec(iter=opts.epochs, size=opts.embs_size, window=opts.window, alpha=opts.learn_rate, min_alpha=opts.lin_decay, min_count=opts.min_count, 
					sg=opts.model_type, negative=opts.neg_samples, seed=opts.seed, sample=None, workers=cpu_count)
	# build Word2Vec vocabulary
	print('build model vocabulary')
	model.build_vocab(docs)

	# sanity check on model vocab - model.wv.vocab must be equal to term_dict
	assert set(model.wv.vocab.keys()) == set(term_dict.token2id.keys())

	# display model vocab statistics
	print('number of unique tokens: {}'.format(len(model.wv.vocab)))

	# initialize callback class to perform ranking after each training epoch
	epoch_ranker = EpochRanker(dict(zip(docnos, docs)), qdict, weights, utils, False, opts, model_folder, rankings_folder, qrels_folder)

	# train and evaluate model in terms of retrieval effectiveness
	print('train the model for {} epochs and evaluate it in terms of IR effectiveness'.format(opts.epochs))
	model.train(docs, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[epoch_ranker])
	print('model training finished!')

	# get best model in terms of reference measure
	best_epoch = epoch_ranker.best_epoch
	best_score = epoch_ranker.best_score
	print('best model found at epoch {} with {}: {}'.format(best_epoch, opts.ref_measure, best_score))


if __name__ == "__main__":
	main()