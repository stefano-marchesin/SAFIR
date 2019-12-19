import os
import glob
import numpy as np
import tensorflow as tf

import safir_utils

from safir_index import Index
from safir import SAFIR

flags = tf.app.flags

flags.DEFINE_integer("ngram_size", 16,
					 "The number of words to predict to the left and right "
					 "of the target word.")
flags.DEFINE_bool("include_poly", False,
				  "Whether to consider the polysemous nature of (index) terms.")
flags.DEFINE_integer("seed", 42,
					 "Answer to ultimate question of life, the universe and everything.")
flags.DEFINE_string("query_field", "summary", "Query target field.")
flags.DEFINE_string("corpus_name", "TREC_CDS14_15", "Target corpus name.")
flags.DEFINE_string("query_fname", "topics2015A.xml", "Query file name.")
flags.DEFINE_string("qrels_fname", "qrels-treceval-2015", "Qrels file name.")
flags.DEFINE_string("inf_qrels_fname", "qrels-sampleval-2015", "Inferred qrels file name.")
flags.DEFINE_string("reference_measure", "infNDCG", "Reference measure to be used for model optimization.")
flags.DEFINE_string("stored_model", "", "Model stored after training.")
flags.DEFINE_string("model_name", "", "Model name.")
FLAGS = flags.FLAGS


class Options(object):
	"""options used by SAFIR"""

	def __init__(self):
		self.ngram_size = FLAGS.ngram_size
		# term polysemy
		self.poly = FLAGS.include_poly
		# seed
		self.seed = FLAGS.seed
		# query field
		self.field = FLAGS.query_field
		# corpus name
		self.corpus_name = FLAGS.corpus_name
		# query file name
		self.query_fname = FLAGS.query_fname
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname
		# inferred qrels file name
		self.inf_qrels_fname = FLAGS.inf_qrels_fname
		# reference measure
		self.ref_measure = FLAGS.reference_measure
		# stored model
		self.stored_model = FLAGS.stored_model
		# model name
		self.model_name = FLAGS.model_name


def main(_):
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set folders
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	model_folder = 'corpus/' + opts.corpus_name + '/models'
	data_folder = 'corpus/' + opts.corpus_name + '/data'
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings'
	# set stoplist path
	stopwords_fname = './indri_stopwords.txt'

	# create folders 
	if not os.path.exists(rankings_folder + '/' + opts.model_name):
		os.makedirs(rankings_folder + '/' + opts.model_name)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# set random seed - enable reproducibility
	np.random.seed(opts.seed)
	# load stop tokens
		#stops = safir_utils.load_stopwords(swords_fname) | set(string.punctuation)
	stops = safir_utils.load_stopwords(stopwords_fname)
	# set Index instance 
	index = Index(stops=stops)

	# load processed data
	print("load index data")
	index.load_index(index_folder + '/' + opts.corpus_name + '_ix.txt')
	index.load_token2cuis(data_folder + '/token2cuis.json')
	print("load dictionaries")
	index.load_term_dict(data_folder + '/term_dict.json')
	index.load_concept_dict(data_folder + '/concept_dict.json')
	print("load doc ids")
	doc_ids = index.load_doc_ids(data_folder + '/doc_ids.json')

	# print statistics
	print("corpus size: {}".format(index.ix.num_docs))
	print("corpus unique tokens: {}".format(len(index.ix.values())))
	print("term dictionary size: {}".format(len(index.term_dict)))
	print("concept dictionary size: {}".format(len(index.concept_dict)))

	# load queries
	print("load {} queries".format(opts.corpus_name))
	if "OHSUMED" in opts.corpus_name:  # load ohsumed queries
		qdict = safir_utils.read_ohsu_queries(query_folder + '/' + opts.query_fname)  
	elif "TREC_CDS" in opts.corpus_name:  # load cds queries
		qdict = safir_utils.read_cds_queries(query_folder + '/' + opts.query_fname)  
	elif "TREC_PM" in opts.corpus_name:  # load pm queries
		qdict = safir_utils.read_pm_queries(query_folder + '/' + opts.query_fname)

	measure_per_epoch = []
	# loop over stored models and test them over given set of queries
	meta_graphs = glob.glob(model_folder + '/' + opts.stored_model + '/*.ckpt.meta')
	for metha_graph in meta_graphs:
		# get model checkpoint path from meta graph path
		model_ckpt = '.'.join(metha_graph.split('.')[:2])
		# get model epoch
		epoch = int(model_ckpt.split('.')[0].split('AA')[-1][:])
		print("testing model at epoch: {}".format(epoch))
		# start test graph
		with tf.Session() as sess: 
			print("restoring model...")
			# restore model and get required tensors
			saver = tf.train.import_meta_graph(metha_graph)
			saver.restore(sess, model_ckpt)
			word_embs = sess.run(tf.get_default_graph().get_tensor_by_name('word_embs:0'))
			if opts.poly:
				concept_embs = sess.run(tf.get_default_graph().get_tensor_by_name('concept_embs:0'))
			proj_weights = sess.run(tf.get_default_graph().get_tensor_by_name('proj_weights:0'))
			doc_embs = sess.run(tf.get_default_graph().get_tensor_by_name('doc_embs:0'))
			print("model restored!")
		# evaluate model effectiveness
		print("ranking w/ model obtained at epoch {}".format(epoch))
		queries = []
		query_ids = []
		# loop over queries and generate rankings
		for qid, qtext in qdict.items():  # TODO: make field choice as a flag param
			# prepare query for document matching (i.e., project query into doc space)
			if opts.poly:
				proj_query = index.project_query(qtext[opts.field], opts.corpus_name, word_embs, proj_weights, concept_embs)
			else:
				proj_query = index.project_query(qtext[opts.field], opts.corpus_name, word_embs, proj_weights)
			# append projected query only if it contains known terms
			if proj_query is not None: 
				queries.append(proj_query)
				query_ids.append(qid)
		# convert queries to np array
		queries = np.array(queries)
		# perform search and evaluate model effectiveness
		index.semantic_search(doc_ids, doc_embs, query_ids, queries, rankings_folder + '/' + opts.model_name, opts.model_name + '_' + str(epoch + 1))
		if "OHSUMED" in opts.corpus_name:  # evaluate OHSUMED collection
			scores = safir_utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder + '/' + opts.model_name, opts.model_name + '_' + str(epoch + 1), qrels_folder, opts.qrels_fname)
			# append reference measure for current epoch
			measure_per_epoch.append(scores[opts.ref_measure])
		elif "TREC_CDS" in opts.corpus_name:  # evaluate TREC CDS collections
			scores =  safir_utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], rankings_folder + '/' + opts.model_name, opts.model_name + '_' + str(epoch + 1), qrels_folder, opts.qrels_fname)  
			inf_scores = safir_utils.evaluate_inferred(['infNDCG'], rankings_folder + '/' + opts.model_name, opts.model_name + '_' + str(epoch + 1), qrels_folder, opts.inf_qrels_fname) 
			if 'inf' in opts.ref_measure:  # optimize model wrt chosen inferred measure
				measure_per_epoch.append(inf_scores[opts.ref_measure])
			else:  # optimize model wrt chosen measure
				measure_per_epoch.append(scores[opts.ref_measure])
	# print best epoch in terms of map
	print("best model (in terms of {}) found at epoch: {}".format(opts.ref_measure, np.argsort(measure_per_epoch)[-1] + 1))


if __name__ == "__main__":
	tf.app.run()
