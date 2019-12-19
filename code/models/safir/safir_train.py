import os
import numpy as np
import tensorflow as tf

import safir_utils

from safir_index import Index
from safir import SAFIR

flags = tf.app.flags

flags.DEFINE_integer("word_embs_size", 300, "The word embedding dimension size.")
flags.DEFINE_integer("concept_embs_size", 300, "The concept embedding dimension size.")
flags.DEFINE_integer("doc_embs_size", 256, "The document embedding dimension size.")
flags.DEFINE_integer("epochs", 15,
	"Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_integer("summary_steps", 500, 
	"Number of steps before computing TensorBoard summaries.")
flags.DEFINE_integer("negative_samples", 10,
					 "Negative samples per training example.")
flags.DEFINE_integer("num_true", 1,
					 "Number of true labels per training example.")
flags.DEFINE_float("regularization_term", 1e-2, "Regularization parameter.")
flags.DEFINE_float("semantic_term", 1.0,
				   "Weighting parameter for semantic matching objective.")
flags.DEFINE_integer("batch_size", 51200,
					 "Number of training examples processed per step "
					 "(size of a minibatch).")
flags.DEFINE_integer("dict_size", 131072, 
					 "Size of term dictionary.")
flags.DEFINE_integer("ngram_size", 16,
					 "The number of words to predict to the left and right "
					 "of the target word.")
flags.DEFINE_bool("l2_norm_ngrams", True,
				  "Whether to l2 normalize ngram representations.")
flags.DEFINE_bool("include_poly", True,
				  "Whether to consider the polysemous nature of (index) terms.")
flags.DEFINE_bool("default_semtypes", True,
				  "Whether to consider default QuickUMLS semtypes or custom ones.")
flags.DEFINE_float("threshold", 0.7, 
				   "Minimum similarity value between strings for QuickUMLS." )
flags.DEFINE_integer("seed", 42,
					 "Answer to ultimate question of life, the universe and everything.")
flags.DEFINE_string("query_field", "summary", "Query target field.")
flags.DEFINE_string("corpus_name", "TREC_CDS14_15", "Target corpus name.")
flags.DEFINE_string("query_fname", "topics2014.xml", "Query file name.")
flags.DEFINE_string("qrels_fname", "qrels-treceval-2014", "Qrels file name.")
flags.DEFINE_string("inf_qrels_fname", "qrels-sampleval-2014", "Inferred qrels file name.")
flags.DEFINE_string("reference_measure", "infNDCG", "Reference measure to be used for model optimization.")
flags.DEFINE_string("model_name", "", "Model name.")
FLAGS = flags.FLAGS


class Options(object):
	"""options used by SAFIR"""

	def __init__(self):
		# word embeddings dimension
		self.word_size = FLAGS.word_embs_size
		# concept embeddings dimension
		self.concept_size = FLAGS.concept_embs_size
		# document embeddings dimension
		self.doc_size = FLAGS.doc_embs_size
		# number of negative samples per example
		self.neg_samples = FLAGS.negative_samples
		# number of true labels per example
		self.num_true = FLAGS.num_true
		# regularization term
		self.reg_term = FLAGS.regularization_term
		# semantic term
		self.sem_term = FLAGS.semantic_term
		# epochs to train
		self.epochs = FLAGS.epochs
		# number of steps before computing summaries
		self.summary_steps = FLAGS.summary_steps
		# batch size
		self.batch_size = FLAGS.batch_size
		# dict size
		self.dict_size = FLAGS.dict_size
		# ngram size
		self.ngram_size = FLAGS.ngram_size
		# l2 normalization for ngrams
		self.l2_norm = FLAGS.l2_norm_ngrams
		# term polysemy
		self.poly = FLAGS.include_poly
		# default semtypes
		self.default_semtypes = FLAGS.default_semtypes
		# quickumls threshold
		self.threshold = FLAGS.threshold
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
		# model name
		self.model_name = FLAGS.model_name


def main(_):
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# load options
	opts = Options()
	# set folders
	corpus_folder = 'corpus/' + opts.corpus_name + '/' + opts.corpus_name
	logs_folder = 'corpus/' + opts.corpus_name + '/logs/' + opts.model_name
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	model_folder = 'corpus/' + opts.corpus_name + '/models'
	data_folder = 'corpus/' + opts.corpus_name + '/data'
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings'
	# set file paths
	if opts.default_semtypes:  # use default QuickUMLS semantic types
		stypes_fname = None
	else:  # use user-specified semantic types
		stypes_fname = './semantic_types.txt'
	stopwords_fname = './indri_stopwords.txt'

	# create folders 
	if not os.path.exists(logs_folder):
		os.makedirs(logs_folder)
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
	if not os.path.exists(index_folder):
		os.makedirs(index_folder)
	if not os.path.exists(rankings_folder + '/' + opts.model_name):
		os.makedirs(rankings_folder + '/' + opts.model_name)
	if not os.path.exists(model_folder + '/' + opts.model_name):
		os.makedirs(model_folder + '/' + opts.model_name)
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

	if not os.path.exists(data_folder + '/enc_corpus.json'):  # process data
		if not os.path.exists(data_folder + '/pproc_corpus.json'):  # pre process data
			# pre process corpus: tokenize, remove stopwords, ecc.
			print("pre process corpus")
			pproc_corpus, doc_ids = index.preprocess_corpus(corpus_folder, opts.corpus_name, data_folder + '/pproc_corpus.json', data_folder + '/doc_ids.json')
		else:  # load pre processed corpus and doc ids
			print("load pre processed corpus")
			pproc_corpus = index.load_pproc_corpus(data_folder + '/pproc_corpus.json')
			print("load doc ids")
			doc_ids = index.load_doc_ids(data_folder + '/doc_ids.json')
		# index corpus 
		print("index corpus")
		index.index_corpus(pproc_corpus, index_folder + '/' + opts.corpus_name + '_ix.txt')
		# create term dictionary 
		print("build term dictionary")
		index.build_term_dict(pproc_corpus, data_folder + '/term_dict.json', dict_size=opts.dict_size)
		# index semantic data and encode data
		print("index semantic data and encode corpus")
		enc_corpus = index.encode_corpus(pproc_corpus, opts.corpus_name, data_folder + '/enc_corpus.json', 
																		 data_folder + '/token2cuis.json', 
																		 data_folder + '/concept_dict.json', 
																		 data_folder + '/synsets.json',
										 threshold=opts.threshold, stypes_fname=stypes_fname)
		# delete pproc_corpus to free memory space
		del pproc_corpus
	else:  # load processed data
		print("load index data")
		index.load_index(index_folder + '/' + opts.corpus_name + '_ix.txt')
		index.load_token2cuis(data_folder + '/token2cuis.json')
		print("load dictionaries")
		index.load_term_dict(data_folder + '/term_dict.json')
		index.load_concept_dict(data_folder + '/concept_dict.json')
		index.load_synsets(data_folder + '/synsets.json')
		print("load encoded corpus")
		enc_corpus = index.load_enc_corpus(data_folder + '/enc_corpus.json')
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
	# start session
	with tf.Graph().as_default(), tf.Session() as sess:
		# get information required for training
		print("get synsets from index")
		synsets = index.get_sense_pairs()
		print("get docs with length >= ngram size")
		allowed_docs = safir_utils.get_allowed_docs(enc_corpus, opts.ngram_size)
		print("compute number of batches per epoch")
		num_batches = safir_utils.compute_num_batches(enc_corpus, opts.batch_size, opts.ngram_size)
		print("number of batches per epoch: {}".format(num_batches))

		# add checkpoints to training 
		store_summaries_every = opts.summary_steps
		save_embeddings_every = num_batches  # one checkpoint per epoch
		print_loss_every = num_batches  # one checkpoint per epoch

		# setup the model
		if opts.poly:
			model = SAFIR(len(index.term_dict), len(enc_corpus), synsets, opts, len(index.concept_dict))
		else:
			model = SAFIR(len(index.term_dict), len(enc_corpus), synsets, opts)

		# create model saving operation - keeps as many saved models as number of epochs
		saver = tf.train.Saver(max_to_keep=opts.epochs)
		# merge all summaries and write them out to logs_folder 
		merged = tf.summary.merge_all()
		# initialize TensorFlow writer
		tb_writer = tf.summary.FileWriter(logs_folder, sess.graph)
		# initialize the variables using global_variables_initializer()
		sess.run(tf.global_variables_initializer())

		measure_per_epoch = []
		print("start training")
		for epoch in range(opts.epochs):
			loss = []
			loss_at_step = []

			print("training epoch {}".format(epoch + 1))
			for i in range(num_batches):
				# generate batch data and feed to feed_dict
				batch_data = safir_utils.generate_batch_data(enc_corpus, allowed_docs, opts.batch_size, opts.ngram_size, opts.neg_samples)
				feed_dict = {model.ngram_words: batch_data[0][:, :, 0],
							 model.labels: batch_data[1],
							 model.negative_labels: batch_data[2]}
				if opts.poly:
					feed_dict[model.ngram_concepts] = batch_data[0][:, :, 1]
				# run train_op
				sess.run(model.train_op, feed_dict=feed_dict)

				# store variable summaries for TensorBoard visualization
				if (i + 1) % store_summaries_every == 0:
					summary = sess.run(merged, feed_dict=feed_dict)
					tb_writer.add_summary(summary, i + 1)

				# compute loss values for each sub-loss function + total
				if (i + 1) % print_loss_every == 0:
					if opts.sem_term > 0.0:
						summary, loss_value, text_loss, sem_loss, reg_loss = sess.run([merged, model.loss, model.text_loss, model.sem_loss, model.reg_loss], feed_dict=feed_dict)
						print("total loss: {}, text loss: {}, semantic loss: {}, regularization loss: {}".format(loss_value, text_loss, sem_loss, reg_loss))
					else: 
						summary, loss_value, text_loss, reg_loss = sess.run([merged, model.loss, model.text_loss, model.reg_loss], feed_dict=feed_dict)
						print("total loss: {}, text loss: {}, regularization loss: {}".format(loss_value, text_loss, reg_loss))
					tb_writer.add_summary(summary, i + 1)
					loss.append(loss_value)
					loss_at_step.append(i + 1)
					print("loss at step {}: {}".format(i + 1, loss_value))

				# save embeddings and extract them for validation
				if (i + 1) % save_embeddings_every == 0:
					model_checkpoint_path = os.path.join(os.getcwd(), model_folder + '/' + opts.model_name, opts.model_name + str(epoch + 1) + '.ckpt')
					save_path = saver.save(sess, model_checkpoint_path)
					print("model saved in file: {}".format(save_path))
					if opts.poly:
						word_embs, concept_embs, proj_weights, doc_embs = sess.run([model.word_embs, model.concept_embs, model.proj_weights, model.doc_embs])
					else:
						word_embs, proj_weights, doc_embs = sess.run([model.word_embs, model.proj_weights, model.doc_embs])

			# evaluate model effectiveness after each epoch
			print("ranking at epoch {}".format(epoch + 1))
			queries = []
			query_ids = []
			# loop over queries and generate rankings
			for qid, qtext in qdict.items(): 
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
