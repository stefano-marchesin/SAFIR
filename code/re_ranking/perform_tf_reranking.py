import os
import argparse
import subprocess
import collections 
import tempfile
import sklearn.model_selection
import pytrec_eval
import json
import numpy as np 
import tensorflow as tf

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from cvangysel import trec_utils

import safir_utils

from safir_index import Index	


flags = argparse.ArgumentParser()

flags.add_argument("--lexical_run", default="", type=str, help="Lexical baseline run.")
flags.add_argument("--semantic_model", default="", type=str, help="Semantic model path.")
flags.add_argument("--include_concepts", default=True, type=bool, help="Whether to consider concept embeddings or not.")
flags.add_argument("--qrels_fname", default="qrels-treceval-2016", type=str, help="Qrels filename.")
flags.add_argument("--inf_qrels_fname", default="qrels-sampleval-2016", type=str, help="Inferred qrels filename.")
flags.add_argument("--query_fname", default="topics2016.xml", type=str, help="Query filename.")
flags.add_argument("--query_field", default="summary", type=str, help="Query field to consider for retrieval.")
flags.add_argument("--ref_measure", default="infNDCG", type=str, help="Reference measure to consider for optimization.")
flags.add_argument("--gamma", default=0, type=float, help="Perform re-ranking using a fixed gamma instead of a 2-fold cross validation optimized one. Set gamma to 0 to perform k-fold cross validation.")
flags.add_argument("--num_folds", default=2, type=int, help="Number of folds to consider for cross validation.")
flags.add_argument("--sweep", default=0.05, type=float, help="Combine models sweeping the weight.")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--corpus_name", default="TREC_CDS16", type=str, help="Corpus to consider.")
flags.add_argument("--run_base_name", default="", type=str, help="Re-ranked (base) run name.")
flags.add_argument("--normalizer", default="minmax", type=str, help="Selected normalizer - possible normalizers: 'standardize', 'minmax', 'none'.")

FLAGS = flags.parse_args()


def parse_inf_qrels(inf_qrels_file):
	"""parse inferred qrels"""
	inf_qrels = collections.defaultdict(dict)
	for line in inf_qrels_file:
		query_id, _, object_id, relevance, inf_relevance = line.strip().split()
		assert object_id not in inf_qrels[query_id]
		inf_qrels[query_id][object_id] = (int(relevance), int(inf_relevance))
	return inf_qrels


def evaluate_ref_measure(ranking, qrels_file, evaluator):
	"""evaluate (online) ranking for given reference measure"""
	qrels_folder = '/'.join(qrels_file.name.split('/')[:-1])
	qrels_name = qrels_file.name.split('/')[-1].split('.')[0]
	# open a temporary file to store ranking
	rank_file = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
	# write ranking into temporary file
	rank_file.write('\n'.join([qid + ' ' + 'Q0' + ' ' + docno + ' ' + str(ix) + ' ' + str(score) + ' ' + 'tmp' 
					for qid, rank in ranking.items() for ix, (docno, score) in enumerate(rank.items())]))
	rank_folder = '/'.join(rank_file.name.split('/')[:-1])
	rank_name = rank_file.name.split('/')[-1].split('.')[0]
	# compute aggregated measure score for (inferred) reference measure
	agg_measure_score = evaluator.evaluate_inferred(FLAGS.ref_measure, rank_folder, rank_name, qrels_folder, qrels_name)[FLAGS.ref_measure]
	# close temporary file (delete it)
	rank_file.close()
	return agg_measure_score


def compute_combined_run(run, qfield, query_ids, query_embs, docnos, doc_embs, norm, weight):
	"""compute combined rank between lexical run and semantic model"""
	combined_run = {}
	# convert docnos into dicts {docno: idx} and {idx: docno}
	docno2idx = {docno: idx for idx, docno in enumerate(docnos)}
	idx2docno = {idx: docno for idx, docno in enumerate(docnos)}
	# loop over qids
	print('combine lexical and semantic models w/ weight: {}'.format(weight))
	for qid, qemb in zip(query_ids, query_embs):
		# query ranking
		qrankings = collections.defaultdict(list)
		# check whether run has ranking for current query
		if run[qid]:
			# get run ranking (lexical baseline)
			lex_ranking = run[qid] 
			# perform semantic matching for non-empty queries
			if qemb is not None:
				# get doc indexes from baseline (lexical) ranking
				doc_idxs = [docno2idx[docno] for docno in lex_ranking.keys() if docno in docno2idx]
				# get hashmap from doc indexes
				idx2pos = {idx: pos for pos, idx in enumerate(doc_idxs)}
				# compute cosine similarity between query and doc embeddings
				cosine_scores = cosine_similarity(doc_embs[doc_idxs], qemb.reshape(1, -1))
				# convert cosine_scores to dict {idx: score}
				sem_ranking = dict(enumerate(cosine_scores.flatten()))
				# compute ranking normalization
				lex_norm = norm(list(lex_ranking.values()))
				sem_norm = norm(list(sem_ranking.values()))
				# iterate over docs within run ranking 
				for docno, score in lex_ranking.items():
					# append weighted (normalized) doc scores to qrankings
					qrankings[docno].append(weight * lex_norm(score))  # lexical score
					if docno in docno2idx:
						qrankings[docno].append((1-weight) * sem_norm(sem_ranking[idx2pos[docno2idx[docno]]]))  # semantic score
		# compute combined ranking for given query
		combined_run[qid] = {docno: np.sum(scores) for docno, scores in qrankings.items()}
	return combined_run


def perform_reranking(run, qfield, query_ids, query_embs, docnos, doc_embs, evaluator, inf_qrels=None):
	"""perform re-ranking of input run w/ semantic model"""
	print("re-ranking models")
	# loop over weight values with sweep equal to sweep
	for weight in np.arange(FLAGS.sweep, 1.0, FLAGS.sweep):
		weight = round(weight, 4)  # round weight up to 4 decimals
		# generate combined run with current weight
		combined_run = compute_combined_run(run, qfield, query_ids, query_embs, docnos, doc_embs, SCORE_NORMALIZERS[FLAGS.normalizer], weight)
		# evaluate combined run
		if 'inf' in FLAGS.ref_measure:  # evaluate for inferred reference measure
			agg_measure_score = evaluate_ref_measure(combined_run, inf_qrels, evaluator)
		else:  # evaluate for standard reference measure
			results = evaluator.evaluate(combined_run)
			# compute aggregated measure score
			agg_measure_score = pytrec_eval.compute_aggregated_measure(FLAGS.ref_measure, [qmeasures[FLAGS.ref_measure] for qmeasures in results.values()])
		# return aggregated mesure score and weight
		yield agg_measure_score, weight


class StandardizationNormalizer(object):
	# apply standard deviation normalization
	def __init__(self, scores):
		self.mean = np.mean(scores)
		self.std = np.std(scores)

	def __call__(self, score):
		return (score - self.mean) / self.std


class MinMaxNormalizer(object):
	# apply minmax normalization
	def __init__(self, scores):
		self.min = np.min(scores)
		self.max = np.max(scores)

	def __call__(self, score):
		return (score - self.min) / (self.max - self.min)


class IdentityNormalizer(object):
	# apply identify normalization
	def __init__(self, scores):
		pass

	def __call__(self, score):
		return score


SCORE_NORMALIZERS = {
	'standardize': StandardizationNormalizer,
	'minmax': MinMaxNormalizer,
	'none': IdentityNormalizer
}


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# set folders
	index_folder = 'corpus/' + FLAGS.corpus_name + '/index'
	data_folder = 'corpus/' + FLAGS.corpus_name + '/data'
	query_folder = 'corpus/' + FLAGS.corpus_name + '/queries'
	qrels_folder = 'corpus/' + FLAGS.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + FLAGS.corpus_name + '/rankings/' + FLAGS.run_base_name

	# set file paths
	stopwords_fname = './indri_stopwords.txt'

	# create folders 
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	if FLAGS.sweep > 0:
		if 'inf' in FLAGS.ref_measure:  
			# parse and store inferred qrels
			with open(qrels_folder + '/' + FLAGS.inf_qrels_fname + '.txt', 'r') as qrelf:
				qrels = parse_inf_qrels(qrelf)
		else:  
			# parse and store qrels
			with open(qrels_folder + '/' + FLAGS.qrels_fname + '.txt', 'r') as qrelf:
				qrels = pytrec_eval.parse_qrel(qrelf)
			# set reference category
			if 'cut' in FLAGS.ref_measure:
				ref_category = '_'.join(FLAGS.ref_measure.split('_')[:2])
			else:
				ref_category = FLAGS.ref_measure.split('_')[0]
			# initialize evaluator over qrels
			evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ref_category})  # evaluate on reference measure

	# set random seed - enable repro
	np.random.seed(FLAGS.seed)
	# load stopwords
	stops = safir_utils.load_stopwords(stopwords_fname)

	# parse lexical run
	print('parse lexical run')
	with open(FLAGS.lexical_run, 'r') as lex_run:
		run = pytrec_eval.parse_run(lex_run)

	# load data required for semantic models
	print('load processed data required to perform re-ranking over lexical model w/ semantic model')
	# set index instance
	index = Index(stops=stops)
	# load data
	index.load_index(index_folder + '/' + FLAGS.corpus_name + '_ix.txt')
	index.load_token2cuis(data_folder + '/token2cuis.json')
	index.load_term_dict(data_folder + '/term_dict.json')
	index.load_concept_dict(data_folder + '/concept_dict.json')

	# load doc ids
	print('load doc ids')
	doc_ids = index.load_doc_ids(data_folder + '/doc_ids.json')

	# load queries
	print("load {} queries".format(FLAGS.corpus_name))
	if "OHSUMED" in FLAGS.corpus_name:  # load ohsumed queries
		qdict = safir_utils.read_ohsu_queries(query_folder + '/' + FLAGS.query_fname)  
	elif "TREC_CDS" in FLAGS.corpus_name:  # load cds queries
		qdict = safir_utils.read_cds_queries(query_folder + '/' + FLAGS.query_fname)  
	# get query ids
	qids = list(qdict.keys())

	# load semantic model
	print('load semantic model')
	with tf.Session() as sess:
		# restore model and get required tensors
		saver = tf.train.import_meta_graph(FLAGS.semantic_model + '.ckpt.meta')
		saver.restore(sess, FLAGS.semantic_model + '.ckpt')
		word_embs = sess.run(tf.get_default_graph().get_tensor_by_name('word_embs:0'))
		if FLAGS.include_concepts:
			concept_embs = sess.run(tf.get_default_graph().get_tensor_by_name('concept_embs:0'))
		proj_weights = sess.run(tf.get_default_graph().get_tensor_by_name('proj_weights:0'))
		doc_embs = sess.run(tf.get_default_graph().get_tensor_by_name('doc_embs:0'))
	print('model loaded!')

	
	'''
	PERFORM RE-RANKING
	'''

	# project queries into embedding space
	print('project queries into the embedding space')
	query_embs = []
	query_ids = []
	# loop over queries
	for qid, qtext in qdict.items(): 
		# project query
		if FLAGS.include_concepts:  # include concepts when performing query projection 
			proj_query = index.project_query(qtext[FLAGS.query_field], FLAGS.corpus_name, word_embs, proj_weights, concept_embs)
		else:  # include words only when performing query projection
			proj_query = index.project_query(qtext[FLAGS.query_field], FLAGS.corpus_name, word_embs, proj_weights)
		# keep projected query regardless of whether it contains known terms or not
		query_embs.append(proj_query)
		query_ids.append(qid)

	# shuffle query ids
	np.random.shuffle(qids)

	# perform re-ranking w/ a fixed gamma
	if FLAGS.gamma > 0:  
		print('perform re-ranking w/ gamma={}'.format(FLAGS.gamma))
		
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(FLAGS.run_base_name + '_gamma_' + str(FLAGS.gamma))
		# combine rankings using fixed gamma
		comb_run = compute_combined_run(run, FLAGS.query_field, query_ids, query_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[FLAGS.normalizer], FLAGS.gamma)
		# store test ranking in combined run
		for qid, doc_ids_and_scores in comb_run.items():
			crun.add_ranking(qid, [(score, docno) for docno, score in doc_ids_and_scores.items()])
		
		# close and store run 
		crun.close_and_write(out_path=rankings_folder + '/' + FLAGS.run_base_name + '_gamma_' + str(FLAGS.gamma) + '.txt', overwrite=True)
		print('combined run stored in {}'.format(rankings_folder))
		
		# evalaute combined run
		print('evaluate run combined w/ gamma={}'.format(FLAGS.gamma))
		safir_utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], 
							  rankings_folder, FLAGS.run_base_name + '_gamma_' + str(FLAGS.gamma), 
							  qrels_folder, FLAGS.qrels_fname)
		if 'TREC_CDS' in FLAGS.corpus_name:
			safir_utils.evaluate_inferred(['infNDCG'], 
										   rankings_folder, FLAGS.run_base_name + '_gamma_' + str(FLAGS.gamma), 
										   qrels_folder, FLAGS.inf_qrels_fname)
		return True

	# learn optimal weight to combine runs
	else:
		print("learn weights used to perform re-ranking with sweep={}".format(FLAGS.sweep))
		
		# set variable to store scores and weights
		scores_and_weights = []
		# initialize kfold with FLAGS.num_folds
		kfold = sklearn.model_selection.KFold(n_splits=FLAGS.num_folds)
		
		for fold, (train_qids, test_qids) in enumerate(kfold.split(qids)):
			print('fold n. {}'.format(fold))
			
			# split queries into train and test
			qtrain_ids = [query_ids[ix] for ix in train_qids]
			qtrain_embs = [query_embs[ix] for ix in train_qids]

			qtest_ids = [query_ids[ix] for ix in test_qids]
			qtest_embs = [query_embs[ix] for ix in test_qids]
			
			if 'inf' in FLAGS.ref_measure:
				# split qrels into train and test
				train_qrels = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
				train_qrels.write('\n'.join([qids[ix] + ' 0 ' + docno + ' ' + ' '.join([str(rel) for rel in rels]) 
											for ix in train_qids for docno, rels in qrels[qids[ix]].items()]))
				test_qrels = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
				test_qrels.write('\n'.join([qids[ix] + ' 0 ' + docno + ' ' + ' '.join([str(rel) 
											for rel in rels]) for ix in test_qids for docno, rels in qrels[qids[ix]].items()]))

				# obtain best combination on training queries
				train_score, best_train_weight = max(perform_reranking(run, FLAGS.query_field, qtrain_ids, qtrain_embs, doc_ids, doc_embs, safir_utils, inf_qrels=train_qrels))
				print('fold {}: best_train_weight={}, {}={}'.format(fold, best_train_weight, FLAGS.ref_measure, train_score))

				# compute re-ranking with best_train_weight on test queries
				test_crun = compute_combined_run(run, FLAGS.query_field, qtest_ids, qtest_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[FLAGS.normalizer], best_train_weight)
				# compute aggregated measure score for test queries
				test_score = evaluate_ref_measure(test_crun, test_qrels, safir_utils)
				# close qrels 
				train_qrels.close()
				test_qrels.close()
			else: 
				# obtain best combination on training queries
				train_score, best_train_weight = max(perform_reranking(run, FLAGS.query_field, qtrain_ids, qtrain_embs, doc_ids, doc_embs, evaluator))
				print('fold {}: best_train_weight={}, {}={}'.format(fold, best_train_weight, FLAGS.ref_measure, train_score))

				# compute re-ranking with best_train_weight on test queries
				test_crun = compute_combined_run(run, FLAGS.query_field, qtest_ids, qtest_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[FLAGS.normalizer], best_train_weight)
				# evaluate test run
				test_res = evaluator.evaluate(test_crun)
				# compute aggregated measure score for test queries
				test_score = pytrec_eval.compute_aggregated_measure(FLAGS.ref_measure, [qscore[FLAGS.ref_measure] for qscore in test_res.values()])
			# store averaged scores w/ best_train_weight
			scores_and_weights.append((np.mean([train_score, test_score]), best_train_weight))

		# get (best) weight that produces the highest averaged score
		best_score, best_weight = max(scores_and_weights)
		print('found best weight={}'.format(best_weight))
		
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(FLAGS.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + FLAGS.ref_measure)
		# compute combined run based on best_weight
		comb_run = compute_combined_run(run, FLAGS.query_field, query_ids, query_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[FLAGS.normalizer], best_weight) 
		
		# store re-ranking in crun
		for qid, doc_ids_and_scores in comb_run.items():
			crun.add_ranking(qid, [(score, doc_id) for doc_id, score in doc_ids_and_scores.items()])
		
		# close and store run
		crun.close_and_write(out_path=rankings_folder + '/' + FLAGS.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + FLAGS.ref_measure + '.txt', overwrite=True)
		print('combined run stored in {}'.format(rankings_folder))
		
		# evalaute combined run
		print('evaluate run combined w/ {}-fold cross validation and best weight={}'.format(FLAGS.num_folds, best_weight))
		safir_utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], 
							  rankings_folder, FLAGS.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + FLAGS.ref_measure, 
							  qrels_folder, FLAGS.qrels_fname)
		if 'TREC_CDS' in FLAGS.corpus_name:
			safir_utils.evaluate_inferred(['infNDCG'], 
										   rankings_folder, FLAGS.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + FLAGS.ref_measure, 
										   qrels_folder, FLAGS.inf_qrels_fname)
		return True


if __name__ == "__main__":
	main()  


