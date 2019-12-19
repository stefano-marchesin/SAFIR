import os
import argparse
import subprocess
import collections 
import tempfile
import sklearn.model_selection
import pytrec_eval
import numpy as np 

from tqdm import tqdm
from cvangysel import trec_utils

import safir_utils


flags = argparse.ArgumentParser()

flags.add_argument("--lexical_run", default="", type=str, help="Lexical run path.")
flags.add_argument("--neural_run", default="", type=str, help="Neural run path.")
flags.add_argument("--qrels_fname", default="qrels-treceval-2015", type=str, help="Qrels file path.")
flags.add_argument("--inf_qrels_fname", default="qrels-sampleval-2015", type=str, help="Inferred qrels file path.")
flags.add_argument("--query_fname", default="topics2015A.xml", type=str, help="Query file path.")
flags.add_argument("--supervised", default=True, type=bool, help="Set to False for unsupervised rank fusion.")
flags.add_argument("--num_folds", default=2, type=int, help="Number of folds to consider for cross validation.")
flags.add_argument("--sweep", default=0.05, type=float, help="Sweep value for optimized weights.")
flags.add_argument("--weight", default=0, type=float, help="Fixed weight to combine runs in unsupervised mode. Set --supervised to False")
flags.add_argument("--method", default="CombSUM", type=str, help="Classic fusion method - possible methods: 'CombMNZ', 'CombANZ', 'CombSUM', 'CombAVG'. Set --supervised to False and --weight to 0")
flags.add_argument("--seed", default=42, type=int, help="Answer to ultimate question of life, the universe and everything.")
flags.add_argument("--normalizer", default="minmax", type=str, help="Selected normalizer - possible normalizers: 'standardize', 'minmax', 'none'.")
flags.add_argument("--ref_measure", default="infNDCG", type=str, help="Reference measure used to optimize cross validation.")
flags.add_argument("--corpus_name", default="TREC_CDS14_15", type=str, help="Corpus name.")
flags.add_argument("--run_base_name", default="", type=str, help="Combined run (base) name.")


FLAGS = flags.parse_args()


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


def classic_rank_fusion(runs, qids, method, norm):
	"""combine input runs using classic rank fusion techniques"""
	combined_run = {}
	# loop over query ids
	for qid in qids:
		# query ranking
		combined_ranking = collections.defaultdict(list)
		# iterate over runs
		for idx, run in enumerate(runs):
			# check whether run has ranking for current query
			if run[qid]:
				# get run ranking
				ranking = run[qid]
				# compute normalization
				normalizer = norm(list(ranking.values()))
				# iterate over docs within ranking
				for doc_id, score in ranking.items():
					# append normalized doc score to combined ranking
					combined_ranking[doc_id].append(normalizer(score))
		# append combined ranking to combined run
		if method == 'CombMNZ':
			combined_run[qid] = {doc_id: len(scores) * np.sum(scores) for doc_id, scores in combined_ranking.items()}
		elif method == 'CombANZ':
			combined_run[qid] = {doc_id: np.sum(scores) / len(scores) for doc_id, scores in combined_ranking.items()}
		elif method == 'CombSUM':
			combined_run[qid] = {doc_id: np.sum(scores) for doc_id, scores in combined_ranking.items()}
		elif method == 'CombAVG':
			combined_run[qid] = {doc_id: np.sum(scores) / len(runs) for doc_id, scores in combined_ranking.items()}
	return combined_run


def compute_combined_run(runs, weights, qids, norm):
	"""combine input runs for given queries using specific weights and normalizer"""
	combined_run = {}
	# loop over query ids
	for qid in qids:
		# query ranking
		combined_ranking = collections.defaultdict(list)
		# iterate over runs
		for idx, run in enumerate(runs):
			# check whether run has ranking for current query
			if run[qid]:
				# get run ranking
				ranking = run[qid]
				# compute normalization
				normalizer = norm(list(ranking.values()))
				# iterate over docs within ranking
				for doc_id, score in ranking.items():
					# append weighted normalized doc score to combined ranking
					combined_ranking[doc_id].append(weights[idx] * normalizer(score))
		# append combined ranking to combined run
		combined_run[qid] = dict(collections.Counter({doc_id: np.mean(scores) for doc_id, scores in combined_ranking.items()}).most_common(1000))
	return combined_run


def _generate(runs, qids, evaluator, inf_qrels=None):
	"""generate combined run from input runs"""
	print("combining {} models".format(len(runs)))
	for alpha in tqdm(np.arange(FLAGS.sweep, 1.0, FLAGS.sweep)):
		alpha = round(alpha, 4)  # round alpha up to 4 decimals
		# assign params to weights
		weights = [alpha, 1-alpha]
		# generate combined run with current params as weights
		combined_run = compute_combined_run(runs, weights, qids, norm=SCORE_NORMALIZERS[FLAGS.normalizer])
		# evaluate combined runs
		if 'inf' in FLAGS.ref_measure:  # evaluate for inferred reference measure
			agg_measure_score = evaluate_ref_measure(combined_run, inf_qrels, evaluator)
		else:  # evaluate for standard reference measure
			results = evaluator.evaluate(combined_run)
			# compute aggregated measure score
			agg_measure_score = pytrec_eval.compute_aggregated_measure(FLAGS.ref_measure, [qmeasures[FLAGS.ref_measure] for qmeasures in results.values()])
		# return aggregated measure score and return params
		yield agg_measure_score, alpha


def main():
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	# set folders
	query_folder = 'corpus/' + FLAGS.corpus_name + '/queries'
	qrels_folder = 'corpus/' + FLAGS.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + FLAGS.corpus_name + '/rankings/' + FLAGS.run_base_name

	# create folders
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False	

	# set random seed - enable repro
	np.random.seed(FLAGS.seed)

	# load queries
	print("load {} queries".format(FLAGS.corpus_name))
	if "OHSUMED" in FLAGS.corpus_name:  # load ohsumed queries
		qdict = safir_utils.read_ohsu_queries(query_folder + '/' + FLAGS.query_fname)  
	elif "TREC_CDS" in FLAGS.corpus_name:  # load cds queries
		qdict = safir_utils.read_cds_queries(query_folder + '/' + FLAGS.query_fname)  
	# extract query ids from queries
	qids = list(qdict.keys())

	# parse and store runs
	runs = []
	# consider lexical run for combination
	if FLAGS.lexical_run:  
		with open(FLAGS.lexical_run, 'r') as lex_run:
			runs.append(pytrec_eval.parse_run(lex_run))
	# consider neural run for combination
	if FLAGS.neural_run:  
		with open(FLAGS.neural_run, 'r') as neu_run:
			runs.append(pytrec_eval.parse_run(neu_run))

	# assert that len(runs) is equal to 2
	if len(runs) != 2:
		print("please provide two runs")
		return False

	if FLAGS.supervised:
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
	
	# learn optimal weights to combine runs using k-fold cross validation
	if FLAGS.supervised:
		print("learn optimal values to combine runs with sweep: {}".format(FLAGS.sweep))
		
		# set variable to store scores and weights
		scores_and_weights = []
		# shuffle query ids
		np.random.shuffle(qids)
		# initialize kfold with FLAGS.num_folds
		kfold = sklearn.model_selection.KFold(n_splits=FLAGS.num_folds)

		for fold_ix, (train_qids, test_qids) in enumerate(kfold.split(qids)):
			print("fold n. {}".format(fold_ix))
			
			if 'inf' in FLAGS.ref_measure:
				# split qrels into train and test
				train_qrels = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
				train_qrels.write('\n'.join([qids[ix] + ' 0 ' + docno + ' ' + ' '.join([str(rel) for rel in rels]) 
											for ix in train_qids for docno, rels in qrels[qids[ix]].items()]))
				test_qrels = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
				test_qrels.write('\n'.join([qids[ix] + ' 0 ' + docno + ' ' + ' '.join([str(rel) 
											for rel in rels]) for ix in test_qids for docno, rels in qrels[qids[ix]].items()]))
				
				# obtain best combination on train ids
				train_score, best_train_alpha = max(_generate(runs, [qids[ix] for ix in train_qids], safir_utils, inf_qrels=train_qrels))
				print('fold %d: best_train_alpha=%.2f, best train %s=%.4f' % (fold_ix, best_train_alpha, FLAGS.ref_measure, train_score))
				# evaluate best combination on test ids
				test_crun = compute_combined_run(runs, [best_train_alpha, 1-best_train_alpha], [qids[ix] for ix in test_qids], norm=SCORE_NORMALIZERS[FLAGS.normalizer])
				# compute aggregated measure score for test queries
				test_score = evaluate_ref_measure(test_crun, test_qrels, safir_utils)
				# close qrels 
				train_qrels.close()
				test_qrels.close()
			else:
				# obtain best combination on train ids
				train_score, best_train_alpha = max(_generate(runs, [qids[ix] for ix in train_qids], evaluator))
				print('fold %d: best_train_alpha=%.2f, best train %s=%.4f' % (fold_ix, best_train_alpha, FLAGS.ref_measure, train_score))
				# evaluate best combination on test ids
				test_crun = compute_combined_run(runs, [best_train_alpha, 1-best_train_alpha], [qids[ix] for ix in test_qids], norm=SCORE_NORMALIZERS[FLAGS.normalizer])
				# evaluate test run
				test_res = evaluator.evaluate(test_crun)
				# compute aggregated measure score for test queries
				test_score = pytrec_eval.compute_aggregated_measure(FLAGS.ref_measure, [qscore[FLAGS.ref_measure] for qscore in test_res.values()])

			# store averaged scores w/ best_train_weight
			scores_and_weights.append((np.mean([train_score, test_score]), best_train_alpha))

		# get (best) weight that produces the highest averaged score
		best_score, best_weight = max(scores_and_weights)
		print('found best weight={}'.format(best_weight))
		
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(FLAGS.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + FLAGS.ref_measure)
		# compute combined run based on best_weight
		comb_run = compute_combined_run(runs, [best_weight, 1-best_weight], qids, norm=SCORE_NORMALIZERS[FLAGS.normalizer]) 
		
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

	# use fixed weight to combine runs
	elif FLAGS.weight > 0:  
		print("use fixed weight={} to combine runs".format(FLAGS.weight))
		
		# initialize combined run
		crun = trec_utils.OnlineTRECRun(FLAGS.run_base_name + '_gamma_' + FLAGS.weight)
		# evaluate combination with fixed weights on test ids
		test_combined_run = compute_combined_run(runs, [FLAGS.weight,1-FLAGS.weight], qids, norm=SCORE_NORMALIZERS[FLAGS.normalizer])
		# store test ranking in combined run
		for qid, doc_ids_and_scores in test_combined_run.items():
			crun.add_ranking(qid, [(score, doc_id) for doc_id, score in doc_ids_and_scores.items()]) 
		
		# close run and store it in out_path
		crun.close_and_write(out_path=rankings_folder + '/' + FLAGS.run_base_name + '_gamma_' + FLAGS.weight + '.txt', overwrite=True)
		print("combined run stored in {}".format(rankings_folder))
		
		# evaluate combined run
		print("evaluate combined run using trec_eval")
		# evaluate combined run
		print("evaluate combined run using trec_eval")
		safir_utils.evaluate_rankings(['ndcg', 'P_10', 'num_rel_ret'], 
									   rankings_folder + '/' + FLAGS.run_base_name + '_gamma_' + FLAGS.weight + '.txt', 
									   qrels_folder, FLAGS.qrels_fname)
		if 'TREC_CDS' in FLAGS.corpus_name:
			safir_utils.evaluate_inferred(['infNDCG'], 
									   	   rankings_folder + '/' + FLAGS.run_base_name + '_gamma_' + FLAGS.weight + '.txt', 
									   	   qrels_folder, FLAGS.inf_qrels_fname)
		return True

	else:  # use classic rank fusion techniques
		print("use {} technique to perform rank fusion".format(FLAGS.method))
		
		# initialize combined run
		crun = trec_utils.OnlineTRECRun(FLAGS.run_base_name + '_' + FLAGS.method)
		# evaluate combination with classic method on test ids
		test_combined_run = classic_rank_fusion(runs, qids, FLAGS.method, norm=SCORE_NORMALIZERS[FLAGS.normalizer])
		# store test ranking in combined run
		for qid, doc_ids_and_scores in test_combined_run.items():
			crun.add_ranking(qid, [(score, doc_id) for doc_id, score in doc_ids_and_scores.items()])
		
		# close run and store it in out_path
		crun.close_and_write(out_path=rankings_folder + '/' + FLAGS.run_base_name + '_' + FLAGS.method + '.txt', overwrite=True)
		print("combined run stored in {}".format(rankings_folder))
		# evaluate combined run
		print("evaluate combined run using trec_eval")
		safir_utils.evaluate_rankings(['ndcg', 'P_10', 'num_rel_ret'], 
									   rankings_folder + '/' + FLAGS.run_base_name + '_' + FLAGS.method + '.txt', 
									   qrels_folder, FLAGS.qrels_fname)
		if 'TREC_CDS' in FLAGS.corpus_name:
			safir_utils.evaluate_inferred(['infNDCG'], 
									   	   rankings_folder + '/' + FLAGS.run_base_name + '_' + FLAGS.method + '.txt', 
									   	   qrels_folder, FLAGS.inf_qrels_fname)
		return True


if __name__ == "__main__":
	main()	


