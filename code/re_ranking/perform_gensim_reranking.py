import os
import argparse
import subprocess
import collections 
import tempfile
import gensim
import umls
import sklearn.model_selection
import pytrec_eval
import json
import numpy as np 

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from cvangysel import trec_utils

from gensim_utils import Utils


flags = argparse.ArgumentParser()

flags.add_argument("--lexical_run", default="", type=str, help="Lexical baseline run.")
flags.add_argument("--semantic_model", default="", type=str, help="Semantic model path.")
flags.add_argument("--include_docs", default=True, type=bool, help="Whether to consider Word2Vec or Doc2Vec models.")
flags.add_argument("--use_concepts", default=False, type=bool, help="Whether to consider concept-based models or not.")
flags.add_argument("--retrofitted", default=True, type=bool, help="Whether to consider retrofitted models or not.")
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


class Options(object):
	"""options used to perform re-ranking w/ gensim models"""

	def __init__(self):
		# lexical run
		self.lexical_run = FLAGS.lexical_run
		# semantic model
		self.semantic_model = FLAGS.semantic_model
		# include docs
		self.include_docs = FLAGS.include_docs
		# include concepts
		self.use_concepts = FLAGS.use_concepts
		# retrofitted models
		self.retrofitted = FLAGS.retrofitted
		# qrels file name
		self.qrels_fname = FLAGS.qrels_fname
		# inferred qrels file name
		self.inf_qrels_fname = FLAGS.inf_qrels_fname
		# query file name
		self.query_fname = FLAGS.query_fname
		# query field
		self.query_field = FLAGS.query_field
		# reference measure
		self.ref_measure = FLAGS.ref_measure
		# gamma 
		self.gamma = FLAGS.gamma
		# number of folds
		self.num_folds = FLAGS.num_folds
		# sweep 
		self.sweep = FLAGS.sweep
		# seed 
		self.seed = FLAGS.seed
		# corpus name
		self.corpus_name = FLAGS.corpus_name
		# run (base) name
		self.run_base_name = FLAGS.run_base_name
		# normalizer 
		self.normalizer = FLAGS.normalizer


def parse_inf_qrels(inf_qrels_file):
	"""parse inferred qrels"""
	inf_qrels = collections.defaultdict(dict)
	for line in inf_qrels_file:
		query_id, _, object_id, relevance, inf_relevance = line.strip().split()
		assert object_id not in inf_qrels[query_id]
		inf_qrels[query_id][object_id] = (int(relevance), int(inf_relevance))
	return inf_qrels


def evaluate_ref_measure(ranking, qrels_file, evaluator, opts):
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
	agg_measure_score = evaluator.evaluate_inferred(opts.ref_measure, rank_folder, rank_name, qrels_folder, qrels_name)[opts.ref_measure]
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


def perform_reranking(run, qfield, query_ids, query_embs, docnos, doc_embs, evaluator, opts, inf_qrels=None):
	"""perform re-ranking of input run w/ semantic model"""
	print("re-ranking models")
	# loop over weight values with sweep equal to sweep
	for weight in np.arange(opts.sweep, 1.0, opts.sweep):
		weight = round(weight, 4)  # round weight up to 4 decimals
		# generate combined run with current weight
		combined_run = compute_combined_run(run, qfield, query_ids, query_embs, docnos, doc_embs, SCORE_NORMALIZERS[opts.normalizer], weight)
		# evaluate combined run 
		if 'inf' in opts.ref_measure:  # evaluate for inferred reference measure
			agg_measure_score = evaluate_ref_measure(combined_run, inf_qrels, evaluator, opts)
		else:  # evaluate for standard reference measure
			results = evaluator.evaluate(combined_run)
			# compute aggregated measure score
			agg_measure_score = pytrec_eval.compute_aggregated_measure(opts.ref_measure, [qmeasures[opts.ref_measure] for qmeasures in results.values()])
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
	# load options
	opts = Options()
	# set folders
	index_folder = 'corpus/' + opts.corpus_name + '/index'
	data_folder = 'corpus/' + opts.corpus_name + '/data'
	query_folder = 'corpus/' + opts.corpus_name + '/queries'
	qrels_folder = 'corpus/' + opts.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + opts.corpus_name + '/rankings/' + opts.run_base_name

	# create folders
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	if opts.sweep > 0:
		if 'inf' in opts.ref_measure:  
			# parse and store inferred qrels
			with open(qrels_folder + '/' + opts.inf_qrels_fname + '.txt', 'r') as qrelf:
				qrels = parse_inf_qrels(qrelf)
		else:  
			# parse and store qrels
			with open(qrels_folder + '/' + opts.qrels_fname + '.txt', 'r') as qrelf:
				qrels = pytrec_eval.parse_qrel(qrelf)
			# set reference category
			if 'cut' in opts.ref_measure:
				ref_category = '_'.join(opts.ref_measure.split('_')[:2])
			else:
				ref_category = opts.ref_measure.split('_')[0]
			# initialize evaluator over qrels
			evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ref_category})  # evaluate on reference measure

	# initialize utils funcs
	utils = Utils()
	# load UMLS lookup funcs
	umls_lookup = umls.UMLSLookup()
	# set random seed to enable repro
	np.random.seed(opts.seed)

	# load queries
	print('load {} queries'.format(opts.corpus_name))
	if 'OHSUMED' in opts.corpus_name:  # load ohsu queries
		qdict = utils.read_ohsu_queries(query_folder + '/' + opts.query_fname)
	elif 'TREC_CDS' in opts.corpus_name:  # load cds queries
		qdict = utils.read_cds_queries(query_folder + '/' + opts.query_fname)
	
	if not opts.include_docs or opts.use_concepts:  # load required data for Word2Vec models or cDoc2Vec
		# load required data
		print('load index')
		ix = gensim.corpora.Dictionary.load_from_text(index_folder + '/' + opts.corpus_name + '_ix.txt')
		ix.id2token = {v: k for k, v in ix.token2id.items()}

	if not opts.include_docs:  # load required data to compute doc embeddings w/ Word2Vec models
		# load required data
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
		weights = utils.compute_idfs(term_dict)

	if opts.use_concepts:  # convert terms into concepts within queries
		# load required data
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
		# disambiguate query
		for qid, qbody in qdict.items():
			enc_query = utils.disambiguate_query(ix, ix2ix, ix2dict, token2cuis, qbody[opts.query_field], opts.corpus_name)
			qdict[qid] = {opts.query_field: enc_query}
		# remove '__NULL__' concept from safir concept dict
		ix2dict.pop('__NULL__', None)
		# reverse safir concept dict
		ix2dict = {v: k for k, v in ix2dict.items()}
		print('convert queries to concepts-only')
		for qid, qbody in qdict.items():
			cuis_query = ' '.join([ix2dict[cui] for _, cui in qdict[qid][opts.query_field] if cui in ix2dict])
			qdict[qid][opts.query_field] = cuis_query
		print('converted {} queries become ... "{}" ...'.format(opts.corpus_name, list(qdict.values())[0][opts.query_field]))
	
	# get query ids 
	qids = list(qdict.keys())

	# parse lexical run
	print('parse lexical run')
	with open(opts.lexical_run, 'r') as lex_run:
		run = pytrec_eval.parse_run(lex_run)

	# load semantic model
	print('load semantic model')
	if opts.include_docs:  # load Doc2Vec models
		if opts.retrofitted:  # load rDoc2Vec 
			# get document vectors
			docvecs = gensim.models.doc2vec.Doc2VecKeyedVectors.load(opts.semantic_model + '.docs')
			doc_embs = docvecs.vectors
			doc_ids = np.array(list(docvecs.vocab.keys()))
			del docvecs  # free memory space
			# get word vectors 
			model = gensim.models.Word2Vec.load(opts.semantic_model + '.tokens')
		else:  # load Doc2Vec
			model = gensim.models.Doc2Vec.load(opts.semantic_model)
			# get document vectors
			doc_embs = model.docvecs.vectors_docs
			doc_ids = np.array(list(model.docvecs.doctags.keys()))
	else:  # load Word2Vec models
		if opts.retrofitted:  # load rWord2Vec
			model = gensim.models.word2vec.Word2VecKeyedVectors.load(opts.semantic_model)
		else:  # load Word2Vec
			model = gensim.models.Word2Vec.load(opts.semantic_model)
		# get document vectors
		print('compute doc vectors for Word2Vec models')
		doc_ids_embs = utils.compute_doc_embs(dict(zip(doc_ids, docs)), model, weights)
		del docs  # free memory space
		# get doc embs and ids from doc_ids_embs
		doc_embs = np.array(list(doc_ids_embs.values()))
		doc_ids = np.array(list(doc_ids_embs.keys()))
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
		proj_query = utils.query2emb(qtext[opts.query_field], model, use_concepts=opts.use_concepts)
		# keep projected query regardless of whether it contains known terms or not
		query_embs.append(proj_query)
		query_ids.append(qid)

	# shuffle query ids
	np.random.shuffle(qids)

	# perform re-ranking w/ a fixed gamma
	if opts.gamma > 0:  
		print('perform re-ranking w/ gamma={}'.format(opts.gamma))
		
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(opts.run_base_name + '_gamma_' + str(opts.gamma))
		# combine rankings using fixed gamma
		comb_run = compute_combined_run(run, opts.query_field, query_ids, query_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[opts.normalizer], opts.gamma)
		# store test ranking in combined run
		for qid, doc_ids_and_scores in comb_run.items():
			crun.add_ranking(qid, [(score, docno) for docno, score in doc_ids_and_scores.items()])
		
		# close and store run 
		crun.close_and_write(out_path=rankings_folder + '/' + opts.run_base_name + '_gamma_' + str(opts.gamma) + '.txt', overwrite=True)
		print('combined run stored in {}'.format(rankings_folder))
		
		# evalaute combined run
		print('evaluate run combined w/ gamma={}'.format(opts.gamma))
		utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], 
						rankings_folder, opts.run_base_name + '_gamma_' + str(opts.gamma), 
						qrels_folder, opts.qrels_fname)
		if 'TREC_CDS' in opts.corpus_name:
			utils.evaluate_inferred(['infNDCG'], 
									 rankings_folder, opts.run_base_name + '_gamma_' + str(opts.gamma), 
									 qrels_folder, opts.inf_qrels_fname)
		return True

	# learn optimal weight to combine runs
	else:
		print("learn weights used to perform re-ranking with sweep={}".format(opts.sweep))
		
		# set variable to store scores and weights
		scores_and_weights = []
		# initialize kfold with opts.num_folds
		kfold = sklearn.model_selection.KFold(n_splits=opts.num_folds)
		
		for fold, (train_qids, test_qids) in enumerate(kfold.split(qids)):
			print('fold n. {}'.format(fold))
			
			# split queries into train and test
			qtrain_ids = [query_ids[ix] for ix in train_qids]
			qtrain_embs = [query_embs[ix] for ix in train_qids]

			qtest_ids = [query_ids[ix] for ix in test_qids]
			qtest_embs = [query_embs[ix] for ix in test_qids]
			
			if 'inf' in opts.ref_measure:
				# split qrels into train and test
				train_qrels = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
				train_qrels.write('\n'.join([qids[ix] + ' 0 ' + docno + ' ' + ' '.join([str(rel) for rel in rels]) 
											for ix in train_qids for docno, rels in qrels[qids[ix]].items()]))
				test_qrels = tempfile.NamedTemporaryFile(dir='.', suffix='.txt', mode='w')
				test_qrels.write('\n'.join([qids[ix] + ' 0 ' + docno + ' ' + ' '.join([str(rel) 
											for rel in rels]) for ix in test_qids for docno, rels in qrels[qids[ix]].items()]))

				# obtain best combination on training queries
				train_score, best_train_weight = max(perform_reranking(run, opts.query_field, qtrain_ids, qtrain_embs, doc_ids, doc_embs, utils, opts, inf_qrels=train_qrels))
				print('fold {}: best_train_weight={}, {}={}'.format(fold, best_train_weight, opts.ref_measure, train_score))

				# compute re-ranking with best_train_weight on test queries
				test_crun = compute_combined_run(run, opts.query_field, qtest_ids, qtest_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[opts.normalizer], best_train_weight)
				# compute aggregated measure score for test queries
				test_score = evaluate_ref_measure(test_crun, test_qrels, utils, opts)
				# close qrels 
				train_qrels.close()
				test_qrels.close()
			else: 
				# obtain best combination on training queries
				train_score, best_train_weight = max(perform_reranking(run, opts.query_field, qtrain_ids, qtrain_embs, doc_ids, doc_embs, evaluator, opts))
				print('fold {}: best_train_weight={}, {}={}'.format(fold, best_train_weight, opts.ref_measure, train_score))

				# compute re-ranking with best_train_weight on test queries
				test_crun = compute_combined_run(run, opts.query_field, qtest_ids, qtest_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[opts.normalizer], best_train_weight)
				# evaluate test run
				test_res = evaluator.evaluate(test_crun)
				# compute aggregated measure score for test queries
				test_score = pytrec_eval.compute_aggregated_measure(opts.ref_measure, [qscore[opts.ref_measure] for qscore in test_res.values()])
			
			# store averaged scores w/ best_train_weight
			scores_and_weights.append((np.mean([train_score, test_score]), best_train_weight))

		# get (best) weight that produces the highest averaged score
		best_score, best_weight = max(scores_and_weights)
		print('found best weight={}'.format(best_weight))
		
		# initialize combined (output) run
		crun = trec_utils.OnlineTRECRun(opts.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + opts.ref_measure)
		# compute combined run based on best_weight
		comb_run = compute_combined_run(run, opts.query_field, query_ids, query_embs, doc_ids, doc_embs, SCORE_NORMALIZERS[opts.normalizer], best_weight) 
		
		# store re-ranking in crun
		for qid, doc_ids_and_scores in comb_run.items():
			crun.add_ranking(qid, [(score, doc_id) for doc_id, score in doc_ids_and_scores.items()])
		
		# close and store run
		crun.close_and_write(out_path=rankings_folder + '/' + opts.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + opts.ref_measure + '.txt', overwrite=True)
		print('combined run stored in {}'.format(rankings_folder))
		
		# evalaute combined run
		print('evaluate run combined w/ {}-fold cross validation and best weight={}'.format(opts.num_folds, best_weight))
		utils.evaluate(['ndcg', 'P_10', 'num_rel_ret'], 
						rankings_folder, opts.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + opts.ref_measure, 
						qrels_folder, opts.qrels_fname)
		if 'TREC_CDS' in opts.corpus_name:
			utils.evaluate_inferred(['infNDCG'], 
									 rankings_folder, opts.run_base_name + '_opt_gamma_' + str(best_weight) + '_' + opts.ref_measure, 
									 qrels_folder, opts.inf_qrels_fname)
		return True


if __name__ == "__main__":
	main()  