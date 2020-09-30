import os
import argparse

from elasticsearch import Elasticsearch
from whoosh.analysis import SimpleAnalyzer

from es_utils import settings
from es_utils import index
from es_utils import utils


flags = argparse.ArgumentParser()

flags.add_argument("--stopwords", default=True, type=bool,
				   help="Whether to remove stopwords from corpus (Indri stopword list applied).")
flags.add_argument("--delete_ix", default=False, type=bool, help="Delete index, if any.")
flags.add_argument("--corpus_name", default="OHSUMED", type=str, help="Used corpus.")
flags.add_argument("--query_fname", default="topics_all", type=str, help="Queries file name.")
flags.add_argument("--qrels_fname", default="qrels_all", type=str, help="Qrels file name.")
flags.add_argument("--inf_qrels_fname", default="", type=str, help="Inferred qrels file name.")
flags.add_argument("--query_field", default="desc", type=str, help="Query field to consider when performing retrieval.")
flags.add_argument("--model_name", default="LMDirichlet", type=str, help="Model name. Tip: model names are, among the others, 'BM25', 'LMDirichlet', 'DFR'.")
flags.add_argument("--feedback_docs", default=10, type=int, help="Number of feedback documents considered for pseudo-releance feedback.")
flags.add_argument("--feedback_terms", default=10, type=int, help="Number of feedback terms considered for pseudo-releance feedback.")
flags.add_argument("--query_weight", default=0.5, type=float, help="The weight applied to terms belonging to the original query when interpolating.")
flags.add_argument("--run_name", default="qlm_rm3_qlm", type=str, help="Run name.")

FLAGS = flags.parse_args()


def main():
	# set folders
	corpus_folder = 'corpus/' + FLAGS.corpus_name + '/' + FLAGS.corpus_name
	query_folder = 'corpus/' + FLAGS.corpus_name + '/queries'
	qrels_folder = 'corpus/' + FLAGS.corpus_name + '/qrels'
	rankings_folder = 'corpus/' + FLAGS.corpus_name + '/rankings/rm3_runs_orig'

	# create folders
	if not os.path.exists(rankings_folder):
		os.makedirs(rankings_folder)
	if not os.path.exists(query_folder) or not os.path.exists(qrels_folder):
		print('folders containing queries and qrels are required - please add them')
		return False

	# set constants and properties
	es_settings = settings.ESS(FLAGS.corpus_name)
	constants = es_settings.constants
	properties = es_settings.properties
	del es_settings

	# set index
	ix = index.Index(constants, properties)

	if FLAGS.delete_ix:  # delete index if required
		if constants.ES.indices.exists(index=FLAGS.corpus_name.lower()):
			print('delete {} index'.format(FLAGS.corpus_name))
			res = constants.ES.indices.delete(index=FLAGS.corpus_name.lower())
		else: 
			print('index {} does not exists'.format(FLAGS.corpus_name))

	# index corpus
	print('index {} collection'.format(FLAGS.corpus_name))
	ix.index_corpus(corpus_folder)

	# read queries
	if 'OHSUMED' in FLAGS.corpus_name:
		qdict = utils.read_ohsu_queries(query_folder + '/' + FLAGS.query_fname)
	elif 'CDS' in FLAGS.corpus_name:
		qdict = utils.read_cds_queries(query_folder + '/' + FLAGS.query_fname)

	if FLAGS.model_name == 'BM25':
		print('change model to BM25 with k2={} and b={}'.format(1.2, 0.75))
		ix.change_model(model='BM25', k2=1.2, b=0.75)
	elif FLAGS.model_name == 'LMDirichlet':
		print('change model to QLM (Dirichlet) with mu={}'.format(2000))
		ix.change_model(model='LMDirichlet', mu=2000)
	elif FLAGS.model_name == 'DFR':
		print('change model to DFR with basic_model={}, after_effect={}, and normalization={}'.format('if', 'b', 'h2'))
		ix.change_model(model='DFR', basic_model='if', after_effect='b', normalization='h2')
	else: 
		print('select a model between BM25, LMDirichlet, and DFR')
		return False

	# perform retrieval
	print('perform lexical search using {} over {} field for {} queries'.format(FLAGS.model_name, FLAGS.query_field, FLAGS.corpus_name))
	ix.lexical_search(qdict, FLAGS.query_field, rankings_folder, FLAGS.run_name)

	# evaluate ranking
	print('evaluate ranking obtained with {} over {} for {} queries'.format(FLAGS.model_name, FLAGS.query_field, FLAGS.corpus_name))
	utils.evaluate(['ndcg_cut_1000', 'ndcg_cut_100', 'ndcg_cut_10', 'P_10', 'recall_1000'], rankings_folder, FLAGS.run_name, qrels_folder, FLAGS.qrels_fname)
	if 'CDS' in FLAGS.corpus_name:
		utils.evaluate_inferred(['infNDCG'], rankings_folder, FLAGS.run_name, qrels_folder, FLAGS.inf_qrels_fname) 

	# perform RM3 pseudo-relevance feedback
	print('perform rm3 pseudo-relevance feedback with {} feedback documents and {} feedback_terms and interpolating with weight {}'.format(FLAGS.feedback_docs, FLAGS.feedback_terms, FLAGS.query_weight))
	ix.perform_rm3_prf(qdict, FLAGS.query_field, rankings_folder, FLAGS.run_name, FLAGS.feedback_docs, FLAGS.feedback_terms, FLAGS.query_weight)

	# evaluate ranking
	print('evaluate ranking obtained with {} over {} for {} queries'.format(FLAGS.model_name, FLAGS.query_field, FLAGS.corpus_name))
	utils.evaluate(['ndcg_cut_1000', 'ndcg_cut_100', 'ndcg_cut_10', 'P_10', 'recall_1000'], rankings_folder, FLAGS.run_name, qrels_folder, FLAGS.qrels_fname)
	if 'CDS' in FLAGS.corpus_name:
		utils.evaluate_inferred(['infNDCG'], rankings_folder, FLAGS.run_name, qrels_folder, FLAGS.inf_qrels_fname) 


if __name__ == "__main__":
	main()