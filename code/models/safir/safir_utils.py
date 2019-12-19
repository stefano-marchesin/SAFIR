import math 
import subprocess
import glob
import numpy as np 
import xml.etree.ElementTree as ETree

from tqdm import tqdm

from pubmed_parser.pubmed_oa_parser import list_xml_path, parse_pubmed_xml


def load_stopwords(stopwords_path):
	"""read stopwords file into list"""
	with open(stopwords_path, 'r') as sl:
		stop_words = {stop.strip() for stop in sl}
	return stop_words


def load_semtypes(semtypes_path):
	"""read semantic types into list"""
	with open(semtypes_path, 'r') as st:
		semtypes = [semtype.split('|')[1] for semtype in st]
	return semtypes


def get_trec_corpus(corpus_path):
	"""convert trec style corpus into (valid) xml"""
	docs = glob.glob(corpus_path + '/**/*.txt', recursive=True)
	for doc in docs:
		with open(doc, 'r') as f:  # read doc
			xml = f.read()
		# convert into true xml
		xml = '<ROOT>' + xml + '</ROOT>'
		# fix bad-formed tokens
		xml = xml.replace('&', '&amp;')
		yield xml


def gen_trec_doc(corpus_path):
	"""generate doc from batch of TREC-style docs"""
	ohsu = get_trec_corpus(corpus_path)
	# loop over batches
	for batch in ohsu:
		# parse xml
		root = ETree.fromstring(batch)
		# loop through each doc in the batch
		for doc in tqdm(root):
			docno = ''
			body = ''
			# loop through each element (tag, value)
			for elem in doc:
				if elem.tag == 'DOCNO':
					docno = elem.text.strip()
				else:
					body = elem.text.strip()
			# return doc to index
			yield docno, body


def gen_cds_doc(corpus_path):
	"""generate doc from batch of NXML-style docs"""
	cds = list_xml_path(corpus_path)
	# loop over doc paths
	for doc_path in tqdm(cds):
		# parse doc and get required data
		doc_data = parse_pubmed_xml(doc_path)
		docno = doc_data['pmc'].strip()
		title = doc_data['full_title'].strip()
		abstract = doc_data['abstract'].strip()
		body = ' '.join([title, abstract])
		# return doc to index
		yield docno, body


def compute_num_batches(enc_corpus, batch_size, ngram_size):
	"""compute number of batch iterations per epoch"""
	docs_length = [len(doc) for doc in enc_corpus]
	# compute number of batches
	num_batches = math.ceil(sum([max(doc_length - ngram_size + 1, 0) for doc_length in docs_length]) / batch_size)
	return num_batches  


def get_allowed_docs(enc_corpus, ngram_size):
	"""return list of allowed documents for the given ngram size"""
	allowed_docs = list()
	filtered_docs = list()
	# loop over documents and store doc indexes when len(doc) >= ngram_size
	for idx, doc in enumerate(enc_corpus):
		if len(doc) >= ngram_size:
			allowed_docs.append(idx)
		else:
			filtered_docs.append(idx)
	print('filtered {} docs'.format(len(filtered_docs)))
	return np.array(allowed_docs)


def generate_batch_data(enc_corpus, allowed_docs, batch_size, ngram_size, neg_samples):
	"""generate a batch of data for given corpus (optimized)"""
	corpus_size = len(enc_corpus)
	# select random documents from allowed documents (i.e. documents with len(doc) >= ngram_size)
	rand_docs_idx = np.random.choice(allowed_docs, size=batch_size)
	# compute documents length
	docs_length = [len(enc_corpus[rand_doc_idx]) for rand_doc_idx in rand_docs_idx]
	# store position of last prefixes + 1 (one above the highest prefix available)
	last_prefixes = [doc_length - ngram_size + 1 for doc_length in docs_length]
	# sample random prefixes lower than or equal to last_prefixes
	prefixes = [np.random.randint(last_prefix) for last_prefix in last_prefixes]
	# slices = prefixes + ngram_size
	ngrams = [enc_corpus[rand_doc_idx][prefix:prefix+ngram_size] for rand_doc_idx, prefix in zip(rand_docs_idx, prefixes)]
	# generate negative labels - discrete uniform distribution
	negative_labels = np.random.randint(corpus_size, size=[batch_size, neg_samples])
	# convert batch data to numpy array
	ngrams = np.array(ngrams)
	# return batch data in the form: (ngrams, true labels, negative labels)
	return ngrams, rand_docs_idx, negative_labels


def read_ohsu_queries(query_path):
	"""read query file and return a dict[id] = {title: <string>, desc: <string>}"""
	with open(query_path, 'r') as qf:
		q = qf.read()
	q = [query.split('\n') for query in q.split('\n\n') if query]
	# loop through each query and fill dict
	qdict = dict()
	for query in q:
		qid = query[1].split()[-1]
		qdict[qid] = dict()
		qdict[qid]['title'] = query[2].split('<title>')[1].strip()
		qdict[qid]['desc'] = query[4]
	return qdict


def read_cds_queries(query_path):
	"""read query file and return a dict[id] = {note: <string>, description: <string>, summary: <string>}"""
	with open(query_path, 'r') as qf:
		queries = qf.read()
	# convert queries to xml
	q = ETree.fromstring(queries)
	# loop through each query and fill dict
	qdict = dict()
	for query in q:
		qid = query.attrib['number']
		qdict[qid] = dict()
		# store query versions (i.e. note, description, summary)
		for version in query:
			qdict[qid][version.tag] = version.text.strip()
	return qdict


def get_score(run, qrels, measure):
	"""return np array of scores for a given measure"""
	if "P_" in measure:
		cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + " " + qrels + " " + run
	elif "ndcg_cut" in measure:
		cmd = "./trec_eval/trec_eval -q -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
	else:
		cmd = "./trec_eval/trec_eval -q -m " + measure + " " + qrels + " " + run
	# run trev_eval as a subprocess
	process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
	result = process.stdout.decode('utf-8').split('\n')
	# get scores
	scores = np.array([(score.split('\t')[-2], score.split('\t')[-1]) for score in result if score.split('\t')[0].strip() == measure and score.split('\t')[-2] != 'all'])
	return scores


def get_averaged_measure_score(run, qrels, measure):
	"""return averaged measure score over topics"""
	if "P_" in measure:
		cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + " " + qrels + " " + run
	elif "ndcg_cut" in measure:
		cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + '_' + measure.split('_')[1] + " " + qrels + " " + run
	else:
		cmd = "./trec_eval/trec_eval -m " + measure + " " + qrels + " " + run
	process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
	result = process.stdout.decode('utf-8').split('\n')
	qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
	qscore = qscore.astype(np.float)[0]
	return qscore


def get_averaged_inferred_measure_score(run, qrels, measure):
    """return averaged measure score over topics"""
    cmd = "perl sample_eval.pl " + qrels + " " + run
    process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    result = process.stdout.decode('utf-8').split('\n')
    qscore = np.array([score.split('\t')[-1] for score in result if score.split('\t')[0].strip() == measure])
    qscore = qscore.astype(np.float)[0]
    return qscore


def evaluate(measures, ranking_folder, ranking_name, qrels_folder, qrels_name):
	"""evaluate models on given measures"""
	scores = []
	print('evaluate model ranking')
	if type(measures) == list:  # list of measures provided
		for measure in measures:
			scores.append(get_averaged_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure)) 
	else:  # single measure provided
		scores.append(get_averaged_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
	print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
	return {measure: score for measure, score in zip(measures, scores)}


def evaluate_inferred(measures, ranking_folder, ranking_name, qrels_folder, qrels_name):
	"""evaluate models on given inferred measures"""
	scores = []
	print('evaluate model ranking')
	if type(measures) == list:  # list of inferred measures provided
		for measure in measures:
			scores.append(get_averaged_inferred_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measure))
	else:  # single inferred measure provided
		scores.append(get_averaged_inferred_measure_score(ranking_folder + '/' + ranking_name + '.txt', qrels_folder + '/' + qrels_name + '.txt', measures))
	print("\t".join([measure + ": " + str(score) for measure, score in zip(measures, scores)]))
	return {measure: score for measure, score in zip(measures, scores)}
				