import math
import subprocess
import glob
import numpy as np
import xml.etree.ElementTree as ETree

from tqdm import tqdm


def load_stopwords(stopwords_path):
        """read stopwords file into list"""
        with open(stopwords_path, 'r') as sl:
            stop_words = [stop.strip() for stop in sl]
        return stop_words


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
    scores = np.array([(score.split('\t')[-2], score.split('\t')[-1]) for score in result 
        if score.split('\t')[0].strip() == measure and score.split('\t')[-2] != 'all'])
    return scores


def get_averaged_measure_score(run, qrels, measure):
    """return averaged measure score over topics"""
    if "P_" in measure:
        cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + " " + qrels + " " + run
    elif "recall_" in measure:
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
    return [(measure, score) for measure, score in zip(measures, scores)]


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
