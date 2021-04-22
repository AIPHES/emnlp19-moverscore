from __future__ import division
import os
import json
import codecs
import copy

from moverscore import get_idf_dict, word_mover_score
#from moverscore_v2 import get_idf_dict, word_mover_score, plot_example

BASE_FOLDER = "data"
name = "tac.09.mds.gen.resp-pyr"

def load_json(filename):
    filepath = os.path.join(BASE_FOLDER, filename)
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        return json.loads(f.read())
    
def normalize_responsiveness(dataset):
    max_resp = 0.
    for k,v in dataset.items():
        for annot in v['annotations']:
            if annot['responsiveness'] > max_resp:
                max_resp = annot['responsiveness']
    for k,v in dataset.items():
        for annot in v['annotations']:
            annot['responsiveness'] /= float(max_resp)
    return dataset

tac_09_mds_gen_resp_pyr = normalize_responsiveness(load_json(name))

def merge_datasets(lst_datasets):
    merged_dataset = {}
    for dataset in lst_datasets:
        merged_dataset.update(copy.deepcopy(dataset))
    return merged_dataset

import pprint
pp = pprint.PrettyPrinter(indent=4)
import numpy as np
def print_average_correlation(corr_mat):
    corr_mat = np.array(corr_mat)   
    results = dict(zip(['kendall','pearson', 'spearman'], 
                       [np.mean(corr_mat[:,0]), 
                       np.mean(corr_mat[:,1]),
                       np.mean(corr_mat[:,2])]))
    pp.pprint(results)
    
resp_data = merge_datasets([tac_09_mds_gen_resp_pyr])
pyr_data = merge_datasets([tac_09_mds_gen_resp_pyr])        
    
pyr_data = dict(list(pyr_data.items()))
resp_data = dict(list(resp_data.items()))

human_scores = ['pyr_score', 'responsiveness']
dataset = [list(pyr_data.items()), list(resp_data.items())]

with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = set(f.read().strip().split(' '))

import scipy.stats as stats
from tqdm import tqdm 

def micro_averaging(dataset, target, device='cuda:0'):
    references, summaries = [], []
    for topic in dataset:
        k,v = topic
        references.extend([' '.join(ref['text']) for ref in v['references']])
        summaries.extend([' '.join(annot['text']) for annot in v['annotations']])
 
    idf_dict_ref = get_idf_dict(references)
    idf_dict_hyp = get_idf_dict(summaries)

    correlations = []
    for topic in tqdm(dataset): 
        k,v = topic
        references = [' '.join(ref['text']) for ref in v['references']]
        num_refs = len(references)
        target_scores, prediction_scores = [], []      

        for annot in v['annotations']:            
            if len(annot['text']) > 1:
                target_scores.append(float(annot[target]))

                scores = word_mover_score(references, [' '.join(annot['text'])] * num_refs, idf_dict_ref, idf_dict_hyp, stop_words,
                                        n_gram=1, remove_subwords=True, batch_size=48)

                prediction_scores.append(np.mean(scores))

        correlations.append([
                         stats.kendalltau(target_scores, prediction_scores)[0], 
                         stats.pearsonr(target_scores, prediction_scores)[0],
                         stats.spearmanr(target_scores, prediction_scores)[0]])
    return np.array(correlations)


for i in range(len(human_scores)):
    print(human_scores[i])
    bert_corr = micro_averaging(dataset[i], human_scores[i], device='cuda:0')
    print_average_correlation(bert_corr)
    
    
