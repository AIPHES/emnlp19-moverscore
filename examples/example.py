#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 02:01:42 2021

@author: zhao
"""
from typing import List, Union, Iterable
from itertools import zip_longest
import sacrebleu
from moverscore_v2 import word_mover_score
from collections import defaultdict
import numpy as np
def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score

def corpus_score(sys_stream: List[str],
                     ref_streams:Union[str, List[Iterable[str]]], trace=0):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    fhs = [sys_stream] + ref_streams

    corpus_score = 0
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
            
        hypo, *refs = lines
        corpus_score += sentence_score(hypo, refs, trace=0)
        
    corpus_score /= len(sys_stream)

    return corpus_score

def test_corpus_score():
    
    refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
            ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    
    bleu = sacrebleu.corpus_bleu(sys, refs)
    mover = corpus_score(sys, refs)
    print(bleu.score)
    print(mover)
    
def test_sentence_score():
    
    refs = ['The dog bit the man.', 'The dog had bit the man.']
    sys = 'The dog bit the man.'
    
    bleu = sacrebleu.sentence_bleu(sys, refs)
    mover = sentence_score(sys, refs)
    
    print(bleu.score)
    print(mover)

if __name__ == '__main__':   

    test_sentence_score()
    
    test_corpus_score()
    
        
