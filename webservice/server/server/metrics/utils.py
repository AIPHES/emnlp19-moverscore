#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
stopset = frozenset(stopwords.words('english'))

def stem_word(word):
    return stemmer.stem(normalize_word(word))


def normalize_word(word):
    return word.lower()


def get_len(element):
    return len(tokenizer.tokenize(element))


def sentence_tokenizer(sentence):
    return tokenizer.tokenize(sentence)


def get_ngrams(sentence, N):
    tokens = tokenizer.tokenize(sentence.lower())
    clean = [stemmer.stem(token) for token in tokens]
    return [gram for gram in ngrams(clean, N)]


def get_words(sentence, stem=True):
    if stem:
        words = [stemmer.stem(r) for r in tokenizer.tokenize(sentence)]
        return [normalize_word(w) for w in words]
    else:
        return [normalize_word(w) for w in tokenizer.tokenize(sentence)]
    
def tokenize(text):
    return [w.lower() for sent in text for w in tokenizer.tokenize(sent) if w not in stopset]

from metrics import ROUGE

def ROUGE_N(summary, references, external_info):
    N = external_info['N']
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return ROUGE.rouge_n(summary, references_text, N, alpha=0)

def ROUGE_L(summary, references, external_info=None):
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return ROUGE.rouge_l(summary, references_text, alpha=0)

def ROUGE_WE(summary, references, external_info):
    N = external_info['N']
    we = external_info['we']
    references_text = []
    for ref in references:
        references_text.append(ref['text'])
    return ROUGE.rouge_n_we(summary, references_text, we, N, alpha=0)
