
MoverScore ([Zhao et.al, 2019](https://arxiv.org/pdf/1909.02622.pdf)) provides evaluation metrics for text generation tasks such as machine translation, summarization, etc. It achieves high correlation with human judgments and can be considered a successor of the BLEU score metrics.

# Overview

MoverScore is a simply and easy-to-use evaluation metric, which combines contextualized word embeddings from BERT finetuned on MNLI and set-based similarity criterion like Earth Mover Distance, as illustrated below:

<p align="center">
<img src="./MoverScore.png">
</p>

# QUICK START

Install the Python module (Python 3 only)

    pip3 install moverscore

# Using MoverScore for Evaluating Machine Translation

### MoverScore Specification

```bash
scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, \
                          n_gram=1, remove_subwords=True)
```
| Parameters       | Description                        |
|----------------|----------------------------|
| references       | a list of reference texts      |
| translations     | a list of system translation texts            |
| idf_dict_ref     | idf dictionary extracted from the reference corpus | 
| idf_dict_hyp     | idf dictionary extracted from the system hypothesis corpus | 
| n_gram           | unigram-based MoverScore (n-gram=1), bigram-based MoverScore (n-gram=2) | 
| remove_subwords  | if the subwords (verb tense) like 'ING/ED' need to be removed | 

### The Results in Machine Translation

System                  | cs-en | de-en | ru-en | tr-en | zh-en
----------------------- | :------: | :----------: | :------: | :------: | :------:
SentBLEU                | 0.435 | 0.432 | 0.484 |  0.538 | 0.512 
RUSE(supervised metric) | 0.624 | 0.644 | 0.673 | 0.716 | 0.691 
BERTScore               | 0.670 | 0.686 | 0.729 | 0.714 | 0.704 
WMD-1+BERTMNLI+PMeans   | 0.670    | 0.708     | **0.738** | 0.762| **0.744**
WMD-2+BERTMNLI+PMeans   | **0.679** | **0.710**     | 0.736 | **0.763**| 0.740

This repo knows the dataset in WMT17 and handles downloading & preprocessing silently. 

Obtain the results in WMT17 with one line code:

```bash
python examples/run_MT.py
```

# Reference
If you find our source code useful, please consider citing our work.
```
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
```
