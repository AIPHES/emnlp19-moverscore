
MoverScore ([Zhao, 2019](http://aclweb.org/anthology/W18-6319)) provides shareable, comparable evaluation metrics for text generation tasks, which achieves high correlation with human judgments and considers to be the successor of BLEU scores.

# QUICK START

Install the Python module (Python 3 only)

    pip3 install moverscore

# Using MoverScore for the MT Evaluation

System                  | cs-en | de-en | ru-en | tr-en | zh-en
----------------------- | :------: | :----------: | :------: | :------: | :------:
RUSE(supervised metric) | 0.624 | 0.644 | 0.673 | 0.716 | 0.691 | 0.685 
BERTScore               | 0.670 | 0.686 | 0.729 | 0.714 | 0.704 | 0.719 
WMD-1+BERTMNLI+PMeans   | 0.670    | 0.708     | **0.738** | 0.762| **0.744**
WMD-2+BERTMNLI+PMeans   | **0.679** | **0.710**     | 0.736 | **0.763**| 0.740

To reproduce the above numbers, please access the MT folder and then download the BERT model finetuned on [MNLI](https://drive.google.com/open?id=1LyWbyMg4CVHktbGPcm2pgtIPeiLg0W0g) before runing experiment.

```bash
python ./run_MT.py
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
