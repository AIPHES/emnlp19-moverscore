<h1 align="left">Evaluation-as-Service</h1>

<h2 align="left">What is MoverScore and EvalSerivce</h2>

**MoverScore** measures semantic distance between system and reference texts by aligning semantically similar words and finding the corresponding travel costs.

**EvalSerivce** is a evaluation framework for NLG tasks, assigning scores (e.g., ROUGE ans MoverScore) to system-generated text by comparing it against human references for content matching.

<h2 align="left">Installation</h2>

Install the server and client via `pip`. They can be installed separately or even on *different* machines:
```bash
cd server/
python3 setup.py install # server
cd client/
python3 setup.py install # client
```

Note that the server MUST be running on **Python >= 3.5**. Again, the server does not support Python 2!

The client can be running on both Python 2 and 3 [for the following consideration](#q-can-i-run-it-in-python-2).

<h2 align="left">Getting Started</h2>

#### 1. Start the evaluation service
After installing the server, you should start a serivce as follows:
```bash
summ-eval-start -data_dir ../../ -num_worker=4
```
This will start the service with four workers, meaning that it can handle up to four **concurrent** requests.

#### 2. Use Client to Get Evaluation scores
Now you can get scores:
```python
from summ_eval.client import EvalClient
ec = EvalClient()

system = ['A guy with a read jacket is standing on a boat']
references = ['A man wearing a lifevest is sitting in a canoe','A small white ferry rides through water']

example_1 = [system, references, 'rouge_1']
example_2 = [system, references, 'rouge_2']
example_3 = [system, references, 'wmd_1'] # BERTWordMover-unigram
example_4 = [system, references, 'wmd_2'] # BERTWordMover-bigram
example_5 = [system, references, 'smd'] # BERTSentMover

ec.eval([example_1,example_2,example_3,example_4,example_5])
```
