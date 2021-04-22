import tqdm
import pandas as pd
from mosestokenizer import MosesDetokenizer
from mt_utils import load_data, load_metadata, output_MT_correlation
from moverscore import get_idf_dict, word_mover_score
#from moverscore_v2 import get_idf_dict, word_mover_score, plot_example
import os 

USERHOME = os.path.expanduser("~")
MOVERSCORE_DIR = os.environ.get('MOVERSCORE', os.path.join(USERHOME, '.moverscore'))
data_dir = os.path.join(MOVERSCORE_DIR, 'MT')

reference_list = dict({
        "newstest2017-csen-ref.en": "cs-en",
        "newstest2017-deen-ref.en": "de-en",
        "newstest2017-ruen-ref.en": "ru-en",
        "newstest2017-tren-ref.en": "tr-en",
        "newstest2017-zhen-ref.en": "zh-en"
        })
#from collections import defaultdict
metric = 'MoverScore'

data = []
for _ in reference_list.items():
    reference_path, lp = _
    references = load_data(os.path.join(data_dir, reference_path))
    with MosesDetokenizer('en') as detokenize:
        references = [detokenize(ref.split(' ')) for ref in references]

    idf_dict_ref = get_idf_dict(references) #defaultdict(lambda: 1.)
    
    all_meta_data = load_metadata(os.path.join(data_dir, lp))
    for i in tqdm.tqdm(range(len(all_meta_data))):
        path, testset, lp, system = all_meta_data[i]
        translations = load_data(path)        
        with MosesDetokenizer('en') as detokenize:
            translations = [detokenize(hyp.split(' ')) for hyp in translations]
        idf_dict_hyp = get_idf_dict(translations)
        
        df_system = pd.DataFrame(columns=('metric', 'lp', 'testset', 'system', 'sid', 'score'))
        scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True,
                                      batch_size=64)
        num_samples = len(references)
        df_system = pd.DataFrame({'metric': [metric] * num_samples,
                               'lp': [lp] * num_samples,
                               'testset': [testset] * num_samples,
                               'system': [system] * num_samples,
                               'sid': [_ for _ in range(1, num_samples + 1)],
                               'score': scores,
                             })
        data.append(df_system) 

results = pd.concat(data, ignore_index=True)
results.to_csv(metric + '.seg.score', sep='\t', index=False, header=False)
output_MT_correlation(lp_set=list(reference_list.values()), eval_metric=metric)

reference = 'they are now equipped with air conditioning and new toilets.'
translation = 'they have air conditioning and new toilets.'
plot_example(True, reference, translation)
