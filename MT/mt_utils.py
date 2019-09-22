import glob
from io import StringIO
import os

import pandas as pd
from scipy.stats import pearsonr
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    return "pearson: " + str(pearson_corr)

def load_data(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines

def load_metadata(lp):
    files_path = []
    for root, directories, files in os.walk(lp):
        for file in files:
            if '.hybrid' not in file:
                raw = file.split('.')
                testset = raw[0]
                lp = raw[-1]
                system = '.'.join(raw[1:-1])
                files_path.append((os.path.join(root, file), testset, lp, system))
    return files_path

def output_MT_correlation(lp_set, eval_metric, f = "DA-seglevel.csv"):    
    lines = [line.rstrip('\n') for line in open(f)]
    lines.pop(0)
    manual = {}
    for l in lines:
        l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")
        c = l.split()
     
        lp, data, system, sid, score = c[0], c[1], c[2], c[3], c[4]    
        c = system.split("+")
        system = c[0]
    
        if lp not in manual:
            manual[lp] = {}
        if system not in manual[lp]:
            manual[lp][system+"::"+sid] = score
     
    missing = 0
    met_names = {}
    lms = {}
    lsm = {}
    
    submissions = ["*.seg.score"]
    for s in submissions:
        files = glob.glob(s)
        for f in files:
            lines = [line.rstrip('\n') for line in open(f)]
            for l in lines:
                l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")                       
                if (l.find("hybrid")==-1) and (l.find("himl")==-1): 
                    c = l.split()
                    if len(c) < 6:
                        missing = missing + 1   
                    else:
                        metric, lp, data, system, sid, score = c[0], c[1], c[2], c[3], c[4], c[5]                                            
                        system = system + "::" + sid
                        if lp not in lms:
                            lms[lp] = {}
                        if metric not in lms[lp]:
                            lms[lp][metric] = {}
                        if system not in lms[lp][metric]:
                            lms[lp][metric][system] = score 
                        if lp not in lsm:
                            lsm[lp] = {}
                        if system not in lsm[lp]:
                            lsm[lp][system] = {}
                        if system not in lsm[lp][system]:
                            lsm[lp][system][metric] = score 
    for lp in manual:
        if lp not in lp_set: continue
        for metric in lms[lp]:
            allthere = True
            for trans in manual[lp]:
                if not trans in lms[lp][metric]:
                    allthere = False
                    print (lp+" "+metric+" "+trans)
            if allthere:  
                if lp not in met_names:
                    met_names[lp] = {}
            if metric not in met_names[lp]:
                met_names[lp][metric] = 1
            else:
                print ("segment mismatch "+lp+" "+metric)                
    for lp in manual:
        if lp not in lp_set: continue
    
        s = "LP SYSTEM HUMAN"
        for metric in sorted(met_names[lp]):  
            s = s+" "+metric
        s = s+"\n"
        for system in manual[lp]:
            s = s+lp+" "+system+" "+manual[lp][system]    
            for metric in sorted(met_names[lp]):
                s = s +" "+lsm[lp][system][metric]
            s = s+"\n"        
        results = pd.read_csv(StringIO(s), sep=" ")    
        print(lp +"\t" + pearson_and_spearman(results['HUMAN'], results[eval_metric]))

