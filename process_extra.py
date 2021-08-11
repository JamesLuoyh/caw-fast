import argparse
import pandas as pd
from tqdm import tqdm
import sys

def preprocess(data_name):
    u_list, i_list, ts_list, label_list, e_from_u, e_from_i, u_start, i_start = [], [], [], [], [], [], [], []
    idx_list = []
    neighbors = {}
    u_ngh_n = []
    i_ngh_n = []
    i_post_n = []
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in tqdm(enumerate(f)):
            e = line.strip().split(',')
            u = int(e[1])
            i = int(e[2])
            if u not in neighbors:
                neighbors[u] = [u]
            if i not in neighbors:
                neighbors[i] = [i]
            u_ngh_n.append(len(neighbors[u]))
            i_ngh_n.append(len(neighbors[i]))
            if i not in neighbors[u]:
                neighbors[u].append(i)
            if u not in neighbors[i]:
                neighbors[i].append(u)
            i_post_n.append(len(neighbors[i]))
            ts = float(e[3])
            label = int(e[4])
                        
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
    max_idx = max(max(u_list), max(i_list))
    
    curr_idx = 1 # reserve 0 for null
    start_idx = {}
    for i in range(max_idx + 1):
        if i in neighbors:
            start_idx[i] = curr_idx
            curr_idx += len(neighbors[i])


    for idx in range(len(u_list)):
        u = u_list[idx]
        i = i_list[idx]
        u_idx = start_idx[u]
        i_idx = start_idx[i]
        u_start.append(u_idx)
        i_start.append(i_idx)
        e_from_u.append(u_idx + neighbors[u].index(i))
        e_from_i.append(i_idx + neighbors[i].index(u))

    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list,
                         'e_from_u': e_from_u,
                         'e_from_i': e_from_i,
                         'u_start': u_start,
                         'i_start': i_start,
                         'u_ngh_n': u_ngh_n,
                         'i_ngh_n': i_ngh_n,
                         'i_post_n': i_post_n # used for negative examples
                        })

def run(args):
    data_name = args.dataset
    PATH = './processed/ml_{}.csv'.format(data_name)
    OUT_DF = './processed/ml2_{}.csv'.format(data_name)
    print('preprocess {} dataset ...'.format(data_name))
    out = preprocess(PATH)

    out.to_csv(OUT_DF)

parser = argparse.ArgumentParser('Interface for second round of propressing csv source data for CATAW framework')
parser.add_argument('--dataset', choices = ['wikipedia', 'reddit', 'socialevolve', 'uci', 'enron', 'socialevolve_1month', 'socialevolve_2weeks'], 
                   help='specify one dataset to preprocess')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
run(args)