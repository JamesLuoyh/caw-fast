import argparse
import pandas as pd
from tqdm import tqdm
import sys
import random
import numpy as np

def generate_idx(i, u, ts, neighbors_idx_keeper, neighbors_idx_pointer, neighbors_recency_keeper, max_nghs):
    if i in neighbors_idx_keeper[u]:
        i_in_u = np.where(neighbors_idx_keeper[u] == i)[0][0]
    else:
        # i_in_u = neighbors_idx_pointer[u]
        # neighbors_idx_pointer[u] = (neighbors_idx_pointer[u] + 1) % max_nghs
        i_in_u = np.argmin(neighbors_recency_keeper[u])
        neighbors_idx_keeper[u, i_in_u] = i
    neighbors_recency_keeper[u, i_in_u] = ts
    return i_in_u

def preprocess(data_name, max_nghs, seed, replace='random'):
    u_list, i_list, ts_list, label_list, e_from_u, e_from_i, u_start, i_start = [], [], [], [], [], [], [], []
    idx_list = []
    neighbors = {}
    u_ngh_n = []
    i_ngh_n = []
    i_post_n = []
    max_nghs_and_self = max_nghs + 1

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
            u_ngh_n.append(min(len(neighbors[u]), max_nghs_and_self))
            i_ngh_n.append(min(len(neighbors[i]), max_nghs_and_self))
            if i not in neighbors[u]:
                neighbors[u].append(i)
            if u not in neighbors[i]:
                neighbors[i].append(u)
            i_post_n.append(min(len(neighbors[i]), max_nghs_and_self))
            ts = float(e[3])
            label = int(float(e[4]))
                        
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
    max_idx = max(max(u_list), max(i_list))
    
    curr_idx = 1 # reserve 0 for null
    start_idx = {}
    neighbors_idx_keeper = np.zeros((max_idx + 1, max_nghs))
    neighbors_recency_keeper = np.zeros((max_idx + 1, max_nghs))
    count = 0
    for i in range(max_idx + 1):
        if i in neighbors:
            start_idx[i] = curr_idx
            curr_idx += min(len(neighbors[i]), max_nghs_and_self)
            if len(neighbors[i]) <= 2:
                count += 1

    neighbors_idx_pointer = np.zeros(max_idx + 1, dtype = int)
    for idx in range(len(u_list)):
        u = u_list[idx]
        i = i_list[idx]
        u_idx = start_idx[u]
        i_idx = start_idx[i]
        ts = ts_list[idx]
        u_start.append(u_idx)
        i_start.append(i_idx)
        if replace == 'oldest':
            i_in_u = generate_idx(i, u, ts, neighbors_idx_keeper, neighbors_idx_pointer, neighbors_recency_keeper, max_nghs)
            u_in_i = generate_idx(u, i, ts, neighbors_idx_keeper, neighbors_idx_pointer, neighbors_recency_keeper, max_nghs)
            e_from_u.append(i_in_u + 1)
            e_from_i.append(u_in_i + 1)
        else: 
            i_in_u = neighbors[u].index(i)
            u_in_i = neighbors[i].index(u)
            if i_in_u >= max_nghs_and_self:
                i_in_u = (hash(i_in_u) ^ seed) % max_nghs + 1
            if u_in_i >= max_nghs_and_self:
                u_in_i = (hash(u_in_i) ^ seed) % max_nghs + 1
            e_from_u.append(i_in_u)
            e_from_i.append(u_in_i)
        

    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list,
                         'src': u_list,
                         'tgt': i_list,
                         'e_from_u': e_from_u,
                         'e_from_i': e_from_i,
                         'u_start': u_start,
                         'i_start': i_start,
                         'u_ngh_n': u_ngh_n,
                         'i_ngh_n': i_ngh_n
                        })

def run(args):
    data_name = args.dataset
    PATH = './processed/ml_{}.csv'.format(data_name)
    OUT_DF = './processed/ml2_{}.csv'.format(data_name)
    print('preprocess {} dataset ...'.format(data_name))
    out = preprocess(PATH, args.max_neighbors, args.seed, args.replace)

    out.to_csv(OUT_DF)

parser = argparse.ArgumentParser('Interface for second round of propressing csv source data for CATAW framework')
parser.add_argument('--dataset', choices = ['wikipedia', 'reddit', 'socialevolve', 'uci', 'enron', 'socialevolve_1month', 'socialevolve_2weeks', 'sx-superuser'], 
                   help='specify one dataset to preprocess')
parser.add_argument('--max_neighbors', type=int, default=32, help='number of neighbors to store per node')
parser.add_argument('--seed', type=int, default=1, help='seed for randomization of neighborhood store index')
parser.add_argument('--replace', choices = ['random', 'oldest'], default=random,
                   help='specify one replacement mechanism when max neighbors reached')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
run(args)