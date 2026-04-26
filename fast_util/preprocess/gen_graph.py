import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

t_start = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='LASTFM', help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')
args = parser.parse_args()

df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)

ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

src_array = df['src'].values.astype(np.int32)
dst_array = df['dst'].values.astype(np.int32)
time_array = df['time'].values.astype(np.float32)
idx_array = np.arange(len(df), dtype=np.int32)

for idx in tqdm(range(len(df)), total=len(df), desc='Building adjacency'):
    src = src_array[idx]
    dst = dst_array[idx]
    
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(float(time_array[idx]))
    ext_full_eid[src].append(int(idx_array[idx]))
    
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(float(time_array[idx]))
        ext_full_eid[dst].append(int(idx_array[idx]))

cumulative_sum = 0
for i in tqdm(range(num_nodes), desc='Calculating indptr'):
    ext_full_indptr[i] = cumulative_sum
    cumulative_sum += len(ext_full_indices[i])
ext_full_indptr[num_nodes] = cumulative_sum

print('Flattening arrays...')
ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)), dtype=np.int32)
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)), dtype=np.float32)
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)), dtype=np.int32)

print('Sorting by timestamp...')

def sort_all_nodes(indptr, indices, ts, eid):
    """Sort the adjacency lists of all nodes by timestamp"""
    sorted_indices = np.empty_like(indices)
    sorted_ts = np.empty_like(ts)
    sorted_eid = np.empty_like(eid)
    
    for i in tqdm(range(num_nodes), desc='Sorting nodes'):
        beg = indptr[i]
        end = indptr[i + 1]
        if end > beg:
            sidx = np.argsort(ts[beg:end])
            sorted_indices[beg:end] = indices[beg:end][sidx]
            sorted_ts[beg:end] = ts[beg:end][sidx]
            sorted_eid[beg:end] = eid[beg:end][sidx]
    
    return sorted_indices, sorted_ts, sorted_eid

ext_full_indices, ext_full_ts, ext_full_eid = sort_all_nodes(
    ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid
)

print('Saving...')
ext_full_indptr = ext_full_indptr.astype(np.int32)
ext_full_indices = ext_full_indices.astype(np.int32)
ext_full_ts = ext_full_ts.astype(np.float32)
ext_full_eid = ext_full_eid.astype(np.int32)

np.savez('DATA/{}/ext_full.npz'.format(args.data), 
         indptr=ext_full_indptr, 
         indices=ext_full_indices, 
         ts=ext_full_ts,
         eid=ext_full_eid)

print("Cost Time {:.4f} s".format(time.perf_counter() - t_start))