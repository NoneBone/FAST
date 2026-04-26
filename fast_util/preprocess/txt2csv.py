import pandas as pd
import numpy as np
import argparse
import time

t_start = time.perf_counter()

commonDict=""
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default = "BITCOIN")
parser.add_argument('--txt', type=str, help='txt file path', default = commonDict+"DATA/BITCOIN/soc-bitcoin.edges")
args=parser.parse_args()


print('open txt file...')
with open(args.txt, "r") as f:
    data = f.readlines()
for i in range(len(data)):
    data[i] = data[i].split()

print('to DataFrame...')
df = pd.DataFrame(data[1:])
df.columns = ['src','dst','time']

print('reindex...')
#reindex src & dst
src = df.src.astype(int)
dst = df.dst.astype(int)
cut = len(src)
total = np.concatenate((src,dst))
total,inv = np.unique(total,return_inverse=True)

new_data = np.arange(len(total))
df['src'] = new_data[inv][:cut]
df['dst'] = new_data[inv][cut:]

print('sort by time...')
# sort by time
df.time =  df.time.astype(int)
df.time -= df.time.min()
df = df.sort_values(by=['time'])
df = df.reset_index(drop=True)

print('split the data...')
#data split
values = np.zeros(len(df),dtype=int)
values[int(0.7*len(df)):int(0.85*len(df))] = 1
values[int(0.85*len(df)):] = 2
df['ext_roll'] = values

print('save csv...')
df.to_csv(commonDict+'DATA/{}/edges.csv'.format(args.data))
print("Cost Time {:.4f} s".format(time.perf_counter() - t_start))