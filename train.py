import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str,default='LASTFM', help='dataset name: LASTFM MOOC WIKITALK WIKIPEDIA REDDIT STACKOVERFLOW BITCOIN GDELT')
parser.add_argument('--model_name', type=str, default='TGAT', help='name of stored model, such as TGAT, TGN, DySAT')
parser.add_argument('--neighbor_num', type=lambda x: [int(i) for i in x.split(',')], default=[10, 10], help='control layer and sample Num, usage --neighbor_num "10,20"  ')
parser.add_argument('--batch_size', type=int, default=2000, help='path to config file')
parser.add_argument('--epochNum', type=int, default=1, help='manually set ,do not use the config file')

parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
# Sampler technique
parser.add_argument('--threadNum', type=int, default=8, help='used in parallel sampler by TGL')
parser.add_argument('--thread_bind', type=int, default=4, help='rebind omp thread, 0 for node0, 1 for node 1 ,2 for node{0,1}, 3 for dynamic, 4 for static')
# IO technique 
parser.add_argument('--slim_trans', type=int, default=2, help='the slim schedule, 2 for slim trans with CT')
parser.add_argument('--cache_method', type=int, default=2, help='the cache schedule, 2 for Greedy selection')
parser.add_argument("--cache_ratio",type=float,default=1,help='the cache rate for all node and edge, set 1 for full cache, auto reduce when vdram not enough')
parser.add_argument('--csr_out', type=int, default=1, help='use the CT default')
# compute technique
parser.add_argument('--fast_esm', type=int, default=1, help='the technique in fast edge-softmax.')
parser.add_argument('--fast_agg', type=int, default=1, help='the technique in fast aggegrate.')
parser.add_argument('--tqdm_on', type=int, default=1, help='open the tqdm for better visualization of training process')
# deperated
parser.add_argument('--config', type=str,default='', help='(deperated) config by model_name and neighbor_num')

args=parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# model and layer
if len(args.neighbor_num)==1:
    args.config = f"config/{args.model_name}-1.yml"
else:
    args.config = f"config/{args.model_name}.yml"

# forward manager and CT engine control
from fast_util.topologyAware import calculate_similarity, getBlossomBindMode
from fast_util.slimCache import *
from fast_util.helper import *
from fast_util.efficientGOP import fm
fm.init_gop(args.fast_agg, args.fast_esm, args.csr_out)

import torch
import time
import random
import dgl
import numpy as np
from modules import *
    
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

class NegLinkSampler:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)

def deviceInfo():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        props = torch.cuda.get_device_properties(current_device)
        
        print(f"Device: {current_device}: {device_name}",end=" ")
        print(f"with: {props.total_memory / 1024**3:.2f} GB.")
    else:
        print("CUDA is not available. Running on CPU.")

set_seed(42)
deviceInfo()

# graph message
g, df = load_graph(args.data)
allnodeNum = g['indptr'].shape[0] - 1
allEdgeNum = g['eid'].shape[0]
node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
rawFeats=[node_feats, edge_feats]
allnodeFeatSize = allnodeNum * node_feats.shape[1] * 4 / 1024 / 1024
allEdgeFeatSize = allEdgeNum * edge_feats.shape[1] * 4 / 1024 / 1024
print("Sum: {:.4f} GB, node: {:.4f} GB, edge: {:.4f} GB".format((allnodeFeatSize+allEdgeFeatSize)/1024,allnodeFeatSize/1024,allEdgeFeatSize/1024))

# model config
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
sample_param['neighbor'] = args.neighbor_num
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
CSR_CONFIG = [1,1,1] if args.csr_out and args.cache_ratio!=1 else []

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]
    
gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()
sampler = None

if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              args.threadNum, 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        batch_groups = eval_df.groupby(eval_df.index // args.batch_size)
        total_batches = len(eval_df) // args.batch_size + 1

        if args.tqdm_on:
            iterator = tqdm(batch_groups, total=total_batches, 
                        desc=f"val batches (size={args.batch_size})", leave=False)
        else:
            iterator = batch_groups
        for _, rows in iterator:
            sampler.configOMP(args.threadNum,2,[]) # val and test no rebind
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end], CSR_CONFIG)
                else:
                    sampler.sample(root_nodes, ts, CSR_CONFIG)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                fm.cntE=[]
                if args.csr_out and (args.fast_esm):
                    for idx,_ in enumerate(ret):
                        inv_idx = len(ret)-1-idx
                        fm.cntE.append(torch.from_numpy(ret[inv_idx].cntE()).cuda())

                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

if not os.path.isdir('models'):
    os.mkdir('models')
if args.model_name == '':
    path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
else:
    path_saver = 'models/{}.pkl'.format(args.model_name)
best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // args.batch_size))
if 'reorder' in train_param:
    # random chunk shceduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), args.batch_size // reorder)
    group_idx = np.tile(group_idx, train_edge_end // args.batch_size + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(args.batch_size // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])

# args.epochNum = train_param['epoch']
for e in range(args.epochNum):
    total_loss = 0
    useNum = 2
    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    # slim cache config
    if e ==0:
        print(args)
    if args.cache_method or args.slim_trans:
        if e == 0:
            cacheConfig = subGraphCacheConfig(args, args.data, args.cache_ratio, args.slim_trans, args.cache_method)
            args.cache_ratio = cacheConfig.init_cache(rawFeats)

    group_key = group_indexes[random.randint(0, len(group_indexes) - 1)]
    grouped = df[:train_edge_end].groupby(group_key)
    grouped_data = df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)])
    if args.tqdm_on:
        grouped_list = list(grouped_data)
        iterator = tqdm(grouped_list, 
                        total=len(grouped_list),
                        desc=f"Processing by {group_key}", 
                        leave=False)
    else:
        iterator = grouped_data
    
    if(args.thread_bind==3):# dynamic generate by pre-analyse
        output_filename = f"./DATA/{args.data}/{args.batch_size}/bind/{args.threadNum}/dynamic_{args.threadNum}.npy"
        loaded_data = np.load(output_filename)
    elif(args.thread_bind==4):# static TA, generate once and reuse
        output_common = f"./DATA/{args.data}/{args.batch_size}/bind/{args.threadNum}/static_{args.threadNum}.npy"
        if os.path.exists(output_common):
            loaded_data = np.load(output_common)[0]
        else:
            # online generate
            simList = np.zeros((args.threadNum, args.threadNum), dtype=np.int64)
            for itr, rows in iterator:
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                simList += calculate_similarity(root_nodes, args.threadNum)
            loaded_data = getBlossomBindMode(simList, args.threadNum)
            all_bind_modes=[]
            all_bind_modes.append(loaded_data)
            if all_bind_modes:
                os.makedirs(os.path.dirname(output_common), exist_ok=True)
                all_bind_modes_array = np.vstack(all_bind_modes)
                np.save(output_common, all_bind_modes_array)
    for itr, rows in iterator:
    # for itr, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]): 
        if(args.thread_bind==3):
            npBindMode = loaded_data[itr]
        elif(args.thread_bind==4):
            npBindMode = loaded_data
        else:
            npBindMode = []
        timeStack.time_push("t_tot")
        sampler.configOMP(args.threadNum,3 if args.thread_bind == 4 else args.thread_bind,npBindMode)
        timeStack.time_push("t_sample")

        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        # import pdb; pdb.set_trace()
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = root_nodes.shape[0] * 2 // 3
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end], CSR_CONFIG)
            else:
                sampler.sample(root_nodes, ts, CSR_CONFIG)
            ret = sampler.get_ret()
        if itr==0: timeStack.timeDict["t_engine"] = 0
        timeStack.timeDict["t_engine"] += ret[0].coo_time() # engine time spilt from sample stage
        # import pdb; pdb.set_trace()
        timeStack.time_pop() # "t_sample"
        timeStack.timeDict["t_sample"] -= ret[0].coo_time() # engine time spilt from sample stage

        useNum -= 1

        timeStack.time_push("t_prep")
        if gnn_param['arch'] != 'identity':
            fm.cntE=[]
            if args.csr_out and (args.fast_esm):
                for idx,_ in enumerate(ret):
                    inv_idx = len(ret)-1-idx
                    fm.cntE.append(torch.from_numpy(ret[inv_idx].cntE()).cuda())
            
        if  (args.cache_method or args.slim_trans) and ( args.cache_ratio!=1 or "TGN" in args.config):
            # 2 block
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            dstFeat=[]
            sample_id_t, ctData = mfgs_fetch(ret, mfgs, sample_param['history'], sample_param['layer'], 1 if cacheConfig.slimEdgeFlag!=0 and cacheConfig.cache_ratio==0 else 0) 
            for i in range(sample_param['history']):
                dstFeat.append(cacheConfig.trans2Device(None, rawFeats, sample_id_t[i], ctData[i] if len(ctData)>0 else None))
            mfgs = prepare_input_slim_cache(mfgs, dstFeat)
            if mailbox is not None:
                mailbox.prep_input_mails_sc(mfgs[0],cacheConfig.node_uni_inv)
            cacheConfig.node_uni_inv=None
        else:
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
        timeStack.time_pop("t_prep")

        useNum -= 1
        timeStack.time_push("t_forward")
        optimizer.zero_grad()
        pred_pos, pred_neg = model(mfgs)
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * args.batch_size
        timeStack.time_pop() # "t_forward"

        timeStack.time_push("t_backward")
        loss.backward()
        optimizer.step()
        
        timeStack.time_pop() # "t_backward"

        timeStack.time_push("t_prep")
        if mailbox is not None:
            mem_edge_feats = edge_feats[rows['Unnamed: 0'].iloc[0]:rows['Unnamed: 0'].iloc[-1]+1]
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
        timeStack.time_pop() # "t_prep"
        timeStack.time_pop() # "t_tot"
    
    ap, auc = eval('val')
    if e > 2 and ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
    print('Epoch {:d}:'.format(e))
    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
    timeStack.save_print_time(pFlag=True)

timeStack.print_avg_stats()
print('Loading model at epoch {}...'.format(best_e))

# test
# model.load_state_dict(torch.load(path_saver))
# model.eval()
# if sampler is not None:
#     sampler.reset()
# if mailbox is not None:
#     mailbox.reset()
#     model.memory_updater.last_updated_nid = None
#     eval('train')
#     eval('val')
# ap, auc = eval('test')
# if args.eval_neg_samples > 1:
#     print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
# else:
#     print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
