import numpy as np
import torch
import random
from utils import *
from sampler_core import ParallelSampler

cacheT = nvtxTimeTable()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class preSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.count = []

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)
    
    def filtRandom(self, n, npData):
        srcData = np.random.randint(self.num_nodes, size=n)
        mask = np.isin(srcData, npData)
        self.count.append(np.sum(mask) / len(npData))
        return srcData[mask]
        # return np.random.randint(self.num_nodes, size=n)
    
    def avg_cnt(self):
        if len(self.count) == 0:
            return 0.0
        
        avg_rate = sum(self.count) / len(self.count)
        return avg_rate

def get_num(data):
    if (data == 'BITCOIN'):
        node_num = 24575383
        edge_num = 122948162
    elif (data == 'WIKITALK'):
        node_num = 1140149
        edge_num = 7833140
    elif (data == 'STACKOVERFLOW'):
        node_num = 2601977
        edge_num = 63497049
    elif (data == 'GDELT'):
        node_num = 16682
        edge_num = 191290883
    elif (data == 'LASTFM'):
        node_num = 1980
        edge_num = 1293103
    elif (data == 'MOOC'):
        node_num = 7144
        edge_num = 411749
    elif (data == 'REDDIT'):
        node_num = 10984
        edge_num = 672447
    elif (data == 'WIKIPEDIA'):
        node_num = 9228
        edge_num = 157474
    return (node_num, edge_num)


def maxinumCacheHitRate(cap, score1, score2):
    """Batch processing based on priority queue """
    n1 = score1.size(0)
    n2 = score2.size(0)
    
    if cap >= n1 + n2:
        return n1, n2
    
    scores_combined = torch.cat([score1, score2])
    indices = torch.cat([torch.zeros(n1, dtype=torch.int32, device=score1.device),
                         torch.ones(n2, dtype=torch.int32, device=score1.device)])
    
    top_k_values, top_k_indices = torch.topk(scores_combined, k=min(cap, scores_combined.size(0)))
    
    top_sources = indices[top_k_indices]
    x = int((top_sources == 0).sum().item())
    y = int((top_sources == 1).sum().item())
    
    return x, y

class subGraphCacheConfig:
    """
    Cache config for subgraph features
    """

    def __init__(self, args, dataName, cache_ratio=0.4, slimEnable=True, cacheEnable=True):
        self.preRate = 0.5
        self.cache_method = 0
        self.node_num,self.edge_num  = get_num(dataName)
        self.node_uni_inv=None
        self.args = args
        self.devID = 'cuda'
        self.cacheEnable = cacheEnable
        self.cache_ratio = cache_ratio

        self.cacheFeats_node = None
        self.cacheFeats_edge = None

        self.uniqueCPUFlag = False
        self.slimEdgeFlag = False
        if(slimEnable==1):
            self.uniqueCPUFlag = False
            self.slimEdgeFlag = False
        elif(slimEnable==2):
            self.uniqueCPUFlag = False
            self.slimEdgeFlag = True
            
        # masks for manage the feature locations: default in CPU
        self.gpu_flag_node = torch.zeros(self.node_num, device=self.devID).bool()
        self.gpu_flag_edge = torch.zeros(self.edge_num, device=self.devID).bool()
        self.gpu_flag_node.requires_grad_(False)
        self.gpu_flag_edge.requires_grad_(False)

        with torch.cuda.device(self.devID):
            self.localid2cacheid_node = torch.zeros(self.node_num, dtype=torch.int32, device=self.devID)
            self.localid2cacheid_edge = torch.zeros(self.edge_num, dtype=torch.int32, device=self.devID)
            self.localid2cacheid_node.requires_grad_(False)
            self.localid2cacheid_edge.requires_grad_(False)

    def get_hot_cts(self, preRate=1.):
        '''
        preRate: if 1: save/load npy, else: regenerate.
        '''
        all_num_node, all_num_edge  = get_num(self.args.data)

        sample_num_node = torch.zeros(all_num_node).cuda()
        sample_num_edge = torch.zeros(all_num_edge).cuda()
        # If the file exists, it will be read; otherwise, resampling will be performed.
        output_common = f"./EXP_ITERATION/00-DATA/{self.args.data}/{self.args.batch_size}/09-hotID/TYPE.npy"
        if preRate==1 and os.path.exists(output_common.replace("TYPE","node_hid")) and os.path.exists(output_common.replace("TYPE","edge_hid")):
            node_hid = torch.from_numpy(np.load(output_common.replace("TYPE","node_hid"), allow_pickle=True))
            edge_hid = torch.from_numpy(np.load(output_common.replace("TYPE","edge_hid"), allow_pickle=True))
            node_sc_ct = torch.from_numpy(np.load(output_common.replace("TYPE","node_sc"), allow_pickle=True)).cuda()
            edge_sc_ct = torch.from_numpy(np.load(output_common.replace("TYPE","edge_sc"), allow_pickle=True)).cuda()

            return [[node_hid], [node_sc_ct]], [[edge_hid], [edge_sc_ct]]
        else:
            # sample
            g, df = load_graph(self.args.data)
            group_indexes = list()
            train_edge_end = int(df[df['ext_roll'].gt(0)].index[0] * preRate)
            group_indexes.append(np.array(df[:train_edge_end].index // self.args.batch_size))
            
            neg_link_sampler = preSampler(g['indptr'].shape[0] - 1)
            sampler = None
            sample_param,b1,b2,b3 = parse_config(self.args.config)
            sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    40, 1, sample_param['layer'], sample_param['neighbor'],
                                    sample_param['strategy']=='recent', sample_param['prop_time'],
                                    sample_param['history'], float(sample_param['duration']))
            
            for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                sampler.configOMP(40,2,[])
                sampler.sample(root_nodes, ts, [])
                ret = sampler.get_ret()
                for idx, r in enumerate(ret):
                    if idx == 0:
                        if len(ret)==1: sample_num_node[torch.from_numpy(r.nodes()).cuda()] += 1 #  when 1 layer  
                        sample_num_edge[torch.from_numpy(r.eid()).cuda()] += 1
                    else:
                        sample_num_node[torch.from_numpy(r.nodes()).cuda()] += 1 
                        sample_num_edge[torch.from_numpy(r.eid()).cuda()] += 1

            node_score_ct, indices_node = torch.sort(sample_num_node,descending=True)
            edge_score_ct, indices_edge = torch.sort(sample_num_edge,descending=True)
            if preRate==1:
                os.makedirs(os.path.dirname(output_common), exist_ok=True)
                arr0 = node_score_ct.cpu().numpy().astype(np.int64)
                arr1 = indices_node.cpu().numpy().astype(np.int64)
                arr2 = edge_score_ct.cpu().numpy().astype(np.int64)
                arr3 = indices_edge.cpu().numpy().astype(np.int64)
                np.save(output_common.replace("TYPE","node_hid"), arr0)
                np.save(output_common.replace("TYPE","node_sc"), arr1)
                np.save(output_common.replace("TYPE","edge_hid"), arr2)
                np.save(output_common.replace("TYPE","edge_sc"), arr3)
            return [[indices_node.cpu()], [node_score_ct]], [[indices_edge.cpu()], [edge_score_ct]]
    
    def init_cache(self, input_feats):
        # get available GPU memory
        peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.devID)
        peak_cached_mem = torch.cuda.max_memory_reserved(device=self.devID)
        total_mem = torch.cuda.get_device_properties(self.devID).total_memory
        available_in_bytes = total_mem - peak_allocated_mem - peak_cached_mem \
                    - 2 * 1024 * 1024 * 1024
        avail_length = int(available_in_bytes / (input_feats[1].size(1) * 4)) # input_feats_bytes = length * dim(1) * sizeof(float32)

        # set capability
        if(self.cache_ratio > 1):
             self.cache_ratio = 1
        scale_n2e = input_feats[0].size(1)/input_feats[1].size(1) # num of feat is based on edge dim
        needed_length = int((self.node_num * scale_n2e + self.edge_num) * self.cache_ratio)
        if (needed_length > avail_length):
            tmpRatio = round((avail_length/needed_length*self.cache_ratio),1)
            print(f"GPU memory is not enough, set cache to {available_in_bytes/(1024 * 1024 * 1024)} GB, equal to cacheRate={tmpRatio}")
            print(f"{avail_length} < {needed_length} = ({self.node_num} * {scale_n2e}+ {self.edge_num}) * {self.cache_ratio}, ")
            self.capability = avail_length
            self.cache_ratio = tmpRatio
        else:
            self.capability = needed_length

        if(self.cache_ratio==1):
            self.cache_method = 9
            self.cacheFeats_node = input_feats[0].cuda(self.devID)
            self.cacheFeats_edge = input_feats[1].cuda(self.devID)
        elif((self.cache_ratio<1)&(self.cache_ratio>0)):
            self.cache_method = 1
            # Presample
            self.node_hot_list,self.edge_hot_list = self.get_hot_cts(self.preRate)
            node_hot_t, node_score = self.node_hot_list[0][0], self.node_hot_list[1][0]
            edge_hot_t, edge_score  = self.edge_hot_list[0][0], self.edge_hot_list[1][0]
            if self.cacheEnable==1:
                # node prim
                if node_score.size()[0]> self.capability:
                    node_length = self.capability
                    edge_length = 0
                else:
                    node_length = node_score.size()[0]
                    edge_length = self.capability - node_length
            elif self.cacheEnable==2:
                # greedy
                edge_length , node_length = maxinumCacheHitRate(self.capability, edge_score, node_score)
            else:
                # avg
                node_length = int(self.capability * (self.node_num/(self.edge_num+self.node_num)))
                edge_length = self.capability - node_length
            
            fetch_nid = node_hot_t[:node_length]
            fetch_eid = edge_hot_t[:edge_length]
            # # length, hit, miss
            # print(f"\tnode, {node_length}, {int(torch.sum(node_score[:node_length]))}, {int(torch.sum(node_score[node_length:]))},"+
            #       f"edge, {edge_length},{int(torch.sum(edge_score[:edge_length]))}, {int(torch.sum(edge_score[edge_length:]))}",end="")
            self.fresh_cache(input_feats, fetch_nid, fetch_eid)
        else:
            self.cache_method = 0
        
        return self.cache_ratio

    def fresh_cache(self, input_feats, fetch_nid, fetch_eid, iteration=0):
        # get input_nodes
        self.cacheFeats_node = torch.index_select(input_feats[0], 0, fetch_nid).to(self.devID)
        self.cacheFeats_edge = torch.index_select(input_feats[1], 0, fetch_eid).to(self.devID)

        if iteration > 0:
            self.localid2cacheid_node.fill_(0)
            self.localid2cacheid_edge.fill_(0)
        self.localid2cacheid_node[fetch_nid] = torch.arange(fetch_nid.size(0),device=self.devID).int()
        self.localid2cacheid_edge[fetch_eid] = torch.arange(fetch_eid.size(0),device=self.devID).int()

        if iteration > 0:
            self.gpu_flag_node.fill_(False)
            self.gpu_flag_edge.fill_(False)
        self.gpu_flag_node[fetch_nid] = True
        self.gpu_flag_edge[fetch_eid] = True

    def trans2Device(self, dstFeat, srcFeat, sample_3ID_t, csrData=None):
        '''
        Build dstFeat (CUDA), using srcFeat (CPU + Cache)
        '''
        cacheT.time_push("t_gpu")
        if dstFeat == None:
            dstFeat=[]
            for i in range(len(sample_3ID_t)):
                sID_t = sample_3ID_t[i]
                dstFeat.append(torch.empty((sID_t.shape[0],srcFeat[0 if i==0 else 1].shape[1]), device='cuda', dtype=torch.float32))
        cacheT.time_pop("")
       
        if self.cache_method == 0:
            # baseline
            if not self.uniqueCPUFlag and not self.slimEdgeFlag:
                cacheT.time_push("t_cpu")
                for i in range(len(sample_3ID_t)):
                    sID_t = sample_3ID_t[i]
                    dstFeat[i] = srcFeat[0 if i==0 else 1][sID_t.to(srcFeat[0].device)].cuda()
                cacheT.time_pop("")
                return dstFeat
            
            cacheT.time_push("t_cpu")
            if self.uniqueCPUFlag:
                sample_ID_t=[]
                inv=[]
                for i in range(len(sample_3ID_t)):
                    sID_t = sample_3ID_t[i]
                    if i==0:
                        if self.node_uni_inv:
                            [_uni, _inv] = self.node_uni_inv
                        else:
                            _uni, _inv=torch.unique(sID_t, return_inverse=True)
                            self.node_uni_inv = [_uni.cpu(), _inv]
                    else: 
                        _uni, _inv=torch.unique(sID_t, return_inverse=True)
                    sample_ID_t.append(_uni)
                    inv.append(_inv)
            # CT engine
            elif self.slimEdgeFlag:
                inv = []
                sample_ID_t = []
                for i in range(len(sample_3ID_t)):
                    sID_t = sample_3ID_t[i]
                    if i==0:
                        _uni, _inv=torch.unique(sID_t, return_inverse=True)
                        sample_ID_t.append(_uni)
                        inv_ = _inv.to(self.devID)
                        inv.append(inv_)
                        self.node_uni_inv = [_uni.cpu(), inv_]
                    else:
                        sample_ID_t.append(torch.from_numpy(csrData[0 if i==1 else 2]))
                        inv.append(torch.from_numpy(csrData[1 if i==1 else 3]).to(self.devID))
            cacheT.time_pop("")
            cacheT.time_push("t_gpu")
            for i in range(len(sample_ID_t)):
                sID_t = sample_ID_t[i]
                dstFeat[i] = srcFeat[0 if i==0 else 1][sID_t].cuda()[inv[i]]
            cacheT.time_pop("")
            return dstFeat

        elif self.cache_method == 9:
            # all cache
            cacheT.time_push("t_cpu")
            cacheT.time_pop("")
            cacheT.time_push("t_gpu")
            for i in range(len(sample_3ID_t)):
                sID_t = sample_3ID_t[i]
                if i==0:
                    dstFeat[i] = self.cacheFeats_node[sID_t]
                else:
                    dstFeat[i] = self.cacheFeats_edge[sID_t]
            cacheT.time_pop("")
            return dstFeat
        
        elif self.cache_method == 1:
            # In non-full caching scenarios, enabling edge CT means that the nodes can only temporarily compress.
            if self.slimEdgeFlag:
                inv = []
                sample_ID_t = []
                for i in range(len(sample_3ID_t)):
                    sID_t = sample_3ID_t[i]
                    if i==0:
                        if sID_t.device.type == 'cuda':
                            _uni, _inv=torch.unique(sID_t, return_inverse=True)
                            sample_ID_t.append(_uni.cpu())
                            inv.append(_inv)
                            self.node_uni_inv=[_uni, _inv]
                        else:
                            _uni, _inv=torch.unique(sID_t, return_inverse=True)
                            sample_ID_t.append(_uni) 
                            inv_ = _inv.cuda()
                            inv.append(inv_)
                            self.node_uni_inv=[_uni, inv_]
                    else:
                        sample_ID_t.append(torch.from_numpy(csrData[0 if i==1 else 2]))
                        inv.append(torch.from_numpy(csrData[1 if i==1 else 3]).to(self.devID))
                dstFeatSlim=[]
                for i in range(len(sample_ID_t)):
                    sID_t = sample_ID_t[i]
                    dstFeatSlim.append(torch.empty((sID_t.shape[0],srcFeat[0 if i==0 else 1].shape[1]), device='cuda', dtype=torch.float32))
            else:
                dstFeatSlim = dstFeat

            # hot static cache
            for i in range(len(sample_3ID_t)):
                cacheT.time_push("t_gpu")
                if self.slimEdgeFlag:
                    sID_t = sample_ID_t[i]
                else:
                    sID_t = sample_3ID_t[i]
                if i==0:
                    gpu_flag = self.gpu_flag_node
                    localid2cacheid = self.localid2cacheid_node
                    cacheFeats = self.cacheFeats_node
                else:
                    gpu_flag = self.gpu_flag_edge
                    localid2cacheid = self.localid2cacheid_edge
                    cacheFeats = self.cacheFeats_edge

                gpu_mask = gpu_flag[sID_t]
                nids_in_gpu = sID_t[gpu_mask.to(sID_t.device)]

                cacheid = localid2cacheid[nids_in_gpu]
                dstFeatSlim[i][gpu_mask] = cacheFeats[cacheid]
                cacheT.time_pop("t_gpu")

                cacheT.time_push("t_cpu")
                cpu_mask = ~(gpu_mask)

                if self.uniqueCPUFlag: # without using the slimEdgeFlag, temporary compression is not employed.
                    _nids_in_cpu_, _inv_ = torch.unique(sID_t[cpu_mask.to(sID_t.device)], return_inverse=True)
                    if len(_nids_in_cpu_) != 0: dstFeatSlim[i][cpu_mask] = torch.index_select(srcFeat[0 if i==0 else 1], 0, _nids_in_cpu_.to(srcFeat[0].device).long()).to(self.devID)[_inv_]
                elif self.slimEdgeFlag:
                    _nids_in_cpu_ = sID_t[cpu_mask.to(sID_t.device)]
                    dstFeatSlim[i][cpu_mask] = torch.index_select(srcFeat[0 if i==0 else 1], 0, _nids_in_cpu_.long()).to(self.devID)

                    dstFeat[i] = dstFeatSlim[i][inv[i]]
                else:
                    _nids_in_cpu_ = sID_t[cpu_mask.to(sID_t.device)]
                    dstFeatSlim[i][cpu_mask] = torch.index_select(srcFeat[0 if i==0 else 1], 0, _nids_in_cpu_.to(srcFeat[0].device).long()).to(self.devID)
                cacheT.time_pop("")
            return dstFeat if self.slimEdgeFlag else dstFeatSlim
