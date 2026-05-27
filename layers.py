import torch
import dgl
import math
import numpy as np

from fast_util.helper import *
from fast_util.efficientGOP import *
from fast_util.trainingVerifier import (
    dgl_aggregate_reference,
    print_bad_segments_once,
    verify_forward_aggregate,
    verify_forward_edge_softmax,
)

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        nsysForward.time_push("t_linear")
        y = self.w(t.reshape((-1, 1)))
        nsysForward.time_pop("t_linear")
        if fm.fastTIME:
            nsysForward.time_push("t_cos")
            nsysForward.time_pop("t_cos")
            return y
        else:
            nsysForward.time_push("t_cos")
            output = torch.cos(y)
            nsysForward.time_pop("t_cos")
            return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        nsysForward.time_push("t_mlp")
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        pos_t, neg_t = self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
        nsysForward.time_pop()
        return pos_t, neg_t
    

class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.negative_slope = 0.2
        self.att_act = torch.nn.LeakyReLU(self.negative_slope)
        self.combined = combined
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:
            if dim_node_feat > 0:
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if fm.fastKQV:
                if dim_node_feat + dim_time > 0:
                    self.weight_q = nn.Parameter(torch.randn(dim_out, dim_node_feat + dim_time))
                    self.bias_q = nn.Parameter(torch.zeros(dim_out))
                self.weight_k = nn.Parameter(torch.randn(dim_out, dim_node_feat + dim_edge_feat  + dim_time))
                self.bias_k = nn.Parameter(torch.zeros(dim_out))
                self.weight_v = nn.Parameter(torch.randn(dim_out, dim_node_feat + dim_edge_feat  + dim_time))
                self.bias_v = nn.Parameter(torch.zeros(dim_out))
            else:
                if dim_node_feat + dim_time > 0:
                    self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
                self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
                self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b, _layers, _hist):
        # nsysForward.time_push("t_timeEncode")
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            if fm.fastTIME:
                _zero = torch.zeros(1, dtype=torch.float32, device=torch.device('cuda:0'))
                zero_time_feat = self.time_enc(_zero)
            else:
                _zero = torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0'))
                zero_time_feat = self.time_enc(_zero)
        # nsysForward.time_pop("t_timeEncode")
        nsysForward.time_push("t_qkvMM")
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.num_dst_nodes():])
                V = self.w_v(b.srcdata['h'][b.num_dst_nodes():])
            elif self.dim_time == 0:# DySAT
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:# TGN TGAT
                if fm.fastKQV:
                    Q, K, V = FAST_QKV.apply(self.weight_q, self.weight_k, self.weight_v, self.bias_q, self.bias_k, self.bias_v,\
                                             b.srcdata['h'], b.edata['f'], b.edges()[1], time_feat, zero_time_feat, b.num_dst_nodes(), self.dim_out)
                else:
                    Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                    K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                    V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            nsysForward.time_pop()
            nsysForward.time_push("t_eleM")
            if fm.fastESMFlag == 3:
                f_tmp = torch.sum(Q*K, dim=2) # multiply and sum no need to be fused 
            else:
                f_tmp = self.att_act(torch.sum(Q*K, dim=2))
            if fm.fastESMFlag:
                if fm.CSR_OUT and fm.fastESMFlag != 2:
                    if self.dim_time == 0:
                        all_hist_num = int(len(fm.cntE)/fm.layerAllNum)
                        densePtr = fm.cntE[all_hist_num - 1 - _hist + _layers*all_hist_num]
                    else:
                        densePtr = fm.cntE[_layers]
                else:
                    densePtr = unique2indptr(b.edges()[1], b.num_dst_nodes())
                indices = densePtr # no use
            nsysForward.time_pop()
            nsysForward.time_push("t_esm")
            # edge_softmax
            if fm.fastESMFlag == 2 or fm.fastESMFlag == 1:
                att = ESM_Function_TGAT.apply(densePtr, indices, f_tmp, b.num_dst_nodes())
            elif fm.fastESMFlag == 3:
                aggTemp = FUSED_TGAT.apply(f_tmp, V, densePtr, densePtr, b.edges()[1], b.edges()[0], b.num_dst_nodes(), b.num_src_nodes(), self.num_head, self.negative_slope)
            else:
                att = dgl.ops.edge_softmax(b, f_tmp)
            if fm.FORWARD_VERIFY_ERR and fm.fastESMFlag in (1, 2):
                ref_att = dgl.ops.edge_softmax(b, f_tmp)
                verify_forward_edge_softmax(
                    ref_att,
                    att,
                    fm.FORWARD_VERIFY_ERR,
                    f"(layer={_layers}, hist={_hist}, fastESMFlag={fm.fastESMFlag})",
                )
            if fm.fastESMFlag == 3:
                nsysForward.time_pop()
            else:
                nsysForward.time_pop()
                nsysForward.time_push("t_drop+reshape+cat")
                att = self.att_dropout(att)
                if fm.fastDrop:
                    src_v = FAST_DROP.apply(att, V, b.num_dst_nodes())
                else:
                    V = torch.reshape(V * att[:, :, None], (V.shape[0], -1))
                    src_v = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
                    b.srcdata['v'] = src_v
                nsysForward.time_pop()
                nsysForward.time_push("t_agg")
                if not fm.fastAggFlag:
                    if fm.fastDrop:
                        b.srcdata['v'] = src_v
                    b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
                elif fm.fastAggFlag == 1:
                    aggTemp = MA_Function_TGAT.apply(b.edges()[1], b.edges()[0], src_v, b.num_dst_nodes(), 10, 2, b.srcdata['h'], self.dim_node_feat) # 
                elif fm.fastAggFlag == 2:
                    if not fm.fastESMFlag:
                        if fm.CSR_OUT:
                            densePtr = fm.cntE[_layers]
                        else:
                            densePtr = unique2indptr(b.edges()[1], b.num_dst_nodes())
                    aggTemp = MA_Function_TGAT.apply(b.edges()[1], b.edges()[0], src_v, b.num_dst_nodes(), 10 ,2, densePtr, densePtr)
                nsysForward.time_pop()
        nsysForward.time_push("t_beforeMLP")
        if (fm.fastAggFlag or fm.fastESMFlag == 3):
            if self.dim_node_feat != 0:
                rst = torch.cat([aggTemp, b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
            else:
                rst = aggTemp
        else:
            if self.dim_node_feat != 0:
                rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
            else:
                rst = b.dstdata['h']
        if fm.FORWARD_VERIFY_ERR and (fm.fastAggFlag or fm.fastESMFlag == 3):
            if fm.fastAggFlag == 2:
                print_bad_segments_once(
                    b.edges()[1],
                    densePtr,
                    f"(layer={_layers}, hist={_hist}, fastESMFlag={fm.fastESMFlag}, fastAggFlag={fm.fastAggFlag})",
                )
            if fm.fastESMFlag == 3:
                if not self.training or self.att_dropout.p == 0:
                    ref_scores = self.att_act(torch.sum(Q * K, dim=2))
                    ref_att = self.att_dropout(dgl.ops.edge_softmax(b, ref_scores))
                    ref_v = torch.reshape(V * ref_att[:, :, None], (V.shape[0], -1))
                    ref_src_v = torch.cat([torch.zeros((b.num_dst_nodes(), ref_v.shape[1]), device=torch.device('cuda:0')), ref_v], dim=0)
                    verify_forward_aggregate(
                        dgl_aggregate_reference(b, ref_src_v),
                        aggTemp,
                        fm.FORWARD_VERIFY_ERR,
                        f"(layer={_layers}, hist={_hist}, fastESMFlag={fm.fastESMFlag}, fastAggFlag={fm.fastAggFlag})",
                    )
            else:
                verify_forward_aggregate(
                    dgl_aggregate_reference(b, src_v),
                    aggTemp,
                    fm.FORWARD_VERIFY_ERR,
                    f"(layer={_layers}, hist={_hist}, fastESMFlag={fm.fastESMFlag}, fastAggFlag={fm.fastAggFlag})",
                )
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        tmp = self.layer_norm(rst)
        nsysForward.time_pop()

        return tmp

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            
