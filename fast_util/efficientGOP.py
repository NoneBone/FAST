import torch
import dgl
import fastAgg
from fast_util.helper import *

class forwardManager(object):
    def __init__(self):
        self.fastAggFlag = 0
        self.fastESMFlag = 0
        self.CSR_OUT = 0
        self.reduceSize = 16
        self.node_num_per_block = 1
        self.aggFeat = 1
        self.aggNodePB = 1 
        # No use
        self.fastKQV = 0
        self.fastTIME = 0
        self.fastDrop = 0

    def init_gop(self, fast_agg=0, fast_esm=0, csr_out=0):
        self.fastAggFlag = fast_agg
        self.fastESMFlag = fast_esm
        self.CSR_OUT = csr_out
        
        # block set by the method, 1 for FAST, 2 for naive
        if fast_esm==2:
            fm.reduceSize = 16
            fm.node_num_per_block = max(1, int(1024/fm.reduceSize/2))
        elif fast_esm==1:
            fm.node_num_per_block = 512
            fm.reduceSize = 16

        if fast_agg==2:
            fm.aggNodePB = 8
            fm.aggFeat = 32
        elif fast_agg==1:
            fm.aggNodePB = 8
            fm.aggFeat = 128
        
fm = forwardManager()

def see1see(t1,t2,str):
    '''
    :param t1: Automatically search for non-zero values
    :param t2: Automatically search for non-zero values
    '''
    print(str,end="")
    print(f"in shape: {t1.shape}, out shape: {t2.shape}")
    t = t1
    nonzero = (t != 0).nonzero(as_tuple=False)
    if len(nonzero) > 0:
        row, col = nonzero[0][0].item(), nonzero[0][1].item()
        value = t[row, col].item()
        print(f"first non-zero: t1[{row}, {col}] = {value}", end=" ")
    else:
        print("all zero")
    t = t2
    nonzero = (t != 0).nonzero(as_tuple=False)
    if len(nonzero) > 0:
        row, col = nonzero[0][0].item(), nonzero[0][1].item()
        value = t[row, col].item()
        print(f"t2[{row}, {col}] = {value}", end="\r\n")
    else:
        print("all zero")

class ESM_Function_TGAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dst_eid, src_eid_9764, input_feat, node_num_7882, num_heads=2):
        indptr = dst_eid
        indices = src_eid_9764 
        if fm.fastESMFlag == 1:
            X_prime = fastAgg.forward_edge_softmax(indptr, indices, input_feat, fm.node_num_per_block, num_heads)[0]
        elif fm.fastESMFlag == 2:
            X_prime = fastAgg.forward_csr_edge_softmax(indptr, indices, input_feat, fm.node_num_per_block, fm.reduceSize)[0]
        ctx.src_nodes = node_num_7882 
        ctx.num_heads = num_heads
        ctx.save_for_backward(indptr, indices, X_prime)
        return X_prime
    
    @staticmethod
    def backward(ctx, d_input):
        indptr, indices, output = ctx.saved_tensors
        node_num_7882 = ctx.src_nodes 
        num_heads = ctx.num_heads
        if not d_input.is_contiguous():
            d_input = d_input.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()
        if fm.fastESMFlag == 1:
            d_x = fastAgg.backward_edge_softmax(indptr, indices, d_input, output,  fm.node_num_per_block, num_heads)[0]
        elif fm.fastESMFlag == 2:
            d_x = fastAgg.backward_csr_edge_softmax(indptr, indices, d_input, output, fm.node_num_per_block, fm.reduceSize)[0]
        else:
            raise ValueError("fastESMFlag must be 1 or 2")       
        return None, None, d_x, None, None
    
class MA_Function_TGAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_edge, dst_edge_9764, input_feat, num_dst_nodes, neighbor_num=10, num_heads=2, densePtr=None, denseInd=None):
        if fm.fastAggFlag == 2:
            X_prime = fastAgg.forward_tgat_shd(densePtr, denseInd, src_edge, dst_edge_9764, input_feat, num_dst_nodes, fm.aggNodePB, fm.aggFeat)[0]
            ctx.save_for_backward(src_edge, dst_edge_9764, densePtr, denseInd)
        elif fm.fastAggFlag == 1:
            X_prime = fastAgg.forward_tgat(src_edge, dst_edge_9764, input_feat, num_dst_nodes, fm.aggNodePB, fm.aggFeat)[0]
            ctx.save_for_backward(src_edge, dst_edge_9764)
        else:
            raise ValueError("fastAggFlag must be 1 or 2")
        
        ctx.src_nodes = input_feat.size(0)
        ctx.num_heads = num_heads
        ctx.neighbor_num = neighbor_num
        
        return X_prime

    @staticmethod
    def backward(ctx, d_input):
        if fm.fastAggFlag == 2:
            src_edge, dst_edge_9764, densePtr, denseInd = ctx.saved_tensors
        else:
            src_edge, dst_edge_9764 = ctx.saved_tensors
        node_num_7882 = ctx.src_nodes
        num_heads = ctx.num_heads
        if not d_input.is_contiguous():
            d_input = d_input.contiguous()
        if fm.fastAggFlag == 2:
            d_x = fastAgg.backward_tgat_shd(densePtr, denseInd, src_edge, dst_edge_9764, d_input, node_num_7882, fm.aggNodePB, fm.aggFeat)[0]
        elif fm.fastAggFlag == 1:
            d_x = fastAgg.backward_tgat(src_edge, dst_edge_9764, d_input, node_num_7882, fm.aggNodePB, fm.aggFeat)[0]
        else:
            raise ValueError("fastAggFlag must be 1 or 2")
        return None, None, d_x, None, None, None, None, None

def unique2indptr(dstID, dstNum):
    localID_t, deg_t = torch.unique(dstID, return_counts=True)
    indptr = fastAgg.exclusive_sum(deg_t.int())
    return indptr

############################################################################################
# the following is deperated
############################################################################################
class FAST_QKV(torch.autograd.Function):
    '''
    leakeyRelu + esm + m, drop, reshape, cat, agg
    '''
    @staticmethod
    def forward(ctx, w_q, w_k, w_v, b_q, b_k, b_v, nFeat, eFeat, eid, timeFeat, zeroFeat, num_nodes, dim_out=100):
        device = nFeat.device
        with torch.no_grad():
            if fm.fastTIME:
                q_input = torch.cat([nFeat[:num_nodes], torch.cos(zeroFeat).repeat(num_nodes,1)], dim=1)
                kv_input = torch.cat([nFeat[num_nodes:], eFeat, torch.cos(timeFeat)], dim=1)
                Q = torch.nn.functional.linear(q_input, w_q, b_q)[eid]
                K = torch.nn.functional.linear(kv_input, w_k, b_k)
                V = torch.nn.functional.linear(kv_input, w_v, b_v)
            else:
                q_input = torch.cat([nFeat[:num_nodes], zeroFeat], dim=1)
                kv_input = torch.cat([nFeat[num_nodes:], eFeat, timeFeat], dim=1)
                Q = torch.nn.functional.linear(q_input, w_q, b_q)[eid]
                K = torch.nn.functional.linear(kv_input, w_k, b_k)
                V = torch.nn.functional.linear(kv_input, w_v, b_v)
        
            ctx.w_q_weight = w_q
            ctx.w_k_weight = w_k
            ctx.w_v_weight = w_v
            ctx.save_for_backward(nFeat, eFeat, timeFeat, zeroFeat, eid)
    
            ctx.num_nodes = num_nodes
            ctx.device = device
            ctx.dim_out = dim_out
            return Q, K, V

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        '''
        Re-calculate kqv, update the weights, and calculate the backpropagation values.
        '''
        with torch.no_grad():
            nFeat, eFeat, timeFeat, zeroFeat, eid = ctx.saved_tensors
            num_nodes = ctx.num_nodes
            device = ctx.device
            
            src_feat_dim = nFeat.shape[1]
            e_feat_dim = eFeat.shape[1]
            
            q_input = torch.cat([nFeat[:num_nodes], zeroFeat.repeat(num_nodes,1)], dim=1)
            kv_input = torch.cat([nFeat[num_nodes:], eFeat, timeFeat], dim=1)
            
            w_q = ctx.w_q_weight
            w_k = ctx.w_k_weight
            w_v = ctx.w_v_weight

            if not grad_q.is_contiguous():
                grad_q = grad_q.contiguous()
            if not grad_k.is_contiguous():
                grad_k = grad_k.contiguous()
            if not grad_v.is_contiguous():
                grad_v = grad_v.contiguous()
            # w_q: grad_w_q = grad_q^T * q_input
            # grad_q_full = torch.zeros((q_input.shape[0], ctx.dim_out), device=device)
            # grad_q_full.scatter_add_(0, eid.unsqueeze(1).expand(-1, ctx.dim_out), grad_q)
            #  index_add or scatter_add
            grad_q_full = torch.zeros(num_nodes, ctx.dim_out, device=grad_q.device)
            grad_q_full.index_add_(0, eid, grad_q)
            
            grad_w_q_weight = torch.matmul(grad_q_full.t(), q_input)
            grad_w_q_bias = grad_q_full.sum(dim=0)
            grad_w_k_weight = torch.matmul(grad_k.t(), kv_input)
            grad_w_k_bias = grad_k.sum(dim=0)
            grad_w_v_weight = torch.matmul(grad_v.t(), kv_input)
            grad_w_v_bias = grad_v.sum(dim=0)
            
            # dL/dq_input = grad_q_full * w_q.weight^T
            grad_q_input = torch.matmul(grad_q_full, w_q)
            
            # dL/dkv_input = grad_k * w_k.weight^T + grad_v * w_v.weight^T
            grad_k_slice = torch.matmul(grad_k, w_k[:, src_feat_dim + e_feat_dim:])
            grad_v_slice = torch.matmul(grad_v, w_v[:, src_feat_dim + e_feat_dim:])
            grad_kv_input = grad_k_slice + grad_v_slice
            
            sin_zero = torch.sin(zeroFeat)
            grad_zeroFeat = grad_q_input[:, src_feat_dim:] * (-sin_zero.repeat(num_nodes, 1))
            
            grad_timeFeat = grad_kv_input * (-torch.sin(timeFeat))
            
            del q_input, kv_input, grad_q_input, grad_kv_input, sin_zero, grad_k_slice, grad_v_slice
            if 'grad_q_full' in locals():
                del grad_q_full
            
            # if torch.cuda.is_available() and torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
            torch.cuda.empty_cache()
            return grad_w_q_weight, grad_w_k_weight, grad_w_v_weight, grad_w_q_bias, grad_w_k_bias, grad_w_v_bias, None, None, None, grad_timeFeat, grad_zeroFeat, None, None


class FUSED_TGAT(torch.autograd.Function):
    '''
    leakeyRelu + esm + m, drop, reshape, cat, agg
    '''
    @staticmethod
    def forward(ctx, input_feat, Vval, indptr, indices, dst_eid, src_eid_9764, node_num_7882, node_num_9764, num_heads=2, negative_slope=0.2, attn_drop=0.1):
        if fm.fastESMFlag == 3:
            Y_prime, attn, edgeMask = fastAgg.forward_fused_attn(input_feat, Vval, indptr, indices, dst_eid, src_eid_9764, node_num_7882, num_heads, negative_slope, attn_drop)
        ctx.node_num = node_num_9764
        ctx.num_heads = num_heads
        ctx.negative_slope = negative_slope
        ctx.attn_drop = attn_drop
        # see1see(input_feat, Y_prime, ">>> fused forward:")
        ctx.save_for_backward(Vval, attn, edgeMask, indptr, indices, dst_eid, src_eid_9764) 
        return Y_prime
    
    @staticmethod
    def backward(ctx, d_input):
        Vval, attn, edgeMask, indptr, indices, dst_eid, src_eid_9764 = ctx.saved_tensors
        node_num_9764 = ctx.node_num
        num_heads = ctx.num_heads
        negative_slope = ctx.negative_slope
        attn_drop = ctx.attn_drop
        if not d_input.is_contiguous():
            d_input = d_input.contiguous()
        if not attn.is_contiguous():
            attn = attn.contiguous()
        if fm.fastESMFlag == 3:
            d_x1,d_x2 = fastAgg.backward_fused_attn(d_input, Vval, attn, edgeMask, indptr, indices, dst_eid, src_eid_9764, node_num_9764, num_heads, negative_slope, attn_drop)
        # see1see(d_input, d_x1, ">>> backward:")
        return d_x1, d_x2, None, None, None, None, None, None, None, None, None

class FAST_DROP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, Vval, node_num_7882):
        with torch.no_grad():
            V = torch.reshape(Vval*attn[:, :, None], (Vval.shape[0], -1)) # 1882 * 100
            Y_prime = torch.cat([torch.zeros((node_num_7882, V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            ctx.save_for_backward(Vval, attn)
            ctx.node_num = node_num_7882
        
            return Y_prime # 9764 * 100
    
    @staticmethod
    def backward(ctx, d_input):
        with torch.no_grad():
            Vval, attn = ctx.saved_tensors
            node_num_7882 = ctx.node_num

            if not d_input.is_contiguous():
                d_input = d_input.contiguous()
            d_input_reshaped = d_input[node_num_7882:].reshape(Vval.shape[0], Vval.shape[1], -1)
            grad_V = d_input_reshaped * attn[:, :, None]
            grad_attn = torch.sum(d_input_reshaped * Vval, dim=-1)
            
            return grad_attn, grad_V, None
