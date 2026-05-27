import atexit

import dgl
import torch

FORWARD_VERIFY_RTOL = 1e-5
FORWARD_VERIFY_ATOL = 1e-8
BACKWARD_VERIFY_RTOL = 1e-5
BACKWARD_VERIFY_ATOL = 1e-8


def _new_stats():
    return {
        'count': 0,
        'sum': 0.0,
        'min': None,
        'max': None,
        'max_abs': 0.0,
    }


FORWARD_EDGE_STATS = _new_stats()
FORWARD_AGG_STATS = _new_stats()
BACKWARD_ESM_STATS = _new_stats()
BACKWARD_MA_STATS = _new_stats()
_BAD_SEG_CHECK_DONE = False


def _record_ratio(stats, diff_ratio, max_abs_err):
    stats['count'] += 1
    stats['sum'] += diff_ratio
    if stats['min'] is None or diff_ratio < stats['min']:
        stats['min'] = diff_ratio
    if stats['max'] is None or diff_ratio > stats['max']:
        stats['max'] = diff_ratio
    if max_abs_err > stats['max_abs']:
        stats['max_abs'] = max_abs_err


def _print_ratio_summary(prefix, tag, stats):
    count = stats['count']
    if count == 0:
        return
    avg_ratio = stats['sum'] / count
    min_ratio = stats['min']
    max_ratio = stats['max']
    max_abs_err = stats['max_abs']
    print(
        f"[{prefix}] {tag} mismatch ratio summary: "
        f"count={count}, avg={avg_ratio:.6%}, min={min_ratio:.6%}, "
        f"max={max_ratio:.6%}, max_abs_err={max_abs_err:.6e}"
    )

atexit.register(lambda: _print_ratio_summary("BACKWARD_VERIFY_ERR", "MA backward", BACKWARD_MA_STATS))
atexit.register(lambda: _print_ratio_summary("BACKWARD_VERIFY_ERR", "ESM backward", BACKWARD_ESM_STATS))
atexit.register(lambda: _print_ratio_summary("FORWARD_VERIFY_ERR", "aggregate", FORWARD_AGG_STATS))
atexit.register(lambda: _print_ratio_summary("FORWARD_VERIFY_ERR", "edge_softmax", FORWARD_EDGE_STATS))

def _verify_ratio(reference, candidate, stats, rtol, atol, prefix, tag, extra_msg=""):
    if reference.shape != candidate.shape:
        print(f"[{prefix}] {tag} shape mismatch: {reference.shape} vs {candidate.shape}. {extra_msg}")
        return

    ref = reference.detach()
    cur = candidate.detach()
    nonzero_mask = (ref != 0) | (cur != 0)
    total_nonzero = int(nonzero_mask.sum().item())
    if total_nonzero == 0:
        diff_ratio = 0.0
    else:
        diff_mask = nonzero_mask & (~torch.isclose(ref, cur, rtol=rtol, atol=atol))
        diff_ratio = int(diff_mask.sum().item()) / total_nonzero
    max_abs_err = 0.0 if ref.numel() == 0 else torch.max(torch.abs(cur - ref)).item()
    _record_ratio(stats, diff_ratio, max_abs_err)


def verify_forward_edge_softmax(reference, candidate, enabled, extra_msg=""):
    if not enabled:
        return
    _verify_ratio(
        reference,
        candidate,
        FORWARD_EDGE_STATS,
        FORWARD_VERIFY_RTOL,
        FORWARD_VERIFY_ATOL,
        "FORWARD_VERIFY_ERR",
        "edge_softmax",
        extra_msg,
    )


def verify_forward_aggregate(reference, candidate, enabled, extra_msg=""):
    if not enabled:
        return
    _verify_ratio(
        reference,
        candidate,
        FORWARD_AGG_STATS,
        FORWARD_VERIFY_RTOL,
        FORWARD_VERIFY_ATOL,
        "FORWARD_VERIFY_ERR",
        "aggregate",
        extra_msg,
    )


def dgl_aggregate_reference(b, src_v):
    with b.local_scope():
        b.srcdata['v'] = src_v
        b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        return b.dstdata['h']


def print_bad_segments_once(dst_edges, dense_ptr, extra_msg="", max_print=10):
    global _BAD_SEG_CHECK_DONE
    if _BAD_SEG_CHECK_DONE or dense_ptr.numel() < 2:
        return
    _BAD_SEG_CHECK_DONE = True

    bad_segments = []
    seg_num = dense_ptr.numel() - 1
    for seg_id in range(seg_num):
        lb = int(dense_ptr[seg_id].item())
        lh = int(dense_ptr[seg_id + 1].item())
        if lh <= lb:
            continue
        seg_dst = dst_edges[lb:lh]
        if not torch.all(seg_dst == seg_dst[0]):
            bad_segments.append((seg_id, lb, lh))

    print(f"[FORWARD_VERIFY_ERR] bad_seg_num={len(bad_segments)}/{seg_num}. {extra_msg}")
    for seg_id, lb, lh in bad_segments[:max_print]:
        seg_dst = dst_edges[lb:lh]
        uniq_list = torch.unique(seg_dst)[:8].detach().cpu().tolist()
        print(
            f"  bad_seg id={seg_id}, range=[{lb}, {lh}), "
            f"first_dst={int(seg_dst[0].item())}, unique_dst={uniq_list}"
        )


def _esm_backward_reference(indptr, grad_output, output):
    # TODO: Completely adopt the internal implementation of DGL
    # dgl/backend/pytorch/sparse.py: class EdgeSoftmax: backward
    ref_grad = torch.zeros_like(grad_output)
    node_num = indptr.numel() - 1
    for node_id in range(node_num):
        lb = int(indptr[node_id].item())
        lh = int(indptr[node_id + 1].item())
        if lh <= lb:
            continue
        out_seg = output[lb:lh]
        grad_seg = grad_output[lb:lh]
        seg_sum = torch.sum(out_seg * grad_seg, dim=0, keepdim=True)
        ref_grad[lb:lh] = out_seg * (grad_seg - seg_sum)
    return ref_grad


def _ma_backward_reference(dst_index, src_index, grad_output, input_rows):
    ref_grad = torch.zeros(
        (input_rows, grad_output.shape[1]),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    ref_grad.index_add_(0, src_index, grad_output[dst_index])
    return ref_grad


def verify_esm_backward(indptr, grad_output, output, grad_input, enabled, fast_esm_flag):
    if not enabled:
        return
    ref_grad = _esm_backward_reference(indptr, grad_output, output)
    _verify_ratio(
        ref_grad,
        grad_input,
        BACKWARD_ESM_STATS,
        BACKWARD_VERIFY_RTOL,
        BACKWARD_VERIFY_ATOL,
        "BACKWARD_VERIFY_ERR",
        "ESM backward",
        f"(fastESMFlag={fast_esm_flag})",
    )


def verify_ma_backward(dst_index, src_index, grad_output, grad_input, enabled, fast_agg_flag):
    if not enabled:
        return
    ref_grad = _ma_backward_reference(dst_index, src_index, grad_output, grad_input.shape[0])
    _verify_ratio(
        ref_grad,
        grad_input,
        BACKWARD_MA_STATS,
        BACKWARD_VERIFY_RTOL,
        BACKWARD_VERIFY_ATOL,
        "BACKWARD_VERIFY_ERR",
        "MA backward",
        f"(fastAggFlag={fast_agg_flag})",
    )
