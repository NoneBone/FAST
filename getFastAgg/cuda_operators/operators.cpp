#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/********************************************************************************************************************************************* */
// 0.Declare the kernel startup function
std::vector<torch::Tensor> computation_aware_forward_tgat_shd_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_edges,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int dst_nodes,
    int aggNodePB,
    int aggFeat
);
std::vector<torch::Tensor> computation_aware_backward_tgat_shd_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_edges,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int aggNodePB,
    int aggFeat
);
std::vector<torch::Tensor> computation_aware_forward_tgat_cuda(// temporal
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor input_feat,
    int dst_num,
    int aggNodePB,
    int aggFeat
);
std::vector<torch::Tensor> computation_aware_backward_tgat_cuda(// temporal
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor d_input,
    int src_num,
    int aggNodePB,
    int aggFeat
);

std::vector<torch::Tensor> forward_edge_softmax_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int node_num_per_block,
    int num_heads
);
std::vector<torch::Tensor> backward_edge_softmax_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int num_heads
);
std::vector<torch::Tensor> forward_balanced_edge_softmax_cuda(// balanced_softmax
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int edge_num_per_block,
    int reduce_size
);
std::vector<torch::Tensor> backward_balanced_edge_softmax_cuda(// balanced_softmax backward
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int edge_num_per_block,
    int reduce_size
);
std::vector<torch::Tensor> forward_csr_edge_softmax_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int node_num_per_block,
    int reduce_size
);
std::vector<torch::Tensor> backward_csr_edge_softmax_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int reduce_size
);
std::vector<torch::Tensor> forward_fused_attn_cuda(
    torch::Tensor input_feat,
    torch::Tensor V_val,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_eid,
    torch::Tensor src_eid_9764,
    int node_num_7882,
    int num_heads,
    float negative_slope,
    float attn_drop
);
std::vector<torch::Tensor> backward_fused_attn_cuda(
    torch::Tensor d_output_agg,
    torch::Tensor V_val,
    torch::Tensor output_attn,
    torch::Tensor edge_mask,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_eid,
    torch::Tensor src_eid_9764,
    int node_num_7882,
    int num_heads,
    float negative_slope,
    float attn_drop
);
/********************************************************************************************************************************************* */
// 1.forward
std::vector<torch::Tensor> computation_aware_forward_tgat_shd(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor input_feat,
    int dst_nodes,
    int aggNodePB,
    int aggFeat
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(dst_edges);
    CHECK_INPUT(input_feat);

    return computation_aware_forward_tgat_shd_cuda(
        indptr,
        indices,
        src_edges,
        dst_edges,
        input_feat,
        dst_nodes,
        aggNodePB,
        aggFeat);
}
std::vector<torch::Tensor> computation_aware_forward_tgat(// temporal
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor input_feat,
    int dst_num,
    int aggNodePB,
    int aggFeat
){
    CHECK_INPUT(src_edges);
    CHECK_INPUT(dst_edges);
    CHECK_INPUT(input_feat);

    return computation_aware_forward_tgat_cuda(
        src_edges,
        dst_edges,
        input_feat,
        dst_num,
        aggNodePB,
        aggFeat);
}

std::vector<torch::Tensor> forward_edge_softmax(// edge_softmax
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int node_num_per_block,
    int num_heads
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(input_feat);

    auto result = forward_edge_softmax_cuda(
        indptr,
        indices,
        input_feat,
        node_num_per_block,
        num_heads);
    if (!result.empty()) {
        auto& d_input = result[0];
        
        if (d_input.numel() > 0) {
            auto d_input_cpu = d_input.cpu();

            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "  ERROR: forward_edge_softmax_cuda contains NaN!" << std::endl;
                std::cout << "[DEBUG] forward_edge_softmax_cuda output:" << std::endl;
                std::cout << "  d_input: " << d_input.sizes() << std::endl;
                        std::cout   << "  d_input first 2 val:" << std::endl;
                for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
                    for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
                        std::cout << "    [" << i << "," << j << "]: " 
                                << d_input_cpu[i][j].item<float>() << std::endl;
                    }
                }
            }
        }
    }
    return result;
}

std::vector<torch::Tensor> forward_balanced_edge_softmax(// csr_softmax
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int edge_num_per_block,
    int reduce_size
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(input_feat);

    auto result = forward_balanced_edge_softmax_cuda(
        indptr,
        indices,
        input_feat,
        edge_num_per_block,
        reduce_size);
    if (!result.empty()) {
        auto& d_input = result[0];
        // std::cout << "[DEBUG] forward_balanced_edge_softmax output:" << std::endl;
        // std::cout << "  d_input: " << d_input.sizes() << std::endl;
        
        if (d_input.numel() > 0) {
            auto d_input_cpu = d_input.cpu();
            // std::cout << "  d_input first 2 val:" << std::endl;
            // for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
            //     for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
            //         std::cout << "    [" << i << "," << j << "]: " 
            //                 << d_input_cpu[i][j].item<float>() << std::endl;
            //     }
            // }
            
            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "  ERROR: forward_balanced_edge_softmax contains NaN!" << std::endl;
            }
        }
    }
    return result;
}
std::vector<torch::Tensor> forward_csr_edge_softmax(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int node_num_per_block,
    int reduce_size
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(input_feat);

    auto result = forward_csr_edge_softmax_cuda(
        indptr,
        indices,
        input_feat,
        node_num_per_block,
        reduce_size);
    if (!result.empty()) {
        auto& d_input = result[0];
        
        if (d_input.numel() > 0) {
            auto d_input_cpu = d_input.cpu();

            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "  ERROR: forward_csr_edge_softmax contains NaN!" << std::endl;
                std::cout << "[DEBUG] forward_csr_edge_softmax output:" << std::endl;
                std::cout << "  d_input: " << d_input.sizes() << std::endl;
                        std::cout   << "  d_input first 2 val:" << std::endl;
                for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
                    for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
                        std::cout << "    [" << i << "," << j << "]: " 
                                << d_input_cpu[i][j].item<float>() << std::endl;
                    }
                }
            }
        }
    }
    return result;
}
std::vector<torch::Tensor> forward_fused_attn(
    torch::Tensor input_feat,
    torch::Tensor V_val,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_eid,
    torch::Tensor src_eid_9764,
    int node_num_7882,
    int num_heads,
    float negative_slope,
    float attn_drop
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(input_feat);

    auto result = forward_fused_attn_cuda(
        input_feat,
        V_val,
        indptr,
        indices,
        dst_eid,
        src_eid_9764,
        node_num_7882,
        num_heads,
        negative_slope,
        attn_drop);
    if (!result.empty()) {
        auto& d_input = result[0];
        
        if (d_input.numel() > 0) {
            auto d_input_cpu = d_input.cpu();

            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "[DEBUG] forward_fused_attn output:" << std::endl;
                std::cout << "  input: " << d_input.sizes() << std::endl;
                std::cout << "  input first 2 val:" << std::endl;
                for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
                    for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
                        std::cout << "    [" << i << "," << j << "]: " 
                                << d_input_cpu[i][j].item<float>() << std::endl;
                    }
                }
                std::cout << "  ERROR: forward_fused_attn contains NaN!" << std::endl;
            }
        }
    }
    return result;
}

/********************************************************************************************************************************************* */
// 2.backward
std::vector<torch::Tensor> computation_aware_backward_tgat_shd(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_edges,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int aggNodePB,
    int aggFeat
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(dst_edges);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(d_input);

    return computation_aware_backward_tgat_shd_cuda(
        indptr,
        indices,
        dst_edges,
        src_edges,
        d_input,
        src_nodes,
        aggNodePB,
        aggFeat
    );
}
std::vector<torch::Tensor> computation_aware_backward_tgat(// temporal
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor d_input,
    int src_num,
    int aggNodePB,
    int aggFeat
){
    CHECK_INPUT(src_edges);
    CHECK_INPUT(dst_edges);
    CHECK_INPUT(d_input);

    return computation_aware_backward_tgat_cuda(
        src_edges,
        dst_edges,
        d_input,
        src_num,
        aggNodePB,
        aggFeat
        );
}
std::vector<torch::Tensor> backward_edge_softmax(// edge_softmax backward
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int num_heads
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(d_output);
    CHECK_INPUT(output);

    auto result =  backward_edge_softmax_cuda(
        indptr,
        indices,
        d_output,
        output,
        node_num_per_block,
        num_heads);
    if (!result.empty()) {
        auto& d_input = result[0];
        
        if (d_input.numel() > 0) {
            
            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "  ERROR: backward_edge_softmax_cuda contains NaN!" << std::endl;
                // std::cout << "[DEBUG] backward_edge_softmax_cuda output:" << std::endl;
                std::cout << "  d_input: " << d_input.sizes() << std::endl;

                auto d_input_cpu = d_input.cpu();
                std::cout << "  d_input first 2 val:" << std::endl;
                for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
                    for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
                        std::cout << "    [" << i << "," << j << "]: " 
                                << d_input_cpu[i][j].item<float>() << std::endl;
                    }
                }
            }
        }
    }
    return result;
}
std::vector<torch::Tensor> backward_balanced_edge_softmax(// csr_softmax backward
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int reduce_size
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(d_output);
    CHECK_INPUT(output);

    auto result =  backward_balanced_edge_softmax_cuda(
        indptr,
        indices,
        d_output,
        output,
        node_num_per_block,
        reduce_size);
    if (!result.empty()) {
        auto& d_input = result[0];
        // std::cout << "[DEBUG] backward_balanced_edge_softmax output:" << std::endl;
        // std::cout << "  d_input: " << d_input.sizes() << std::endl;
        
        if (d_input.numel() > 0) {
            // auto d_input_cpu = d_input.cpu();
            // std::cout << "  d_input first 2 val:" << std::endl;
            // for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
            //     for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
            //         std::cout << "    [" << i << "," << j << "]: " 
            //                 << d_input_cpu[i][j].item<float>() << std::endl;
            //     }
            // }
            
            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "  ERROR: backward_balanced_edge_softmax contains NaN!" << std::endl;
            }
        }
    }
    return result;
}
std::vector<torch::Tensor> backward_csr_edge_softmax(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int reduce_size
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(d_output);
    CHECK_INPUT(output);

    auto result =  backward_csr_edge_softmax_cuda(
        indptr,
        indices,
        d_output,
        output,
        node_num_per_block,
        reduce_size);
    if (!result.empty()) {
        auto& d_input = result[0];
        
        if (d_input.numel() > 0) {
            
            if (torch::isnan(d_input).any().item<bool>()) {
                std::cout << "  ERROR: backward_csr_edge_softmax contains NaN!" << std::endl;
                // std::cout << "[DEBUG] backward_csr_edge_softmax_cuda output:" << std::endl;
                std::cout << "  d_input: " << d_input.sizes() << std::endl;

                auto d_input_cpu = d_input.cpu();
                std::cout << "  d_input first 2 val:" << std::endl;
                for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
                    for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
                        std::cout << "    [" << i << "," << j << "]: " 
                                << d_input_cpu[i][j].item<float>() << std::endl;
                    }
                }
            }
        }
    }
    return result;
}
std::vector<torch::Tensor> backward_fused_attn(
    torch::Tensor d_output,
    torch::Tensor V_val,
    torch::Tensor output,
    torch::Tensor edge_mask,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_eid,
    torch::Tensor src_eid_9764,
    int node_num_7882,
    int num_heads,
    float negative_slope,
    float attn_drop
){
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(d_output);
    CHECK_INPUT(output);

    auto result = backward_fused_attn_cuda(
        d_output,
        V_val,
        output,
        edge_mask,
        indptr,
        indices,
        dst_eid,
        src_eid_9764,
        node_num_7882,
        num_heads,
        negative_slope,
        attn_drop);
    if (!result.empty()) {
        auto& d_input = result[0];
        if (d_input.numel() > 0) {
            if (torch::isnan(d_input).any().item<bool>()) {
            // std::cout << "[DEBUG] backward_balanced_edge_softmax output:" << std::endl;
            std::cout << "  d_input: " << d_input.sizes() << std::endl;
            auto d_input_cpu = d_input.cpu();
            std::cout << "  d_input first 2 val:" << std::endl;
            for (int i = 0; i < std::min(2, (int)d_input.size(0)); i++) {
                for (int j = 0; j < std::min(2, (int)d_input.size(1)); j++) {
                    std::cout << "    [" << i << "," << j << "]: " 
                            << d_input_cpu[i][j].item<float>() << std::endl;
                }
            }
            
                std::cout << "  ERROR: backward_fused_attn contains NaN!" << std::endl;
            }
        }
    }
    return result;
}
/********************************************************************************************************************************************* */
// 3.utils for subgraph-csr
at::Tensor exclusive_sum_cuda(
    torch::Tensor in_degs
);

at::Tensor cal_deg_cuda(
    torch::Tensor edges,
    int nodes_num
);
at::Tensor get_balanced_cuda(
    torch::Tensor densePtr,
    int threshold
);
at::Tensor exclusive_sum(
    torch::Tensor in_degs
){
    CHECK_INPUT(in_degs);

    return exclusive_sum_cuda(in_degs);
}

at::Tensor cal_deg(
    torch::Tensor edges,
    int nodes_num
){
    CHECK_INPUT(edges);

    return cal_deg_cuda(edges,nodes_num);
}

at::Tensor getBalancedPtr(
    torch::Tensor densePtr,
    int threshold
){
    CHECK_INPUT(densePtr);

    return get_balanced_cuda(densePtr, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cal_deg", &cal_deg, "cal_deg (CUDA)");
  m.def("exclusive_sum", &exclusive_sum, "exclusive_sum (CUDA)");
  m.def("getBalancedPtr", &getBalancedPtr, "balanced_ptr (CUDA)");
  // global-based
  m.def("forward_tgat", &computation_aware_forward_tgat, "computation_aware_forward_tgat (CUDA)");
  m.def("backward_tgat", &computation_aware_backward_tgat, "computation_aware_backward_tgat (CUDA)");
  m.def("forward_edge_softmax", &forward_edge_softmax, "esm (CUDA)");
  m.def("backward_edge_softmax", &backward_edge_softmax, "esm (CUDA)");
  // csr-based
  m.def("forward_tgat_shd", &computation_aware_forward_tgat_shd, "computation_aware_forward_tgat (CUDA)");
  m.def("backward_tgat_shd", &computation_aware_backward_tgat_shd, "computation_aware_backward_tgat_shd (CUDA)");
  m.def("forward_balanced_edge_softmax", &forward_balanced_edge_softmax, "esm_csr (CUDA)");
  m.def("backward_balanced_edge_softmax", &backward_balanced_edge_softmax, "esm_csr (CUDA)");
  m.def("forward_csr_edge_softmax", &forward_csr_edge_softmax, "esm_online (CUDA)");
  m.def("backward_csr_edge_softmax", &backward_csr_edge_softmax, "esm_online (CUDA)");
  // fused
  m.def("forward_fused_attn", &forward_fused_attn, "fused attn (CUDA)");
  m.def("backward_fused_attn", &backward_fused_attn, "fused attn (CUDA)");
}
