#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <time.h>
// util
#include <curand_kernel.h>
#include <cub/cub.cuh>


template <typename scalar_t>
__global__ void computation_aware_forward_tgat_cuda_kernel(// temporal
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int dim,
                    int edge_num
);

template <typename scalar_t>
__global__ void computation_aware_backward_tgat_cuda_kernel(// temporal
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int src_num,
                    int edge_num,
                    int neighbor_num
);

/**************************************************************  Forward cuda ******************************************************************************* */
std::vector<torch::Tensor> computation_aware_forward_tgat_cuda(// temporal
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor input_feat,
    int dst_num,
    int aggNodePB,
    int aggFeat
)
{   
    int dim = input_feat.size(1);// 128
    auto output = torch::zeros({dst_num, dim}, torch::kCUDA);// 7882 * 100
    const dim3 threads(aggFeat, aggNodePB);// 32, 8
    int edge_num = src_edges.size(0);// 1882
    const dim3 blocks((dim+threads.x-1)/threads.x, (edge_num+threads.y-1)/threads.y);// cover all edges

    // float agg_start_time = clock();
    
    // const int numBlocks = dst_num;
    AT_DISPATCH_FLOATING_TYPES(input_feat.type(), "computation_aware_forward_tgat_cuda_kernel", ([&] {
                                    computation_aware_forward_tgat_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                        src_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,
                                        edge_num
                                    );
                                }));
    // kernel<<<grids, blocks>>>();
    // float agg_end_time = clock();

    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    return {output};
}

/**************************************************************  Forward Kernel ******************************************************************************* */
template <typename scalar_t>
__global__ void computation_aware_forward_tgat_cuda_kernel(// temporal
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int dim,
                    int edge_num
)
{
    int current_dst_id = blockIdx.y * blockDim.y + threadIdx.y;// 0~8

    int feat_id = blockIdx.x * blockDim.x + threadIdx.x;// 0~31

    if((current_dst_id>=edge_num) || (feat_id>=dim)){
        return;
    }
    // global memory
    if(current_dst_id<edge_num){
        atomicAdd(&output[src_edges[current_dst_id]][feat_id], 
                input_feat[dst_edges[current_dst_id]][feat_id]);
    }
}

/**************************************************************  backward cuda  ******************************************************************************* */
std::vector<torch::Tensor> computation_aware_backward_tgat_cuda(// temporal
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor d_input,
    int src_nodes,
    int aggNodePB,
    int aggFeat
)
{   
    int dst_num = d_input.size(0);// 6000
    int dim = d_input.size(1);// 100
    auto d_output = torch::zeros({src_nodes, dim}, torch::kCUDA);// 7882 * 100
    int shared_memory = 64 * sizeof(float);
    // const dim3 threads(64); 
    // const dim3 blocks(dst_num, (dim+threads.x-1)/threads.x);
    const dim3 threads(aggFeat, aggNodePB);
    int edge_num = src_edges.size(0);// 1882
    const dim3 blocks((dim+threads.x-1)/threads.x, (edge_num+threads.y-1)/threads.y);
    int neighbor_num = 10;// no use

    // float agg_start_time = clock();
    
    // const int numBlocks = dst_num;
    AT_DISPATCH_FLOATING_TYPES(d_input.type(), "computation_aware_backward_tgat_cuda_kernel", ([&] {
                                    computation_aware_backward_tgat_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        src_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        d_output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,
                                        dst_num,
                                        edge_num,
                                        neighbor_num
                                    );
                                }));
    // kernel<<<grids, blocks>>>();
    // float agg_end_time = clock();

    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    return {d_output};
}

/**************************************************************  backward kernel  ******************************************************************************* */
template <typename scalar_t>
__global__ void computation_aware_backward_tgat_cuda_kernel(// temporal
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_num,
                    int edge_num,
                    int neighbor_num
)
{
    int current_dst_id = blockIdx.y * blockDim.y + threadIdx.y;
    int feat_id = blockIdx.x * blockDim.x + threadIdx.x;
    if((current_dst_id>=edge_num) || (feat_id>=dim)){
        return;
    }
    if(current_dst_id<edge_num){
        atomicAdd(&d_output[dst_edges[current_dst_id]][feat_id], 
                d_input[src_edges[current_dst_id]][feat_id]);
    }
}
