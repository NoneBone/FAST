#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <time.h>

#define SHD_METHOD 0 // 1 OR 0

template <typename scalar_t>
__global__ void computation_aware_forward_tgat_shd_cuda_kernel(// static2temporal
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int dim,
                    int num_node,
                    int dst_nodes
);
template <typename scalar_t>
__global__ void computation_aware_backward_tgat_shd_cuda_kernel(// static2temporal
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int num_node,
                    int dst_nodes
);

/**************************************************************  Forward cuda ******************************************************************************* */
std::vector<torch::Tensor> computation_aware_forward_tgat_shd_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor input_feat,
    int dst_nodes,
    int aggNodePB,
    int aggFeat  
)
{   
# if SHD_METHOD
    int dim = input_feat.size(1);// 128
    int num_node = indptr.size(0)-1;
    int unit_dim = dim/num_heads;// 64
    auto output = torch::zeros({dst_nodes, dim}, torch::kCUDA);
    int shared_memory = 1 * sizeof(float);
    const dim3 threads(DEGREE_SIZE, FEAT_PER_BLOCK);
    const dim3 blocks((num_node)/1, (dim+threads.y-1)/threads.y);
# else
    int dim = input_feat.size(1);// 128
    int num_node = indptr.size(0)-1;
    auto output = torch::zeros({dst_nodes, dim}, torch::kCUDA);// dstN * 128
    int shared_memory = aggNodePB * aggFeat * sizeof(float);// 8*32*4 = 1024 Byte
    const dim3 threads(aggNodePB, aggFeat); // x, 32
    const dim3 blocks((num_node+threads.x-1)/threads.x, (dim+threads.y-1)/threads.y);// 369, 1
# endif
    // float agg_start_time = clock();
    AT_DISPATCH_FLOATING_TYPES(input_feat.scalar_type(), "computation_aware_forward_tgat_shd_cuda_kernel", ([&] {
                                    computation_aware_forward_tgat_shd_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        indptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        indices.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,// 128
                                        num_node,// 64
                                        dst_nodes// 7882
                                    );
                                }));
    // kernel<<<grids, blocks>>>();
    // float agg_end_time = clock();

    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    cudaDeviceSynchronize();
    cudaError_t sync_error = cudaGetLastError();
    
    if (sync_error != cudaSuccess) {
        printf("computation_aware_forward_tgat_shd_cuda_kernel CUDA sync error: %s\n", cudaGetErrorString(sync_error));
        printf("Error code: %d\n", sync_error);
        std::cout << "input_feat shape: (" << input_feat.size(0) << ", " 
              << input_feat.size(1) << ")" << std::endl;
        std::cout << "output shape: (" << output.size(0) << ", " 
              << output.size(1) << ")" << std::endl;
    }
    return {output};
}
/**************************************************************  Forward Kernel ******************************************************************************* */
template <typename scalar_t>
__global__ void computation_aware_forward_tgat_shd_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,// no use
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int dim,
                    int num_node,
                    int dst_nodes
)
{
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 ){
    //         printf("blockDim=(%d,%d,%d), gridDim=(%d,%d,%d)\n",
    //        blockDim.x, blockDim.y, blockDim.z,
    //        gridDim.x, gridDim.y, gridDim.z);
    // }
# if SHD_METHOD
    int node_id = blockIdx.x;// node 0~367
    int sid = threadIdx.y;// shared 0 ~ 31
    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;// 0~127

    int tid = threadIdx.x; // 0 ~ 9

    if((node_id>=num_node) || (feat_id>=dim)){
        return;
    }

    int lb = indptr[node_id];
    int lh = indptr[node_id+1];
    int src_num = lh - lb;

    extern __shared__ float shared_space[];
    float *partial_sum = (float*)&shared_space[0];// 0 ~ 31 * 16

    // read
    float sum = 0.0f;
    if(tid < src_num) {
        int src_idx = dst_edges[lb + tid];
        
        if(src_idx >= 0 && src_idx < input_feat.size(0) && feat_id < dim){
            sum = input_feat[src_idx][feat_id];
        }
    }

    // reduce
    __syncthreads();
    for (int stride = DEGREE_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < src_num && tid + stride < src_num) {
            sum += __shfl_xor_sync(0xffffffff, sum, stride, 32);
        }
    }
    if (tid == 0){
        partial_sum[0] = sum;
    }
    __syncthreads();

    // write
    if( tid < src_num) {
        int dst_idx = src_edges[lb + tid];
        if(dst_idx >= 0 && dst_idx < output.size(0) && feat_id < dim) {
            output[dst_idx][feat_id] = partial_sum[0];
        }
    }
# else 
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;// node 0~367
    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;// 0~127
    
    int fid = threadIdx.y;// 0
    int tid = threadIdx.x;// 0~127
    
    if((node_id>=num_node) || (feat_id>=dim)){
        return;
    }

    int lb = indptr[node_id];
    int lh = indptr[node_id+1];
    int src_num = lh - lb;

    extern __shared__ float shared_space[];
    float *partial_output = shared_space;
    
    // sum
    // float sum = 0.0f;
    #pragma unroll
    int shared_idx = tid * blockDim.y + fid;
    if (shared_idx < blockDim.x * blockDim.y) {
        partial_output[shared_idx] = 0.0f;
    }
    __syncthreads();

    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < src_num; i++) {
        int src_idx = dst_edges[lb + i];
        if(src_idx < input_feat.size(0)) {
            thread_sum += input_feat[src_idx][feat_id];
        }
    }

    partial_output[shared_idx] = thread_sum;
    __syncthreads();
    
    // write
    if (src_num > 0 && feat_id < dim) {
        for (int i = 0; i < src_num; i++) {
            int edge_dst_idx = src_edges[lb + i];
            if(edge_dst_idx >= 0 && edge_dst_idx < output.size(0)) {
                output[edge_dst_idx][feat_id] = partial_output[shared_idx];
            }
        }
    }
# endif
}
/**************************************************************  backward cuda  ******************************************************************************* */
std::vector<torch::Tensor> computation_aware_backward_tgat_shd_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_edges,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,// 9764
    int aggNodePB,
    int aggFeat
)
{   
# if SHD_METHOD
    int dst_nodes = d_input.size(0);// 7882
    int feat_dim = d_input.size(1);// 128
    int num_node = indptr.size(0)-1;
    auto d_output = torch::zeros({src_nodes, feat_dim}, torch::kCUDA);
    int shared_memory = 16 * FEAT_PER_BLOCK * sizeof(float);// 8*10*4 + 8*32*4 = 1344 B = 1.25 KB
    const dim3 threads(DEGREE_SIZE, FEAT_PER_BLOCK); // 10,32
    const dim3 blocks((num_node)/1, (feat_dim+threads.y-1)/threads.y);// 368,4
# else
    int dst_nodes = d_input.size(0);// 7882
    int feat_dim = d_input.size(1);// 128
    int num_node = indptr.size(0)-1;
    auto d_output = torch::zeros({src_nodes, feat_dim}, torch::kCUDA);
    int shared_memory = aggNodePB * aggFeat * sizeof(float);// 8*32*4 = 1024 Byte
    const dim3 threads(aggNodePB, aggFeat); // 8, 32
    const dim3 blocks((num_node+threads.x-1)/threads.x, (feat_dim+threads.y-1)/threads.y);// 47, 4
# endif
    AT_DISPATCH_FLOATING_TYPES(d_input.scalar_type(), "computation_aware_backward_tgat_shd_cuda_kernel", ([&] {
                                    computation_aware_backward_tgat_shd_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        indptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        indices.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        d_output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        feat_dim,
                                        num_node,
                                        dst_nodes// 7882
                                    );
                                }));
    cudaDeviceSynchronize();
    cudaError_t sync_error = cudaGetLastError();
    if (sync_error != cudaSuccess) {
        printf("computation_aware_backward_tgat_shd_cuda_kernel CUDA sync error: %s\n", cudaGetErrorString(sync_error));
        printf("Error code: %d\n", sync_error);
        std::cout << "d_input shape: (" << d_input.size(0) << ", " 
              << d_input.size(1) << ")" << std::endl;
        std::cout << "d_output shape: (" << d_output.size(0) << ", " 
              << d_output.size(1) << ")" << std::endl;
    }
    return {d_output};
}
/**************************************************************  backward kernel  ******************************************************************************* */
template <typename scalar_t>
__global__ void computation_aware_backward_tgat_shd_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int num_node,
                    int dst_nodes
)
{
# if SHD_METHOD
    int node_id = blockIdx.x; // node 0~367
    int sid = threadIdx.y;    // shared 0 ~ 31
    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;// 0 ~ 100/128-1
    int tid = threadIdx.x;    // 0 ~ 15

    if((node_id >= num_node) || (feat_id >= dim)) {
        return;
    }

    int lb = indptr[node_id];
    int lh = indptr[node_id+1];
    int src_num = lh - lb;

    extern __shared__ float shared_space[];
    float *partial_sum = &shared_space[sid * blockDim.x];

    // read
    if(tid < src_num) {
        int src_idx = dst_edges[lb + tid];
        
        if(src_idx >= 0 && src_idx < d_input.size(0) && feat_id < dim){
            partial_sum[tid] = d_input[src_idx][feat_id];
        }
    } else {
        partial_sum[tid] = 0.0f;
    }
    __syncthreads();

    // reduce
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(tid < stride && tid + stride < src_num) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // write
    if( tid < src_num) {
        int dst_idx = src_edges[lb + tid];
        if(dst_idx >= 0 && dst_idx < d_output.size(0) && feat_id < dim) {
            d_output[dst_idx][feat_id] = partial_sum[0];
        }
    }
# else
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;// node 0~367
    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;
    
    int bid = threadIdx.y; // 0~7
    int tid = threadIdx.x; // 0~31
    
    if((node_id >= num_node) || (feat_id >= dim)){
        return;
    }

    int lb = indptr[node_id];
    int lh = indptr[node_id+1];
    int src_num = lh - lb;

    extern __shared__ float shared_space[];
    float *node_grad_sums = shared_space;
    node_grad_sums[tid] = 0.0f;
    
    float grad_sum = 0.0f;
    for (int i = 0; i < src_num; i++) {
        int src_idx = dst_edges[lb + i];
        if(src_idx >= 0 && src_idx < d_input.size(0) && feat_id < dim) {
            // A simple addition aggregation, other aggregation methods can be extended.
            float val = d_input[src_idx][feat_id];
            grad_sum = fmaf(val, 1.0f, grad_sum);  
        }
    }
    __syncthreads();

    int node_offset_in_block = tid;
    if (bid == 0) {
        node_grad_sums[node_offset_in_block] = grad_sum;
    }
    __syncthreads();
    
    if (src_num > 0 && feat_id < dim) {
        for (int i = 0; i < src_num; i++) {
            int edge_dst_idx = src_edges[lb + i];
            if(edge_dst_idx >= 0 && edge_dst_idx < d_output.size(0)) {
                d_output[edge_dst_idx][feat_id] = grad_sum;
            }
        }
    }
# endif
}
