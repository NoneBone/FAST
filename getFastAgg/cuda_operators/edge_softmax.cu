#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <time.h>

#define MAX(a, b) ((a < b) ? b : a)

template <typename scalar_t>
__global__ void forward_edge_softmax_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
);
template <typename scalar_t>
__global__ void backward_edge_softmax_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input// gradient to be calculated
);
// optmize kernel 
template <typename scalar_t>
__global__ void forward_csr_edge_softmax_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
);
template <typename scalar_t>
__global__ void backward_csr_edge_softmax_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input// gradient to be calculated
);

template <typename scalar_t>
__global__ void forward_balanced_edge_softmax_cuda_kernel(
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
);
template <typename scalar_t>
__global__ void backward_balanced_edge_softmax_cuda_kernel(
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input// gradient to be calculated
);
/************************************************************** edge_softmax cuda ******************************************************************************* */
std::vector<torch::Tensor> forward_edge_softmax_cuda(// best version
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int node_num_per_block,
    int num_heads
)
{   
    int edge_num = input_feat.size(0);// 1882
    int node_num = indptr.size(0)-1;// 396
    auto output = torch::zeros({edge_num, num_heads}, torch::kCUDA);// edge_num * 2
    const dim3 threads(num_heads, node_num_per_block, 1);// 32, 2
    const dim3 blocks((node_num+node_num_per_block-1)/node_num_per_block, 1);// 369, 1
    // int shared_memory = 2 * 2 * node_num_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(input_feat.type(), "forward_edge_softmax_cuda_kernel", ([&] {
                                    forward_edge_softmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                        indptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
                                    );
                                }));
    cudaDeviceSynchronize();
    return {output};
}
std::vector<torch::Tensor> backward_edge_softmax_cuda(// best version
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int num_heads
)
{   
    int dimFeat = d_output.size(1);// 2
    int edge_num = d_output.size(0);// 1882
    int node_num = indptr.size(0)-1;// 396
    auto d_input = torch::zeros({edge_num, dimFeat}, torch::kCUDA);// edge_num * 2
    const dim3 threads(num_heads, node_num_per_block, 1);
    const dim3 blocks((node_num + node_num_per_block-1)/node_num_per_block, 1);// 369, 1
    // int shared_memory = 2 * node_num_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "backward_edge_softmax_cuda_kernel", ([&] {
        backward_edge_softmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
            indptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    cudaDeviceSynchronize();
    return {d_input};
}
std::vector<torch::Tensor> forward_balanced_edge_softmax_cuda(// csr
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int edge_num_per_block,
    int reduce_size// no use
)
{   
    int num_heads = input_feat.size(1);// 2
    int edge_num = input_feat.size(0);// 1882
    // int node_num = indptr.size(0)-1;// 369
    int balanceNum = indices.size(0);// less
    auto output = torch::zeros({edge_num, num_heads}, torch::kCUDA);// edge_num * 2
    const dim3 threads(edge_num_per_block, num_heads, 1);// 64, 2, 1
    const dim3 blocks(balanceNum, 1);
    int shared_memory = 2 * 2 * edge_num_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(input_feat.type(), "forward_balanced_edge_softmax_cuda_kernel", ([&] {
                                    forward_balanced_edge_softmax_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        indptr.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        indices.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
                                    );
                                }));
    cudaDeviceSynchronize();
    return {output};
}

std::vector<torch::Tensor> backward_balanced_edge_softmax_cuda(// csr
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int edge_num_per_block,
    int reduce_size
)
{   
    int num_heads = d_output.size(1);// 2
    int edge_num = d_output.size(0);// 1882
    int balanceNum = indices.size(0);// less
    auto d_input = torch::zeros({edge_num, num_heads}, torch::kCUDA);// edge_num * 2
    const dim3 threads(edge_num_per_block, num_heads, 1);// 32, 2
    const dim3 blocks(balanceNum, 1);// less369, 1
    int shared_memory = 2 * edge_num_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "backward_balanced_edge_softmax_cuda_kernel", ([&] {
        backward_balanced_edge_softmax_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
            indptr.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
            d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));
    // cudaDeviceSynchronize();
    
    // std::cout << "=== backward_balanced_edge_softmax_cuda ===" << std::endl;
    // std::cout << "d_input size: [" << d_input.size(0) << ", " << d_input.size(1) << "]" << std::endl;
    
    // std::cout << "d_input requires_grad: " << d_input.requires_grad() << std::endl;
    
    // auto d_input_cpu = d_input.cpu();
    // std::cout << "d_input 2 val: " << std::endl;
    // for (int i = 0; i < std::min(2, (int)d_input_cpu.size(0)); i++) {
    //     for (int j = 0; j < d_input_cpu.size(1); j++) {
    //         std::cout << "  [" << i << ", " << j << "]: " << d_input_cpu[i][j].item<float>() << std::endl;
    //     }
    // }
    cudaDeviceSynchronize();
    return {d_input};
}

std::vector<torch::Tensor> forward_csr_edge_softmax_cuda(// csr
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor input_feat,
    int node_num_per_block,// 7882
    int reduce_size
)
{   
    int num_heads = input_feat.size(1);// 2
    int edge_num = input_feat.size(0);// 1882
    int node_num = indptr.size(0)-1;// 396
    auto output = torch::zeros({edge_num, num_heads}, torch::kCUDA);// edge_num * 2
    auto edge_max = torch::full({node_num, num_heads}, 
                          -std::numeric_limits<float>::infinity(),  // -1e38f
                          torch::dtype(torch::kFloat).device(torch::kCUDA));// torch::zeros({node_num, dimFeat}, torch::kCUDA);// node_num * 2
    auto edge_sum = torch::zeros({node_num, num_heads}, torch::kCUDA);// node_num * 2
    const dim3 threads(reduce_size, num_heads, node_num_per_block);// 32, 2
    const dim3 blocks((node_num + node_num_per_block-1)/node_num_per_block, 1);// 369, 1
    int shared_memory = 2 * 2 * node_num_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(input_feat.type(), "forward_csr_edge_softmax_cuda_kernel", ([&] {
                                    forward_csr_edge_softmax_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        indptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
                                    );
                                }));
    cudaDeviceSynchronize();
    return {output};
}
std::vector<torch::Tensor> backward_csr_edge_softmax_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor d_output,
    torch::Tensor output,
    int node_num_per_block,
    int reduce_size
)
{   
    int num_heads = 2;
    int dimFeat = d_output.size(1);// 2
    int edge_num = d_output.size(0);// 1882
    int node_num = indptr.size(0)-1;// 396
    auto d_input = torch::zeros({edge_num, dimFeat}, torch::kCUDA);// edge_num * 2
    const dim3 threads(reduce_size, num_heads, node_num_per_block);// 32, 2
    const dim3 blocks((node_num + node_num_per_block-1)/node_num_per_block, 1);// 369, 1
    int shared_memory = 2 * node_num_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "backward_csr_edge_softmax_cuda_kernel", ([&] {
        backward_csr_edge_softmax_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
            indptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    cudaDeviceSynchronize();
    return {d_input};
}

/**************************************************************  edge_softmax Kernel ******************************************************************************* */
#define MAX(a, b) ((a < b) ? b : a)
__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    int new_val_int = __float_as_int(val);
    
    do {
        expected = old;
        float old_val = __int_as_float(old);
        if (old_val >= val || isnan(val)) {
            break;
        }
        old = atomicCAS(address_as_int, expected, new_val_int);
    } while (old != expected);
}

template <typename scalar_t>
__global__ void forward_edge_softmax_cuda_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
)
{
    int hid = threadIdx.x;
    int node_id = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (node_id >= indptr.size(0) - 1) return;
    
    int lb = indptr[node_id];
    int lh = indptr[node_id + 1];
    int degree = lh - lb;
    
    float max_val = -1e38f;
    for (int i = 0; i < degree; ++i) {
        int idx = lb + i;
        float val = static_cast<float>(input_feat[idx][hid]);
        max_val = fmaxf(max_val, val);
    }
    
    float exp_sum = 0.0f;
    for (int i = 0; i < degree; ++i) {
        int idx = lb + i;
        float val = static_cast<float>(input_feat[idx][hid]);
        float exp_val = expf(val - max_val);
        exp_sum += exp_val;
    }
    
    float inv_exp_sum = 1.0f / exp_sum;
    for (int i = 0; i < degree; ++i) {
        int idx = lb + i;
        float val = static_cast<float>(input_feat[idx][hid]);
        float softmax_val = expf(val - max_val) * inv_exp_sum;
        output[idx][hid] = static_cast<scalar_t>(softmax_val);
    }
}

template <typename scalar_t>
__global__ void backward_edge_softmax_cuda_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input   // gradient to be calculated
)
{
    int hid = threadIdx.x;
    int node_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (node_id >= indptr.size(0) - 1) return;
    
    int lb = indptr[node_id];
    int lh = indptr[node_id + 1];
    int degree = lh - lb;
    
    // 1. sum_j = Σ_i (output_i * d_output_i)
    float sum_val = 0.0f;
    for (int i = 0; i < degree; ++i) {
        int idx = lb + i;
        float output_val = static_cast<float>(output[idx][hid]);
        float d_output_val = static_cast<float>(d_output[idx][hid]);
        sum_val += output_val * d_output_val;
    }
    
    // 2. d_input_i = output_i * (d_output_i - sum_val)
    for (int i = 0; i < degree; ++i) {
        int idx = lb + i;
        float output_val = static_cast<float>(output[idx][hid]);
        float d_output_val = static_cast<float>(d_output[idx][hid]);
        float grad = output_val * (d_output_val - sum_val);
        d_input[idx][hid] = static_cast<scalar_t>(grad);
    }
}
template <typename scalar_t>
__global__ void forward_balanced_edge_softmax_cuda_kernel(
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> indptr,// batch : featBias
    torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> indices,// edge2node batch * 64
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
)
{
    int tid = threadIdx.x;      // 0 ~ 63
    int hid = threadIdx.y;      // 0 ~ num_heads-1
    int num_heads = blockDim.y;
    int block_size = blockDim.x;

    int node_batch_id = blockIdx.x;   // 0 ~ node_batch_num-1
    int edge_num = indptr[node_batch_id+1] - indptr[node_batch_id];
    int nodeID = 0;
    if (tid < edge_num) {
        nodeID = indices[node_batch_id][tid];
    }
    extern __shared__ float shared_mem[];
    float *s_weightMax = &shared_mem[0];// nodeNum * 2
    float *s_expAll = &shared_mem[block_size * 2];
    s_expAll[nodeID * num_heads + hid] = 0.0f;
    

    // 1. max
    // float weight = -1e38f;
    float input_once = -1e38f;
    int edge_id = indptr[node_batch_id] + tid;
    if (tid < edge_num) {
        input_once = static_cast<float>(input_feat[edge_id][hid]);
        // weight = input_once;
        atomicMaxFloat(&s_weightMax[nodeID * num_heads + hid], input_once);
    }
    __syncthreads();
    
    // 2. exp and sum
    float sum_weight = 0.0f;
    if (tid < edge_num) {
        sum_weight = expf(input_once - s_weightMax[nodeID * num_heads + hid]);
        atomicAdd(&s_expAll[nodeID * num_heads + hid], sum_weight);
    }
    __syncthreads();

    // 3. softmax
    
    if (tid < edge_num) {
        float softmax_val = expf(input_once - s_weightMax[nodeID * num_heads + hid]) / s_expAll[nodeID * num_heads + hid];
        output[edge_id][hid] = static_cast<scalar_t>(softmax_val);
    }
}

template <typename scalar_t>
__global__ void backward_balanced_edge_softmax_cuda_kernel(
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input   // gradient to be calculated
)
{
    int tid = threadIdx.x;      // 0 ~ 63
    int hid = threadIdx.y;      // 0 ~ num_heads-1
    int num_heads = blockDim.y;
    
    int node_batch_id = blockIdx.x;
    
    int edge_start = indptr[node_batch_id];
    int edge_end = indptr[node_batch_id + 1];
    int edge_num = edge_end - edge_start;
    if (edge_num <= 0) {
        return;
    }
    extern __shared__ float shared_mem[];
    float *s_sum = &shared_mem[0];
    int max_nodes = indices.size(1);
    int shared_mem_size = max_nodes * num_heads;
    
    int nodeID = 0;
    if (tid < edge_num) {
        nodeID = indices[node_batch_id][tid];
        if (nodeID < 0 || nodeID >= max_nodes) {
            printf("Warning: nodeID=%d out of range [0, %d) at batch=%d, tid=%d\n", 
                   nodeID, max_nodes, node_batch_id, tid);
            return;
        }
    }
    int sum_idx = nodeID * num_heads + hid;
    if (sum_idx < 0 || sum_idx >= shared_mem_size) {
        printf("Error: sum_idx=%d out of range [0, %d) nodeID=%d, hid=%d\n",
                sum_idx, shared_mem_size, nodeID, hid);
        return;
    }
    s_sum[sum_idx] = 0.0f;
    __syncthreads();
    
    // 1. sum_j = Σ_i (output_i * d_output_i)
    int edge_id = indptr[node_batch_id] + tid;
    if (tid < edge_num) {
        float output_val = static_cast<float>(output[edge_id][hid]);
        float d_output_val = static_cast<float>(d_output[edge_id][hid]);
        float product = output_val * d_output_val;
        
        atomicAdd(&s_sum[nodeID * num_heads + hid], product);
        // if ((blockIdx.x == 0 || blockIdx.x == 1) && blockIdx.y == 0 && tid < 2) {
        //     printf("  product = %.6f * %.6f = %.6f\n", 
        //            output_val, d_output_val, product);
        // }
    }
    __syncthreads();
    
    // 2. d_input_i = output_i * (d_output_i - sum_val)
    if (tid < edge_num) {
        float output_val = static_cast<float>(output[edge_id][hid]);
        float d_output_val = static_cast<float>(d_output[edge_id][hid]);
        float grad = output_val * (d_output_val - s_sum[nodeID * num_heads + hid]);
        d_input[edge_id][hid] = static_cast<scalar_t>(grad);
    }
}

template <typename scalar_t>
__global__ void forward_csr_edge_softmax_cuda_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
)
{
    int tid = threadIdx.x;
    int hid = threadIdx.y;
    int node_id = threadIdx.z;
    int node_batch_id = blockIdx.x;
    int num_heads = blockDim.y;
    int node_num_per_block = blockDim.z;
    int global_node_id = node_batch_id * node_num_per_block + node_id;
    if (global_node_id + 1 > indptr.size(0) - 1) {
        return;
    }
    int lb = indptr[global_node_id];
    int lh = indptr[global_node_id + 1];
    int degree = lh - lb;

    extern __shared__ float shared_mem[];
    float *s_expAll = &shared_mem[0];
    float *s_weightMax = &shared_mem[node_num_per_block*2];
    s_expAll[node_id * num_heads + hid] = 0.0f;

    // 1. max
    float weight = -1e38f;
    float input_once = 0.0f;
    if (tid < degree) {
        int idx = lb + tid;
        input_once = static_cast<float>(input_feat[idx][hid]);
        weight = input_once;
    }
    // max reduce
    __syncwarp();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32);
        weight = MAX(tmp, weight);
    }
    
    if (tid == 0) {
        s_weightMax[node_id * num_heads+ hid] = weight;
    }
    __syncwarp();
    
    // 2. exp and sum
    float sum_weight = 0.0f;
    if (tid < degree) {
        sum_weight = expf(input_once - s_weightMax[node_id * num_heads+ hid]);
    }
    // add reduce
    __syncwarp();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, sum_weight, stride, 32);
        sum_weight += tmp;
    }

    if (tid == 0){
        s_expAll[node_id * num_heads + hid] = sum_weight;
    }
    // 3. softmax
    if (tid < degree) {
        int idx = lb + tid;
        float w = input_once;
        float softmax_val = expf(w - s_weightMax[node_id * num_heads+ hid]) / s_expAll[node_id * num_heads + hid];
        output[idx][hid] = static_cast<scalar_t>(softmax_val);
    }
}

template <typename scalar_t>
__global__ void backward_csr_edge_softmax_cuda_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input   // gradient to be calculated
)
{
    int tid = threadIdx.x;
    int hid = threadIdx.y;
    int node_id = threadIdx.z;

    int node_batch_id = blockIdx.x;
    int num_heads = blockDim.y; 
    int node_num_per_block = blockDim.z;
    int global_node_id = node_batch_id * node_num_per_block + node_id;
    if (global_node_id+1 > indptr.size(0) - 1) {
        return;
    }
    int lb = indptr[global_node_id];
    int lh = indptr[global_node_id + 1];
    int degree = lh - lb;
    
    extern __shared__ float s_data[];
    float* s_sum = &s_data[0];
    s_sum[node_id * num_heads + hid] = 0.0f;
    __syncwarp();// sync after smem using

    // 1. sum_j = Σ_i (output_i * d_output_i)
    float product = 0.0f;
    int idx = 0;
    float d_output_once = 0.0f;
    if (tid < degree) {
        idx = lb + tid;
        d_output_once = static_cast<float>(d_output[idx][hid]);
        product = static_cast<float>(output[idx][hid]) * d_output_once;
        // if ((blockIdx.x == 0 || blockIdx.x == 1) && blockIdx.y == 0 && tid < 2) {
        //     printf("  product = %.6f * %.6f = %.6f\n", 
        //            output_val, d_output_val, product);
        // }
    }

    // 2. sum reduce    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, product, stride, 32);
        product += tmp;
    }
    if (tid == 0){
        s_sum[node_id * num_heads + hid] = product;
    }
    __syncwarp();// sync after smem using

    // 3. d_input_i = output_i * (d_output_i - sum_val)
    if (tid < degree) {
        idx = lb + tid;
        d_input[idx][hid] = static_cast<scalar_t>(static_cast<float>(output[idx][hid]) * (d_output_once - s_sum[node_id * num_heads + hid]));
    }
}