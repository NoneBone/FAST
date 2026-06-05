// the fellowing no use in fast
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define CURAND_CALL(x)                                                         \
  do {                                                                         \
    curandStatus_t _status = (x);                                              \
    if (_status != CURAND_STATUS_SUCCESS) {                                    \
      printf("Error at %s:%d, CURAND error code: %d\n", __FILE__, __LINE__, _status); \
      throw std::runtime_error("CURAND error");                                \
    }                                                                          \
  } while (0)

#define MAX(a, b) ((a < b) ? b : a)
#define LeakyRelu(x, negative_slope) ((x > 0) ? (x) : ((x)*negative_slope))
constexpr size_t WARP_SIZE_indegree = 16; 
constexpr size_t SHAPE_Z = 32;

template <typename scalar_t>
__global__ void forward_fused_attn_cuda_kernel(
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> V_val,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> attn,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out_agg,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> edge_mask,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_eid,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_eid_9764,
                    int edge_num,
                    int node_num_396,
                    int node_num_7882,
                    int num_heads,
                    int shape3,
                    float negative_slope,
                    float attn_drop
);
template <typename scalar_t>
__global__ void backward_fused_attn_cuda_kernel(
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
                    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> V_val,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_attn,
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> edge_mask,
                    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> edge_tmp,
                    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_V,   // gradient to be calculated for V
                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,   // gradient to be calculated
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_eid,
                    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_eid_9764,
                    int edge_num,
                    int node_num_396,
                    int node_num_9764,
                    int num_heads,
                    float negative_slope,
                    float attn_drop
);

/**************************************************************  edge_softmax CUDA ******************************************************************************* */
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
)
{   
    int dimFeat = input_feat.size(1);// 2
    int shape3 = V_val.size(2);// 1882, 2, 50
    int edge_num = indices.size(0);// 1882
    int node_num_396 = indptr.size(0)-1;// 369
    auto attn = torch::zeros({edge_num, dimFeat}, torch::kCUDA);// edge_num * 2
    auto out_tmp = torch::zeros({edge_num, dimFeat*shape3}, torch::kCUDA);// edge_num_1882 * 100
    auto out_agg = torch::zeros({node_num_7882, dimFeat*shape3}, torch::kCUDA);// node_num_7882 * 100
    // auto zeroFeat = torch::zeros({node_num_7882, dimFeat*shape3}, torch::kCUDA);// 7882*100 zero

    // dropout mask
    auto edge_mask = torch::empty({edge_num, dimFeat}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    float* edge_mask_ptr = edge_mask.data_ptr<float>();
    long seed = clock();
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CALL(curandGenerateUniform(gen, edge_mask_ptr, edge_num * dimFeat));

    const dim3 threads(WARP_SIZE_indegree, num_heads, SHAPE_Z);// 32, 2
    const dim3 blocks(node_num_396, 1, (shape3+SHAPE_Z-1)/SHAPE_Z);// 369, 1
    int shared_memory = 2 * num_heads * sizeof(float) + num_heads * SHAPE_Z * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(input_feat.scalar_type(), "forward_fused_attn_cuda_kernel", ([&] {
                                    forward_fused_attn_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        input_feat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        V_val.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                        attn.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), 
                                        out_tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), 
                                        out_agg.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), 
                                        edge_mask.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), 
                                        indptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        indices.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_eid.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        src_eid_9764.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                                        edge_num,
                                        node_num_396,
                                        node_num_7882,
                                        num_heads,
                                        shape3,
                                        negative_slope,
                                        attn_drop
                                    );
                                }));
    cudaDeviceSynchronize();
    cudaError_t sync_error = cudaGetLastError();
    
    if (sync_error != cudaSuccess) {
        printf("forward_fused_attn_cuda CUDA sync error: %s\n", cudaGetErrorString(sync_error));
        printf("Error code: %d\n", sync_error);
        std::cout << "input_feat shape: (" << input_feat.size(0) << ", " 
              << input_feat.size(1) << ")" << std::endl;
        std::cout << "attn shape: (" << attn.size(0) << ", " 
              << attn.size(1) << attn.size(2) << ")" << std::endl;
        std::cout << "output shape: (" << out_agg.size(0) << ", " 
              << out_agg.size(1) << out_agg.size(2) << ")" << std::endl;
    }
    // return {torch::cat({zeroFeat, output}, /*dim=*/0)};
    return {out_agg, attn, edge_mask};
}
std::vector<torch::Tensor> backward_fused_attn_cuda(
    torch::Tensor d_output_agg,// 7882 * 100
    torch::Tensor V_val,// 1882 * 2 * 50
    torch::Tensor output_attn,
    torch::Tensor edge_mask,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor dst_eid,
    torch::Tensor src_eid_9764,
    int node_num_9764,
    int num_heads,
    float negative_slope,
    float attn_drop
)
{   
    int dimFeat = V_val.size(1);// 2
    int shape3 = V_val.size(2);
    int edge_num = indices.size(0);// 1882
    int node_num_396 = indptr.size(0)-1;// 396
    // auto d_input = torch::zeros({edge_num, dimFeat}, torch::kCUDA);// edge_num * 2
    auto d_input = torch::zeros({edge_num, dimFeat}, torch::kCUDA);// edge_num * 2
    auto edge_tmp = torch::zeros({edge_num, dimFeat, shape3}, torch::kCUDA);// 1882 * 2 * 50
    auto grad_V = torch::zeros({edge_num, dimFeat, shape3}, torch::kCUDA);// 9764 * 100
    const dim3 threads(WARP_SIZE_indegree, num_heads, SHAPE_Z);// 16, 2， 32
    const dim3 blocks(node_num_396, 1, (shape3+SHAPE_Z-1)/SHAPE_Z);// 369, 1， 2
    int shared_memory = 2 * num_heads * sizeof(float) + num_heads * SHAPE_Z * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(d_output_agg.scalar_type(), "backward_fused_attn_cuda_kernel", ([&] {
        backward_fused_attn_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
            d_output_agg.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            V_val.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            output_attn.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            edge_mask.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            edge_tmp.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grad_V.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            indptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            dst_eid.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            src_eid_9764.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            edge_num,
            node_num_396,
            node_num_9764,
            num_heads,
            negative_slope,
            attn_drop
        );
    }));
    cudaDeviceSynchronize();
    cudaError_t sync_error = cudaGetLastError();
    
    if (sync_error != cudaSuccess) {
        printf("backward_fused_attn_cuda CUDA sync error: %s\n", cudaGetErrorString(sync_error));
        printf("Error code: %d\n", sync_error);
        std::cout << "d_output_agg shape: (" << d_output_agg.size(0) << ", " 
              << d_output_agg.size(1) << ")" << std::endl;
        std::cout << "grad_V shape: (" << grad_V.size(0) << ", " 
              << grad_V.size(1) << grad_V.size(2) << ")" << std::endl;
        std::cout << "d_input shape: (" << d_input.size(0) << ", " 
              << d_input.size(1) << ")" << std::endl;
    }
    return {d_input, grad_V};
}

/**************************************************************  fused Kernel ******************************************************************************* */
template <typename scalar_t>
__global__ void forward_fused_attn_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_feat,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> V_val,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> attn,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out_tmp,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out_agg,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> edge_mask,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_eid,
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_eid_9764,
    int edge_num,
    int node_num_396,
    int node_num_7882,
    int num_heads,
    int shape3,
    float negative_slope,
    float attn_drop
)
{
    int tid = threadIdx.x;      // 0 ~ blockDim.x-1
    int hid = threadIdx.y;      // 0 ~ num_heads-1
    int node_id = blockIdx.x;   // 0 ~ node_num_7882-1
    int sid = blockIdx.z * blockDim.z+ threadIdx.z;// 0 ~ 63(49)
    int lb = indptr[node_id];
    int lh = indptr[node_id+1];
    int indegree = lh - lb;
    
    if (node_id >= node_num_396) {
        return;
    }
    
    __shared__ float weightMax[2];
    __shared__ float expAll[2];
    float *s_expAll = &expAll[hid];
    float *s_weightMax = &weightMax[hid];
    s_expAll[0] = 0.0f;

    // 1. max
    float weight = -1e38f;
    if (tid < indegree) {
        int idx = indices[lb + tid];
        weight = static_cast<float>(input_feat[idx][hid]);
        weight = LeakyRelu(weight, negative_slope);
    }
    // max reduce
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, weight, stride, 32);
        weight = MAX(tmp, weight);
    }
    
    if (tid == 0) {
        s_weightMax[0] = weight;
    }
    __syncthreads();
    
    // 2. exp and sum
    float sum_weight = 0.0f;
    if (tid < indegree) {
        int idx = indices[lb + tid];
        sum_weight = LeakyRelu(static_cast<float>(input_feat[idx][hid]), negative_slope);
        sum_weight = expf(sum_weight - s_weightMax[0]);
    }
    // add reduce
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, sum_weight, stride, 32);
        sum_weight += tmp;
    }
    if (tid == 0){
        s_expAll[tid] = sum_weight;
    }
    // 3. softmax
    float softmax_val = 0.0f;
    if (tid < indegree) {
        int idx = indices[lb + tid];
        float w = static_cast<float>(input_feat[idx][hid]);
        w = LeakyRelu(w, negative_slope);
        softmax_val = expf(w - s_weightMax[0]) / s_expAll[0];
        attn[idx][hid] = static_cast<scalar_t>(softmax_val);// save for backward
        if(edge_mask[idx][hid]>attn_drop && sid < shape3){// dropout
                // out_tmp[idx][hid][sid] = static_cast<scalar_t>(softmax_val) * static_cast<scalar_t>(V_val[idx][hid][sid]);
                out_tmp[idx][hid * shape3 + sid] = static_cast<scalar_t>(softmax_val / (1.0 - attn_drop)) * static_cast<scalar_t>(V_val[idx][hid][sid]);
        }
    }
    // 4. aggregator 
    int _feat_id = blockIdx.z * blockDim.z + threadIdx.y * gridDim.z * blockDim.z + threadIdx.z; // blockIdx.y * blockDim.y + threadIdx.y;// 0~127
    int _sid = threadIdx.y*blockDim.z+threadIdx.z; // 0~63 for agg
    if((node_id>=node_num_396) || (_feat_id>=out_agg.size(1))){
        return;
    }

    __shared__ float shared_space[SHAPE_Z * 2];
    float *agg_sum = shared_space;
    
    // sum
    if (tid == 0)
    {
        float _sum = 0.0f;
#pragma unroll
        for (int i = 0; i < indegree; i++)
        { // loop in one thread，not banlance now
            int src_idx = src_eid_9764[lb + i];
            if (src_idx < node_num_7882 + edge_num && src_idx >= node_num_7882)
            {
                _sum += out_tmp[src_idx - node_num_7882][_feat_id];
            }
        }
        agg_sum[_sid] = _sum;
    }

    __syncthreads();

    // write
    if (indegree > 0 && _feat_id < out_agg.size(1)) {
#pragma unroll
        for (int i = 0; i < indegree; i++) {
            int edge_dst_idx = dst_eid[lb + i];
            if(edge_dst_idx >= 0 && edge_dst_idx < out_agg.size(0)) {
                out_agg[edge_dst_idx][_feat_id] = agg_sum[_sid];
            }
        }
    }
}

template <typename scalar_t>
__global__ void backward_fused_attn_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> V_val,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_attn,    // edge_num * 2,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> edge_mask,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> edge_tmp,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_V,   // gradient to be calculated for V
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,   // gradient to be calculated
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indptr,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> dst_eid,
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> src_eid_9764,
    int edge_num,
    int node_num_396,
    int node_num_9764,
    int num_heads,
    float negative_slope,
    float attn_drop
)
{
    // AGG Grad Calculate
    int node_id = blockIdx.x;
    int _feat_id = blockIdx.z * blockDim.z + threadIdx.y * gridDim.z * blockDim.z + threadIdx.z;
    int _sid = threadIdx.y * blockDim.z + threadIdx.z;
    int hid = threadIdx.y;
    int tid = threadIdx.x;
    int dimFeat = d_output.size(1) / V_val.size(2);
    int node_num_7882 = d_output.size(0);
    
    if((node_id >= node_num_396) || (_feat_id >= d_output.size(1))){
        return;
    }

    int lb = indptr[node_id];
    int lh = indptr[node_id+1];
    int indegree = lh - lb;

    __shared__ float shared_space[SHAPE_Z * 2];
    float *agg_sum = shared_space;
    
    if (tid == 0)
    {
        float grad_sum = 0.0f;
        for (int i = 0; i < indegree; i++) {
            int src_idx = dst_eid[lb + i];
            if(src_idx >= 0 && src_idx < node_num_7882) {
                grad_sum += d_output[src_idx][_feat_id];
            }
        }
        agg_sum[_sid] = grad_sum;
    }
    __syncthreads();

    int shape3 = V_val.size(2);
    int edge_feat_idx = _feat_id / shape3;
    
    if (indegree > 0 && _feat_id < d_output.size(1)) {
        for (int i = 0; i < indegree; i++) {
            int edge_dst_idx = src_eid_9764[lb + i];
            if(edge_dst_idx >= node_num_7882 && edge_dst_idx < node_num_9764) {
                int biasIdx = edge_dst_idx - node_num_7882;
                int feat_in_head = hid;
                
                if(edge_mask[biasIdx][feat_in_head] > attn_drop) {
                    int local_sid = threadIdx.z;
                    // int stride = blockDim.z;
                    float grad_drop = agg_sum[_sid] / (1.0f - attn_drop);
                    
                    if(local_sid < shape3) {
                        atomicAdd(&edge_tmp[biasIdx][feat_in_head][local_sid], 
                                grad_drop * static_cast<float>(V_val[biasIdx][feat_in_head][local_sid]));
                    }
                    
                    if(local_sid == 0) {
                        grad_V[biasIdx][feat_in_head][edge_feat_idx] = 
                            grad_drop * static_cast<scalar_t>(output_attn[biasIdx][feat_in_head]);
                    }
                }
            }
        }
    }
    __syncthreads();
    // ESM Grad Calculate
    if (node_id >= node_num_396) {
        return;
    }
    
    extern __shared__ float s_data[2];
    float* s_sum = &s_data[hid];
    s_sum[0] = scalar_t(0);// 0.0f;

    // 1. sum_j = Σ_i (output_i * d_output_i)
    float product = 0.0f;
    int idx = 0;
    if (tid < indegree) {
        idx = indices[lb + tid];
        // product = static_cast<float>(output_attn[idx][hid]) * static_cast<float>(d_output[idx][hid]);
        product = static_cast<float>(output_attn[idx][hid]) * static_cast<float>(edge_tmp[idx][hid][0]);
    }
    // sum reduce    
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, product, stride, 32);
        product += tmp;
    }
    if (tid == 0){
        s_sum[0] = product;
    }
    __syncthreads();
    
    // 2. d_input_i = output_i * (d_output_i - sum_val)
    if (tid < indegree) {
        idx = indices[lb + tid];
        d_input[idx][hid] = static_cast<scalar_t>(static_cast<float>(output_attn[idx][hid]) * (static_cast<float>(edge_tmp[idx][hid][0]) - s_sum[0]));
    }
}
