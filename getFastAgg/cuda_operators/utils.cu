#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
// util
#include <curand_kernel.h>
#include <cub/cub.cuh>

constexpr size_t BLOCK_SIZE = 128;

/*
Sequential accumulation and degree calculation, used for parallelization of node centers
*/
at::Tensor exclusive_sum_cuda(
    torch::Tensor in_degs
){
    // auto in_degs_pad = torch::zeros({1}, torch::kCUDA).to(at::kInt);
    auto in_degs_pad = torch::cat({in_degs,torch::zeros({1}, torch::kCUDA).to(at::kInt)});// all 1
    int num_items = in_degs_pad.size(0);// 172266
    int* input_data = in_degs_pad.data<int>();

    auto edge_ptr = torch::zeros({num_items}, torch::kCUDA).to(at::kInt);// out: 172266

    int* output_ptr = edge_ptr.data<int>();

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_ptr, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_ptr, num_items);

    // cudaFree(input_data);
    cudaFree(d_temp_storage);

    return edge_ptr;
}

__global__ void cal_deg_cuda_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> deg,
    int edge_num

){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=edge_num){
        return;
    }

    int id = edges[tid];

    atomicAdd((int*)&deg[id],1);
}

at::Tensor cal_deg_cuda(
    torch::Tensor edges,
    int nodes_num
){
    int edge_num = edges.size(0);
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((edge_num+threads.x-1)/threads.x);
    int device = edges.get_device();
    auto deg = torch::zeros({nodes_num}).to(at::Device(at::kCUDA, device)).to(at::kInt);

    AT_DISPATCH_ALL_TYPES(deg.type(), "cal_deg_cuda_kernel", ([&] {
                                    cal_deg_cuda_kernel<<<blocks, threads>>>(
                                        edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        deg.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_num
                                    );
                                }));

    return deg;
}

// template <typename scalar_t>
__global__ void getBalancedPtrKernel(
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> densePtr, 
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> output, 
    int n, 
    int threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n - 1) return;
    
    int degree = static_cast<int>(densePtr[idx + 1]) - static_cast<int>(densePtr[idx]);
    
    extern __shared__ int s_data[];
    int* s_degrees = s_data;
    int* s_prefix = s_data + blockDim.x;
    
    s_degrees[threadIdx.x] = degree;
    __syncthreads();
    
    s_prefix[threadIdx.x] = s_degrees[threadIdx.x];
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        int val = (threadIdx.x >= stride) ? s_prefix[threadIdx.x - stride] : 0;
        __syncthreads();
        s_prefix[threadIdx.x] += val;
    }
    __syncthreads();
    
    int total_before = (threadIdx.x > 0) ? s_prefix[threadIdx.x - 1] : 0;
    int total_after = s_prefix[threadIdx.x];
    
    output[idx] = (total_after / threshold > total_before / threshold) ? 1 : 0;
}

at::Tensor get_balanced_cuda(
    torch::Tensor densePtr,
    int threshold
){
    int num = densePtr.size(0);
    const dim3 threads(256);
    const dim3 blocks((num+threads.x-1)/threads.x);
    const int shared_mem = 2 * threads.x * sizeof(int);
    int device = densePtr.get_device();

    auto mask = torch::zeros({num-1}).to(at::Device(at::kCUDA, device)).to(at::kInt);

    AT_DISPATCH_ALL_TYPES(densePtr.type(), "getBalancedPtrKernel", ([&] {
                                    getBalancedPtrKernel<<<blocks, threads, shared_mem>>>(
                                        densePtr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        mask.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        num,
                                        threshold
                                    );
                                }));
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }
    return mask;
}