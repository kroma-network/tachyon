#ifndef TACHYON_DEVICE_GPU_GPU_DEVICE_FUNCTIONS_H_
#define TACHYON_DEVICE_GPU_GPU_DEVICE_FUNCTIONS_H_

#if TACHYON_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // TACHYON_CUDA

// clang-format off
#if TACHYON_CUDA
using gpuError_t = cudaError_t;
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString

#define gpuDeviceReset cudaDeviceReset
#define gpuDeviceSynchronize cudaDeviceSynchronize

using gpuEvent_t = cudaEvent_t;
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDestroy cudaEventDestroy
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize

#if CUDA_VERSION >= 11020  // CUDA 11.2
// See https://developer.nvidia.com/blog/enhancing-memory-allocation-with-new-cuda-11-2-features/
using gpuMemPool_t = cudaMemPool_t;
using gpuMemPoolProps = cudaMemPoolProps;
#define gpuMemPoolCreate cudaMemPoolCreate
#define gpuMemPoolDestroy cudaMemPoolDestroy
#define gpuMemPoolSetAttribute cudaMemPoolSetAttribute
#define gpuMemPoolGetAttribute cudaMemPoolGetAttribute
#endif

using gpuStream_t = cudaStream_t;
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent

#define gpuMalloc cudaMalloc
#define gpuMallocHost cudaMallocHost
#define gpuMallocFromPoolAsync cudaMallocFromPoolAsync

#define gpuFree cudaFree
#define gpuFreeHost cudaFreeHost
#define gpuFreeAsync cudaFreeAsync

#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset

#define gpuMemGetInfo cudaMemGetInfo
#elif TACHYON_USE_ROCM
using cudaError = int;
using cudaError_t = int;
#define cudaSuccess 0
#define cudaGetLastError hipGetLastError
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString cudaGetErrorString
static std::string cudaGetErrorString(int err) { return std::to_string(err); }

using gpuError_t = int;

#define gpuDeviceSynchronize hipDeviceSynchronize

using gpuEvent_t = hipEvent_t;
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDestroy hipEventDestroy
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize

using gpuMemPool_t = hipMemPool_t;
using gpuMemPoolProps = hipMemPoolProps;
#define gpuMemPoolCreate hipMemPoolCreate
#define gpuMemPoolDestroy hipMemPoolDestroy
#define gpuMemPoolSetAttribute hipMemPoolSetAttribute
#define gpuMemPoolGetAttribute hipMemPoolGetAttribute

using gpuStream_t = hipStream_t;
#define gpuStreamCreate hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent

#define gpuMalloc hipMalloc
#define gpuMallocHost hipMallocHost
#define gpuMallocFromPoolAsync hipMallocFromPoolAsync

#define gpuFree hipFree
#define gpuFreeHost hipFreeHost
#define gpuFreeAsync hipFreeAsync

#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset

#define gpuMemGetInfo hipMemGetInfo
#endif
// clang-format on

#endif  // TACHYON_DEVICE_GPU_GPU_DEVICE_FUNCTIONS_H_
