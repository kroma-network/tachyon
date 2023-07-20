#ifndef TACHYON_DEVICE_GPU_GPU_DEVICE_FUNCTIONS_H_
#define TACHYON_DEVICE_GPU_GPU_DEVICE_FUNCTIONS_H_

#if TACHYON_CUDA
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // TACHYON_CUDA

#if TACHYON_CUDA
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventDestroy cudaEventDestroy
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuFree cudaFree
#elif TACHYON_USE_ROCM
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using cudaError = int;
using cudaError_t = int;
#define cudaSuccess 0
#define cudaGetLastError hipGetLastError
#define gpuEventRecord hipEventRecord
#define gpuEventDestroy hipEventDestroy
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuFree hipFree
static std::string cudaGetErrorString(int err) { return std::to_string(err); }
#endif

#endif  // TACHYON_DEVICE_GPU_GPU_DEVICE_FUNCTIONS_H_
