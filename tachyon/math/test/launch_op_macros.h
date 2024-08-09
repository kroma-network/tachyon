#ifndef TACHYON_MATH_TEST_LAUNCH_OP_MACROS_H_
#define TACHYON_MATH_TEST_LAUNCH_OP_MACROS_H_

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_logging.h"

#define DEFINE_LAUNCH_UNARY_OP(thread_num, method, type, result_type)   \
  gpuError_t Launch##method(const type* x, result_type* result,         \
                            size_t count) {                             \
    ::tachyon::math::kernels::                                          \
        method<<<(count - 1) / thread_num + 1, thread_num>>>(x, result, \
                                                             count);    \
    gpuError_t error = LOG_IF_GPU_LAST_ERROR("Failed " #method "()");   \
    return error == gpuSuccess                                          \
               ? LOG_IF_GPU_ERROR(gpuDeviceSynchronize(),               \
                                  "Failed gpuDeviceSynchronize()")      \
               : error;                                                 \
  }

#define DEFINE_LAUNCH_BINARY_OP(thread_num, method, type, result_type)         \
  gpuError_t Launch##method(const type* x, const type* y, result_type* result, \
                            size_t count) {                                    \
    ::tachyon::math::kernels::                                                 \
        method<<<(count - 1) / thread_num + 1, thread_num>>>(x, y, result,     \
                                                             count);           \
    gpuError_t error = LOG_IF_GPU_LAST_ERROR("Failed " #method "()");          \
    return error == gpuSuccess                                                 \
               ? LOG_IF_GPU_ERROR(gpuDeviceSynchronize(),                      \
                                  "Failed gpuDeviceSynchronize()")             \
               : error;                                                        \
  }

#endif  // TACHYON_MATH_TEST_LAUNCH_OP_MACROS_H_
