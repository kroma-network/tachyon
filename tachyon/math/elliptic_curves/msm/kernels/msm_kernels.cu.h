#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_H_

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon::math::kernels {

TACHYON_EXPORT bool SetMSMKernelAttributes();

}  // namespace tachyon::math::kernels

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_H_
