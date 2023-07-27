#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_H_

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon {
namespace math {
namespace kernels {

TACHYON_EXPORT bool SetMSMKernelAttributes();

}  // namespace kernels
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_KERNELS_H_
