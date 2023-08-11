#ifndef VENDORS_HALO2_INCLUDE_MSM_H_
#define VENDORS_HALO2_INCLUDE_MSM_H_

#include "rust/cxx.h"

namespace tachyon {
namespace halo2 {

struct CppG1Affine;
struct CppG1Jacobian;
struct CppFr;

rust::Box<CppG1Jacobian> msm(rust::Slice<const CppG1Affine> bases,
                             rust::Slice<const CppFr> scalars);

void init_msm_gpu(uint8_t degree);

void release_msm_gpu();

rust::Box<CppG1Jacobian> msm_gpu(rust::Slice<const CppG1Affine> bases,
                                 rust::Slice<const CppFr> scalars);

}  // namespace halo2
}  // namespace tachyon

#endif  // VENDORS_HALO2_INCLUDE_MSM_H_
