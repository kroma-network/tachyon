#ifndef VENDORS_HALO2_INCLUDE_MSM_H_
#define VENDORS_HALO2_INCLUDE_MSM_H_

#include "rust/cxx.h"

namespace tachyon {
namespace halo2 {

struct CppMSM;
struct CppMSMGpu;

struct CppG1Affine;
struct CppG1Jacobian;
struct CppFr;

rust::Box<CppMSM> create_msm(uint8_t degree);

void destroy_msm(rust::Box<CppMSM> msm);

rust::Box<CppG1Jacobian> msm(CppMSM* msm, rust::Slice<const CppG1Affine> bases,
                             rust::Slice<const CppFr> scalars);

rust::Box<CppMSMGpu> create_msm_gpu(uint8_t degree, int algorithm);

void destroy_msm_gpu(rust::Box<CppMSMGpu> msm);

rust::Box<CppG1Jacobian> msm_gpu(CppMSMGpu* msm,
                                 rust::Slice<const CppG1Affine> bases,
                                 rust::Slice<const CppFr> scalars);

}  // namespace halo2
}  // namespace tachyon

#endif  // VENDORS_HALO2_INCLUDE_MSM_H_
