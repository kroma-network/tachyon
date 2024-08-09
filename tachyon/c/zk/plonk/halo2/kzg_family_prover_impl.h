#ifndef TACHYON_C_ZK_PLONK_HALO2_KZG_FAMILY_PROVER_IMPL_H_
#define TACHYON_C_ZK_PLONK_HALO2_KZG_FAMILY_PROVER_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/c/zk/plonk/halo2/prover_impl_base.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon::c::zk::plonk::halo2 {

template <typename PS>
class KZGFamilyProverImpl : public ProverImplBase<PS> {
 public:
  using Callback = typename ProverImplBase<PS>::Callback;
  using PCS = typename PS::PCS;
  using AffinePoint = typename PCS::Commitment;
  using ProjectivePoint =
      tachyon::math::ProjectivePoint<typename AffinePoint::Curve>;
  using ScalarField = typename AffinePoint::ScalarField;
  using CAffinePoint = typename math::PointTraits<AffinePoint>::CCurvePoint;
  using CProjectivePoint =
      typename math::PointTraits<ProjectivePoint>::CCurvePoint;
  using CScalarField = typename math::PointTraits<AffinePoint>::CScalarField;

  using ProverImplBase<PS>::ProverImplBase;

  CProjectivePoint* CommitRaw(const std::vector<ScalarField>& scalars) const {
#if TACHYON_CUDA
    if (this->pcs_.UsesGPU()) {
      return DoMSM(this->pcs_.GetDeviceG1PowersOfTau(), scalars);
    }
#endif
    return DoMSM(this->pcs_.GetG1PowersOfTau(), scalars);
  }

  CProjectivePoint* CommitLagrangeRaw(
      const std::vector<ScalarField>& scalars) const {
#if TACHYON_CUDA
    if (this->pcs_.UsesGPU()) {
      return DoMSM(this->pcs_.GetDeviceG1PowersOfTauLagrange(), scalars);
    }
#endif
    return DoMSM(this->pcs_.GetG1PowersOfTauLagrange(), scalars);
  }

 private:
  template <typename BaseContainer>
  CProjectivePoint* DoMSM(const BaseContainer& bases,
                          const std::vector<ScalarField>& scalars) const {
    ProjectivePoint* ret = new ProjectivePoint();
    CHECK(this->pcs_.DoMSM(bases, scalars, ret));

    return c::base::c_cast(ret);
  }
};

}  // namespace tachyon::c::zk::plonk::halo2

#endif  // TACHYON_C_ZK_PLONK_HALO2_KZG_FAMILY_PROVER_IMPL_H_
