#ifndef TACHYON_C_ZK_PLONK_HALO2_KZG_FAMILY_PROVER_IMPL_H_
#define TACHYON_C_ZK_PLONK_HALO2_KZG_FAMILY_PROVER_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/point_traits_forward.h"
#include "tachyon/c/zk/plonk/halo2/prover_impl_base.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon::c::zk::plonk::halo2 {

template <typename PCS>
class KZGFamilyProverImpl : public ProverImplBase<PCS> {
 public:
  using Callback = typename ProverImplBase<PCS>::Callback;
  using AffinePoint = typename PCS::Commitment;
  using JacobianPoint =
      tachyon::math::JacobianPoint<typename AffinePoint::Curve>;
  using ScalarField = typename AffinePoint::ScalarField;
  using CAffinePoint = typename math::PointTraits<AffinePoint>::CCurvePoint;
  using CJacobianPoint = typename math::PointTraits<JacobianPoint>::CCurvePoint;
  using CScalarField = typename math::PointTraits<AffinePoint>::CScalarField;

  using ProverImplBase<PCS>::ProverImplBase;

  CJacobianPoint* Commit(const std::vector<ScalarField>& scalars) const {
    return DoMSM(this->pcs_.GetG1PowersOfTau(), scalars);
  }

  CJacobianPoint* CommitLagrange(
      const std::vector<ScalarField>& scalars) const {
    return DoMSM(this->pcs_.GetG1PowersOfTauLagrange(), scalars);
  }

 private:
  static CJacobianPoint* DoMSM(const std::vector<AffinePoint>& bases,
                               const std::vector<ScalarField>& scalars) {
    using MSM = tachyon::math::VariableBaseMSM<AffinePoint>;
    MSM msm;
    typename MSM::Bucket bucket;
    absl::Span<const AffinePoint> bases_span(
        bases.data(), std::min(bases.size(), scalars.size()));
    CHECK(msm.Run(bases_span, scalars, &bucket));
    JacobianPoint* ret = new JacobianPoint(bucket.ToJacobian());
    return reinterpret_cast<CJacobianPoint*>(ret);
  }
};

}  // namespace tachyon::c::zk::plonk::halo2

#endif  // TACHYON_C_ZK_PLONK_HALO2_KZG_FAMILY_PROVER_IMPL_H_
