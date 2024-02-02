#ifndef TACHYON_C_ZK_PLONK_HALO2_SHPLONK_PROVER_IMPL_BASE_H_
#define TACHYON_C_ZK_PLONK_HALO2_SHPLONK_PROVER_IMPL_BASE_H_

#include <algorithm>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/zk/plonk/halo2/prover_impl_base.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"

namespace tachyon::c::zk::plonk::halo2 {

// NOTE(chokobole): We set |kMaxDegree| and |kMaxExtendedDegree| to |SIZE_MAX|
// on purpose to avoid creating variant apis corresponding to the set of each
// degree.
constexpr size_t kMaxDegree = SIZE_MAX;
constexpr size_t kMaxExtendedDegree = SIZE_MAX;

using PCS = tachyon::zk::SHPlonkExtension<math::bn254::BN254Curve, kMaxDegree,
                                          kMaxExtendedDegree,
                                          math::bn254::G1AffinePoint>;

class SHPlonkProverImplBase : public ProverImplBase<PCS> {
 public:
  using ProverImplBase<PCS>::ProverImplBase;

  tachyon_bn254_g1_jacobian* Commit(
      const std::vector<math::bn254::Fr>& scalars) const {
    return DoMSM(pcs_.GetG1PowersOfTau(), scalars);
  }

  tachyon_bn254_g1_jacobian* CommitLagrange(
      const std::vector<math::bn254::Fr>& scalars) const {
    return DoMSM(pcs_.GetG1PowersOfTauLagrange(), scalars);
  }

 private:
  static tachyon_bn254_g1_jacobian* DoMSM(
      const std::vector<math::bn254::G1AffinePoint>& bases,
      const std::vector<math::bn254::Fr>& scalars) {
    math::VariableBaseMSM<math::bn254::G1AffinePoint> msm;
    math::VariableBaseMSM<math::bn254::G1AffinePoint>::Bucket bucket;
    absl::Span<const math::bn254::G1AffinePoint> bases_span =
        absl::Span<const math::bn254::G1AffinePoint>(
            bases.data(), std::min(bases.size(), scalars.size()));
    CHECK(msm.Run(bases_span, scalars, &bucket));
    math::bn254::G1JacobianPoint* ret =
        new math::bn254::G1JacobianPoint(bucket.ToJacobian());
    return reinterpret_cast<tachyon_bn254_g1_jacobian*>(ret);
  }
};

}  // namespace tachyon::c::zk::plonk::halo2

#endif  // TACHYON_C_ZK_PLONK_HALO2_SHPLONK_PROVER_IMPL_BASE_H_
