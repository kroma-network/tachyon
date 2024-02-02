#ifndef TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_IMPL_H_
#define TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/c/zk/plonk/halo2/prover_impl_base.h"

namespace tachyon::c::zk::plonk::halo2::bn254 {

class SHPlonkProverImpl : public ProverImplBase<PCS> {
 public:
  using Callback = ProverImplBase<PCS>::Callback;

  SHPlonkProverImpl(Callback callback, uint8_t transcript_type)
      : ProverImplBase<PCS>(std::move(callback)),
        transcript_type_(transcript_type) {}

  tachyon_bn254_g1_jacobian* Commit(
      const std::vector<tachyon::math::bn254::Fr>& scalars) const {
    return DoMSM(pcs_.GetG1PowersOfTau(), scalars);
  }

  tachyon_bn254_g1_jacobian* CommitLagrange(
      const std::vector<tachyon::math::bn254::Fr>& scalars) const {
    return DoMSM(pcs_.GetG1PowersOfTauLagrange(), scalars);
  }

  uint8_t transcript_type() const { return transcript_type_; }

 private:
  static tachyon_bn254_g1_jacobian* DoMSM(
      const std::vector<tachyon::math::bn254::G1AffinePoint>& bases,
      const std::vector<tachyon::math::bn254::Fr>& scalars) {
    using MSM =
        tachyon::math::VariableBaseMSM<tachyon::math::bn254::G1AffinePoint>;
    MSM msm;
    MSM::Bucket bucket;
    absl::Span<const tachyon::math::bn254::G1AffinePoint> bases_span =
        absl::Span<const tachyon::math::bn254::G1AffinePoint>(
            bases.data(), std::min(bases.size(), scalars.size()));
    CHECK(msm.Run(bases_span, scalars, &bucket));
    tachyon::math::bn254::G1JacobianPoint* ret =
        new tachyon::math::bn254::G1JacobianPoint(bucket.ToJacobian());
    return reinterpret_cast<tachyon_bn254_g1_jacobian*>(ret);
  }

  uint8_t transcript_type_;
};

}  // namespace tachyon::c::zk::plonk::halo2::bn254

#endif  // TACHYON_C_ZK_PLONK_HALO2_BN254_SHPLONK_PROVER_IMPL_H_
