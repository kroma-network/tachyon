#ifndef VENDORS_HALO2_SRC_BN254_SHPLONK_PROVER_IMPL_H_
#define VENDORS_HALO2_SRC_BN254_SHPLONK_PROVER_IMPL_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <utility>

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"
#include "vendors/halo2/src/degrees.h"
#include "vendors/halo2/src/prover_impl.h"

namespace tachyon::halo2_api::bn254 {

using PCS =
    zk::SHPlonkExtension<math::bn254::BN254Curve, kMaxDegree,
                         kMaxExtendedDegree, math::bn254::G1AffinePoint>;

class SHPlonkProverImpl : public ProverImpl<PCS> {
 public:
  SHPlonkProverImpl(uint32_t k, const Fr& s)
      : ProverImpl<PCS>([k, &s]() {
          math::bn254::BN254Curve::Init();

          PCS pcs;
          size_t n = size_t{1} << k;
          math::bn254::Fr::BigIntTy bigint;
          memcpy(bigint.limbs, reinterpret_cast<const uint8_t*>(&s),
                 sizeof(uint64_t) * math::bn254::Fr::kLimbNums);
          CHECK(pcs.UnsafeSetup(n, math::bn254::Fr::FromMontgomery(bigint)));
          base::Uint8VectorBuffer write_buf;
          std::unique_ptr<crypto::TranscriptWriter<math::bn254::G1AffinePoint>>
              writer = std::make_unique<
                  zk::halo2::Blake2bWriter<math::bn254::G1AffinePoint>>(
                  std::move(write_buf));
          zk::halo2::Prover<PCS> prover = zk::halo2::Prover<PCS>::CreateFromRNG(
              std::move(pcs), std::move(writer), /*rng=*/nullptr,
              /*blinding_factors=*/0);
          prover.set_domain(PCS::Domain::Create(n));
          return prover;
        }) {}
};

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_SRC_BN254_SHPLONK_PROVER_IMPL_H_
