#ifndef VENDORS_HALO2_INCLUDE_BN254_SHPLONK_PROVER_H_
#define VENDORS_HALO2_INCLUDE_BN254_SHPLONK_PROVER_H_

#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;
struct G1JacobianPoint;
class SHPlonkProverImpl;
class SHPlonkProvingKey;

class SHPlonkProver {
 public:
  SHPlonkProver(uint32_t k, const Fr& s);

  uint32_t k() const;
  uint64_t n() const;
  rust::Box<G1JacobianPoint> commit(rust::Slice<const Fr> scalars) const;
  rust::Box<G1JacobianPoint> commit_lagrange(
      rust::Slice<const Fr> scalars) const;
  void set_rng(rust::Slice<const uint8_t> state);
  void set_transcript(rust::Slice<const uint8_t> state);
  void set_extended_domain(const SHPlonkProvingKey& pk);

 private:
  std::shared_ptr<SHPlonkProverImpl> impl_;
};

std::unique_ptr<SHPlonkProver> new_shplonk_prover(uint32_t k, const Fr& s);

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_SHPLONK_PROVER_H_
