#ifndef VENDORS_HALO2_INCLUDE_BN254_SHPLONK_PROVER_H_
#define VENDORS_HALO2_INCLUDE_BN254_SHPLONK_PROVER_H_

#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_prover.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;
struct G1JacobianPoint;
struct InstanceSingle;
struct AdviceSingle;
class ProvingKey;
class Evals;
class RationalEvals;
class Poly;

class SHPlonkProver {
 public:
  SHPlonkProver(uint32_t k, const Fr& s);
  SHPlonkProver(const SHPlonkProver& other) = delete;
  SHPlonkProver& operator=(const SHPlonkProver& other) = delete;
  ~SHPlonkProver();

  const tachyon_halo2_bn254_shplonk_prover* prover() const { return prover_; }

  uint32_t k() const;
  uint64_t n() const;
  rust::Box<G1JacobianPoint> commit(const Poly& poly) const;
  rust::Box<G1JacobianPoint> commit_lagrange(const Evals& evals) const;
  std::unique_ptr<Evals> empty_evals() const;
  std::unique_ptr<RationalEvals> empty_rational_evals() const;
  std::unique_ptr<Poly> ifft(const Evals& evals) const;
  void batch_evaluate(
      rust::Slice<const std::unique_ptr<RationalEvals>> rational_evals,
      rust::Slice<std::unique_ptr<Evals>> evals) const;
  void set_rng(rust::Slice<const uint8_t> state);
  void set_transcript(rust::Slice<const uint8_t> state);
  void set_extended_domain(const ProvingKey& pk);
  void create_proof(const ProvingKey& key,
                    rust::Slice<InstanceSingle> instance_singles,
                    rust::Slice<AdviceSingle> advice_singles,
                    rust::Slice<const Fr> challenges);
  rust::Vec<uint8_t> get_proof() const;

 private:
  tachyon_halo2_bn254_shplonk_prover* prover_;
};

std::unique_ptr<SHPlonkProver> new_shplonk_prover(uint32_t k, const Fr& s);

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_SHPLONK_PROVER_H_
