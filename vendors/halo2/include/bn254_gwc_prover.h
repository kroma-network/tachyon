#ifndef VENDORS_HALO2_INCLUDE_BN254_GWC_PROVER_H_
#define VENDORS_HALO2_INCLUDE_BN254_GWC_PROVER_H_

#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/plonk/halo2/bn254_gwc_prover.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;
struct G1JacobianPoint;
struct G2AffinePoint;
struct InstanceSingle;
struct AdviceSingle;
class ProvingKey;
class Evals;
class RationalEvals;
class Poly;

class GWCProver {
 public:
  GWCProver(uint8_t transcript_type, uint32_t k, const Fr& s);
  GWCProver(uint8_t transcript_type, uint32_t k, const uint8_t* params,
            size_t params_len);
  GWCProver(const GWCProver& other) = delete;
  GWCProver& operator=(const GWCProver& other) = delete;
  ~GWCProver();

  const tachyon_halo2_bn254_gwc_prover* prover() const { return prover_; }

  uint32_t k() const;
  uint64_t n() const;
  // TODO(dongchangYoo): avoid copying through the use of |rust::Box|.
  rust::Box<G2AffinePoint> s_g2() const;
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
  void create_proof(ProvingKey& key,
                    rust::Slice<InstanceSingle> instance_singles,
                    rust::Slice<AdviceSingle> advice_singles,
                    rust::Slice<const Fr> challenges);
  rust::Vec<uint8_t> get_proof() const;

 private:
  tachyon_halo2_bn254_gwc_prover* prover_;
};

std::unique_ptr<GWCProver> new_gwc_prover(uint8_t transcript_type, uint32_t k,
                                          const Fr& s);

std::unique_ptr<GWCProver> new_gwc_prover_from_params(
    uint8_t transcript_type, uint32_t k, rust::Slice<const uint8_t> params);

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_GWC_PROVER_H_
