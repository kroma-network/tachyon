#ifndef VENDORS_HALO2_INCLUDE_BN254_PROVING_KEY_H_
#define VENDORS_HALO2_INCLUDE_BN254_PROVING_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;
class SHPlonkProver;
class ProvingKeyImpl;

class ProvingKey {
 public:
  explicit ProvingKey(rust::Slice<const uint8_t> pk_bytes);

  const ProvingKeyImpl* impl() const { return impl_.get(); }

  rust::Vec<uint8_t> advice_column_phases() const;
  uint32_t blinding_factors() const;
  rust::Vec<uint8_t> challenge_phases() const;
  rust::Vec<size_t> constants() const;
  size_t num_advice_columns() const;
  size_t num_challenges() const;
  size_t num_instance_columns() const;
  rust::Vec<uint8_t> phases() const;
  rust::Box<Fr> transcript_repr(const SHPlonkProver& prover);

 private:
  std::shared_ptr<ProvingKeyImpl> impl_;
};

std::unique_ptr<ProvingKey> new_proving_key(
    rust::Slice<const uint8_t> pk_bytes);

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_PROVING_KEY_H_
