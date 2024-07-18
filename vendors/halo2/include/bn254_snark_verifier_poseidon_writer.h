#ifndef VENDORS_HALO2_INCLUDE_BN254_SNARK_VERIFIER_POSEIDON_WRITER_H_
#define VENDORS_HALO2_INCLUDE_BN254_SNARK_VERIFIER_POSEIDON_WRITER_H_

#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"

namespace tachyon::halo2_api::bn254 {

struct Fr;

class SnarkVerifierPoseidonWriter {
 public:
  SnarkVerifierPoseidonWriter();
  SnarkVerifierPoseidonWriter(const SnarkVerifierPoseidonWriter& other) =
      delete;
  SnarkVerifierPoseidonWriter& operator=(
      const SnarkVerifierPoseidonWriter& other) = delete;
  ~SnarkVerifierPoseidonWriter();

  void update(rust::Slice<const uint8_t> data);
  rust::Box<Fr> squeeze();
  rust::Vec<uint8_t> state() const;

 private:
  tachyon_halo2_bn254_transcript_writer* writer_;
};

std::unique_ptr<SnarkVerifierPoseidonWriter>
new_snark_verifier_poseidon_writer();

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_SNARK_VERIFIER_POSEIDON_WRITER_H_
