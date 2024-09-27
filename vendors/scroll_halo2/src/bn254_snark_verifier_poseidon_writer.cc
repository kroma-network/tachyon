#include "vendors/scroll_halo2/include/bn254_snark_verifier_poseidon_writer.h"

#include "tachyon/rs/base/container_util.h"

namespace tachyon::halo2_api::bn254 {

SnarkVerifierPoseidonWriter::SnarkVerifierPoseidonWriter()
    : writer_(tachyon_halo2_bn254_transcript_writer_create(
          TACHYON_HALO2_SNARK_VERIFIER_POSEIDON_TRANSCRIPT)) {}

SnarkVerifierPoseidonWriter::~SnarkVerifierPoseidonWriter() {
  tachyon_halo2_bn254_transcript_writer_destroy(writer_);
}

void SnarkVerifierPoseidonWriter::update(rust::Slice<const uint8_t> data) {
  tachyon_halo2_bn254_transcript_writer_update(writer_, data.data(),
                                               data.size());
}

rust::Box<Fr> SnarkVerifierPoseidonWriter::squeeze() {
  tachyon_bn254_fr* ret = new tachyon_bn254_fr;
  *ret = tachyon_halo2_bn254_transcript_writer_squeeze(writer_);
  return rust::Box<Fr>::from_raw(reinterpret_cast<Fr*>(ret));
}

rust::Vec<uint8_t> SnarkVerifierPoseidonWriter::state() const {
  size_t state_size;
  tachyon_halo2_bn254_transcript_writer_get_state(writer_, nullptr,
                                                  &state_size);
  rust::Vec<uint8_t> ret = rs::CreateDefaultVector<uint8_t>(state_size);
  tachyon_halo2_bn254_transcript_writer_get_state(writer_, ret.data(),
                                                  &state_size);
  return ret;
}

std::unique_ptr<SnarkVerifierPoseidonWriter>
new_snark_verifier_poseidon_writer() {
  return std::make_unique<SnarkVerifierPoseidonWriter>();
}

}  // namespace tachyon::halo2_api::bn254
