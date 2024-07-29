#include "vendors/scroll_halo2/include/bn254_poseidon_writer.h"

namespace tachyon::halo2_api::bn254 {

PoseidonWriter::PoseidonWriter()
    : writer_(tachyon_halo2_bn254_transcript_writer_create(
          TACHYON_HALO2_POSEIDON_TRANSCRIPT)) {}

PoseidonWriter::~PoseidonWriter() {
  tachyon_halo2_bn254_transcript_writer_destroy(writer_);
}

void PoseidonWriter::update(rust::Slice<const uint8_t> data) {
  tachyon_halo2_bn254_transcript_writer_update(writer_, data.data(),
                                               data.size());
}

rust::Box<Fr> PoseidonWriter::squeeze() {
  tachyon_bn254_fr* ret = new tachyon_bn254_fr;
  *ret = tachyon_halo2_bn254_transcript_writer_squeeze(writer_);
  return rust::Box<Fr>::from_raw(reinterpret_cast<Fr*>(ret));
}

rust::Vec<uint8_t> PoseidonWriter::state() const {
  size_t state_size;
  tachyon_halo2_bn254_transcript_writer_get_state(writer_, nullptr,
                                                  &state_size);
  rust::Vec<uint8_t> ret;
  // NOTE(chokobole): |rust::Vec<uint8_t>| doesn't have |resize()|.
  ret.reserve(state_size);
  for (size_t i = 0; i < state_size; ++i) {
    ret.push_back(0);
  }
  tachyon_halo2_bn254_transcript_writer_get_state(writer_, ret.data(),
                                                  &state_size);
  return ret;
}

std::unique_ptr<PoseidonWriter> new_poseidon_writer() {
  return std::make_unique<PoseidonWriter>();
}

}  // namespace tachyon::halo2_api::bn254
