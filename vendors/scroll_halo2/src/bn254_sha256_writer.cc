#include "vendors/scroll_halo2/include/bn254_sha256_writer.h"

#include <string.h>

#include "tachyon/base/logging.h"
#include "tachyon/rs/base/container_util.h"

namespace tachyon::halo2_api::bn254 {

Sha256Writer::Sha256Writer()
    : writer_(tachyon_halo2_bn254_transcript_writer_create(
          TACHYON_HALO2_SHA256_TRANSCRIPT)) {}

Sha256Writer::~Sha256Writer() {
  tachyon_halo2_bn254_transcript_writer_destroy(writer_);
}

void Sha256Writer::update(rust::Slice<const uint8_t> data) {
  tachyon_halo2_bn254_transcript_writer_update(writer_, data.data(),
                                               data.size());
}

void Sha256Writer::finalize(std::array<uint8_t, SHA256_DIGEST_LENGTH>& result) {
  uint8_t data[SHA256_DIGEST_LENGTH];
  size_t data_size;
  tachyon_halo2_bn254_transcript_writer_finalize(writer_, data, &data_size);
  CHECK_EQ(data_size, size_t{SHA256_DIGEST_LENGTH});
  memcpy(result.data(), data, data_size);
}

rust::Vec<uint8_t> Sha256Writer::state() const {
  constexpr size_t kStateSize = sizeof(sha256_state_st);
  rust::Vec<uint8_t> ret = rs::CreateEmptyVector<uint8_t>(kStateSize);
  size_t state_size;
  tachyon_halo2_bn254_transcript_writer_get_state(writer_, ret.data(),
                                                  &state_size);
  CHECK_EQ(state_size, kStateSize);
  return ret;
}

std::unique_ptr<Sha256Writer> new_sha256_writer() {
  return std::make_unique<Sha256Writer>();
}

}  // namespace tachyon::halo2_api::bn254
