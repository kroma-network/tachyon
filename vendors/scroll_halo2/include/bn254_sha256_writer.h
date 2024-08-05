#ifndef VENDORS_SCROLL_HALO2_INCLUDE_BN254_SHA256_WRITER_H_
#define VENDORS_SCROLL_HALO2_INCLUDE_BN254_SHA256_WRITER_H_

#include <stdint.h>

#include <array>
#include <memory>

#include "openssl/sha.h"
#include "rust/cxx.h"

#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"

namespace tachyon::halo2_api::bn254 {

class Sha256Writer {
 public:
  Sha256Writer();
  Sha256Writer(const Sha256Writer& other) = delete;
  Sha256Writer& operator=(const Sha256Writer& other) = delete;
  ~Sha256Writer();

  void update(rust::Slice<const uint8_t> data);
  void finalize(std::array<uint8_t, SHA256_DIGEST_LENGTH>& result);
  rust::Vec<uint8_t> state() const;

 private:
  tachyon_halo2_bn254_transcript_writer* writer_;
};

std::unique_ptr<Sha256Writer> new_sha256_writer();

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_SCROLL_HALO2_INCLUDE_BN254_SHA256_WRITER_H_
