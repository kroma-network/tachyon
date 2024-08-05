#ifndef VENDORS_SCROLL_HALO2_INCLUDE_BN254_BLAKE2B_WRITER_H_
#define VENDORS_SCROLL_HALO2_INCLUDE_BN254_BLAKE2B_WRITER_H_

#include <stdint.h>

#include <array>
#include <memory>

#include "openssl/blake2.h"
#include "rust/cxx.h"

#include "tachyon/c/zk/plonk/halo2/bn254_transcript.h"

namespace tachyon::halo2_api::bn254 {

class Blake2bWriter {
 public:
  Blake2bWriter();
  Blake2bWriter(const Blake2bWriter& other) = delete;
  Blake2bWriter& operator=(const Blake2bWriter& other) = delete;
  ~Blake2bWriter();

  void update(rust::Slice<const uint8_t> data);
  void finalize(std::array<uint8_t, BLAKE2B512_DIGEST_LENGTH>& result);
  rust::Vec<uint8_t> state() const;

 private:
  tachyon_halo2_bn254_transcript_writer* writer_;
};

std::unique_ptr<Blake2bWriter> new_blake2b_writer();

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_SCROLL_HALO2_INCLUDE_BN254_BLAKE2B_WRITER_H_
