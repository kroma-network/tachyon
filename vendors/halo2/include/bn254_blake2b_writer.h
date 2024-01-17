#ifndef VENDORS_HALO2_INCLUDE_BN254_BLAKE2B_WRITER_H_
#define VENDORS_HALO2_INCLUDE_BN254_BLAKE2B_WRITER_H_

#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

namespace tachyon::halo2_api::bn254 {

class Blake2bWriter {
 public:
  Blake2bWriter();

  void update(rust::Slice<const uint8_t> data);
  void finalize(std::array<uint8_t, 64>& result);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

std::unique_ptr<Blake2bWriter> new_blake2b_writer();

}  // namespace tachyon::halo2_api::bn254

#endif  // VENDORS_HALO2_INCLUDE_BN254_BLAKE2B_WRITER_H_
