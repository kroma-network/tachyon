#ifndef VENDORS_HALO2_SRC_BLAKE2B_WRITER_IMPL_H_
#define VENDORS_HALO2_SRC_BLAKE2B_WRITER_IMPL_H_

#include <stdint.h>

#include <vector>

#include "rust/cxx.h"

#include "tachyon/zk/plonk/halo2/blake2b_transcript.h"

namespace tachyon::halo2_api {

template <typename AffinePoint>
class Blake2bWriterImpl {
 public:
  using ScalarField = typename AffinePoint::ScalarField;

  Blake2bWriterImpl() : writer_(base::Uint8VectorBuffer()) {}

  void Update(rust::Slice<const uint8_t> data) {
    writer_.Update(data.data(), data.size());
  }

  void Finalize(std::array<uint8_t, 64>& result) {
    uint8_t stack_result[64];
    writer_.Finalize(stack_result);
    memcpy(result.data(), stack_result, sizeof(stack_result));
  }

  std::vector<uint8_t> GetState() const { return writer_.GetState(); }

 private:
  zk::halo2::Blake2bWriter<AffinePoint> writer_;
};

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_SRC_BLAKE2B_WRITER_IMPL_H_
