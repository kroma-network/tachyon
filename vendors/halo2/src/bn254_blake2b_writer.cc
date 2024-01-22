#include "vendors/halo2/include/bn254_blake2b_writer.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "vendors/halo2/src/blake2b_writer_impl.h"

namespace tachyon::halo2_api::bn254 {

class Blake2bWriter::Impl
    : public Blake2bWriterImpl<math::bn254::G1AffinePoint> {};

Blake2bWriter::Blake2bWriter() : impl_(new Impl()) {}

void Blake2bWriter::update(rust::Slice<const uint8_t> data) {
  return impl_->Update(data);
}

void Blake2bWriter::finalize(std::array<uint8_t, 64>& result) {
  return impl_->Finalize(result);
}

std::unique_ptr<Blake2bWriter> new_blake2b_writer() {
  return std::make_unique<Blake2bWriter>();
}

}  // namespace tachyon::halo2_api::bn254
