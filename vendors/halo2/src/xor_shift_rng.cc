#include "vendors/halo2/include/xor_shift_rng.h"

#include <vector>

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/random/xor_shift/xor_shift_rng.h"
#include "tachyon/rs/base/container_util.h"

namespace tachyon::halo2_api {

class XORShiftRng::Impl {
 public:
  explicit Impl(std::array<uint8_t, 16> seed) {
    uint8_t seed_copy[16];
    memcpy(seed_copy, seed.data(), 16);
    rng_ = crypto::XORShiftRNG::FromSeed(seed_copy);
  }
  Impl(const Impl& other) : rng_(other.rng_) {}

  uint32_t NextUint32() { return rng_.NextUint32(); }

  rust::Vec<uint8_t> GetState() const {
    base::Uint8VectorBuffer buffer;
    CHECK(buffer.Grow(16));
    CHECK(buffer.Write32LE(rng_.x()));
    CHECK(buffer.Write32LE(rng_.y()));
    CHECK(buffer.Write32LE(rng_.z()));
    CHECK(buffer.Write32LE(rng_.w()));
    return rs::ConvertCppVecToRustVec(buffer.owned_buffer());
  }

 private:
  crypto::XORShiftRNG rng_;
};

XORShiftRng::XORShiftRng(std::array<uint8_t, 16> seed)
    : impl_(new Impl(seed)) {}

uint32_t XORShiftRng::next_u32() { return impl_->NextUint32(); }

std::unique_ptr<XORShiftRng> XORShiftRng::clone() const {
  std::unique_ptr<XORShiftRng> ret(new XORShiftRng());
  ret->impl_.reset(new Impl(*impl_));
  return ret;
}

rust::Vec<uint8_t> XORShiftRng::state() const { return impl_->GetState(); }

std::unique_ptr<XORShiftRng> new_xor_shift_rng(std::array<uint8_t, 16> seed) {
  return std::make_unique<XORShiftRng>(seed);
}

}  // namespace tachyon::halo2_api
