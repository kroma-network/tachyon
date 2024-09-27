#include "vendors/sp1/include/baby_bear_poseidon2_duplex_challenger.h"

#include "tachyon/rs/base/container_util.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

DuplexChallenger::DuplexChallenger()
    : challenger_(tachyon_sp1_baby_bear_poseidon2_duplex_challenger_create()) {}

DuplexChallenger::~DuplexChallenger() {
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger_destroy(challenger_);
}

void DuplexChallenger::observe(const TachyonBabyBear& value) {
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger_observe(
      challenger_, reinterpret_cast<const tachyon_baby_bear*>(&value));
}

rust::Box<TachyonBabyBear> DuplexChallenger::sample() {
  tachyon_baby_bear* ret = new tachyon_baby_bear;
  *ret = tachyon_sp1_baby_bear_poseidon2_duplex_challenger_sample(challenger_);
  return rust::Box<TachyonBabyBear>::from_raw(
      reinterpret_cast<TachyonBabyBear*>(ret));
}

std::unique_ptr<DuplexChallenger> DuplexChallenger::clone() const {
  return std::make_unique<DuplexChallenger>(
      tachyon_sp1_baby_bear_poseidon2_duplex_challenger_clone(challenger_));
}

rust::Vec<uint8_t> DuplexChallenger::write_hint() const {
  size_t size;
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger_write_hint(challenger_,
                                                               nullptr, &size);
  rust::Vec<uint8_t> ret = rs::CreateDefaultVector<uint8_t>(size);
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger_write_hint(
      challenger_, ret.data(), &size);
  return ret;
}

std::unique_ptr<DuplexChallenger> new_duplex_challenger() {
  return std::make_unique<DuplexChallenger>();
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
