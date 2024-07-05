#include "vendors/plonky3/include/baby_bear_poseidon2_duplex_challenger.h"

namespace tachyon::plonky3_api::baby_bear_poseidon2 {

DuplexChallenger::DuplexChallenger()
    : challenger_(
          tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_create()) {}

DuplexChallenger::~DuplexChallenger() {
  tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_destroy(challenger_);
}

void DuplexChallenger::observe(const BabyBear& value) {
  tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_observe(
      challenger_, reinterpret_cast<const tachyon_baby_bear*>(&value));
}

rust::Box<BabyBear> DuplexChallenger::sample() {
  tachyon_baby_bear* ret = new tachyon_baby_bear;
  *ret =
      tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_sample(challenger_);
  return rust::Box<BabyBear>::from_raw(reinterpret_cast<BabyBear*>(ret));
}

std::unique_ptr<DuplexChallenger> DuplexChallenger::clone() const {
  return std::make_unique<DuplexChallenger>(
      tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_clone(challenger_));
}

std::unique_ptr<DuplexChallenger> new_duplex_challenger() {
  return std::make_unique<DuplexChallenger>();
}

}  // namespace tachyon::plonky3_api::baby_bear_poseidon2
