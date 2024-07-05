#ifndef VENDORS_PLONKY3_INCLUDE_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_H_
#define VENDORS_PLONKY3_INCLUDE_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_H_

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/plonky3/challenger/baby_bear_poseidon2_duplex_challenger.h"

namespace tachyon::plonky3_api::baby_bear_poseidon2 {

struct BabyBear;

class DuplexChallenger {
 public:
  DuplexChallenger();
  explicit DuplexChallenger(
      tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger)
      : challenger_(challenger) {}
  DuplexChallenger(const DuplexChallenger& other) = delete;
  DuplexChallenger& operator=(const DuplexChallenger& other) = delete;
  ~DuplexChallenger();

  void observe(const BabyBear& value);
  rust::Box<BabyBear> sample();
  std::unique_ptr<DuplexChallenger> clone() const;

 private:
  tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger_;
};

std::unique_ptr<DuplexChallenger> new_duplex_challenger();

}  // namespace tachyon::plonky3_api::baby_bear_poseidon2

#endif  // VENDORS_PLONKY3_INCLUDE_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_H_
