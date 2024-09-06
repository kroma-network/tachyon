#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENED_VALUES_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENED_VALUES_H_

#include <stddef.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear4;

class OpenedValues {
 public:
  explicit OpenedValues(
      tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values)
      : opened_values_(opened_values) {}
  OpenedValues(const OpenedValues& other) = delete;
  OpenedValues& operator=(const OpenedValues& other) = delete;
  ~OpenedValues();

  const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values() const {
    return opened_values_;
  }

  void allocate_outer(size_t round, size_t rows, size_t cols);
  void allocate_inner(size_t round, size_t row, size_t cols, size_t size);
  void set(size_t round, size_t row, size_t col, size_t idx,
           const TachyonBabyBear4& value);

 private:
  tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values_;
};

std::unique_ptr<OpenedValues> new_opened_values(size_t rounds);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENED_VALUES_H_
