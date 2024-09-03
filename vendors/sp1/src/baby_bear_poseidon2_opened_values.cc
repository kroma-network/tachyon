#include "vendors/sp1/include/baby_bear_poseidon2_opened_values.h"

#include "vendors/sp1/src/baby_bear_poseidon2.rs.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

OpenedValues::~OpenedValues() {
  tachyon_sp1_baby_bear_poseidon2_opened_values_destroy(opened_values_);
}

void OpenedValues::allocate_outer(size_t round, size_t rows, size_t cols) {
  tachyon_sp1_baby_bear_poseidon2_opened_values_allocate_outer(
      opened_values_, round, rows, cols);
}

void OpenedValues::allocate_inner(size_t round, size_t row, size_t cols,
                                  size_t size) {
  tachyon_sp1_baby_bear_poseidon2_opened_values_allocate_inner(
      opened_values_, round, row, cols, size);
}

void OpenedValues::set(size_t round, size_t row, size_t col, size_t idx,
                       const TachyonBabyBear4& value) {
  tachyon_sp1_baby_bear_poseidon2_opened_values_set(
      opened_values_, round, row, col, idx,
      reinterpret_cast<const tachyon_baby_bear4*>(&value));
}

std::unique_ptr<OpenedValues> new_opened_values(size_t rounds) {
  return std::make_unique<OpenedValues>(
      tachyon_sp1_baby_bear_poseidon2_opened_values_create(rounds));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
