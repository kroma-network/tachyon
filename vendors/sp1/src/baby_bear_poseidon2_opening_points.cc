#include "vendors/sp1/include/baby_bear_poseidon2_opening_points.h"

#include "vendors/sp1/src/baby_bear_poseidon2.rs.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

OpeningPoints::~OpeningPoints() {
  tachyon_sp1_baby_bear_poseidon2_opening_points_destroy(opening_points_);
}

std::unique_ptr<OpeningPoints> OpeningPoints::clone() const {
  return std::make_unique<OpeningPoints>(
      tachyon_sp1_baby_bear_poseidon2_opening_points_clone(opening_points_));
}

void OpeningPoints::allocate(size_t round, size_t rows, size_t cols) {
  tachyon_sp1_baby_bear_poseidon2_opening_points_allocate(opening_points_,
                                                          round, rows, cols);
}

void OpeningPoints::set(size_t round, size_t row, size_t col,
                        const TachyonBabyBear4& point) {
  tachyon_sp1_baby_bear_poseidon2_opening_points_set(
      opening_points_, round, row, col,
      reinterpret_cast<const tachyon_baby_bear4*>(&point));
}

std::unique_ptr<OpeningPoints> new_opening_points(size_t rounds) {
  return std::make_unique<OpeningPoints>(
      tachyon_sp1_baby_bear_poseidon2_opening_points_create(rounds));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2
