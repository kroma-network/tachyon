#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENING_POINTS_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENING_POINTS_H_

#include <stddef.h>

#include <memory>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear4;

class OpeningPoints {
 public:
  explicit OpeningPoints(
      tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points)
      : opening_points_(opening_points) {}
  OpeningPoints(const OpeningPoints& other) = delete;
  OpeningPoints& operator=(const OpeningPoints& other) = delete;
  ~OpeningPoints();

  const tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points() const {
    return opening_points_;
  }

  std::unique_ptr<OpeningPoints> clone() const;
  void allocate(size_t round, size_t rows, size_t cols);
  void set(size_t round, size_t row, size_t col, const TachyonBabyBear4& point);

 private:
  tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points_ = nullptr;
};

std::unique_ptr<OpeningPoints> new_opening_points(size_t rounds);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_OPENING_POINTS_H_
