#ifndef TACHYON_MATH_BASE_PARALLELIZE_THRESHOLD_H_
#define TACHYON_MATH_BASE_PARALLELIZE_THRESHOLD_H_

namespace tachyon::math {

struct ParallelizeThreshold {
  // The threshold for parallelizing a loop. If the size of the loop is less
  // than this threshold, the loop will be executed sequentially.
  static constexpr int kFieldInit = 1e6;
  static constexpr int kFieldSimpleOp = 1e5;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_PARALLELIZE_THRESHOLD_H_
