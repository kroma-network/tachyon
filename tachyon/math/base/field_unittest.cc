#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

TEST(FieldTest, BatchInverse) {
#if defined(TACHYON_HAS_OPENMP)
  size_t size = size_t{1} << (static_cast<size_t>(omp_get_max_threads()) /
                              GF7::kParallelBatchInverseDivisorThreshold);
#else
  size_t size = 5;
#endif

  std::vector<GF7> fields =
      base::CreateVector(size, []() { return GF7::Random(); });
  std::vector<GF7> inverses;
  inverses.resize(fields.size());
  ASSERT_TRUE(GF7::BatchInverse(fields, &inverses));
  for (size_t i = 0; i < fields.size(); ++i) {
    if (fields[i].IsZero()) {
      EXPECT_TRUE(inverses[i].IsZero());
    } else {
      EXPECT_TRUE((inverses[i] * fields[i]).IsOne());
    }
  }

  ASSERT_TRUE(GF7::BatchInverseInPlace(fields));
  EXPECT_EQ(fields, inverses);
}

}  // namespace tachyon::math
