#include "tachyon/crypto/sumcheck/sumcheck_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/multivariate/multilinear_dense_evaluations.h"

namespace tachyon::crypto {

namespace {

const size_t kMaxDegree = 4;

using Poly = math::MultilinearDenseEvaluations<math::GF7, kMaxDegree>;

class ProvingKeyTest : public math::FiniteFieldTest<math::GF7> {};

}  // namespace

TEST_F(ProvingKeyTest, Copyable) {
  math::LinearCombination<Poly> random_linear_combination =
      math::LinearCombination<Poly>::Random(kMaxDegree,
                                            base::Range<size_t>(6, 12), 7);
  ProvingKey<Poly> expected =
      ProvingKey<Poly>::Build(random_linear_combination);

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  ProvingKey<Poly> value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::crypto
