#include "tachyon/zk/plonk/halo2/argument_data.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

constexpr size_t kMaxDegree = 7;

class ArgumentDataTest : public math::FiniteFieldTest<math::GF7> {
 public:
  using Poly = math::UnivariateDensePolynomial<math::GF7, kMaxDegree>;
  using Evals = math::UnivariateEvaluations<math::GF7, kMaxDegree>;
};

}  // namespace

TEST_F(ArgumentDataTest, Copyable) {
  constexpr size_t kNumCircuits = 2;
  constexpr size_t kNumAdvices = 3;
  constexpr size_t kNumBlinds = 4;
  constexpr size_t kNumChallenges = 2;
  constexpr size_t kNumInstances = 2;

  std::vector<std::vector<Evals>> advice_columns_vec =
      base::CreateVector(kNumCircuits, []() {
        return base::CreateVector(kNumAdvices,
                                  []() { return Evals::Random(kMaxDegree); });
      });
  std::vector<std::vector<math::GF7>> advice_blinds_vec =
      base::CreateVector(kNumCircuits, []() {
        return base::CreateVector(kNumBlinds,
                                  []() { return math::GF7::Random(); });
      });
  std::vector<math::GF7> challenges =
      base::CreateVector(kNumChallenges, []() { return math::GF7::Random(); });
  std::vector<std::vector<Evals>> instance_columns_vec =
      base::CreateVector(kNumCircuits, []() {
        return base::CreateVector(kNumInstances,
                                  []() { return Evals::Random(kMaxDegree); });
      });
  std::vector<std::vector<Poly>> instance_polys_vec =
      base::CreateVector(kNumCircuits, []() {
        return base::CreateVector(kNumInstances,
                                  []() { return Poly::Random(kMaxDegree); });
      });

  ArgumentData<Poly, Evals> expected(
      std::move(advice_columns_vec), std::move(advice_blinds_vec),
      std::move(challenges), std::move(instance_columns_vec),
      std::move(instance_polys_vec));

  std::vector<uint8_t> vec;
  vec.resize(base::EstimateSize(expected));
  base::Buffer write_buf(vec.data(), vec.size());
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  ArgumentData<Poly, Evals> value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk::plonk::halo2
