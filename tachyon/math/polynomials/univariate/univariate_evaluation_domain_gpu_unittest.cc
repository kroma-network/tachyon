#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/mixed_radix_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"

namespace tachyon::math {

namespace {

template <typename Domain>
class UnivariateEvaluationDomainGpuTest
    : public FiniteFieldTest<typename Domain::Field> {};

}  // namespace

using UnivariateEvaluationDomainTypes =
    testing::Types<Radix2EvaluationDomain<bn254::Fr>,
                   MixedRadixEvaluationDomain<bn254::Fr>>;
TYPED_TEST_SUITE(UnivariateEvaluationDomainGpuTest,
                 UnivariateEvaluationDomainTypes);

template <typename Domain>
void Test(Domain* domain, IcicleNTTHolder<typename Domain::Field>* holder) {
  using DensePoly = typename Domain::DensePoly;
  using Evals = typename Domain::Evals;

  {
    auto poly = DensePoly::Random(domain->size() - 1);
    poly.at(0) = *Domain::Field::FromDecString(
        "1019095245415693755667175276439778880498898940628134960481081498899812"
        "1232998");
    domain->set_icicle(nullptr);
    Evals expected_evals = domain->FFT(poly);
    domain->set_icicle(holder);
    Evals evals = domain->FFT(std::move(poly));
    EXPECT_EQ(evals, expected_evals);
  }
}

TYPED_TEST(UnivariateEvaluationDomainGpuTest, FFTCorrectness) {
  using Domain = TypeParam;
  using F = typename Domain::Field;

  for (size_t i = 0; i < 1; ++i) {
    SCOPED_TRACE(absl::Substitute("test: $0", static_cast<int>(i)));
    IcicleNTTHolder<F> holder = IcicleNTTHolder<F>::Create();
    size_t size = size_t{1} << i;
    auto domain = Domain::Create(size);
    CHECK(holder->Init(domain->group_gen()));
    Test(domain.get(), &holder);

    auto coset_domain =
        domain->GetCoset(F::FromMontgomery(F::Config::kSubgroupGenerator));
    Test(coset_domain.get(), &holder);
  }
}

}  // namespace tachyon::math
