// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/contains.h"
#include "tachyon/base/functional/function_ref.h"
#include "tachyon/base/optional.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn384_small_two_adicity/fq.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/mixed_radix_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"

namespace tachyon::math {

namespace {

constexpr size_t kNumCoeffs = 12;

template <typename Domain>
class UnivariateEvaluationDomainTest
    : public FiniteFieldTest<typename Domain::Field> {
 public:
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;

 protected:
  void TestDomains(size_t num_coeffs,
                   base::FunctionRef<void(const BaseDomain&)> callback) {
    std::unique_ptr<BaseDomain> domain = Domain::Create(num_coeffs);
    std::unique_ptr<BaseDomain> coset_domain =
        domain->GetCoset(F::FromMontgomery(F::Config::kSubgroupGenerator));
    for (bool use_coset : {true, false}) {
      const std::unique_ptr<BaseDomain>& d = use_coset ? coset_domain : domain;
      callback(*d);
    }
  }
};

}  // namespace

using UnivariateEvaluationDomainTypes =
    testing::Types<Radix2EvaluationDomain<bls12_381::Fr>,
                   MixedRadixEvaluationDomain<bn384_small_two_adicity::Fq>>;
TYPED_TEST_SUITE(UnivariateEvaluationDomainTest,
                 UnivariateEvaluationDomainTypes);

TYPED_TEST(UnivariateEvaluationDomainTest, VanishingPolynomialEvaluation) {
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;
  using SparsePoly = typename Domain::SparsePoly;

  for (size_t coeffs = 0; coeffs < kNumCoeffs; ++coeffs) {
    this->TestDomains(coeffs, [](const BaseDomain& d) {
      SparsePoly z = d.GetVanishingPolynomial();
      for (size_t i = 0; i < 100; ++i) {
        F point = F::Random();
        EXPECT_EQ(z.Evaluate(point), d.EvaluateVanishingPolynomial(point));
      }
    });
  }
}

TYPED_TEST(UnivariateEvaluationDomainTest,
           VanishingPolynomialVanishesOnDomain) {
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;
  using SparsePoly = typename Domain::SparsePoly;

  for (size_t coeffs = 0; coeffs < kNumCoeffs; ++coeffs) {
    this->TestDomains(coeffs, [](const BaseDomain& d) {
      SparsePoly z = d.GetVanishingPolynomial();
      for (const F& element : d.GetElements()) {
        EXPECT_TRUE(z.Evaluate(element).IsZero());
      }
    });
  }
}

TYPED_TEST(UnivariateEvaluationDomainTest, FilterPolynomial) {
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;
  using DensePoly = typename Domain::DensePoly;

  if constexpr (std::is_same_v<F, bls12_381::Fr>) {
    for (size_t log_domain_size = 1; log_domain_size < 4; ++log_domain_size) {
      size_t domain_size = size_t{1} << log_domain_size;
      std::unique_ptr<Domain> domain = Domain::Create(domain_size);
      for (size_t log_subdomain_size = 1; log_subdomain_size <= log_domain_size;
           ++log_subdomain_size) {
        size_t subdomain_size = size_t{1} << log_subdomain_size;
        std::unique_ptr<Domain> subdomain = Domain::Create(subdomain_size);

        // Obtain all possible offsets of |subdomain| within |domain|.
        std::vector<bls12_381::Fr> possible_offsets = {bls12_381::Fr::One()};
        const bls12_381::Fr& domain_generator = domain->group_gen();

        bls12_381::Fr offset = domain_generator;
        const bls12_381::Fr& subdomain_generator = subdomain->group_gen();
        while (offset != subdomain_generator) {
          possible_offsets.push_back(offset);
          offset *= domain_generator;
        }
        EXPECT_EQ(possible_offsets.size(), domain_size / subdomain_size);

        // Get all possible cosets of |subdomain| within |domain|.
        for (const bls12_381::Fr& offset : possible_offsets) {
          std::unique_ptr<BaseDomain> coset = subdomain->GetCoset(offset);
          std::vector<bls12_381::Fr> coset_elements = coset->GetElements();
          DensePoly filter_poly = domain->GetFilterPolynomial(*coset);
          EXPECT_EQ(filter_poly.Degree(), domain_size - subdomain_size);
          for (const bls12_381::Fr& element : domain->GetElements()) {
            bls12_381::Fr evaluation = unwrap<bls12_381::Fr>(
                domain->EvaluateFilterPolynomial(*coset, element));
            EXPECT_EQ(evaluation, filter_poly.Evaluate(element));
            if (base::Contains(coset_elements, element)) {
              EXPECT_TRUE(evaluation.IsOne());
            } else {
              EXPECT_TRUE(evaluation.IsZero());
            }
          }
        }
      }
    }
  } else {
    GTEST_SKIP()
        << "Skip testing FilterPolynomial on MixedRadixEvaluationDomain";
  }
}

TYPED_TEST(UnivariateEvaluationDomainTest, SizeOfElements) {
  using Domain = TypeParam;

  for (size_t coeffs = 0; coeffs < kNumCoeffs; ++coeffs) {
    size_t size = size_t{1} << coeffs;
    std::unique_ptr<Domain> domain = Domain::Create(size);
    EXPECT_EQ(domain->size(), domain->GetElements().size());
  }
}

TYPED_TEST(UnivariateEvaluationDomainTest, ContentsOfElements) {
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;

  for (size_t coeffs = 0; coeffs < kNumCoeffs; ++coeffs) {
    size_t size = size_t{1} << coeffs;
    this->TestDomains(size, [size](const BaseDomain& d) {
      std::vector<F> elements = d.GetElements();
      for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(elements[i], d.offset() * d.group_gen().Pow(i));
        EXPECT_EQ(elements[i], d.GetElement(i));
      }
    });
  }
}

// Test that lagrange interpolation for a random polynomial at a random
// point works.
TYPED_TEST(UnivariateEvaluationDomainTest, NonSystematicLagrangeCoefficients) {
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;
  using DensePoly = typename Domain::DensePoly;
  using Evals = typename Domain::Evals;

  for (size_t domain_dim = 1; domain_dim < 5; ++domain_dim) {
    size_t domain_size = size_t{1} << domain_dim;
    F rand_pt = F::Random();
    DensePoly rand_poly = DensePoly::Random(domain_size - 1);
    F actual_eval = rand_poly.Evaluate(rand_pt);
    this->TestDomains(domain_size, [domain_size, &rand_pt, &rand_poly,
                                    &actual_eval](const BaseDomain& d) {
      std::vector<F> lagrange_coeffs =
          d.EvaluateAllLagrangeCoefficients(rand_pt);

      std::vector<F> sub_lagrange_coeffs =
          d.EvaluatePartialLagrangeCoefficients(
              rand_pt, base::Range<size_t>(1, domain_size - 1));

      if (domain_size > 2) {
        sub_lagrange_coeffs.push_back(lagrange_coeffs.back());
        sub_lagrange_coeffs.insert(sub_lagrange_coeffs.begin(),
                                   lagrange_coeffs.front());
        EXPECT_EQ(lagrange_coeffs, sub_lagrange_coeffs);
      }

      Evals poly_evals = d.FFT(rand_poly);

      // Do lagrange interpolation, and compare against the actual
      // evaluation
      F interpolated_eval = F::Zero();
      for (size_t i = 0; i < domain_size; ++i) {
        interpolated_eval += lagrange_coeffs[i] * poly_evals[i];
      }
      EXPECT_EQ(actual_eval, interpolated_eval);
    });
  }
}

/// Test that lagrange coefficients for a point in the domain is correct
TYPED_TEST(UnivariateEvaluationDomainTest, SystematicLagrangeCoefficients) {
  // This runs in time O(N²) in the domain size, so keep the domain dimension
  // low. We generate lagrange coefficients for each element in the domain.
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;

  for (size_t domain_dim = 1; domain_dim < 5; ++domain_dim) {
    size_t domain_size = size_t{1} << domain_dim;
    this->TestDomains(domain_size, [domain_size](const BaseDomain& d) {
      for (size_t i = 0; i < domain_size; ++i) {
        F x = d.GetElement(i);
        std::vector<F> lagrange_coeffs = d.EvaluateAllLagrangeCoefficients(x);

        std::vector<F> sub_lagrange_coeffs =
            d.EvaluatePartialLagrangeCoefficients(
                x, base::Range<size_t>(1, domain_size - 1));

        if (domain_size > 2) {
          sub_lagrange_coeffs.push_back(lagrange_coeffs.back());
          sub_lagrange_coeffs.insert(sub_lagrange_coeffs.begin(),
                                     lagrange_coeffs.front());
          EXPECT_EQ(lagrange_coeffs, sub_lagrange_coeffs);
        }

        for (size_t j = 0; j < domain_size; ++j) {
          // Lagrange coefficient for the evaluation point,
          // which should be 1 if i == j
          if (i == j) {
            EXPECT_TRUE(lagrange_coeffs[j].IsOne());
          } else {
            EXPECT_TRUE(lagrange_coeffs[j].IsZero());
          }
        }
      }
    });
  }
}

// Tests that the ffts output the correct result.
// This assumes a correct polynomial evaluation at point procedure.
// It tests consistency of FFT/IFFT, and coset_fft/coset_ifft,
// along with testing that each individual evaluation is correct.
TYPED_TEST(UnivariateEvaluationDomainTest, FFTCorrectness) {
  // Runs in time O(degree²)
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;
  using DensePoly = typename Domain::DensePoly;
  using Evals = typename Domain::Evals;

  // NOTE(TomTaehoonKim): |degree| is set to 2⁵ -1 for default, but for the
  // |Radix2EvaluationDomain| test with openmp, it is updated to
  // |Radix2EvaluationDomain::kDefaultMinNumChunksForCompaction| - 1.
  size_t log_degree = 5;
  size_t degree = (size_t{1} << log_degree) - 1;
  DensePoly rand_poly = DensePoly::Random(degree);
  for (size_t log_domain_size = log_degree; log_domain_size < log_degree + 2;
       ++log_domain_size) {
    size_t domain_size = size_t{1} << log_domain_size;
    this->TestDomains(
        domain_size, [domain_size, &rand_poly](const BaseDomain& d) {
          Evals poly_evals = d.FFT(rand_poly);
          for (size_t i = 0; i < domain_size; ++i) {
            EXPECT_EQ(poly_evals[i], rand_poly.Evaluate(d.GetElement(i)));
          }
          EXPECT_EQ(rand_poly, d.IFFT(std::move(poly_evals)));
        });
  }
}

// Test that the degree aware FFT (O(n log d)) matches the regular FFT
// (O(n log n)).
TYPED_TEST(UnivariateEvaluationDomainTest, DegreeAwareFFTCorrectness) {
  using Domain = TypeParam;
  using F = typename Domain::Field;
  using BaseDomain = UnivariateEvaluationDomain<F, Domain::kMaxDegree>;
  using DensePoly = typename Domain::DensePoly;
  using Evals = typename Domain::Evals;

  if constexpr (std::is_same_v<F, bls12_381::Fr>) {
    const size_t log_degree = 5;
    const size_t degree = (size_t{1} << log_degree) - 1;
    DensePoly rand_poly = DensePoly::Random(degree);
    size_t domain_size = (degree + 1) * Domain::kDegreeAwareFFTThresholdFactor;
    this->TestDomains(domain_size, [domain_size,
                                    &rand_poly](const BaseDomain& d) {
      Evals deg_aware_fft_evals = d.FFT(rand_poly);
      for (size_t i = 0; i < domain_size; ++i) {
        EXPECT_EQ(deg_aware_fft_evals[i], rand_poly.Evaluate(d.GetElement(i)));
      }
    });
  } else {
    GTEST_SKIP() << "Skip testing DegreeAwareFFTCorrectness on "
                    "MixedRadixEvaluationDomain";
  }
}

TYPED_TEST(UnivariateEvaluationDomainTest, RootsOfUnity) {
  using Domain = TypeParam;
  using F = typename Domain::Field;

  for (size_t coeffs = 0; coeffs < kNumCoeffs; ++coeffs) {
    std::unique_ptr<Domain> domain = Domain::Create(coeffs);
    std::vector<F> actual_roots =
        domain->GetRootsOfUnity(domain->size(), domain->group_gen());
    for (const F& value : actual_roots) {
      EXPECT_TRUE(domain->EvaluateVanishingPolynomial(value).IsZero());
    }
    EXPECT_EQ(actual_roots.size(), domain->size());
    std::vector<F> expected_roots = domain->GetElements();
    EXPECT_EQ(absl::MakeConstSpan(actual_roots),
              absl::Span(expected_roots.data(), actual_roots.size()));
  }
}

}  // namespace tachyon::math
