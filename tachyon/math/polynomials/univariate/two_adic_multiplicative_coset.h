// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_MULTIPLICATIVE_COSET_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_MULTIPLICATIVE_COSET_H_

#include <optional>
#include <utility>
#include <vector>
#include <memory>

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/optional.h"
#include "tachyon/math/base/lagrange_selectors.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"

namespace tachyon::math {

template <typename F,
          size_t MaxDegree = (size_t{1} << F::Config::kTwoAdicity) - 1>
class TwoAdicMultiplicativeCoset
    : public UnivariateEvaluationDomain<F, MaxDegree> {
 public:
  using Evals = UnivariateEvaluations<F, MaxDegree>;
  using DensePoly = UnivariateDensePolynomial<F, MaxDegree>;

  // UnivariateEvaluationDomain methods
  constexpr void DoFFT(Evals& evals) const override {}
  constexpr void DoIFFT(DensePoly& poly) const override {}
  constexpr std::unique_ptr<UnivariateEvaluationDomain<F, MaxDegree>> Clone()
      const override {
    return absl::WrapUnique(new TwoAdicMultiplicativeCoset(*this));
  }

  using UnivariateEvaluationDomain<F, MaxDegree>::UnivariateEvaluationDomain;
  TwoAdicMultiplicativeCoset(
      const UnivariateEvaluationDomain<F, MaxDegree>* domain)
      : UnivariateEvaluationDomain<F, MaxDegree>(*domain) {}

  static std::unique_ptr<TwoAdicMultiplicativeCoset> Create(size_t log_n,
                                                            F shift) {
    auto ret = new TwoAdicMultiplicativeCoset(1 << log_n, log_n);
    return absl::WrapUnique(
        new TwoAdicMultiplicativeCoset(ret->GetCoset(shift).get()));
  }

  template <typename ExtField>
  ExtField GetNextPoint(const ExtField& x) const {
    return x * this->group_gen_;
  }

  std::unique_ptr<TwoAdicMultiplicativeCoset<F>> CreateDisjointDomain(
      size_t min_size) const {
    return Create(
        base::bits::SafeLog2Ceiling(min_size),
        this->offset_ * F::FromMontgomery(F::Config::kSubgroupGenerator));
  }

  // exp_power_of_2 = x^2^log_n
  template <typename ExtField>
  ExtField GetZpAtPoint(const ExtField& point) const {
    ExtField root = point * this->offset_inv_;
    ExtField::GetRootOfUnity(this->log_size_of_group_, root);
    return root - ExtField::One();
  }

  std::vector<std::unique_ptr<TwoAdicMultiplicativeCoset<F>>> SplitDomains(
      size_t num_chunks) const {
    CHECK(base::bits::IsPowerOfTwo(num_chunks));
    size_t log_chunks = base::bits::Log2Ceiling(num_chunks);
    return base::CreateVector(num_chunks, [log_chunks, this](size_t i) {
      return Create(this->log_size_of_group_ - log_chunks,
                    this->offset_ * this->group_gen_.Pow(i));
    });
  }

  template <typename ExtField>
  LagrangeSelectors<ExtField> GetSelectorsAtPoint(const ExtField& point) const {
    ExtField unshifted_point = point * this->offset_inv_;
    ExtField root = point * this->offset_inv_;
    ExtField::GetRootOfUnity(this->log_size_of_group_, root);
    ExtField z_h = ExtField(std::move(root)) - ExtField::One();
    std::optional<ExtField> first_row =
        z_h / (unshifted_point - ExtField::One());
    CHECK(first_row);
    std::optional<ExtField> last_row =
        z_h / (unshifted_point - ExtField(this->group_gen_inv_));
    CHECK(last_row);
    ExtField transition =
        std::move(unshifted_point) - ExtField(this->group_gen_inv_);
    ExtField inv_zeroifier = unwrap(std::move(z_h).Inverse());
    return {std::move(*first_row), std::move(*last_row), std::move(transition),
            std::move(inv_zeroifier)};
  }

  LagrangeSelectors<std::vector<F>> GetSelectorsOnCoset(
      const TwoAdicMultiplicativeCoset& coset) const {
    F coset_shift = coset.offset();

    CHECK_EQ(this->offset_, F::One());
    CHECK_NE(coset_shift, F::One());
    CHECK_GE(this->log_size_of_group_, coset.log_size_of_group());
    size_t rate_bits = coset.log_size_of_group() - this->log_size_of_group_;

    F s_pow_n = coset_shift;
    // NOTE(ashjeong): CHECK(F::GetRootOfUnity(this->size_, &s_pow_n)); is
    // invalid
    for (size_t i = 0; i < this->log_size_of_group_; ++i) {
      s_pow_n.SquareInPlace();
    }
    // Evals of Z_H(X) = Xⁿ - 1
    std::vector<F> evals =
        base::Map(this->GetRootsOfUnity(1 << rate_bits, this->group_gen_),
                  [&s_pow_n](F& x) { return s_pow_n * x - F::One(); });

    std::vector<F> xs =
        base::Map(this->GetRootsOfUnity(coset.size(), coset.group_gen()),
                  [&coset_shift](F& x) { return x * coset_shift; });
    // FIRST_ROW
    F coset_i = this->group_gen_.Pow(0);
    std::vector<F> inv_denoms =
        base::Map(xs, [&coset_i](F& x) { return x - coset_i; });
    CHECK(F::BatchInverseInPlace(inv_denoms));
    std::vector<F> first_row =
        base::CreateVector(inv_denoms.size(), [&evals, &inv_denoms](size_t i) {
          return inv_denoms[i] * evals[i % evals.size()];
        });
    // LAST_ROW
    coset_i = this->group_gen_.Pow(this->size_ - 1);

    inv_denoms = base::Map(xs, [&coset_i](F& x) { return x - coset_i; });
    CHECK(F::BatchInverseInPlace(inv_denoms));
    std::vector<F> last_row =
        base::CreateVector(inv_denoms.size(), [&evals, &inv_denoms](size_t i) {
          return inv_denoms[i] * evals[i % evals.size()];
        });
    // TRANSITION
    std::vector<F> transition =
        base::Map(xs, [this](F& x) { return x - this->group_gen_inv_; });
    // INV_ZEROIFIER
    inv_denoms = evals;
    CHECK(F::BatchInverseInPlace(inv_denoms));
    std::vector<F> inv_zeroifier = base::CreateVector(
        coset.size(),
        [&inv_denoms](size_t i) { return inv_denoms[i % inv_denoms.size()]; });
    return {std::move(first_row), std::move(last_row), std::move(transition),
            std::move(inv_zeroifier)};
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_MULTIPLICATIVE_COSET_H_
