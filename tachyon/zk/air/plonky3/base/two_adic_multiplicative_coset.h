// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_ZK_AIR_PLONKY3_BASE_TWO_ADIC_MULTIPLICATIVE_COSET_H_
#define TACHYON_ZK_AIR_PLONKY3_BASE_TWO_ADIC_MULTIPLICATIVE_COSET_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/optional.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"
#include "tachyon/zk/air/plonky3/base/lagrange_selectors.h"

namespace tachyon::zk::air::plonky3 {

template <typename F>
class TwoAdicMultiplicativeCoset {
  static_assert(F::Config::kModulusBits <= 32);

 public:
  constexpr TwoAdicMultiplicativeCoset() = default;

  TwoAdicMultiplicativeCoset(size_t log_n, const F& shift) {
    domain_.reset(static_cast<math::Radix2EvaluationDomain<F>*>(
        math::Radix2EvaluationDomain<F>::Create(1 << log_n)
            ->GetCoset(shift)
            .release()));
  }

  const math::Radix2EvaluationDomain<F>* domain() const {
    return domain_.get();
  }

  template <typename ExtField>
  ExtField GetNextPoint(const ExtField& x) const {
    return x * domain_->group_gen();
  }

  TwoAdicMultiplicativeCoset<F> CreateDisjointDomain(size_t min_size) const {
    return {
        base::bits::SafeLog2Ceiling(min_size),
        domain_->offset() * F::FromMontgomery(F::Config::kSubgroupGenerator)};
  }

  template <typename ExtField>
  ExtField GetZpAtPoint(const ExtField& point) const {
    return (point * domain_->offset_inv())
               .ExpPowOfTwo(domain_->log_size_of_group()) -
           ExtField::One();
  }

  std::vector<TwoAdicMultiplicativeCoset<F>> SplitDomains(
      size_t num_chunks) const {
    CHECK(base::bits::IsPowerOfTwo(num_chunks));
    uint32_t log_chunks = base::bits::Log2Ceiling(num_chunks);
    F f = domain_->offset();
    return base::CreateVector(num_chunks, [this, log_chunks, &f](size_t i) {
      TwoAdicMultiplicativeCoset ret{domain_->log_size_of_group() - log_chunks,
                                     f};
      f *= domain_->group_gen();
      return ret;
    });
  }

  template <typename ExtField>
  LagrangeSelectors<ExtField> GetSelectorsAtPoint(const ExtField& point) const {
    ExtField unshifted_point = point * domain_->offset_inv();
    ExtField z_h = unshifted_point.ExpPowOfTwo(domain_->log_size_of_group()) -
                   ExtField::One();
    ExtField first_row = unwrap(z_h / (unshifted_point - ExtField::One()));
    ExtField transition = unshifted_point - ExtField(domain_->group_gen_inv());
    ExtField last_row = unwrap(z_h / transition);
    ExtField inv_zeroifier = unwrap(z_h.Inverse());
    return {std::move(first_row), std::move(last_row), std::move(transition),
            std::move(inv_zeroifier)};
  }

  LagrangeSelectors<std::vector<F>> GetSelectorsOnCoset(
      const TwoAdicMultiplicativeCoset& coset) const {
    const F& coset_shift = coset.domain()->offset();

    CHECK_EQ(domain_->offset(), F::One());
    CHECK_NE(coset_shift, F::One());
    CHECK_GE(domain_->log_size_of_group(), coset.domain()->log_size_of_group());
    size_t rate_bits =
        coset.domain()->log_size_of_group() - domain_->log_size_of_group();
    F s_pow_n = coset_shift.ExpPowOfTwo(domain_->log_size_of_group());

    // TODO(ashjeong): Reduce the number of thread joins since |evals| and |xs|
    // can be created independently.
    // Evals of Z_H(X) = Xⁿ - 1
    std::vector<F> evals =
        F::GetSuccessivePowers(1 << rate_bits, domain_->group_gen(), s_pow_n);
    std::vector<F> inv_denoms_inv_zeroifier(evals.size());
    base::Parallelize(evals, [&evals, &inv_denoms_inv_zeroifier](
                                 absl::Span<F> chunk, size_t chunk_offset,
                                 size_t chunk_size) {
      size_t start = chunk_offset * chunk_size;
      for (size_t i = 0; i < chunk_size; ++i) {
        size_t idx = start + i;
        evals[idx] -= F::One();
        inv_denoms_inv_zeroifier[idx] = evals[idx];
      }
      absl::Span<F> inv_denoms_inv_zeroifier_chunk =
          absl::MakeSpan(&inv_denoms_inv_zeroifier[start], chunk_size);
      CHECK(F::BatchInverseInPlaceSerial(inv_denoms_inv_zeroifier_chunk));
    });

    size_t sz = coset.domain()->size();
    std::vector<F> xs =
        F::GetSuccessivePowers(sz, coset.domain()->group_gen(), coset_shift);

    F coset_i = domain_->group_gen().Pow(domain_->size() - 1);

    std::vector<F> first_row(sz);
    std::vector<F> last_row(sz);
    std::vector<F> transition(sz);
    std::vector<F> inv_zeroifier(sz);

    base::Parallelize(
        xs, [this, &evals, &coset_i, &inv_denoms_inv_zeroifier, &first_row,
             &last_row, &transition, &inv_zeroifier](
                absl::Span<F> chunk, size_t chunk_offset, size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          for (size_t i = 0; i < chunk_size; ++i) {
            size_t idx = start + i;
            first_row[idx] = chunk[i] - F::One();
            last_row[idx] = chunk[i] - coset_i;
          }
          absl::Span<F> first_row_chunk =
              absl::MakeSpan(&first_row[start], chunk_size);
          CHECK(F::BatchInverseInPlaceSerial(first_row_chunk));
          absl::Span<F> last_row_chunk =
              absl::MakeSpan(&last_row[start], chunk_size);
          CHECK(F::BatchInverseInPlaceSerial(last_row_chunk));

          for (size_t i = 0; i < chunk_size; ++i) {
            size_t idx = start + i;
            size_t evals_i = idx % evals.size();
            first_row[idx] *= evals[evals_i];
            last_row[idx] *= evals[evals_i];
            transition[idx] = chunk[i] - domain_->group_gen_inv();
            inv_zeroifier[idx] = inv_denoms_inv_zeroifier[evals_i];
          }
        });

    return {std::move(first_row), std::move(last_row), std::move(transition),
            std::move(inv_zeroifier)};
  }

 private:
  std::unique_ptr<math::Radix2EvaluationDomain<F>> domain_;
};

}  // namespace tachyon::zk::air::plonky3

#endif  // TACHYON_ZK_AIR_PLONKY3_BASE_TWO_ADIC_MULTIPLICATIVE_COSET_H_
