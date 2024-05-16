// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_FIXED_BASE_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_FIXED_BASE_MSM_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/base/semigroups.h"
#include "tachyon/math/elliptic_curves/msm/msm_ctx.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::math {

// MSM(Multi-Scalar Multiplication): s₀ * g⁰ + s₁ * g¹ + ... + sₙ₋₁ * gⁿ⁻¹
// Fixed-base MSM is an operation that multiplies multiples of generators
// with respective scalars, unlike the Variable-base MSM, which uses the
// different base point for all multiplications.
template <typename Point>
class FixedBaseMSM {
 public:
  using AddResult = typename internal::AdditiveSemigroupTraits<Point>::ReturnTy;
  using ScalarField = typename Point::ScalarField;

  constexpr void Reset(size_t size, const Point& base) {
    ctx_ = MSMCtx::CreateDefault<ScalarField>(size);
    UpdateWindowTable(base);
  }

  constexpr AddResult ScalarMul(const ScalarField& scalar) const {
    // modulus_bits = 254
    // window_bits = 5
    // window_count = (254 + 4) / 5 = 51
    //
    // clang-format off
    // SG = ((2⁰S₀ + 2¹S₁ + 2²S₂ + 2³S₃ + 2⁴S₄) + 2⁵(2⁰S₅ + 2¹S₆ + 2²S₇ + 2³S₈ + 2⁴S₉) + ... +
    //       2²⁴⁵(2⁰S₂₄₅ + 2¹S₂₄₆ + 2²S₂₄₇ + 2³S₂₄₈ + 2⁴S₂₄₉) + 2²⁵⁰(2⁰S₂₅₀ + 2¹S₂₅₁ + 2²S₂₅₂ + 2³S₂₅₃))G
    // clang-format on
    using BigInt = typename ScalarField::BigIntTy;

    BigInt scalar_bigint = scalar.ToBigInt();
    auto it = BitIteratorLE<BigInt>::begin(&scalar_bigint);

    unsigned int modulus_bits = ScalarField::Config::kModulusBits;

    AddResult ret;
    for (size_t i = 0; i < ctx_.window_count; ++i) {
      size_t j = 0;
      for (size_t bit_offset = 0; bit_offset < ctx_.window_bits; ++bit_offset) {
        size_t bit_index = i * ctx_.window_bits + bit_offset;
        if (bit_index < modulus_bits && *(it++)) {
          j |= (size_t{1} << bit_offset);
        }
      }
      ret += base_multiples_[i][j];
    }
    return ret;
  }

  template <typename ScalarIterator, typename OutputIterator>
  [[nodiscard]] bool RunSerial(ScalarIterator scalars_first,
                               ScalarIterator scalars_last,
                               OutputIterator outputs_first,
                               OutputIterator outputs_last) {
    if (std::distance(scalars_first, scalars_last) !=
        std::distance(outputs_first, outputs_last)) {
      LOG(ERROR) << "the size of scalar and output iterators don't match ";
      return false;
    }
    auto scalars_it = scalars_first;
    auto outputs_it = outputs_first;
    while (scalars_it != scalars_last) {
      *outputs_it = ScalarMul(*scalars_it);
      ++scalars_it;
      ++outputs_it;
    }
    return true;
  }

  template <typename ScalarContainer, typename OutputContainer>
  [[nodiscard]] bool RunSerial(const ScalarContainer& scalars,
                               OutputContainer& outputs) {
    return RunSerial(std::begin(scalars), std::end(scalars),
                     std::begin(outputs), std::end(outputs));
  }

  template <typename ScalarIterator, typename OutputIterator>
  [[nodiscard]] bool Run(ScalarIterator scalars_first,
                         ScalarIterator scalars_last,
                         OutputIterator outputs_first,
                         OutputIterator outputs_last) {
    using scalar_iterator_category =
        typename std::iterator_traits<ScalarIterator>::iterator_category;
    using output_iterator_category =
        typename std::iterator_traits<OutputIterator>::iterator_category;
    return Run(std::move(scalars_first), std::move(scalars_last),
               std::move(outputs_first), std::move(outputs_last),
               scalar_iterator_category(), output_iterator_category());
  }

  template <typename ScalarContainer, typename OutputContainer>
  [[nodiscard]] bool Run(const ScalarContainer& scalars,
                         OutputContainer* outputs) {
    return Run(std::begin(scalars), std::end(scalars), std::begin(*outputs),
               std::end(*outputs));
  }

 private:
  template <typename ScalarIterator, typename OutputIterator>
  [[nodiscard]] bool Run(ScalarIterator scalars_first,
                         ScalarIterator scalars_last,
                         OutputIterator outputs_first,
                         OutputIterator outputs_last,
                         const std::random_access_iterator_tag&,
                         const std::random_access_iterator_tag&) {
    using difference_type =
        typename std::iterator_traits<ScalarIterator>::difference_type;
    difference_type size = std::distance(scalars_first, scalars_last);
    if (size != std::distance(outputs_first, outputs_last)) {
      LOG(ERROR) << "the size of scalar and output iterators don't match ";
      return false;
    }
    OPENMP_PARALLEL_FOR(difference_type i = 0; i < size; ++i) {
      *(outputs_first + i) = ScalarMul(*(scalars_first + i));
    }
    return true;
  }

  template <typename ScalarIterator, typename OutputIterator>
  [[nodiscard]] bool Run(ScalarIterator scalars_first,
                         ScalarIterator scalars_last,
                         OutputIterator outputs_first,
                         OutputIterator outputs_last,
                         const std::input_iterator_tag&,
                         const std::input_iterator_tag&) {
    return RunSerial(std::move(scalars_first), std::move(scalars_last),
                     std::move(outputs_first), std::move(outputs_last));
  }

  OPENMP_CONSTEXPR void UpdateWindowTable(const Point& base) {
    AddResult window_base;
    if constexpr (std::is_same_v<AddResult, Point>) {
      window_base = base;
    } else {
      window_base = math::ConvertPoint<AddResult>(base);
    }
    unsigned int window_bits = ctx_.window_bits;
    unsigned int window_count = ctx_.window_count;
    unsigned int window_size =
        static_cast<unsigned int>(size_t{1} << window_bits);
    unsigned int modulus_bits = ScalarField::Config::kModulusBits;
    unsigned int last_window_size = static_cast<unsigned int>(
        size_t{1} << (modulus_bits - (window_count - 1) * window_bits));
    // modulus_bits = 254
    // window_bits = 5
    // window_count = (254 + 4) / 5 = 51
    // window_size = 2⁵ = 32
    // last_window_size = 2^(254 - 50 * 5) = 2⁴ = 16
    //
    // The contents of |window_bases| looks like following:
    //
    // |   0   |   1   |  ...  |   49  |   50  |
    // +-------+-------+-------+-------+-------+
    // |  2⁰G  |  2⁵G  |  ...  | 2²⁴⁵G | 2²⁵⁰G |
    // +-------+-------+-------+-------+-------+
    std::vector<AddResult> window_bases =
        base::CreateVector(window_count, [&window_base, window_bits]() {
          AddResult window_base_copy = window_base;
          for (size_t i = 0; i < window_bits; ++i) {
            window_base.DoubleInPlace();
          }
          return window_base_copy;
        });

    // clang-format off
    // SG = ((2⁰S₀ + 2¹S₁ + 2²S₂ + 2³S₃ + 2⁴S₄) + 2⁵(2⁰S₅ + 2¹S₆ + 2²S₇ + 2³S₈ + 2⁴S₉) + ... +
    //       2²⁴⁵(2⁰S₂₄₅ + 2¹S₂₄₆ + 2²S₂₄₇ + 2³S₂₄₈ + 2⁴S₂₄₉) + 2²⁵⁰(2⁰S₂₅₀ + 2¹S₂₅₁ + 2²S₂₅₂ + 2³S₂₅₃))G
    // clang-format on

    // The contents of |base_multiples_| looks like following:
    //
    //  j \ i |     0     |     1     |  ...  |     49     |     50     |
    // -------+-----------+-----------+-------+------------+------------+
    //    0   |    2⁰G    |    2⁵G    |  ...  |    2²⁴⁵G   |    2²⁵⁰G   |
    // -------+-----------+-----------+-------+------------+------------+
    //    1   |  2 * 2⁰G  |  2 * 2⁵G  |  ...  |  2 * 2²⁴⁵G |  2 * 2²⁵⁰G |
    // -------+-----------+-----------+-------+------------+------------+
    //   ...  |    ...    |    ...    |  ...  |     ...    |     ...    |
    // -------+-----------+-----------+-------+------------+------------+
    //    30  |  30 * 2⁰G |  30 * 2⁵G |  ...  | 30 * 2²⁴⁵G |     0G     |
    // -------+-----------+-----------+-------+------------+------------+
    //    31  |  31 * 2⁰G |  31 * 2⁵G |  ...  | 31 * 2²⁴⁵G |     0G     |
    // -------+-----------+-----------+-------+------------+------------+

    base_multiples_ = std::vector<std::vector<AddResult>>(
        window_count, std::vector<AddResult>(window_size));
    OPENMP_PARALLEL_FOR(size_t i = 0; i < window_count; ++i) {
      size_t cur_window_size =
          i == window_count - 1 ? last_window_size : window_size;

      std::vector<AddResult>& cur_window = base_multiples_[i];
      AddResult base_multiple = AddResult::Zero();
      const AddResult& base_multiple_base = window_bases[i];
      for (size_t j = 0; j < cur_window_size; ++j) {
        cur_window[j] = base_multiple;
        base_multiple += base_multiple_base;
      }
    }
  }

  MSMCtx ctx_;
  std::vector<std::vector<AddResult>> base_multiples_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_FIXED_BASE_MSM_H_
