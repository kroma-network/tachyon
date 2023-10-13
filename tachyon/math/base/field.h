#ifndef TACHYON_MATH_BASE_FIELD_H_
#define TACHYON_MATH_BASE_FIELD_H_

#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/internal/sysinfo.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/math/base/rings.h"

namespace tachyon::math {

// Field is any set of elements that satisfies the field axioms for both
// addition and multiplication and is commutative division algebra
// Simply put, a field is a ring in which multiplicative commutativity exists,
// and every non-zero element has a multiplicative inverse.
// See https://mathworld.wolfram.com/Field.html

// The Field supports SumOfProducts, inheriting the properties of both
// AdditiveGroup and MultiplicativeGroup.
template <typename F>
class Field : public AdditiveGroup<F>, public MultiplicativeGroup<F> {
 public:
  // Sum of products: a₁ * b₁ + a₂ * b₂ + ... + aₙ * bₙ
  template <
      typename InputIterator,
      std::enable_if_t<std::is_same_v<F, base::iter_value_t<InputIterator>>>* =
          nullptr>
  constexpr static F SumOfProducts(InputIterator a_first, InputIterator a_last,
                                   InputIterator b_first,
                                   InputIterator b_last) {
    return Ring<F>::SumOfProducts(std::move(a_first), std::move(a_last),
                                  std::move(b_first), std::move(b_last));
  }

  template <typename Container>
  constexpr static F SumOfProducts(const Container& a, const Container& b) {
    return Ring<F>::SumOfProducts(a, b);
  }

  template <typename Container>
  constexpr static bool BatchInverseInPlace(Container& fields,
                                            const F& coeff = F::One()) {
    return BatchInverse(fields, fields, coeff);
  }

  // This is taken and modified from
  // https://github.com/arkworks-rs/algebra/blob/5dfeedf560da6937a5de0a2163b7958bd32cd551/ff/src/fields/mod.rs#L355-L418.
  // Batch inverse: [b₁, b₂, ..., bₙ] = [a₁⁻¹, a₂⁻¹, ... , aₙ⁻¹]
  template <typename InputContainer, typename OutputContainer>
  constexpr static bool BatchInverse(const InputContainer& fields,
                                     OutputContainer& inverses,
                                     const F& coeff = F::One()) {
    if (fields.size() != inverses.size()) return false;

#if defined(TACHYON_HAS_OPENMP)
    using R = decltype(std::declval<F>().Inverse());

    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
    if (fields.size() >=
        size_t{1} << (thread_nums / kParallelBatchInverseDivisorThreshold)) {
      size_t num_elem_per_thread =
          (fields.size() + thread_nums - 1) / thread_nums;

      auto fields_chunks = base::Chunked(fields, num_elem_per_thread);
      auto inverses_chunks = base::Chunked(inverses, num_elem_per_thread);
      auto zipped = base::Zipped(fields_chunks, inverses_chunks);
      auto zipped_vector = base::Map(
          zipped.begin(), zipped.end(),
          [](const std::tuple<absl::Span<const F>, absl::Span<R>>& v) {
            return v;
          });

#pragma omp parallel for
      for (size_t i = 0; i < zipped_vector.size(); ++i) {
        const auto& [fields_chunk, inverses_chunk] = zipped_vector[i];
        DoBatchInverse(fields_chunk, inverses_chunk, coeff);
      }
      return true;
    }
#endif
    DoBatchInverse(absl::MakeConstSpan(fields), absl::MakeSpan(inverses),
                   coeff);
    return true;
  }

 private:
  // NOTE(chokobole): This value was chosen empirically that
  // |batch_inverse_benchmark| performs better at fewer input compared to the
  // number of cpu cores.
  constexpr static size_t kParallelBatchInverseDivisorThreshold = 4;

  FRIEND_TEST(FieldTest, BatchInverse);

  template <typename R>
  constexpr static void DoBatchInverse(const absl::Span<const F>& fields,
                                       absl::Span<R> inverses, const F& coeff) {
    // Montgomery’s Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned
    // vector by |coeff|.

    // First pass: compute [a₁, a₁ * a₂, ..., a₁ * a₂ * ... * aₙ]
    std::vector<R> productions;
    productions.reserve(fields.size() + 1);
    productions.push_back(R::One());
    R product = R::One();
    for (const F& f : fields) {
      if (!f.IsZero()) {
        product *= f;
        productions.push_back(product);
      }
    }

    // Invert |product|.
    // (a₁ * a₂ * ... *  aₙ)⁻¹
    R product_inv = product.Inverse();

    // Multiply |product_inv| by |coeff|, so all inverses will be scaled by
    // |coeff|.
    // c * (a₁ * a₂ * ... *  aₙ)⁻¹
    product_inv *= coeff;

    // Second pass: iterate backwards to compute inverses.
    //              [c * a₁⁻¹, c * a₂,⁻¹ ..., c * aₙ⁻¹]
    auto prod_it = productions.rbegin();
    ++prod_it;
    for (size_t i = fields.size() - 1; i != std::numeric_limits<size_t>::max();
         --i) {
      const F& f = fields[i];
      if (!f.IsZero()) {
        // c * (a₁ * a₂ * ... *  aᵢ)⁻¹ * aᵢ = c * (a₁ * a₂ * ... *  aᵢ₋₁)⁻¹
        R new_product_inv = product_inv * f;
        // v = c * (a₁ * a₂ * ... *  aᵢ)⁻¹ * (a₁ * a₂ * ... aᵢ₋₁)
        //   = c * aᵢ⁻¹
        inverses[i] = product_inv * (*(prod_it++));
        product_inv = std::move(new_product_inv);
      } else {
        inverses[i] = R::Zero();
      }
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_FIELD_H_
