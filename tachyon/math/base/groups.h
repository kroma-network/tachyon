#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/math/base/semigroups.h"

namespace tachyon::math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Div);
SUPPORTS_BINARY_OPERATOR(Mod);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Inverse);
SUPPORTS_BINARY_OPERATOR(Sub);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Neg);

}  // namespace internal

// Group 'G' is a set of elements together with a binary operation (called the
// group operation) that together satisfy the four fundamental properties of
// closure, associative, the identity property, and the inverse property.
// See https://mathworld.wolfram.com/Group.html

// MultiplicativeGroup is a group with the group operation '*'.
// MultiplicativeGroup supports division and inversion, inheriting the
// properties of MultiplicativeSemigroup.
template <typename G>
class MultiplicativeGroup : public MultiplicativeSemigroup<G> {
 public:
  // Division:
  //   1) a / b if division is supported.
  //   2) a * b⁻¹ otherwise
  template <
      typename G2,
      std::enable_if_t<internal::SupportsMul<G, G2>::value ||
                       internal::SupportsMulInPlace<G, G2>::value ||
                       internal::SupportsDiv<G, G2>::value ||
                       internal::SupportsDivInPlace<G, G2>::value>* = nullptr>
  constexpr auto operator/(const G2& other) const {
    if constexpr (internal::SupportsDiv<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Div(other);
    } else if constexpr (internal::SupportsDivInPlace<G, G2>::value) {
      G g = *static_cast<const G*>(this);
      return g.DivInPlace(other);
    } else {
      return this->operator*(other.Inverse());
    }
  }

  // Division in place: a /= b
  //   1) a /= b if division is supported.
  //   2) a *= b⁻¹ otherwise
  template <
      typename G2,
      std::enable_if_t<internal::SupportsDivInPlace<G, G2>::value ||
                       internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  constexpr G& operator/=(const G2& other) {
    if constexpr (internal::SupportsDivInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->DivInPlace(other);
    } else if constexpr (internal::SupportsMulInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->MulInPlace(other.Inverse());
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // Inverse: a⁻¹
  template <
      typename G2 = G,
      std::enable_if_t<internal::SupportsInverseInPlace<G2>::value>* = nullptr>
  [[nodiscard]] constexpr auto Inverse() const {
    G ret = *static_cast<const G*>(this);
    return ret.InverseInPlace();
  }

  template <typename Container>
  constexpr static bool BatchInverseInPlace(Container& groups,
                                            const G& coeff = G::One()) {
    return BatchInverse(groups, &groups, coeff);
  }

  // This is taken and modified from
  // https://github.com/arkworks-rs/algebra/blob/5dfeedf560da6937a5de0a2163b7958bd32cd551/ff/src/fields/mod.rs#L355-L418.
  // Batch inverse: [a₁, a₂, ..., aₙ] -> [a₁⁻¹, a₂⁻¹, ... , aₙ⁻¹]
  template <typename InputContainer, typename OutputContainer>
  constexpr static bool BatchInverse(const InputContainer& groups,
                                     OutputContainer* inverses,
                                     const G& coeff = G::One()) {
    if (std::size(groups) != std::size(*inverses)) {
      LOG(ERROR) << "Size of |groups| and |inverses| do not match";
      return false;
    }

#if defined(TACHYON_HAS_OPENMP)
    using G2 = decltype(std::declval<G>().Inverse());
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
    if (std::size(groups) >=
        size_t{1} << (thread_nums / kParallelBatchInverseDivisorThreshold)) {
      size_t num_elem_per_thread =
          (std::size(groups) + thread_nums - 1) / thread_nums;

      auto groups_chunks = base::Chunked(groups, num_elem_per_thread);
      auto inverses_chunks = base::Chunked(*inverses, num_elem_per_thread);
      auto zipped = base::Zipped(groups_chunks, inverses_chunks);
      auto zipped_vector = base::Map(
          zipped.begin(), zipped.end(),
          [](const std::tuple<absl::Span<const G2>, absl::Span<G2>>& v) {
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
    DoBatchInverse(absl::MakeConstSpan(groups), absl::MakeSpan(*inverses),
                   coeff);
    return true;
  }

 private:
  // NOTE(chokobole): This value was chosen empirically that
  // |batch_inverse_benchmark| performs better at fewer input compared to the
  // number of cpu cores.
  constexpr static size_t kParallelBatchInverseDivisorThreshold = 4;

  FRIEND_TEST(GroupsTest, BatchInverse);

  constexpr static void DoBatchInverse(absl::Span<const G> groups,
                                       absl::Span<G> inverses, const G& coeff) {
    // Montgomery’s Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned
    // vector by |coeff|.

    // First pass: compute [a₁, a₁ * a₂, ..., a₁ * a₂ * ... * aₙ]
    std::vector<G> productions;
    productions.reserve(groups.size() + 1);
    productions.push_back(G::One());
    G product = G::One();
    for (const G& g : groups) {
      if (!g.IsZero()) {
        product *= g;
        productions.push_back(product);
      }
    }

    // Invert |product|.
    // (a₁ * a₂ * ... *  aₙ)⁻¹
    G product_inv = product.Inverse();

    // Multiply |product_inv| by |coeff|, so all inverses will be scaled by
    // |coeff|.
    // c * (a₁ * a₂ * ... *  aₙ)⁻¹
    product_inv *= coeff;

    // Second pass: iterate backwards to compute inverses.
    //              [c * a₁⁻¹, c * a₂,⁻¹ ..., c * aₙ⁻¹]
    auto prod_it = productions.rbegin();
    ++prod_it;
    for (size_t i = groups.size() - 1; i != std::numeric_limits<size_t>::max();
         --i) {
      const G& g = groups[i];
      if (!g.IsZero()) {
        // c * (a₁ * a₂ * ... *  aᵢ)⁻¹ * aᵢ = c * (a₁ * a₂ * ... *  aᵢ₋₁)⁻¹
        G new_product_inv = product_inv * g;
        // v = c * (a₁ * a₂ * ... *  aᵢ)⁻¹ * (a₁ * a₂ * ... aᵢ₋₁) = c * aᵢ⁻¹
        inverses[i] = product_inv * (*(prod_it++));
        product_inv = std::move(new_product_inv);
      } else {
        inverses[i] = G::Zero();
      }
    }
  }
};

// AdditiveGroup is a group with the group operation '+'.
// AdditiveGroup supports subtraction and negation, inheriting the
// properties of AdditiveSemigroup.
template <typename G>
class AdditiveGroup : public AdditiveSemigroup<G> {
 public:
  // Subtraction:
  //   1) a - b if subtraction is supported.
  //   2) a + (-b) otherwise
  template <
      typename G2,
      std::enable_if_t<internal::SupportsAdd<G, G2>::value ||
                       internal::SupportsAddInPlace<G, G2>::value ||
                       internal::SupportsSub<G, G2>::value ||
                       internal::SupportsSubInPlace<G, G2>::value>* = nullptr>
  constexpr auto operator-(const G2& other) const {
    if constexpr (internal::SupportsSub<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Sub(other);
    } else if constexpr (internal::SupportsSubInPlace<G, G2>::value) {
      G g = *static_cast<const G*>(this);
      return g.SubInPlace(other);
    } else {
      return this->operator+(-other);
    }
  }

  // Subtraction in place:
  //   1) a -= b if subtraction is supported.
  //   2) a += (-b) otherwise
  template <
      typename G2,
      std::enable_if_t<internal::SupportsSubInPlace<G, G2>::value ||
                       internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr G& operator-=(const G2& other) {
    if constexpr (internal::SupportsSubInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->SubInPlace(other);
    } else if constexpr (internal::SupportsAddInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->AddInPlace(-other);
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // Negation: -a
  constexpr auto operator-() const {
    if constexpr (internal::SupportsNegInPlace<G>::value) {
      G g = *static_cast<const G*>(this);
      return g.NegInPlace();
    } else {
      const G* g = static_cast<const G*>(this);
      return g->Negative();
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GROUPS_H_
