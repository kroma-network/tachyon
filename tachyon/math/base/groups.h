#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include <atomic>
#include <limits>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/math/base/semigroups.h"

namespace tachyon::math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Sub);

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
  // NOTE(chokobole): This value was chosen empirically such that
  // |batch_inverse_benchmark| performs better at fewer inputs compared to the
  // number of cpu cores.
  constexpr static size_t kParallelBatchInverseDivisorThreshold = 2;

  // Division: a * b⁻¹
  template <typename G2>
  constexpr std::optional<G> operator/(const G2& other) const {
    const std::optional<G> other_inv = other.Inverse();
    const G* g = static_cast<const G*>(this);
    if (LIKELY(other_inv)) return g->Mul(*other_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  // Division in place: a *= b⁻¹
  template <
      typename G2,
      std::enable_if_t<internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  [[nodiscard]] constexpr std::optional<G*> operator/=(const G2& other) {
    const std::optional<G> other_inv = other.Inverse();
    G* g = static_cast<G*>(this);
    if (LIKELY(other_inv)) return &g->MulInPlace(*other_inv);
    LOG_IF_NOT_GPU(ERROR) << "Division by zero attempted";
    return std::nullopt;
  }

  template <typename Container>
  [[nodiscard]] constexpr static bool BatchInverseInPlace(
      Container& groups, const G& coeff = G::One()) {
    return BatchInverse(groups, &groups, coeff);
  }

  template <typename Container>
  [[nodiscard]] constexpr static bool BatchInverseInPlaceSerial(
      Container& groups, const G& coeff = G::One()) {
    return BatchInverseSerial(groups, &groups, coeff);
  }

  // This is taken and modified from
  // https://github.com/arkworks-rs/algebra/blob/5dfeedf/ff/src/fields/mod.rs#L355-L418.
  // Batch inverse: [a₁, a₂, ..., aₙ] -> [a₁⁻¹, a₂⁻¹, ... , aₙ⁻¹]
  template <typename InputContainer, typename OutputContainer>
  [[nodiscard]] CONSTEXPR_IF_NOT_OPENMP static bool BatchInverse(
      const InputContainer& groups, OutputContainer* inverses,
      const G& coeff = G::One()) {
    size_t size = std::size(groups);
    if (size != std::size(*inverses)) {
      LOG(ERROR) << "Size of |groups| and |inverses| do not match";
      return false;
    }

#if defined(TACHYON_HAS_OPENMP)
    size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
    if (size >=
        size_t{1} << (thread_nums / kParallelBatchInverseDivisorThreshold)) {
      size_t chunk_size = base::GetNumElementsPerThread(groups);
      size_t num_chunks = (size + chunk_size - 1) / chunk_size;
      std::atomic<bool> check_valid(true);
      OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
        size_t len = i == num_chunks - 1 ? size - i * chunk_size : chunk_size;
        absl::Span<const G> groups_chunk(std::data(groups) + i * chunk_size,
                                         len);
        absl::Span<G> inverses_chunk(std::data(*inverses) + i * chunk_size,
                                     len);
        if (UNLIKELY(!DoBatchInverse(groups_chunk, inverses_chunk, coeff))) {
          check_valid.store(false, std::memory_order_relaxed);
          continue;
        }
      }
      if (UNLIKELY(!check_valid.load(std::memory_order_relaxed))) {
        LOG(ERROR) << "Inverse of zero attempted";
        return false;
      }
      return true;
    }
#endif
    return DoBatchInverse(groups, absl::MakeSpan(*inverses), coeff);
  }

  template <typename InputContainer, typename OutputContainer>
  [[nodiscard]] constexpr static bool BatchInverseSerial(
      const InputContainer& groups, OutputContainer* inverses,
      const G& coeff = G::One()) {
    if (std::size(groups) != std::size(*inverses)) {
      LOG(ERROR) << "Size of |groups| and |inverses| do not match";
      return false;
    }
    return DoBatchInverse(groups, absl::MakeSpan(*inverses), coeff);
  }

 private:
  [[nodiscard]] constexpr static bool DoBatchInverse(absl::Span<const G> groups,
                                                     absl::Span<G> inverses,
                                                     const G& coeff) {
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
    std::optional<G> product_inv_opt = product.Inverse();
    if (UNLIKELY(!product_inv_opt)) {
      LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
      return false;
    }
    G product_inv = std::move(*product_inv_opt);

    // Multiply |product_inv| by |coeff|, so all inverses will be scaled by
    // |coeff|.
    // c * (a₁ * a₂ * ... *  aₙ)⁻¹
    if (!coeff.IsOne()) product_inv *= coeff;

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
    return true;
  }
};

template <typename G>
std::ostream& operator<<(std::ostream& os, const MultiplicativeGroup<G>& g) {
  const G& derived = static_cast<const G&>(g);
  return os << derived.ToString();
}

// AdditiveGroup is a group with the group operation '+'.
// AdditiveGroup supports subtraction and negation, inheriting the
// properties of AdditiveSemigroup.
template <typename G>
class AdditiveGroup : public AdditiveSemigroup<G> {
 public:
  // Subtraction:
  //   1) a - b if subtraction is supported.
  //   2) a + (-b) otherwise
  template <typename G2,
            std::enable_if_t<internal::SupportsAdd<G, G2>::value ||
                             internal::SupportsSub<G, G2>::value>* = nullptr>
  constexpr auto operator-(const G2& other) const {
    if constexpr (internal::SupportsSub<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Sub(other);
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
    } else {
      G* g = static_cast<G*>(this);
      return g->AddInPlace(-other);
    }
  }

  // Negation: -a
  constexpr auto operator-() const {
    const G* g = static_cast<const G*>(this);
    return g->Negate();
  }
};

template <typename G>
std::ostream& operator<<(std::ostream& os, const AdditiveGroup<G>& g) {
  const G& derived = static_cast<const G&>(g);
  return os << derived.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GROUPS_H_
