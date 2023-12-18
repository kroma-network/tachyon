#ifndef TACHYON_MATH_BASE_RINGS_H_
#define TACHYON_MATH_BASE_RINGS_H_

#include <type_traits>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/math/base/groups.h"

namespace tachyon::math {

// Ring is a set S with operations + and * that satisfies the followings:
// 1. Additive associativity: (a + b) + c = a + (b + c)
// 2. Additive commutativity: a + b = b + a
// 3. Additive identity: a + 0 = 0 + a = a
// 4. Additive inverse: a + (-a) = (-a) + a = 0
// 5. Distributivity: a * (b + c) = (a * b) + (a * c)
// 6. Multiplicative associativity: (a * b) * c = a * (b * c)
// See https://mathworld.wolfram.com/Ring.html

// The Ring supports SumOfProducts, inheriting the properties of both
// AdditiveGroup and MultiplicativeSemigroup.
template <typename F>
class Ring : public AdditiveGroup<F>, public MultiplicativeSemigroup<F> {
 public:
  // This is taken and modified from
  // https://github.com/arkworks-rs/algebra/blob/5dfeedf560da6937a5de0a2163b7958bd32cd551/ff/src/fields/mod.rs#L298C1-L305
  // Sum of products: a₁ * b₁ + a₂ * b₂ + ... + aₙ * bₙ
  // TODO(chokobole): If I call |SumOfProducts()| instead of
  // |SumOfProductsSerial| for all call sites, it gets stuck when doing
  // unittests. I think we need a some general threshold to check whether it is
  // good to doing parallelization.
  template <typename Container>
  constexpr static F SumOfProducts(const Container& a, const Container& b) {
    size_t size = std::size(a);
    CHECK_EQ(size, std::size(b));
    CHECK_NE(size, size_t{0});
    std::vector<F> partial_sum_of_products = base::ParallelizeMap(
        a,
        [&b](absl::Span<const F> chunk, size_t chunk_idx, size_t chunk_size) {
          F sum = F::Zero();
          size_t i = chunk_idx * chunk_size;
          for (size_t j = 0; j < chunk.size(); ++j) {
            sum += (chunk[j] * b[i + j]);
          }
          return sum;
        });
    return std::accumulate(partial_sum_of_products.begin(),
                           partial_sum_of_products.end(), F::Zero(),
                           [](F& acc, const F& partial_sum_of_product) {
                             return acc += partial_sum_of_product;
                           });
  }

  template <typename Container>
  constexpr static F SumOfProductsSerial(const Container& a,
                                         const Container& b) {
    size_t size = std::size(a);
    CHECK_EQ(size, std::size(b));
    CHECK_NE(size, size_t{0});
    return DoSumOfProductsSerial(a, b);
  }

 private:
  template <typename Container>
  constexpr static F DoSumOfProductsSerial(const Container& a,
                                           const Container& b) {
    size_t n = std::size(a);
    F sum = F::Zero();
    for (size_t i = 0; i < n; ++i) {
      sum += (a[i] * b[i]);
    }
    return sum;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_RINGS_H_
