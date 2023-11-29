#ifndef TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_

#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/prover.h"

namespace tachyon::zk {

class GrandProductArgument {
 public:
  template <typename PCSTy, typename Callable,
            typename Poly = typename PCSTy::Poly>
  static BlindedPolynomial<Poly> Commit(Prover<PCSTy>* prover,
                                        Callable numerator_callback,
                                        Callable denominator_callback) {
    using Evals = typename PCSTy::Evals;

    size_t size = prover->pcs().N();
    size_t blinding_factors = prover->blinder().blinding_factors();
    Evals z = CreatePolynomial<Evals>(size, blinding_factors,
                                      std::move(numerator_callback),
                                      std::move(denominator_callback));
    CHECK(prover->blinder().Blind(z));

    BlindedPolynomial<Poly> ret;
    CHECK(prover->CommitEvalsWithBlind(z, &ret));
    return ret;
  }

 private:
  FRIEND_TEST(LookupPermutedTest, ComputePermutationProduct);

  template <typename Evals, typename Callable>
  static Evals CreatePolynomial(size_t size, size_t blinding_factors,
                                Callable numerator_callback,
                                Callable denominator_callback) {
    using F = typename Evals::Field;

    std::vector<F> grand_product(size, F::Zero());

    base::Parallelize(grand_product, std::move(denominator_callback));

    F::BatchInverseInPlace(grand_product);

    base::Parallelize(grand_product, std::move(numerator_callback));

    std::vector<F> z;
    z.resize(size);
    z[0] = F::One();
    for (size_t i = 0; i < size - blinding_factors - 1; ++i) {
      z[i + 1] = z[i] * grand_product[i];
    }
    return Evals(std::move(z));
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_
