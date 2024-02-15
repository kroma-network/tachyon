#ifndef TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"

namespace tachyon::zk {
namespace lookup::halo2 {

class LookupArgumentRunnerTest_ComputePermutationProduct_Test;

}  // namespace lookup::halo2

namespace plonk {

class GrandProductArgument {
 public:
  // If the number of rows is within than the supported size of polynomial
  // commitment scheme, you should use this version. See lookup argument for use
  // case.
  template <typename PCS, typename Callable, typename Poly = typename PCS::Poly>
  static BlindedPolynomial<Poly> Commit(ProverBase<PCS>* prover,
                                        Callable numerator_callback,
                                        Callable denominator_callback) {
    using Evals = typename PCS::Evals;

    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex size = static_cast<RowIndex>(prover->pcs().N());
    RowIndex usable_rows = prover->GetUsableRows();
    Evals z = CreatePolynomial<Evals>(size, usable_rows,
                                      std::move(numerator_callback),
                                      std::move(denominator_callback));
    CHECK(prover->blinder().Blind(z));
    return prover->CommitAndWriteToProofWithBlind(z);
  }

  // If the number of rows is out of the supported size of polynomial
  // commitment scheme, you should use this version. See permutation argument
  // for use case.
  // See
  // https://zcash.github.io/halo2/design/proving-system/permutation.html#spanning-a-large-number-of-columns
  template <typename PCS, typename Callable, typename F,
            typename Poly = typename PCS::Poly>
  static BlindedPolynomial<Poly> CommitExcessive(ProverBase<PCS>* prover,
                                                 Callable numerator_callback,
                                                 Callable denominator_callback,
                                                 size_t num_cols, F& last_z) {
    using Evals = typename PCS::Evals;

    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex size = static_cast<RowIndex>(prover->pcs().N());
    RowIndex usable_rows = prover->GetUsableRows();
    Evals z = CreatePolynomialExcessive<Evals>(
        size, usable_rows, num_cols, last_z, std::move(numerator_callback),
        std::move(denominator_callback));
    CHECK(prover->blinder().Blind(z));
    return prover->CommitAndWriteToProofWithBlind(z);
  }

 private:
  friend class zk::lookup::halo2::
      LookupArgumentRunnerTest_ComputePermutationProduct_Test;

  template <typename Evals, typename Callable>
  static Evals CreatePolynomial(RowIndex size, RowIndex usable_rows,
                                Callable numerator_callback,
                                Callable denominator_callback) {
    using F = typename Evals::Field;

    std::vector<F> z = base::CreateVector(size + 1, F::Zero());
    absl::Span<F> grand_product = absl::MakeSpan(z).subspan(1);

    base::Parallelize(grand_product, std::move(denominator_callback));

    CHECK(F::BatchInverseInPlace(grand_product));

    base::Parallelize(grand_product, std::move(numerator_callback));

    F last_z = F::One();
    return DoCreatePolynomial<Evals>(size, usable_rows, last_z, std::move(z));
  }

  template <typename Evals, typename F, typename Callable>
  static Evals CreatePolynomialExcessive(RowIndex size, RowIndex usable_rows,
                                         size_t num_cols, F& last_z,
                                         Callable numerator_callback,
                                         Callable denominator_callback) {
    std::vector<F> z = base::CreateVector(size + 1, F::One());
    absl::Span<F> grand_product = absl::MakeSpan(z).subspan(1);

    for (size_t i = 0; i < num_cols; ++i) {
      base::Parallelize(grand_product, denominator_callback(i));
    }

    CHECK(F::BatchInverseInPlace(grand_product));

    for (size_t i = 0; i < num_cols; ++i) {
      base::Parallelize(grand_product, numerator_callback(i));
    }

    return DoCreatePolynomial<Evals>(size, usable_rows, last_z, std::move(z));
  }

  template <typename Evals, typename F>
  static Evals DoCreatePolynomial(RowIndex size, RowIndex usable_rows,
                                  F& last_z, std::vector<F>&& grand_product) {
    absl::Span<F> z = absl::MakeSpan(grand_product);
    z[0] = last_z;
    for (RowIndex i = 0; i < usable_rows; ++i) {
      z[i + 1] = z[i] * grand_product[i + 1];
    }
    last_z = z[usable_rows];
    grand_product.pop_back();
    return Evals(std::move(grand_product));
  }
};

}  // namespace plonk
}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_
