#ifndef TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/entities/prover_base.h"

namespace tachyon::zk::plonk {

class GrandProductArgument {
 public:
  // If the number of rows is within than the supported size of polynomial
  // commitment scheme, you should use this version. See lookup argument for use
  // case.
  template <typename PCS, typename Callable,
            typename Evals = typename PCS::Evals>
  static Evals CreatePoly(ProverBase<PCS>* prover, Callable numerator_callback,
                          Callable denominator_callback) {
    using F = typename Evals::Field;

    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex size = static_cast<RowIndex>(prover->pcs().N());
    std::vector<F> z(size + 1);
    absl::Span<F> grand_product = absl::MakeSpan(z).subspan(1);

    base::Parallelize(grand_product, std::move(denominator_callback));

    CHECK(F::BatchInverseInPlace(grand_product));

    base::Parallelize(grand_product, std::move(numerator_callback));

    F last_z = F::One();
    return DoCreatePoly(prover, last_z, std::move(z));
  }

  template <typename PCS, typename Callable,
            typename Evals = typename PCS::Evals>
  static Evals CreatePolySerial(ProverBase<PCS>* prover,
                                Callable numerator_callback,
                                Callable denominator_callback) {
    using F = typename Evals::Field;

    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex size = static_cast<RowIndex>(prover->pcs().N());
    std::vector<F> z(size + 1);
    absl::Span<F> grand_product = absl::MakeSpan(z).subspan(1);

    for (RowIndex i = 0; i < size; ++i) {
      grand_product[i] = denominator_callback(i);
    }

    CHECK(F::BatchInverseInPlaceSerial(grand_product));

    for (RowIndex i = 0; i < size; ++i) {
      grand_product[i] *= numerator_callback(i);
    }

    F last_z = F::One();
    return DoCreatePoly(prover, last_z, std::move(z));
  }

  // If the number of rows is out of the supported size of polynomial
  // commitment scheme, you should use this version. See permutation argument
  // for use case.
  // See
  // https://zcash.github.io/halo2/design/proving-system/permutation.html#spanning-a-large-number-of-columns
  template <typename PCS, typename Callable, typename F,
            typename Evals = typename PCS::Evals>
  static Evals CreateExcessivePoly(ProverBase<PCS>* prover,
                                   Callable numerator_callback,
                                   Callable denominator_callback,
                                   size_t num_cols, F& last_z) {
    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex size = static_cast<RowIndex>(prover->pcs().N());
    std::vector<F> z(size + 1, F::One());
    absl::Span<F> grand_product = absl::MakeSpan(z).subspan(1);

    base::Parallelize(
        grand_product.size(),
        [&grand_product, numerator_callback, denominator_callback, num_cols](
            size_t len, size_t chunk_offset, size_t chunk_size) {
          RowIndex start = chunk_offset * chunk_size;
          for (size_t i = 0; i < num_cols; ++i) {
            for (RowIndex j = start; j < start + len; ++j) {
              grand_product[j] *= denominator_callback(i, j);
            }
          }

          auto grand_subspan = grand_product.subspan(start, len);
          CHECK(F::BatchInverseInPlaceSerial(grand_subspan));

          for (size_t i = 0; i < num_cols; ++i) {
            for (RowIndex j = start; j < start + len; ++j) {
              grand_product[j] *= numerator_callback(i, j);
            }
          }
        });

    return DoCreatePoly(prover, last_z, std::move(z));
  }

 private:
  template <typename PCS, typename F, typename Evals = typename PCS::Evals>
  static Evals DoCreatePoly(ProverBase<PCS>* prover, F& last_z,
                            std::vector<F>&& grand_product) {
    RowIndex usable_rows = prover->GetUsableRows();

    // TODO(chokobole): Apply the same optimization trick used in grand sum.
    absl::Span<F> z(grand_product);
    z[0] = last_z;
    for (RowIndex i = 0; i < usable_rows; ++i) {
      z[i + 1] = z[i] * grand_product[i + 1];
    }
    last_z = z[usable_rows];
    grand_product.pop_back();

    Evals z_evals(std::move(grand_product));
    CHECK(prover->blinder().Blind(z_evals));
    return z_evals;
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_GRAND_PRODUCT_ARGUMENT_H_
