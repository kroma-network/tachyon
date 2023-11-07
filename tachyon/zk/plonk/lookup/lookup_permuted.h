// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_

#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/evals_pair.h"
#include "tachyon/zk/plonk/lookup/lookup_committed.h"
#include "tachyon/zk/plonk/lookup/permute_expression_pair.h"

namespace tachyon::zk {

template <typename PCSTy>
class LookupPermuted {
 public:
  using F = typename PCSTy::Field;
  using Evals = typename PCSTy::Evals;
  using Poly = typename PCSTy::Poly;
  using Commitment = typename PCSTy::Commitment;
  using Domain = typename PCSTy::Domain;

  LookupPermuted() = default;
  LookupPermuted(EvalsPair<Evals> compressed_evals_pair,
                 EvalsPair<Evals> permuted_evals_pair,
                 BlindedPolynomial<Poly> permuted_input_poly,
                 BlindedPolynomial<Poly> permuted_table_poly)
      : compressed_evals_pair_(std::move(compressed_evals_pair)),
        permuted_evals_pair_(std::move(permuted_evals_pair)),
        permuted_input_poly_(std::move(permuted_input_poly)),
        permuted_table_poly_(std::move(permuted_table_poly)) {}

  const EvalsPair<Evals>& compressed_evals_pair() {
    return compressed_evals_pair_;
  }
  const EvalsPair<Evals>& permuted_evals_pair() { return permuted_evals_pair_; }
  const BlindedPolynomial<Poly>& permuted_input_poly() {
    return permuted_input_poly_;
  }
  const BlindedPolynomial<Poly>& permuted_table_poly() {
    return permuted_table_poly_;
  }

  template <typename ProverTy>
  LookupCommitted<PCSTy> CommitProduct(ProverTy& prover, const F& beta,
                                       const F& gamma) && {
    size_t blinding_factors = prover.blinder().blinded_factors();
    Evals z = ComputePermutationProduct(blinding_factors, beta, gamma,
                                        prover.pcs().N());
    prover.blinder().Blind(z);

    BlindedPolynomial<PCSTy> product_poly;
    CHECK(prover.CommitEvalsWithBlind(z, &product_poly));

    return LookupCommitted<PCSTy>(std::move(permuted_input_poly_),
                                  std::move(permuted_table_poly_),
                                  std::move(product_poly));
  }

 private:
  FRIEND_TEST(LookupPermutedTest, ComputePermutationProduct);

  Evals ComputePermutationProduct(size_t blinding_factors, const F& beta,
                                  const F& gamma, size_t params_size) const {
    std::vector<F> lookup_product(params_size, F::Zero());

    // 1. lookup_product[i] =
    // (A'(xᵢ) + β) * (S'(xᵢ) + γ)
    base::Parallelize(lookup_product, [&beta, &gamma, this](absl::Span<F> chunk,
                                                            size_t chunk_index,
                                                            size_t chunk_size) {
      size_t i = chunk_index * chunk_size;
      for (F& value : chunk) {
        value = (*permuted_evals_pair_.input()[i] + beta) *
                (*permuted_evals_pair_.table()[i] + gamma);
        ++i;
      }
    });

    // 2. lookup_product[i] =
    //               1
    //   ─────────────────────────
    //   (A'(xᵢ) + β) * (S'(xᵢ) + γ)
    F::BatchInverseInPlace(lookup_product);

    // 3. lookup_product[i] =
    //  (A_compressed(xᵢ) + β) * (S_compressed(xᵢ) + γ)
    //  ─────────────────────────────────────────────
    //            (A'(xᵢ) + β) * (S'(xᵢ) + γ)
    base::Parallelize(lookup_product, [&beta, &gamma, this](absl::Span<F> chunk,
                                                            size_t chunk_index,
                                                            size_t chunk_size) {
      size_t i = chunk_index * chunk_size;
      for (F& value : chunk) {
        value *= (*compressed_evals_pair_.input()[i] + beta);
        value *= (*compressed_evals_pair_.table()[i] + gamma);
        ++i;
      }
    });

    std::vector<F> z;
    z.resize(params_size);
    z[0] = F::One();
    for (size_t i = 0; i < params_size - blinding_factors - 1; ++i) {
      z[i + 1] = z[i] * lookup_product[i];
    }
    return Evals(std::move(z));
  }

  EvalsPair<Evals> compressed_evals_pair_;
  EvalsPair<Evals> permuted_evals_pair_;
  BlindedPolynomial<Poly> permuted_input_poly_;
  BlindedPolynomial<Poly> permuted_table_poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_
