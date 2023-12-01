// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_

#include <utility>

#include "gtest/gtest_prod.h"

#include "tachyon/zk/base/evals_pair.h"
#include "tachyon/zk/plonk/lookup/lookup_committed.h"
#include "tachyon/zk/plonk/lookup/permute_expression_pair.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class LookupPermuted {
 public:
  using F = typename Poly::Field;

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

  template <typename PCSTy, typename ExtendedDomain>
  LookupCommitted<Poly> CommitGrandProduct(
      Prover<PCSTy, ExtendedDomain>* prover, const F& beta, const F& gamma) && {
    BlindedPolynomial<Poly> grand_product_poly = GrandProductArgument::Commit(
        prover, CreateNumeratorCallback<F>(beta, gamma),
        CreateDenominatorCallback<F>(beta, gamma));

    return LookupCommitted<Poly>(std::move(permuted_input_poly_),
                                 std::move(permuted_table_poly_),
                                 std::move(grand_product_poly));
  }

 private:
  FRIEND_TEST(LookupPermutedTest, ComputePermutationProduct);

  template <typename F>
  base::ParallelizeCallback3<F> CreateNumeratorCallback(const F& beta,
                                                        const F& gamma) const {
    // (A_compressed(xᵢ) + β) * (S_compressed(xᵢ) + γ)
    return [&beta, &gamma, this](absl::Span<F> chunk, size_t chunk_index,
                                 size_t chunk_size) {
      size_t i = chunk_index * chunk_size;
      for (F& value : chunk) {
        value *= (*compressed_evals_pair_.input()[i] + beta);
        value *= (*compressed_evals_pair_.table()[i] + gamma);
        ++i;
      }
    };
  }

  template <typename F>
  base::ParallelizeCallback3<F> CreateDenominatorCallback(
      const F& beta, const F& gamma) const {
    // (A'(xᵢ) + β) * (S'(xᵢ) + γ)
    return [&beta, &gamma, this](absl::Span<F> chunk, size_t chunk_index,
                                 size_t chunk_size) {
      size_t i = chunk_index * chunk_size;
      for (F& value : chunk) {
        value = (*permuted_evals_pair_.input()[i] + beta) *
                (*permuted_evals_pair_.table()[i] + gamma);
        ++i;
      }
    };
  }

  EvalsPair<Evals> compressed_evals_pair_;
  EvalsPair<Evals> permuted_evals_pair_;
  BlindedPolynomial<Poly> permuted_input_poly_;
  BlindedPolynomial<Poly> permuted_table_poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_PERMUTED_H_
