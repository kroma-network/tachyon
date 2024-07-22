// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_PROVER_IMPL_H_
#define TACHYON_ZK_LOOKUP_HALO2_PROVER_IMPL_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/lookup/halo2/permute_expression_pair.h"
#include "tachyon/zk/lookup/halo2/prover.h"
#include "tachyon/zk/plonk/expressions/compress_expression.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"

namespace tachyon::zk::lookup::halo2 {

// static
template <typename Poly, typename Evals>
template <typename Domain>
Pair<Evals> Prover<Poly, Evals>::CompressPair(
    const Domain* domain, const Argument<F>& argument, const F& theta,
    const plonk::ProvingEvaluator<Evals>& evaluator_tpl) {
  // A_compressedᵢ(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
  Evals compressed_input = plonk::CompressExpressions(
      domain, argument.input_expressions(), theta, evaluator_tpl);

  // S_compressedᵢ(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
  Evals compressed_table = plonk::CompressExpressions(
      domain, argument.table_expressions(), theta, evaluator_tpl);

  return {std::move(compressed_input), std::move(compressed_table)};
}

template <typename Poly, typename Evals>
template <typename Domain>
void Prover<Poly, Evals>::CompressPairs(
    const Domain* domain, const std::vector<Argument<F>>& arguments,
    const F& theta, const plonk::ProvingEvaluator<Evals>& evaluator_tpl) {
  compressed_pairs_ = base::Map(
      arguments, [domain, &theta, &evaluator_tpl](const Argument<F>& argument) {
        return CompressPair(domain, argument, theta, evaluator_tpl);
      });
}

// static
template <typename Poly, typename Evals>
template <typename Domain>
void Prover<Poly, Evals>::BatchCompressPairs(
    std::vector<Prover>& lookup_provers, const Domain* domain,
    const std::vector<Argument<F>>& arguments, const F& theta,
    const std::vector<plonk::MultiPhaseRefTable<Evals>>& tables) {
  CHECK_EQ(lookup_provers.size(), tables.size());
  // NOTE(chokobole): It's safe to downcast because domain is already checked.
  int32_t n = static_cast<int32_t>(domain->size());
  for (size_t i = 0; i < lookup_provers.size(); ++i) {
    plonk::ProvingEvaluator<Evals> proving_evaluator(0, n, 1, tables[i]);
    lookup_provers[i].CompressPairs(domain, arguments, theta,
                                    proving_evaluator);
  }
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
Pair<BlindedPolynomial<Poly, Evals>> Prover<Poly, Evals>::PermutePair(
    ProverBase<PCS>* prover, const Pair<Evals>& compressed_pair) {
  // A'ᵢ(X), S'ᵢ(X)
  Pair<Evals> permuted_pair;
  CHECK(PermuteExpressionPair(prover, compressed_pair, &permuted_pair));

  F input_blind = prover->blinder().Generate();
  F table_blind = prover->blinder().Generate();
  return {{std::move(permuted_pair).TakeInput(), std::move(input_blind)},
          {std::move(permuted_pair).TakeTable(), std::move(table_blind)}};
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::PermutePairs(ProverBase<PCS>* prover) {
  permuted_pairs_ = base::Map(compressed_pairs_,
                              [prover](const Pair<Evals>& compressed_pair) {
                                return PermutePair(prover, compressed_pair);
                              });
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::BatchCommitPermutedPairs(
    const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
    size_t& commit_idx) {
  if (lookup_provers.empty()) return;

  if constexpr (PCS::kSupportsBatchMode) {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair :
           lookup_prover.permuted_pairs_) {
        prover->BatchCommitAt(permuted_pair.input().evals(), commit_idx++);
        prover->BatchCommitAt(permuted_pair.table().evals(), commit_idx++);
      }
    }
  } else {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair :
           lookup_prover.permuted_pairs_) {
        prover->CommitAndWriteToProof(permuted_pair.input().evals());
        prover->CommitAndWriteToProof(permuted_pair.table().evals());
      }
    }
  }
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
BlindedPolynomial<Poly, Evals> Prover<Poly, Evals>::CreateGrandProductPoly(
    ProverBase<PCS>* prover, const Pair<Evals>& compressed_pair,
    const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair, const F& beta,
    const F& gamma) {
  return {plonk::GrandProductArgument::CreatePolySerial(
              prover, CreateNumeratorCallback(compressed_pair, beta, gamma),
              CreateDenominatorCallback(permuted_pair, beta, gamma)),
          prover->blinder().Generate()};
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::CreateGrandProductPolys(ProverBase<PCS>* prover,
                                                  const F& beta,
                                                  const F& gamma) {
  // Zₗ,ᵢ(X)
  CHECK_EQ(compressed_pairs_.size(), permuted_pairs_.size());
  grand_product_polys_.resize(compressed_pairs_.size());

  // NOTE(dongchangYoo): do not change this code to parallelized logic.
  grand_product_polys_ = base::Map(
      compressed_pairs_, [this, prover, &beta, &gamma](
                             size_t i, const Pair<Evals>& compressed_pair) {
        return CreateGrandProductPoly(prover, compressed_pair,
                                      permuted_pairs_[i], beta, gamma);
      });
  compressed_pairs_.clear();
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::BatchCommitGrandProductPolys(
    const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
    size_t& commit_idx) {
  if (lookup_provers.empty()) return;

  if constexpr (PCS::kSupportsBatchMode) {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
           lookup_prover.grand_product_polys_) {
        prover->BatchCommitAt(grand_product_poly.evals(), commit_idx++);
      }
    }
  } else {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
           lookup_prover.grand_product_polys_) {
        prover->CommitAndWriteToProof(grand_product_poly.evals());
      }
    }
  }
}

template <typename Poly, typename Evals>
template <typename Domain>
void Prover<Poly, Evals>::TransformEvalsToPoly(const Domain* domain) {
  for (Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair : permuted_pairs_) {
    permuted_pair.input().TransformEvalsToPoly(domain);
    permuted_pair.table().TransformEvalsToPoly(domain);
  }
  for (BlindedPolynomial<Poly, Evals>& grand_product_poly :
       grand_product_polys_) {
    grand_product_poly.TransformEvalsToPoly(domain);
  }
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::Evaluate(ProverBase<PCS>* prover,
                                   const OpeningPointSet<F>& point_set) const {
  size_t size = grand_product_polys_.size();
  CHECK_EQ(size, permuted_pairs_.size());

#define EVALUATE(polynomial, point) \
  prover->EvaluateAndWriteToProof(polynomial.poly(), point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/lookup/prover.rs#L309-L337.
  for (size_t i = 0; i < size; ++i) {
    const BlindedPolynomial<Poly, Evals>& grand_product_poly =
        grand_product_polys_[i];
    const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair =
        permuted_pairs_[i];

    // Zₗ,ᵢ(x)
    EVALUATE(grand_product_poly, x);
    // Zₗ,ᵢ(ω * x)
    EVALUATE(grand_product_poly, x_next);
    // A'ᵢ(x)
    EVALUATE(permuted_pair.input(), x);
    // A'ᵢ(ω⁻¹ * x)
    EVALUATE(permuted_pair.input(), x_prev);
    // S'ᵢ(x)
    EVALUATE(permuted_pair.table(), x);
  }
#undef EVALUATE
}

template <typename Poly, typename Evals>
void Prover<Poly, Evals>::Open(
    const OpeningPointSet<F>& point_set,
    std::vector<crypto::PolynomialOpening<Poly>>& openings) const {
  size_t size = grand_product_polys_.size();
  CHECK_EQ(size, permuted_pairs_.size());

#define OPENING(polynomial, point)                            \
  base::Ref<const Poly>(&polynomial.poly()), point_set.point, \
      polynomial.poly().Evaluate(point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/lookup/prover.rs#L340-L381.
  for (size_t i = 0; i < size; ++i) {
    const BlindedPolynomial<Poly, Evals>& grand_product_poly =
        grand_product_polys_[i];
    const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair =
        permuted_pairs_[i];

    // Zₗ,ᵢ(x)
    openings.emplace_back(OPENING(grand_product_poly, x));
    // A'ᵢ(x)
    openings.emplace_back(OPENING(permuted_pair.input(), x));
    // S'ᵢ(x)
    openings.emplace_back(OPENING(permuted_pair.table(), x));
    // A'ᵢ(ω⁻¹ * x)
    openings.emplace_back(OPENING(permuted_pair.input(), x_prev));
    // Zₗ,ᵢ(ω * x)
    openings.emplace_back(OPENING(grand_product_poly, x_next));
  }
#undef OPENING
}

// static
template <typename Poly, typename Evals>
std::function<typename Poly::Field(RowIndex)>
Prover<Poly, Evals>::CreateNumeratorCallback(const Pair<Evals>& compressed_pair,
                                             const F& beta, const F& gamma) {
  // (A_compressedᵢ(x) + β) * (S_compressedᵢ(x) + γ)
  return [&compressed_pair, &beta, &gamma](RowIndex row_index) {
    return (compressed_pair.input()[row_index] + beta) *
           (compressed_pair.table()[row_index] + gamma);
  };
}

// static
template <typename Poly, typename Evals>
std::function<typename Poly::Field(RowIndex)>
Prover<Poly, Evals>::CreateDenominatorCallback(
    const Pair<BlindedPolynomial<Poly, Evals>>& permuted_pair, const F& beta,
    const F& gamma) {
  return [&permuted_pair, &beta, &gamma](RowIndex row_index) {
    // (A'ᵢ(x) + β) * (S'ᵢ(x) + γ)
    return (permuted_pair.input().evals()[row_index] + beta) *
           (permuted_pair.table().evals()[row_index] + gamma);
  };
}

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_PROVER_IMPL_H_
