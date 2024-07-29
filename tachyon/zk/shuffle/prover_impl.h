// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_PROVER_IMPL_H_
#define TACHYON_ZK_SHUFFLE_PROVER_IMPL_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/plonk/expressions/compress_expression.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"
#include "tachyon/zk/shuffle/prover.h"

namespace tachyon::zk::shuffle {

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
  Evals compressed_shuffle = plonk::CompressExpressions(
      domain, argument.shuffle_expressions(), theta, evaluator_tpl);

  return {std::move(compressed_input), std::move(compressed_shuffle)};
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
    std::vector<Prover>& shuffle_provers, const Domain* domain,
    const std::vector<Argument<F>>& arguments, const F& theta,
    const std::vector<plonk::MultiPhaseRefTable<Evals>>& tables) {
  CHECK_EQ(shuffle_provers.size(), tables.size());
  // NOTE(chokobole): It's safe to downcast because domain is already checked.
  int32_t n = static_cast<int32_t>(domain->size());
  for (size_t i = 0; i < shuffle_provers.size(); ++i) {
    plonk::ProvingEvaluator<Evals> proving_evaluator(0, n, 1, tables[i]);
    shuffle_provers[i].CompressPairs(domain, arguments, theta,
                                     proving_evaluator);
  }
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
BlindedPolynomial<Poly, Evals> Prover<Poly, Evals>::CreateGrandProductPoly(
    ProverBase<PCS>* prover, const Pair<Evals>& compressed_pair,
    const F& gamma) {
  return {plonk::GrandProductArgument::CreatePolySerial(
              prover, CreateNumeratorCallback(compressed_pair.input(), gamma),
              CreateDenominatorCallback(compressed_pair.shuffle(), gamma)),
          prover->blinder().Generate()};
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::CreateGrandProductPolys(ProverBase<PCS>* prover,
                                                  const F& gamma) {
  // Zₛ,ᵢ(X)
  grand_product_polys_.resize(compressed_pairs_.size());

  // NOTE(dongchangYoo): do not change this code to parallelized logic.
  grand_product_polys_ =
      base::Map(compressed_pairs_,
                [prover, &gamma](size_t i, const Pair<Evals>& compressed_pair) {
                  return CreateGrandProductPoly(prover, compressed_pair, gamma);
                });
  compressed_pairs_.clear();
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::BatchCommitGrandProductPolys(
    const std::vector<Prover>& shuffle_provers, ProverBase<PCS>* prover,
    size_t& commit_idx) {
  if (shuffle_provers.empty()) return;

  if constexpr (PCS::kSupportsBatchMode) {
    for (const Prover& shuffle_prover : shuffle_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
           shuffle_prover.grand_product_polys_) {
        prover->BatchCommitAt(grand_product_poly.evals(), commit_idx++);
      }
    }
  } else {
    for (const Prover& shuffle_prover : shuffle_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_product_poly :
           shuffle_prover.grand_product_polys_) {
        prover->CommitAndWriteToProof(grand_product_poly.evals());
      }
    }
  }
}

template <typename Poly, typename Evals>
template <typename Domain>
void Prover<Poly, Evals>::TransformEvalsToPoly(const Domain* domain) {
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

#define EVALUATE(polynomial, point) \
  prover->EvaluateAndWriteToProof(polynomial.poly(), point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/scroll-tech/halo2/blob/e5ddf67/halo2_proofs/src/plonk/shuffle/prover.rs#L204-L225.
  for (size_t i = 0; i < size; ++i) {
    const BlindedPolynomial<Poly, Evals>& grand_product_poly =
        grand_product_polys_[i];

    // Zₛ,ᵢ(x)
    EVALUATE(grand_product_poly, x);
    // Zₛ,ᵢ(ω * x)
    EVALUATE(grand_product_poly, x_next);
  }
#undef EVALUATE
}

template <typename Poly, typename Evals>
void Prover<Poly, Evals>::Open(
    const OpeningPointSet<F>& point_set,
    std::vector<crypto::PolynomialOpening<Poly>>& openings) const {
  size_t size = grand_product_polys_.size();

#define OPENING(polynomial, point)                            \
  base::Ref<const Poly>(&polynomial.poly()), point_set.point, \
      polynomial.poly().Evaluate(point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/scroll-tech/halo2/blob/e5ddf67/halo2_proofs/src/plonk/shuffle/prover.rs#L229-L249
  for (size_t i = 0; i < size; ++i) {
    const BlindedPolynomial<Poly, Evals>& grand_product_poly =
        grand_product_polys_[i];

    // Zₛ,ᵢ(x)
    openings.emplace_back(OPENING(grand_product_poly, x));
    // Zₛ,ᵢ(ω * x)
    openings.emplace_back(OPENING(grand_product_poly, x_next));
  }
#undef OPENING
}

// static
template <typename Poly, typename Evals>
std::function<typename Poly::Field(RowIndex)>
Prover<Poly, Evals>::CreateNumeratorCallback(const Evals& input,
                                             const F& gamma) {
  // A_compressedᵢ(x) + γ
  return
      [&input, &gamma](RowIndex row_index) { return input[row_index] + gamma; };
}

// static
template <typename Poly, typename Evals>
std::function<typename Poly::Field(RowIndex)>
Prover<Poly, Evals>::CreateDenominatorCallback(const Evals& shuffle,
                                               const F& gamma) {
  return [&shuffle, &gamma](RowIndex row_index) {
    // S_compressedᵢ(x) + γ
    return shuffle[row_index] + gamma;
  };
}

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_PROVER_IMPL_H_
