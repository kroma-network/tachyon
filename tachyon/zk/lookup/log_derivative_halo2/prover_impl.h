// Copyright (c) 2022-2024 Scroll
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.scroll and the LICENCE-APACHE.scroll
// file.

#ifndef TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_PROVER_IMPL_H_
#define TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_PROVER_IMPL_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/lookup/log_derivative_halo2/prover.h"
#include "tachyon/zk/plonk/expressions/compress_expression.h"

namespace tachyon::zk::lookup::log_derivative_halo2 {

// static
template <typename Poly, typename Evals>
template <typename Domain>
std::vector<Evals> Prover<Poly, Evals>::CompressInputs(
    const Domain* domain, const Argument<F>& argument, const F& theta,
    const plonk::ProvingEvaluator<Evals>& evaluator_tpl) {
  // f_compressedᵢ(X) = θᵐ⁻¹f₀(X) + θᵐ⁻²f₁(X) + ... + θfₘ₋₂(X) + fₘ₋₁(X)
  return base::Map(argument.inputs_expressions(),
                   [&domain, &theta, &evaluator_tpl](
                       const std::vector<std::unique_ptr<Expression<F>>>&
                           input_expressions) {
                     return plonk::CompressExpressions(
                         domain, input_expressions, theta, evaluator_tpl);
                   });
}

// static
template <typename Poly, typename Evals>
template <typename Domain>
Evals Prover<Poly, Evals>::CompressTable(
    const Domain* domain, const Argument<F>& argument, const F& theta,
    const plonk::ProvingEvaluator<Evals>& evaluator_tpl) {
  // t_compressedᵢ(X) = θᵐ⁻¹t₀(X) + θᵐ⁻²t₁(X) + ... + θtₘ₋₂(X) + tₘ₋₁(X)
  return plonk::CompressExpressions(domain, argument.table_expressions(), theta,
                                    evaluator_tpl);
}

template <typename Poly, typename Evals>
template <typename Domain>
void Prover<Poly, Evals>::CompressPairs(
    const Domain* domain, const std::vector<Argument<F>>& arguments,
    const F& theta, const plonk::ProvingEvaluator<Evals>& evaluator_tpl) {
  compressed_inputs_vec_.reserve(arguments.size());
  compressed_tables_.reserve(arguments.size());
  for (const Argument<F>& argument : arguments) {
    compressed_inputs_vec_.push_back(
        CompressInputs(domain, argument, theta, evaluator_tpl));
    compressed_tables_.push_back(
        CompressTable(domain, argument, theta, evaluator_tpl));
  }
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

template <typename BigInt>
struct LessThan {
  bool operator()(const TableEvalWithIndex<BigInt>& a, const BigInt& b) const {
    return a.eval < b;
  }

  bool operator()(const BigInt& a, const TableEvalWithIndex<BigInt>& b) const {
    return a < b.eval;
  }
};

// static
template <typename Poly, typename Evals>
template <typename PCS>
BlindedPolynomial<Poly, Evals> Prover<Poly, Evals>::ComputeMPoly(
    ProverBase<PCS>* prover, const std::vector<Evals>& compressed_inputs,
    const Evals& compressed_table, ComputeMPolysTempStorage<BigInt>& storage) {
  RowIndex usable_rows = prover->GetUsableRows();

  OPENMP_PARALLEL_FOR(RowIndex i = 0; i < usable_rows; ++i) {
    storage.sorted_table_with_indices[i] = {i, compressed_table[i].ToBigInt()};
  }

  // TODO(chokobole): Use https://github.com/timsort/cpp-TimSort or
  // https://github.com/sebawild/powersort for better performance.
  std::stable_sort(storage.sorted_table_with_indices.begin(),
                   storage.sorted_table_with_indices.end());

  std::vector<F> m_values(prover->pcs().N());
  OMP_PARALLEL {
    OMP_NESTED_FOR(size_t i = 0; i < compressed_inputs.size(); ++i) {
      for (RowIndex j = 0; j < usable_rows; ++j) {
        BigInt input = compressed_inputs[i][j].ToBigInt();
        auto it = base::BinarySearchByKey(
            storage.sorted_table_with_indices.begin(),
            storage.sorted_table_with_indices.end(), input, LessThan<BigInt>{});
        if (it != storage.sorted_table_with_indices.end()) {
          storage.m_values_atomic[it->index].fetch_add(
              1, std::memory_order_relaxed);
        }
      }
    }

    // Convert atomic |m_values| to |Evals|.
    OMP_FOR(RowIndex i = 0; i < usable_rows; ++i) {
      m_values[i] =
          F(storage.m_values_atomic[i].exchange(0, std::memory_order_relaxed));
    }
  }

  BlindedPolynomial<Poly, Evals> m_poly(Evals(std::move(m_values)),
                                        prover->blinder().Generate());
  return m_poly;
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::ComputeMPolys(
    ProverBase<PCS>* prover, ComputeMPolysTempStorage<BigInt>& storage) {
  CHECK_EQ(compressed_inputs_vec_.size(), compressed_tables_.size());
  m_polys_ =
      base::Map(compressed_inputs_vec_,
                [this, prover, &storage](
                    size_t i, const std::vector<Evals>& compressed_inputs) {
                  return ComputeMPoly(prover, compressed_inputs,
                                      compressed_tables_[i], storage);
                });
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::BatchCommitMPolys(
    const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
    size_t& commit_idx) {
  if (lookup_provers.empty()) return;

  if constexpr (PCS::kSupportsBatchMode) {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const BlindedPolynomial<Poly, Evals>& m_poly :
           lookup_prover.m_polys()) {
        prover->BatchCommitAt(m_poly.evals(), commit_idx++);
      }
    }
  } else {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const BlindedPolynomial<Poly, Evals>& m_poly :
           lookup_prover.m_polys()) {
        prover->CommitAndWriteToProof(m_poly.evals());
      }
    }
  }
}

// static
template <typename Poly, typename Evals>
void Prover<Poly, Evals>::ComputeLogDerivatives(const Evals& evals,
                                                const F& beta,
                                                std::vector<F>& ret) {
  base::Parallelize(ret,
                    [&evals, &beta](absl::Span<F> chunk, size_t chunk_offset,
                                    size_t chunk_size) {
                      size_t start = chunk_offset * chunk_size;
                      for (size_t i = 0; i < chunk.size(); ++i) {
                        chunk[i] = beta + evals.evaluations()[start + i];
                      }
                      CHECK(F::BatchInverseInPlaceSerial(chunk));
                    });
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
BlindedPolynomial<Poly, Evals> Prover<Poly, Evals>::CreateGrandSumPoly(
    ProverBase<PCS>* prover, const Evals& m_values,
    const std::vector<Evals>& compressed_inputs, const Evals& compressed_table,
    const F& beta, GrandSumPolysTempStorage<F>& storage) {
  size_t n = prover->pcs().N();
  RowIndex usable_rows = prover->GetUsableRows();

  // Σ 1/(φᵢ(X))
  // NOTE(chokobole): To save memory, |input_log_derivatives| uses the storage
  // space of |storage.table_log_derivatives| since
  // |storage.table_log_derivatives| is currently empty and will be assigned
  // with relevant values after |input_log_derivatives| is finished being used.
  std::vector<F>& input_log_derivatives = storage.table_log_derivatives;
  for (size_t i = 0; i < compressed_inputs.size(); ++i) {
    ComputeLogDerivatives(compressed_inputs[i], beta, input_log_derivatives);

    if (i == 0) {
      OPENMP_PARALLEL_FOR(size_t j = 0; j < usable_rows; ++j) {
        storage.inputs_log_derivatives[j] = input_log_derivatives[j];
      }
    } else {
      OPENMP_PARALLEL_FOR(size_t j = 0; j < usable_rows; ++j) {
        storage.inputs_log_derivatives[j] += input_log_derivatives[j];
      }
    }
  }

  // 1 / τ(X)
  ComputeLogDerivatives(compressed_table, beta, storage.table_log_derivatives);

  std::vector<F> grand_sum(n);
  grand_sum[0] = F::Zero();

  // (Σ 1/φᵢ(X)) - m(X) / τ(X)
  // NOTE(chokobole): To save memory, |log_derivatives_diff| overwrites
  // |storage.inputs_log_derivatives| since the current values of
  // |storage.inputs_log_derivatives| are not needed anymore.
  std::vector<F>& log_derivatives_diff = storage.inputs_log_derivatives;
  OPENMP_PARALLEL_FOR(size_t i = 0; i < usable_rows; ++i) {
    log_derivatives_diff[i] -= m_values[i] * storage.table_log_derivatives[i];
    if (i != usable_rows - 1) {
      grand_sum[i + 1] = log_derivatives_diff[i];
    }
  }

// let L(X) = (Σ 1/φᵢ(X)) - m(X) / τ(X)
// ϕ(ω⁰) = 0
// ϕ(ω¹) = L(ω⁰)
// ...
// ϕ(ω^last) = L(ω⁰) + L(ω¹) + ... + L(ω^{usable_rows - 1})
#if defined(TACHYON_HAS_OPENMP)
  // NOTE(chokobole): To save memory, |segment_sum| overwrites
  // |storage.table_log_derivatives| since the current values of
  // |storage.table_log_derivatives| are not needed anymore.
  std::vector<F>& segment_sum = storage.table_log_derivatives;

  absl::Span<F> grand_sum_sub_span =
      absl::MakeSpan(grand_sum).subspan(0, usable_rows);
  base::Parallelize(
      grand_sum_sub_span,
      [&log_derivatives_diff, &segment_sum](
          absl::Span<F> chunk, size_t chunk_idx, size_t chunk_size) {
        size_t start = chunk_idx * chunk_size;
        for (size_t i = 1; i < chunk.size(); ++i) {
          chunk[i] = chunk[i - 1] + log_derivatives_diff[start + i - 1];
        }
        segment_sum[chunk_idx] = chunk.back();
      });

  for (size_t i = 1; i < segment_sum.size(); ++i) {
    segment_sum[i] += segment_sum[i - 1];
  }

  base::Parallelize(
      grand_sum_sub_span,
      [&segment_sum](absl::Span<F> chunk, size_t chunk_idx, size_t chunk_size) {
        if (chunk_idx == 0) return;
        const F& prefix_sum = segment_sum[chunk_idx - 1];
        for (F& v : chunk) {
          v += prefix_sum;
        }
      });
#else
  std::partial_sum(log_derivatives_diff.begin(), log_derivatives_diff.end() - 1,
                   grand_sum.begin() + 1);
#endif

  Evals grand_sum_poly(std::move(grand_sum));

  CHECK(prover->blinder().Blind(grand_sum_poly));

  return BlindedPolynomial<Poly, Evals>(std::move(grand_sum_poly),
                                        prover->blinder().Generate());
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::CreateGrandSumPolys(
    ProverBase<PCS>* prover, const F& beta,
    GrandSumPolysTempStorage<F>& storage) {
  CHECK_EQ(compressed_inputs_vec_.size(), compressed_tables_.size());

  grand_sum_polys_ =
      base::Map(compressed_inputs_vec_,
                [this, &prover, &beta, &storage](
                    size_t i, const std::vector<Evals>& compressed_inputs) {
                  return CreateGrandSumPoly(
                      prover, m_polys_[i].evals(), compressed_inputs,
                      compressed_tables_[i], beta, storage);
                });
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::BatchCommitGrandSumPolys(
    const std::vector<Prover>& lookup_provers, ProverBase<PCS>* prover,
    size_t& commit_idx) {
  if (lookup_provers.empty()) return;

  if constexpr (PCS::kSupportsBatchMode) {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_sum_poly :
           lookup_prover.grand_sum_polys_) {
        prover->BatchCommitAt(grand_sum_poly.evals(), commit_idx++);
      }
    }
  } else {
    for (const Prover& lookup_prover : lookup_provers) {
      for (const BlindedPolynomial<Poly, Evals>& grand_sum_poly :
           lookup_prover.grand_sum_polys_) {
        prover->CommitAndWriteToProof(grand_sum_poly.evals());
      }
    }
  }
}

template <typename Poly, typename Evals>
template <typename Domain>
void Prover<Poly, Evals>::TransformEvalsToPoly(const Domain* domain) {
  for (BlindedPolynomial<Poly, Evals>& m_poly : m_polys_) {
    m_poly.TransformEvalsToPoly(domain);
  }
  for (BlindedPolynomial<Poly, Evals>& grand_sum_poly : grand_sum_polys_) {
    grand_sum_poly.TransformEvalsToPoly(domain);
  }
}

template <typename Poly, typename Evals>
template <typename PCS>
void Prover<Poly, Evals>::Evaluate(
    ProverBase<PCS>* prover, const halo2::OpeningPointSet<F>& point_set) const {
  size_t size = grand_sum_polys_.size();
  CHECK_EQ(size, m_polys_.size());

#define EVALUATE(polynomial, point) \
  prover->EvaluateAndWriteToProof(polynomial.poly(), point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/scroll-tech/halo2/blob/1070391/halo2_proofs/src/plonk/mv_lookup/prover.rs#L428-L453
  for (size_t i = 0; i < size; ++i) {
    const BlindedPolynomial<Poly, Evals>& grand_sum_poly = grand_sum_polys_[i];
    const BlindedPolynomial<Poly, Evals>& m_poly = m_polys_[i];

    // ϕᵢ(x)
    EVALUATE(grand_sum_poly, x);
    // ϕᵢ(ω * x)
    EVALUATE(grand_sum_poly, x_next);
    // mᵢ(x)
    EVALUATE(m_poly, x);
  }
#undef EVALUATE
}

template <typename Poly, typename Evals>
void Prover<Poly, Evals>::Open(
    const halo2::OpeningPointSet<F>& point_set,
    std::vector<crypto::PolynomialOpening<Poly>>& openings) const {
  size_t size = grand_sum_polys_.size();
  CHECK_EQ(size, m_polys_.size());

#define OPENING(polynomial, point)                            \
  base::Ref<const Poly>(&polynomial.poly()), point_set.point, \
      polynomial.poly().Evaluate(point_set.point)

  // THE ORDER IS IMPORTANT!! DO NOT CHANGE!
  // See
  // https://github.com/scroll-tech/halo2/blob/1070391/halo2_proofs/src/plonk/mv_lookup/prover.rs#L455-L480
  for (size_t i = 0; i < size; ++i) {
    const BlindedPolynomial<Poly, Evals>& grand_sum_poly = grand_sum_polys_[i];
    const BlindedPolynomial<Poly, Evals>& m_poly = m_polys_[i];
    // ϕᵢ(x)
    openings.emplace_back(OPENING(grand_sum_poly, x));
    // ϕᵢ(ω * x)
    openings.emplace_back(OPENING(grand_sum_poly, x_next));
    // mᵢ(x)
    openings.emplace_back(OPENING(m_poly, x));
  }
#undef OPENING
}

}  // namespace tachyon::zk::lookup::log_derivative_halo2

#endif  // TACHYON_ZK_LOOKUP_LOG_DERIVATIVE_HALO2_PROVER_IMPL_H_
