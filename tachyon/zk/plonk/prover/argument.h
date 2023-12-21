#ifndef TACHYON_ZK_PLONK_PROVER_ARGUMENT_H_
#define TACHYON_ZK_PLONK_PROVER_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/circuit/ref_table.h"

namespace tachyon::zk {

// Data class including all arguments for creating proof.
template <typename PCSTy>
class Argument {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;
  using Evals = typename PCSTy::Evals;
  using Domain = typename PCSTy::Domain;

  Argument() = default;
  static Argument Create(ProverBase<PCSTy>* prover, size_t num_circuits,
                         const std::vector<Evals>* fixed_columns,
                         const std::vector<Poly>* fixed_polys,
                         std::vector<std::vector<Evals>>&& advice_columns_vec,
                         std::vector<std::vector<F>>&& advice_blinds_vec,
                         std::vector<std::vector<Evals>>&& instance_columns_vec,
                         std::vector<F>&& challenges) {
    std::vector<std::vector<Poly>> instance_polys_vec =
        GenerateInstancePolys(prover, instance_columns_vec);

    return Argument(num_circuits, fixed_columns, fixed_polys,
                    std::move(advice_columns_vec), std::move(advice_blinds_vec),
                    std::move(instance_columns_vec),
                    std::move(instance_polys_vec), std::move(challenges));
  }

  bool advice_transformed() const { return advice_transformed_; }

  // Generate a vector of advice coefficient-formed polynomials with a vector
  // of advice evaluation-formed columns. (a.k.a. Batch IFFT)
  // And for memory optimization, every evaluations of advice will be released
  // as soon as transforming it to coefficient form.
  void TransformAdvice(const Domain* domain) {
    CHECK(!advice_transformed_);
    advice_polys_vec_ = base::Map(
        advice_columns_vec_, [domain](std::vector<Evals>& advice_columns) {
          return base::Map(advice_columns, [domain](Evals& advice_column) {
            Poly poly = domain->IFFT(advice_column);
            // Release advice evals for memory optimization.
            advice_column = Evals::Zero();
            return poly;
          });
        });
    // Deallocate evaluations for memory optimization.
    advice_columns_vec_.clear();
    advice_transformed_ = true;
  }

  // Return tables including every type of polynomials in evaluation form.
  std::vector<RefTable<Evals>> ExportColumnTables() const {
    CHECK(!advice_transformed_);
    absl::Span<const Evals> fixed_columns =
        absl::MakeConstSpan(*fixed_columns_);

    base::Range<size_t> circuit_range =
        base::Range<size_t>::Until(num_circuits_);
    return base::Map(circuit_range.begin(), circuit_range.end(),
                     [fixed_columns, this](size_t i) {
                       absl::Span<const Evals> advice_columns =
                           absl::MakeConstSpan(advice_columns_vec_[i]);
                       absl::Span<const Evals> instance_columns =
                           absl::MakeConstSpan(instance_columns_vec_[i]);
                       return RefTable<Evals>(fixed_columns, advice_columns,
                                              instance_columns);
                     });
  }

  // Return a table including every type of polynomials in coefficient form.
  std::vector<RefTable<Poly>> ExportPolyTables() const {
    CHECK(advice_transformed_);
    absl::Span<const Poly> fixed_polys = absl::MakeConstSpan(*fixed_polys_);

    base::Range<size_t> circuit_range =
        base::Range<size_t>::Until(num_circuits_);
    return base::Map(circuit_range.begin(), circuit_range.end(),
                     [fixed_polys, this](size_t i) {
                       absl::Span<const Poly> advice_polys =
                           absl::MakeConstSpan(advice_polys_vec_[i]);
                       absl::Span<const Poly> instance_polys =
                           absl::MakeConstSpan(instance_polys_vec_[i]);
                       return RefTable<Poly>(fixed_polys, advice_polys,
                                             instance_polys);
                     });
  }

  const std::vector<F>& GetAdviceBlinds(size_t circuit_idx) const {
    CHECK_LT(circuit_idx, num_circuits_);
    return advice_blinds_vec_[circuit_idx];
  }

  const std::vector<F>& challenges() const { return challenges_; }

 private:
  Argument(size_t num_circuits, const std::vector<Evals>* fixed_columns,
           const std::vector<Poly>* fixed_polys,
           std::vector<std::vector<Evals>>&& advice_columns_vec,
           std::vector<std::vector<F>>&& advice_blinds_vec,
           std::vector<std::vector<Evals>>&& instance_columns_vec,
           std::vector<std::vector<Poly>>&& instance_polys_vec,
           std::vector<F>&& challenges)
      : num_circuits_(num_circuits),
        fixed_columns_(fixed_columns),
        fixed_polys_(fixed_polys),
        advice_columns_vec_(std::move(advice_columns_vec)),
        advice_blinds_vec_(std::move(advice_blinds_vec)),
        instance_columns_vec_(std::move(instance_columns_vec)),
        instance_polys_vec_(std::move(instance_polys_vec)),
        challenges_(std::move(challenges)) {
    CHECK_EQ(num_circuits_, advice_columns_vec_.size());
    CHECK_EQ(num_circuits_, advice_blinds_vec_.size());
    CHECK_EQ(num_circuits_, instance_columns_vec_.size());
    CHECK_EQ(num_circuits_, instance_polys_vec_.size());
  }

  // Generate a vector of instance coefficient-formed polynomials with a vector
  // of instance evaluation-formed columns. (a.k.a. Batch IFFT)
  static std::vector<std::vector<Poly>> GenerateInstancePolys(
      ProverBase<PCSTy>* prover,
      std::vector<std::vector<Evals>> instance_columns_vec) {
    return base::Map(instance_columns_vec,
                     [prover](const std::vector<Evals>& instance_columns) {
                       return base::Map(
                           instance_columns,
                           [prover](const Evals& instance_column) {
                             if constexpr (PCSTy::kQueryInstance) {
                               CHECK(prover->CommitEvals(instance_column));
                             } else {
                               for (size_t i = 0; i < prover->pcs().N(); ++i) {
                                 CHECK(prover->GetWriter()->WriteToTranscript(
                                     *instance_column[i]));
                               }
                             }
                             return prover->domain()->IFFT(instance_column);
                           });
                     });
  }

  size_t num_circuits_ = 0;
  // not owned
  const std::vector<Evals>* fixed_columns_ = nullptr;
  // not owned
  const std::vector<Poly>* fixed_polys_ = nullptr;

  std::vector<std::vector<Evals>> advice_columns_vec_;
  std::vector<std::vector<F>> advice_blinds_vec_;
  // Note(dongchangYoo): to optimize memory usage, release every advice
  // evaluations after generating an advice polynomial. That is, when
  // |advice_transformed_| is set to true, |advice_values_by_circuits| is
  // released, and only |advice_polys_by_circuits| becomes available for use.
  std::vector<std::vector<Poly>> advice_polys_vec_;
  bool advice_transformed_ = false;

  std::vector<std::vector<Evals>> instance_columns_vec_;
  std::vector<std::vector<Poly>> instance_polys_vec_;

  std::vector<F> challenges_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PROVER_ARGUMENT_H_
