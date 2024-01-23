#ifndef TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_
#define TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/circuit/ref_table.h"

namespace tachyon::zk::halo2 {

// Data class including all arguments for creating proof.
template <typename PCS>
class Argument {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Domain = typename PCS::Domain;

  Argument() = default;
  static Argument Create(
      ProverBase<PCS>* prover, size_t num_circuits,
      const std::vector<Evals>* fixed_columns,
      const std::vector<Poly>* fixed_polys,
      std::vector<std::vector<Evals>>&& advice_columns_vec,
      std::vector<std::vector<F>>&& advice_blinds_vec,
      std::vector<F>&& challenges,
      std::vector<std::vector<Evals>>&& instance_columns_vec) {
    std::vector<std::vector<Poly>> instance_polys_vec =
        GenerateInstancePolys(prover, instance_columns_vec);

    return Argument(num_circuits, fixed_columns, fixed_polys,
                    std::move(advice_columns_vec), std::move(advice_blinds_vec),
                    std::move(challenges), std::move(instance_columns_vec),
                    std::move(instance_polys_vec));
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

    return base::CreateVector(num_circuits_, [fixed_columns, this](size_t i) {
      absl::Span<const Evals> advice_columns =
          absl::MakeConstSpan(advice_columns_vec_[i]);
      absl::Span<const Evals> instance_columns =
          absl::MakeConstSpan(instance_columns_vec_[i]);
      return RefTable<Evals>(fixed_columns, advice_columns, instance_columns);
    });
  }

  // Return a table including every type of polynomials in coefficient form.
  std::vector<RefTable<Poly>> ExportPolyTables() const {
    CHECK(advice_transformed_);
    absl::Span<const Poly> fixed_polys = absl::MakeConstSpan(*fixed_polys_);
    return base::CreateVector(num_circuits_, [fixed_polys, this](size_t i) {
      absl::Span<const Poly> advice_polys =
          absl::MakeConstSpan(advice_polys_vec_[i]);
      absl::Span<const Poly> instance_polys =
          absl::MakeConstSpan(instance_polys_vec_[i]);
      return RefTable<Poly>(fixed_polys, advice_polys, instance_polys);
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
           std::vector<F>&& challenges,
           std::vector<std::vector<Evals>>&& instance_columns_vec,
           std::vector<std::vector<Poly>>&& instance_polys_vec)
      : num_circuits_(num_circuits),
        fixed_columns_(fixed_columns),
        fixed_polys_(fixed_polys),
        advice_columns_vec_(std::move(advice_columns_vec)),
        advice_blinds_vec_(std::move(advice_blinds_vec)),
        challenges_(std::move(challenges)),
        instance_columns_vec_(std::move(instance_columns_vec)),
        instance_polys_vec_(std::move(instance_polys_vec)) {
    CHECK_EQ(num_circuits_, advice_columns_vec_.size());
    CHECK_EQ(num_circuits_, advice_blinds_vec_.size());
    CHECK_EQ(num_circuits_, instance_columns_vec_.size());
    CHECK_EQ(num_circuits_, instance_polys_vec_.size());
  }

  // Generate a vector of instance coefficient-formed polynomials with a vector
  // of instance evaluation-formed columns. (a.k.a. Batch IFFT)
  static std::vector<std::vector<Poly>> GenerateInstancePolys(
      ProverBase<PCS>* prover,
      const std::vector<std::vector<std::vector<F>>>& instance_columns_vec) {
    size_t num_circuit = instance_columns_vec.size();
    CHECK_GT(num_circuit, size_t{0});
    size_t num_instance_columns = instance_columns_vec[0].size();
    if constexpr (PCS::kSupportsBatchMode && PCS::kQueryInstance) {
      size_t num_commitment = num_circuit * num_instance_columns;
      prover->pcs().SetBatchMode(num_commitment);
    }

    std::vector<std::vector<Poly>> instance_polys_vec;
    instance_polys_vec.reserve(num_circuit);
    for (size_t i = 0; i < num_circuit; ++i) {
      const std::vector<std::vector<F>>& instance_columns =
          instance_columns_vec[i];
      std::vector<Poly> instance_polys;
      instance_polys.reserve(num_instance_columns);
      for (size_t j = 0; j < num_instance_columns; ++j) {
        const std::vector<F>& instances = instance_columns[j];
        if constexpr (PCS::kQueryInstance) {
          // FIXME(dongchangYoo): This causes copy internally and I want to
          // avoid this.
          Evals instance_column(instances);
          if constexpr (PCS::kSupportsBatchMode) {
            prover->BatchCommitAt(instance_column,
                                  i * num_instance_columns + j);
            instance_polys.push_back(prover->domain()->IFFT(instance_column));
          } else {
            CHECK(prover->GetWriter()->WriteToTranscript(instance_column));
            instance_polys.push_back(prover->domain()->IFFT(instance_column));
          }
        } else {
          for (const F& instance : instances) {
            CHECK(prover->GetWriter()->WriteToTranscript(instance));
          }
          instance_polys.push_back(prover->domain()->IFFT(Evals(instances)));
        }
      }
      instance_polys_vec.push_back(std::move(instance_polys));
    }
    if constexpr (PCS::kSupportsBatchMode && PCS::kQueryInstance) {
      prover->RetrieveAndWriteBatchCommitmentsToTranscript();
    }
    return instance_polys_vec;
  }

  size_t num_circuits_ = 0;
  // not owned
  const std::vector<Evals>* fixed_columns_ = nullptr;
  // not owned
  const std::vector<Poly>* fixed_polys_ = nullptr;

  // NOTE(dongchangYoo): to optimize memory usage, release every advice
  // evaluations after generating an advice polynomial. That is, when
  // |advice_transformed_| is set to true, |advice_columns_vec_| is
  // released, and only |advice_polys_vec_| becomes available for use.
  bool advice_transformed_ = false;
  std::vector<std::vector<Evals>> advice_columns_vec_;
  std::vector<std::vector<Poly>> advice_polys_vec_;
  std::vector<std::vector<F>> advice_blinds_vec_;
  std::vector<F> challenges_;

  std::vector<std::vector<Evals>> instance_columns_vec_;
  std::vector<std::vector<Poly>> instance_polys_vec_;
};

}  // namespace tachyon::zk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_ARGUMENT_H_
