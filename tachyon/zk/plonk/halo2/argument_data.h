#ifndef TACHYON_ZK_PLONK_HALO2_ARGUMENT_DATA_H_
#define TACHYON_ZK_PLONK_HALO2_ARGUMENT_DATA_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/base/ref_table.h"
#include "tachyon/zk/plonk/halo2/argument_data.h"
#include "tachyon/zk/plonk/halo2/synthesizer.h"

namespace tachyon {
namespace zk::plonk::halo2 {

template <typename Poly, typename Evals>
class ArgumentData {
 public:
  using F = typename Poly::Field;

  ArgumentData() = default;

  // NOTE(chokobole): This constructor is used by the c api.
  explicit ArgumentData(size_t num_circuits) {
    advice_columns_vec_.resize(num_circuits);
    advice_blinds_vec_.resize(num_circuits);
    instance_columns_vec_.resize(num_circuits);
    instance_polys_vec_.resize(num_circuits);
  }

  ArgumentData(std::vector<std::vector<Evals>>&& advice_columns_vec,
               std::vector<std::vector<F>>&& advice_blinds_vec,
               std::vector<F>&& challenges,
               std::vector<std::vector<Evals>>&& instance_columns_vec,
               std::vector<std::vector<Poly>>&& instance_polys_vec)
      : advice_columns_vec_(std::move(advice_columns_vec)),
        advice_blinds_vec_(std::move(advice_blinds_vec)),
        challenges_(std::move(challenges)),
        instance_columns_vec_(std::move(instance_columns_vec)),
        instance_polys_vec_(std::move(instance_polys_vec)) {
    size_t num_circuits = advice_columns_vec_.size();
    CHECK_EQ(num_circuits, advice_blinds_vec_.size());
    CHECK_EQ(num_circuits, instance_columns_vec_.size());
    CHECK_EQ(num_circuits, instance_polys_vec_.size());
  }

  template <typename PCS, typename Circuit>
  static ArgumentData Create(
      ProverBase<PCS>* prover, std::vector<Circuit>& circuits,
      const ConstraintSystem<F>& constraint_system,
      std::vector<std::vector<Evals>>&& instance_columns_vec) {
    // Generate instance polynomial and write it to transcript.
    std::vector<std::vector<Poly>> instance_polys_vec =
        GenerateInstancePolys(prover, instance_columns_vec);

    // Append leading zeros to each column of |instance_columns_vec|.
    size_t num_circuits = circuits.size();
    size_t n = prover->pcs().N();
    for (size_t i = 0; i < num_circuits; ++i) {
      for (Evals& instance_column : instance_columns_vec[i]) {
        instance_column.evaluations().resize(n);
      }
    }

    // Generate advice poly by synthesizing circuit and write it to transcript.
    Synthesizer<Evals> synthesizer(num_circuits, &constraint_system);
    synthesizer.GenerateAdviceColumns(prover, circuits, instance_columns_vec);

    return ArgumentData(std::move(synthesizer).TakeAdviceColumnsVec(),
                        std::move(synthesizer).TakeAdviceBlindsVec(),
                        std::move(synthesizer).TakeChallenges(),
                        std::move(instance_columns_vec),
                        std::move(instance_polys_vec));
  }

  // NOTE(chokobole): These getters are used by the c api.
  std::vector<std::vector<Evals>>& advice_columns_vec() {
    return advice_columns_vec_;
  }
  std::vector<std::vector<F>>& advice_blinds_vec() {
    return advice_blinds_vec_;
  }
  std::vector<F>& challenges() { return challenges_; }
  std::vector<std::vector<Evals>>& instance_columns_vec() {
    return instance_columns_vec_;
  }
  std::vector<std::vector<Poly>>& instance_polys_vec() {
    return instance_polys_vec_;
  }

  size_t GetNumCircuits() const { return instance_columns_vec_.size(); }

  absl::Span<const F> GetAdviceBlinds(size_t circuit_idx) const {
    CHECK_LT(circuit_idx, GetNumCircuits());
    return absl::MakeConstSpan(advice_blinds_vec_[circuit_idx]);
  }

  absl::Span<const F> GetChallenges() const {
    return absl::MakeConstSpan(challenges_);
  }

  // Generate a vector of advice coefficient-formed polynomials with a vector
  // of advice evaluation-formed columns. (a.k.a. Batch IFFT)
  // And for memory optimization, every evaluations of advice will be released
  // as soon as transforming it to coefficient form.
  template <typename Domain>
  void TransformAdvice(const Domain* domain) {
    VLOG(2) << "Transform advice columns to polys";
    CHECK(!advice_transformed_);
    advice_polys_vec_ = base::Map(
        advice_columns_vec_, [domain](std::vector<Evals>& advice_columns) {
          return base::Map(advice_columns, [domain](Evals& advice_column) {
            Poly poly = domain->IFFT(advice_column);
            // NOTE(dongchangYoo): Release advice evals for memory optimization.
            advice_column = Evals::Zero();
            return poly;
          });
        });
    // NOTE(dongchangYoo): Deallocate evaluations for memory optimization.
    advice_columns_vec_.clear();
    advice_transformed_ = true;
  }

  // Return tables including every type of columns in evaluation form.
  std::vector<RefTable<Evals>> ExportColumnTables(
      absl::Span<const Evals> fixed_columns) const {
    CHECK(!advice_transformed_);
    return base::CreateVector(GetNumCircuits(), [fixed_columns,
                                                 this](size_t i) {
      absl::Span<const Evals> advice_columns =
          absl::MakeConstSpan(advice_columns_vec_[i]);
      absl::Span<const Evals> instance_columns =
          absl::MakeConstSpan(instance_columns_vec_[i]);
      return RefTable<Evals>(fixed_columns, advice_columns, instance_columns);
    });
  }

  // Return a table including every type of columns in coefficient form.
  std::vector<RefTable<Poly>> ExportPolyTables(
      absl::Span<const Poly> fixed_polys) const {
    CHECK(advice_transformed_);
    return base::CreateVector(GetNumCircuits(), [fixed_polys, this](size_t i) {
      absl::Span<const Poly> advice_polys =
          absl::MakeConstSpan(advice_polys_vec_[i]);
      absl::Span<const Poly> instance_polys =
          absl::MakeConstSpan(instance_polys_vec_[i]);
      return RefTable<Poly>(fixed_polys, advice_polys, instance_polys);
    });
  }

  bool operator==(const ArgumentData& other) const {
    return advice_transformed_ == other.advice_transformed_ &&
           advice_columns_vec_ == other.advice_columns_vec_ &&
           advice_polys_vec_ == other.advice_polys_vec_ &&
           advice_blinds_vec_ == other.advice_blinds_vec_ &&
           challenges_ == other.challenges_ &&
           instance_columns_vec_ == other.instance_columns_vec_ &&
           instance_polys_vec_ == other.instance_polys_vec_;
  }
  bool operator!=(const ArgumentData& other) const {
    return !operator==(other);
  }

 private:
  friend class base::Copyable<ArgumentData>;

  // Generate a vector of instance coefficient-formed polynomials with a vector
  // of instance evaluation-formed columns. (a.k.a. Batch IFFT)
  template <typename PCS>
  static std::vector<std::vector<Poly>> GenerateInstancePolys(
      ProverBase<PCS>* prover,
      const std::vector<std::vector<Evals>>& instance_columns_vec) {
    VLOG(2) << "Generating instance polys";
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
      const std::vector<Evals>& instance_columns = instance_columns_vec[i];
      std::vector<Poly> instance_polys;
      instance_polys.reserve(num_instance_columns);
      for (size_t j = 0; j < num_instance_columns; ++j) {
        const Evals& instance_column = instance_columns[j];
        if constexpr (PCS::kQueryInstance && PCS::kSupportsBatchMode) {
          prover->BatchCommitAt(instance_column, i * num_instance_columns + j);
        } else if constexpr (PCS::kQueryInstance && !PCS::kSupportsBatchMode) {
          prover->CommitAndWriteToTranscript(instance_column);
        } else {
          for (const F& instance : instance_column.evaluations()) {
            CHECK(prover->GetWriter()->WriteToTranscript(instance));
          }
        }
        instance_polys.push_back(prover->domain()->IFFT(instance_column));
      }
      instance_polys_vec.push_back(std::move(instance_polys));
    }
    if constexpr (PCS::kSupportsBatchMode && PCS::kQueryInstance) {
      prover->RetrieveAndWriteBatchCommitmentsToTranscript();
    }
    return instance_polys_vec;
  }

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

}  // namespace zk::plonk::halo2

namespace base {

template <typename Poly, typename Evals>
class Copyable<zk::plonk::halo2::ArgumentData<Poly, Evals>> {
 public:
  using F = typename Poly::Field;

  static bool WriteTo(const zk::plonk::halo2::ArgumentData<Poly, Evals>& data,
                      Buffer* buffer) {
    return buffer->WriteMany(data.advice_columns_vec_, data.advice_blinds_vec_,
                             data.challenges_, data.instance_columns_vec_,
                             data.instance_polys_vec_);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       zk::plonk::halo2::ArgumentData<Poly, Evals>* data) {
    std::vector<std::vector<Evals>> advice_columns_vec;
    std::vector<std::vector<F>> advice_blinds_vec;
    std::vector<F> challenges;
    std::vector<std::vector<Evals>> instance_columns_vec;
    std::vector<std::vector<Poly>> instance_polys_vec;
    if (!buffer.ReadMany(&advice_columns_vec, &advice_blinds_vec, &challenges,
                         &instance_columns_vec, &instance_polys_vec))
      return false;
    *data = zk::plonk::halo2::ArgumentData<Poly, Evals>(
        std::move(advice_columns_vec), std::move(advice_blinds_vec),
        std::move(challenges), std::move(instance_columns_vec),
        std::move(instance_polys_vec));
    return true;
  }

  static size_t EstimateSize(
      const zk::plonk::halo2::ArgumentData<Poly, Evals>& data) {
    return base::EstimateSize(data.advice_columns_vec_) +
           base::EstimateSize(data.advice_blinds_vec_) +
           base::EstimateSize(data.challenges_) +
           base::EstimateSize(data.instance_columns_vec_) +
           base::EstimateSize(data.instance_polys_vec_);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_ARGUMENT_DATA_H_
