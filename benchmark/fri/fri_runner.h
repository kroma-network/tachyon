#ifndef BENCHMARK_FRI_FRI_RUNNER_H_
#define BENCHMARK_FRI_FRI_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/fri/fri_config.h"
#include "benchmark/simple_reporter.h"
// clang-format on
#include "tachyon/base/time/time.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/crypto/commitments/fri/two_adic_multiplicative_coset.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::benchmark {

template <typename PCS>
class FRIRunner {
 public:
  using F = typename PCS::F;
  using ExtF = typename PCS::ExtF;
  using Domain = typename PCS::Domain;
  using Challenger = typename PCS::Challenger;
  using Commitment = typename PCS::Commitment;
  using ProverData = typename PCS::ProverData;
  using FRIProof = typename PCS::FRIProof;
  using OpenedValues = typename PCS::OpenedValues;

  typedef tachyon_baby_bear* (*ExternalFn)(const tachyon_baby_bear* data,
                                           size_t input_num, size_t round_num,
                                           size_t max_degree, size_t batch_size,
                                           uint32_t log_blowup,
                                           uint64_t* duration);

  FRIRunner(SimpleReporter& reporter, const FRIConfig& config, PCS& pcs,
            Challenger& challenger)
      : reporter_(reporter),
        config_(config),
        pcs_(pcs),
        challenger_(challenger) {}

  ExtF Run(Vendor vendor, const math::RowMajorMatrix<F>& input) {
    size_t max_degree = static_cast<size_t>(input.rows());
    std::vector<Commitment> commits_by_round(config_.round_num());
    std::vector<ProverData> data_by_round(config_.round_num());
    std::vector<std::vector<Domain>> domains_by_round(config_.round_num());
    std::vector<std::vector<math::RowMajorMatrix<F>>> inner_polys_by_round(
        config_.round_num());

    Challenger p_challenger = challenger_;

    for (size_t round = 0; round < config_.round_num(); round++) {
      std::vector<size_t> degrees = GetInputDegrees(max_degree, round);

      domains_by_round[round] = base::Map(degrees, [this](size_t degree) {
        return this->pcs_.GetNaturalDomainForDegree(degree);
      });

      inner_polys_by_round[round] =
          base::Map(degrees, [this, round, input](size_t degree) {
            math::RowMajorMatrix<F> ret =
                Eigen::Map<const math::RowMajorMatrix<F>>(
                    &input.data()[round], degree, config_.batch_size());
            return ret;
          });
    }

    base::TimeTicks start = base::TimeTicks::Now();
    for (size_t round = 0; round < config_.round_num(); round++) {
      CHECK(pcs_.Commit(domains_by_round[round], inner_polys_by_round[round],
                        &commits_by_round[round], &data_by_round[round]));
    }
    p_challenger.ObserveContainer2D(commits_by_round);
    ExtF zeta = p_challenger.template SampleExtElement<ExtF>();

    std::vector<std::vector<std::vector<ExtF>>> points_by_round(
        config_.round_num());
    for (size_t round = 0; round < config_.round_num(); ++round) {
      points_by_round[round] =
          std::vector<std::vector<ExtF>>(config_.input_num(), {zeta});
    }
    OpenedValues openings;
    FRIProof fri_proof;
    CHECK(pcs_.CreateOpeningProof(data_by_round, points_by_round, p_challenger,
                                  &openings, &fri_proof, pow_witness_));
    reporter_.AddTime(vendor, base::TimeTicks::Now() - start);
    return fri_proof.final_eval;
  }

  ExtF RunExternal(Vendor vendor, ExternalFn fn,
                   const math::RowMajorMatrix<F>& input) {
    size_t max_degree = static_cast<size_t>(input.rows());

    uint64_t duration_in_us = 0;
    tachyon_baby_bear* data =
        fn(c::base::c_cast(input.data()), config_.input_num(),
           config_.round_num(), max_degree, config_.batch_size(),
           config_.log_blowup(), &duration_in_us);
    reporter_.AddTime(vendor, base::Microseconds(duration_in_us));
    pow_witness_ = c::base::native_cast(data)[0];
    ExtF final_eval{
        c::base::native_cast(data)[1], c::base::native_cast(data)[2],
        c::base::native_cast(data)[3], c::base::native_cast(data)[4]};
    return final_eval;
  }

 private:
  std::vector<size_t> GetInputDegrees(size_t max_degree, size_t round) {
    std::vector<size_t> degrees;
    degrees.reserve(config_.input_num());
    for (size_t i = 0; i < config_.input_num(); ++i) {
      degrees.push_back(max_degree >> (i + round));
    }
    return degrees;
  }

  SimpleReporter& reporter_;
  const FRIConfig& config_;
  PCS& pcs_;
  Challenger& challenger_;
  std::optional<F> pow_witness_;
};

}  // namespace tachyon::benchmark

#endif  // BENCHMARK_FRI_FRI_RUNNER_H_
