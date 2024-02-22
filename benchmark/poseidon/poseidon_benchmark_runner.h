#ifndef BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
#define BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "benchmark/poseidon/simple_poseidon_benchmark_reporter.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/cc/math/finite_fields/prime_field_traits_forward.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

namespace tachyon {

template <typename Field>
class PoseidonBenchmarkRunner {
 public:
  using CField = typename cc::math::PrimeFieldTraits<Field>::CPrimeField;

  typedef CField* (*PoseidonExternalFn)(const CField* pre_images,
                                        size_t absorbing_num,
                                        size_t squeezing_num,
                                        uint64_t* duration);

  explicit PoseidonBenchmarkRunner(SimplePoseidonBenchmarkReporter* reporter,
                                   size_t repeating_num, size_t absorbing_num,
                                   size_t squeezing_num)
      : reporter_(reporter),
        repeating_num_(repeating_num),
        absorbing_num_(absorbing_num),
        squeezing_num_(squeezing_num) {
    pre_images_ =
        base::CreateVector(absorbing_num, []() { return Field::Random(); });

    results_.resize(reporter->GetVendorNum());
    for (std::vector<Field>& result : results_) {
      result.reserve(repeating_num_);
    }
  }

  void AddResult(size_t vendor_idx, Field result) {
    results_[vendor_idx].push_back(result);
  }

  void Run() {
    // Measure config time for poseidon hash.
    crypto::PoseidonConfig<Field> config =
        crypto::PoseidonConfig<Field>::CreateCustom(8, 5, 8, 63, 0);
    crypto::PoseidonSponge<Field> sponge(config);

    // Measure test times for poseidon hash.
    for (size_t i = 0; i < repeating_num_; ++i) {
      base::TimeTicks start = base::TimeTicks::Now();
      for (size_t j = 0; j < absorbing_num_; ++j) {
        sponge.Absorb(pre_images_[j]);
      }
      std::vector<Field> squeezed_values =
          sponge.SqueezeFieldElements(squeezing_num_);
      reporter_->AddTime(i, (base::TimeTicks::Now() - start).InSecondsF());
      Field squeezed_sum = std::accumulate(
          squeezed_values.begin(), squeezed_values.end(), Field::Zero(),
          [](Field& total, const Field& value) { return total + value; });
      results_[current_vendor_idx].push_back(std::move(squeezed_sum));
    }
  }

  void RunExternal(PoseidonExternalFn fn) {
    ++current_vendor_idx;

    for (size_t i = 0; i < repeating_num_; ++i) {
      std::vector<uint64_t> durations(repeating_num_, 0);
      uint64_t duration;
      std::unique_ptr<CField> squeezed_sum;
      squeezed_sum.reset(fn(reinterpret_cast<const CField*>(pre_images_.data()),
                            absorbing_num_, squeezing_num_, &duration));
      reporter_->AddTime(i, base::Microseconds(duration).InSecondsF());
      results_[current_vendor_idx].push_back(
          *reinterpret_cast<Field*>(squeezed_sum.get()));
    }
  }

  void CheckResults() const {
    for (size_t i = 1; i < reporter_->GetVendorNum(); ++i) {
      for (size_t j = 0; j < repeating_num_; ++j) {
        CHECK_EQ(results_[0][j], results_[i][j])
            << "The results from Tachyon and " << reporter_->GetVendorName(i)
            << "the vendor differ.";
      }
    }
  }

 private:
  // not owned
  SimplePoseidonBenchmarkReporter* reporter_ = nullptr;
  std::vector<Field> pre_images_;
  size_t repeating_num_ = 0;
  size_t absorbing_num_ = 0;
  size_t squeezing_num_ = 0;

  size_t current_vendor_idx = 0;
  std::vector<std::vector<Field>> results_;
};

}  // namespace tachyon

#endif  // BENCHMARK_POSEIDON_POSEIDON_BENCHMARK_RUNNER_H_
