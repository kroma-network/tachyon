#include "benchmark/msm/arkworks/include/arkworks_benchmark.h"

#include <iostream>
#include <string>

#include "absl/types/span.h"

// clang-format off
#include "benchmark/msm/arkworks/src/main.rs.h"
#include "benchmark/simple_benchmark_reporter.h"
// clang-format on
#include "tachyon/base/console/iostream.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/time/time_interval.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/msm/msm.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/bn254_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon {

using namespace math;

rust::Vec<uint64_t> get_test_nums(rust::Slice<const rust::String> shared_argv) {
  std::vector<char*> argv;
  for (const rust::String& shared_string : shared_argv) {
    argv.push_back(const_cast<char*>(shared_string.data()));
  }
  size_t argc = argv.size();

  std::vector<uint64_t> test_log_nums;
  base::FlagParser parser;
  parser.AddFlag<base::Flag<std::vector<uint64_t>>>(&test_log_nums)
      .set_short_name("-n")
      .set_required()
      .set_help("The log number of points to test");
  {
    std::string error;
    if (!parser.Parse(argc, argv.data(), &error)) {
      tachyon_cerr << error << std::endl;
    }
  }
  std::cout << "test_log_nums: " << test_log_nums.size() << std::endl;

  base::ranges::sort(test_log_nums);

  rust::Vec<uint64_t> test_nums;
  for (uint64_t test_log_num : test_log_nums) {
    test_nums.push_back(1 << test_log_num);
  }

  return test_nums;
}

void arkworks_benchmark(rust::Slice<const rust::u64> test_nums,
                        rust::Slice<const CppG1Affine> bases,
                        rust::Slice<const CppFr> scalars,
                        rust::Slice<const CppG1Jacobian> results_arkworks_in,
                        rust::Slice<const rust::f64> durations_arkworks) {
  std::vector<std::string> names;
  std::vector<double> results;
  names.reserve(test_nums.size() * 2);
  results.reserve(test_nums.size() * 2);
  for (uint64_t test_num : test_nums) {
    names.push_back(absl::Substitute("Arkworks/$0", std::log2(test_num)));
  }
  for (uint64_t test_num : test_nums) {
    names.push_back(absl::Substitute("Tachyon/$0", std::log2(test_num)));
  }
  for (const auto& duration : durations_arkworks) {
    results.push_back(duration);
  }
  SimpleBenchmarkReporter reporter(std::move(names), std::move(results));

  std::cout << std::endl;
  std::cout << "Executing Tachyon MSM..." << std::endl;
  base::TimeInterval interval(base::TimeTicks::Now());

  std::vector<bn254::G1JacobianPoint> results_tachyon;
  for (uint64_t test_num : test_nums) {
    std::unique_ptr<tachyon_bn254_g1_jacobian> ret;
    ret.reset(tachyon_msm_g1_affine(
        reinterpret_cast<const tachyon_bn254_g1_affine*>(bases.data()),
        test_num, reinterpret_cast<const tachyon_bn254_fr*>(scalars.data()),
        test_num));
    results_tachyon.push_back(
        *reinterpret_cast<bn254::G1JacobianPoint*>(ret.get()));
    tachyon::base::TimeDelta duration = interval.GetTimeDelta();
    std::cout << "calculate: " << duration.InSecondsF() << std::endl;
    reporter.AddResult(duration.InSecondsF());
  }

  absl::Span<const bn254::G1JacobianPoint> results_arkworks =
      absl::MakeConstSpan(reinterpret_cast<const bn254::G1JacobianPoint*>(
                              results_arkworks_in.data()),
                          results_arkworks_in.size());
  CHECK(results_arkworks == absl::MakeConstSpan(results_tachyon))
      << "Result not matched";

  reporter.Show();
}

}  // namespace tachyon
