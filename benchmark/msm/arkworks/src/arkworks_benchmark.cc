#include "benchmark/msm/arkworks/include/arkworks_benchmark.h"

#include <iostream>
#include <string>

#include "benchmark/msm/arkworks/src/main.rs.h"

namespace benchmark::msm::arkworks {

void arkworks_benchmark(const std::string& s) {
  std::cout << "Benchmark initiated: " << s << std::endl;
  std::string test_set = "test-set from Tachyon";
  group_operation(test_set);
}

}  // namespace benchmark::msm::arkworks
