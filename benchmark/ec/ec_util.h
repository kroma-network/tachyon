#ifndef BENCHMARK_EC_EC_UTIL_H_
#define BENCHMARK_EC_EC_UTIL_H_

#include <stdint.h>

#include <vector>

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon {

std::vector<math::bn254::G1AffinePoint> CreateRandomBn254Points(uint64_t nums);
std::vector<math::bn254::Fr> CreateRandomBn254Scalars(uint64_t nums);

}  // namespace tachyon

#endif  // BENCHMARK_EC_EC_UTIL_H_
