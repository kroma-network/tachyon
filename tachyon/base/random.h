#ifndef TACHYON_BASE_RANDOM_H_
#define TACHYON_BASE_RANDOM_H_

#include <vector>

#include "absl/random/random.h"

#include "tachyon/export.h"

namespace tachyon::base {

TACHYON_EXPORT absl::BitGen& GetAbslBitGen();

template <typename L, typename R>
auto Uniform(L min, R max) {
  return absl::Uniform(GetAbslBitGen(), min, max);
}

template <typename TagType, typename L, typename R>
auto Uniform(TagType tag, L min, R max) {
  return absl::Uniform(tag, GetAbslBitGen(), min, max);
}

template <typename T>
const T& Uniform(const std::vector<T>& vec) {
  return vec[Uniform(static_cast<size_t>(0), vec.size())];
}

TACHYON_EXPORT bool Bernoulli(double probability);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_RANDOM_H_
