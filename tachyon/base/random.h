#ifndef TACHYON_BASE_RANDOM_H_
#define TACHYON_BASE_RANDOM_H_

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

TACHYON_EXPORT bool Bernoulli(double probability);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_RANDOM_H_
