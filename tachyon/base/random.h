#ifndef TACHYON_BASE_RANDOM_H_
#define TACHYON_BASE_RANDOM_H_

#include <vector>

#include "absl/random/random.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/range.h"
#include "tachyon/export.h"

namespace tachyon::base {

TACHYON_EXPORT absl::BitGen& GetAbslBitGen();

template <typename T, bool IsStartInclusive, bool IsEndInclusive>
T Uniform(const Range<T, IsStartInclusive, IsEndInclusive>& range) {
  CHECK(!range.IsEmpty());
  if constexpr (IsStartInclusive && IsEndInclusive) {
    return absl::Uniform(absl::IntervalClosedClosed, GetAbslBitGen(),
                         range.from, range.to);
  } else if constexpr (IsStartInclusive && !IsEndInclusive) {
    return absl::Uniform(absl::IntervalClosedOpen, GetAbslBitGen(), range.from,
                         range.to);
  } else if constexpr (!IsStartInclusive && IsEndInclusive) {
    return absl::Uniform(absl::IntervalOpenClosed, GetAbslBitGen(), range.from,
                         range.to);
  } else {
    return absl::Uniform(absl::IntervalOpenOpen, GetAbslBitGen(), range.from,
                         range.to);
  }
}

template <typename Container,
          typename R = decltype(std::declval<Container>()[0])>
R UniformElement(Container&& container) {
  return container[Uniform(Range<size_t>::Until(std::size(container)))];
}

TACHYON_EXPORT bool Bernoulli(double probability);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_RANDOM_H_
