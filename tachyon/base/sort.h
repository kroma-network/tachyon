#ifndef TACHYON_BASE_SORT_H_
#define TACHYON_BASE_SORT_H_

#include "third_party/pdqsort/include/pdqsort.h"
#include "third_party/powersort/include/sorts/powersort.h"

namespace tachyon::base {

template <typename Iter>
void UnstableSort(Iter begin, Iter end) {
  return pdqsort(begin, end);
}

template <typename Iter, typename Compare>
void UnstableSort(Iter begin, Iter end, Compare compare) {
  return pdqsort(begin, end, compare);
}

// TODO(chokobole): Add StableSort() with compare version.
template <typename Iter>
void StableSort(Iter begin, Iter end) {
  algorithms::powersort<Iter> sort;
  sort.sort(begin, end);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_SORT_H_
