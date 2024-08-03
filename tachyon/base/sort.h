#ifndef TACHYON_BASE_SORT_H_
#define TACHYON_BASE_SORT_H_

#include "third_party/pdqsort/include/pdqsort.h"

namespace tachyon::base {

template <typename Iter>
void UnstableSort(Iter begin, Iter end) {
  return pdqsort(begin, end);
}

template <typename Iter, typename Compare>
void UnstableSort(Iter begin, Iter end, Compare compare) {
  return pdqsort(begin, end, compare);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_SORT_H_
