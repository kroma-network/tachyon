#ifndef VENDORS_HALO2_SRC_DEGREES_H_
#define VENDORS_HALO2_SRC_DEGREES_H_

#include <stddef.h>

namespace tachyon::halo2_api {

constexpr size_t kMaxDegree = (size_t{1} << 5) - 1;
constexpr size_t kMaxExtendedDegree = (size_t{1} << 7) - 1;

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_SRC_DEGREES_H_
