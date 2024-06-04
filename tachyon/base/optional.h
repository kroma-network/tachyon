#ifndef TACHYON_BASE_OPTIONAL_H_
#define TACHYON_BASE_OPTIONAL_H_

#include <optional>
#include <utility>

#include "tachyon/base/logging.h"

template <typename T>
T&& unwrap(std::optional<T>&& optional_value) {
  CHECK(optional_value);
  return std::move(*optional_value);
}

#endif  // TACHYON_BASE_OPTIONAL_H_
