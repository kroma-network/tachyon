// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FLAG_FLAG_VALUE_TRAITS_H_
#define TACHYON_BASE_FLAG_FLAG_VALUE_TRAITS_H_

#include <stdint.h>

#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon::base {

template <typename T, typename SFINAE = void>
class FlagValueTraits;

template <typename T>
class FlagValueTraits<
    T, std::enable_if_t<std::is_integral<T>::value &&
                        std::is_signed<T>::value && (sizeof(T) <= 32)>> {
 public:
  static bool ParseValue(std::string_view input, T* value,
                         std::string* reason) {
    int value_tmp;
    if (!StringToInt(input, &value_tmp)) {
      *reason = absl::Substitute("failed to convert int (\"$0\")", input);
      return false;
    }
    if (value_tmp <= std::numeric_limits<T>::max() &&
        value_tmp >= std::numeric_limits<T>::min()) {
      *value = static_cast<T>(value_tmp);
      return true;
    }
    *reason = absl::Substitute("$0 is out of its range", input);
    return false;
  }
};

template <typename T>
class FlagValueTraits<
    T,
    std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value &&
                     !std::is_same<T, bool>::value && (sizeof(T) <= 32)>> {
 public:
  static bool ParseValue(std::string_view input, T* value,
                         std::string* reason) {
    unsigned value_tmp;
    if (!StringToUint(input, &value_tmp)) {
      *reason =
          absl::Substitute("failed to convert unsigned int (\"$0\")", input);
      return false;
    }
    if (value_tmp <= std::numeric_limits<T>::max()) {
      *value = static_cast<T>(value_tmp);
      return true;
    }
    *reason = absl::Substitute("$0 is out of its range", input);
    return false;
  }
};

template <>
class FlagValueTraits<float> {
 public:
  static bool ParseValue(std::string_view input, float* value,
                         std::string* reason) {
    if (!StringToFloat(input, value)) {
      *reason = absl::Substitute("failed to convert to float (\"$0\")", input);
      return false;
    }
    return true;
  }
};

template <>
class FlagValueTraits<double> {
 public:
  static bool ParseValue(std::string_view input, double* value,
                         std::string* reason) {
    if (!StringToDouble(input, value)) {
      *reason = absl::Substitute("failed to convert to double (\"$0\")", input);
      return false;
    }
    return true;
  }
};

template <>
class FlagValueTraits<int64_t> {
 public:
  static bool ParseValue(std::string_view input, int64_t* value,
                         std::string* reason) {
    if (!StringToInt64(input, value)) {
      *reason =
          absl::Substitute("failed to convert to int64_t (\"$0\")", input);
      return false;
    }
    return true;
  }
};

template <>
class FlagValueTraits<uint64_t> {
 public:
  static bool ParseValue(std::string_view input, uint64_t* value,
                         std::string* reason) {
    if (!StringToUint64(input, value)) {
      *reason =
          absl::Substitute("failed to convert to uint64_t (\"$0\")", input);
      return false;
    }
    return true;
  }
};

template <>
class FlagValueTraits<bool> {
 public:
  static bool ParseValue(std::string_view input, bool* value,
                         std::string* reason) {
    *value = true;
    return true;
  }
};

template <>
class FlagValueTraits<std::string> {
 public:
  static bool ParseValue(std::string_view input, std::string* value,
                         std::string* reason) {
    if (input.length() == 0) {
      *reason = "input is empty";
      return false;
    }
    *value = std::string(input);
    return true;
  }
};

template <typename T>
class FlagValueTraits<std::vector<T>> {
 public:
  static bool ParseValue(std::string_view input, std::vector<T>* value,
                         std::string* reason) {
    T element;
    if (FlagValueTraits<T>::ParseValue(input, &element, reason)) {
      value->push_back(std::move(element));
      return true;
    }
    return false;
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FLAG_FLAG_VALUE_TRAITS_H_
