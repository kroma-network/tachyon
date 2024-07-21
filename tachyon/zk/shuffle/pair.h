// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_PAIR_H_
#define TACHYON_ZK_SHUFFLE_PAIR_H_

#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/json/json.h"

namespace tachyon {
namespace zk::shuffle {

template <typename T, typename U = T>
class Pair {
 public:
  Pair() = default;
  Pair(T input, U shuffle)
      : input_(std::move(input)), shuffle_(std::move(shuffle)) {}

  const T& input() const { return input_; }
  const U& shuffle() const { return shuffle_; }
  T& input() { return input_; }
  U& shuffle() { return shuffle_; }

  T&& TakeInput() && { return std::move(input_); }
  U&& TakeShuffle() && { return std::move(shuffle_); }

  bool operator==(const Pair& other) const {
    return input_ == other.input_ && shuffle_ == other.shuffle_;
  }
  bool operator!=(const Pair& other) const { return !operator==(other); }

 private:
  T input_;
  U shuffle_;
};

template <typename T, typename U = T>
using Pairs = std::vector<Pair<T, U>>;

}  // namespace zk::shuffle

namespace base {

template <typename T, typename U>
class RapidJsonValueConverter<zk::shuffle::Pair<T, U>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const zk::shuffle::Pair<T, U>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "input", value.input(), allocator);
    AddJsonElement(object, "shuffle", value.shuffle(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 zk::shuffle::Pair<T, U>* value, std::string* error) {
    T input;
    U shuffle;
    if (!ParseJsonElement(json_value, "input", &input, error)) return false;
    if (!ParseJsonElement(json_value, "shuffle", &shuffle, error)) return false;
    *value = zk::shuffle::Pair<T, U>(std::move(input), std::move(shuffle));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_SHUFFLE_PAIR_H_
