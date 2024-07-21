// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_PAIR_H_
#define TACHYON_ZK_LOOKUP_PAIR_H_

#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/json/json.h"

namespace tachyon {
namespace zk::lookup {

template <typename T, typename U = T>
class Pair {
 public:
  Pair() = default;
  Pair(T input, U table) : input_(std::move(input)), table_(std::move(table)) {}

  const T& input() const { return input_; }
  const U& table() const { return table_; }
  T& input() { return input_; }
  U& table() { return table_; }

  T&& TakeInput() && { return std::move(input_); }
  U&& TakeTable() && { return std::move(table_); }

  bool operator==(const Pair& other) const {
    return input_ == other.input_ && table_ == other.table_;
  }
  bool operator!=(const Pair& other) const { return !operator==(other); }

 private:
  T input_;
  U table_;
};

template <typename T, typename U = T>
using Pairs = std::vector<Pair<T, U>>;

}  // namespace zk::lookup

namespace base {

template <typename T, typename U>
class RapidJsonValueConverter<zk::lookup::Pair<T, U>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const zk::lookup::Pair<T, U>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "input", value.input(), allocator);
    AddJsonElement(object, "table", value.table(), allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 zk::lookup::Pair<T, U>* value, std::string* error) {
    T input;
    U table;
    if (!ParseJsonElement(json_value, "input", &input, error)) return false;
    if (!ParseJsonElement(json_value, "table", &table, error)) return false;
    *value = zk::lookup::Pair<T, U>(std::move(input), std::move(table));
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_LOOKUP_PAIR_H_
