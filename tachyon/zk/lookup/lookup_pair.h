// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_PAIR_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_PAIR_H_

#include <utility>
#include <vector>

namespace tachyon::zk {

template <typename T, typename U = T>
class LookupPair {
 public:
  LookupPair() = default;
  LookupPair(T input, U table)
      : input_(std::move(input)), table_(std::move(table)) {}

  const T& input() const& { return input_; }
  const U& table() const& { return table_; }

  T&& input() && { return std::move(input_); }
  U&& table() && { return std::move(table_); }

 private:
  T input_;
  U table_;
};

template <typename T, typename U = T>
using LookupPairs = std::vector<LookupPair<T, U>>;

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_PAIR_H_
