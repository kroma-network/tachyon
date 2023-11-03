// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_EVALS_PAIR_H_
#define TACHYON_ZK_BASE_EVALS_PAIR_H_

#include <utility>

namespace tachyon::zk {

template <typename Evals>
class EvalsPair {
 public:
  EvalsPair() = default;
  EvalsPair(const Evals& input, const Evals& table)
      : input_(input), table_(table) {}

  EvalsPair(Evals&& input, Evals&& table)
      : input_(std::move(input)), table_(std::move(table)) {}

  const Evals& input() const { return input_; }

  const Evals& table() const { return table_; }

 private:
  Evals input_;
  Evals table_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_EVALS_PAIR_H_
