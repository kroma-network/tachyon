// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_EVALUATION_INPUT_H_
#define TACHYON_ZK_PLONK_VANISHING_EVALUATION_INPUT_H_

#include <stddef.h>

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/owned_table.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class EvaluationInput {
 public:
  using F = typename Poly::Field;

  EvaluationInput(std::vector<F>&& intermediates,
                  std::vector<int32_t>&& rotations,
                  const OwnedTable<Evals>* table,
                  absl::Span<const F> challenges, const F* beta, const F* gamma,
                  const F* theta, const F* y, int32_t n)
      : intermediates_(std::move(intermediates)),
        rotations_(std::move(rotations)),
        table_(table),
        challenges_(challenges),
        beta_(beta),
        gamma_(gamma),
        theta_(theta),
        y_(y),
        n_(n) {}

  const std::vector<F> intermediates() const { return intermediates_; }
  std::vector<F>& intermediates() { return intermediates_; }
  const std::vector<int32_t>& rotations() const { return rotations_; }
  std::vector<int32_t>& rotations() { return rotations_; }
  const OwnedTable<Evals>& table() const { return *table_; }
  absl::Span<const F> challenges() const { return challenges_; }
  const F& beta() const { return *beta_; }
  const F& gamma() const { return *gamma_; }
  const F& theta() const { return *theta_; }
  const F& y() const { return *y_; }
  int32_t n() const { return n_; }

 private:
  std::vector<F> intermediates_;
  std::vector<int32_t> rotations_;
  // not owned
  const OwnedTable<Evals>* table_ = nullptr;
  absl::Span<const F> challenges_;
  // not owned
  const F* beta_ = nullptr;
  // not owned
  const F* gamma_ = nullptr;
  // not owned
  const F* theta_ = nullptr;
  // not owned
  const F* y_ = nullptr;
  int32_t n_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_EVALUATION_INPUT_H_
