// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_CONSTRUCTED_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_CONSTRUCTED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/vanishing/vanishing_committed.h"

namespace tachyon::zk {

template <typename PCS>
class VanishingConstructed {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;
  using ExtendedPoly = typename PCS::ExtendedPoly;

  VanishingConstructed() = default;
  VanishingConstructed(std::vector<Poly>&& h_pieces, std::vector<F>&& h_blinds,
                       VanishingCommitted<Poly>&& committed)
      : h_pieces_(std::move(h_pieces)),
        h_blinds_(std::move(h_blinds)),
        committed_(std::move(committed)) {}

  const std::vector<Poly>& h_pieces() const { return h_pieces_; }
  const std::vector<F>& h_blinds() const { return h_blinds_; }

  VanishingCommitted<Poly>&& TakeCommitted() && {
    return std::move(committed_);
  }

 private:
  std::vector<Poly> h_pieces_;
  std::vector<F> h_blinds_;
  VanishingCommitted<Poly> committed_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_CONSTRUCTED_H_
