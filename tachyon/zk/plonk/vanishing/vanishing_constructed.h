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

namespace tachyon::zk::plonk {

template <typename Poly, typename ExtendedPoly>
class VanishingConstructed {
 public:
  using F = typename Poly::Field;

  VanishingConstructed() = default;
  VanishingConstructed(ExtendedPoly&& h_poly, std::vector<F>&& h_blinds,
                       VanishingCommitted<Poly>&& committed)
      : h_poly_(std::move(h_poly)),
        h_blinds_(std::move(h_blinds)),
        committed_(std::move(committed)) {}

  const ExtendedPoly& h_poly() const { return h_poly_; }
  ExtendedPoly& h_poly() { return h_poly_; }
  const std::vector<F>& h_blinds() const { return h_blinds_; }

  VanishingCommitted<Poly>&& TakeCommitted() && {
    return std::move(committed_);
  }

 private:
  ExtendedPoly h_poly_;
  std::vector<F> h_blinds_;
  VanishingCommitted<Poly> committed_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_CONSTRUCTED_H_
