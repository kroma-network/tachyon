// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFYING_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace zk {

template <typename PCSTy>
class PermutationVerifyingKey {
 public:
  using Commitment = typename PCSTy::ResultTy;
  using Commitments = std::vector<Commitment>;

  PermutationVerifyingKey() = default;

  explicit PermutationVerifyingKey(const Commitments& commitments)
      : commitments_(commitments) {}
  explicit PermutationVerifyingKey(Commitments&& commitments)
      : commitments_(std::move(commitments)) {}

  const Commitments& commitments() const { return commitments_; }

  size_t BytesLength() const { return base::EstimateSize(this); }

  bool operator==(const PermutationVerifyingKey& other) const {
    return commitments_ == other.commitments_;
  }
  bool operator!=(const PermutationVerifyingKey& other) const {
    return commitments_ != other.commitments_;
  }

 private:
  Commitments commitments_;
};

}  // namespace zk

namespace base {

template <typename PCSTy>
class Copyable<zk::PermutationVerifyingKey<PCSTy>> {
 public:
  static bool WriteTo(const zk::PermutationVerifyingKey<PCSTy>& vk,
                      Buffer* buffer) {
    return buffer->Write(vk.commitments());
  }

  static bool ReadFrom(const Buffer& buffer,
                       zk::PermutationVerifyingKey<PCSTy>* vk) {
    typename zk::PermutationVerifyingKey<PCSTy>::Commitments commitments;
    if (!buffer.Read(&commitments)) return false;
    *vk = zk::PermutationVerifyingKey<PCSTy>(std::move(commitments));
    return true;
  }

  static size_t EstimateSize(const zk::PermutationVerifyingKey<PCSTy>& vk) {
    return base::EstimateSize(vk.commitments());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFYING_KEY_H_
