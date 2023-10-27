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
#include "tachyon/zk/plonk/commitment.h"

namespace tachyon {
namespace zk {

template <typename Curve>
class PermutationVerifyingKey {
 public:
  PermutationVerifyingKey() = default;

  explicit PermutationVerifyingKey(const Commitments<Curve>& commitments)
      : commitments_(commitments) {}
  explicit PermutationVerifyingKey(Commitments<Curve>&& commitments)
      : commitments_(std::move(commitments)) {}

  const Commitments<Curve>& commitments() const { return commitments_; }

  size_t BytesLength() const { return base::EstimateSize(this); }

  bool operator==(const PermutationVerifyingKey& other) const {
    return commitments_ == other.commitments_;
  }
  bool operator!=(const PermutationVerifyingKey& other) const {
    return commitments_ != other.commitments_;
  }

 private:
  Commitments<Curve> commitments_;
};

}  // namespace zk

namespace base {

template <typename Curve>
class Copyable<zk::PermutationVerifyingKey<Curve>> {
 public:
  static bool WriteTo(const zk::PermutationVerifyingKey<Curve>& vk,
                      Buffer* buffer) {
    return buffer->Write(vk.commitments());
  }

  static bool ReadFrom(const Buffer& buffer,
                       zk::PermutationVerifyingKey<Curve>* vk) {
    zk::Commitments<Curve> commitments;
    if (!buffer.Read(&commitments)) return false;
    *vk = zk::PermutationVerifyingKey<Curve>(std::move(commitments));
    return true;
  }

  static size_t EstimateSize(const zk::PermutationVerifyingKey<Curve>& vk) {
    return base::EstimateSize(vk.commitments());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFYING_KEY_H_
