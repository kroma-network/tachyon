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
namespace zk::plonk {

template <typename Commitment>
class PermutationVerifyingKey {
 public:
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

}  // namespace zk::plonk

namespace base {

template <typename Commitment>
class Copyable<zk::plonk::PermutationVerifyingKey<Commitment>> {
 public:
  static bool WriteTo(const zk::plonk::PermutationVerifyingKey<Commitment>& vk,
                      Buffer* buffer) {
    return buffer->Write(vk.commitments());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       zk::plonk::PermutationVerifyingKey<Commitment>* vk) {
    typename zk::plonk::PermutationVerifyingKey<Commitment>::Commitments
        commitments;
    if (!buffer.Read(&commitments)) return false;
    *vk =
        zk::plonk::PermutationVerifyingKey<Commitment>(std::move(commitments));
    return true;
  }

  static size_t EstimateSize(
      const zk::plonk::PermutationVerifyingKey<Commitment>& vk) {
    return base::EstimateSize(vk.commitments());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_VERIFYING_KEY_H_
