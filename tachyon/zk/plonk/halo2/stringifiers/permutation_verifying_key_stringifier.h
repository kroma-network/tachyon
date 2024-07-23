// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_PERMUTATION_VERIFYING_KEY_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_PERMUTATION_VERIFYING_KEY_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/halo2/stringifiers/point_stringifier.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

namespace tachyon::base::internal {

template <typename Commitment>
class RustDebugStringifier<zk::plonk::PermutationVerifyingKey<Commitment>> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const zk::plonk::PermutationVerifyingKey<Commitment>& vk) {
    // NOTE(chokobole): See
    // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/plonk/permutation.rs#L80-L84
    return os << fmt.DebugStruct("VerifyingKey")
                     .Field("commitments", vk.commitments())
                     .Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_PERMUTATION_VERIFYING_KEY_STRINGIFIER_H_
