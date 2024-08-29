// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_SIMPLE_FRI_STORAGE_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_SIMPLE_FRI_STORAGE_H_

#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_tree_storage.h"

namespace tachyon::crypto {

template <typename Hash>
class SimpleFRIStorage {
 public:
  virtual ~SimpleFRIStorage() = default;

  virtual void Allocate(size_t size) = 0;
  virtual BinaryMerkleTreeStorage<Hash>* GetLayer(size_t index) = 0;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_SIMPLE_FRI_STORAGE_H_
