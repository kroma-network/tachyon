#ifndef TACHYON_CRYPTO_COMMITMENTS_BATCH_COMMITMENT_STATE_H_
#define TACHYON_CRYPTO_COMMITMENTS_BATCH_COMMITMENT_STATE_H_

#include <stddef.h>

#include "tachyon/export.h"

namespace tachyon::crypto {

struct TACHYON_EXPORT BatchCommitmentState {
  bool batch_mode = false;
  size_t batch_count = 0;

  constexpr BatchCommitmentState() = default;
  constexpr BatchCommitmentState(bool batch_mode, size_t batch_count)
      : batch_mode(batch_mode), batch_count(batch_count) {}

  void Reset() {
    batch_mode = false;
    batch_count = 0;
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_BATCH_COMMITMENT_STATE_H_
