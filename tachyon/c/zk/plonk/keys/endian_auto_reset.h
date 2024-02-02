#ifndef TACHYON_C_ZK_PLONK_KEYS_ENDIAN_AUTO_RESET_H_
#define TACHYON_C_ZK_PLONK_KEYS_ENDIAN_AUTO_RESET_H_

#include "tachyon/base/buffer/buffer.h"

namespace tachyon::c::zk::plonk {

struct EndianAutoReset {
  explicit EndianAutoReset(const base::ReadOnlyBuffer& buffer,
                           base::Endian endian)
      : buffer(buffer), old_endian(buffer.endian()) {
    buffer.set_endian(endian);
  }
  ~EndianAutoReset() { buffer.set_endian(old_endian); }

  const base::ReadOnlyBuffer& buffer;
  base::Endian old_endian;
};

}  // namespace tachyon::c::zk::plonk

#endif  // TACHYON_C_ZK_PLONK_KEYS_ENDIAN_AUTO_RESET_H_
