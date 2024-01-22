#ifndef VENDORS_HALO2_SRC_ENDIAN_AUTO_RESET_H_
#define VENDORS_HALO2_SRC_ENDIAN_AUTO_RESET_H_

#include "tachyon/base/buffer/buffer.h"

namespace tachyon::halo2_api {

struct EndianAutoReset {
  explicit EndianAutoReset(base::Buffer& buffer, base::Endian endian)
      : buffer(buffer), old_endian(buffer.endian()) {
    buffer.set_endian(endian);
  }
  ~EndianAutoReset() { buffer.set_endian(old_endian); }

  base::Buffer& buffer;
  base::Endian old_endian;
};

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_SRC_ENDIAN_AUTO_RESET_H_
