#ifndef TACHYON_BASE_BUFFER_ENDIAN_AUTO_RESET_H_
#define TACHYON_BASE_BUFFER_ENDIAN_AUTO_RESET_H_

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/export.h"

namespace tachyon::base {

struct TACHYON_EXPORT EndianAutoReset {
  explicit EndianAutoReset(const base::ReadOnlyBuffer& buffer,
                           base::Endian endian)
      : buffer(buffer), old_endian(buffer.endian()) {
    buffer.set_endian(endian);
  }
  ~EndianAutoReset() { buffer.set_endian(old_endian); }

  const base::ReadOnlyBuffer& buffer;
  base::Endian old_endian;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_ENDIAN_AUTO_RESET_H_
