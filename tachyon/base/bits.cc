#include "tachyon/base/bits.h"

namespace tachyon::base {
namespace bits {

uint64_t BitRev(uint64_t n) {
#if defined(__clang__) && HAS_BUILTIN(__builtin_convertvector)
  return __builtin_bitreverse64(n);
#else
  size_t count = 63;
  uint64_t rev = n;
  while ((n >>= 1) > 0) {
    rev <<= 1;
    rev |= n & 1;
    --count;
  }
  rev <<= count;
  return rev;
#endif
}

}  // namespace bits
}  // namespace tachyon::base
