#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_

#include <stdint.h>

#include "tachyon/c/export.h"

struct TACHYON_C_EXPORT tachyon_bn254_fr {
  uint64_t limbs[4];
};

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
