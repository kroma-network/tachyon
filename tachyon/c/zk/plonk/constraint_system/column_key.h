#ifndef TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_COLUMN_KEY_H_
#define TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_COLUMN_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/zk/plonk/constraint_system/phase.h"

struct tachyon_fixed_column_key {
  size_t index;
};

struct tachyon_instance_column_key {
  size_t index;
};

struct tachyon_advice_column_key {
  size_t index;
  struct tachyon_phase phase;
};

#endif  // TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_COLUMN_KEY_H_
