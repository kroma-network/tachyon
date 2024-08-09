/**
 * @file column_key.h
 * @brief Column Keys for PLONK Constraint System.
 *
 * This header file defines structures for various types of column keys used in
 * the PLONK constraint system. Column keys uniquely identify columns within the
 * system and may include additional information such as phases for advice
 * columns.
 */
#ifndef TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_COLUMN_KEY_H_
#define TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_COLUMN_KEY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/zk/plonk/constraint_system/phase.h"

/**
 * @struct tachyon_fixed_column_key
 * @brief Key for identifying a fixed column in the PLONK constraint system.
 *
 * This structure represents a key that uniquely identifies a fixed column
 * within a PLONK constraint system. Fixed columns contain constants or other
 * values that do not change between different executions of the protocol.
 */
struct tachyon_fixed_column_key {
  size_t index;
};

/**
 * @struct tachyon_instance_column_key
 * @brief Key for identifying an instance column in the PLONK constraint system.
 *
 * This structure represents a key that uniquely identifies an instance column
 * within a PLONK constraint system. Instance columns contain public inputs to
 * the zero-knowledge proof.
 */
struct tachyon_instance_column_key {
  size_t index;
};

/**
 * @struct tachyon_advice_column_key
 * @brief Key for identifying an advice column in the PLONK constraint system.
 *
 * This structure represents a key that uniquely identifies an advice column
 * within a PLONK constraint system. Advice columns contain auxiliary witness
 * information generated during the proof construction. Each advice column is
 * associated with a phase, indicating when its values are to be used during the
 * proof verification process.
 */
struct tachyon_advice_column_key {
  size_t index;
  struct tachyon_phase phase;
};

#endif  // TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_COLUMN_KEY_H_
