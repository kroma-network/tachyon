/**
 * @file phase.h
 * @brief Phase Structure for PLONK Constraint System.
 *
 * This header file defines the tachyon_phase structure, which represents the
 * phase of an operation or element within the PLONK constraint system. Phases
 * are used to organize and control the execution order of various operations in
 * PLONK proofs, ensuring the correct application of constraints and
 * evaluations.
 */
#ifndef TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_PHASE_H_
#define TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_PHASE_H_

#include <stdint.h>

/**
 * @struct tachyon_phase
 * @brief Represents a phase in the PLONK constraint system.
 *
 * A phase is a conceptual tool used in the PLONK constraint system to denote
 * the timing or sequencing of operations. It helps in organizing the constraint
 * system and ensuring that operations are performed in the intended order. Each
 * phase is associated with certain columns or operations within the PLONK
 * setup, enabling fine-grained control over the proof generation and
 * verification processes.
 */
struct tachyon_phase {
  uint8_t value;
};

#endif  // TACHYON_C_ZK_PLONK_CONSTRAINT_SYSTEM_PHASE_H_
