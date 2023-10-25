// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_ERROR_H_
#define TACHYON_ZK_PLONK_ERROR_H_

namespace tachyon::zk {

enum class Error {
  // Success case
  kNone,
  // This is an error that can occur during synthesis of the circuit, for
  // example, when the witness is not present.
  kSynthesis,
  // The provided instances do not match the circuit parameters.
  kInvalidInstances,
  // The constraint system is not satisfied.
  kConstraintSystemFailure,
  // Out of bounds index passed to a backend
  kBoundsFailure,
  // Opening error
  kOpening,
  // Transcript error
  kTranscript,
  // The provided degree is too small for the given circuit.
  kNotEnoughRowsAvailable,
  // Instance provided exceeds number of available rows
  kInstanceTooLarge,
  // Circuit synthesis requires global constants, but circuit configuration
  // did not call |ConstraintSystem::EnableConstant| on fixed columns with
  // sufficient space.
  kNotEnoughColumnsForConstants,
  // The instance sets up a copy constraint involving a column that has not
  // been included in the permutation.
  kColumnNotInPermutation,
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_ERROR_H_
