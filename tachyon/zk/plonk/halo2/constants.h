#ifndef TACHYON_ZK_PLONK_HALO2_CONSTANTS_H_
#define TACHYON_ZK_PLONK_HALO2_CONSTANTS_H_

#include <stdint.h>

namespace tachyon::zk::plonk::halo2 {

constexpr char kTranscriptStr[] = "Halo2-Transcript";

// Prefix to a prover's message soliciting a challenge
constexpr uint8_t kBlake2bPrefixChallenge[1] = {0};

// Prefix to a prover's message containing a curve point
constexpr uint8_t kBlake2bPrefixPoint[1] = {1};

// Prefix to a prover's message containing a scalar
constexpr uint8_t kBlake2bPrefixScalar[1] = {2};

// Dummy zeros that come before prefix to a prover's message
constexpr uint8_t kShaPrefixZeros[31] = {0};

// Prefix to a prover's message soliciting a challenge
constexpr uint8_t kShaPrefixChallenge[1] = {0};

// Prefix to a prover's message containing a curve point
constexpr uint8_t kShaPrefixPoint[1] = {1};

// Prefix to a prover's message containing a scalar
constexpr uint8_t kShaPrefixScalar[1] = {2};

constexpr char kVerifyingKeyStr[] = "Halo2-Verify-Key";

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_CONSTANTS_H_
