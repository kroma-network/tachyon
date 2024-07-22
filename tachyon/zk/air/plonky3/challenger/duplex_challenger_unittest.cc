// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/zk/air/plonky3/challenger/duplex_challenger.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/base/bits.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::air::plonky3 {

using F = math::BabyBear;
using Poseidon2 = crypto::Poseidon2Sponge<
    crypto::Poseidon2ExternalMatrix<crypto::Poseidon2Plonky3ExternalMatrix<F>>>;

namespace {

class DuplexChallengerTest : public math::FiniteFieldTest<F> {
 public:
  constexpr static size_t kWidth = 16;
  constexpr static size_t kRate = 4;

  void SetUp() override {
    crypto::Poseidon2Config<F> config =
        crypto::Poseidon2Config<F>::CreateCustom(
            15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    Poseidon2 sponge(std::move(config));
    challenger_.reset(
        new DuplexChallenger<Poseidon2, kWidth, kRate>(std::move(sponge)));
  }

 protected:
  std::unique_ptr<DuplexChallenger<Poseidon2, kWidth, kRate>> challenger_;
};

}  // namespace

TEST_F(DuplexChallengerTest, Sample) {
  for (uint32_t i = 0; i < 20; ++i) {
    challenger_->Observe(F(i));
  }

  F answers[] = {
      F(1091695522), F(747772208), F(1145639564), F(1789312616), F(567623980),
      F(179016966),  F(125050365), F(1725901131), F(65962335),   F(1086560956),
  };
  for (size_t i = 0; i < std::size(answers); ++i) {
    EXPECT_EQ(challenger_->Sample(), answers[i]);
  }
}

TEST_F(DuplexChallengerTest, Grind) {
  const uint32_t kBits = 3;
  F witness = challenger_->Grind(kBits, base::Range<uint32_t>(0, 100));
  EXPECT_TRUE(challenger_->CheckWitness(kBits, witness));
}

}  // namespace tachyon::zk::air::plonky3
