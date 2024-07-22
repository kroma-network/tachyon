// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/zk/air/plonky3/challenger/multi_field32_challenger.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/elliptic_curves/bn/bn254/poseidon2.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"

namespace tachyon::zk::air::plonky3 {

using F = math::BabyBear;
using Poseidon2 = crypto::Poseidon2Sponge<crypto::Poseidon2ExternalMatrix<
    crypto::Poseidon2Plonky3ExternalMatrix<math::bn254::Fr>>>;

namespace {

class MultiField32ChallengerTest : public testing::Test {
 public:
  constexpr static size_t kWidth = 3;

  static void SetUpTestSuite() {
    F::Init();
    math::bn254::Fr::Init();
  }

  void SetUp() override {
    crypto::Poseidon2Config<math::bn254::Fr> config =
        crypto::Poseidon2Config<math::bn254::Fr>::CreateCustom(
            2, 5, 8, 56, math::bn254::GetPoseidon2InternalDiagonalVector<3>());
    Poseidon2 sponge(std::move(config));
    challenger_.reset(
        new MultiField32Challenger<F, Poseidon2, kWidth>(std::move(sponge)));
  }

 protected:
  std::unique_ptr<MultiField32Challenger<F, Poseidon2, kWidth>> challenger_;
};

}  // namespace

TEST_F(MultiField32ChallengerTest, Sample) {
  for (uint32_t i = 0; i < 20; ++i) {
    challenger_->Observe(F(i));
  }

  F answers[] = {
      F(72199253),   F(733473132), F(442816494),  F(326641700),  F(1342573676),
      F(1242755868), F(887300172), F(1831922292), F(1518709680),
  };
  for (size_t i = 0; i < std::size(answers); ++i) {
    EXPECT_EQ(challenger_->Sample(), answers[i]);
  }
}

TEST_F(MultiField32ChallengerTest, Grind) {
  const uint32_t kBits = 3;
  F witness = challenger_->Grind(kBits, base::Range<uint32_t>(0, 100));
  EXPECT_TRUE(challenger_->CheckWitness(kBits, witness));
}

}  // namespace tachyon::zk::air::plonky3
