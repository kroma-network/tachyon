// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/zk/air/plonky3/challenger/hash_challenger.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::air::plonky3 {

using F = math::BabyBear;
using Poseidon2 = crypto::Poseidon2Sponge<
    crypto::Poseidon2ExternalMatrix<crypto::Poseidon2Plonky3ExternalMatrix<F>>>;
using Hasher = crypto::PaddingFreeSponge<Poseidon2, 8, 8>;

namespace {

class HashChallengerTest : public math::FiniteFieldTest<F> {
 public:
  void SetUp() override {
    crypto::Poseidon2Config<F> config =
        crypto::Poseidon2Config<F>::CreateCustom(
            15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    Poseidon2 sponge(std::move(config));
    Hasher hasher(std::move(sponge));

    std::vector<F> initial_state =
        base::CreateVector(10, [](size_t i) { return F(i + 1); });
    challenger_.reset(new HashChallenger<Hasher>(std::move(initial_state),
                                                 std::move(hasher)));
  }

 protected:
  std::unique_ptr<HashChallenger<Hasher>> challenger_;
};

}  // namespace

TEST_F(HashChallengerTest, Sample) {
  for (uint32_t i = 0; i < 20; ++i) {
    challenger_->Observe(F(i));
  }

  F answers[] = {
      F(886174168),  F(1457271233), F(1952268252), F(1595005924), F(796215768),
      F(1553987485), F(1108393593), F(1336137665), F(971109448),  F(1853357459),
  };
  for (size_t i = 0; i < std::size(answers); ++i) {
    EXPECT_EQ(challenger_->Sample(), answers[i]);
  }
}

// NOTE(chokobole): Grind is not tested since |HashChallenger| doesn't implement
// |GrindingChallenger| traits.
// See
// https://github.com/Plonky3/Plonky3/blob/7bb6db50594e159010f11c97d110aa3ee121069b/challenger/src/grinding_challenger.rs.

}  // namespace tachyon::zk::air::plonky3
