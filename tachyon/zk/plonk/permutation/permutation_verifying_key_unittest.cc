// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/zk/base/halo2/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class PermutationVerifyingKeyTest : public Halo2ProverTest {
 public:
  using VerifyingKey = PermutationVerifyingKey<PCS>;
};

}  // namespace

TEST_F(PermutationVerifyingKeyTest, Copyable) {
  VerifyingKey expected({math::bn254::G1AffinePoint::Random()});
  VerifyingKey value;

  base::VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);

  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk
