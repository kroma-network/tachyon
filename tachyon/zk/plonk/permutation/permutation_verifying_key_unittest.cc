// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk {

namespace {

class PermutationVerifyingKeyTest : public halo2::ProverTest {
 public:
  using VerifyingKey = PermutationVerifyingKey<Commitment>;
};

}  // namespace

TEST_F(PermutationVerifyingKeyTest, Copyable) {
  VerifyingKey expected({math::bn254::G1AffinePoint::Random(),
                         math::bn254::G1AffinePoint::Random(),
                         math::bn254::G1AffinePoint::Random()});

  std::vector<uint8_t> vec;
  vec.resize(base::EstimateSize(expected));
  base::Buffer write_buf(vec.data(), vec.size());
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  VerifyingKey value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk::plonk
