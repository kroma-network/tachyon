// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::zk {

namespace {

class PermutationVerifyingKeyTest : public testing::Test {
 public:
  using PCS = crypto::KZGCommitmentScheme<math::bn254::G1AffinePoint,
                                          math::bn254::G2AffinePoint,
                                          math::bn254::G1AffinePoint>;
  using VerifyingKey = PermutationVerifyingKey<PCS>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(PermutationVerifyingKeyTest, Copyable) {
  VerifyingKey expected({math::bn254::G1AffinePoint::Random()});
  VerifyingKey value;

  base::VectorBuffer write_buf;
  write_buf.Write(expected);

  write_buf.set_buffer_offset(0);

  write_buf.Read(&value);
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk
