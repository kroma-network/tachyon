// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

namespace {

class PoseidonTest : public math::FiniteFieldTest<math::bls12_381::Fr> {};

}  // namespace

TEST_F(PoseidonTest, AbsorbSqueeze) {
  using Fr = math::bls12_381::Fr;

  PoseidonConfig<Fr> config = PoseidonConfig<Fr>::CreateDefault(2, false);
  PoseidonSponge<Fr> sponge(config);
  std::vector<Fr> inputs = {Fr(0), Fr(1), Fr(2)};
  ASSERT_TRUE(sponge.Absorb(inputs));
  std::vector<Fr> result = sponge.SqueezeNativeFieldElements(3);
  std::vector<Fr> expected = {
      *Fr::FromDecString(
          "404427934635713040283377530022421867103101638970489622"
          "78675457993207843616876"),
      *Fr::FromDecString(
          "266437446169989800029115314522409928771122402171620296"
          "0480903840045233645301"),
      *Fr::FromDecString(
          "501910788280669236620702282565306929518015040434228440"
          "38937334196346054068797"),
  };
  EXPECT_EQ(result, expected);
}

TEST_F(PoseidonTest, Copyable) {
  using Fr = math::bls12_381::Fr;

  PoseidonConfig<Fr> config = PoseidonConfig<Fr>::CreateDefault(2, false);
  PoseidonSponge<Fr> expected(config);

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  PoseidonSponge<Fr> value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::crypto
