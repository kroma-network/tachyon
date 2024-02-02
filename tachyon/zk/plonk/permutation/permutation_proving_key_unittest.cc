// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk::plonk {

namespace {

class PermutationProvingKeyTest : public halo2::ProverTest {
 public:
  using ProvingKey = PermutationProvingKey<Poly, Evals>;
};

}  // namespace

TEST_F(PermutationProvingKeyTest, Copyable) {
  const Domain* domain = prover_->domain();
  ProvingKey expected(
      {domain->Random<Evals>(), domain->Random<Evals>(),
       domain->Random<Evals>()},
      {domain->Random<Poly>(), domain->Random<Poly>(), domain->Random<Poly>()});

  std::vector<uint8_t> vec;
  vec.resize(base::EstimateSize(expected));
  base::Buffer write_buf(vec.data(), vec.size());
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  ProvingKey value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk::plonk
