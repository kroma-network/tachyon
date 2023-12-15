// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk {

namespace {

class PermutationProvingKeyTest : public halo2::ProverTest {
 public:
  using ProvingKey = PermutationProvingKey<Poly, Evals>;
};

}  // namespace

TEST_F(PermutationProvingKeyTest, Copyable) {
  const Domain* domain = prover_->domain();
  ProvingKey expected({domain->Random<Evals>()}, {domain->Random<Poly>()});
  ProvingKey value;

  base::VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);

  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk
