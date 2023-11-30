// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/zk/base/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class PermutationProvingKeyTest : public Halo2ProverTest {
 public:
  using ProvingKey = PermutationProvingKey<PCS>;
};

}  // namespace

TEST_F(PermutationProvingKeyTest, Copyable) {
  // NOTE(chokobole): Since https://github.com/kroma-network/tachyon/pull/139,
  // I intentionally use |Evals::Zero()| instead of |Evals::Random()| due to
  // performance issues.
  ProvingKey expected({Evals::Zero()}, {Poly::Random(5)});
  ProvingKey value;

  base::VectorBuffer write_buf;
  write_buf.Write(expected);

  write_buf.set_buffer_offset(0);

  write_buf.Read(&value);
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk
