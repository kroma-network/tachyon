// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/keys/proving_key.h"

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2/halo2_prover_test.h"

namespace tachyon::zk {

namespace {

class ProvingKeyTest : public Halo2ProverTest {};

}  // namespace

// TODO(chokobole): Implement test codes.
TEST_F(ProvingKeyTest, Generate) { ProvingKey<PCS> proving_key; }

}  // namespace tachyon::zk
