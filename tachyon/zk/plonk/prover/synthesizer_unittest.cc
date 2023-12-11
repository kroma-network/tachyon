// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/prover/synthesizer.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/zk/base/halo2/halo2_prover_test.h"
#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"
#include "tachyon/zk/plonk/keys/halo2/pinned_verifying_key.h"

namespace tachyon::zk {
namespace {

class SynthesizerTest : public Halo2ProverTest {
 public:
  using F = typename PCS::Field;

  void SetUp() override {
    Halo2ProverTest::SetUp();

    circuits_ = {SimpleCircuit<F>(), SimpleCircuit<F>()};

    CHECK(VerifyingKey<PCS>::Generate<SimpleCircuit<F>>(
        prover_.get(), circuits_[0], &verifying_key_));

    synthesizer_ =
        Synthesizer<PCS>(circuits_.size(), &verifying_key_.constraint_system());
  }

 protected:
  std::vector<SimpleCircuit<F>> circuits_;
  VerifyingKey<PCS> verifying_key_;
  Synthesizer<PCS> synthesizer_;
};

}  // namespace

// TODO(dongchangYoo): it should be verified if it produces the expected values.
TEST_F(SynthesizerTest, GenerateAdviceColumns) {
  std::vector<std::vector<Evals>> instance_columns_vec = base::CreateVector(
      circuits_.size(),
      []() { return base::CreateVector(1, Evals::Random(kMaxDegree)); });
  synthesizer_.GenerateAdviceColumns(prover_.get(), circuits_,
                                     instance_columns_vec);

  std::vector<F> challenges = synthesizer_.ExportChallenges();
}

}  // namespace tachyon::zk
