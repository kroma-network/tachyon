// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#include "tachyon/crypto/commitments/fri/simple_fri.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/simple_binary_merkle_tree_storage.h"
#include "tachyon/crypto/transcripts/simple_transcript.h"
#include "tachyon/math/finite_fields/goldilocks/goldilocks.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::crypto {

namespace {

class SimpleHasher
    : public BinaryMerkleHasher<math::Goldilocks, math::Goldilocks> {
 public:
  // BinaryMerkleHasher<math::Goldilocks, math::Goldilocks> methods
  math::Goldilocks ComputeLeafHash(
      const math::Goldilocks& leaf) const override {
    return leaf;
  }
  math::Goldilocks ComputeParentHash(
      const math::Goldilocks& left,
      const math::Goldilocks& right) const override {
    return left + right.Double();
  }
};

class SimpleFRIStorageImpl : public SimpleFRIStorage<math::Goldilocks> {
 public:
  const std::vector<SimpleBinaryMerkleTreeStorage<math::Goldilocks>>& layers()
      const {
    return layers_;
  }

  // FRIStorage<math::Goldilocks> methods
  void Allocate(size_t size) override { layers_.resize(size); }
  BinaryMerkleTreeStorage<math::Goldilocks>* GetLayer(size_t index) override {
    return &layers_[index];
  }

 private:
  std::vector<SimpleBinaryMerkleTreeStorage<math::Goldilocks>> layers_;
};

class FRITest : public math::FiniteFieldTest<math::Goldilocks> {
 public:
  constexpr static size_t K = 3;
  constexpr static size_t N = size_t{1} << K;
  constexpr static size_t kMaxDegree = N - 1;

  using PCS = SimpleFRI<math::Goldilocks, kMaxDegree>;
  using F = PCS::Field;
  using Poly = PCS::Poly;
  using Commitment = PCS::Commitment;
  using Domain = PCS::Domain;

  void SetUp() override {
    domain_ = Domain::Create(N);
    pcs_ = PCS(domain_.get(), &storage_, &hasher_);
  }

 protected:
  std::unique_ptr<Domain> domain_;
  SimpleFRIStorageImpl storage_;
  SimpleHasher hasher_;
  PCS pcs_;
};

}  // namespace

TEST_F(FRITest, CommitAndVerify) {
  Poly poly = Poly::Random(kMaxDegree);
  base::Uint8VectorBuffer write_buffer;
  SimpleTranscriptWriter<F> writer(std::move(write_buffer));
  ASSERT_TRUE(pcs_.Commit(poly, &writer));

  size_t index = base::Uniform(base::Range<size_t>::Until(kMaxDegree + 1));
  SimpleFRIProof<math::Goldilocks> proof;
  ASSERT_TRUE(pcs_.CreateOpeningProof(index, &proof));

  SimpleTranscriptReader<F> reader(std::move(writer).TakeBuffer());
  reader.buffer().set_buffer_offset(0);
  ASSERT_TRUE(pcs_.VerifyOpeningProof(reader, index, proof));
}

}  // namespace tachyon::crypto
