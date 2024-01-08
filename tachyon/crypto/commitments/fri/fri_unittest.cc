// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#include "tachyon/crypto/commitments/fri/fri.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/simple_binary_merkle_tree_storage.h"
#include "tachyon/crypto/transcripts/simple_transcript.h"
#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks.h"
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

class SimpleFRIStorage : public FRIStorage<math::Goldilocks> {
 public:
  using Layer =
      SimpleBinaryMerkleTreeStorage<math::Goldilocks, math::Goldilocks>;

  const std::vector<Layer>& layers() const { return layers_; }

  std::vector<math::Goldilocks> GetRoots() const {
    return base::Map(layers_,
                     [](const Layer& layer) { return layer.GetRoot(); });
  }

  // FRIStorage<math::Goldilocks> methods
  void Allocate(size_t size) override { layers_.resize(size); }
  BinaryMerkleTreeStorage<math::Goldilocks, math::Goldilocks>* GetLayer(
      size_t index) override {
    return &layers_[index];
  }

 private:
  std::vector<Layer> layers_;
};

class FRITest : public testing::Test {
 public:
  constexpr static size_t K = 3;
  constexpr static size_t N = size_t{1} << K;
  constexpr static size_t kMaxDegree = N - 1;

  using PCS = FRI<math::Goldilocks, kMaxDegree,
                  SimpleTranscriptReader<math::Goldilocks>,
                  SimpleTranscriptWriter<math::Goldilocks>>;
  using F = PCS::Field;
  using Poly = PCS::Poly;
  using Domain = PCS::Domain;
  using TranscriptReader = PCS::TranscriptReader;
  using TranscriptWriter = PCS::TranscriptWriter;

  static void SetUpTestSuite() { math::Goldilocks::Init(); }

  void SetUp() override {
    domain_ = Domain::Create(N);
    pcs_ = PCS(domain_.get(), &storage_, &hasher_);
  }

 protected:
  std::unique_ptr<Domain> domain_;
  SimpleFRIStorage storage_;
  SimpleHasher hasher_;
  PCS pcs_;
};

}  // namespace

TEST_F(FRITest, CommitAndVerify) {
  Poly poly = Poly::Random(kMaxDegree);
  base::Uint8VectorBuffer write_buffer;
  TranscriptWriter writer(std::move(write_buffer));
  std::vector<F> roots;
  ASSERT_TRUE(pcs_.Commit(poly, &roots, &writer));
  roots.pop_back();
  EXPECT_EQ(roots, storage_.GetRoots());

  size_t index = base::Uniform(base::Range<size_t>::Until(kMaxDegree + 1));
  FRIProof<math::Goldilocks> proof;
  ASSERT_TRUE(pcs_.CreateOpeningProof(index, &proof));

  TranscriptReader reader(std::move(writer).TakeBuffer());
  reader.buffer().set_buffer_offset(0);
  ASSERT_TRUE(pcs_.VerifyOpeningProof(index, proof, reader));
}

}  // namespace tachyon::crypto
