#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree.h"
#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/geometry/dimensions.h"

#if TACHYON_CUDA
#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2_holder.h"
#endif

namespace tachyon::crypto {

constexpr size_t kRate = 8;
constexpr size_t kChunk = 8;
constexpr size_t kN = 2;

using F = math::BabyBear;
using PackedF = math::PackedBabyBear;
using Params = Poseidon2Params<Poseidon2Vendor::kPlonky3,
                               Poseidon2Vendor::kPlonky3, F, 15, 7>;
using PackedParams = Poseidon2Params<Poseidon2Vendor::kPlonky3,
                                     Poseidon2Vendor::kPlonky3, PackedF, 15, 7>;
using Poseidon2 = Poseidon2Sponge<Params>;
using PackedPoseidon2 = Poseidon2Sponge<PackedParams>;
using MyHasher = PaddingFreeSponge<Poseidon2, kRate, kChunk>;
using MyPackedHasher = PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
using MyCompressor = TruncatedPermutation<Poseidon2, kChunk, kN>;
using MyPackedCompressor = TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
using Tree = FieldMerkleTree<F, kChunk>;

namespace {

class FieldMerkleTreeGpuTest : public math::FiniteFieldTest<PackedF> {
 protected:
  MyHasher hasher_;
  MyCompressor compressor_;
  MyPackedHasher packed_hasher_;
  MyPackedCompressor packed_compressor_;
};

}  // namespace

TEST_F(FieldMerkleTreeGpuTest, CommitGpu) {
  struct Config {
    size_t num;
    math::Dimensions dimensions;
  };

  struct {
    std::vector<Config> configs;
  } tests[] = {
      // NOTE(chokobole): Icicle supports only rows that are powers of 2.
      {{{4, {8, 1024}}, {5, {8, 128}}, {6, {8, 8}}}},
      {
          {{1, {1, 32}},
           {1, {2, 32}},
           {1, {3, 32}},
           {1, {4, 32}},
           {1, {5, 32}},
           {1, {6, 32}},
           {1, {7, 32}},
           {1, {8, 32}},
           {1, {9, 32}},
           {1, {10, 32}}},
      },
  };

  IciclePoseidon2Holder<F> poseidon2_holder =
      IciclePoseidon2Holder<F>::Create<kRate>(hasher_.derived().config);
  IcicleMMCSHolder<F> mmcs_holder = IcicleMMCSHolder<F>::Create<kRate>(
      poseidon2_holder.get()->impl(), poseidon2_holder.get()->impl());

  for (const auto& test : tests) {
    std::vector<math::RowMajorMatrix<F>> matrices;
    for (size_t i = 0; i < test.configs.size(); ++i) {
      math::Dimensions dimensions = test.configs[i].dimensions;
      for (size_t j = 0; j < test.configs[i].num; ++j) {
        matrices.push_back(math::RowMajorMatrix<F>::Random(dimensions.height,
                                                           dimensions.width));
      }
    }

    std::vector<math::RowMajorMatrix<F>> matrices_tmp = matrices;
    Tree tree_cpu = Tree::MaybeBuildOwnedGpu(hasher_, packed_hasher_,
                                             compressor_, packed_compressor_,
                                             std::move(matrices_tmp), nullptr);
    Tree tree_gpu = Tree::MaybeBuildOwnedGpu(hasher_, packed_hasher_,
                                             compressor_, packed_compressor_,
                                             std::move(matrices), &mmcs_holder);
    EXPECT_EQ(tree_cpu.GetRoot(), tree_gpu.GetRoot());
    EXPECT_EQ(tree_cpu.leaves(), tree_gpu.leaves());
    EXPECT_EQ(tree_cpu.digest_layers(), tree_gpu.digest_layers());
  }
}

}  // namespace tachyon::crypto
