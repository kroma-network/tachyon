#include <iostream>

#include "absl/strings/substitute.h"

// clang-format off
#include "benchmark/simple_reporter.h"
#include "benchmark/fri/fri_config.h"
#include "benchmark/fri/fri_runner.h"
// clang-format on
#include "tachyon/base/profiler.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/crypto/challenger/duplex_challenger.h"
#include "tachyon/crypto/commitments/fri/fri_config.h"
#include "tachyon/crypto/commitments/fri/two_adic_fri.h"
#include "tachyon/crypto/commitments/fri/two_adic_multiplicative_coset.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/extension_field_merkle_tree_mmcs.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree_mmcs.h"
#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"

namespace tachyon::benchmark {

extern "C" tachyon_baby_bear* run_fri_plonky3_baby_bear(
    const tachyon_baby_bear* data, const size_t* raw_degrees,
    size_t num_of_degrees, size_t batch_size, uint32_t log_blowup,
    uint64_t* duration);

template <typename Commitment>
void CheckResult(bool check_results, const Commitment& tachyon_result,
                 const Commitment& vendor_result) {
  if (check_results) {
    CHECK_EQ(tachyon_result, vendor_result) << "Results not matched";
  }
}

void Run(const FRIConfig& config) {
  constexpr size_t kRate = 8;
  constexpr size_t kChunk = 8;
  constexpr size_t kN = 2;

  using F = math::BabyBear;
  using ExtF = math::BabyBear4;
  using PackedF = math::PackedBabyBear;
  using ExtPackedF = math::PackedBabyBear4;
  using Poseidon2 = crypto::Poseidon2Sponge<crypto::Poseidon2ExternalMatrix<
      crypto::Poseidon2HorizenExternalMatrix<F>>>;
  using PackedPoseidon2 =
      crypto::Poseidon2Sponge<crypto::Poseidon2ExternalMatrix<
          crypto::Poseidon2HorizenExternalMatrix<PackedF>>>;
  using MyHasher = crypto::PaddingFreeSponge<Poseidon2, kRate, kChunk>;
  using MyPackedHasher =
      crypto::PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
  using MyCompressor = crypto::TruncatedPermutation<Poseidon2, kChunk, kN>;
  using MyPackedCompressor =
      crypto::TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
  using MMCS =
      crypto::FieldMerkleTreeMMCS<F, MyHasher, MyPackedHasher, MyCompressor,
                                  MyPackedCompressor, kChunk>;
  using ExtMMCS =
      crypto::FieldMerkleTreeMMCS<ExtF, MyHasher, MyPackedHasher, MyCompressor,
                                  MyPackedCompressor, kChunk>;
  using ChallengeMMCS = crypto::ExtensionFieldMerkleTreeMMCS<ExtF, ExtMMCS>;
  using Challenger = crypto::DuplexChallenger<Poseidon2, 16, kRate>;
  using MyPCS = crypto::TwoAdicFRI<ExtF, MMCS, ChallengeMMCS, Challenger>;

  PackedF::Init();
  ExtPackedF::Init();

  crypto::Poseidon2Config<F> poseidon2_config =
      crypto::Poseidon2Config<F>::CreateCustom(
          15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftArray<15>());
  Poseidon2 sponge(poseidon2_config);
  MyHasher hasher(sponge);
  MyCompressor compressor(sponge);

  crypto::Poseidon2Config<PackedF> packed_config =
      crypto::Poseidon2Config<PackedF>::CreateCustom(
          15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftArray<15>());
  PackedPoseidon2 packed_sponge(packed_config);
  MyPackedHasher packed_hasher(packed_sponge);
  MyPackedCompressor packed_compressor(std::move(packed_sponge));
  MMCS mmcs(hasher, packed_hasher, compressor, packed_compressor);

  ChallengeMMCS challenge_mmcs(
      ExtMMCS(std::move(hasher), std::move(packed_hasher),
              std::move(compressor), std::move(packed_compressor)));

  crypto::FRIConfig<ChallengeMMCS> fri_config{config.log_blowup(), 10, 8,
                                              challenge_mmcs};
  MyPCS pcs = MyPCS(std::move(mmcs), std::move(fri_config));

  SimpleReporter reporter;
  std::string name;
  name = absl::Substitute("FRI Benchmark (b: $0, l: $1)", config.batch_size(),
                          config.log_blowup());
  reporter.set_title(name);
  reporter.set_x_label("Max Exponent");
  reporter.set_column_labels(base::Map(
      config.exponents(),
      [](uint32_t exponent) { return base::NumberToString(exponent); }));

  FRIRunner runner(reporter, config, pcs);

  std::vector<size_t> degrees = config.GetDegrees();

  reporter.AddVendor(Vendor::Tachyon());
  for (const Vendor vendor : config.vendors()) {
    reporter.AddVendor(vendor);
  }

  for (size_t degree : degrees) {
    math::RowMajorMatrix<F> input =
        math::RowMajorMatrix<F>::Random(degree, config.batch_size());

    F tachyon_commit = runner.Run(Vendor::Tachyon(), input);
    for (const Vendor vendor : config.vendors()) {
      if (vendor.value() == Vendor::kPlonky3) {
        F vendor_commit =
            runner.RunExternal(vendor, run_fri_plonky3_baby_bear, input);
        CheckResult(config.check_results(), tachyon_commit, vendor_commit);
      } else {
        NOTREACHED();
      }
    }
  }

  reporter.Show();
}

int RealMain(int argc, char** argv) {
  base::FilePath tmp_file;
  CHECK(base::GetTempDir(&tmp_file));
  tmp_file = tmp_file.Append("fri_benchmark.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  FRIConfig config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  Run(config);
  return 0;
}

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
