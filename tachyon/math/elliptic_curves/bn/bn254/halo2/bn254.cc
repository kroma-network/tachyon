#include "tachyon/math/elliptic_curves/bn/bn254/halo2/bn254.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::math::halo2 {

void OverrideSubgroupGenerator() {
  // See
  // https://github.com/kroma-network/halo2curves/blob/c0ac193/src/bn256/fr.rs#L68-L70.
  bn254::Fr::Config::kSubgroupGenerator = BigInt<4>({
      UINT64_C(3483395353741361115),
      UINT64_C(3494632259903994625),
      UINT64_C(6657987792994187913),
      UINT64_C(108272644256946680),
  });
  // See
  // https://github.com/kroma-network/halo2curves/blob/c0ac193/src/bn256/fr.rs#L74-L83.
  bn254::Fr::Config::kTwoAdicRootOfUnity = BigInt<4>({
      UINT64_C(10822932506504462008),
      UINT64_C(10978899855858987673),
      UINT64_C(12888607242213977304),
      UINT64_C(2119232853909229097),
  });
  bn254::Fr::Config::kLargeSubgroupRootOfUnity = BigInt<4>({
      UINT64_C(9055134861510678988),
      UINT64_C(3166494206591041163),
      UINT64_C(11983946130272577941),
      UINT64_C(1690279781341100183),
  });
}

ScopedSubgroupGeneratorOverrider::ScopedSubgroupGeneratorOverrider() {
  subgroup_generator = bn254::Fr::Config::kSubgroupGenerator;
  two_adic_root_of_unity = bn254::Fr::Config::kTwoAdicRootOfUnity;
  large_subgroup_root_of_unity = bn254::Fr::Config::kLargeSubgroupRootOfUnity;

  OverrideSubgroupGenerator();
}

ScopedSubgroupGeneratorOverrider::~ScopedSubgroupGeneratorOverrider() {
  bn254::Fr::Config::kSubgroupGenerator = subgroup_generator;
  bn254::Fr::Config::kTwoAdicRootOfUnity = two_adic_root_of_unity;
  bn254::Fr::Config::kLargeSubgroupRootOfUnity = large_subgroup_root_of_unity;
}

}  // namespace tachyon::math::halo2
