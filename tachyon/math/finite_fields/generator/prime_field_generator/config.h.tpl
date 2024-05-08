// clang-format off
#include "tachyon/export.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/big_int.h"

namespace %{namespace} {

class TACHYON_EXPORT %{class}Config {
 public:
  constexpr static const char* kName = "%{namespace}::%{class}";

  constexpr static bool kUseMontgomery = %{use_montgomery};
%{if kIsSmallField}
  constexpr static bool kIsSpecialPrime = false;
%{endif kIsSmallField}
%{if !kIsSmallField}
%{if kUseAsm}
#if ARCH_CPU_X86_64
  constexpr static bool kIsSpecialPrime = true;
  constexpr static bool %{flag} = true;
#else
%{endif kUseAsm}
  constexpr static bool kIsSpecialPrime = false;
%{if kUseAsm}
#endif
%{endif kUseAsm}
%{endif !kIsSmallField}

  constexpr static size_t kModulusBits = %{modulus_bits};
  constexpr static BigInt<%{n}> kModulus = BigInt<%{n}>({
    %{modulus}
  });
  constexpr static BigInt<%{n}> kModulusMinusOneDivTwo = BigInt<%{n}>({
    %{modulus_minus_one_div_two}
  });
  constexpr static BigInt<%{n}> kModulusPlusOneDivFour = BigInt<%{n}>({
    %{modulus_plus_one_div_four}
  });
  constexpr static BigInt<%{n}> kTrace = BigInt<%{n}>({
    %{trace}
  });
  constexpr static BigInt<%{n}> kTraceMinusOneDivTwo = BigInt<%{n}>({
    %{trace_minus_one_div_two}
  });
  constexpr static bool kModulusModFourIsThree = %{modulus_mod_four_is_three};
  constexpr static bool kModulusModSixIsOne = %{modulus_mod_six_is_one};
  constexpr static bool kModulusHasSpareBit = %{modulus_has_spare_bit};
%{if !kIsSmallField}
  constexpr static bool kCanUseNoCarryMulOptimization = %{can_use_no_carry_mul_optimization};
  constexpr static BigInt<%{n}> kMontgomeryR = BigInt<%{n}>({
    %{r}
  });
  constexpr static BigInt<%{n}> kMontgomeryR2 = BigInt<%{n}>({
    %{r2}
  });
  constexpr static BigInt<%{n}> kMontgomeryR3 = BigInt<%{n}>({
    %{r3}
  });
  constexpr static uint64_t kInverse64 = UINT64_C(%{inverse64});
%{endif !kIsSmallField}
  constexpr static uint32_t kInverse32 = %{inverse32};

  constexpr static BigInt<%{n}> kOne = BigInt<%{n}>({
    %{one}
  });

  constexpr static bool kHasTwoAdicRootOfUnity = %{has_two_adic_root_of_unity};

  constexpr static bool kHasLargeSubgroupRootOfUnity = %{has_large_subgroup_root_of_unity};

%{if kHasTwoAdicRootOfUnity}
  constexpr static BigInt<%{n}> kSubgroupGenerator = BigInt<%{n}>({
    %{subgroup_generator}
  });
  constexpr static uint32_t kTwoAdicity = %{two_adicity};
  constexpr static BigInt<%{n}> kTwoAdicRootOfUnity = BigInt<%{n}>({
    %{two_adic_root_of_unity}
  });


%{if kHasLargeSubgroupRootOfUnity}
  constexpr static uint32_t kSmallSubgroupBase = %{small_subgroup_base};
  constexpr static uint32_t kSmallSubgroupAdicity = %{small_subgroup_adicity};
  constexpr static BigInt<%{n}> kLargeSubgroupRootOfUnity = BigInt<%{n}>({
    %{large_subgroup_root_of_unity}
  });
%{endif kHasLargeSubgroupRootOfUnity}
%{endif kHasTwoAdicRootOfUnity}

%{if kIsSmallField}
  constexpr static uint32_t AddMod(uint32_t a, uint32_t b) {
    // NOTE(chokobole): This assumes that the 2m - 2 < 2³², where m is modulus.
    return Reduce(a + b);
  }

  constexpr static uint32_t SubMod(uint32_t a, uint32_t b) {
    // NOTE(chokobole): This assumes that the 2m - 2 < 2³², where m is modulus.
    return Reduce(a + static_cast<uint32_t>(kModulus[0]) - b);
  }

  constexpr static uint32_t Reduce(uint32_t v) {
    %{reduce32}
  }

  constexpr static uint32_t Reduce(uint64_t v) {
    %{reduce64}
  }

  constexpr static uint32_t ToMontgomery(uint32_t v) {
    return (uint64_t{v} << 32) % static_cast<uint32_t>(kModulus[0]);
  }

  constexpr static uint32_t FromMontgomery(uint64_t v) {
    constexpr uint64_t kMask = (uint64_t{1} << 32) - 1;
    uint64_t t = (v * kInverse32) & kMask;
    uint64_t u = t * kModulus[0];
    uint32_t ret = (v - u) >> 32;
    if (v < u) {
      return ret + static_cast<uint32_t>(kModulus[0]);
    } else {
      return ret;
    }
  }
%{endif kIsSmallField}
};

}  // namespace %{namespace}
// clang-format on
