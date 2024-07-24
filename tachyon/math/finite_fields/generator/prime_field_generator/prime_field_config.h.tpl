// clang-format off
#include "tachyon/export.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/big_int.h"

namespace %{namespace} {

class TACHYON_EXPORT %{class}Config {
 public:
  constexpr static const char* kName = "%{namespace}::%{class}";

  constexpr static bool kUseMontgomery = %{use_montgomery};
%{if kUseAsm}
#if ARCH_CPU_X86_64
  constexpr static bool kUseAsm = %{use_asm};
  constexpr static bool %{asm_flag} = true;
#else
  constexpr static bool kUseAsm = false;
#endif
%{endif kUseAsm}
%{if !kUseAsm}
  constexpr static bool kUseAsm = false;
%{endif !kUseAsm}

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
  constexpr static uint32_t kInverse32 = %{inverse32};

  constexpr static BigInt<%{n}> kOne = BigInt<%{n}>({
    %{one}
  });

  constexpr static BigInt<%{n}> kMinusOne = BigInt<%{n}>({
    %{minus_one}
  });

  constexpr static bool kHasTwoAdicRootOfUnity = %{has_two_adic_root_of_unity};

  constexpr static bool kHasLargeSubgroupRootOfUnity = %{has_large_subgroup_root_of_unity};

%{if kHasTwoAdicRootOfUnity}
  static BigInt<%{n}> kSubgroupGenerator;
  constexpr static uint32_t kTwoAdicity = %{two_adicity};
  static BigInt<%{n}> kTwoAdicRootOfUnity;

%{if kHasLargeSubgroupRootOfUnity}
  constexpr static uint32_t kSmallSubgroupBase = %{small_subgroup_base};
  constexpr static uint32_t kSmallSubgroupAdicity = %{small_subgroup_adicity};
  static BigInt<%{n}> kLargeSubgroupRootOfUnity;
%{endif kHasLargeSubgroupRootOfUnity}
%{endif kHasTwoAdicRootOfUnity}

  static void Init() {
%{if kHasTwoAdicRootOfUnity}
    kSubgroupGenerator = BigInt<%{n}>({
      %{subgroup_generator}
    });
    kTwoAdicRootOfUnity = BigInt<%{n}>({
      %{two_adic_root_of_unity}
    });
%{if kHasLargeSubgroupRootOfUnity}
    kLargeSubgroupRootOfUnity = BigInt<%{n}>({
      %{large_subgroup_root_of_unity}
    });
%{endif kHasLargeSubgroupRootOfUnity}
%{endif kHasTwoAdicRootOfUnity}
  }
};

}  // namespace %{namespace}
// clang-format on
