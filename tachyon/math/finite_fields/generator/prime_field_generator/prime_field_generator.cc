#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"
#include "tachyon/math/finite_fields/prime_field_util.h"

namespace tachyon {

struct ModulusInfo {
  bool modulus_has_spare_bit;
  bool can_use_no_carry_mul_optimization;
  mpz_class r;
  mpz_class r2;
  uint64_t inverse64;
  uint32_t inverse32;

  template <size_t N>
  static ModulusInfo From(const mpz_class& m_in) {
    math::BigInt<N> m;
    math::gmp::CopyLimbs(m_in, m.limbs);

    ModulusInfo ret;
    ret.modulus_has_spare_bit = math::Modulus<N>::HasSpareBit(m);
    ret.can_use_no_carry_mul_optimization =
        math::Modulus<N>::CanUseNoCarryMulOptimization(m);
    math::BigInt<N> r = math::Modulus<N>::MontgomeryR(m);
    math::gmp::WriteLimbs(r.limbs, N, &ret.r);
    math::BigInt<N> r2 = math::Modulus<N>::MontgomeryR2(m);
    math::gmp::WriteLimbs(r2.limbs, N, &ret.r2);
    ret.inverse64 = math::Modulus<N>::template Inverse<uint64_t>(m);
    ret.inverse32 = math::Modulus<N>::template Inverse<uint32_t>(m);
    return ret;
  }

  static ModulusInfo From(const mpz_class& m) {
    size_t limb_size = math::gmp::GetLimbSize(m);
    switch (limb_size) {
      case 1:
        return From<1>(m);
      case 2:
        return From<2>(m);
      case 3:
        return From<3>(m);
      case 4:
        return From<4>(m);
      case 5:
        return From<5>(m);
      case 6:
        return From<6>(m);
      case 7:
        return From<7>(m);
      case 8:
        return From<8>(m);
      case 9:
        return From<9>(m);
    }
    NOTREACHED();
    return {};
  }
};

struct GenerationConfig : public build::CcWriter {
  std::string ns_name;
  std::string class_name;
  std::string modulus;
  std::string subgroup_generator;
  std::string small_subgroup_base;
  std::string small_subgroup_adicity;
  std::string hdr_include_override;
  std::string special_prime_override;

  int GenerateConfigHdr() const;
  int GenerateConfigCpp() const;
  int GenerateConfigGpuHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  // clang-format off
  std::vector<std::string> tpl = {
      "#include \"tachyon/export.h\"",
      "#include \"tachyon/math/finite_fields/prime_field.h\"",
      "#if defined(TACHYON_GMP_BACKEND)",
      "#include \"tachyon/math/finite_fields/prime_field_gmp.h\"",
      "#endif  // defined(TACHYON_GMP_BACKEND)",
      "",
      "namespace %{namespace} {",
      "",
      "class TACHYON_EXPORT %{class}Config {",
      " public:",
      "  constexpr static bool kIsSpecialPrime = false;",
      "",
      "  constexpr static size_t kModulusBits = %{modulus_bits};",
      "  constexpr static BigInt<%{n}> kModulus = BigInt<%{n}>({",
      "    %{modulus}",
      "  });",
      "  constexpr static bool kModulusHasSpareBit = %{modulus_has_spare_bit};",
      "  constexpr static bool kCanUseNoCarryMulOptimization = "
      "%{can_use_no_carry_mul_optimization};",
      "  constexpr static BigInt<%{n}> kMontgomeryR = BigInt<%{n}>({",
      "    %{r}",
      "  });",
      "  constexpr static BigInt<%{n}> kMontgomeryR2 = BigInt<%{n}>({",
      "    %{r2}",
      "  });",
      "  constexpr static uint64_t kInverse64 = UINT64_C(%{inverse64});",
      "  constexpr static uint32_t kInverse32 = %{inverse32};",
      "",
      "  constexpr static BigInt<%{n}> kOne = BigInt<%{n}>({",
      "    %{one_mont_form}",
      "  });",
      "",
      "  constexpr static bool kHasTwoAdicRootOfUnity = false;",
      "",
      "  constexpr static bool kHasLargeSubgroupRootOfUnity = false;",
      "",
      "  static void Init();",
      "};",
      "",
      "using %{class} = PrimeField<%{class}Config>;",
      "#if defined(TACHYON_GMP_BACKEND)",
      "using %{class}Gmp = PrimeFieldGmp<%{class}Config>;",
      "#endif  // defined(TACHYON_GMP_BACKEND)",
      "",
      "}  // namespace %{namespace}",
  };
  // clang-format on

  if (!hdr_include_override.empty()) {
    for (size_t i = 0; i < tpl.size(); ++i) {
      size_t idx =
          tpl[i].find("#include \"tachyon/math/finite_fields/prime_field.h\"");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + i;
        tpl.erase(it);
        tpl.insert(it, hdr_include_override);
        break;
      }
    }
  }

  if (!special_prime_override.empty()) {
    for (size_t i = 0; i < tpl.size(); ++i) {
      size_t idx = tpl[i].find("kIsSpecialPrime");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + i;
        tpl.erase(it, it + 1);
        tpl.insert(it, special_prime_override);
        break;
      }
    }
  }

  mpz_class m = math::gmp::FromDecString(modulus);
  auto it = math::BitIteratorBE<mpz_class>::begin(&m, true);
  auto end = math::BitIteratorBE<mpz_class>::end(&m);
  size_t num_bits = 0;
  while (it != end) {
    ++it;
    ++num_bits;
  }
  size_t n = math::gmp::GetLimbSize(m);

  ModulusInfo modulus_info = ModulusInfo::From(m);

  mpz_class trace = math::ComputeTrace(2, m - mpz_class(1));
  if (!subgroup_generator.empty()) {
    uint32_t two_adicity = math::ComputeAdicity(2, m - mpz_class(1));
    mpz_class two_adic_root_of_unity;
    mpz_powm(two_adic_root_of_unity.get_mpz_t(),
             math::gmp::FromDecString(subgroup_generator).get_mpz_t(),
             trace.get_mpz_t(), m.get_mpz_t());

    std::vector<std::string> lines;
    // clang-format off
    lines.push_back("  constexpr static bool kHasTwoAdicRootOfUnity = true;");
    lines.push_back("  constexpr static BigInt<1> kSubgroupGenerator = BigInt<1>({");
    lines.push_back(absl::Substitute("      UINT64_C($0)", subgroup_generator));
    lines.push_back("  });");
    lines.push_back(absl::Substitute("  constexpr static uint32_t kTwoAdicity = $0;", two_adicity));
    lines.push_back("  constexpr static BigInt<%{n}> kTwoAdicRootOfUnity = BigInt<%{n}>({");
    lines.push_back(absl::Substitute("    $0", math::MpzClassToMontString(two_adic_root_of_unity, m)));
    lines.push_back("  });");
    // clang-format on

    for (size_t i = 0; i < tpl.size(); ++i) {
      size_t idx =
          tpl[i].find("constexpr static bool kHasTwoAdicRootOfUnity = false;");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + i;
        tpl.erase(it);
        tpl.insert(it, lines.begin(), lines.end());
        break;
      }
    }
  }

  if (!small_subgroup_base.empty()) {
    CHECK(!small_subgroup_adicity.empty());

    mpz_class small_subgroup_base_pow_adicity;
    mpz_powm(small_subgroup_base_pow_adicity.get_mpz_t(),
             math::gmp::FromDecString(small_subgroup_base).get_mpz_t(),
             math::gmp::FromDecString(small_subgroup_adicity).get_mpz_t(),
             m.get_mpz_t());
    mpz_class remaining_subgroup_size;
    mpz_div(remaining_subgroup_size.get_mpz_t(), trace.get_mpz_t(),
            small_subgroup_base_pow_adicity.get_mpz_t());
    mpz_class large_subgroup_root_of_unity;
    mpz_powm(large_subgroup_root_of_unity.get_mpz_t(),
             math::gmp::FromDecString(subgroup_generator).get_mpz_t(),
             remaining_subgroup_size.get_mpz_t(), m.get_mpz_t());

    std::vector<std::string> lines;
    // clang-format off
    lines.push_back("  constexpr static bool kHasLargeSubgroupRootOfUnity = true;");
    lines.push_back(absl::Substitute("  constexpr static uint32_t kSmallSubgroupBase = $0;", small_subgroup_base));
    lines.push_back(absl::Substitute("  constexpr static uint32_t kSmallSubgroupAdicity = $0;", small_subgroup_adicity));
    lines.push_back("  constexpr static BigInt<%{n}> kLargeSubgroupRootOfUnity = BigInt<%{n}>({");
    lines.push_back(absl::Substitute("    $0", math::MpzClassToMontString(large_subgroup_root_of_unity, m)));
    lines.push_back("  });");
    // clang-format on

    for (size_t i = 0; i < tpl.size(); ++i) {
      size_t idx = tpl[i].find(
          "constexpr static bool kHasLargeSubgroupRootOfUnity = false;");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + i;
        tpl.erase(it);
        tpl.insert(it, lines.begin(), lines.end());
        break;
      }
    }
  } else {
    CHECK(small_subgroup_adicity.empty());
  }

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{namespace}", ns_name},
          {"%{class}", class_name},
          {"%{modulus_bits}", base::NumberToString(num_bits)},
          {"%{n}", base::NumberToString(n)},
          {"%{modulus}", math::MpzClassToString(m)},
          {"%{modulus_has_spare_bit}",
           base::BoolToString(modulus_info.modulus_has_spare_bit)},
          {"%{can_use_no_carry_mul_optimization}",
           base::BoolToString(modulus_info.can_use_no_carry_mul_optimization)},
          {"%{r}", math::MpzClassToString(modulus_info.r)},
          {"%{r2}", math::MpzClassToString(modulus_info.r2)},
          {"%{inverse64}", base::NumberToString(modulus_info.inverse64)},
          {"%{inverse32}", base::NumberToString(modulus_info.inverse32)},
          {"%{one_mont_form}", math::MpzClassToMontString(mpz_class(1), m)},
      });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateConfigCpp() const {
  std::string_view tpl[] = {
      "namespace %{namespace} {",
      "",
      "// static",
      "void %{class}Config::Init() {",
      "#if defined(TACHYON_GMP_BACKEND)",
      "  %{class}Gmp::Init();",
      "#endif  // defined(TACHYON_GMP_BACKEND)",
      "}",
      "",
      "}  // namespace %{namespace}",
  };
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content =
      absl::StrReplaceAll(tpl_content, {
                                           {"%{namespace}", ns_name},
                                           {"%{class}", class_name},
                                       });
  return WriteSrc(content);
}

int GenerationConfig::GenerateConfigGpuHdr() const {
  std::string_view tpl[] = {
      "#include \"%{header_path}\"",
      "",
      "#include \"tachyon/math/finite_fields/prime_field_gpu.h\"",
      "",
      "namespace %{namespace} {",
      "",
      "using %{class}Gpu = PrimeFieldGpu<%{class}Config>;",
      "",
      "}  // namespace %{namespace}",
  };
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{header_path}", math::ConvertToCpuHdr(GetHdrPath()).value()},
          {"%{namespace}", ns_name},
          {"%{class}", class_name},
      });
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/finite_fields/prime_field_field_generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.ns_name)
      .set_long_name("--namespace")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.class_name).set_long_name("--class");
  parser.AddFlag<base::StringFlag>(&config.modulus)
      .set_long_name("--modulus")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.subgroup_generator)
      .set_long_name("--subgroup_generator");
  parser.AddFlag<base::StringFlag>(&config.small_subgroup_base)
      .set_long_name("--small_subgroup_base");
  parser.AddFlag<base::StringFlag>(&config.small_subgroup_adicity)
      .set_long_name("--small_subgroup_adicity");
  parser.AddFlag<base::StringFlag>(&config.hdr_include_override)
      .set_long_name("--hdr_include_override");
  parser.AddFlag<base::StringFlag>(&config.special_prime_override)
      .set_long_name("--special_prime_override");

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "_gpu.h")) {
    return config.GenerateConfigGpuHdr();
  } else if (base::EndsWith(config.out.value(), ".h")) {
    return config.GenerateConfigHdr();
  } else if (base::EndsWith(config.out.value(), ".cc")) {
    return config.GenerateConfigCpp();
  } else {
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
