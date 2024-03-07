#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
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
  mpz_class r3;
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
    math::BigInt<N> r3 = math::Modulus<N>::MontgomeryR3(m);
    math::gmp::WriteLimbs(r3.limbs, N, &ret.r3);
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
  std::string flag;
  base::FilePath x86_hdr_tpl_path;
  base::FilePath fail_hdr_tpl_path;
  base::FilePath fail_src_tpl_path;
  std::string subgroup_generator;
  std::string small_subgroup_base;
  std::string small_subgroup_adicity;

  int GenerateFailHdr() const;
  int GenerateFailSrc() const;
  int GeneratePrimeFieldX86Hdr() const;
  int GenerateConfigHdr() const;
  int GenerateCpuHdr() const;
  int GenerateGpuHdr() const;

  std::string GetPrefix() const {
    return absl::Substitute("$0_$1",
                            absl::StrReplaceAll(ns_name, {{"::", "_"}}),
                            base::ToLowerASCII(class_name));
  }
};

int GenerationConfig::GenerateFailHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(fail_hdr_tpl_path, &tpl_content));

  std::string content =
      absl::StrReplaceAll(tpl_content, {{"%{prefix}", GetPrefix()}});
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateFailSrc() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(fail_src_tpl_path, &tpl_content));

  std::string content =
      absl::StrReplaceAll(tpl_content, {{"%{prefix}", GetPrefix()}});
  return WriteSrc(content);
}

int GenerationConfig::GeneratePrimeFieldX86Hdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(x86_hdr_tpl_path, &tpl_content));

  mpz_class m = math::gmp::FromDecString(modulus);
  size_t n = math::gmp::GetLimbSize(m);

  std::string content =
      absl::StrReplaceAll(tpl_content, {
                                           {"%{prefix}", GetPrefix()},
                                           {"%{n}", base::NumberToString(n)},
                                           {"%{flag}", flag},
                                       });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateConfigHdr() const {
  // clang-format off
  std::vector<std::string> tpl = {
      "#include \"tachyon/export.h\"",
      "#include \"tachyon/build/build_config.h\"",
      "#include \"tachyon/math/base/big_int.h\"",
      "",
      "namespace %{namespace} {",
      "",
      "class TACHYON_EXPORT %{class}Config {",
      " public:",
      "  constexpr static const char* kName = \"%{namespace}::%{class}\";",
      "#if ARCH_CPU_X86_64",
      "  constexpr static bool kIsSpecialPrime = true;",
      "  constexpr static bool %{flag} = true;",
      "#else",
      "  constexpr static bool kIsSpecialPrime = false;",
      "#endif",
      "",
      "  constexpr static size_t kModulusBits = %{modulus_bits};",
      "  constexpr static BigInt<%{n}> kModulus = BigInt<%{n}>({",
      "    %{modulus}",
      "  });",
      "  constexpr static BigInt<%{n}> kModulusMinusOneDivTwo = BigInt<%{n}>({",
      "    %{modulus_minus_one_div_two}",
      "  });",
      "  constexpr static BigInt<%{n}> kModulusPlusOneDivFour = BigInt<%{n}>({",
      "    %{modulus_plus_one_div_four}",
      "  });",
      "  constexpr static BigInt<%{n}> kTrace = BigInt<%{n}>({",
      "    %{trace}",
      "  });",
      "  constexpr static BigInt<%{n}> kTraceMinusOneDivTwo = BigInt<%{n}>({",
      "    %{trace_minus_one_div_two}",
      "  });",
      "  constexpr static bool kModulusModFourIsThree = %{modulus_mod_four_is_three};",
      "  constexpr static bool kModulusModSixIsOne = %{modulus_mod_six_is_one};",
      "  constexpr static bool kModulusHasSpareBit = %{modulus_has_spare_bit};",
      "  constexpr static bool kCanUseNoCarryMulOptimization = "
      "%{can_use_no_carry_mul_optimization};",
      "  constexpr static BigInt<%{n}> kMontgomeryR = BigInt<%{n}>({",
      "    %{r}",
      "  });",
      "  constexpr static BigInt<%{n}> kMontgomeryR2 = BigInt<%{n}>({",
      "    %{r2}",
      "  });",
      "  constexpr static BigInt<%{n}> kMontgomeryR3 = BigInt<%{n}>({",
      "    %{r3}",
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
      "};",
      "",
      "}  // namespace %{namespace}",
  };
  // clang-format on

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
  mpz_class subgroup_generator_mpz;
  if (!subgroup_generator.empty()) {
    // 1) 2ˢ * t = m - 1,
    // According to Fermat's Little Theorem, the following equations hold:
    // 2) g^(m - 1) = 1 (mod m)
    // 3) g^(2ˢ * t) = 1 (mod m)
    // 4) gᵗ^(2ˢ) = 1 (mod m)
    // Where subgroup_generator = g, two_adicity = s, trace = t and
    // two_adic_root_of_unity = gᵗ.
    uint32_t two_adicity = math::ComputeAdicity(2, m - mpz_class(1));
    mpz_class two_adic_root_of_unity;
    subgroup_generator_mpz = math::gmp::FromDecString(subgroup_generator);
    mpz_powm(two_adic_root_of_unity.get_mpz_t(),
             subgroup_generator_mpz.get_mpz_t(), trace.get_mpz_t(),
             m.get_mpz_t());

    std::vector<std::string> lines;
    // clang-format off
    lines.push_back("  constexpr static bool kHasTwoAdicRootOfUnity = true;");
    lines.push_back("  constexpr static BigInt<%{n}> kSubgroupGenerator = BigInt<%{n}>({");
    lines.push_back(absl::Substitute("    $0", math::MpzClassToMontString(subgroup_generator_mpz, m)));
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

    if (!small_subgroup_base.empty()) {
      CHECK(!small_subgroup_adicity.empty());
      // 5) gᵗ^(2ˢ) = 1 (mod m)
      // 6) g^(t / bᵃ)^(2ˢ * bᵃ) = 1 (mod m)
      // Where small_subgroup_base = b, small_subgroup_adicity = a,
      // remaining_subgroup_size = t / bᵃ
      // and large_subgroup_root_of_unity = g^(t / bᵃ).
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
               subgroup_generator_mpz.get_mpz_t(),
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
  } else {
    CHECK(small_subgroup_base.empty());
    CHECK(small_subgroup_adicity.empty());
  }

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{namespace}", ns_name},
          {"%{class}", class_name},
          {"%{flag}", flag},
          {"%{modulus_bits}", base::NumberToString(num_bits)},
          {"%{n}", base::NumberToString(n)},
          {"%{modulus}", math::MpzClassToString(m)},
          {"%{modulus_minus_one_div_two}",
           math::MpzClassToString((m - mpz_class(1)) / mpz_class(2))},
          {"%{modulus_plus_one_div_four}",
           math::MpzClassToString((m + mpz_class(1)) / mpz_class(4))},
          {"%{trace}", math::MpzClassToString(trace)},
          {"%{trace_minus_one_div_two}",
           math::MpzClassToString((trace - mpz_class(1)) / mpz_class(2))},
          {"%{modulus_mod_four_is_three}",
           base::BoolToString(m % mpz_class(4) == mpz_class(3))},
          {"%{modulus_mod_six_is_one}",
           base::BoolToString(m % mpz_class(6) == mpz_class(1))},
          {"%{modulus_has_spare_bit}",
           base::BoolToString(modulus_info.modulus_has_spare_bit)},
          {"%{can_use_no_carry_mul_optimization}",
           base::BoolToString(modulus_info.can_use_no_carry_mul_optimization)},
          {"%{r}", math::MpzClassToString(modulus_info.r)},
          {"%{r2}", math::MpzClassToString(modulus_info.r2)},
          {"%{r3}", math::MpzClassToString(modulus_info.r3)},
          {"%{inverse64}", base::NumberToString(modulus_info.inverse64)},
          {"%{inverse32}", base::NumberToString(modulus_info.inverse32)},
          {"%{one_mont_form}", math::MpzClassToMontString(mpz_class(1), m)},
      });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateCpuHdr() const {
  std::string_view tpl[] = {
      "#include \"%{config_header_path}\"",
      "",
      "#if ARCH_CPU_X86_64",
      "#include \"%{prime_field_x86_hdr}\"",
      "#else",
      "#include \"tachyon/math/finite_fields/prime_field_generic.h\"",
      "#endif",
      "",
      "namespace %{namespace} {",
      "",
      "using %{class} = PrimeField<%{class}Config>;",
      "",
      "}  // namespace %{namespace}",
  };
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  base::FilePath hdr_path = GetHdrPath();
  base::FilePath basename = hdr_path.BaseName().RemoveExtension();
  base::FilePath config_header_path =
      hdr_path.DirName().Append(basename.value() + "_config.h");
  base::FilePath prime_field_x86_hdr_path =
      hdr_path.DirName().Append(basename.value() + "_prime_field_x86.h");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{config_header_path}", config_header_path.value()},
          {"%{prime_field_x86_hdr}", prime_field_x86_hdr_path.value()},
          {"%{namespace}", ns_name},
          {"%{class}", class_name},
      });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateGpuHdr() const {
  std::string_view tpl[] = {
      "#include \"%{config_header_path}\"",
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
      tpl_content, {
                       {"%{config_header_path}",
                        math::ConvertToConfigHdr(GetHdrPath()).value()},
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
  parser.AddFlag<base::StringFlag>(&config.flag)
      .set_long_name("--flag")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.x86_hdr_tpl_path)
      .set_long_name("--x86_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.fail_hdr_tpl_path)
      .set_long_name("--fail_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.fail_src_tpl_path)
      .set_long_name("--fail_src_tpl_path")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.subgroup_generator)
      .set_long_name("--subgroup_generator");
  parser.AddFlag<base::StringFlag>(&config.small_subgroup_base)
      .set_long_name("--small_subgroup_base");
  parser.AddFlag<base::StringFlag>(&config.small_subgroup_adicity)
      .set_long_name("--small_subgroup_adicity");

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "_fail.h")) {
    return config.GenerateFailHdr();
  } else if (base::EndsWith(config.out.value(), "_fail.cc")) {
    return config.GenerateFailSrc();
  } else if (base::EndsWith(config.out.value(), "_prime_field_x86.h")) {
    return config.GeneratePrimeFieldX86Hdr();
  } else if (base::EndsWith(config.out.value(), "_config.h")) {
    return config.GenerateConfigHdr();
  } else if (base::EndsWith(config.out.value(), "_gpu.h")) {
    return config.GenerateGpuHdr();
  } else if (base::EndsWith(config.out.value(), ".h")) {
    return config.GenerateCpuHdr();
  } else {
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
