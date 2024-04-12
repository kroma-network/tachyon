#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
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

size_t GetNumBits(const mpz_class& m) {
  auto it = math::BitIteratorBE<mpz_class>::begin(&m, true);
  auto end = math::BitIteratorBE<mpz_class>::end(&m);
  size_t num_bits = 0;
  while (it != end) {
    ++it;
    ++num_bits;
  }
  return num_bits;
}

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
  base::FilePath config_hdr_tpl_path;
  base::FilePath cpu_hdr_tpl_path;
  base::FilePath fail_src_tpl_path;
  base::FilePath fail_hdr_tpl_path;
  base::FilePath gpu_hdr_tpl_path;
  base::FilePath x86_hdr_tpl_path;

  std::string ns_name;
  std::string class_name;
  std::string modulus;
  std::string flag;
  std::string reduce;
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
  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{flag}", flag},
  };

  mpz_class m = math::gmp::FromDecString(modulus);
  replacements["%{modulus}"] = math::MpzClassToString(m);
  replacements["%{modulus_minus_one_div_two}"] =
      math::MpzClassToString((m - mpz_class(1)) / mpz_class(2));
  replacements["%{modulus_plus_one_div_four}"] =
      math::MpzClassToString((m + mpz_class(1)) / mpz_class(4));
  replacements["%{modulus_mod_four_is_three}"] =
      base::BoolToString(m % mpz_class(4) == mpz_class(3));
  replacements["%{modulus_mod_six_is_one}"] =
      base::BoolToString(m % mpz_class(6) == mpz_class(1));
  replacements["%{one_mont_form}"] =
      math::MpzClassToMontString(mpz_class(1), m);

  mpz_class trace = math::ComputeTrace(2, m - mpz_class(1));
  replacements["%{trace}"] = math::MpzClassToString(trace);
  replacements["%{trace_minus_one_div_two}"] =
      math::MpzClassToString((trace - mpz_class(1)) / mpz_class(2));

  size_t num_bits = GetNumBits(m);
  replacements["%{n}"] = base::NumberToString(math::gmp::GetLimbSize(m));
  replacements["%{modulus_bits}"] = base::NumberToString(num_bits);

  ModulusInfo modulus_info = ModulusInfo::From(m);
  replacements["%{modulus_has_spare_bit}"] =
      base::BoolToString(modulus_info.modulus_has_spare_bit);
  replacements["%{can_use_no_carry_mul_optimization}"] =
      base::BoolToString(modulus_info.can_use_no_carry_mul_optimization);
  replacements["%{r}"] = math::MpzClassToString(modulus_info.r);
  replacements["%{r2}"] = math::MpzClassToString(modulus_info.r2);
  replacements["%{r3}"] = math::MpzClassToString(modulus_info.r3);
  replacements["%{inverse64}"] = base::NumberToString(modulus_info.inverse64);
  replacements["%{inverse32}"] = base::NumberToString(modulus_info.inverse32);

  std::string tpl_content;
  CHECK(base::ReadFileToString(config_hdr_tpl_path, &tpl_content));
  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, "\n");

  bool is_small_field = num_bits <= 32;
  RemoveOptionalLines(tpl_lines, "kIsSmallField", is_small_field);
  RemoveOptionalLines(tpl_lines, "!kIsSmallField", !is_small_field);
  if (is_small_field) replacements["%{reduce}"] = reduce;

  bool has_two_adic_root_of_unity = !subgroup_generator.empty();
  replacements["%{has_two_adic_root_of_unity}"] =
      base::BoolToString(has_two_adic_root_of_unity);
  RemoveOptionalLines(tpl_lines, "kHasTwoAdicRootOfUnity",
                      has_two_adic_root_of_unity);
  mpz_class subgroup_generator_mpz;
  if (has_two_adic_root_of_unity) {
    subgroup_generator_mpz = math::gmp::FromDecString(subgroup_generator);
    replacements["%{subgroup_generator}"] =
        math::MpzClassToMontString(subgroup_generator_mpz, m);

    // 1) 2ˢ * t = m - 1,
    // According to Fermat's Little Theorem, the following equations hold:
    // 2) g^(m - 1) = 1 (mod m)
    // 3) g^(2ˢ * t) = 1 (mod m)
    // 4) gᵗ^(2ˢ) = 1 (mod m)
    // Where subgroup_generator = g, two_adicity = s, trace = t and
    // two_adic_root_of_unity = gᵗ.
    mpz_class two_adic_root_of_unity;
    mpz_powm(two_adic_root_of_unity.get_mpz_t(),
             subgroup_generator_mpz.get_mpz_t(), trace.get_mpz_t(),
             m.get_mpz_t());
    replacements["%{two_adicity}"] =
        base::NumberToString(math::ComputeAdicity(2, m - mpz_class(1)));
    replacements["%{two_adic_root_of_unity}"] =
        math::MpzClassToMontString(two_adic_root_of_unity, m);
  }

  bool has_large_subgroup_root_of_unity = !small_subgroup_base.empty();
  replacements["%{has_large_subgroup_root_of_unity}"] =
      base::BoolToString(has_large_subgroup_root_of_unity);
  RemoveOptionalLines(tpl_lines, "kHasLargeSubgroupRootOfUnity",
                      has_large_subgroup_root_of_unity);
  if (has_two_adic_root_of_unity && has_large_subgroup_root_of_unity) {
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
    mpz_class large_subgroup_root_of_unity;
    {
      mpz_class remaining_subgroup_size;
      mpz_div(remaining_subgroup_size.get_mpz_t(), trace.get_mpz_t(),
              small_subgroup_base_pow_adicity.get_mpz_t());
      mpz_powm(large_subgroup_root_of_unity.get_mpz_t(),
               subgroup_generator_mpz.get_mpz_t(),
               remaining_subgroup_size.get_mpz_t(), m.get_mpz_t());
    }

    replacements["%{small_subgroup_base}"] = small_subgroup_base;
    replacements["%{small_subgroup_adicity}"] = small_subgroup_adicity;
    replacements["%{large_subgroup_root_of_unity}"] =
        math::MpzClassToMontString(large_subgroup_root_of_unity, m);
  }

  tpl_content = absl::StrJoin(tpl_lines, "\n");
  std::string content =
      absl::StrReplaceAll(tpl_content, std::move(replacements));

  return WriteHdr(content, false);
}

int GenerationConfig::GenerateCpuHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(cpu_hdr_tpl_path, &tpl_content));

  base::FilePath hdr_path = GetHdrPath();
  base::FilePath basename = hdr_path.BaseName().RemoveExtension();
  base::FilePath config_header_path =
      hdr_path.DirName().Append(basename.value() + "_config.h");
  base::FilePath prime_field_x86_hdr_path =
      hdr_path.DirName().Append(basename.value() + "_prime_field_x86.h");

  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{config_header_path}", config_header_path.value()},
      {"%{prime_field_x86_hdr}", prime_field_x86_hdr_path.value()},
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
  };

  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, "\n");

  mpz_class m = math::gmp::FromDecString(modulus);
  size_t num_bits = GetNumBits(m);
  bool is_small_field = num_bits <= 32;
  RemoveOptionalLines(tpl_lines, "kIsSmallField", is_small_field);
  RemoveOptionalLines(tpl_lines, "!kIsSmallField", !is_small_field);

  tpl_content = absl::StrJoin(tpl_lines, "\n");

  std::string content =
      absl::StrReplaceAll(tpl_content, std::move(replacements));

  return WriteHdr(content, false);
}

int GenerationConfig::GenerateGpuHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(gpu_hdr_tpl_path, &tpl_content));

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
  parser.AddFlag<base::StringFlag>(&config.reduce).set_long_name("--reduce");
  parser.AddFlag<base::FilePathFlag>(&config.x86_hdr_tpl_path)
      .set_long_name("--x86_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.fail_hdr_tpl_path)
      .set_long_name("--fail_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.fail_src_tpl_path)
      .set_long_name("--fail_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.config_hdr_tpl_path)
      .set_long_name("--config_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.cpu_hdr_tpl_path)
      .set_long_name("--cpu_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.gpu_hdr_tpl_path)
      .set_long_name("--gpu_hdr_tpl_path")
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
