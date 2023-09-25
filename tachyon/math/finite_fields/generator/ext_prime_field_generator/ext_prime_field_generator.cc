#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_iterator.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  std::string ns_name;
  std::string class_name;
  int degree;
  int non_residue;
  std::string base_field_hdr;
  std::string base_field;

  int GenerateConfigHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  // clang-format off
  std::vector<std::string> tpl = {
      "#include \"tachyon/math/finite_fields/fp%{degree}.h\"",
      "#include \"%{base_field_hdr}\"",
      "",
      "namespace %{namespace} {",
      "",
      "template <typename _BaseField>",
      "class %{class}Config {",
      " public:",
      "  using BaseField = _BaseField;",
      "",
      "  // NOTE(chokobole): This can't be constexpr because of PrimeFieldGmp support.",
      "  static BaseField kNonResidue;",
      "",
      "  constexpr static bool kNonResidueIsMinusOne = %{non_residue_is_minus_one};",
      "  constexpr static size_t kDegreeOverBaseField = %{degree_over_base_field};",
      "",
      "  static BaseField MulByNonResidue(const BaseField& v) {",
      "    BaseField ret = v;",
      "    return ret%{mul_by_non_residue};",
      "  }",
      "",
      "  static void Init() {",
      "%{init}",
      "  }",
      "};",
      "",
      "template <typename BaseField>",
      "BaseField %{class}Config<BaseField>::kNonResidue;",
      "",
      "using %{class} = Fp%{degree}<%{class}Config<%{base_field}>>;",
      "#if defined(TACHYON_GMP_BACKEND)",
      "using %{class}Gmp = Fp%{degree}<%{class}Config<%{base_field}Gmp>>;",
      "#endif  // defined(TACHYON_GMP_BACKEND)",
      "",
      "}  // namespace %{namespace}",
  };
  // clang-format on
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  bool is_negative = non_residue < 0;
  uint64_t abs_non_residue = is_negative ? -non_residue : non_residue;
  math::BigInt<1> scalar(abs_non_residue);
  std::string mul_by_non_residue;
  auto it = math::BitIteratorBE<math::BigInt<1>>::begin(&scalar, true);
  ++it;
  auto end = math::BitIteratorBE<math::BigInt<1>>::end(&scalar);
  {
    std::stringstream ss;
    while (it != end) {
      ss << ".DoubleInPlace()";
      if (*it) {
        ss << ".AddInPlace(v)";
      }
      ++it;
    }
    if (is_negative) ss << ".NegInPlace()";
    mul_by_non_residue = ss.str();
  }

  std::string init;
  std::vector<std::string> init_components;
  init_components.push_back(
      "    using BigIntTy = typename BaseField::BigIntTy;");
  if (is_negative) {
    init_components.push_back(absl::Substitute(
        "    kNonResidue = BaseField::FromBigInt(BaseField::Config::kModulus "
        "- BigIntTy($0));",
        abs_non_residue));
  } else {
    init_components.push_back(absl::Substitute(
        "    kNonResidue = BaseField::FromBigInt(BigIntTy($0));",
        abs_non_residue));
  }
  init = absl::StrJoin(init_components, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {

          {"%{namespace}", ns_name},
          {"%{class}", class_name},
          {"%{degree}", base::NumberToString(degree)},
          {"%{degree_over_base_field}", base::NumberToString(degree)},
          {"%{base_field_hdr}", base_field_hdr},
          {"%{base_field}", base_field},
          {"%{non_residue_is_minus_one}",
           base::BoolToString(non_residue == -1)},
          {"%{mul_by_non_residue}", mul_by_non_residue},
          {"%{init}", init},
      });
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/finite_fields/ext_prime_field_generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.ns_name)
      .set_long_name("--namespace")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.class_name)
      .set_long_name("--class")
      .set_required();
  parser.AddFlag<base::IntFlag>(&config.degree)
      .set_long_name("--degree")
      .set_required();
  parser.AddFlag<base::IntFlag>(&config.non_residue)
      .set_long_name("--non_residue")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.base_field_hdr)
      .set_long_name("--base_field_hdr")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.base_field)
      .set_long_name("--base_field")
      .set_required();

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), ".h")) {
    return config.GenerateConfigHdr();
  } else {
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
