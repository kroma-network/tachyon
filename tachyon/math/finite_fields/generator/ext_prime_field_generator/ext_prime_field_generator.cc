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
  int base_field_degree;
  std::vector<int> non_residue;
  std::string base_field_hdr;
  std::string base_field;
  std::string mul_by_non_residue_override;

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
      "%{mul_by_non_residue}",
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

  int degree_over_base_field = degree / base_field_degree;

  bool non_residue_is_minus_one = false;
  std::string mul_by_non_residue;
  std::string init;
  if (non_residue.size() == 1) {
    bool is_negative = non_residue[0] < 0;
    uint64_t abs_non_residue = is_negative ? -non_residue[0] : non_residue[0];
    non_residue_is_minus_one = is_negative && abs_non_residue == 1;
    math::BigInt<1> scalar(abs_non_residue);
    auto it = math::BitIteratorBE<math::BigInt<1>>::begin(&scalar, true);
    ++it;
    auto end = math::BitIteratorBE<math::BigInt<1>>::end(&scalar);
    {
      std::stringstream ss;
      ss << "    BaseField ret = v;";
      ss << std::endl;
      ss << "    return ret";
      while (it != end) {
        ss << ".DoubleInPlace()";
        if (*it) {
          ss << ".AddInPlace(v)";
        }
        ++it;
      }
      if (is_negative) ss << ".NegInPlace()";
      ss << ";";
      mul_by_non_residue = ss.str();
    }
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
  } else {
    non_residue_is_minus_one =
        non_residue[0] == -1 &&
        std::all_of(non_residue.begin() + 1, non_residue.end(),
                    [](int e) { return e == 0; });

    std::vector<std::string> init_components;
    init_components.push_back("    using F = typename BaseField::BaseField;");
    std::stringstream ss;
    ss << "    kNonResidue = BaseField(";
    for (size_t i = 0; i < non_residue.size(); ++i) {
      uint64_t abs_non_residue;
      if (non_residue[i] < 0) {
        ss << "-";
        abs_non_residue = -non_residue[i];
      } else {
        abs_non_residue = non_residue[i];
      }
      if (abs_non_residue == 0) {
        ss << "F::Zero()";
      } else if (abs_non_residue == 1) {
        ss << "F::One()";
      } else {
        ss << "F(" << abs_non_residue << ")";
      }
      if (i != non_residue.size() - 1) {
        ss << ", ";
      }
    }
    ss << ");";
    mul_by_non_residue = "    return v * kNonResidue;";
    init_components.push_back(ss.str());
    init = absl::StrJoin(init_components, "\n");
  }

  if (!mul_by_non_residue_override.empty()) {
    mul_by_non_residue = mul_by_non_residue_override;
  }

  std::string content = absl::StrReplaceAll(
      tpl_content, {

                       {"%{namespace}", ns_name},
                       {"%{class}", class_name},
                       {"%{degree}", base::NumberToString(degree)},
                       {"%{degree_over_base_field}",
                        base::NumberToString(degree_over_base_field)},
                       {"%{base_field_hdr}", base_field_hdr},
                       {"%{base_field}", base_field},
                       {"%{non_residue_is_minus_one}",
                        base::BoolToString(non_residue_is_minus_one)},
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
  parser.AddFlag<base::IntFlag>(&config.base_field_degree)
      .set_long_name("--base_field_degree")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<int>>>(&config.non_residue)
      .set_long_name("--non_residue")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.base_field_hdr)
      .set_long_name("--base_field_hdr")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.base_field)
      .set_long_name("--base_field")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.mul_by_non_residue_override)
      .set_long_name("--mul_by_non_residue_override");

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
