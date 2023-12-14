#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  std::string ns_name;
  std::string class_name;
  int degree;
  int base_field_degree;
  std::vector<std::string> non_residue;
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
      "  using BasePrimeField = %{base_prime_field};",
      "  using FrobeniusCoefficient = %{frobenius_coefficient};",
      "",
      "  // TODO(chokobole): Make them constexpr.",
      "  static BaseField kNonResidue;",
      "  static FrobeniusCoefficient kFrobeniusCoeffs[%{frob_coeffs_size}];",
      "  static FrobeniusCoefficient kFrobeniusCoeffs2[%{frob_coeffs_size}];",
      "",
      "  constexpr static bool kNonResidueIsMinusOne = %{non_residue_is_minus_one};",
      "  constexpr static uint64_t kDegreeOverBaseField = %{degree_over_base_field};",
      "  constexpr static uint64_t kDegreeOverBasePrimeField = %{degree_over_base_prime_field};",
      "",
      "  static BaseField MulByNonResidue(const BaseField& v) {",
      "%{mul_by_non_residue}",
      "  }",
      "",
      "  static void Init() {",
      "    BaseField::Init();",
      "%{init}",
      "  }",
      "};",
      "",
      "template <typename BaseField>",
      "BaseField %{class}Config<BaseField>::kNonResidue;",
      "template <typename BaseField>",
      "typename %{class}Config<BaseField>::FrobeniusCoefficient %{class}Config<BaseField>::kFrobeniusCoeffs[%{frob_coeffs_size}];",
      "template <typename BaseField>",
      "typename %{class}Config<BaseField>::FrobeniusCoefficient %{class}Config<BaseField>::kFrobeniusCoeffs2[%{frob_coeffs_size}];",
      "",
      "using %{class} = Fp%{degree}<%{class}Config<%{base_field}>>;",
      "",
      "}  // namespace %{namespace}",
  };

  int degree_over_base_field = degree / base_field_degree;

  if (degree_over_base_field != 3) {
    for (size_t j = 0; j < tpl.size(); ++j) {
      size_t idx = tpl[j].find("kFrobeniusCoeffs2");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + j;
        tpl.erase(it);
        break;
      }
    }
    for (size_t j = 0; j < tpl.size(); ++j) {
      size_t idx = tpl[j].find("kFrobeniusCoeffs2");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + j - 1;
        tpl.erase(it, it + 2);
        break;
      }
    }
  }

  // clang-format on
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  bool non_residue_is_minus_one = false;
  bool mul_by_non_residue_fast = false;
  std::string init;
  if (non_residue.size() == 1) {
    mul_by_non_residue_fast = true;

    init = math::GenerateInitField("kNonResidue", "BaseField", non_residue[0]);
  } else {
    mul_by_non_residue_fast =
        std::all_of(non_residue.begin() + 1, non_residue.end(),
                    [](const std::string& e) { return e == "0"; });

    init = math::GenerateInitExtField("kNonResidue", "BaseField",
                                      absl::MakeConstSpan(non_residue),
                                      /*is_prime_field=*/degree != 12);
  }

  std::string mul_by_non_residue;
  if (!mul_by_non_residue_override.empty()) {
    mul_by_non_residue = mul_by_non_residue_override;
  } else if (mul_by_non_residue_fast) {
    int64_t a_value;
    CHECK(base::StringToInt64(non_residue[0], &a_value));
    non_residue_is_minus_one = a_value == -1;
    std::stringstream ss;
    ss << "    BaseField ret = v;";
    ss << std::endl;
    ss << "    return ret";
    ss << math::GenerateFastMultiplication(a_value);
    ss << ";";
    mul_by_non_residue = ss.str();
  } else if (degree == 4) {
    std::stringstream ss;
    // clang-format off
    ss << "    // See [[DESD06, Section 5.1]](https://eprint.iacr.org/2006/471.pdf).";
    ss << std::endl;
    ss << "    return BaseField(BaseField::Config::MulByNonResidue(v.c1()), v.c0());";
    // clang-format on
    mul_by_non_residue = ss.str();
  } else if (degree == 12) {
    std::stringstream ss;
    // clang-format off
    ss << "    // See [[DESD06, Section 6.1]](https://eprint.iacr.org/2006/471.pdf).";
    ss << std::endl;
    ss << "    return BaseField(BaseField::Config::MulByNonResidue(v.c2()), v.c0(), v.c1());";
    // clang-format on
    mul_by_non_residue = ss.str();
  } else {
    mul_by_non_residue = "    return v * kNonResidue;";
  }

  std::string frobenius_coefficient;
  if (degree == 4 || (degree == 6 && base_field_degree == 3) || degree == 12) {
    // See fp4.h, fp6.h and fp12.h for details.
    frobenius_coefficient = "typename BaseField::BaseField";
  } else {
    // See fp2.h, fp3.h and fp6.h for details.
    frobenius_coefficient = "BaseField";
  }

  std::map<std::string, std::string> replacements = {{
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{degree}", base::NumberToString(degree)},
      {"%{degree_over_base_field}",
       base::NumberToString(degree_over_base_field)},
      {"%{degree_over_base_prime_field}", base::NumberToString(degree)},
      {"%{base_field_hdr}", base_field_hdr},
      {"%{base_field}", base_field},
      {"%{base_prime_field}", degree == degree_over_base_field
                                  ? "BaseField"
                                  : "typename BaseField::BasePrimeField"},
      {"%{non_residue_is_minus_one}",
       base::BoolToString(non_residue_is_minus_one)},
      {"%{mul_by_non_residue}", mul_by_non_residue},
      {"%{init}", init},
      {"%{frobenius_coefficient}", frobenius_coefficient},
      {"%{frob_coeffs_size}", base::NumberToString(degree)},
  }};

  if (degree_over_base_field == 3) {
    replacements["%{frob_coeffs2_size}"] = base::NumberToString(degree);
  }

  std::string content = absl::StrReplaceAll(tpl_content, replacements);
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
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.non_residue)
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
