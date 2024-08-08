#include "absl/container/flat_hash_map.h"
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
  base::FilePath fp_hdr_tpl_path;

  std::string ns_name;
  std::string class_name;
  int degree;
  int base_field_degree;
  std::vector<std::string> non_residue;
  std::string base_field_hdr;
  std::string base_field;
  bool is_packed;
  std::string mul_by_non_residue_override;

  std::string GenerateFastMulByNonResidueCode(int64_t a_value) const;
  std::string GenerateMulByNonResidueCodeByDegree() const;
  std::string GenerateInitCode(bool mul_by_non_residue_fast) const;
  int GenerateConfigHdr() const;
};

std::string GenerationConfig::GenerateInitCode(
    bool mul_by_non_residue_fast) const {
  std::string init;
  if (mul_by_non_residue_fast) {
    if (is_packed) {
      init = math::GenerateInitPackedField("kNonResidue", "BaseField",
                                           non_residue[0]);
    } else {
      init =
          math::GenerateInitField("kNonResidue", "BaseField", non_residue[0]);
    }
  } else {
    init = math::GenerateInitExtField("kNonResidue", "BaseField", non_residue,
                                      base_field_degree);
  }
  return init;
}

std::string GenerationConfig::GenerateFastMulByNonResidueCode(
    int64_t a_value) const {
  std::stringstream ss;
  // clang-format off
  ss << "    BaseField ret = v;" << std::endl;
  ss << "    return ret" << math::GenerateFastMultiplication(a_value) << ";";
  // clang-format on
  return ss.str();
}

std::string GenerationConfig::GenerateMulByNonResidueCodeByDegree() const {
  std::stringstream ss;
  // clang-format off

  if (degree == 4 && non_residue.size() == 2 && non_residue[0] == "0" && non_residue[1] == "1") {
    ss << "    // See [[DESD06, Section 5.1]](https://eprint.iacr.org/2006/471.pdf)." << std::endl;
    ss << "    return BaseField(BaseField::Config::MulByNonResidue(v.c1()), v.c0());";
  } else if (degree == 12 && non_residue.size() == 3 && non_residue[0] == "0" && non_residue[1] == "1" && non_residue[2] == "0") {
    ss << "    // See [[DESD06, Section 6.1]](https://eprint.iacr.org/2006/471.pdf)." << std::endl;
    ss << "    return BaseField(BaseField::Config::MulByNonResidue(v.c2()), v.c0(), v.c1());";
  } else {
    return "    return v * kNonResidue;";
  }
  // clang-format on
  return ss.str();
}

int GenerationConfig::GenerateConfigHdr() const {
  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{degree}", base::NumberToString(degree)},
      {"%{degree_over_base_prime_field}", base::NumberToString(degree)},
      {"%{base_field}", base_field},
      {"%{base_field_hdr}", base_field_hdr},
      {"%{frobenius_coeffs_size}", base::NumberToString(degree)}};

  int degree_over_base_field = degree / base_field_degree;
  replacements["%{degree_over_base_field}"] =
      base::NumberToString(degree_over_base_field);
  replacements["%{base_prime_field}"] =
      degree == degree_over_base_field ? "BaseField"
                                       : "typename BaseField::BasePrimeField";
  if (degree_over_base_field == 3) {
    replacements["%{frobenius_coeffs2_size}"] = base::NumberToString(degree);
  }

  bool mul_by_non_residue_fast =
      non_residue.size() == 1
          ? true
          : std::all_of(non_residue.begin() + 1, non_residue.end(),
                        [](const std::string& e) { return e == "0"; });

  replacements["%{init_code}"] = GenerateInitCode(mul_by_non_residue_fast);

  int64_t a_value;
  CHECK(base::StringToInt64(non_residue[0], &a_value));
  replacements["%{non_residue_is_minus_one}"] =
      base::BoolToString(a_value == -1);

  if (!mul_by_non_residue_override.empty()) {
    replacements["%{mul_by_non_residue_code}"] = mul_by_non_residue_override;
  } else if (mul_by_non_residue_fast) {
    replacements["%{mul_by_non_residue_code}"] =
        GenerateFastMulByNonResidueCode(a_value);
  } else {
    replacements["%{mul_by_non_residue_code}"] =
        GenerateMulByNonResidueCodeByDegree();
  }

  replacements["%{frobenius_coefficient}"] =
      ((degree == 4 && base_field_degree == 2) ||
       (degree == 6 && base_field_degree == 3) || degree == 12)
          ? "typename BaseField::BaseField"
          : "BaseField";

  std::string tpl_content;
  CHECK(base::ReadFileToString(fp_hdr_tpl_path, &tpl_content));

  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, '\n');

  RemoveOptionalLines(tpl_lines, "FrobeniusCoefficient2",
                      degree_over_base_field >= 3);
  RemoveOptionalLines(tpl_lines, "FrobeniusCoefficient3",
                      degree_over_base_field >= 4);

  tpl_content = absl::StrJoin(tpl_lines, "\n");
  std::string content = absl::StrReplaceAll(tpl_content, replacements);
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/finite_fields/ext_field_generator";

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
  parser.AddFlag<base::BoolFlag>(&config.is_packed)
      .set_long_name("--is_packed")
      .set_default_value(false);
  parser.AddFlag<base::FilePathFlag>(&config.fp_hdr_tpl_path)
      .set_long_name("--fp_hdr_tpl_path")
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
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
