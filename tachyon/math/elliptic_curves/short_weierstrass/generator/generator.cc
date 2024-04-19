#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"

namespace tachyon {

std::string GenerateMulByAFunc(bool mul_by_a_fast, int64_t a_value) {
  if (a_value == 0) return "    return BaseField::Zero();";
  if (mul_by_a_fast) {
    std::stringstream ss;
    ss << "    return ";
    ss << math::GenerateFastMultiplication(a_value) << ";";
    return ss.str();
  }
  return "    return kA * v;";
}

struct GenerationConfig : public build::CcWriter {
  base::FilePath cpu_hdr_tpl_path;
  base::FilePath gpu_hdr_tpl_path;

  std::string ns_name;
  std::string class_name;
  std::string base_field;
  base::FilePath base_field_hdr;
  std::string scalar_field;
  base::FilePath scalar_field_hdr;
  std::vector<std::string> a;
  std::vector<std::string> b;
  std::vector<std::string> x;
  std::vector<std::string> y;
  std::string mul_by_a_override;

  // For GLV
  std::vector<std::string> endomorphism_coefficient;
  std::string lambda;
  std::vector<std::string> glv_coefficients;

  int GenerateConfigHdr() const;
  int GenerateConfigGpuHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  CHECK_EQ(b.size(), a.size());
  CHECK_EQ(x.size(), a.size());
  CHECK_EQ(y.size(), a.size());

  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{base_field}", base_field},
      {"%{base_field_hdr}", base_field_hdr.value()},
      {"%{scalar_field}", scalar_field},
      {"%{scalar_field_hdr}", scalar_field_hdr.value()},
      {"%{base_prime_field}",
       a.size() == 1 ? "BaseField" : "typename BaseField::BasePrimeField"}};

  if (a.size() == 1) {
    replacements["%{a_init}"] =
        math::GenerateInitField("kA", "BaseField", a[0]);
    replacements["%{b_init}"] =
        math::GenerateInitField("kB", "BaseField", b[0]);
    replacements["%{x_init}"] =
        math::GenerateInitField("kGenerator.x", "BaseField", x[0]);
    replacements["%{y_init}"] =
        math::GenerateInitField("kGenerator.y", "BaseField", y[0]);
  } else {
    bool is_prime_field = true;
    replacements["%{a_init}"] =
        math::GenerateInitExtField("kA", "BaseField", a, is_prime_field);
    replacements["%{b_init}"] =
        math::GenerateInitExtField("kB", "BaseField", b, is_prime_field);
    replacements["%{x_init}"] = math::GenerateInitExtField(
        "kGenerator.x", "BaseField", x, is_prime_field);
    replacements["%{y_init}"] = math::GenerateInitExtField(
        "kGenerator.y", "BaseField", y, is_prime_field);
  }

  int64_t a_value;
  CHECK(base::StringToInt64(a[0], &a_value));
  bool a_is_zero = a_value == 0;
  replacements["%{a_is_zero}"] = base::BoolToString(a_is_zero);

  bool mul_by_a_fast =
      a.size() == 1
          ? true
          : std::all_of(a.begin() + 1, a.end(), [](const std::string& a) {
              return math::gmp::FromDecString(a) == mpz_class(0);
            });

  replacements["%{mul_by_a_code}"] =
      !mul_by_a_override.empty() ? mul_by_a_override
                                 : GenerateMulByAFunc(mul_by_a_fast, a_value);

  std::string tpl_content;
  CHECK(base::ReadFileToString(cpu_hdr_tpl_path, &tpl_content));
  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, '\n');

  bool has_glv_coefficients = !glv_coefficients.empty();
  if (has_glv_coefficients) {
    if (endomorphism_coefficient.size() == 1) {
      replacements["%{endomorphism_coefficient_init_code}"] =
          math::GenerateInitField("kEndomorphismCoefficient", "BaseField",
                                  endomorphism_coefficient[0]);
    } else {
      replacements["%{endomorphism_coefficient_init_code}"] =
          math::GenerateInitExtField("kEndomorphismCoefficient", "BaseField",
                                     endomorphism_coefficient,
                                     /*is_prime_field=*/
                                     true);
    }
    replacements["%{lambda_init_code}"] =
        math::GenerateInitField("kLambda", "ScalarField", lambda);
    replacements["%{glv_coeffs_init_code}"] = absl::StrJoin(
        base::CreateVector(4,
                           [this](size_t i) {
                             return math::GenerateInitMpzClass(
                                 absl::Substitute("kGLVCoeffs[$0]", i),
                                 glv_coefficients[i]);
                           }),
        "\n");

    RemoveOptionalLines(tpl_lines, "HasGLVCoefficients", has_glv_coefficients);
  }
  tpl_content = absl::StrJoin(tpl_lines, "\n");
  std::string content = absl::StrReplaceAll(tpl_content, replacements);
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateConfigGpuHdr() const {
  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{base_field}", base_field},
      {"%{base_field_header}", math::ConvertToGpuHdr(base_field_hdr).value()},
      {"%{scalar_field}", scalar_field},
      {"%{scalar_field_header}",
       math::ConvertToGpuHdr(scalar_field_hdr).value()},
      {"%{cpu_header_path}", math::ConvertToCpuHdr(GetHdrPath()).value()},
  };

  std::string tpl_content;
  CHECK(base::ReadFileToString(gpu_hdr_tpl_path, &tpl_content));
  std::string content = absl::StrReplaceAll(tpl_content, replacements);
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator =
      "//tachyon/math/elliptic_curves/short_weierstrass/generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.ns_name)
      .set_long_name("--namespace")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.class_name).set_long_name("--class");
  parser.AddFlag<base::StringFlag>(&config.base_field)
      .set_long_name("--base_field")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.base_field_hdr)
      .set_long_name("--base_field_hdr")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.scalar_field)
      .set_long_name("--scalar_field")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.scalar_field_hdr)
      .set_long_name("--scalar_field_hdr")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.cpu_hdr_tpl_path)
      .set_long_name("--cpu_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.gpu_hdr_tpl_path)
      .set_long_name("--gpu_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.a)
      .set_short_name("-a")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.b)
      .set_short_name("-b")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.x)
      .set_short_name("-x")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.y)
      .set_short_name("-y")
      .set_required();
  parser
      .AddFlag<base::Flag<std::vector<std::string>>>(
          &config.endomorphism_coefficient)
      .set_long_name("--endomorphism_coefficient");
  parser.AddFlag<base::StringFlag>(&config.lambda).set_long_name("--lambda");
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.glv_coefficients)
      .set_long_name("--glv_coefficients");
  parser.AddFlag<base::StringFlag>(&config.mul_by_a_override)
      .set_long_name("--mul_by_a_override");

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "_gpu.h")) {
    return config.GenerateConfigGpuHdr();
  } else if (base::EndsWith(config.out.value(), ".h")) {
    return config.GenerateConfigHdr();
  } else {
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
