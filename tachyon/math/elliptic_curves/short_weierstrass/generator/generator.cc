#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
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
  std::vector<std::string_view> tpl = {
      // clang-format off
      "#include \"tachyon/base/logging.h\"",
      "#include \"%{base_field_hdr}\"",
      "#include \"%{scalar_field_hdr}\"",
      "#include \"tachyon/math/base/gmp/gmp_util.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/affine_point.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/projective_point.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/sw_curve.h\"",
      "",
      "namespace %{namespace} {",
      "",
      "template <typename _BaseField, typename _ScalarField>",
      "class %{class}CurveConfig {",
      " public:",
      "  using BaseField = _BaseField;",
      "  using BasePrimeField = %{base_prime_field};",
      "  using ScalarField = _ScalarField;",
      "",
      "  using CpuBaseField = typename BaseField::CpuField;",
      "  using CpuScalarField = typename ScalarField::CpuField;",
      "  using GpuBaseField = typename BaseField::GpuField;",
      "  using GpuScalarField = typename ScalarField::GpuField;",
      "  using CpuCurveConfig = %{class}CurveConfig<CpuBaseField, CpuScalarField>;",
      "  using GpuCurveConfig = %{class}CurveConfig<GpuBaseField, GpuScalarField>;",
      "",
      "  constexpr static bool kAIsZero = %{a_is_zero};",
      "",
      "  // TODO(chokobole): Make them constexpr.",
      "  static BaseField kA;",
      "  static BaseField kB;",
      "  static Point2<BaseField> kGenerator;",
      "  static BaseField kEndomorphismCoefficient;",
      "  static ScalarField kLambda;",
      "  static mpz_class kGLVCoeffs[4];",
      "",
      "  static void Init() {",
      "%{a_init}",
      "%{b_init}",
      "%{x_init}",
      "%{y_init}",
      "%{endomorphism_coefficient_init}",
      "%{lambda_init}",
      "%{glv_coeffs_init}",
      "    VLOG(1) << \"%{namespace}::%{class} initialized\";",
      "  }",
      "",
      "  constexpr static BaseField MulByA(const BaseField& v) {",
      "%{mul_by_a}",
      "  }",
      "};",
      "",
      "template <typename BaseField, typename ScalarField>",
      "BaseField %{class}CurveConfig<BaseField, ScalarField>::kA;",
      "template <typename BaseField, typename ScalarField>",
      "BaseField %{class}CurveConfig<BaseField, ScalarField>::kB;",
      "template <typename BaseField, typename ScalarField>",
      "Point2<BaseField> %{class}CurveConfig<BaseField, ScalarField>::kGenerator;",
      "template <typename BaseField, typename ScalarField>",
      "BaseField %{class}CurveConfig<BaseField, ScalarField>::kEndomorphismCoefficient;",
      "template <typename BaseField, typename ScalarField>",
      "ScalarField %{class}CurveConfig<BaseField, ScalarField>::kLambda;",
      "template <typename BaseField, typename ScalarField>",
      "mpz_class %{class}CurveConfig<BaseField, ScalarField>::kGLVCoeffs[4];",
      "",
      "using %{class}Curve = SWCurve<%{class}CurveConfig<%{base_field}, %{scalar_field}>>;",
      "using %{class}AffinePoint = AffinePoint<%{class}Curve>;",
      "using %{class}ProjectivePoint = ProjectivePoint<%{class}Curve>;",
      "using %{class}JacobianPoint = JacobianPoint<%{class}Curve>;",
      "using %{class}PointXYZZ = PointXYZZ<%{class}Curve>;",
      "",
      "}  // namespace %{namespace}",
      // clang-format on
  };

  if (glv_coefficients.empty()) {
    for (size_t j = 0; j < tpl.size(); ++j) {
      size_t idx = tpl[j].find("gmp_util.h");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + j;
        tpl.erase(it);
        break;
      }
    }
    for (size_t j = 0; j < tpl.size(); ++j) {
      size_t idx = tpl[j].find("kEndomorphismCoefficient");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + j;
        tpl.erase(it, it + 3);
        break;
      }
    }
    for (size_t j = 0; j < tpl.size(); ++j) {
      size_t idx = tpl[j].find("endomorphism_coefficient_init");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + j;
        tpl.erase(it, it + 3);
        break;
      }
    }
    for (size_t j = 0; j < tpl.size(); ++j) {
      size_t idx = tpl[j].find("kEndomorphismCoefficient");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + j - 1;
        tpl.erase(it, it + 3 * 2);
        break;
      }
    }
  }

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  bool a_is_zero = false;
  bool mul_by_a_fast = false;
  std::string mul_by_a;
  std::string a_init;
  std::string b_init;
  std::string x_init;
  std::string y_init;
  CHECK_EQ(b.size(), a.size());
  CHECK_EQ(x.size(), a.size());
  CHECK_EQ(y.size(), a.size());
  if (a.size() == 1) {
    mul_by_a_fast = true;

    a_init = math::GenerateInitField("kA", "BaseField", a[0]);
    b_init = math::GenerateInitField("kB", "BaseField", b[0]);
    x_init = math::GenerateInitField("kGenerator.x", "BaseField", x[0]);
    y_init = math::GenerateInitField("kGenerator.y", "BaseField", y[0]);
  } else {
    mul_by_a_fast =
        std::all_of(a.begin() + 1, a.end(), [](const std::string& a) {
          return math::gmp::FromDecString(a) == mpz_class(0);
        });

    a_init = math::GenerateInitExtField("kA", "BaseField", a,
                                        /*is_prime_field=*/true);
    b_init = math::GenerateInitExtField("kB", "BaseField", b,
                                        /*is_prime_field=*/true);
    x_init = math::GenerateInitExtField("kGenerator.x", "BaseField", x,
                                        /*is_prime_field=*/true);
    y_init = math::GenerateInitExtField("kGenerator.y", "BaseField", y,
                                        /*is_prime_field=*/true);
  }

  if (!mul_by_a_override.empty()) {
    mul_by_a = mul_by_a_override;
  } else if (mul_by_a_fast) {
    int64_t a_value;
    CHECK(base::StringToInt64(a[0], &a_value));
    std::stringstream ss;
    ss << "    return ";
    if (a_value == 0) {
      a_is_zero = true;
      ss << "BaseField::Zero()";
    } else {
      ss << math::GenerateFastMultiplication(a_value);
    }
    ss << ";";
    mul_by_a = ss.str();
  } else {
    mul_by_a = "    return kA * v;";
  }

  std::map<std::string, std::string> replacements = {
      {{"%{base_field_hdr}", base_field_hdr.value()},
       {"%{scalar_field_hdr}", scalar_field_hdr.value()},
       {"%{namespace}", ns_name},
       {"%{class}", class_name},
       {"%{base_field}", base_field},
       {"%{base_prime_field}",
        a.size() == 1 ? "BaseField" : "typename BaseField::BasePrimeField"},
       {"%{scalar_field}", scalar_field},
       {"%{a_is_zero}", base::BoolToString(a_is_zero)},
       {"%{a_init}", a_init},
       {"%{b_init}", b_init},
       {"%{x_init}", x_init},
       {"%{y_init}", y_init},
       {"%{mul_by_a}", mul_by_a}}};

  if (!glv_coefficients.empty()) {
    std::string endomorphism_coefficient_init;
    std::string lambda_init;
    std::string glv_coeffs_init;
    if (endomorphism_coefficient.size() == 1) {
      endomorphism_coefficient_init = math::GenerateInitField(
          "kEndomorphismCoefficient", "BaseField", endomorphism_coefficient[0]);
    } else {
      endomorphism_coefficient_init = math::GenerateInitExtField(
          "kEndomorphismCoefficient", "BaseField", endomorphism_coefficient,
          /*is_prime_field=*/true);
    }
    lambda_init = math::GenerateInitField("kLambda", "ScalarField", lambda);
    glv_coeffs_init = absl::StrJoin(
        base::CreateVector(4,
                           [this](size_t i) {
                             return math::GenerateInitMpzClass(
                                 absl::Substitute("kGLVCoeffs[$0]", i),
                                 glv_coefficients[i]);
                           }),
        "\n");

    replacements["%{endomorphism_coefficient_init}"] =
        endomorphism_coefficient_init;
    replacements["%{lambda_init}"] = lambda_init;
    replacements["%{glv_coeffs_init}"] = glv_coeffs_init;
  }

  std::string content = absl::StrReplaceAll(tpl_content, replacements);
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateConfigGpuHdr() const {
  std::string_view tpl[] = {
      // clang-format off
      "#include \"%{base_field_header}\"",
      "#include \"%{scalar_field_header}\"",
      "#include \"%{header_path}\"",
      "",
      "namespace %{namespace} {",
      "",
      "using %{class}CurveGpu = SWCurve<%{class}CurveConfig<%{base_field}Gpu, %{scalar_field}Gpu>>;",
      "using %{class}AffinePointGpu = AffinePoint<%{class}CurveGpu>;",
      "using %{class}ProjectivePointGpu = ProjectivePoint<%{class}CurveGpu>;",
      "using %{class}JacobianPointGpu = JacobianPoint<%{class}CurveGpu>;",
      "using %{class}PointXYZZGpu = PointXYZZ<%{class}CurveGpu>;",
      "",
      "}  // namespace %{namespace}",
      // clang-format on
  };
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{base_field_header}",
           math::ConvertToGpuHdr(base_field_hdr).value()},
          {"%{scalar_field_header}",
           math::ConvertToGpuHdr(scalar_field_hdr).value()},
          {"%{header_path}", math::ConvertToCpuHdr(GetHdrPath()).value()},
          {"%{namespace}", ns_name},
          {"%{class}", class_name},
          {"%{base_field}", base_field},
          {"%{scalar_field}", scalar_field},
      });
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
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
