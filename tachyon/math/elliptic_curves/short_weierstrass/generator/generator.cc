#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/cxx20_erase_vector.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  std::string ns_name;
  std::string class_name;
  std::string a;
  std::string b;
  std::string x;
  std::string y;
  std::string fq_modulus;

  // For GLV
  std::string endomorphism_coefficient;
  std::string lambda;
  std::vector<std::string> glv_coefficients;
  std::string fr_modulus;

  int GenerateConfigHdr() const;
  int GenerateConfigGpuHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  std::vector<std::string_view> tpl = {
      // clang-format off
      "#include \"%{header_dir_name}/fq.h\"",
      "#include \"%{header_dir_name}/fr.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/affine_point.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/projective_point.h\"",
      "",
      "namespace %{namespace} {",
      "",
      "template <typename Fq, typename Fr>",
      "class %{class}CurveConfig {",
      " public:",
      "  using BaseField = Fq;",
      "  using ScalarField = Fr;",
      "",
      "  using CpuBaseField = typename Fq::CpuField;",
      "  using CpuScalarField = typename Fr::CpuField;",
      "  using GpuBaseField = typename Fq::GpuField;",
      "  using GpuScalarField = typename Fr::GpuField;",
      "  using CpuCurveConfig = %{class}CurveConfig<CpuBaseField, CpuScalarField>;",
      "  using GpuCurveConfig = %{class}CurveConfig<GpuBaseField, GpuScalarField>;",
      "",
      "  constexpr static BigInt<%{fq_n}> kA = BigInt<%{fq_n}>({",
      "    %{a_mont_form}",
      "  });",
      "  constexpr static BigInt<%{fq_n}> kB = BigInt<%{fq_n}>({",
      "    %{b_mont_form}",
      "  });",
      "  constexpr static Point2<BigInt<%{fq_n}>> kGenerator = Point2<BigInt<%{fq_n}>>(",
      "    BigInt<%{fq_n}>({",
      "      %{x_mont_form}",
      "    }),",
      "    BigInt<%{fq_n}>({",
      "      %{y_mont_form}",
      "    })",
      "  );",
      "  constexpr static BigInt<%{fq_n}> kEndomorphismCoefficient = BigInt<%{fq_n}>({",
      "    %{endomorphism_coeff_mont_form}",
      "  });",
      "  constexpr static BigInt<%{fr_n}> kLambda = BigInt<%{fr_n}>({",
      "    %{lambda_mont_form}",
      "  });",
      "  constexpr static BigInt<%{fr_n}> kGLVCoeff00 = BigInt<%{fr_n}>({",
      "    %{glv_coeff_00_mont_form}",
      "  });",
      "  constexpr static BigInt<%{fr_n}> kGLVCoeff01 = BigInt<%{fr_n}>({",
      "    %{glv_coeff_01_mont_form}",
      "  });",
      "  constexpr static BigInt<%{fr_n}> kGLVCoeff10 = BigInt<%{fr_n}>({",
      "    %{glv_coeff_10_mont_form}",
      "  });",
      "  constexpr static BigInt<%{fr_n}> kGLVCoeff11 = BigInt<%{fr_n}>({",
      "    %{glv_coeff_11_mont_form}",
      "  });",
      "};",
      "",
      "using %{class}AffinePoint = AffinePoint<SWCurve<%{class}CurveConfig<Fq, Fr>>>;",
      "using %{class}ProjectivePoint = ProjectivePoint<SWCurve<%{class}CurveConfig<Fq, Fr>>>;",
      "using %{class}JacobianPoint = JacobianPoint<SWCurve<%{class}CurveConfig<Fq, Fr>>>;",
      "using %{class}PointXYZZ = PointXYZZ<SWCurve<%{class}CurveConfig<Fq, Fr>>>;",
      "#if defined(TACHYON_GMP_BACKEND)",
      "using %{class}AffinePointGmp = AffinePoint<SWCurve<%{class}CurveConfig<FqGmp, FrGmp>>>;",
      "using %{class}ProjectivePointGmp = ProjectivePoint<SWCurve<%{class}CurveConfig<FqGmp, FrGmp>>>;",
      "using %{class}JacobianPointGmp = JacobianPoint<SWCurve<%{class}CurveConfig<FqGmp, FrGmp>>>;",
      "using %{class}PointXYZZGmp = PointXYZZ<SWCurve<%{class}CurveConfig<FqGmp, FrGmp>>>;",
      "#endif  // defined(TACHYON_GMP_BACKEND)",
      "",
      "}  // namespace %{namespace}",
      // clang-format on
  };

  if (glv_coefficients.empty()) {
    for (std::size_t i = 0; i < tpl.size(); ++i) {
      size_t idx = tpl[i].find("kEndomorphismCoefficient");
      if (idx != std::string::npos) {
        auto it = tpl.begin() + i;
        tpl.erase(it, it + 3 * 6);
        break;
      }
    }
  }

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  mpz_class fq_modulus = math::gmp::FromDecString(this->fq_modulus);
  mpz_class a = math::gmp::FromDecString(this->a);
  mpz_class b = math::gmp::FromDecString(this->b);
  mpz_class x = math::gmp::FromDecString(this->x);
  mpz_class y = math::gmp::FromDecString(this->y);

  size_t fq_n = math::gmp::GetLimbSize(fq_modulus);

  std::map<std::string, std::string> replacements = {{
      {"%{header_dir_name}", GetHdrPath().DirName().value()},
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{fq_n}", absl::StrCat(fq_n)},
      {"%{a_mont_form}", math::MpzClassToMontString(a, fq_modulus)},
      {"%{b_mont_form}", math::MpzClassToMontString(b, fq_modulus)},
      {"%{x_mont_form}", math::MpzClassToMontString(x, fq_modulus)},
      {"%{y_mont_form}", math::MpzClassToMontString(y, fq_modulus)},
  }};

  if (!glv_coefficients.empty()) {
    mpz_class fr_modulus = math::gmp::FromDecString(this->fr_modulus);
    mpz_class endomorphism_coefficient =
        math::gmp::FromDecString(this->endomorphism_coefficient);
    mpz_class lambda = math::gmp::FromDecString(this->lambda);
    std::vector<mpz_class> glv_coefficients = {
        math::gmp::FromDecString(this->glv_coefficients[0]),
        math::gmp::FromDecString(this->glv_coefficients[1]),
        math::gmp::FromDecString(this->glv_coefficients[2]),
        math::gmp::FromDecString(this->glv_coefficients[3]),
    };

    size_t fr_n = math::gmp::GetLimbSize(fr_modulus);

    replacements["%{endomorphism_coeff_mont_form}"] =
        math::MpzClassToMontString(endomorphism_coefficient, fq_modulus);
    replacements["%{lambda_mont_form}"] =
        math::MpzClassToMontString(lambda, fr_modulus);
    replacements["%{glv_coeff_00_mont_form}"] =
        math::MpzClassToMontString(glv_coefficients[0], fr_modulus);
    replacements["%{glv_coeff_01_mont_form}"] =
        math::MpzClassToMontString(glv_coefficients[1], fr_modulus);
    replacements["%{glv_coeff_10_mont_form}"] =
        math::MpzClassToMontString(glv_coefficients[2], fr_modulus);
    replacements["%{glv_coeff_11_mont_form}"] =
        math::MpzClassToMontString(glv_coefficients[3], fr_modulus);
    replacements["%{fr_n}"] = absl::StrCat(fr_n);
  }
  std::string content = absl::StrReplaceAll(tpl_content, replacements);
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateConfigGpuHdr() const {
  std::string_view tpl[] = {
      // clang-format off
      "#include \"%{header_dir_name}/fq_gpu.h\"",
      "#include \"%{header_dir_name}/fr_gpu.h\"",
      "#include \"%{header_path}\"",
      "#include \"tachyon/math/elliptic_curves/short_weierstrass/sw_curve_gpu.h\"",
      "",
      "namespace %{namespace} {",
      "",
      "using %{class}AffinePointGpu = AffinePoint<SWCurveGpu<%{class}CurveConfig<FqGpu, FrGpu>>>;",
      "using %{class}ProjectivePointGpu = ProjectivePoint<SWCurveGpu<%{class}CurveConfig<FqGpu, FrGpu>>>;",
      "using %{class}JacobianPointGpu = JacobianPoint<SWCurveGpu<%{class}CurveConfig<FqGpu, FrGpu>>>;",
      "using %{class}PointXYZZGpu = PointXYZZ<SWCurveGpu<%{class}CurveConfig<FqGpu, FrGpu>>>;",
      "",
      "}  // namespace %{namespace}",
      // clang-format on
  };
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  base::FilePath hdr_path = GetHdrPath();
  std::string basename = hdr_path.BaseName().value();
  basename = basename.substr(0, basename.find("_gpu"));
  std::string header_path = hdr_path.DirName().Append(basename + ".h").value();
  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", hdr_path.DirName().value()},
                       {"%{header_path}", header_path},
                       {"%{namespace}", ns_name},
                       {"%{class}", class_name},
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
  parser.AddFlag<base::StringFlag>(&config.a)
      .set_short_name("-a")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.b)
      .set_short_name("-b")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.x)
      .set_short_name("-x")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.y)
      .set_short_name("-y")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.fq_modulus)
      .set_long_name("--fq_modulus")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.endomorphism_coefficient)
      .set_long_name("--endomorphism_coefficient");
  parser.AddFlag<base::StringFlag>(&config.lambda).set_long_name("--lambda");
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.glv_coefficients)
      .set_long_name("--glv_coefficients");
  parser.AddFlag<base::StringFlag>(&config.fr_modulus)
      .set_long_name("--fr_modulus");

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
