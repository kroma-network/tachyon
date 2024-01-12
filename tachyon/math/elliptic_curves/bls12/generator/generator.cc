#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/elliptic_curves/pairing/twist_type.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  std::string ns_name;
  std::string class_name;
  base::FilePath fq12_hdr;
  base::FilePath g1_hdr;
  base::FilePath g2_hdr;
  std::string x;
  math::TwistType twist_type;

  int GenerateConfigHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  std::vector<std::string_view> tpl = {
      // clang-format off
      "#include \"tachyon/math/elliptic_curves/bls12/bls12_curve.h\"",
      "#include \"%{fq12_hdr}\"",
      "#include \"%{g1_hdr}\"",
      "#include \"%{g2_hdr}\"",
      "#include \"tachyon/math/elliptic_curves/pairing/twist_type.h\"",
      "",
      "namespace %{namespace} {",
      "",
      "template <typename Fq, typename Fq2, typename Fq6, typename Fq12, typename _G1Curve, typename _G2Curve>",
      "class %{class}Config {",
      " public:",
      "  constexpr static size_t kXLimbNums = %{x_size};",
      "  constexpr static BigInt<%{x_size}> kX = BigInt<%{x_size}>({",
      "    %{x}",
      "  });",
      "  constexpr static bool kXIsNegative = %{x_is_negative};",
      "  constexpr static TwistType kTwistType = TwistType::k%{twist_type};",
      "",
      "  using Fp = Fq;",
      "  using Fp2 = Fq2;",
      "  using Fp6 = Fq6;",
      "  using Fp12 = Fq12;",
      "  using G1Curve = _G1Curve;",
      "  using G2Curve = _G2Curve;",
      "",
      "  static void Init() {}",
      "};",
      "",
      "using %{class}Curve = BLS12Curve<%{class}Config<Fq, Fq2, Fq6, Fq12, G1Curve, G2Curve>>;",
      "",
      "}  // namespace %{namespace}",
      // clang-format on
  };

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  mpz_class x_mpz = math::gmp::FromDecString(x);
  bool x_is_negative = math::gmp::IsNegative(x_mpz);
  x_mpz = math::gmp::GetAbs(x_mpz);
  size_t x_size = math::gmp::GetLimbSize(x_mpz);

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{fq12_hdr}", fq12_hdr.value()},
                       {"%{g1_hdr}", g1_hdr.value()},
                       {"%{g2_hdr}", g2_hdr.value()},
                       {"%{namespace}", ns_name},
                       {"%{class}", class_name},
                       {"%{x_size}", base::NumberToString(x_size)},
                       {"%{x}", math::MpzClassToString(x_mpz)},
                       {"%{x_is_negative}", base::BoolToString(x_is_negative)},
                       {"%{twist_type}", TwistTypeToString(twist_type)},
                   });
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/elliptic_curves/bls12/generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.ns_name)
      .set_long_name("--namespace")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.class_name).set_long_name("--class");
  parser.AddFlag<base::FilePathFlag>(&config.fq12_hdr)
      .set_long_name("--fq12_hdr")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g1_hdr)
      .set_long_name("--g1_hdr")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g2_hdr)
      .set_long_name("--g2_hdr")
      .set_required();
  parser.AddFlag<base::Flag<std::string>>(&config.x)
      .set_short_name("-x")
      .set_required();
  parser.AddFlag<base::Flag<math::TwistType>>(&config.twist_type)
      .set_long_name("--twist_type")
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
