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

template <size_t N>
std::vector<int8_t> ComputeAteLoopCount(const mpz_class& six_x_plus_2) {
  math::BigInt<N> x;
  math::gmp::CopyLimbs(six_x_plus_2, x.limbs);
  return x.ToNAF();
}

std::vector<int8_t> ComputeAteLoopCount(const mpz_class& x) {
  mpz_class six_x_plus_2 = mpz_class(6) * x + mpz_class(2);
  size_t limb_size = math::gmp::GetLimbSize(six_x_plus_2);
  switch (limb_size) {
    case 1:
      return ComputeAteLoopCount<1>(six_x_plus_2);
    case 2:
      return ComputeAteLoopCount<2>(six_x_plus_2);
    case 3:
      return ComputeAteLoopCount<3>(six_x_plus_2);
    case 4:
      return ComputeAteLoopCount<4>(six_x_plus_2);
    case 5:
      return ComputeAteLoopCount<5>(six_x_plus_2);
    case 6:
      return ComputeAteLoopCount<6>(six_x_plus_2);
    case 7:
      return ComputeAteLoopCount<7>(six_x_plus_2);
    case 8:
      return ComputeAteLoopCount<8>(six_x_plus_2);
    case 9:
      return ComputeAteLoopCount<9>(six_x_plus_2);
  }
  NOTREACHED();
  return {};
}

struct GenerationConfig : public build::CcWriter {
  base::FilePath curve_hdr_tpl_path;

  std::string ns_name;
  std::string class_name;
  base::FilePath fq12_hdr;
  base::FilePath g1_hdr;
  base::FilePath g2_hdr;
  std::string x;
  std::vector<std::string> twist_mul_by_q_x;
  std::vector<std::string> twist_mul_by_q_y;
  math::TwistType twist_type;

  int GenerateConfigHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  std::map<std::string, std::string> replace_map = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{fq12_hdr}", fq12_hdr.value()},
      {"%{g1_hdr}", g1_hdr.value()},
      {"%{g2_hdr}", g2_hdr.value()},
      {"%{twist_type}", TwistTypeToString(twist_type)},
  };

  mpz_class x_mpz = math::gmp::FromDecString(x);
  replace_map["%{x_is_negative}"] =
      base::BoolToString(math::gmp::IsNegative(x_mpz));
  x_mpz = math::gmp::GetAbs(x_mpz);
  replace_map["%{x}"] = math::MpzClassToString(x_mpz);
  replace_map["%{x_size}"] =
      base::NumberToString(math::gmp::GetLimbSize(x_mpz));

  std::vector<int8_t> ate_loop_count = ComputeAteLoopCount(x_mpz);
  replace_map["%{ate_loop_count}"] = absl::StrJoin(ate_loop_count, ", ");

  replace_map["%{twist_mul_by_q_x_init_code}"] =
      math::GenerateInitExtField("kTwistMulByQX", "Fq2", twist_mul_by_q_x,
                                 /*is_prime_field=*/true);
  replace_map["%{twist_mul_by_q_y_init_code}"] =
      math::GenerateInitExtField("kTwistMulByQY", "Fq2", twist_mul_by_q_y,
                                 /*is_prime_field=*/true);

  std::string tpl_content;
  CHECK(base::ReadFileToString(curve_hdr_tpl_path, &tpl_content));
  std::string content = absl::StrReplaceAll(tpl_content, replace_map);
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/elliptic_curves/bn/generator";

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
  parser.AddFlag<base::FilePathFlag>(&config.curve_hdr_tpl_path)
      .set_long_name("--curve_hdr_path")
      .set_required();
  parser.AddFlag<base::Flag<std::string>>(&config.x)
      .set_short_name("-x")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.twist_mul_by_q_x)
      .set_long_name("--twist_mul_by_q_x")
      .set_required();
  parser.AddFlag<base::Flag<std::vector<std::string>>>(&config.twist_mul_by_q_y)
      .set_long_name("--twist_mul_by_q_y")
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
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
