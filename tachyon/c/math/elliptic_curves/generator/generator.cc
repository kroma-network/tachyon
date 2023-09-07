#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/c/math/elliptic_curves/generator/generator_util.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  std::string type;
  int fq_limb_nums;
  int fr_limb_nums;

  int GeneratePrimeFieldHdr(std::string_view suffix) const;
  int GenerateFqHdr() const;
  int GenerateFrHdr() const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
};

int GenerationConfig::GeneratePrimeFieldHdr(std::string_view suffix) const {
  std::vector<std::string_view> tpl = {
      "#include <stdint.h>",
      "",
      "#include \"tachyon/c/export.h\"",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_%{suffix} {",
      "  uint64_t limbs[%{limb_nums}];",
      "};",
  };

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{type}", type},
          {"%{suffix}", std::string(suffix)},
          {"%{limb_nums}",
           absl::StrCat(suffix == "fq" ? fq_limb_nums : fr_limb_nums)},
      });
  return WriteHdr(content);
}

int GenerationConfig::GenerateFqHdr() const {
  return GeneratePrimeFieldHdr("fq");
}

int GenerationConfig::GenerateFrHdr() const {
  return GeneratePrimeFieldHdr("fr");
}

int GenerationConfig::GenerateG1Hdr() const {
  // clang-format off
  std::string_view tpl[] = {
      "#include \"tachyon/c/export.h\"",
      "#include \"%{header_path}\"",
      "",
      "struct TACHYON_C_EXPORT __attribute__((aligned(32))) tachyon_%{type}_g1_affine {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  // needs to occupy 32 byte",
      "  // NOTE(chokobole): See LimbsAlignment() in tachyon/math/base/big_int.h",
      "  bool infinity = false;",
      "};",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_g1_projective {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "};",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "};",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_g1_xyzz {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq zz;",
      "  tachyon_%{type}_fq zzz;",
      "};",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_g1_point2 {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "};",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_g1_point3 {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "};",
      "",
      "struct TACHYON_C_EXPORT tachyon_%{type}_g1_point4 {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "  tachyon_%{type}_fq w;",
      "};",
      "",
      "TACHYON_C_EXPORT void tachyon_%{type}_g1_init();",
  };
  // clang-format on
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  base::FilePath hdr_path = GetHdrPath();
  std::string basename = hdr_path.BaseName().value();
  std::string header_path = hdr_path.DirName().Append("fq.h").value();
  std::string content =
      absl::StrReplaceAll(tpl_content, {
                                           {"%{header_path}", header_path},
                                           {"%{type}", type},
                                       });
  return WriteHdr(content);
}

int GenerationConfig::GenerateG1Src() const {
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/g1.h\"",
      "",
      "void tachyon_%{type}_init() {",
      "  tachyon::math::%{type}::G1AffinePoint::Curve::Init();",
      "}",
  };

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                   });
  return WriteSrc(content);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/c/math/elliptic_curves/generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.type)
      .set_long_name("--type")
      .set_required();
  parser.AddFlag<base::IntFlag>(&config.fq_limb_nums)
      .set_long_name("--fq_limb_nums")
      .set_required();
  parser.AddFlag<base::IntFlag>(&config.fr_limb_nums)
      .set_long_name("--fr_limb_nums")
      .set_required();

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "fq.h")) {
    return config.GenerateFqHdr();
  } else if (base::EndsWith(config.out.value(), "fr.h")) {
    return config.GenerateFrHdr();
  } else if (base::EndsWith(config.out.value(), "g1.h")) {
    return config.GenerateG1Hdr();
  } else if (base::EndsWith(config.out.value(), "g1.cc")) {
    return config.GenerateG1Src();
  } else {
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
