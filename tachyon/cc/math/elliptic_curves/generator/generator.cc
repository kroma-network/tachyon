#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/c/math/elliptic_curves/generator/generator_util.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  base::FilePath prime_field_hdr_tpl_path;
  base::FilePath prime_field_src_tpl_path;
  base::FilePath g1_hdr_tpl_path;
  base::FilePath g1_src_tpl_path;

  std::string type;
  int fq_limb_nums;
  int fr_limb_nums;

  int GeneratePrimeFieldHdr(std::string_view suffix) const;
  int GenerateFqHdr() const;
  int GenerateFrHdr() const;
  int GeneratePrimeFieldSrc(std::string_view suffix) const;
  int GenerateFqSrc() const;
  int GenerateFrSrc() const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
};

int GenerationConfig::GeneratePrimeFieldHdr(std::string_view suffix) const {
  std::string tpl_content;
  base::ReadFileToString(prime_field_hdr_tpl_path, &tpl_content);

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{header_dir_name}", c::math::GetLocation(type)},
          {"%{type}", type},
          {"%{suffix}", suffix},
          {"%{c_field}", absl::Substitute("tachyon_$0_$1", type, suffix)},
          {"%{cc_field}", suffix == "fq" ? "Fq" : "Fr"},
          {"%{limb_nums}",
           base::NumberToString(suffix == "fq" ? fq_limb_nums : fr_limb_nums)},
      });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateFqHdr() const {
  return GeneratePrimeFieldHdr("fq");
}

int GenerationConfig::GenerateFrHdr() const {
  return GeneratePrimeFieldHdr("fr");
}

int GenerationConfig::GeneratePrimeFieldSrc(std::string_view suffix) const {
  std::string tpl_content;
  base::ReadFileToString(prime_field_src_tpl_path, &tpl_content);

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{suffix}", suffix},
                       {"%{cc_field}", suffix == "fq" ? "Fq" : "Fr"},
                   });
  return WriteSrc(content);
}

int GenerationConfig::GenerateFqSrc() const {
  return GeneratePrimeFieldSrc("fq");
}

int GenerationConfig::GenerateFrSrc() const {
  return GeneratePrimeFieldSrc("fr");
}

int GenerationConfig::GenerateG1Hdr() const {
  std::string tpl_content;
  base::ReadFileToString(g1_hdr_tpl_path, &tpl_content);

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{c_fq}", absl::Substitute("tachyon_$0_fq", type)},
                       {"%{c_g1}", absl::Substitute("tachyon_$0_g1", type)},
                       {"%{fq_limb_nums}", base::NumberToString(fq_limb_nums)},
                   });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateG1Src() const {
  std::string tpl_content;
  base::ReadFileToString(g1_src_tpl_path, &tpl_content);

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                   });
  return WriteSrc(content);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/cc/math/elliptic_curves/generator";

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
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_hdr_tpl_path)
      .set_long_name("--prime_field_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_src_tpl_path)
      .set_long_name("--prime_field_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g1_hdr_tpl_path)
      .set_long_name("--g1_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g1_src_tpl_path)
      .set_long_name("--g1_src_tpl_path")
      .set_required();

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "fq.h")) {
    return config.GenerateFqHdr();
  } else if (base::EndsWith(config.out.value(), "fq.cc")) {
    return config.GenerateFqSrc();
  } else if (base::EndsWith(config.out.value(), "fr.h")) {
    return config.GenerateFrHdr();
  } else if (base::EndsWith(config.out.value(), "fr.cc")) {
    return config.GenerateFrSrc();
  } else if (base::EndsWith(config.out.value(), "g1.h")) {
    return config.GenerateG1Hdr();
  } else if (base::EndsWith(config.out.value(), "g1.cc")) {
    return config.GenerateG1Src();
  } else {
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
