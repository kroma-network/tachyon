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
  base::FilePath point_hdr_tpl_path;
  base::FilePath point_src_tpl_path;
  base::FilePath point_traits_hdr_tpl_path;
  base::FilePath point_type_traits_hdr_tpl_path;
  base::FilePath msm_hdr_tpl_path;
  base::FilePath msm_src_tpl_path;
  base::FilePath msm_gpu_hdr_tpl_path;
  base::FilePath msm_gpu_src_tpl_path;

  std::string type;

  int GeneratePointHdr(std::string_view g1_or_g2,
                       std::string_view fq_or_fq2) const;
  int GeneratePointSrc(std::string_view g1_or_g2,
                       std::string_view fq_or_fq2) const;
  int GeneratePointTraitsHdr(std::string_view g1_or_g2,
                             std::string_view fq_or_fq2) const;
  int GeneratePointTypeTraitsHdr(std::string_view g1_or_g2,
                                 std::string_view fq_or_fq2) const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
  int GenerateG1TraitsHdr() const;
  int GenerateG1TypeTraitsHdr() const;
  int GenerateG2Hdr() const;
  int GenerateG2Src() const;
  int GenerateG2TraitsHdr() const;
  int GenerateG2TypeTraitsHdr() const;
  int GenerateMSMHdr() const;
  int GenerateMSMSrc() const;
  int GenerateMSMGpuHdr() const;
  int GenerateMSMGpuSrc() const;
};

int GenerationConfig::GeneratePointHdr(std::string_view g1_or_g2,
                                       std::string_view fq_or_fq2) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(point_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {{"%{header_dir_name}", c::math::GetLocation(type)},
                    {"%{type}", type},
                    {"%{g1_or_g2}", g1_or_g2},
                    {"%{fq_or_fq2}", fq_or_fq2}});
  return WriteHdr(content, true);
}

int GenerationConfig::GenerateG1Hdr() const {
  return GeneratePointHdr("g1", "fq");
}

int GenerationConfig::GenerateG2Hdr() const {
  return GeneratePointHdr("g2", "fq2");
}

int GenerationConfig::GeneratePointSrc(std::string_view g1_or_g2,
                                       std::string_view fq_or_fq2) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(point_src_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {{"%{header_dir_name}", c::math::GetLocation(type)},
                    {"%{type}", type},
                    {"%{g1_or_g2}", g1_or_g2},
                    {"%{G1_or_G2}", base::CapitalizeASCII(g1_or_g2)},
                    {"%{fq_or_fq2}", fq_or_fq2}});
  return WriteSrc(content);
}

int GenerationConfig::GenerateG1Src() const {
  return GeneratePointSrc("g1", "fq");
}

int GenerationConfig::GenerateG2Src() const {
  return GeneratePointSrc("g2", "fq2");
}

int GenerationConfig::GeneratePointTraitsHdr(std::string_view g1_or_g2,
                                             std::string_view fq_or_fq2) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(point_traits_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{g1_or_g2}", g1_or_g2},
                       {"%{G1_or_G2}", base::CapitalizeASCII(g1_or_g2)},
                       {"%{fq_or_fq2}", fq_or_fq2},
                       {"%{Fq_or_Fq2}", base::CapitalizeASCII(fq_or_fq2)},
                   });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateG1TraitsHdr() const {
  return GeneratePointTraitsHdr("g1", "fq");
}

int GenerationConfig::GenerateG2TraitsHdr() const {
  return GeneratePointTraitsHdr("g2", "fq2");
}

int GenerationConfig::GeneratePointTypeTraitsHdr(
    std::string_view g1_or_g2, std::string_view fq_or_fq2) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(point_type_traits_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{g1_or_g2}", g1_or_g2},
                       {"%{G1_or_G2}", base::CapitalizeASCII(g1_or_g2)},
                       {"%{Fq_or_Fq2}", base::CapitalizeASCII(fq_or_fq2)},
                   });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateG1TypeTraitsHdr() const {
  return GeneratePointTypeTraitsHdr("g1", "fq");
}

int GenerationConfig::GenerateG2TypeTraitsHdr() const {
  return GeneratePointTypeTraitsHdr("g2", "fq2");
}

int GenerationConfig::GenerateMSMHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(msm_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {{"%{header_dir_name}", c::math::GetLocation(type)}, {"%{type}", type}});
  return WriteHdr(content, true);
}

int GenerationConfig::GenerateMSMSrc() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(msm_src_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {{"%{header_dir_name}", c::math::GetLocation(type)}, {"%{type}", type}});
  return WriteSrc(content);
}

int GenerationConfig::GenerateMSMGpuHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(msm_gpu_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                   });
  return WriteHdr(content, true);
}

int GenerationConfig::GenerateMSMGpuSrc() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(msm_gpu_src_tpl_path, &tpl_content));

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
  parser.AddFlag<base::FilePathFlag>(&config.point_hdr_tpl_path)
      .set_long_name("--point_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.point_src_tpl_path)
      .set_long_name("--point_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.point_traits_hdr_tpl_path)
      .set_long_name("--point_traits_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.point_type_traits_hdr_tpl_path)
      .set_long_name("--point_type_traits_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.msm_hdr_tpl_path)
      .set_long_name("--msm_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.msm_src_tpl_path)
      .set_long_name("--msm_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.msm_gpu_hdr_tpl_path)
      .set_long_name("--msm_gpu_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.msm_gpu_src_tpl_path)
      .set_long_name("--msm_gpu_src_tpl_path")
      .set_required();

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "g1.h")) {
    return config.GenerateG1Hdr();
  } else if (base::EndsWith(config.out.value(), "g1.cc")) {
    return config.GenerateG1Src();
  } else if (base::EndsWith(config.out.value(), "g1_point_traits.h")) {
    return config.GenerateG1TraitsHdr();
  } else if (base::EndsWith(config.out.value(), "g1_point_type_traits.h")) {
    return config.GenerateG1TypeTraitsHdr();
  } else if (base::EndsWith(config.out.value(), "g2.h")) {
    return config.GenerateG2Hdr();
  } else if (base::EndsWith(config.out.value(), "g2.cc")) {
    return config.GenerateG2Src();
  } else if (base::EndsWith(config.out.value(), "g2_point_traits.h")) {
    return config.GenerateG2TraitsHdr();
  } else if (base::EndsWith(config.out.value(), "g2_point_type_traits.h")) {
    return config.GenerateG2TypeTraitsHdr();
  } else if (base::EndsWith(config.out.value(), "msm.h")) {
    return config.GenerateMSMHdr();
  } else if (base::EndsWith(config.out.value(), "msm.cc")) {
    return config.GenerateMSMSrc();
  } else if (base::EndsWith(config.out.value(), "msm_gpu.h")) {
    return config.GenerateMSMGpuHdr();
  } else if (base::EndsWith(config.out.value(), "msm_gpu.cc")) {
    return config.GenerateMSMGpuSrc();
  } else {
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
