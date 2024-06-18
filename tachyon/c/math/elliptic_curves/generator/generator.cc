#include "absl/strings/str_replace.h"
#include "absl/types/span.h"

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
  base::FilePath prime_field_type_traits_hdr_tpl_path;
  base::FilePath ext_field_hdr_tpl_path;
  base::FilePath ext_field_src_tpl_path;
  base::FilePath ext_field_type_traits_hdr_tpl_path;
  base::FilePath point_hdr_tpl_path;
  base::FilePath point_src_tpl_path;
  base::FilePath point_traits_hdr_tpl_path;
  base::FilePath msm_hdr_tpl_path;
  base::FilePath msm_src_tpl_path;
  base::FilePath msm_gpu_hdr_tpl_path;
  base::FilePath msm_gpu_src_tpl_path;

  std::string type;
  int fq_limb_nums;
  int fr_limb_nums;
  int degree;
  int base_field_degree;
  bool has_specialized_g1_msm_kernels;

  int GeneratePrimeFieldHdr(std::string_view suffix) const;
  int GeneratePrimeFieldSrc(std::string_view suffix) const;
  int GeneratePrimeFieldTypeTraitsHdr(std::string_view suffix) const;
  int GenerateFqHdr() const;
  int GenerateFqSrc() const;
  int GenerateFqTypeTraitsHdr() const;
  int GenerateFrHdr() const;
  int GenerateFrSrc() const;
  int GenerateFrTypeTraitsHdr() const;
  int GenerateExtFieldHdr() const;
  int GenerateExtFieldSrc() const;
  int GenerateExtFieldTypeTraitsHdr() const;
  int GeneratePointHdr(std::string_view g1_or_g2,
                       std::string_view fq_or_fq2) const;
  int GeneratePointSrc(std::string_view g1_or_g2,
                       std::string_view fq_or_fq2) const;
  int GeneratePointTraitsHdr(std::string_view g1_or_g2,
                             std::string_view fq_or_fq2) const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
  int GenerateG1TraitsHdr() const;
  int GenerateG2Hdr() const;
  int GenerateG2Src() const;
  int GenerateG2TraitsHdr() const;
  int GenerateMSMHdr() const;
  int GenerateMSMSrc() const;
  int GenerateMSMGpuHdr() const;
  int GenerateMSMGpuSrc() const;
};

int GenerationConfig::GeneratePrimeFieldHdr(std::string_view suffix) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(prime_field_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{type}", type},
          {"%{suffix}", suffix},
          {"%{limb_nums}",
           base::NumberToString(suffix == "fq" ? fq_limb_nums : fr_limb_nums)},
      });
  return WriteHdr(content, true);
}

int GenerationConfig::GeneratePrimeFieldSrc(std::string_view suffix) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(prime_field_src_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{suffix}", suffix},
                   });
  return WriteSrc(content);
}

int GenerationConfig::GeneratePrimeFieldTypeTraitsHdr(
    std::string_view suffix) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(prime_field_type_traits_hdr_tpl_path,
                               &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{suffix}", std::string(suffix)},
                       {"%{Suffix}", suffix == "fq" ? "Fq" : "Fr"},
                   });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateFqHdr() const {
  return GeneratePrimeFieldHdr("fq");
}

int GenerationConfig::GenerateFqSrc() const {
  return GeneratePrimeFieldSrc("fq");
}

int GenerationConfig::GenerateFqTypeTraitsHdr() const {
  return GeneratePrimeFieldTypeTraitsHdr("fq");
}

int GenerationConfig::GenerateFrHdr() const {
  return GeneratePrimeFieldHdr("fr");
}

int GenerationConfig::GenerateFrSrc() const {
  return GeneratePrimeFieldSrc("fr");
}

int GenerationConfig::GenerateFrTypeTraitsHdr() const {
  return GeneratePrimeFieldTypeTraitsHdr("fr");
}

int GenerationConfig::GenerateExtFieldHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(ext_field_hdr_tpl_path, &tpl_content));

  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, '\n');
  int degree_over_base_field = degree / base_field_degree;
  RemoveOptionalLines(tpl_lines, "IsCubicExtension",
                      degree_over_base_field == 3);
  tpl_content = absl::StrJoin(tpl_lines, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{header_dir_name}", c::math::GetLocation(type)},
          {"%{type}", type},
          {"%{degree}", base::NumberToString(degree)},
          {"%{base_field_degree}",
           base_field_degree == 1 ? ""
                                  : base::NumberToString(base_field_degree)},
      });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateExtFieldSrc() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(ext_field_src_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{header_dir_name}", c::math::GetLocation(type)},
          {"%{type}", type},
          {"%{degree}", degree == 1 ? "" : base::NumberToString(degree)},
      });
  return WriteSrc(content);
}

int GenerationConfig::GenerateExtFieldTypeTraitsHdr() const {
  std::string tpl_content;
  CHECK(
      base::ReadFileToString(ext_field_type_traits_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{type}", type},
          {"%{header_dir_name}", c::math::GetLocation(type)},
          {"%{degree}", degree == 1 ? "" : base::NumberToString(degree)},
      });
  return WriteHdr(content, false);
}

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

  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, '\n');
  RemoveOptionalLines(tpl_lines, "HasSpecializedG1MsmKernels",
                      has_specialized_g1_msm_kernels);
  tpl_content = absl::StrJoin(tpl_lines, "\n");

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
  parser.AddFlag<base::IntFlag>(&config.degree)
      .set_long_name("--degree")
      .set_required();
  parser.AddFlag<base::IntFlag>(&config.base_field_degree)
      .set_long_name("--base_field_degree")
      .set_required();
  parser.AddFlag<base::BoolFlag>(&config.has_specialized_g1_msm_kernels)
      .set_long_name("--has_specialized_g1_msm_kernels")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_hdr_tpl_path)
      .set_long_name("--prime_field_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_src_tpl_path)
      .set_long_name("--prime_field_src_tpl_path")
      .set_required();
  parser
      .AddFlag<base::FilePathFlag>(&config.prime_field_type_traits_hdr_tpl_path)
      .set_long_name("--prime_field_type_traits_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.ext_field_hdr_tpl_path)
      .set_long_name("--ext_field_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.ext_field_src_tpl_path)
      .set_long_name("--ext_field_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.ext_field_type_traits_hdr_tpl_path)
      .set_long_name("--ext_field_type_traits_hdr_tpl_path")
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

  std::string ext_field_name =
      absl::Substitute("fq$0", base::NumberToString(config.degree));

  if (base::EndsWith(config.out.value(), "fq.h")) {
    return config.GenerateFqHdr();
  } else if (base::EndsWith(config.out.value(), "fq.cc")) {
    return config.GenerateFqSrc();
  } else if (base::EndsWith(config.out.value(), "fq_type_traits.h")) {
    return config.GenerateFqTypeTraitsHdr();
  } else if (base::EndsWith(config.out.value(), "fr.h")) {
    return config.GenerateFrHdr();
  } else if (base::EndsWith(config.out.value(), "fr.cc")) {
    return config.GenerateFrSrc();
  } else if (base::EndsWith(config.out.value(), "fr_type_traits.h")) {
    return config.GenerateFrTypeTraitsHdr();
  } else if (base::EndsWith(config.out.value(), ext_field_name + ".h")) {
    return config.GenerateExtFieldHdr();
  } else if (base::EndsWith(config.out.value(), ext_field_name + ".cc")) {
    return config.GenerateExtFieldSrc();
  } else if (base::EndsWith(config.out.value(),
                            ext_field_name + "_type_traits.h")) {
    return config.GenerateExtFieldTypeTraitsHdr();
  } else if (base::EndsWith(config.out.value(), "g1.h")) {
    return config.GenerateG1Hdr();
  } else if (base::EndsWith(config.out.value(), "g1.cc")) {
    return config.GenerateG1Src();
  } else if (base::EndsWith(config.out.value(), "g1_point_traits.h")) {
    return config.GenerateG1TraitsHdr();
  } else if (base::EndsWith(config.out.value(), "g2.h")) {
    return config.GenerateG2Hdr();
  } else if (base::EndsWith(config.out.value(), "g2.cc")) {
    return config.GenerateG2Src();
  } else if (base::EndsWith(config.out.value(), "g2_point_traits.h")) {
    return config.GenerateG2TraitsHdr();
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
