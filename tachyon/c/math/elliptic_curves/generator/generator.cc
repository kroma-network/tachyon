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
  base::FilePath prime_field_traits_hdr_tpl_path;
  base::FilePath g1_hdr_tpl_path;
  base::FilePath g1_src_tpl_path;
  base::FilePath g1_traits_hdr_tpl_path;
  base::FilePath msm_hdr_tpl_path;
  base::FilePath msm_src_tpl_path;
  base::FilePath msm_gpu_hdr_tpl_path;
  base::FilePath msm_gpu_src_tpl_path;

  std::string type;
  int fq_limb_nums;
  int fr_limb_nums;
  bool has_specialized_g1_msm_kernels;

  int GeneratePrimeFieldHdr(std::string_view suffix) const;
  int GeneratePrimeFieldSrc(std::string_view suffix) const;
  int GeneratePrimeFieldTraitsHdr(std::string_view suffix) const;
  int GenerateFqHdr() const;
  int GenerateFqSrc() const;
  int GenerateFqTraitsHdr() const;
  int GenerateFrHdr() const;
  int GenerateFrSrc() const;
  int GenerateFrTraitsHdr() const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
  int GenerateG1TraitsHdr() const;
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

int GenerationConfig::GeneratePrimeFieldTraitsHdr(
    std::string_view suffix) const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(prime_field_traits_hdr_tpl_path, &tpl_content));

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

int GenerationConfig::GenerateFqTraitsHdr() const {
  return GeneratePrimeFieldTraitsHdr("fq");
}

int GenerationConfig::GenerateFrHdr() const {
  return GeneratePrimeFieldHdr("fr");
}

int GenerationConfig::GenerateFrSrc() const {
  return GeneratePrimeFieldSrc("fr");
}

int GenerationConfig::GenerateFrTraitsHdr() const {
  return GeneratePrimeFieldTraitsHdr("fr");
}

int GenerationConfig::GenerateG1Hdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(g1_hdr_tpl_path, &tpl_content));

  base::FilePath hdr_path = GetHdrPath();
  std::string basename = hdr_path.BaseName().value();
  std::string header_path = hdr_path.DirName().Append("fq.h").value();
  std::string content = absl::StrReplaceAll(
      tpl_content, {{"%{header_path}", header_path}, {"%{type}", type}});
  return WriteHdr(content, true);
}

int GenerationConfig::GenerateG1Src() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(g1_src_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {{"%{header_dir_name}", c::math::GetLocation(type)}, {"%{type}", type}});
  return WriteSrc(content);
}

int GenerationConfig::GenerateG1TraitsHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(g1_traits_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {{"%{header_dir_name}", c::math::GetLocation(type)}, {"%{type}", type}});
  return WriteHdr(content, false);
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
  CHECK(base::ReadFileToString(msm_gpu_src_tpl_path, &tpl_content));

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
  parser.AddFlag<base::BoolFlag>(&config.has_specialized_g1_msm_kernels)
      .set_long_name("--has_specialized_g1_msm_kernels")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_hdr_tpl_path)
      .set_long_name("--prime_field_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_src_tpl_path)
      .set_long_name("--prime_field_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.prime_field_traits_hdr_tpl_path)
      .set_long_name("--prime_field_traits_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g1_hdr_tpl_path)
      .set_long_name("--g1_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g1_src_tpl_path)
      .set_long_name("--g1_src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.g1_traits_hdr_tpl_path)
      .set_long_name("--g1_traits_hdr_tpl_path")
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

  if (base::EndsWith(config.out.value(), "fq.h")) {
    return config.GenerateFqHdr();
  } else if (base::EndsWith(config.out.value(), "fq.cc")) {
    return config.GenerateFqSrc();
  } else if (base::EndsWith(config.out.value(), "fq_prime_field_traits.h")) {
    return config.GenerateFqTraitsHdr();
  } else if (base::EndsWith(config.out.value(), "fr.h")) {
    return config.GenerateFrHdr();
  } else if (base::EndsWith(config.out.value(), "fr.cc")) {
    return config.GenerateFrSrc();
  } else if (base::EndsWith(config.out.value(), "fr_prime_field_traits.h")) {
    return config.GenerateFrTraitsHdr();
  } else if (base::EndsWith(config.out.value(), "g1.h")) {
    return config.GenerateG1Hdr();
  } else if (base::EndsWith(config.out.value(), "g1.cc")) {
    return config.GenerateG1Src();
  } else if (base::EndsWith(config.out.value(), "g1_point_traits.h")) {
    return config.GenerateG1TraitsHdr();
  } else if (base::EndsWith(config.out.value(), "msm.h")) {
    return config.GenerateMSMHdr();
  } else if (base::EndsWith(config.out.value(), "msm.cc")) {
    return config.GenerateMSMSrc();
  } else if (base::EndsWith(config.out.value(), "msm_gpu.h")) {
    return config.GenerateMSMGpuHdr();
  } else if (base::EndsWith(config.out.value(), "msm_gpu.cc")) {
    return config.GenerateMSMGpuSrc();
  } else {
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
