#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/base/gmp/bit_traits.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"

namespace tachyon {

size_t GetNumBits(const mpz_class& m) {
  auto it = math::BitIteratorBE<mpz_class>::begin(&m, true);
  auto end = math::BitIteratorBE<mpz_class>::end(&m);
  size_t num_bits = 0;
  while (it != end) {
    ++it;
    ++num_bits;
  }
  return num_bits;
}
struct GenerationConfig : public build::CcWriter {
  base::FilePath binary_config_hdr_tpl_path;
  base::FilePath binary_cpu_hdr_tpl_path;
  base::FilePath binary_gpu_hdr_tpl_path;

  std::string ns_name;
  std::string class_name;
  std::string modulus;

  int GenerateConfigHdr() const;
  int GenerateCpuHdr() const;
  int GenerateGpuHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
  };

  mpz_class m = math::gmp::FromDecString(modulus);
  size_t num_bits = GetNumBits(m);
  if (num_bits == 64 + 1) {
    replacements["%{modulus}"] =
        absl::Substitute("BigInt<2>{$0}", math::MpzClassToString(m));
    replacements["%{modulus_type}"] = "BigInt<2>";
    replacements["%{value_type}"] = "uint64_t";
    replacements["%{value}"] = "1";
  } else {
    replacements["%{modulus}"] = math::MpzClassToString(m);
    std::string modulus_type;
    std::string value_type;
    switch (num_bits) {
      case 1 + 1:
      case 2 + 1:
      case 4 + 1:
        modulus_type = "uint8_t";
        value_type = "uint8_t";
        break;
      case 8 + 1:
        modulus_type = "uint16_t";
        value_type = "uint8_t";
        break;
      case 16 + 1:
        modulus_type = "uint32_t";
        value_type = "uint16_t";
        break;
      case 32 + 1:
        modulus_type = "uint64_t";
        value_type = "uint32_t";
        break;
      default:
        NOTREACHED();
    }
    replacements["%{modulus_type}"] = modulus_type;
    replacements["%{value_type}"] = value_type;
    replacements["%{value}"] = "1";
  }
  replacements["%{modulus_bits}"] = base::NumberToString(num_bits);

  std::string tpl_content;
  CHECK(base::ReadFileToString(binary_config_hdr_tpl_path, &tpl_content));

  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, '\n');
  RemoveOptionalLines(tpl_lines, "NeedsBigInt", num_bits == 64 + 1);

  tpl_content = absl::StrJoin(tpl_lines, "\n");
  std::string content =
      absl::StrReplaceAll(tpl_content, std::move(replacements));

  return WriteHdr(content, false);
}

int GenerationConfig::GenerateCpuHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(binary_cpu_hdr_tpl_path, &tpl_content));

  base::FilePath hdr_path = GetHdrPath();
  base::FilePath basename = hdr_path.BaseName().RemoveExtension();
  base::FilePath config_header_path =
      hdr_path.DirName().Append(basename.value() + "_config.h");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{config_header_path}", config_header_path.value()},
                       {"%{namespace}", ns_name},
                       {"%{class}", class_name},
                   });

  return WriteHdr(content, false);
}

int GenerationConfig::GenerateGpuHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(binary_gpu_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{config_header_path}",
                        math::ConvertToConfigHdr(GetHdrPath()).value()},
                       {"%{namespace}", ns_name},
                       {"%{class}", class_name},
                   });

  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/finite_fields/prime_field_field_generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.ns_name)
      .set_long_name("--namespace")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.class_name).set_long_name("--class");
  parser.AddFlag<base::StringFlag>(&config.modulus)
      .set_long_name("--modulus")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.binary_config_hdr_tpl_path)
      .set_long_name("--binary_config_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.binary_cpu_hdr_tpl_path)
      .set_long_name("--binary_cpu_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.binary_gpu_hdr_tpl_path)
      .set_long_name("--binary_gpu_hdr_tpl_path")
      .set_required();

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "_config.h")) {
    return config.GenerateConfigHdr();
  } else if (base::EndsWith(config.out.value(), "_gpu.h")) {
    return config.GenerateGpuHdr();
  } else if (base::EndsWith(config.out.value(), ".h")) {
    return config.GenerateCpuHdr();
  } else {
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
