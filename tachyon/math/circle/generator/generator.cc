#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/math/finite_fields/generator/generator_util.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  base::FilePath cpu_hdr_tpl_path;

  std::string ns_name;
  std::string class_name;
  std::string base_field;
  base::FilePath base_field_hdr;
  std::string scalar_field;
  base::FilePath scalar_field_hdr;
  std::string x;
  std::string y;

  int GenerateConfigHdr() const;
};

int GenerationConfig::GenerateConfigHdr() const {
  absl::flat_hash_map<std::string, std::string> replacements = {
      {"%{namespace}", ns_name},
      {"%{class}", class_name},
      {"%{base_field}", base_field},
      {"%{base_field_hdr}", base_field_hdr.value()},
      {"%{scalar_field}", scalar_field},
      {"%{scalar_field_hdr}", scalar_field_hdr.value()},
      {"%{x_init}", math::GenerateInitField("kGenerator.x", "BaseField", x)},
      {"%{y_init}", math::GenerateInitField("kGenerator.y", "BaseField", y)}};

  std::string tpl_content;
  CHECK(base::ReadFileToString(cpu_hdr_tpl_path, &tpl_content));

  std::string content = absl::StrReplaceAll(tpl_content, replacements);
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/math/circle/generator";

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
  parser.AddFlag<base::FilePathFlag>(&config.cpu_hdr_tpl_path)
      .set_long_name("--cpu_hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.x)
      .set_short_name("-x")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.y)
      .set_short_name("-y")
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
