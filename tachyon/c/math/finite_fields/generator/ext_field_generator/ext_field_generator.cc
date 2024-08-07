#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/build/cc_writer.h"

namespace tachyon {

struct GenerationConfig : public build::CcWriter {
  base::FilePath hdr_tpl_path;
  base::FilePath src_tpl_path;
  base::FilePath type_traits_hdr_tpl_path;

  std::string class_name;
  std::string base_field_class_name;
  std::string display_name;
  std::string base_field_display_name;
  std::string c_base_field_hdr;
  std::string native_type;
  std::string native_hdr;
  int degree_over_base_field;

  int GenerateHdr() const;
  int GenerateSrc() const;
  int GenerateTypeTraitsHdr() const;
};

int GenerationConfig::GenerateHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(hdr_tpl_path, &tpl_content));

  std::vector<std::string> tpl_lines = absl::StrSplit(tpl_content, '\n');
  RemoveOptionalLines(tpl_lines, "IsCubicExtension",
                      degree_over_base_field == 3);
  RemoveOptionalLines(tpl_lines, "IsQuarticExtension",
                      degree_over_base_field == 4);
  tpl_content = absl::StrJoin(tpl_lines, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{class_name}", class_name},
                       {"%{base_field_class_name}", base_field_class_name},
                       {"%{display_name}", display_name},
                       {"%{base_field_display_name}", base_field_display_name},
                       {"%{c_base_field_hdr}", c_base_field_hdr},
                   });
  return WriteHdr(content, true);
}

int GenerationConfig::GenerateSrc() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(src_tpl_path, &tpl_content));

  const base::FilePath& hdr_path = GetHdrPath();
  std::string basename = GetHdrPath().BaseName().value();
  basename = basename.substr(0, basename.find(".h"));
  base::FilePath c_type_traits_hdr =
      GetHdrPath().DirName().Append(basename + "_type_traits.h");
  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{class_name}", class_name},
                       {"%{c_type_traits_hdr}", c_type_traits_hdr.value()},
                   });
  return WriteSrc(content);
}

int GenerationConfig::GenerateTypeTraitsHdr() const {
  std::string tpl_content;
  CHECK(base::ReadFileToString(type_traits_hdr_tpl_path, &tpl_content));

  const base::FilePath& hdr_path = GetHdrPath();
  std::string basename = GetHdrPath().BaseName().value();
  basename = basename.substr(0, basename.find("_type_traits"));
  base::FilePath c_hdr = GetHdrPath().DirName().Append(basename + ".h");
  std::string content =
      absl::StrReplaceAll(tpl_content, {
                                           {"%{class_name}", class_name},
                                           {"%{c_hdr}", c_hdr.value()},
                                           {"%{native_hdr}", native_hdr},
                                           {"%{native_type}", native_type},
                                       });
  return WriteHdr(content, false);
}

int RealMain(int argc, char** argv) {
  GenerationConfig config;
  config.generator = "//tachyon/c/math/finite_fields/ext_field_generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.class_name)
      .set_long_name("--class_name")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.base_field_class_name)
      .set_long_name("--base_field_class_name")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.display_name)
      .set_long_name("--display_name")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.base_field_display_name)
      .set_long_name("--base_field_display_name")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.c_base_field_hdr)
      .set_long_name("--c_base_field_hdr")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.native_type)
      .set_long_name("--native_type")
      .set_required();
  parser.AddFlag<base::StringFlag>(&config.native_hdr)
      .set_long_name("--native_hdr")
      .set_required();
  parser.AddFlag<base::IntFlag>(&config.degree_over_base_field)
      .set_long_name("--degree_over_base_field")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.hdr_tpl_path)
      .set_long_name("--hdr_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.src_tpl_path)
      .set_long_name("--src_tpl_path")
      .set_required();
  parser.AddFlag<base::FilePathFlag>(&config.type_traits_hdr_tpl_path)
      .set_long_name("--type_traits_hdr_tpl_path")
      .set_required();

  std::string error;
  if (!parser.Parse(argc, argv, &error)) {
    tachyon_cerr << error << std::endl;
    return 1;
  }

  if (base::EndsWith(config.out.value(), "_type_traits.h")) {
    return config.GenerateTypeTraitsHdr();
  } else if (base::EndsWith(config.out.value(), ".h")) {
    return config.GenerateHdr();
  } else if (base::EndsWith(config.out.value(), ".cc")) {
    return config.GenerateSrc();
  } else {
    tachyon_cerr << "suffix not supported:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
