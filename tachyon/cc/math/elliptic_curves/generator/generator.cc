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
  int GeneratePrimeFieldSrc(std::string_view suffix) const;
  int GenerateFqSrc() const;
  int GenerateFrSrc() const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
};

int GenerationConfig::GeneratePrimeFieldHdr(std::string_view suffix) const {
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include <string.h>",
      "",
      "#include <ostream>",
      "",
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}.h\"",
      "#include \"tachyon/cc/export.h\"",
      "",
      "namespace tachyon::cc::math::%{type} {",
      "",
      "class TACHYON_CC_EXPORT %{cc_field} {",
      " public:",
      "  %{cc_field}() = default;",
      "  explicit %{cc_field}(%{c_field} value) {",
      "    memcpy(value_.limbs, value.limbs, sizeof(uint64_t) * %{limb_nums});",
      "  }",
      "",
      "  const %{c_field}& value() const { return value_; }",
      "  %{c_field}& value() { return value_; }",
      "",
      "  const %{c_field}* value_ptr() const { return &value_; }",
      "  %{c_field}* value_ptr() { return &value_; }",
      "",
      "%{creation_ops}",
      "",
      "%{binary_arithmetic_ops}",
      "",
      "  %{cc_field} operator-() const {",
      "    return %{cc_field}(%{c_field}_neg(&value_));",
      "  }",
      "",
      "%{unary_arithmetic_ops}",
      "",
      "%{comparison_ops}",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  %{c_field} value_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const %{cc_field}& value);",
      "",
      "} // namespace tachyon::cc::math::%{type}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string creation_ops;
  std::vector<std::string> creation_ops_components;
  const char* kFieldCreationOps[] = {"Zero", "One", "Random"};
  const char* kCFieldCreationOps[] = {"zero", "one", "random"};
  for (size_t i = 0; i < std::size(kFieldCreationOps); ++i) {
    creation_ops_components.push_back(absl::Substitute(
        // clang-format off
          "  static %{cc_field} $0() {\n"
          "    return %{cc_field}(%{c_field}_$1());\n"
          "  }",
        // clang-format on
        kFieldCreationOps[i], kCFieldCreationOps[i]));
    if (i != std::size(kFieldCreationOps) - 1) {
      creation_ops_components.push_back("");
    }
  }
  creation_ops = absl::StrJoin(creation_ops_components, "\n");

  std::string binary_arithmetic_ops;
  std::vector<std::string> binary_arithmetic_ops_components;
  const char* kFieldBinaryArithmeticOps[] = {"+", "-", "*", "/"};
  const char* kCFieldBinaryArithmeticOps[] = {"add", "sub", "mul", "div"};
  for (size_t i = 0; i < std::size(kFieldBinaryArithmeticOps); ++i) {
    binary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "  %{cc_field} operator$0(const %{cc_field}& other) const {\n"
          "    return %{cc_field}(%{c_field}_$1(&value_, &other.value_));\n"
          "  }",
        // clang-format on
        kFieldBinaryArithmeticOps[i], kCFieldBinaryArithmeticOps[i]));
    binary_arithmetic_ops_components.push_back("");
    binary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "  %{cc_field}& operator$0=(const %{cc_field}& other) {\n"
          "    value_ = %{c_field}_$1(&value_, &other.value_);\n"
          "    return *this;\n"
          "  }",
        // clang-format on
        kFieldBinaryArithmeticOps[i], kCFieldBinaryArithmeticOps[i]));
    if (i != std::size(kFieldBinaryArithmeticOps) - 1) {
      binary_arithmetic_ops_components.push_back("");
    }
  }
  binary_arithmetic_ops = absl::StrJoin(binary_arithmetic_ops_components, "\n");

  std::string unary_arithmetic_ops;
  std::vector<std::string> unary_arithmetic_ops_components;
  const char* kFieldUnaryArithmeticOps[] = {"Double", "Square", "Inverse"};
  const char* kCFieldUnaryArithmeticOps[] = {"dbl", "sqr", "inv"};
  for (size_t i = 0; i < std::size(kFieldUnaryArithmeticOps); ++i) {
    unary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "  %{cc_field} $0() const {\n"
          "    return %{cc_field}(%{c_field}_$1(&value_));\n"
          "  }",
        // clang-format on
        kFieldUnaryArithmeticOps[i], kCFieldUnaryArithmeticOps[i]));
    if (i != std::size(kFieldUnaryArithmeticOps) - 1) {
      unary_arithmetic_ops_components.push_back("");
    }
  }
  unary_arithmetic_ops = absl::StrJoin(unary_arithmetic_ops_components, "\n");

  std::string comparison_ops;
  std::vector<std::string> comparison_ops_components;
  const char* kFieldComparisonOps[] = {"==", "!=", ">", ">=", "<", "<="};
  const char* kCFieldComparisonOps[] = {"eq", "ne", "gt", "ge", "lt", "le"};
  for (size_t i = 0; i < std::size(kFieldComparisonOps); ++i) {
    comparison_ops_components.push_back(absl::Substitute(
        // clang-format off
          "  bool operator$0(const %{cc_field}& other) const {\n"
          "    return %{c_field}_$1(&value_, &other.value_);\n"
          "  }",
        // clang-format on
        kFieldComparisonOps[i], kCFieldComparisonOps[i]));
    if (i != std::size(kFieldComparisonOps) - 1) {
      comparison_ops_components.push_back("");
    }
  }
  comparison_ops = absl::StrJoin(comparison_ops_components, "\n");

  tpl_content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{creation_ops}", creation_ops},
                       {"%{binary_arithmetic_ops}", binary_arithmetic_ops},
                       {"%{unary_arithmetic_ops}", unary_arithmetic_ops},
                       {"%{comparison_ops}", comparison_ops},
                   });

  std::string content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{header_dir_name}", c::math::GetLocation(type)},
          {"%{type}", type},
          {"%{suffix}", suffix},
          {"%{c_field}", absl::Substitute("tachyon_$0_$1", type, suffix)},
          {"%{cc_field}", suffix == "fq" ? "Fq" : "Fr"},
          {"%{limb_nums}",
           absl::StrCat(suffix == "fq" ? fq_limb_nums : fr_limb_nums)},
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
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}_prime_field_traits.h\"",
      "#include \"tachyon/cc/math/finite_fields/prime_field_conversions.h\"",
      "",
      "namespace tachyon::cc::math::%{type} {",
      "",
      "std::string %{cc_field}::ToString() const {",
      "  return ToBigInt(value_).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const %{cc_field}& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "} // namespace tachyon::cc::math::%{type}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

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

int GenerationConfig::GenerateG1Hdr() const { return WriteHdr("", false); }

int GenerationConfig::GenerateG1Src() const { return WriteSrc(""); }

int RealMain(int argc, char** argv) {
  GenerationConfig config;

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
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
