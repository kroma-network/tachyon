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
      "#include \"tachyon/py/base/pybind11.h\"",
      "",
      "namespace tachyon::py::math::%{type} {",
      "",
      "void Add%{cc_field}(py11::module& m);",
      "",
      "} // namespace tachyon::py::math::%{type}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{type}", type},
                       {"%{cc_field}", suffix == "fq" ? "Fq" : "Fr"},
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
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h\"",
      "#include \"tachyon/py/math/finite_fields/prime_field.h\"",
      "",
      "namespace tachyon::py::math::%{type} {",
      "",
      "void Add%{cc_field}(py11::module& m) {",
      "  AddPrimeField<tachyon::math::%{type}::%{cc_field}>(m, \"%{cc_field}\");",
      "}",
      "",
      "} // namespace tachyon::py::math::%{type}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{suffix}", suffix},
                       {"%{type}", type},
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
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/py/base/pybind11.h\"",
      "",
      "namespace tachyon::py::math::%{type} {",
      "",
      "void AddG1(py11::module& m);",
      "",
      "} // namespace tachyon::py::math::%{type}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(tpl_content, {
                                                             {"%{type}", type},
                                                         });
  return WriteHdr(content, false);
}

int GenerationConfig::GenerateG1Src() const {
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/g1.h\"",
      "#include \"tachyon/py/math/elliptic_curves/short_weierstrass/affine_point.h\"",
      "#include \"tachyon/py/math/elliptic_curves/short_weierstrass/projective_point.h\"",
      "#include \"tachyon/py/math/elliptic_curves/short_weierstrass/jacobian_point.h\"",
      "#include \"tachyon/py/math/elliptic_curves/short_weierstrass/point_xyzz.h\"",
      "",
      "namespace tachyon::py::math::%{type} {",
      "",
      "void AddG1(py11::module& m) {",
      "  AddJacobianPoint<tachyon::math::%{type}::G1JacobianPoint>(m, \"G1JacobianPoint\");",
      "  AddAffinePoint<tachyon::math::%{type}::G1AffinePoint>(m, \"G1AffinePoint\");",
      "  AddProjectivePoint<tachyon::math::%{type}::G1ProjectivePoint>(m, \"G1ProjectivePoint\");",
      "  AddPointXYZZ<tachyon::math::%{type}::G1PointXYZZ>(m, \"G1PointXYZZ\");",
      "}",
      "",
      "} // namespace tachyon::py::math::%{type}",
  };
  // clang-format on

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
  config.generator = "//tachyon/py/math/elliptic_curves/generator";

  base::FlagParser parser;
  parser.AddFlag<base::FilePathFlag>(&config.out)
      .set_long_name("--out")
      .set_help("path to output");
  parser.AddFlag<base::StringFlag>(&config.type)
      .set_long_name("--type")
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
