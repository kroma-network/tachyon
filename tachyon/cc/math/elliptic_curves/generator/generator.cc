#include "absl/strings/str_replace.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_number_conversions.h"
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

int GenerationConfig::GenerateG1Hdr() const {
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include <string.h>",
      "",
      "#include <ostream>",
      "",
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/g1.h\"",
      "#include \"tachyon/cc/export.h\"",
      "#include \"tachyon/cc/math/elliptic_curves/%{header_dir_name}/fq.h\"",
      "",
      "namespace tachyon::cc::math::%{type} {",
      "",
      "class TACHYON_CC_EXPORT G1ProjectivePoint {",
      " public:",
      "  G1ProjectivePoint() = default;",
      "  explicit G1ProjectivePoint(const %{c_g1}_projective& point)",
      "      : G1ProjectivePoint(point.x, point.y, point.z) {}",
      "  G1ProjectivePoint(const %{c_fq}& x, const %{c_fq}& y, const %{c_fq}& z) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1ProjectivePoint(const Fq& x, const Fq& y, const Fq& z)",
      "      : x_(x), y_(y), z_(z) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  const Fq& z() const { return z_; }",
      "  Fq& z() { return z_; }",
      "",
      "%{projective_creation_ops}",
      "",
      "%{projective_binary_arithmetic_ops}",
      "",
      "  G1ProjectivePoint operator-() const {",
      "    return {x_, -y_, z_};",
      "  }",
      "",
      "%{projective_unary_arithmetic_ops}",
      "",
      "%{projective_equality_ops}",
      "",
      "  %{c_g1}_projective ToCPoint() const {"
      "    %{c_g1}_projective ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "  Fq z_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1ProjectivePoint& value);",
      "",
      "class TACHYON_CC_EXPORT G1JacobianPoint {",
      " public:",
      "  G1JacobianPoint() = default;",
      "  explicit G1JacobianPoint(const %{c_g1}_jacobian& point)",
      "      : G1JacobianPoint(point.x, point.y, point.z) {}",
      "  G1JacobianPoint(const %{c_fq}& x, const %{c_fq}& y, const %{c_fq}& z) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1JacobianPoint(const Fq& x, const Fq& y, const Fq& z)",
      "      : x_(x), y_(y), z_(z) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  const Fq& z() const { return z_; }",
      "  Fq& z() { return z_; }",
      "",
      "%{jacobian_creation_ops}",
      "",
      "%{jacobian_binary_arithmetic_ops}",
      "",
      "  G1JacobianPoint operator-() const {",
      "    return {x_, -y_, z_};",
      "  }",
      "",
      "%{jacobian_unary_arithmetic_ops}",
      "",
      "%{jacobian_equality_ops}",
      "",
      "  %{c_g1}_jacobian ToCPoint() const {"
      "    %{c_g1}_jacobian ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "  Fq z_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1JacobianPoint& value);",
      "",
      "class TACHYON_CC_EXPORT G1AffinePoint {",
      " public:",
      "  G1AffinePoint() = default;",
      "  explicit G1AffinePoint(const %{c_g1}_affine& point)",
      "      : G1AffinePoint(point.x, point.y, point.infinity) {}",
      "  G1AffinePoint(const %{c_fq}& x, const %{c_fq}& y, bool infinity = false)",
      "      : infinity_(infinity) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1AffinePoint(const Fq& x, const Fq& y, bool infinity = false)",
      "      : x_(x), y_(y), infinity_(infinity) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  const bool infinity() const { return infinity_; }",
      "  bool infinity() { return infinity_; }",
      "",
      "%{affine_creation_ops}",
      "",
      "%{affine_binary_arithmetic_ops}",
      "",
      "  G1AffinePoint operator-() const {",
      "    return {x_, -y_, infinity_};",
      "  }",
      "",
      "%{affine_unary_arithmetic_ops}",
      "",
      "%{affine_equality_ops}",
      "",
      "  %{c_g1}_affine ToCPoint() const {"
      "    %{c_g1}_affine ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    ret.infinity = infinity_;",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "  bool infinity_ = false;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1AffinePoint& value);",
      "",
      "class TACHYON_CC_EXPORT G1PointXYZZ {",
      " public:",
      "  G1PointXYZZ() = default;",
      "  explicit G1PointXYZZ(const %{c_g1}_xyzz& point)",
      "      : G1PointXYZZ(point.x, point.y, point.zz, point.zzz) {}",
      "  G1PointXYZZ(const %{c_fq}& x, const %{c_fq}& y, const %{c_fq}& zz, const %{c_fq}& zzz) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(zz_.value().limbs, zz.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(zzz_.value().limbs, zzz.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1PointXYZZ(const Fq& x, const Fq& y, const Fq& zz, const Fq& zzz)",
      "      : x_(x), y_(y), zz_(zz), zzz_(zzz) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  const Fq& zz() const { return zz_; }",
      "  Fq& zz() { return zz_; }",
      "",
      "  const Fq& zzz() const { return zzz_; }",
      "  Fq& zzz() { return zzz_; }",
      "",
      "%{xyzz_creation_ops}",
      "",
      "%{xyzz_binary_arithmetic_ops}",
      "",
      "  G1PointXYZZ operator-() const {",
      "    return {x_, -y_, zz_, zzz_};",
      "  }",
      "",
      "%{xyzz_unary_arithmetic_ops}",
      "",
      "%{xyzz_equality_ops}",
      "",
      "  %{c_g1}_xyzz ToCPoint() const {"
      "    %{c_g1}_xyzz ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.zz.limbs, zz_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.zzz.limbs, zzz_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "  Fq zz_;",
      "  Fq zzz_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1PointXYZZ& value);",
      "",
      "class TACHYON_CC_EXPORT G1Point2 {",
      " public:",
      "  G1Point2() = default;",
      "  G1Point2(const %{c_g1}_point2& point)",
      "      : G1Point2(point.x, point.y) {}",
      "  G1Point2(const %{c_fq}& x, const %{c_fq}& y) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1Point2(const Fq& x, const Fq& y)",
      "      : x_(x), y_(y) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  %{c_g1}_point2 ToCPoint() const {"
      "    %{c_g1}_point2 ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1Point2& value);",
      "",
      "class TACHYON_CC_EXPORT G1Point3 {",
      " public:",
      "  G1Point3() = default;",
      "  G1Point3(const %{c_g1}_point3& point)",
      "      : G1Point3(point.x, point.y, point.z) {}",
      "  G1Point3(const %{c_fq}& x, const %{c_fq}& y, const %{c_fq}& z) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1Point3(const Fq& x, const Fq& y, const Fq& z)",
      "      : x_(x), y_(y), z_(z) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  const Fq& z() const { return z_; }",
      "  Fq& z() { return z_; }",
      "",
      "  %{c_g1}_point3 ToCPoint() const {"
      "    %{c_g1}_point3 ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "  Fq z_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1Point2& value);",
      "",
      "class TACHYON_CC_EXPORT G1Point4 {",
      " public:",
      "  G1Point4() = default;",
      "  G1Point4(const %{c_g1}_point4& point)",
      "      : G1Point4(point.x, point.y, point.z, point.w) {}",
      "  G1Point4(const %{c_fq}& x, const %{c_fq}& y,const %{c_fq}& z, const %{c_fq}& w) {",
      "    memcpy(x_.value().limbs, x.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(y_.value().limbs, y.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(z_.value().limbs, z.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(w_.value().limbs, w.limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "  }",
      "  G1Point4(const Fq& x, const Fq& y, const Fq& z, const Fq& w)",
      "      : x_(x), y_(y), z_(z), w_(w) {}",
      "",
      "  const Fq& x() const { return x_; }",
      "  Fq& x() { return x_; }",
      "",
      "  const Fq& y() const { return y_; }",
      "  Fq& y() { return y_; }",
      "",
      "  const Fq& z() const { return z_; }",
      "  Fq& z() { return z_; }",
      "",
      "  const Fq& w() const { return w_; }",
      "  Fq& w() { return w_; }",
      "",
      "  %{c_g1}_point4 ToCPoint() const {",
      "    %{c_g1}_point4 ret;",
      "    memcpy(ret.x.limbs, x_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.y.limbs, y_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.z.limbs, z_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    memcpy(ret.w.limbs, w_.value().limbs, sizeof(uint64_t) * %{fq_limb_nums});",
      "    return ret;",
      "  }",
      "",
      "  std::string ToString() const;",
      "",
      " private:",
      "  Fq x_;",
      "  Fq y_;",
      "  Fq z_;",
      "  Fq w_;",
      "};",
      "",
      "TACHYON_CC_EXPORT std::ostream& operator<<(std::ostream& os, const G1Point4& value);",
      "",
      "} // namespace tachyon::cc::math::%{type}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  const char* kG1PointKinds[] = {"G1AffinePoint", "G1ProjectivePoint",
                                 "G1JacobianPoint", "G1PointXYZZ"};
  const char* kCG1PointKinds[] = {"%{c_g1}_affine", "%{c_g1}_projective",
                                  "%{c_g1}_jacobian", "%{c_g1}_xyzz"};
  std::vector<std::string> creation_ops;
  const char* kFieldCreationOps[] = {"Zero", "Generator", "Random"};
  const char* kCFieldCreationOps[] = {"zero", "generator", "random"};
  for (size_t i = 0; i < std::size(kG1PointKinds); ++i) {
    std::vector<std::string> creation_ops_components;
    for (size_t j = 0; j < std::size(kFieldCreationOps); ++j) {
      creation_ops_components.push_back(absl::Substitute(
          // clang-format off
            "  static $0 $1() {\n"
            "    return $0($2_$3());\n"
            "  }",
          // clang-format on
          kG1PointKinds[i], kFieldCreationOps[j], kCG1PointKinds[i],
          kCFieldCreationOps[j]));
      if (j != std::size(kFieldCreationOps) - 1) {
        creation_ops_components.push_back("");
      }
    }
    creation_ops.push_back(absl::StrJoin(creation_ops_components, "\n"));
  }

  std::vector<std::string> binary_arithmetic_ops;
  const char* kBinaryArithmeticOps[] = {"+", "-"};
  const char* kCBinaryArithmeticOps[] = {"add", "sub"};
  for (size_t i = 0; i < std::size(kG1PointKinds); ++i) {
    std::vector<std::string> binary_arithmetic_ops_components;
    for (size_t j = 0; j < std::size(kBinaryArithmeticOps); ++j) {
      binary_arithmetic_ops_components.push_back(absl::Substitute(
          // clang-format off
            "  $2 operator$1(const $0& other) const {\n"
            "    auto a = ToCPoint();\n"
            "    auto b = other.ToCPoint();\n"
            "    return $2($3_$4(&a, &b));\n"
            "  }",
          // clang-format on
          kG1PointKinds[i], kBinaryArithmeticOps[j],
          i == 0 ? "G1JacobianPoint" : kG1PointKinds[i], kCG1PointKinds[i],
          kCBinaryArithmeticOps[j]));
      if (i != 0) {
        binary_arithmetic_ops_components.push_back("");
        binary_arithmetic_ops_components.push_back(absl::Substitute(
            // clang-format off
              "  $0& operator$1=(const $0& other) {\n"
              "    auto a = ToCPoint();\n"
              "    auto b = other.ToCPoint();\n"
              "    *this = $0($2_$3(&a, &b));\n"
              "    return *this;\n"
              "  }",
            // clang-format on
            kG1PointKinds[i], kBinaryArithmeticOps[j], kCG1PointKinds[i],
            kCBinaryArithmeticOps[j]));
      }
      if (j != std::size(kBinaryArithmeticOps) - 1) {
        binary_arithmetic_ops_components.push_back("");
      }
    }
    binary_arithmetic_ops.push_back(
        absl::StrJoin(binary_arithmetic_ops_components, "\n"));
  }

  std::vector<std::string> unary_arithmetic_ops;
  const char* kUnaryArithmeticOps[] = {"Double"};
  const char* kCUnaryArithmeticOps[] = {"dbl"};
  for (size_t i = 0; i < std::size(kG1PointKinds); ++i) {
    std::vector<std::string> unary_arithmetic_ops_components;
    for (size_t j = 0; j < std::size(kUnaryArithmeticOps); ++j) {
      unary_arithmetic_ops_components.push_back(absl::Substitute(
          // clang-format off
            "  $0 $1() const {\n"
            "    auto a = ToCPoint();\n"
            "    return $0($2_$3(&a));\n"
            "  }",
          // clang-format on
          i == 0 ? "G1JacobianPoint" : kG1PointKinds[i], kUnaryArithmeticOps[j],
          kCG1PointKinds[i], kCUnaryArithmeticOps[j]));
      if (j != std::size(kUnaryArithmeticOps) - 1) {
        unary_arithmetic_ops_components.push_back("");
      }
    }
    unary_arithmetic_ops.push_back(
        absl::StrJoin(unary_arithmetic_ops_components, "\n"));
  }

  std::vector<std::string> equality_ops;
  const char* kEqualityOps[] = {"==", "!="};
  const char* kCEqualityOps[] = {"eq", "ne"};
  for (size_t i = 0; i < std::size(kG1PointKinds); ++i) {
    std::vector<std::string> equality_ops_components;
    for (size_t j = 0; j < std::size(kEqualityOps); ++j) {
      equality_ops_components.push_back(absl::Substitute(
          // clang-format off
            "  bool operator$1(const $0& other) const {\n"
            "    auto a = ToCPoint();\n"
            "    auto b = other.ToCPoint();\n"
            "    return $2_$3(&a, &b);\n"
            "  }",
          // clang-format on
          kG1PointKinds[i], kEqualityOps[j], kCG1PointKinds[i],
          kCEqualityOps[j]));
      if (j != std::size(kEqualityOps) - 1) {
        equality_ops_components.push_back("");
      }
    }
    equality_ops.push_back(absl::StrJoin(equality_ops_components, "\n"));
  }

  tpl_content = absl::StrReplaceAll(
      tpl_content,
      {
          {"%{affine_creation_ops}", creation_ops[0]},
          {"%{projective_creation_ops}", creation_ops[1]},
          {"%{jacobian_creation_ops}", creation_ops[2]},
          {"%{xyzz_creation_ops}", creation_ops[3]},
          {"%{affine_binary_arithmetic_ops}", binary_arithmetic_ops[0]},
          {"%{projective_binary_arithmetic_ops}", binary_arithmetic_ops[1]},
          {"%{jacobian_binary_arithmetic_ops}", binary_arithmetic_ops[2]},
          {"%{xyzz_binary_arithmetic_ops}", binary_arithmetic_ops[3]},
          {"%{affine_unary_arithmetic_ops}", unary_arithmetic_ops[0]},
          {"%{projective_unary_arithmetic_ops}", unary_arithmetic_ops[1]},
          {"%{jacobian_unary_arithmetic_ops}", unary_arithmetic_ops[2]},
          {"%{xyzz_unary_arithmetic_ops}", unary_arithmetic_ops[3]},
          {"%{affine_equality_ops}", equality_ops[0]},
          {"%{projective_equality_ops}", equality_ops[1]},
          {"%{jacobian_equality_ops}", equality_ops[2]},
          {"%{xyzz_equality_ops}", equality_ops[3]},
      });

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
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/fq_prime_field_traits.h\"",
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_traits.h\"",
      "#include \"tachyon/cc/math/elliptic_curves/point_conversions.h\"",
      "",
      "namespace tachyon::cc::math::%{type} {",
      "",
      "std::string G1AffinePoint::ToString() const {",
      "  return ToAffinePoint(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1AffinePoint& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "std::string G1ProjectivePoint::ToString() const {",
      "  return ToProjectivePoint(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1ProjectivePoint& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "std::string G1JacobianPoint::ToString() const {",
      "  return ToJacobianPoint(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1JacobianPoint& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "std::string G1PointXYZZ::ToString() const {",
      "  return ToPointXYZZ(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1PointXYZZ& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "std::string G1Point2::ToString() const {",
      "  return ToPoint2(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1Point2& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "std::string G1Point3::ToString() const {",
      "  return ToPoint3(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1Point3& value) {",
      "  return os << value.ToString();",
      "}",
      "",
      "std::string G1Point4::ToString() const {",
      "  return ToPoint4(ToCPoint()).ToString();",
      "}",
      "",
      "std::ostream& operator<<(std::ostream& os, const G1Point4& value) {",
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
