#include "absl/strings/str_replace.h"
#include "absl/types/span.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/files/file_path_flag.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/cc_writer.h"
#include "tachyon/c/math/elliptic_curves/generator/generator_util.h"

namespace tachyon {

const char* kComparisonOps[] = {"eq", "ne", "gt", "ge", "lt", "le"};
const char* kEqualityOps[] = {"eq", "ne"};

const char* kFieldBinaryArithmeticOps[] = {"add", "sub", "mul", "div"};
const char* kFieldUnaryArithmeticOps[] = {"neg", "dbl", "sqr", "inv"};
const char* kFieldCreationOps[] = {"zero", "one", "random"};

const char* kPointCreationOps[] = {"zero", "generator", "random"};

const char* kG1PointKinds[] = {"%{g1}_affine", "%{g1}_projective",
                               "%{g1}_jacobian", "%{g1}_xyzz"};

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
  int GeneratePrimeFieldTraitsHdr(std::string_view suffix) const;
  int GenerateFqTraitsHdr() const;
  int GenerateFrTraitsHdr() const;
  int GenerateG1TraitsHdr() const;
  int GenerateG1Hdr() const;
  int GenerateG1Src() const;
};

int GenerationConfig::GeneratePrimeFieldHdr(std::string_view suffix) const {
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include <stdint.h>",
      "",
      "#include \"tachyon/c/export.h\"",
      "",
      "struct TACHYON_C_EXPORT %{field} {",
      "  uint64_t limbs[%{limb_nums}];",
      "};",
      "",
      "%{creation_ops}",
      "",
      "%{binary_arithmetic_ops}",
      "",
      "%{unary_arithmetic_ops}",
      "",
      "%{comparison_ops}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string creation_ops;
  std::vector<std::string> creation_ops_components;
  for (size_t i = 0; i < std::size(kFieldCreationOps); ++i) {
    creation_ops_components.push_back(absl::Substitute(
        // clang-format off
          "TACHYON_C_EXPORT %{field} %{field}_$0();",
        // clang-format on
        kFieldCreationOps[i]));
    if (i != std::size(kFieldCreationOps) - 1) {
      creation_ops_components.push_back("");
    }
  }
  creation_ops = absl::StrJoin(creation_ops_components, "\n");

  std::string binary_arithmetic_ops;
  std::vector<std::string> binary_arithmetic_ops_components;
  for (size_t i = 0; i < std::size(kFieldBinaryArithmeticOps); ++i) {
    binary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "TACHYON_C_EXPORT %{field} %{field}_$0(const %{field}* a, const %{field}* b);",
        // clang-format on
        kFieldBinaryArithmeticOps[i]));
    if (i != std::size(kFieldBinaryArithmeticOps) - 1) {
      binary_arithmetic_ops_components.push_back("");
    }
  }
  binary_arithmetic_ops = absl::StrJoin(binary_arithmetic_ops_components, "\n");

  std::string unary_arithmetic_ops;
  std::vector<std::string> unary_arithmetic_ops_components;
  for (size_t i = 0; i < std::size(kFieldUnaryArithmeticOps); ++i) {
    unary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "TACHYON_C_EXPORT %{field} %{field}_$0(const %{field}* a);",
        // clang-format on
        kFieldUnaryArithmeticOps[i]));
    if (i != std::size(kFieldUnaryArithmeticOps) - 1) {
      unary_arithmetic_ops_components.push_back("");
    }
  }
  unary_arithmetic_ops = absl::StrJoin(unary_arithmetic_ops_components, "\n");

  std::string comparison_ops;
  std::vector<std::string> comparison_ops_components;
  for (size_t i = 0; i < std::size(kComparisonOps); ++i) {
    comparison_ops_components.push_back(absl::Substitute(
        // clang-format off
          "TACHYON_C_EXPORT bool %{field}_$0(const %{field}* a, const %{field}* b);",
        // clang-format on
        kComparisonOps[i]));
    if (i != std::size(kComparisonOps) - 1) {
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
          {"%{field}", absl::Substitute("tachyon_$0_$1", type, suffix)},
          {"%{limb_nums}",
           absl::StrCat(suffix == "fq" ? fq_limb_nums : fr_limb_nums)},
      });
  return WriteHdr(content);
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
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h\"",
      "",
      "%{creation_ops}",
      "",
      "%{binary_arithmetic_ops}",
      "",
      "%{unary_arithmetic_ops}",
      "",
      "%{comparison_ops}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string creation_ops;
  std::vector<std::string> creation_ops_components;
  const char* kUpperCreationOps[] = {"Zero", "One", "Random"};
  for (size_t i = 0; i < std::size(kFieldCreationOps); ++i) {
    creation_ops_components.push_back(absl::Substitute(
        // clang-format off
          "%{field} %{field}_$0() {\n"
          "  using namespace tachyon::cc::math;\n"
          "  using PrimeFieldTy = typename PrimeFieldTraits<%{field}>::PrimeFieldTy;\n"
          "  return ToCPrimeField(PrimeFieldTy::$1());\n"
          "}",
        // clang-format on
        kFieldCreationOps[i], kUpperCreationOps[i]));
    if (i != std::size(kFieldCreationOps) - 1) {
      creation_ops_components.push_back("");
    }
  }
  creation_ops = absl::StrJoin(creation_ops_components, "\n");

  std::string binary_arithmetic_ops;
  std::vector<std::string> binary_arithmetic_ops_components;
  const char* kUpperBinaryArithmeticOps[] = {"Add", "Sub", "Mul", "Div"};
  for (size_t i = 0; i < std::size(kFieldBinaryArithmeticOps); ++i) {
    binary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "%{field} %{field}_$0(const %{field}* a, const %{field}* b) {\n"
          "  using namespace tachyon::cc::math;\n"
          "  return ToCPrimeField(ToPrimeField(*a).$1InPlace(ToPrimeField(*b)));\n"
          "}",
        // clang-format on
        kFieldBinaryArithmeticOps[i], kUpperBinaryArithmeticOps[i]));
    if (i != std::size(kFieldBinaryArithmeticOps) - 1) {
      binary_arithmetic_ops_components.push_back("");
    }
  }
  binary_arithmetic_ops = absl::StrJoin(binary_arithmetic_ops_components, "\n");

  std::string unary_arithmetic_ops;
  std::vector<std::string> unary_arithmetic_ops_components;
  const char* kUpperUnaryArithmeticOps[] = {"Neg", "Double", "Square",
                                            "Inverse"};
  for (size_t i = 0; i < std::size(kFieldUnaryArithmeticOps); ++i) {
    unary_arithmetic_ops_components.push_back(absl::Substitute(
        // clang-format off
          "%{field} %{field}_$0(const %{field}* a) {\n"
          "  using namespace tachyon::cc::math;\n"
          "  return ToCPrimeField(ToPrimeField(*a).$1InPlace());\n"
          "}",
        // clang-format on
        kFieldUnaryArithmeticOps[i], kUpperUnaryArithmeticOps[i]));
    if (i != std::size(kFieldUnaryArithmeticOps) - 1) {
      unary_arithmetic_ops_components.push_back("");
    }
  }
  unary_arithmetic_ops = absl::StrJoin(unary_arithmetic_ops_components, "\n");

  std::string comparison_ops;
  std::vector<std::string> comparison_ops_components;
  const char* kBinaryComparisonSymbols[] = {"==", "!=", ">", ">=", "<", "<="};
  for (size_t i = 0; i < std::size(kComparisonOps); ++i) {
    comparison_ops_components.push_back(absl::Substitute(
        // clang-format off
          "bool %{field}_$0(const %{field}* a, const %{field}* b) {\n"
          "  using namespace tachyon::cc::math;\n"
          "  return ToPrimeField(*a) $1 ToPrimeField(*b);\n"
          "}",
        // clang-format on
        kComparisonOps[i], kBinaryComparisonSymbols[i]));
    if (i != std::size(kComparisonOps) - 1) {
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
          {"%{suffix}", suffix},
          {"%{field}", absl::Substitute("tachyon_$0_$1", type, suffix)},
      });
  return WriteSrc(content);
}

int GenerationConfig::GenerateFqSrc() const {
  return GeneratePrimeFieldSrc("fq");
}

int GenerationConfig::GenerateFrSrc() const {
  return GeneratePrimeFieldSrc("fr");
}

int GenerationConfig::GeneratePrimeFieldTraitsHdr(
    std::string_view suffix) const {
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}.h\"",
      "#include \"tachyon/cc/math/finite_fields/prime_field_traits.h\"",
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h\"",
      "",
      "namespace tachyon::cc::math {",
      "",
      "template <>",
      "struct PrimeFieldTraits<tachyon_%{type}_%{suffix}> {",
      "  using PrimeFieldTy = tachyon::math::%{type}::%{Suffix};",
      "};",
      "",
      "template <>",
      "struct PrimeFieldTraits<tachyon::math::%{type}::%{Suffix}> {",
      "  using CPrimeFieldTy = tachyon_%{type}_%{suffix};",
      "};",
      "",
      "}  // namespace tachyon::cc::math",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{suffix}", std::string(suffix)},
                       {"%{Suffix}", suffix == "fq" ? "Fq" : "Fr"},
                   });
  return WriteHdr(content);
}

int GenerationConfig::GenerateFqTraitsHdr() const {
  return GeneratePrimeFieldTraitsHdr("fq");
}

int GenerationConfig::GenerateFrTraitsHdr() const {
  return GeneratePrimeFieldTraitsHdr("fr");
}

int GenerationConfig::GenerateG1Hdr() const {
  // clang-format off
  std::string_view tpl[] = {
      "#include \"tachyon/c/export.h\"",
      "#include \"%{header_path}\"",
      "",
      "struct TACHYON_C_EXPORT __attribute__((aligned(32))) %{g1}_affine {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  // needs to occupy 32 byte",
      "  // NOTE(chokobole): See LimbsAlignment() in tachyon/math/base/big_int.h",
      "  bool infinity;",
      "};",
      "",
      "struct TACHYON_C_EXPORT %{g1}_projective {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "};",
      "",
      "struct TACHYON_C_EXPORT %{g1}_jacobian {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "};",
      "",
      "struct TACHYON_C_EXPORT %{g1}_xyzz {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq zz;",
      "  tachyon_%{type}_fq zzz;",
      "};",
      "",
      "struct TACHYON_C_EXPORT %{g1}_point2 {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "};",
      "",
      "struct TACHYON_C_EXPORT %{g1}_point3 {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "};",
      "",
      "struct TACHYON_C_EXPORT %{g1}_point4 {",
      "  tachyon_%{type}_fq x;",
      "  tachyon_%{type}_fq y;",
      "  tachyon_%{type}_fq z;",
      "  tachyon_%{type}_fq w;",
      "};",
      "",
      "TACHYON_C_EXPORT void %{g1}_init();",
      "",
      "%{creation_ops}",
      "",
      "%{equality_ops}",
  };
  // clang-format on
  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string creation_ops;
  std::vector<std::string> creation_ops_components;
  for (size_t i = 0; i < std::size(kPointCreationOps); ++i) {
    for (size_t j = 0; j < std::size(kG1PointKinds); ++j) {
      creation_ops_components.push_back(absl::Substitute(
          // clang-format off
            "TACHYON_C_EXPORT $0 $0_$1();",
          // clang-format on
          kG1PointKinds[j], kPointCreationOps[i]));
      creation_ops_components.push_back("");
    }
    if (i == std::size(kPointCreationOps) - 1) {
      creation_ops_components.pop_back();
    }
  }
  creation_ops = absl::StrJoin(creation_ops_components, "\n");

  std::string equality_ops;
  std::vector<std::string> equality_ops_components;
  for (size_t i = 0; i < std::size(kEqualityOps); ++i) {
    for (size_t j = 0; j < std::size(kG1PointKinds); ++j) {
      equality_ops_components.push_back(absl::Substitute(
          // clang-format off
            "bool TACHYON_C_EXPORT $0_$1(const $0* a, const $0* b);",
          // clang-format on
          kG1PointKinds[j], kEqualityOps[i]));
      equality_ops_components.push_back("");
    }
    if (i == std::size(kEqualityOps) - 1) {
      equality_ops_components.pop_back();
    }
  }
  equality_ops = absl::StrJoin(equality_ops_components, "\n");

  tpl_content =
      absl::StrReplaceAll(tpl_content, {
                                           {"%{creation_ops}", creation_ops},
                                           {"%{equality_ops}", equality_ops},
                                       });

  base::FilePath hdr_path = GetHdrPath();
  std::string basename = hdr_path.BaseName().value();
  std::string header_path = hdr_path.DirName().Append("fq.h").value();
  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_path}", header_path},
                       {"%{type}", type},
                       {"%{g1}", absl::Substitute("tachyon_$0_g1", type)},
                   });
  return WriteHdr(content);
}

int GenerationConfig::GenerateG1Src() const {
  // clang-format off
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/fq_prime_field_traits.h\"",
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/g1_point_traits.h\"",
      "#include \"tachyon/cc/math/elliptic_curves/point_conversions.h\"",
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/g1.h\"",
      "",
      "void tachyon_%{type}_init() {",
      "  tachyon::math::%{type}::G1AffinePoint::Curve::Init();",
      "}",
      "",
      "%{creation_ops}",
      "",
      "%{equality_ops}",
  };
  // clang-format on

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string creation_ops;
  std::vector<std::string> creation_ops_components;
  const char* kUpperCreationOps[] = {"Zero", "Generator", "Random"};
  const char* kUpperPointKinds[] = {"AffinePoint", "ProjectivePoint",
                                    "JacobianPoint", "PointXYZZ"};
  for (size_t i = 0; i < std::size(kPointCreationOps); ++i) {
    for (size_t j = 0; j < std::size(kG1PointKinds); ++j) {
      creation_ops_components.push_back(absl::Substitute(
          // clang-format off
            "$0 $0_$1() {\n"
            "  using namespace tachyon::cc::math;\n"
            "  using CurvePointTy = typename PointTraits<$0>::CurvePointTy;\n"
            "  return ToC$2(CurvePointTy::$3());\n"
            "}",
          // clang-format on
          kG1PointKinds[j], kPointCreationOps[i], kUpperPointKinds[j],
          kUpperCreationOps[i]));
      creation_ops_components.push_back("");
    }
    if (i == std::size(kPointCreationOps) - 1) {
      creation_ops_components.pop_back();
    }
  }
  creation_ops = absl::StrJoin(creation_ops_components, "\n");

  std::string equality_ops;
  std::vector<std::string> equality_ops_components;
  const char* kEqualitySymbols[] = {"==", "!="};
  for (size_t i = 0; i < std::size(kEqualityOps); ++i) {
    for (size_t j = 0; j < std::size(kG1PointKinds); ++j) {
      equality_ops_components.push_back(absl::Substitute(
          // clang-format off
            "bool $0_$1(const $0* a, const $0* b) {\n"
            "  using namespace tachyon::cc::math;\n"
            "  return To$2(*a) $3 To$2(*b);\n"
            "}",
          // clang-format on
          kG1PointKinds[j], kEqualityOps[i], kUpperPointKinds[j],
          kEqualitySymbols[i]));
      equality_ops_components.push_back("");
    }
    if (i == std::size(kEqualityOps) - 1) {
      equality_ops_components.pop_back();
    }
  }
  equality_ops = absl::StrJoin(equality_ops_components, "\n");

  tpl_content =
      absl::StrReplaceAll(tpl_content, {
                                           {"%{creation_ops}", creation_ops},
                                           {"%{equality_ops}", equality_ops},
                                       });

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{type}", type},
                       {"%{g1}", absl::Substitute("tachyon_$0_g1", type)},
                   });
  return WriteSrc(content);
}

int GenerationConfig::GenerateG1TraitsHdr() const {
  std::vector<std::string_view> tpl = {
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/fr.h\"",
      "#include \"tachyon/c/math/elliptic_curves/%{header_dir_name}/g1.h\"",
      "#include \"tachyon/cc/math/elliptic_curves/point_traits.h\"",
      "#include \"tachyon/math/elliptic_curves/%{header_dir_name}/g1.h\"",
      "",
      "namespace tachyon::cc::math {",
      "",
      "template <>",
      "struct PointTraits<tachyon::math::%{type}::G1AffinePoint> {",
      "  using CPointTy = tachyon_%{type}_g1_point2;",
      "  using CCurvePointTy = tachyon_%{type}_g1_affine;",
      "  using CScalarField = tachyon_%{type}_fr;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon::math::%{type}::G1ProjectivePoint> {",
      "  using CPointTy = tachyon_%{type}_g1_point3;",
      "  using CCurvePointTy = tachyon_%{type}_g1_projective;",
      "  using CScalarField = tachyon_%{type}_fr;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon::math::%{type}::G1JacobianPoint> {",
      "  using CPointTy = tachyon_%{type}_g1_point3;",
      "  using CCurvePointTy = tachyon_%{type}_g1_jacobian;",
      "  using CScalarField = tachyon_%{type}_fr;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon::math::%{type}::G1PointXYZZ> {",
      "  using CPointTy = tachyon_%{type}_g1_point4;",
      "  using CCurvePointTy = tachyon_%{type}_g1_xyzz;",
      "  using CScalarField = tachyon_%{type}_fr;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_affine> {",
      "  using PointTy = tachyon::math::Point2<tachyon::math::%{type}::Fq>;",
      "  using CurvePointTy = tachyon::math::%{type}::G1AffinePoint;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_projective> {",
      "  using PointTy = tachyon::math::Point3<tachyon::math::%{type}::Fq>;",
      "  using CurvePointTy = tachyon::math::%{type}::G1ProjectivePoint;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_jacobian> {",
      "  using PointTy = tachyon::math::Point3<tachyon::math::%{type}::Fq>;",
      "  using CurvePointTy = tachyon::math::%{type}::G1JacobianPoint;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_xyzz> {",
      "  using PointTy = tachyon::math::Point4<tachyon::math::%{type}::Fq>;",
      "  using CurvePointTy = tachyon::math::%{type}::G1PointXYZZ;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_point2> {",
      "  using PointTy = tachyon::math::Point2<tachyon::math::%{type}::Fq>;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_point3> {",
      "  using PointTy = tachyon::math::Point3<tachyon::math::%{type}::Fq>;",
      "};",
      "",
      "template <>",
      "struct PointTraits<tachyon_%{type}_g1_point4> {",
      "  using PointTy = tachyon::math::Point4<tachyon::math::%{type}::Fq>;",
      "};",
      "",
      "}  // namespace tachyon::cc::math",
  };

  std::string tpl_content = absl::StrJoin(tpl, "\n");

  std::string content = absl::StrReplaceAll(
      tpl_content, {
                       {"%{header_dir_name}", c::math::GetLocation(type)},
                       {"%{fq_limb_nums}", absl::StrCat(fq_limb_nums)},
                       {"%{fr_limb_nums}", absl::StrCat(fr_limb_nums)},
                       {"%{type}", type},
                   });
  return WriteHdr(content);
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
  } else {
    tachyon_cerr << "not supported suffix:" << config.out << std::endl;
    return 1;
  }
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
