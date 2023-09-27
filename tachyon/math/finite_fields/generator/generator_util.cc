#include "tachyon/math/finite_fields/generator/generator_util.h"

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_iterator.h"

namespace tachyon::math {

// TODO(chokobole): Consider bigendian.
std::string MpzClassToString(const mpz_class& m) {
  size_t limb_size = math::gmp::GetLimbSize(m);
  if (limb_size == 0) {
    return "UINT64_C(0)";
  }

  std::vector<std::string> ret;
  ret.reserve(limb_size);
  for (size_t i = 0; i < limb_size; ++i) {
    ret.push_back(
        absl::Substitute("UINT64_C($0)", math::gmp::GetLimbConstRef(m, i)));
  }
  return absl::StrJoin(ret, ", ");
}

std::string MpzClassToMontString(const mpz_class& v, const mpz_class& m) {
  size_t limb_size = math::gmp::GetLimbSize(m);
  switch (limb_size) {
    case 1:
      return MpzClassToMontString<1>(v, m);
    case 2:
      return MpzClassToMontString<2>(v, m);
    case 3:
      return MpzClassToMontString<3>(v, m);
    case 4:
      return MpzClassToMontString<4>(v, m);
    case 5:
      return MpzClassToMontString<5>(v, m);
    case 6:
      return MpzClassToMontString<6>(v, m);
    case 7:
      return MpzClassToMontString<7>(v, m);
    case 8:
      return MpzClassToMontString<8>(v, m);
    case 9:
      return MpzClassToMontString<9>(v, m);
  }
  NOTREACHED();
  return "";
}

std::string GenerateFastMultiplication(int value) {
  CHECK_NE(value, 0);
  bool is_negative = value < 0;
  math::BigInt<1> scalar(is_negative ? -value : value);
  auto it = math::BitIteratorBE<math::BigInt<1>>::begin(&scalar, true);
  ++it;
  auto end = math::BitIteratorBE<math::BigInt<1>>::end(&scalar);
  std::stringstream ss;
  while (it != end) {
    ss << ".DoubleInPlace()";
    if (*it) {
      ss << ".AddInPlace(v)";
    }
    ++it;
  }
  if (is_negative) ss << ".NegInPlace()";
  return ss.str();
}

base::FilePath ConvertToCpuHdr(const base::FilePath& path) {
  std::string basename = path.BaseName().value();
  basename = basename.substr(0, basename.find("_gpu"));
  return path.DirName().Append(basename + ".h");
}

base::FilePath ConvertToGpuHdr(const base::FilePath& path) {
  std::string basename = path.BaseName().RemoveExtension().value();
  return path.DirName().Append(basename + "_gpu.h");
}

std::string GenerateInitMpzClass(std::string_view name, int value) {
  if (value < 0) {
    return absl::Substitute("    $0 = -gmp::FromDecString($1);", name, -value);
  } else {
    return absl::Substitute("    $0 = gmp::FromDecString($1);", name, value);
  }
}

std::string GenerateInitMpzClass(std::string_view name,
                                 std::string_view value) {
  if (base::ConsumePrefix(&value, "-")) {
    return absl::Substitute("    $0 = -gmp::FromDecString(\"$1\");", name,
                            value);
  } else {
    return absl::Substitute("    $0 = gmp::FromDecString(\"$1\");", name,
                            value);
  }
}

std::string GenerateInitField(std::string_view name, int value,
                              bool is_base_field) {
  std::string_view type = is_base_field ? "BaseField" : "ScalarField";
  if (value < 0) {
    return absl::Substitute("    $0 = -$1($2);", name, type, -value);
  } else {
    return absl::Substitute("    $0 = $1($2);", name, type, value);
  }
}

std::string GenerateInitField(std::string_view name, std::string_view value,
                              bool is_base_field) {
  std::string_view type = is_base_field ? "BaseField" : "ScalarField";
  if (base::ConsumePrefix(&value, "-")) {
    return absl::Substitute("    $0 = -$1::FromDecString(\"$2\");", name, type,
                            value);
  } else {
    return absl::Substitute("    $0 = $1::FromDecString(\"$2\");", name, type,
                            value);
  }
}

std::string GenerateInitExtField(std::string_view name,
                                 absl::Span<const int> values,
                                 bool gen_f_type_alias) {
  std::vector<std::string> init_components;
  if (gen_f_type_alias) {
    init_components.push_back("    using F = typename BaseField::BaseField;");
  }
  std::stringstream ss;
  ss << "    ";
  ss << name;
  ss << " = BaseField(";
  for (size_t i = 0; i < values.size(); ++i) {
    unsigned int abs_value;
    if (values[i] < 0) {
      ss << "-";
      abs_value = -values[i];
    } else {
      abs_value = values[i];
    }
    if (abs_value == 0) {
      ss << "F::Zero()";
    } else if (abs_value == 1) {
      ss << "F::One()";
    } else {
      ss << "F(" << abs_value << ")";
    }

    if (i != values.size() - 1) {
      ss << ", ";
    }
  }
  ss << ");";
  init_components.push_back(ss.str());
  return absl::StrJoin(init_components, "\n");
}

std::string GenerateInitExtField(std::string_view name,
                                 absl::Span<const std::string> values,
                                 bool gen_f_type_alias) {
  std::vector<std::string> init_components;
  if (gen_f_type_alias) {
    init_components.push_back("    using F = typename BaseField::BaseField;");
  }
  std::stringstream ss;
  ss << "    ";
  ss << name;
  ss << " = BaseField(";
  for (size_t i = 0; i < values.size(); ++i) {
    std::string_view value = values[i];
    if (base::ConsumePrefix(&value, "-")) {
      ss << "-F::FromDecString(\"" << value << "\")";
    } else {
      ss << "F::FromDecString(\"" << value << "\")";
    }
    if (i != values.size() - 1) {
      ss << ", ";
    }
  }
  ss << ");";
  init_components.push_back(ss.str());
  return absl::StrJoin(init_components, "\n");
}

}  // namespace tachyon::math
