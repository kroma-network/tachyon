#include "tachyon/math/finite_fields/generator/generator_util.h"

#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/bit_iterator.h"

namespace tachyon::math {

// TODO(chokobole): Consider bigendian.
std::string MpzClassToString(const mpz_class& m) {
  size_t limb_size = gmp::GetLimbSize(m);
  if (limb_size == 0) {
    return "0";
  }

  enum class MaximumType {
    kInt,
    kUint32,
    kUint64,
  };

  // This code checks the maximum type of the vector. It determines the template
  // type |T| of the |std::initializer_list<T>|.
  MaximumType type = MaximumType::kInt;
  for (size_t i = 0; i < limb_size; ++i) {
    mp_limb_t limb = gmp::GetLimb(m, i);
    if (limb <= std::numeric_limits<int>::max()) {
      continue;
    } else if (limb <= std::numeric_limits<uint32_t>::max()) {
      type = MaximumType::kUint32;
    } else {
      type = MaximumType::kUint64;
      break;
    }
  }

  std::vector<std::string> ret =
      base::CreateVector(limb_size, [type, &m](size_t i) {
        mp_limb_t limb = gmp::GetLimb(m, i);
        switch (type) {
          case MaximumType::kInt:
            return absl::StrCat(limb);
          case MaximumType::kUint32:
            return absl::Substitute("UINT32_C($0)", limb);
          case MaximumType::kUint64:
            return absl::Substitute("UINT64_C($0)", limb);
        }
        NOTREACHED();
        return base::EmptyString();
      });
  return absl::StrJoin(ret, ", ");
}

std::string MpzClassToMontString(const mpz_class& v, const mpz_class& m) {
  size_t limb_size = gmp::GetLimbSize(m);
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

std::string GenerateFastMultiplication(int64_t value) {
  CHECK_NE(value, 0);
  bool is_negative = value < 0;
  BigInt<1> scalar(is_negative ? -value : value);
  auto it = BitIteratorBE<BigInt<1>>::begin(&scalar, true);
  ++it;
  auto end = BitIteratorBE<BigInt<1>>::end(&scalar);
  std::stringstream ss;
  while (it != end) {
    ss << ".DoubleInPlace()";
    if (*it) {
      ss << ".AddInPlace(v)";
    }
    ++it;
  }
  if (is_negative) ss << ".NegateInPlace()";
  return ss.str();
}

base::FilePath ConvertToConfigHdr(const base::FilePath& path) {
  std::string basename = path.BaseName().value();
  basename = basename.substr(0, basename.find("_gpu"));
  return path.DirName().Append(basename + "_config.h");
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

std::string GenerateInitMpzClass(std::string_view name,
                                 std::string_view value) {
  std::stringstream ss;
  ss << "    " << name << " = ";
  if (base::ConsumePrefix(&value, "-")) {
    ss << "-";
  }
  ss << "gmp::FromDecString(\"" << value << "\");";
  return ss.str();
}

namespace {

std::string DoGenerateInitField(std::string_view type, std::string_view value) {
  std::stringstream ss;
  if (base::ConsumePrefix(&value, "-")) {
    ss << "-";
  }
  ss << "*" << type << "::FromDecString(\"" << value << "\")";
  return ss.str();
}

}  // namespace

std::string GenerateInitField(std::string_view name, std::string_view type,
                              std::string_view value) {
  std::stringstream ss;
  ss << "    " << name << " = " << DoGenerateInitField(type, value) << ";";
  return ss.str();
}

std::string GenerateInitPackedField(std::string_view name,
                                    std::string_view type,
                                    std::string_view value) {
  std::stringstream ss;
  // clang-format off
  ss << "    using UnpackedPrimeField = typename math::FiniteFieldTraits<" << type << ">::PrimeField;" << std::endl;
  ss << "    " << name << " = " << type << "::Broadcast(" << DoGenerateInitField("UnpackedPrimeField", value) << ");";
  // clang-format on
  return ss.str();
}

// TODO(chokobole): Should be generalized for packed extension field.
std::string GenerateInitExtField(std::string_view name, std::string_view type,
                                 absl::Span<const std::string> values,
                                 size_t degree) {
  std::stringstream ss;
  ss << "    ";
  ss << name;
  ss << " = ";
  ss << type;

  bool is_prime_field = values.size() == degree;
  if (is_prime_field) {
    ss << "::FromBasePrimeFields(std::vector<BasePrimeField>{";
  } else {
    ss << "(";
  }
  for (size_t i = 0; i < values.size(); ++i) {
    std::string_view value = values[i];
    if (base::ConsumePrefix(&value, "-")) {
      ss << "-";
    }
    if (is_prime_field) {
      ss << "*BasePrimeField::FromDecString(\"" << value << "\")";
    } else {
      uint64_t abs_value;
      CHECK(base::StringToUint64(value, &abs_value));
      if (abs_value == 0) {
        ss << "BaseField::BaseField::Zero()";
      } else if (abs_value == 1) {
        ss << "BaseField::BaseField::One()";
      } else {
        ss << "BaseField::BaseField(" << abs_value << ")";
      }
    }
    if (i != values.size() - 1) {
      ss << ", ";
    }
  }
  if (is_prime_field) {
    ss << "}";
  }
  ss << ");";
  return ss.str();
}

}  // namespace tachyon::math
