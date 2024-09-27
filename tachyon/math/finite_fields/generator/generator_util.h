#ifndef TACHYON_MATH_FINITE_FIELDS_GENERATOR_GENERATOR_UTIL_H_
#define TACHYON_MATH_FINITE_FIELDS_GENERATOR_GENERATOR_UTIL_H_

#include <string>

#include "absl/types/span.h"
#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/files/file_path.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/modulus.h"

namespace tachyon::math {

// TODO(chokobole): Consider bigendian.
std::string MpzClassToString(const mpz_class& m);

template <size_t N>
std::string MpzClassToMontString(const mpz_class& v_in, const mpz_class& m_in) {
  BigInt<N> m(0);

  if constexpr (N == 1) {
    // NOLINTNEXTLINE(runtime/int)
    if (m_in < mpz_class(static_cast<unsigned long>(uint64_t{1} << 32))) {
      return MpzClassToString(mpz_class((v_in.get_ui() << 32) % m_in.get_ui()));
    }
  }

  gmp::CopyLimbs(m_in, m.limbs);
  BigInt<N> r2 = Modulus<N>::MontgomeryR2(m);
  uint64_t inv = Modulus<N>::template Inverse<uint64_t>(m);

  BigInt<N> v(0);
  if (gmp::IsNegative(v_in)) {
    gmp::CopyLimbs(m_in + v_in, v.limbs);
  } else {
    gmp::CopyLimbs(v_in, v.limbs);
  }

  BigInt<2 * N> mul_result = v.MulExtend(r2);
  BigInt<N>::template MontgomeryReduce64<false>(mul_result, m, inv, &v);

  mpz_class v_mont;
  gmp::WriteLimbs(v.limbs, N, &v_mont);
  return MpzClassToString(v_mont);
}

std::string MpzClassToMontString(const mpz_class& v, const mpz_class& m);

std::string GenerateFastMultiplication(int64_t value);

base::FilePath ConvertToConfigHdr(const base::FilePath& path);

base::FilePath ConvertToCpuHdr(const base::FilePath& path);

base::FilePath ConvertToGpuHdr(const base::FilePath& path);

std::string GenerateInitMpzClass(std::string_view name, std::string_view value);

std::string GenerateInitField(std::string_view name, std::string_view type,
                              std::string_view value);

std::string GenerateInitPackedField(std::string_view name,
                                    std::string_view type,
                                    std::string_view value);

std::string GenerateInitExtField(std::string_view name, std::string_view type,
                                 absl::Span<const std::string> values,
                                 size_t degree);

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GENERATOR_GENERATOR_UTIL_H_
