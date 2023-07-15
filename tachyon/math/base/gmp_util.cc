#include "tachyon/math/base/gmp_util.h"

#include "absl/base/call_once.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon {
namespace math {
namespace gmp {

namespace {

bool TestBit(const mpz_class* field, size_t index) {
  return mpz_tstbit(field->get_mpz_t(), index) == 1;
}

}  // namespace

// static
BitIteratorBE BitIteratorBE::begin(const mpz_class* field) {
#if ARCH_CPU_BIG_ENDIAN
  return {field, 0};
#else  // ARCH_CPU_LITTLE_ENDIAN
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return end(field);
  }
  return {field, bits - 1};
#endif
}

// static
BitIteratorBE BitIteratorBE::WithoutLeadingZeros(const mpz_class* field) {
  BitIteratorBE ret = begin(field);
#if ARCH_CPU_LITTLE_ENDIAN
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return ret;
  }
#endif  // ARCH_CPU_LITTLE_ENDIAN
  while (*ret == 0) {
    ++ret;
  }
  return ret;
}

// static
BitIteratorBE BitIteratorBE::end(const mpz_class* field) {
#if ARCH_CPU_BIG_ENDIAN
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return begin(field);
  }
  return BitIteratorBE(field, bits);
#else  // ARCH_CPU_LITTLE_ENDIAN
  return BitIteratorBE(field, std::numeric_limits<size_t>::max());
#endif
}

const bool* BitIteratorBE::operator->() const {
  value_ = TestBit(field_, index_);
  return &value_;
}

const bool& BitIteratorBE::operator*() const {
  value_ = TestBit(field_, index_);
  return value_;
}

// static
BitIteratorLE BitIteratorLE::begin(const mpz_class* field) {
#if ARCH_CPU_LITTLE_ENDIAN
  return BitIteratorLE(field, 0);
#else  // ARCH_CPU_BIG_ENDIAN
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return end(field);
  }
  return BitIteratorLE(field, bits - 1);
#endif
}

// static
BitIteratorLE BitIteratorLE::end(const mpz_class* field) {
#if ARCH_CPU_LITTLE_ENDIAN
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return begin(field);
  }
  return {field, bits};
#else  // ARCH_CPU_BIG_ENDIAN
  return {field, std::numeric_limits<size_t>::max()};
#endif
}

// static
BitIteratorLE BitIteratorLE::WithoutTrailingZeros(const mpz_class* field) {
  BitIteratorLE ret = end(field);
#if ARCH_CPU_LITTLE_ENDIAN
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return ret;
  }
#endif
  while (*ret == 0) {
    --ret;
  }
  return ++ret;
}

const bool* BitIteratorLE::operator->() const {
  value_ = TestBit(field_, index_);
  return &value_;
}

const bool& BitIteratorLE::operator*() const {
  value_ = TestBit(field_, index_);
  return value_;
}

gmp_randstate_t& GetRandomState() {
  static gmp_randstate_t random_state;
  static absl::once_flag once;
  absl::call_once(once, []() {
    gmp_randinit_mt(random_state);
    gmp_randseed_ui(random_state, time(NULL));
  });
  return random_state;
}

bool ParseIntoMpz(std::string_view str, int base, mpz_class* out) {
  if (base == 16) {
    base::ConsumePrefix0x(&str);
  }

  return out->set_str(str.data(), base) == 0;
}

void MustParseIntoMpz(std::string_view str, int base, mpz_class* out) {
  CHECK(ParseIntoMpz(str, base, out));
}

mpz_class FromDecString(std::string_view str) {
  mpz_class ret;
  MustParseIntoMpz(str, 10, &ret);
  return ret;
}

mpz_class FromHexString(std::string_view str) {
  mpz_class ret;
  MustParseIntoMpz(str, 16, &ret);
  return ret;
}

Sign GetSign(const mpz_class& out) {
  return ToSign(mpz_sgn(out.get_mpz_t()));
}

bool IsZero(const mpz_class& value) { return GetSign(value) == Sign::kZero; }

bool IsNegative(const mpz_class& value) {
  return GetSign(value) == Sign::kNegative;
}

bool IsPositive(const mpz_class& value) {
  return GetSign(value) == Sign::kPositive;
}

mpz_class GetAbs(const mpz_class& value) {
  mpz_class ret;
  mpz_abs(ret.get_mpz_t(), value.get_mpz_t());
  return ret;
}

size_t GetNumBits(const mpz_class& value) {
  return GetLimbSize(value) * GMP_LIMB_BITS;
}

size_t GetLimbSize(const mpz_class& value) {
  return value.__get_mp()->_mp_size;
}

uint64_t GetLimb(const mpz_class& value, size_t idx) {
  return value.__get_mp()->_mp_d[idx];
}

}  // namespace gmp
}  // namespace math
}  // namespace tachyon
