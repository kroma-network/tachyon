#include "tachyon/math/base/gmp_util.h"

#include "absl/base/call_once.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon {
namespace math {
namespace gmp {

namespace {

int Sign(const mpz_class& out) { return mpz_sgn(out.get_mpz_t()); }

bool TestBit(const mpz_class* field, size_t index) {
  return mpz_tstbit(field->get_mpz_t(), index) == 1;
}

}  // namespace

// static
BitIteratorBE BitIteratorBE::begin(const mpz_class* field) {
#if ARCH_CPU_BIG_ENDIAN == 1
  BitIteratorBE ret(field, 0);
#else
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return end(field);
  }
  BitIteratorBE ret(field, bits - 1);
#endif
  while (*ret == 0) {
    ++ret;
  }
  return ret;
}

// static
BitIteratorBE BitIteratorBE::end(const mpz_class* field) {
#if ARCH_CPU_BIG_ENDIAN == 1
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return begin(field);
  }
  return BitIteratorBE(field, bits);
#else
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
#if ARCH_CPU_LITTLE_ENDIAN == 1
  return BitIteratorLE(field, 0);
#else
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return end(field);
  }
  return BitIteratorLE(field, bits - 1);
#endif
}

// static
BitIteratorLE BitIteratorLE::end(const mpz_class* field) {
#if ARCH_CPU_LITTLE_ENDIAN == 1
  size_t bits = GetNumBits(*field);
  if (bits == 0) {
    return begin(field);
  }
  BitIteratorLE ret(field, bits);
#else
  BitIteratorLE ret(field, std::numeric_limits<size_t>::max());
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

void UnsignedIntegerToMpz(unsigned long int value, mpz_class* out) {
  mpz_set_ui(out->get_mpz_t(), value);
}

bool IsZero(const mpz_class& out) { return Sign(out) == 0; }

bool IsNegative(const mpz_class& out) { return Sign(out) == -1; }

bool IsPositive(const mpz_class& out) { return Sign(out) == 1; }

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
