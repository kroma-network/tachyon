#include "tachyon/math/base/gmp/gmp_util.h"

#include "absl/base/call_once.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon {
namespace math {
namespace gmp {

namespace {

gmp_randstate_t& GetRandomState() {
  static gmp_randstate_t random_state;
  static absl::once_flag once;
  absl::call_once(once, []() {
    gmp_randinit_mt(random_state);
    gmp_randseed_ui(random_state, time(NULL));
  });
  return random_state;
}

}  // namespace

static_assert(sizeof(mp_limb_t) == sizeof(uint64_t), "limb should be 64 bit");

mpz_class Random(mpz_class n) {
  mpz_class value;
  mpz_urandomm(value.get_mpz_t(), GetRandomState(), n.get_mpz_t());
  return value;
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

Sign GetSign(const mpz_class& out) { return ToSign(mpz_sgn(out.get_mpz_t())); }

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

bool TestBit(const mpz_class& value, size_t index) {
  return mpz_tstbit(value.get_mpz_t(), index) == 1;
}

void SetBit(mpz_class& value, size_t index, bool bit_value) {
  bit_value ? SetBit(value, index) : ClearBit(value, index);
}

void SetBit(mpz_class& value, size_t index) {
  mpz_setbit(value.get_mpz_t(), index);
}

void ClearBit(mpz_class& value, size_t index) {
  mpz_clrbit(value.get_mpz_t(), index);
}

uint64_t* GetLimbs(const mpz_class& value) {
  return reinterpret_cast<uint64_t*>(value.__get_mp()->_mp_d);
}

size_t GetLimbSize(const mpz_class& value) {
  return value.__get_mp()->_mp_size;
}

const uint64_t& GetLimbConstRef(const mpz_class& value, size_t idx) {
  return value.__get_mp()->_mp_d[idx];
}

uint64_t& GetLimbRef(mpz_class& value, size_t idx) {
  return value.__get_mp()->_mp_d[idx];
}

void CopyLimbs(const mpz_class& value, uint64_t* limbs) {
  for (size_t i = 0; i < GetLimbSize(value); ++i) {
    limbs[i] = GetLimbConstRef(value, i);
  }
}

void WriteLimbs(const uint64_t* limbs_src, size_t limb_size, mpz_class* out) {
  mp_ptr limbs_dst = mpz_limbs_write(out->get_mpz_t(), limb_size);
  for (size_t i = 0; i < limb_size; ++i) {
    limbs_dst[i] = limbs_src[i];
  }
  mpz_limbs_finish(out->get_mpz_t(), limb_size);
}

mpz_class DivBy2Exp(const mpz_class& value, uint64_t exp) {
  mpz_class ret;
  mpz_fdiv_q_2exp(ret.get_mpz_t(), value.get_mpz_t(), exp);
  return ret;
}

}  // namespace gmp
}  // namespace math
}  // namespace tachyon
