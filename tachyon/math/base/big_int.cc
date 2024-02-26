#include "tachyon/math/base/big_int.h"

#include <string>

#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon::math {

template class BigInt<1>;
template class BigInt<2>;
template class BigInt<3>;
template class BigInt<4>;
template class BigInt<5>;
template class BigInt<6>;
template class BigInt<7>;
template class BigInt<8>;
template class BigInt<9>;

namespace internal {

namespace {

// TODO(chokobole): Remove gmp dependency.
template <size_t Base>
bool DoStringToLimbs(std::string_view str, uint64_t* limbs, size_t limb_nums) {
  mpz_class out;
  if (!gmp::ParseIntoMpz(str, Base, &out)) return false;
  if (gmp::GetLimbSize(out) > limb_nums) return false;
  gmp::CopyLimbs(out, limbs);
  return true;
}

template <size_t Base>
std::string DoLimbsToString(const uint64_t* limbs, size_t limb_nums,
                            bool pad_zero) {
  mpz_class out;
  gmp::WriteLimbs(limbs, limb_nums, &out);
  std::string str = out.get_str(Base);
  if (!pad_zero) return str;
  return base::ToHexStringWithLeadingZero(str, limb_nums * 8 * 2);
}

}  // namespace

bool StringToLimbs(std::string_view str, uint64_t* limbs, size_t limb_nums) {
  return DoStringToLimbs<10>(str, limbs, limb_nums);
}

bool HexStringToLimbs(std::string_view str, uint64_t* limbs, size_t limb_nums) {
  return DoStringToLimbs<16>(str, limbs, limb_nums);
}

std::string LimbsToString(const uint64_t* limbs, size_t limb_nums) {
  return DoLimbsToString<10>(limbs, limb_nums, false);
}

std::string LimbsToHexString(const uint64_t* limbs, size_t limb_nums,
                             bool pad_zero) {
  return base::MaybePrepend0x(DoLimbsToString<16>(limbs, limb_nums, pad_zero));
}

}  // namespace internal
}  // namespace tachyon::math
