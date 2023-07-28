#include "tachyon/math/base/big_int.h"

#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon::math::internal {

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
std::string DoLimbsToString(const uint64_t* limbs, size_t limb_nums) {
  mpz_class out;
  gmp::WriteLimbs(limbs, limb_nums, &out);
  return out.get_str(Base);
}

}  // namespace

bool StringToLimbs(std::string_view str, uint64_t* limbs, size_t limb_nums) {
  return DoStringToLimbs<10>(str, limbs, limb_nums);
}

bool HexStringToLimbs(std::string_view str, uint64_t* limbs, size_t limb_nums) {
  return DoStringToLimbs<16>(str, limbs, limb_nums);
}

std::string LimbsToString(const uint64_t* limbs, size_t limb_nums) {
  return DoLimbsToString<10>(limbs, limb_nums);
}

std::string LimbsToHexString(const uint64_t* limbs, size_t limb_nums) {
  return base::MaybePrepend0x(DoLimbsToString<16>(limbs, limb_nums));
}

}  // namespace tachyon::math::internal
