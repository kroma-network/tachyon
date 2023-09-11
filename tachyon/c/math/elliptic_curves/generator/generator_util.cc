#include "tachyon/c/math/elliptic_curves/generator/generator_util.h"

#include "tachyon/base/logging.h"

namespace tachyon::c::math {

std::string GetLocation(std::string_view type) {
  if (type == "bn254") {
    return "bn/bn254";
  } else if (type == "bls12_381") {
    return "bls/bls12_381";
  }
  NOTREACHED() << "Unsupported type: " << type;
  return "";
}

}  // namespace tachyon::c::math
