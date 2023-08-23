#ifndef TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_H_
#define TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_H_

#include <ostream>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/point2.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/point3.h"

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fq& fq);

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_fr& fr);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_affine& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_projective& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_jacobian& point);

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_g1_xyzz& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_affine& point);

std::ostream& operator<<(std::ostream& os,
                         const tachyon_bn254_g1_projective& point);

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_point2& point);

std::ostream& operator<<(std::ostream& os, const tachyon_bn254_point3& point);

#endif  // TACHYON_CC_MATH_ELLIPTIC_CURVES_BN_BN254_BN254_UTIL_H_
