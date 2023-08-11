#include "tachyon/c/math/elliptic_curves/msm/msm.h"

#include "absl/types/span.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon {

namespace {

using namespace math;

bn254::G1JacobianPoint DoMSM(const tachyon_bn254_point2* bases_in,
                             size_t bases_len,
                             const tachyon_bn254_fr* scalars_in,
                             size_t scalars_len) {
  absl::Span<const Point2<bn254::Fq>> points(
      reinterpret_cast<const Point2<bn254::Fq>*>(bases_in), bases_len);
  std::vector<bn254::G1JacobianPoint> bases =
      base::Map(points, [](const Point2<bn254::Fq>& point) {
        return bn254::G1AffinePoint(point).ToJacobian();
      });
  absl::Span<const bn254::Fr> scalars(
      reinterpret_cast<const bn254::Fr*>(scalars_in), scalars_len);
  return VariableBaseMSM<bn254::G1JacobianPoint>::MSM(bases, scalars);
}

bn254::G1JacobianPoint DoMSM(const tachyon_bn254_g1_affine* bases_in,
                             size_t bases_len,
                             const tachyon_bn254_fr* scalars_in,
                             size_t scalars_len) {
  std::vector<bn254::G1JacobianPoint> bases = base::Map(
      absl::MakeConstSpan(
          reinterpret_cast<const bn254::G1AffinePoint*>(bases_in), bases_len),
      [](const bn254::G1AffinePoint& point) { return point.ToJacobian(); });
  absl::Span<const bn254::Fr> scalars(
      reinterpret_cast<const bn254::Fr*>(scalars_in), scalars_len);
  return VariableBaseMSM<bn254::G1JacobianPoint>::MSM(bases, scalars);
}

tachyon_bn254_g1_jacobian* ToCCPtr(const bn254::G1JacobianPoint& point) {
  tachyon_bn254_g1_jacobian* ret = new tachyon_bn254_g1_jacobian;
  memcpy(&ret->x, point.x().value().limbs, sizeof(uint64_t) * 4);
  memcpy(&ret->y, point.y().value().limbs, sizeof(uint64_t) * 4);
  memcpy(&ret->z, point.z().value().limbs, sizeof(uint64_t) * 4);
  return ret;
}

}  // namespace

}  // namespace tachyon

tachyon_bn254_g1_jacobian* tachyon_msm_g1_point2(
    const tachyon_bn254_point2* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::ToCCPtr(
      tachyon::DoMSM(bases, bases_len, scalars, scalars_len));
}

tachyon_bn254_g1_jacobian* tachyon_msm_g1_affine(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::ToCCPtr(
      tachyon::DoMSM(bases, bases_len, scalars, scalars_len));
}
