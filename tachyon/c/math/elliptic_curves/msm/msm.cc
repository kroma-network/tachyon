#include "tachyon/c/math/elliptic_curves/msm/msm.h"

#include <tuple>

#include "absl/types/span.h"

#include "tachyon/base/console/console_stream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon {

using namespace math;

namespace c::math {

namespace {

std::unique_ptr<MSMInputProvider> g_provider;

void DoInitMSM(uint8_t degree) {
  {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Green();
    std::cout << "DoInitMSM()" << std::endl;
  }
  bn254::G1AffinePoint::Curve::Init();

  std::ignore = degree;
  g_provider.reset(new MSMInputProvider());
}

void DoReleaseMSM() {
  {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Green();
    std::cout << "DoReleaseMSM()" << std::endl;
  }
  g_provider.reset();
}

template <typename T>
tachyon_bn254_g1_jacobian* DoMSM(const T* bases, size_t bases_len,
                                 const tachyon_bn254_fr* scalars,
                                 size_t scalars_len) {
  g_provider->Inject(bases, bases_len, scalars, scalars_len);
  return CreateCPoint3Ptr<tachyon_bn254_g1_jacobian>(
      VariableBaseMSM<bn254::G1AffinePoint>::MSM(g_provider->bases(),
                                                 g_provider->scalars()));
}

}  // namespace

}  // namespace c::math
}  // namespace tachyon

void tachyon_init_msm(uint8_t degree) { tachyon::c::math::DoInitMSM(degree); }

void tachyon_release_msm() { tachyon::c::math::DoReleaseMSM(); }

tachyon_bn254_g1_jacobian* tachyon_bn254_g1_point2_msm(
    const tachyon_bn254_g1_point2* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSM(bases, bases_len, scalars, scalars_len);
}

tachyon_bn254_g1_jacobian* tachyon_bn254_g1_affine_msm(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSM(bases, bases_len, scalars, scalars_len);
}
