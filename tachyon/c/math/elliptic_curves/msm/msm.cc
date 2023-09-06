#include "tachyon/c/math/elliptic_curves/msm/msm.h"

#include <tuple>

#include "absl/types/span.h"

#include "tachyon/base/console/console_stream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/no_destructor.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"
#include "tachyon/cc/math/elliptic_curves/bls/bls12_381/point_traits.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"

namespace tachyon {

using namespace math;

namespace c::math {

namespace {

template <typename PointTy>
struct MSMApi {
  MSMApi() = default;
  MSMApi(const MSMApi& other) = delete;
  MSMApi& operator=(const MSMApi& other) = delete;

  static MSMApi& Get() {
    static base::NoDestructor<MSMApi> api;
    return *api;
  }

  void Init() { PointTy::Curve::Init(); }

  void Release() { provider.Clear(); }

  MSMInputProvider<PointTy> provider;
  VariableBaseMSM<PointTy> msm;
};

void DoInitMSM(uint8_t degree) {
  {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Green();
    std::cout << "DoInitMSM()" << std::endl;
  }
  MSMApi<bn254::G1AffinePoint>::Get().Init();
  MSMApi<bls12_381::G1AffinePoint>::Get().Init();

  std::ignore = degree;
}

void DoReleaseMSM() {
  {
    // NOTE(chokobole): This should be replaced with VLOG().
    // Currently, there's no way to delegate VLOG flags from rust side.
    base::ConsoleStream cs;
    cs.Green();
    std::cout << "DoReleaseMSM()" << std::endl;
  }
  MSMApi<bn254::G1AffinePoint>::Get().Release();
  MSMApi<bls12_381::G1AffinePoint>::Get().Release();
}

template <typename PointTy,
          typename CPointTy = typename cc::math::PointTraits<PointTy>::CPointTy,
          typename CScalarField =
              typename cc::math::PointTraits<PointTy>::CScalarField,
          typename ReturnTy = typename VariableBaseMSM<PointTy>::ReturnTy,
          typename CReturnTy =
              typename cc::math::PointTraits<ReturnTy>::CCurvePointTy>
CReturnTy* DoMSM(const CPointTy* bases, size_t bases_len,
                 const CScalarField* scalars, size_t scalars_len) {
  MSMApi<PointTy>& msm_api = MSMApi<PointTy>::Get();
  msm_api.provider.Inject(bases, bases_len, scalars, scalars_len);
  ReturnTy ret;
  CHECK(msm_api.msm.Run(msm_api.provider.bases(), msm_api.provider.scalars(),
                        &ret));
  return cc::math::CreateCPoint3Ptr<CReturnTy>(ret);
}

}  // namespace

}  // namespace c::math
}  // namespace tachyon

void tachyon_init_msm(uint8_t degree) { tachyon::c::math::DoInitMSM(degree); }

void tachyon_release_msm() { tachyon::c::math::DoReleaseMSM(); }

tachyon_bn254_g1_jacobian* tachyon_bn254_g1_point2_msm(
    const tachyon_bn254_g1_point2* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSM<tachyon::math::bn254::G1AffinePoint>(
      bases, bases_len, scalars, scalars_len);
}

tachyon_bn254_g1_jacobian* tachyon_bn254_g1_affine_msm(
    const tachyon_bn254_g1_affine* bases, size_t bases_len,
    const tachyon_bn254_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSM<tachyon::math::bn254::G1AffinePoint>(
      bases, bases_len, scalars, scalars_len);
}

tachyon_bls12_381_g1_jacobian* tachyon_bls12_381_g1_point2_msm(
    const tachyon_bls12_381_g1_point2* bases, size_t bases_len,
    const tachyon_bls12_381_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSM<tachyon::math::bls12_381::G1AffinePoint>(
      bases, bases_len, scalars, scalars_len);
}

tachyon_bls12_381_g1_jacobian* tachyon_bls12_381_g1_affine_msm(
    const tachyon_bls12_381_g1_affine* bases, size_t bases_len,
    const tachyon_bls12_381_fr* scalars, size_t scalars_len) {
  return tachyon::c::math::DoMSM<tachyon::math::bls12_381::G1AffinePoint>(
      bases, bases_len, scalars, scalars_len);
}
