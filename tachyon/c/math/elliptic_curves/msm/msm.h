#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_H_

#include <tuple>

#include "tachyon/base/console/console_stream.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"
#include "tachyon/c/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/msm/variable_base_msm.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::c::math {

template <typename Point>
struct MSMApi {
  MSMInputProvider<Point> provider;
  tachyon::math::VariableBaseMSM<Point> msm;

  explicit MSMApi(uint8_t degree) {
    // NOTE(chokobole): This constructor accepts |degree| for compatibility with
    // a constructor of MSMGpuApi. We should consider whether it accepts an
    // argument for algorithm selection even though it only supports pippenger
    // in a same manner.
    std::ignore = degree;
    {
      // NOTE(chokobole): This should be replaced with VLOG().
      // Currently, there's no way to delegate VLOG flags from rust side.
      tachyon::base::ConsoleStream cs;
      cs.Green();
      std::cout << "CreateMSMApi()" << std::endl;
    }
  }
};

template <
    typename RetPoint, typename Point, typename CPoint, typename CScalarField,
    typename CRetPoint = typename PointTraits<RetPoint>::CCurvePoint,
    typename Bucket = typename tachyon::math::VariableBaseMSM<Point>::Bucket>
CRetPoint* DoMSM(MSMApi<Point>& msm_api, const CPoint* bases,
                 const CScalarField* scalars, size_t size) {
  msm_api.provider.Inject(bases, scalars, size);
  Bucket bucket;
  CHECK(msm_api.msm.Run(msm_api.provider.bases(), msm_api.provider.scalars(),
                        &bucket));
  auto ret = tachyon::math::ConvertPoint<RetPoint>(bucket);
  CRetPoint* cret = new CRetPoint();
  ToCPoint3(ret, cret);
  return cret;
}

}  // namespace tachyon::c::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_H_
