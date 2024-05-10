#ifndef TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_TYPE_TRAITS_H_
#define TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_TYPE_TRAITS_H_

#include <type_traits>

#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"
#include "tachyon/zk/base/commitments/gwc_extension.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/lookup/halo2/scheme.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk {

using BN254SHPlonk =
    SHPlonkExtension<math::bn254::BN254Curve, halo2::kMaxDegree,
                     halo2::kMaxExtendedDegree, math::bn254::G1AffinePoint>;
using BN254GWC =
    GWCExtension<math::bn254::BN254Curve, halo2::kMaxDegree,
                 halo2::kMaxExtendedDegree, math::bn254::G1AffinePoint>;
using BN254Halo2LS = lookup::halo2::Scheme<typename BN254SHPlonk::Poly,
                                           typename BN254SHPlonk::Evals,
                                           typename BN254SHPlonk::Commitment>;

template <typename Circuit>
constexpr bool IsSimpleFloorPlanner =
    std::is_same_v<typename Circuit::FloorPlanner, SimpleFloorPlanner<Circuit>>;

template <typename Circuit>
constexpr bool IsV1FloorPlanner =
    std::is_same_v<typename Circuit::FloorPlanner, V1FloorPlanner<Circuit>>;

template <typename T>
struct IsSHPlonkImpl {
  static constexpr bool value = false;
};

template <typename Curve, size_t MaxDegree, size_t ExtendedMaxDegree,
          typename Commitment>
struct IsSHPlonkImpl<
    SHPlonkExtension<Curve, MaxDegree, ExtendedMaxDegree, Commitment>> {
  static constexpr bool value = true;
};

template <typename PCS>
constexpr bool IsSHPlonk = IsSHPlonkImpl<PCS>::value;

template <typename T>
struct IsGWCImpl {
  static constexpr bool value = false;
};

template <typename Curve, size_t MaxDegree, size_t ExtendedMaxDegree,
          typename Commitment>
struct IsGWCImpl<
    GWCExtension<Curve, MaxDegree, ExtendedMaxDegree, Commitment>> {
  static constexpr bool value = true;
};

template <typename PCS>
constexpr bool IsGWC = IsGWCImpl<PCS>::value;

template <typename T>
struct IsHalo2LSImpl {
  static constexpr bool value = false;
};

template <typename Poly, typename Evals, typename Commitment>
struct IsHalo2LSImpl<lookup::halo2::Scheme<Poly, Evals, Commitment>> {
  static constexpr bool value = true;
};

template <typename LS>
constexpr bool IsHalo2LS = IsHalo2LSImpl<LS>::value;

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_TYPE_TRAITS_H_
