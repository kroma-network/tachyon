// clang-format off
#include "absl/base/call_once.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_curve.h"
#include "tachyon/math/elliptic_curves/pairing/twist_type.h"
#include "%{fq12_hdr}"
#include "%{g1_hdr}"
#include "%{g2_hdr}"

namespace %{namespace} {

template <typename Fq, typename Fq2, typename Fq6, typename Fq12, typename _G1Curve, typename _G2Curve>
class %{class}Config {
 public:
  constexpr static const char* kName = "%{namespace}::%{class}";

  constexpr static BigInt<%{x_size}> kX = BigInt<%{x_size}>({
    %{x}
  });
  constexpr static size_t kXLimbNums = %{x_size};
  constexpr static bool kXIsNegative = %{x_is_negative};
  constexpr static TwistType kTwistType = TwistType::k%{twist_type};

  using Fp = Fq;
  using Fp2 = Fq2;
  using Fp6 = Fq6;
  using Fp12 = Fq12;
  using G1Curve = _G1Curve;
  using G2Curve = _G2Curve;

  static void Init() {
    static absl::once_flag once;
    absl::call_once(once, [] {
      G1Curve::Init();
      G2Curve::Init();
      VLOG(1) << "%{namespace}::%{class} initialized";
    });
  }
};

using %{class}Curve = BLS12Curve<%{class}Config<Fq, Fq2, Fq6, Fq12, G1Curve, G2Curve>>;

}  // namespace %{namespace}
// clang-format on
