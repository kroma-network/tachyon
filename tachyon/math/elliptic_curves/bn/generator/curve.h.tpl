// clang-format off
#include "tachyon/base/logging.h"
#include "tachyon/math/elliptic_curves/bn/bn_curve.h"
#include "tachyon/math/elliptic_curves/pairing/twist_type.h"
#include "%{fq12_hdr}"
#include "%{g1_hdr}"
#include "%{g2_hdr}"

namespace %{namespace} {

template <typename Fq, typename Fq2, typename Fq6, typename Fq12, typename _G1Curve, typename _G2Curve>
class %{class}Config {
 public:
  constexpr static BigInt<%{x_size}> kX = BigInt<%{x_size}>({
    %{x}
  });

  constexpr static bool kXIsNegative = %{x_is_negative};
  constexpr static int8_t kAteLoopCount[] = {
    %{ate_loop_count}
  };
  constexpr static TwistType kTwistType = TwistType::k%{twist_type};

  using Fp = Fq;
  using Fp2 = Fq2;
  using Fp6 = Fq6;
  using Fp12 = Fq12;
  using G1Curve = _G1Curve;
  using G2Curve = _G2Curve;

  // NOTE(chokobole): Make them constexpr.
  static Fq2 kTwistMulByQX;
  static Fq2 kTwistMulByQY;

  static void Init() {
    // TODO(chokobole): This line below is needed by |GenerateInitExtField()|.
    // Later, I want GenerateInitExtField() to accept BasePrimeField as an argument.
    using BasePrimeField = Fq;
%{twist_mul_by_q_x_init_code}
%{twist_mul_by_q_y_init_code}
    G1Curve::Init();
    G2Curve::Init();
    VLOG(1) << "%{namespace}::%{class} initialized";
  }
};

template <typename Fq, typename Fq2, typename Fq6, typename Fq12, typename G1Curve, typename G2Curve>
Fq2 %{class}Config<Fq, Fq2, Fq6, Fq12, G1Curve, G2Curve>::kTwistMulByQX;
template <typename Fq, typename Fq2, typename Fq6, typename Fq12, typename G1Curve, typename G2Curve>
Fq2 %{class}Config<Fq, Fq2, Fq6, Fq12, G1Curve, G2Curve>::kTwistMulByQY;

using %{class}Curve = BNCurve<%{class}Config<Fq, Fq2, Fq6, Fq12, G1Curve, G2Curve>>;

}  // namespace %{namespace}
// clang-format on
